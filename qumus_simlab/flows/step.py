"""Closed-loop Step (spec 4.2) and StepRunner (correction ladder, spec 4.3).

CONTRACT. Every physical step is a Step subclass:

    preconditions(ctx)    raise PreconditionFailed if the world isn't ready (nothing moved)
    act(ctx, correction)  do it; all device commands go through ctx.gate
    verify(ctx)           measure -> Observation(verdict, measurements, expected)
                          or return NOT_VERIFIABLE("why") - listed in testbench reports
    correct(ctx, obs)     return a Correction for the next act(), or None if impossible
    safe_state(ctx)       put the devices this step uses in safe state
    probe_state(ctx)      after a crash: look at the machine (not the records)

The StepRunner runs: preconditions -> act -> verify -> (correct -> act -> verify)*
up to max_corrections, checkpointing in the run store and emitting events.
Steps never return error strings; they raise qumus_simlab.core.errors.*.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from qumus_simlab.core.errors import (
    DangerousError,
    PreconditionFailed,
    QumusError,
    StopRequested,
    TransientError,
)

if TYPE_CHECKING:
    from qumus_simlab.flows.context import FlowContext


class Verdict(StrEnum):
    OK = "ok"
    CORRECTABLE = "correctable"
    UNCLEAR = "unclear"  # not a known category -> hand back to the agent (L1)
    DANGEROUS = "dangerous"


class ProbeResult(StrEnum):
    DONE = "done"
    NOT_DONE = "not_done"
    UNKNOWN = "unknown"


@dataclass
class Observation:
    verdict: Verdict
    measurements: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": str(self.verdict), "measurements": self.measurements,
                "expected": self.expected, "images": self.images, "message": self.message}


@dataclass(frozen=True)
class NotVerifiable:
    reason: str


def NOT_VERIFIABLE(reason: str) -> NotVerifiable:  # noqa: N802 - spec spelling
    return NotVerifiable(reason)


@dataclass
class Correction:
    """What to change on the next act(). Free-form per step, e.g. {"dx": .., "drot": ..}."""

    action: str
    params: dict[str, Any] = field(default_factory=dict)


class StepStatus(StrEnum):
    SUCCEEDED = "succeeded"  # verified OK first time
    CORRECTED = "corrected"  # verified OK after >= 1 correction
    UNVERIFIED = "unverified"  # act() done, step declares NOT_VERIFIABLE
    NEEDS_AGENT = "needs_agent"  # precondition failed / unclear / corrections exhausted
    DANGEROUS = "dangerous"  # safety violation or dangerous verdict: safe state, stop


@dataclass
class StepOutcome:
    status: StepStatus
    step: str
    attempts: int
    corrections: int
    observations: list[Observation] = field(default_factory=list)
    message: str = ""
    error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status in (StepStatus.SUCCEEDED, StepStatus.CORRECTED, StepStatus.UNVERIFIED)

    def to_dict(self) -> dict[str, Any]:
        return {"status": str(self.status), "step": self.step, "attempts": self.attempts,
                "corrections": self.corrections, "message": self.message, "error": self.error,
                "observations": [o.to_dict() for o in self.observations]}


class Step(ABC):
    name: str = "step"
    max_corrections: int = 2
    #: False for steps that must not be blindly resumed (e.g. mid-heating), spec 8.2
    safe_to_resume: bool = True
    #: may the Stop button pause the job inside this step (at ctx.check_cancel)?
    #: None = same as safe_to_resume; otherwise the step runs to its end and the
    #: job pauses before the next one (runtime/jobs.py)
    pause_inside: bool | None = None
    #: device ids this step needs locked
    devices: tuple[str, ...] = ()

    def preconditions(self, ctx: FlowContext) -> None:  # noqa: B027 - optional
        """Raise PreconditionFailed if not ready."""

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        """What verify() should find; stored with the checkpoint."""
        return {}

    @abstractmethod
    def act(self, ctx: FlowContext, correction: Correction | None) -> None: ...

    @abstractmethod
    def verify(self, ctx: FlowContext) -> Observation | NotVerifiable: ...

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return None

    def safe_state(self, ctx: FlowContext) -> None:
        for d in self.devices:
            ctx.gate.devices.get(d).safe_state()

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.UNKNOWN

    def before_park(self, ctx: FlowContext) -> None:  # noqa: B027
        """Restart recovery is about to park this step for a human: the one move
        that is always safe here (e.g. lift a stamp off a chip). Device safe
        states follow. Default: nothing; arms stay where they are."""

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:  # noqa: B027
        """Update the sample ledger etc. Called once after verification passes."""

    def on_give_up(self, ctx: FlowContext, obs: Observation) -> None:  # noqa: B027
        """The step stops without success after `obs` (unclear, or corrections
        used up): record what the last look showed (e.g. where an object really
        is) so the ledger does not keep a stale belief. Called before safe_state."""


class StepRunner:
    """Runs one Step with the L0 correction ladder and checkpoints it."""

    def run(self, step: Step, ctx: FlowContext, seq: int) -> StepOutcome:
        inside = step.safe_to_resume if step.pause_inside is None else step.pause_inside
        if inside:
            return self._run(step, ctx, seq)
        stop, ctx.cancel = ctx.cancel, threading.Event()  # Stop waits for the step's end
        try:
            return self._run(step, ctx, seq)
        finally:
            if ctx.cancel.is_set():
                stop.set()
            ctx.cancel = stop

    def _run(self, step: Step, ctx: FlowContext, seq: int) -> StepOutcome:
        store, rid = ctx.store, ctx.run_id
        store.begin_step(rid, seq, step.name, expected=step.expected(ctx),
                         safe_to_resume=step.safe_to_resume)
        observations: list[Observation] = []
        corrections = 0
        correction: Correction | None = None
        attempts = 0
        try:
            step.preconditions(ctx)
            while True:
                attempts += 1
                result = self._act_verify(step, ctx, correction)
                if isinstance(result, NotVerifiable):
                    store.emit("step.verified", {"seq": seq, "name": step.name,
                                                 "verdict": "not_verifiable",
                                                 "reason": result.reason}, run_id=rid)
                    step.on_success(ctx, None)
                    store.finish_step(rid, seq, "done", note=f"NOT_VERIFIABLE: {result.reason}")
                    return StepOutcome(StepStatus.UNVERIFIED, step.name, attempts, corrections,
                                       message=result.reason)
                obs = result
                observations.append(obs)
                store.emit("step.verified", {"seq": seq, "name": step.name, "attempt": attempts,
                                             **obs.to_dict()}, run_id=rid)
                if obs.verdict == Verdict.OK:
                    step.on_success(ctx, obs)
                    store.finish_step(rid, seq, "done", measurements=obs.measurements,
                                      corrections=corrections)
                    status = StepStatus.CORRECTED if corrections else StepStatus.SUCCEEDED
                    return StepOutcome(status, step.name, attempts, corrections, observations)
                if obs.verdict == Verdict.DANGEROUS:
                    raise DangerousError(f"{step.name}: dangerous observation: {obs.message}")
                if obs.verdict == Verdict.UNCLEAR or corrections >= step.max_corrections:
                    reason = ("unclear observation" if obs.verdict == Verdict.UNCLEAR
                              else f"still off after {corrections} corrections")
                    if obs.message:
                        reason += f": {obs.message}"
                    step.on_give_up(ctx, obs)
                    step.safe_state(ctx)
                    store.finish_step(rid, seq, "failed", measurements=obs.measurements,
                                      corrections=corrections, note=reason)
                    return StepOutcome(StepStatus.NEEDS_AGENT, step.name, attempts, corrections,
                                       observations, message=reason)
                correction = step.correct(ctx, obs)
                if correction is None:
                    step.on_give_up(ctx, obs)
                    step.safe_state(ctx)
                    store.finish_step(rid, seq, "failed", measurements=obs.measurements,
                                      corrections=corrections, note="no correction available")
                    return StepOutcome(StepStatus.NEEDS_AGENT, step.name, attempts, corrections,
                                       observations, message="no correction available")
                corrections += 1
                store.emit("step.corrected", {"seq": seq, "name": step.name,
                                              "attempt": attempts, "action": correction.action,
                                              "params": correction.params}, run_id=rid)
        except StopRequested:
            raise  # not a failure: the step stays in progress and the job pauses
        except PreconditionFailed as e:
            store.finish_step(rid, seq, "failed", note=f"precondition: {e.message}")
            return StepOutcome(StepStatus.NEEDS_AGENT, step.name, attempts, corrections,
                               observations, message=f"precondition failed: {e.message}",
                               error=e.to_dict())
        except DangerousError as e:
            self._safe(step, ctx)
            store.finish_step(rid, seq, "failed", corrections=corrections, note=e.message)
            return StepOutcome(StepStatus.DANGEROUS, step.name, attempts, corrections,
                               observations, message=e.message, error=e.to_dict())
        except QumusError as e:
            # Transient errors are retried by the job service around whole steps;
            # anything reaching here is handed back with safe state.
            self._safe(step, ctx)
            store.finish_step(rid, seq, "failed", corrections=corrections, note=e.message)
            return StepOutcome(StepStatus.NEEDS_AGENT, step.name, attempts, corrections,
                               observations, message=e.message, error=e.to_dict())

    #: transient errors (serial timeout, dropped frame) are retried with backoff (spec 5.2)
    transient_retries: int = 3
    backoff_s: float = 1.0

    def _act_verify(self, step: Step, ctx: FlowContext,
                    correction: Correction | None) -> Observation | NotVerifiable:
        delay = self.backoff_s
        for attempt in range(self.transient_retries + 1):
            try:
                step.act(ctx, correction)
                return step.verify(ctx)
            except TransientError as e:
                if attempt == self.transient_retries:
                    raise
                ctx.store.emit("notification", {"level": "info", "text":
                               f"{step.name}: transient error, retrying: {e.message}"},
                               run_id=ctx.run_id)
                ctx.clock.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    @staticmethod
    def _safe(step: Step, ctx: FlowContext) -> None:
        try:
            step.safe_state(ctx)
        except Exception as e:  # noqa: BLE001
            ctx.store.emit("notification", {"level": "danger",
                                            "text": f"safe_state failed for {step.name}: {e}"},
                           run_id=ctx.run_id)
