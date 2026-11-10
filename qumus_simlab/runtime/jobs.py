"""Job service (spec 5.3, 6.4).

CONTRACT. Hardware tools never run a flow inside the tool call. They submit a
job and return its id at once; the agent then calls wait_job(job_id, timeout).
So a crash of the model process never cuts a hardware flow in half.

    JobService.submit(flow, params, project_id=, requested_by=) -> run_id
    JobService.wait(run_id, timeout_s) -> JobStatus      (returns early when done)
    JobService.status(run_id) -> JobStatus
    JobService.cancel(run_id)                            (stop at next safe point)
    JobService.stop_all()                                (UI Stop button: every job pauses)
    JobService.resume(run_id)       continue a paused job (UI Continue button)
    JobService.resume_all()         continue every paused job
    JobService.retry(run_id)        re-run a parked job from its first unfinished step
                                    (a paused one: same as resume)
    JobService.reverify(run_id)     re-run only verify() of the failed step ("re-shoot");
                                    OK -> step done and the job continues, else parked again

Stop and Continue: Stop sets every job's cancel event; the job stops at its
next safe point (ctx.check_cancel(), between device moves) with that step left
in progress, exactly as if the process had died there, and becomes `paused`.
Heaters of the job are switched off and their setpoints kept in the run's meta;
arms, the stamp manipulator and vacuums stay as they are (nothing held is
dropped), the step's before_park makes its one always-safe move (lift the
stamp). Continue decides like restart recovery (runtime/recovery.py): probe the
step, resume from it or from the next one, or wait for a human when the state
cannot be told; heaters go back to their setpoints before the first step runs.

The agent's recovery menu (spec 4.3 L1) uses retry/reverify on parked jobs.
A job that parks (handed back to the agent, or waiting for a human) switches
its flow's heaters off the same way and keeps their setpoints in the run's
meta (`held_heaters`); retry brings them back before the first step runs.

A flow may change its own plan when a step fails (Flow.on_step_failed, e.g.
stacking changes a stamp whose film tore): the new params are stored in one
write, the failed step is marked `skipped` and the job goes on with the steps
the new params add after it. Step names up to the failed one stay the same and
seqs are never reused, so a resume rebuilds the same list from the stored
params; `skipped` counts as finished (FINISHED_STEP).

ThreadJobService runs jobs on worker threads in the current process. With
`serve --lab-process` it runs inside the lab process (runtime/labproc.py) and the
app process uses runtime/remote.RemoteJobService, same interface, under
runtime/supervisor.Supervisor (spec 5.3, 5.4).
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol

from pydantic import ValidationError

from qumus_simlab.core.errors import DangerousError, NeedsHumanError, QumusError, StopRequested
from qumus_simlab.devices.base import GasFlowController, Heater
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.flow import get_flow
from qumus_simlab.flows.step import NotVerifiable, StepOutcome, StepRunner, StepStatus, Verdict
from qumus_simlab.flows.workflow import WorkflowMismatch, validate_workflow
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.safety.gate import SafetyGate

TERMINAL = {"succeeded", "failed", "cancelled", "parked", "waiting_human", "paused"}
#: step statuses a resume goes past: done, or skipped by a changed plan (Flow.on_step_failed)
FINISHED_STEP = ("done", "skipped")


def _accepting(method: Callable) -> Callable:
    @wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._submit_lock:
            self._require_accepting()
            return method(self, *args, **kwargs)
    return guarded


@dataclass
class JobStatus:
    run_id: str
    flow: str
    status: str
    done: bool
    steps: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None
    error: Any = None
    workflow: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class JobService(Protocol):
    def submit(self, flow: str, params: dict[str, Any], *, project_id: str | None = None,
               requested_by: str | None = None) -> str: ...

    def wait(self, run_id: str, timeout_s: float) -> JobStatus: ...

    def status(self, run_id: str) -> JobStatus: ...

    def cancel(self, run_id: str) -> None: ...

    def stop_all(self) -> None: ...

    def resume(self, run_id: str) -> dict[str, Any]: ...

    def resume_all(self) -> list[dict[str, Any]]: ...

    def retry(self, run_id: str) -> None: ...

    def reverify(self, run_id: str) -> None: ...

    def review_waiting(self, run_id: str, *, project_id: str, failure_event_id: int,
                       reviewer: str, reason: str) -> dict[str, Any]: ...

    def complete_experiment(self, project_id: str, summary: str) -> dict[str, Any]: ...


#: statuses from which the agent's recovery menu may retry or re-verify a job
#: (retrying a paused job continues it, like the Continue button)
RECOVERABLE = {"parked", "paused"}


# Builds a FlowContext for a run; supplied by whoever wires the lab together.
ContextFactory = Callable[[str, dict[str, Any], str | None], FlowContext]


def waiting_human_failure(store: RunStore, run_id: str) -> dict[str, Any] | None:
    """The current hold's journal identity, for a human's per-job review form."""
    run = store.get_run(run_id)
    if not run or run["status"] != "waiting_human":
        return None
    row = store._x("SELECT id FROM event WHERE run_id=? AND kind='job.status' ORDER BY id DESC LIMIT 1",
                   (run_id,)).fetchone()
    if not row:
        return None
    event = store.events_after(row["id"]-1, limit=1)[0]
    if event.payload.get("status") != "waiting_human":
        return None
    return {"failure_event_id": event.id, "message": event.payload.get("message", "")}


def _review_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _unsafe_review_state(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("category") == "dangerous" or value.get("status") == "dangerous":
            return True
        if any(value.get(key) for key in ("pending_intent", "park_pending_intent", "held_reason")):
            return True
        return any(_unsafe_review_state(item) for item in value.values())
    return isinstance(value, (list, tuple)) and any(_unsafe_review_state(item) for item in value)


def saved_flow(run: dict[str, Any], *, store: RunStore | None = None):
    """One raw-source/parameter preflight, without a process context or I/O."""
    try:
        flow = get_flow(run["flow"])
        flow.validate_saved_params(run["params"], saved_run=run, store=store)
        return flow
    except WorkflowMismatch:
        raise
    except Exception as exc:
        raise WorkflowMismatch("construction_failed",
            "Saved flow source or parameters are unavailable/invalid; review before continuing",
            exception_type=type(exc).__name__) from None


def build_run_steps(run: dict[str, Any], bench: dict[str, Any], *, store: RunStore | None = None) -> tuple[Any, Any, list[Any]]:
    """Reconstruct the existing executor input, checking saved bindings before I/O."""
    saved = run["meta"].get("workflow")
    try:
        flow = saved_flow(run, store=store)
        flow._bench = bench
        params = flow.Params.model_validate(run["params"])
        steps = flow.build_steps(params)
        if saved:
            workflow = flow.build_workflow(params)
            validate_workflow(saved, workflow.manifest() if workflow is not None else None, steps)
        return flow, params, steps
    except WorkflowMismatch:
        raise
    except Exception as exc:
        if saved:
            raise WorkflowMismatch("construction_failed", "Current source cannot reconstruct the saved workflow; review before continuing",
                                   exception_type=type(exc).__name__) from None
        raise


def safe_energy_shutdown(store: RunStore, run_id: str, gate: SafetyGate, *,
                         device_ids=None, reason: str = "workflow_shutdown_failed") -> dict[str, Any]:
    """Confirm trusted heat/gas shutdown and retain failed output locks.

    This does not release existing locks or invoke motion/holding callbacks.
    Callers release locks only after all requested outputs are confirmed off.
    """
    shutdown: dict[str, Any] = {"attempted": [], "confirmed": [], "failed": []}
    pending = unconfirmed_energy_shutdown(store.get_run(run_id) or {})
    for device_id, device in gate.devices.devices.items():
        if (device_ids is not None and device_id not in device_ids
                or not isinstance(device, (Heater, GasFlowController))):
            continue
        shutdown["attempted"].append(device_id)
        try:
            output = gate.heater(device_id) if isinstance(device, Heater) else gate.gas_flow(device_id)
            output.safe_state()
            target = output.get_target()
            if (isinstance(device, Heater) and target is not None
                    or isinstance(device, GasFlowController) and target != 0):
                raise RuntimeError("Output target did not reach its disabled state")
            shutdown["confirmed"].append(device_id)
        except Exception as exc:  # best effort: a failed heater must not prevent gas shutdown
            shutdown["failed"].append(device_id)
            store.emit("notification", {"level": "danger", "reason": reason,
                "device": device_id, "exception_type": type(exc).__name__,
                "text": f"Safe shutdown was not confirmed for {device_id}"}, run_id=run_id)
    # A previously failed output disappearing from the registry is not proof
    # that it stopped. Its existing lock must survive an incompatible view.
    shutdown["failed"].extend(device for device in pending
        if (device_ids is None or device in device_ids) and device not in shutdown["attempted"])
    if shutdown["failed"]:
        shutdown["failed_device_locks_held"] = store.acquire_locks(shutdown["failed"], run_id)
        if not shutdown["failed_device_locks_held"]:
            store.emit("notification", {"level": "danger", "reason": reason,
                "text": "Failed outputs are locked by another job; operator action is required"}, run_id=run_id)
    return shutdown


def unconfirmed_energy_shutdown(run: dict[str, Any]) -> list[str]:
    return (run.get("error") or {}).get("safe_shutdown_failed") or []


def guard_failed_shutdown(store: RunStore, run_id: str, gate: SafetyGate, device_ids) -> bool:
    """A failed/paused job cannot release unconfirmed energy outputs."""
    shutdown = safe_energy_shutdown(store, run_id, gate, device_ids=device_ids, reason="energy_shutdown_failed")
    if shutdown["attempted"]:
        store.emit("job.safe_shutdown", {"safe_shutdown": shutdown}, run_id=run_id)
    if shutdown["failed"]:
        run = store.get_run(run_id)
        error = {**(run["error"] or {}), "safe_shutdown_failed": shutdown["failed"]}
        store.set_run_status(run_id, "waiting_human", error=error,
                             message="Energy shutdown is unconfirmed; output locks retained. Explicit shutdown verification is required.")
    return not shutdown["failed"]


def reject_workflow(store: RunStore, run_id: str, error: dict[str, Any], *,
                    stage: str, gate: SafetyGate, failed_outcomes: list[StepOutcome] | None = None) -> bool:
    """Disable trusted energy outputs before parking; never invoke current Steps.

    The changed flow cannot identify its old devices reliably, so inspect the
    connected registry. Failed shutdown retains locks; holding stays unchanged.
    """
    shutdown = safe_energy_shutdown(store, run_id, gate)
    if shutdown["failed"]:
        error = {**error, "safe_shutdown_failed": shutdown["failed"]}
    else:
        store.release_locks(run_id)
    store.emit("job.workflow_rejected", {"stage": stage, "error": error,
                                         "safe_shutdown": shutdown}, run_id=run_id)
    message, reported_error, result = error["message"], error, None
    if failed_outcomes:
        # An optional post-failure replan refusal cannot replace the observed
        # process failure. The refusal remains in the event above; no work resumes.
        failure = failed_outcomes[-1]
        message = failure.message
        reported_error = failure.error or failure.to_dict()
        if shutdown["failed"]:
            reported_error = {**reported_error, "safe_shutdown_failed": shutdown["failed"]}
        result = [outcome.to_dict() for outcome in failed_outcomes]
    store.set_run_status(run_id, "waiting_human", error=reported_error, result=result, message=message)
    store.emit("notification", {"level": "needs_human", "text": message,
                               "reason": "workflow_mismatch", "stage": stage}, run_id=run_id)
    return not shutdown["failed"]


class FlowExecutor:
    """Runs one flow's steps in order from a given sequence number."""

    def __init__(self, store: RunStore, gate: SafetyGate, runner: StepRunner | None = None):
        self.store, self.gate = store, gate
        self.runner = runner or StepRunner()

    def execute(self, run_id: str, ctx: FlowContext, start_seq: int = 0,
                restore: bool = False) -> str:
        """Run from `start_seq`. `restore`: first bring the heaters back to the
        setpoints they had when the job was paused (Continue)."""
        run = self.store.get_run(run_id)
        assert run is not None
        if unconfirmed_energy_shutdown(run):
            return "parked"  # Never invoke Steps or restore targets while fault locks remain.
        try:
            flow, params, steps = build_run_steps(run, ctx.bench, store=self.store)
        except WorkflowMismatch as exc:
            reject_workflow(self.store, run_id, exc.to_dict(), stage="execute", gate=ctx.gate)
            return "parked"
        release_locks = True
        normal_exit = False
        shutdown_devices = flow.shutdown_devices
        outcomes: list[StepOutcome] = []
        seq = start_seq
        try:
            if ctx.cancel.is_set():  # stopped while it waited in the queue
                locks = self.store.locks()
                shutdown_devices = tuple(device for device in flow.shutdown_devices
                                         if locks.get(device) in (None, run_id))
                return self._pause(run_id, ctx, flow, steps, start_seq, in_progress=False)
            if not self.store.acquire_locks(flow.devices, run_id):
                self.store.set_run_status(run_id, "queued", message="waiting for devices")
                release_locks, normal_exit = False, True
                return "queued"
            self.store.set_run_status(run_id, "running")
            if restore and flow.restore_heaters_on_resume:
                try:
                    self._restore_heaters(run, ctx)
                except StopRequested:
                    return self._pause(run_id, ctx, flow, steps, start_seq, in_progress=False)
                except DangerousError as e:
                    self.store.set_run_status(run_id, "failed", error=e.to_dict(),
                                              message=f"continue: {e.message}")
                    return "failed"
                except QumusError as e:
                    self.store.set_run_status(run_id, "waiting_human", error=e.to_dict(),
                                              message=f"continue: {e.message}")
                    return "parked"
            while seq < len(steps):
                step = steps[seq]
                if ctx.cancel.is_set():
                    return self._pause(run_id, ctx, flow, steps, seq, in_progress=False)
                try:
                    self.gate.watchdog_tick()
                except QumusError as e:
                    # a heater the watchdog cannot read or switch off (offline
                    # while on) is a danger between steps: everything to its
                    # safe state, the run fails with the typed error
                    failed = self.gate.devices.all_safe_state()
                    shutdown_devices = self.gate.devices.devices
                    self.store.emit("notification", {
                        "level": "danger", "text": f"{run['flow']}: watchdog: {e.message}",
                        "safe_state_failed": failed}, run_id=run_id)
                    self.store.set_run_status(run_id, "failed", error=e.to_dict(),
                                              message=f"watchdog: {e.message}")
                    return "failed"
                try:
                    out = self.runner.run(step, ctx, seq)
                except StopRequested:
                    return self._pause(run_id, ctx, flow, steps, seq, in_progress=True)
                outcomes.append(out)
                if out.status == StepStatus.DANGEROUS:
                    failed = self.gate.devices.all_safe_state()
                    shutdown_devices = self.gate.devices.devices
                    self.store.emit("notification", {
                        "level": "danger", "text": f"{run['flow']}: {out.message}",
                        "safe_state_failed": failed}, run_id=run_id)
                    self.store.set_run_status(run_id, "failed", error=out.to_dict(),
                                              message=out.message)
                    return "failed"
                if not out.ok:
                    plan = self._replan(run_id, ctx, flow, params, steps, seq, out)
                    if plan is not None:  # the flow goes on another way (Flow.on_step_failed)
                        params, steps = plan
                        seq += 1
                        continue
                    # Hand back to the agent (L1) with photos and measurements.
                    # Nobody is at the bench while it waits: the flow's heaters go
                    # off, their setpoints kept for a retry (_restore_heaters).
                    self._hold_heaters(run_id, flow)
                    self.store.set_run_status(run_id, "waiting_human" if out.error and
                                              out.error.get("category") == "needs_human"
                                              else "parked",
                                              result=[o.to_dict() for o in outcomes],
                                              message=out.message)
                    return "parked"
                seq += 1
            self.store.set_run_status(run_id, "succeeded",
                                      result=[o.to_dict() for o in outcomes])
            normal_exit = True
            return "succeeded"
        except WorkflowMismatch as exc:
            release_locks = reject_workflow(self.store, run_id, exc.to_dict(), stage="replan", gate=ctx.gate,
                failed_outcomes=outcomes if outcomes and not outcomes[-1].ok else None)
            return "parked"
        finally:
            if release_locks and not normal_exit:
                try:
                    # Abrupt worker exits have not passed _pause/_hold_heaters.
                    # Preserve known targets before mandatory shutdown so a
                    # checked safe resume can restore the original process.
                    if self.store.get_run(run_id)["status"] == "running" and flow.restore_heaters_on_resume:
                        self._hold_heaters(run_id, flow)
                finally:
                    release_locks = guard_failed_shutdown(self.store, run_id, ctx.gate, shutdown_devices)
            if release_locks:
                self.store.release_locks(run_id)

    def _replan(self, run_id: str, ctx: FlowContext, flow: Any, params: Any,
                steps: list[Any], seq: int, out: StepOutcome) -> tuple[Any, list[Any]] | None:
        """Ask the flow whether it can go on after step `seq` gave up. A change
        may only add steps after the failed one (the done steps keep their
        checkpoints); the new params are stored in one write before the failed
        step is marked `skipped`, so a crash in between re-runs that step, whose
        flow then finds the change already planned (resume builds from them)."""
        # A checked replan may extend its tail, but it cannot overwrite evidence
        # of an unrelated source/registration change since job admission.
        build_run_steps(self.store.get_run(run_id), ctx.bench, store=self.store)
        try:
            plan = flow.on_step_failed(params, steps[seq], out, ctx)
        except Exception as e:  # noqa: BLE001 - a broken replan hands back as before
            self.store.emit("notification", {"level": "danger", "text":
                            f"{flow.name}: could not plan around {steps[seq].name}: {e}"},
                            run_id=run_id)
            return None
        if plan is None:
            return None
        new_steps = flow.build_steps(plan.params)
        if len(new_steps) <= seq + 1 or [(s.name, s.safe_to_resume) for s in new_steps[:seq + 1]] != \
                [(s.name, s.safe_to_resume) for s in steps[:seq + 1]]:
            self.store.emit("notification", {"level": "danger", "text":
                            f"{flow.name}: a changed plan must keep the steps up to "
                            f"{steps[seq].name} and add steps after it; handed back"},
                            run_id=run_id)
            return None
        workflow = flow.build_workflow(plan.params)
        manifest = workflow.manifest() if workflow is not None else None
        if self.store.get_run(run_id)["meta"].get("workflow") and manifest is None:
            raise WorkflowMismatch("missing_current_workflow", "Replanning cannot remove this job's saved hierarchy")
        if manifest is not None:
            validate_workflow(manifest, manifest, new_steps)
        resolved = plan.params.model_dump(mode="json")
        with self.store._transaction():
            params_event_id = self.store.set_run_params(run_id, resolved, note=plan.note)
            if manifest is not None:
                self.store.update_run_meta(run_id, workflow=manifest)
                self.store.emit("job.workflow", {"workflow": manifest, "reason": "replanned",
                    "params_event_id": params_event_id}, run_id=run_id)
            self.store.finish_step(run_id, seq, "skipped", note=f"{out.message} -> {plan.note}")
        # The next Step uses this same live context, even without a retry.
        # Publish only the JSON-compatible snapshot that actually committed.
        ctx.params = resolved
        self.store.emit("notification", {"level": "warn", "text": f"{flow.name}: {plan.note}"},
                        run_id=run_id)
        return plan.params, new_steps

    def _flow_heaters(self, flow: Any, run_id: str) -> dict[str, float]:
        """The flow's heaters that are on, with their setpoints."""
        heaters: dict[str, float] = {}
        locks = self.store.locks()
        for dev_id in flow.shutdown_devices:
            if locks.get(dev_id) not in (None, run_id):
                continue  # Cancelling a queued job must not disable another job's heating.
            dev = self.gate.devices.devices.get(dev_id)
            if isinstance(dev, Heater):
                try:
                    target = dev.get_target()
                    if target is not None:
                        heaters[dev_id] = float(target)
                except Exception as exc:  # an unavailable saved target cannot prevent shutdown
                    self.store.emit("notification", {"level": "danger", "reason": "heater_setpoint_unavailable",
                        "device": dev_id, "exception_type": type(exc).__name__,
                        "text": f"Could not preserve {dev_id}'s setpoint; shutdown still requires confirmation"}, run_id=run_id)
        return heaters

    def _hold_heaters(self, run_id: str, flow: Any) -> dict[str, float]:
        """Switch the flow's heaters off and remember their setpoints in the run
        meta (`held_heaters`), so a retry or Continue brings them back first."""
        heaters = self._flow_heaters(flow, run_id)
        for h in heaters:
            try:
                self.gate.heater(h).off()
            except Exception as e:  # noqa: BLE001 - report, keep going
                self.store.emit("notification", {"level": "danger", "text":
                                f"could not switch {h} off while handing back: {e}"},
                                run_id=run_id)
        self.store.update_run_meta(run_id, held_heaters=heaters)
        return heaters

    def _pause(self, run_id: str, ctx: FlowContext, flow: Any, steps: list[Any], seq: int,
               in_progress: bool) -> str:
        """The Stop button reached this job at a safe point (see the module doc)."""
        ctx.cancel = threading.Event()  # the safe moves below must not stop themselves
        heaters = self._flow_heaters(flow, run_id)
        step = steps[seq] if seq < len(steps) else None
        if in_progress and step is not None:
            try:
                step.before_park(ctx)
            except Exception as e:  # noqa: BLE001 - report, keep going
                self.store.emit("notification", {"level": "danger", "text":
                                f"safe move before pausing failed: {e}"}, run_id=run_id)
        self._hold_heaters(run_id, flow)
        name = step.name if step is not None else "end"
        self.store.update_run_meta(run_id, paused={"seq": seq, "step": name,
                                                   "in_progress": in_progress,
                                                   "heaters": heaters})
        where = (f"during step {seq + 1} ({name})" if in_progress
                 else f"before step {seq + 1} ({name})")
        self.store.set_run_status(run_id, "paused", message=f"stopped by user {where}")
        locks = self.store.locks()
        owned = [device for device in flow.shutdown_devices if locks.get(device) in (None, run_id)]
        if not guard_failed_shutdown(self.store, run_id, ctx.gate, owned):
            return "parked"
        return "paused"

    def _restore_heaters(self, run: dict[str, Any], ctx: FlowContext) -> None:
        from qumus_simlab.flows.library.primitives import ramp_heater, wait_temperature

        h = ctx.verification["heating"]
        meta = run["meta"]
        held = meta.get("held_heaters")
        if held is None:  # records from before `held_heaters`
            held = (meta.get("paused") or {}).get("heaters") or {}
        for heater_id, target in held.items():
            ramp_heater(ctx, heater_id, target, h["poll_s"])
            wait_temperature(ctx, heater_id, lambda t, c=target: abs(t - c) <= h["band_c"],
                             what=f"{target:.0f} C again after a stop", timeout_s=h["timeout_s"],
                             stuck_window_s=h["stuck_window_s"],
                             stuck_delta_c=h["stuck_delta_c"], poll_s=h["poll_s"])
        if held:
            self.store.update_run_meta(run["id"], held_heaters={})

    def reverify(self, run_id: str, ctx: FlowContext | None = None) -> str:
        """Re-run verify() of the first unfinished step, without moving anything.
        OK -> the step is marked done and the flow continues from the next one."""
        run = self.store.get_run(run_id)
        assert run is not None
        if failed := unconfirmed_energy_shutdown(run):
            # An explicit reverify here confirms shutdown only. It never calls
            # process verify(), marks a Step done or restores saved targets.
            if (any(not isinstance(self.gate.devices.devices.get(device), (Heater, GasFlowController))
                    for device in failed) or not self.store.acquire_locks(failed, run_id)):
                return "parked"
            if guard_failed_shutdown(self.store, run_id, self.gate, failed):
                error = {**run["error"], "safe_shutdown_failed": []}
                self.store.emit("job.shutdown_verified", {"devices": failed}, run_id=run_id)
                self.store.set_run_status(run_id, "waiting_human", error=error,
                    message="Energy outputs confirmed off; original failure remains unfinished and requires human recovery.")
                self.store.release_locks(run_id)
            return "parked"
        assert ctx is not None
        try:
            flow, _, steps = build_run_steps(run, ctx.bench, store=self.store)
        except WorkflowMismatch as exc:
            reject_workflow(self.store, run_id, exc.to_dict(), stage="reverify", gate=ctx.gate)
            return "parked"
        seq = first_unfinished(self.store, run_id)
        if seq >= len(steps):
            return self.execute(run_id, ctx, start_seq=seq)
        if not self.store.acquire_locks(flow.devices, run_id):
            self.store.set_run_status(run_id, "queued", message="waiting for devices")
            return "queued"
        step = steps[seq]
        verified = False
        shutdown_confirmed = True
        try:
            result = step.verify(ctx)
            verified = not isinstance(result, NotVerifiable) and result.verdict == Verdict.OK
        except QumusError as e:
            self.store.set_run_status(run_id, "parked", error=e.to_dict(), message=f"re-verify failed: {e.message}")
            return "parked"
        finally:
            if not verified:
                shutdown_confirmed = guard_failed_shutdown(self.store, run_id, ctx.gate, flow.shutdown_devices)
            if shutdown_confirmed:
                self.store.release_locks(run_id)
        if not shutdown_confirmed:
            return "parked"
        if isinstance(result, NotVerifiable):
            self.store.set_run_status(run_id, "parked",
                                      message=f"{step.name} cannot be verified: {result.reason}")
            return "parked"
        self.store.emit("step.verified", {"seq": seq, "name": step.name, "attempt": "reverify",
                                          **result.to_dict()}, run_id=run_id)
        if result.verdict != Verdict.OK:
            self.store.set_run_status(run_id, "parked", result=[{
                "status": "needs_agent", "step": step.name, "observations": [result.to_dict()],
                "message": f"re-verify: {result.message or result.verdict}"}],
                message=f"re-verify: {result.message or result.verdict}")
            return "parked"
        step.on_success(ctx, result)
        self.store.finish_step(run_id, seq, "done", measurements=result.measurements,
                               note="re-verified")
        return self.execute(run_id, ctx, start_seq=seq + 1)


def first_unfinished(store: RunStore, run_id: str) -> int:
    """Sequence number of the first step that is not done (resume point)."""
    done = {s["seq"] for s in store.get_steps(run_id) if s["status"] in FINISHED_STEP}
    seq = 0
    while seq in done:
        seq += 1
    return seq


class ThreadJobService:
    def __init__(self, store: RunStore, gate: SafetyGate, context_factory: ContextFactory,
                 workers: int = 2, poll_s: float = 0.05,
                 bench: dict[str, Any] | None = None) -> None:
        self.store, self.gate = store, gate
        self.context_factory = context_factory
        self.bench = bench
        self.executor = FlowExecutor(store, gate)
        self._q: queue.Queue[tuple[str, int, str] | None] = queue.Queue()
        self._cancel: dict[str, threading.Event] = {}
        #: jobs cancel() was called on: they end cancelled instead of paused
        self._abandon: set[str] = set()
        self._done: dict[str, threading.Event] = {}
        self._active_jobs: dict[str, set[int]] = {}
        self._queued_jobs: set[str] = set()
        self._retired_threads: list[threading.Thread] = []
        self._recovery_deferred = False
        self._poll = poll_s
        self._submit_lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._shutdown_stop: threading.Thread | None = None
        self._shutdown_stop_error: str | None = None
        self._shutdown_sentinels = False
        self._shutdown_unconfirmed_energy: list[str] = []
        self._shutdown_evidence_complete = False
        self._shutdown_evidence_checked: set[str] = set()
        # A lifecycle caller retains image/store owners here if settlement times out.
        self._shutdown_owner: Any = None
        self._threads = [self._spawn(i) for i in range(workers)]

    def _require_accepting(self) -> None:
        if self._closing:
            raise ValueError("the local job service is shutting down; new work is refused")

    def _require_unowned(self, run_id: str) -> None:
        if run_id in self._active_jobs or run_id in self._queued_jobs:
            raise ValueError(f"job {run_id} is still owned by a local worker; wait for its actual exit")

    def shutdown(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Dispose local workers without abandoning their normal safety cleanup.

        A timeout is an unfinished shutdown, not permission to close the store or
        remove images. The existing Stop runs once, and no recovery is attempted.
        """
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("shutdown timeout must be finite and nonnegative")
        started = time.monotonic()
        deadline = started + timeout_s
        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        def report(*, admission_stopped: bool) -> dict[str, Any]:
            unsettled = sorted(rid for rid, done in list(self._done.items()) if not done.is_set())
            alive = [thread.name for thread in self._threads + self._retired_threads if thread.is_alive()]
            stop_pending = self._shutdown_stop is not None and self._shutdown_stop.is_alive()
            return {"version": 1, "closed": self._closed, "admission_stopped": admission_stopped,
                    "manual_execution": not self._threads,
                    "active_jobs": sorted(self._active_jobs),
                    "unsettled_jobs": unsettled, "alive_workers": alive, "stop_pending": stop_pending,
                    "stop_error": self._shutdown_stop_error,
                    "unconfirmed_energy_jobs": list(self._shutdown_unconfirmed_energy),
                    "evidence_pending": not self._shutdown_evidence_complete,
                    "safe_to_close_resources": self._closed and not unsettled and not alive
                        and not stop_pending and self._shutdown_stop_error is None
                        and not self._shutdown_unconfirmed_energy and self._shutdown_evidence_complete,
                    "wall_elapsed_s": time.monotonic() - started}

        if not self._shutdown_lock.acquire(timeout=remaining()):
            return report(admission_stopped=self._closing)
        try:
            if not self._submit_lock.acquire(timeout=remaining()):
                return report(admission_stopped=self._closing)
            try:
                self._closing = True
                if self._threads and any(not done.is_set() for done in self._done.values()) and self._shutdown_stop is None:
                    def stop() -> None:
                        try:
                            self.stop_all()
                        except BaseException as exc:
                            self._shutdown_stop_error = f"{type(exc).__name__}: {exc}"
                    self._shutdown_stop = threading.Thread(target=stop, daemon=True, name="job-shutdown-stop")
                    self._shutdown_stop.start()
            finally:
                self._submit_lock.release()
            if not self._threads:
                # A workers=0 owner may execute the executor elsewhere. This
                # service cannot certify its settlement or close its resources.
                return report(admission_stopped=True)
            if self._shutdown_stop is not None:
                self._shutdown_stop.join(remaining())
            if self._threads:
                for done in list(self._done.values()):
                    done.wait(remaining())
            settled = all(done.is_set() for done in self._done.values())
            stop_settled = self._shutdown_stop is None or not self._shutdown_stop.is_alive()
            if settled and stop_settled and not self._shutdown_sentinels:
                self._shutdown_sentinels = True
                for thread in self._threads:
                    if thread.is_alive():
                        self._q.put(None)
            if self._shutdown_sentinels:
                for thread in self._threads + self._retired_threads:
                    thread.join(remaining())
            self._closed = settled and stop_settled and all(not thread.is_alive() for thread in self._threads + self._retired_threads)
            if self._closed and not self._shutdown_evidence_complete and self.store._lock.acquire(timeout=remaining()):
                try:
                    for rid in self._done:
                        if rid in self._shutdown_evidence_checked:
                            continue
                        if remaining() <= 0:
                            break
                        try:
                            run = self.store.get_run(rid)
                            if run is not None and unconfirmed_energy_shutdown(run):
                                self._shutdown_unconfirmed_energy.append(rid)
                        except Exception as exc:
                            self._shutdown_stop_error = f"Shutdown evidence unavailable: {type(exc).__name__}: {exc}"
                        self._shutdown_evidence_checked.add(rid)
                    self._shutdown_evidence_complete = len(self._shutdown_evidence_checked) == len(self._done)
                finally:
                    self.store._lock.release()
            return report(admission_stopped=True)
        finally:
            self._shutdown_lock.release()

    @_accepting
    def review_waiting(self, run_id: str, *, project_id: str, failure_event_id: int,
                       reviewer: str, reason: str) -> dict[str, Any]:
        """Record a person's review, allowing normal recovery without executing it.

        This UI-only operation never queues a job or certifies a process result.
        It is intentionally absent from the agent tool registry.
        """
        self._require_unowned(run_id)
        if (type(failure_event_id) is not int or failure_event_id <= 0
                or not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 120
                or not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 2000):
            raise ValueError("A current failure event, reviewer and review reason are required")
        with self._submit_lock, self.store._transaction():
            run = self.store.get_run(run_id)
            if not run or not project_id or run["project_id"] != project_id:
                raise ValueError("Review only a job in the current project")
            failure = waiting_human_failure(self.store, run_id)
            if not failure or failure["failure_event_id"] != failure_event_id:
                raise ValueError("The waiting job changed; reload its current failure before reviewing")
            if run["error"] is not None and not isinstance(run["error"], dict):
                raise ValueError("The current hold has an unknown failure record")
            if unconfirmed_energy_shutdown(run) or _unsafe_review_state((run["error"], run["result"], run["meta"])):
                raise ValueError("Dangerous, unconfirmed shutdown or pending transfer state cannot be released by review")
            if any(r["status"] in ("queued", "running") for r in self.store.run_statuses()):
                raise ValueError("Finish active laboratory jobs before reviewing a hold")
            flow, _, steps = build_run_steps(run, self.bench, store=self.store)
            seq = first_unfinished(self.store, run_id)
            checkpoint = next((s for s in self.store.get_steps(run_id) if s["seq"] == seq), None)
            if (seq >= len(steps) or not checkpoint or checkpoint["status"] != "failed"
                    or not checkpoint["safe_to_resume"] or not steps[seq].safe_to_resume
                    or checkpoint["name"] != steps[seq].name):
                raise ValueError("Only a known failed step with existing safe-resume rules can be reviewed")
            row = self.store._x("SELECT id FROM event WHERE run_id=? AND kind='step.finished' AND id<? "
                                "ORDER BY id DESC LIMIT 1", (run_id, failure_event_id)).fetchone()
            finished = self.store.events_after(row["id"]-1, limit=1)[0] if row else None
            outcome = (run["result"] or [])[-1] if isinstance(run["result"], list) and run["result"] else {}
            outcome_error = outcome.get("error") if isinstance(outcome, dict) else None
            if (not finished or finished.payload.get("seq") != seq or finished.payload.get("name") != checkpoint["name"]
                    or finished.payload.get("status") != "failed" or not isinstance(outcome, dict)
                    or not isinstance(outcome_error, dict) or outcome.get("step") != checkpoint["name"]
                    or outcome_error.get("category") != "needs_human"):
                raise ValueError("The current hold lacks a known same-step human-required failure")
            if any(owner != run_id for device, owner in self.store.locks().items() if device in flow.devices):
                raise ValueError("Another job holds devices needed by this failed step")
            if any(getattr(device, "motion_active", False) for device in self.gate.devices.devices.values()):
                raise ValueError("Review cannot release a hold during instrument motion")
            targets = []
            for device_id, device in self.gate.devices.devices.items():
                if not isinstance(device, (Heater, GasFlowController)):
                    continue
                output = self.gate.heater(device_id) if isinstance(device, Heater) else self.gate.gas_flow(device_id)
                try:
                    target = output.get_target()
                except Exception as exc:
                    raise ValueError("Energy shutdown cannot be read; the job remains held") from exc
                if (isinstance(device, Heater) and target is not None
                        or isinstance(device, GasFlowController) and (type(target) not in (int, float) or target != 0)):
                    raise ValueError("All energy outputs must be confirmed off before reviewing a hold")
                targets.append({"device": device_id, "kind": "heater" if isinstance(device, Heater) else "gas_flow",
                                "target": target})
            event = self.store.events_after(failure_event_id-1, limit=1)[0]
            body = {"version": 1, "failure_event_id": failure_event_id,
                    "failure_event_sha256": _review_digest(event.payload),
                    "failed_step": {"seq": seq, "name": checkpoint["name"], "event_id": finished.id,
                                    "event_sha256": _review_digest(finished.payload)},
                    "reviewer": reviewer.strip(), "reason": reason.strip(), "source": "ui:job_review",
                    "from_status": "waiting_human", "to_status": "parked", "safe_energy_targets": targets,
                    "scope": "Human review report only; no motion, process success or material identity proved"}
            receipt = self.store.emit("job.human_reviewed", body, run_id=run_id)
            self.store.set_run_status(run_id, "parked", message="Human review recorded; normal bounded recovery is available")
            return {"job_id": run_id, "status": "parked", "review_event_id": receipt,
                    "failure_event_id": failure_event_id, "motion_started": False}

    @_accepting
    def complete_experiment(self, project_id: str, summary: str) -> dict[str, Any]:
        """Queue existing Steps, then commit completion from fresh guarded reads.

        Runs in the lab process too. The successful cooling job owns thermal
        locks during the final transaction; old observations alone cannot pass.
        """
        from qumus_simlab.flows.library.stage_cooling import (
            STAGE_DEVICES,
            stage_readings,
            stages_finished,
        )
        from qumus_simlab.records.experiment_completion import completion_error

        with self._submit_lock:
            if error := completion_error(self.store, project_id):
                return error
            runs = [r for r in self.store.list_runs() if r["project_id"] == project_id
                    and r["flow"] == "cool_experiment_stages"]
            cooling = runs[-1] if runs else None
            if cooling and cooling["status"] != "succeeded":
                active = cooling["status"] in ("queued", "running")
                return {"status": "pending" if active else "needs_attention",
                        "summary": "Experiment is not complete: stage cooling " + cooling["status"]
                            + ". Wait for the job, or inspect/recover it, then request completion again.",
                        "data": {"job_id": cooling["id"], "cooling_status": cooling["status"]}}
            if cooling:
                rid = cooling["id"]
                if not self.store.acquire_locks(STAGE_DEVICES, rid):
                    return {"status": "needs_attention", "summary": "Stage devices are busy; experiment completion is unverified."}
                try:
                    with self.store._transaction():
                        # A worker may publish succeeded just before releasing
                        # this same run's locks. Check ownership again while the
                        # transaction excludes that release and other admission.
                        locks = self.store.locks()
                        if any(locks.get(device) != rid for device in STAGE_DEVICES):
                            return {"status": "needs_attention", "summary": "Stage locks changed during final verification; retry completion after the worker settles."}
                        if error := completion_error(self.store, project_id):
                            return error
                        readings = stage_readings(self.gate)
                        if stages_finished(readings):
                            self.store.emit("experiment.stages_verified", {"project_id": project_id,
                                "cooling_job_id": rid, "below_c": 40.0, "readings": readings}, run_id=rid)
                            self.store.update_project_state(project_id, summary=summary, status="done")
                            return {"status": "ok", "summary": "Experiment complete: both stages measured strictly below 40 C, heaters off.",
                                    "data": {"cooling_job_id": rid, "readings": readings}}
                except (QumusError, KeyError) as exc:
                    return {"status": "needs_attention", "summary": "Stage completion feedback is unavailable or invalid; experiment remains incomplete.",
                            "data": {"exception_type": type(exc).__name__}}
                finally:
                    self.store.release_locks(rid)
            rid = self.submit("cool_experiment_stages", {}, project_id=project_id, requested_by="orchestrator")
            self.store.update_project_state(project_id, status="active")
            return {"status": "pending", "summary": "Experiment is cooling, not complete. Wait for this job, then call save_project_status(done) again for fresh readings.",
                    "data": {"job_id": rid, "below_c": 40.0}}

    @_accepting
    def submit(self, flow: str, params: dict[str, Any], *, project_id: str | None = None,
               requested_by: str | None = None) -> str:
        f = get_flow(flow)
        f._bench = self.bench
        # resolve + book under one lock: two submits in a row (the hands agent
        # queueing a second exfoliation while the first runs) must not plan on
        # the same free slot, bare chip or stamp
        with self._submit_lock:
            try:
                p = f.resolve_params(f.Params.model_validate(params), self.store)
            except ValidationError as e:
                raise ValueError(f"bad params for {flow}: {e}") from e
            workflow = f.build_workflow(p)
            manifest = workflow.manifest() if workflow is not None else None
            if manifest is not None:
                validate_workflow(manifest, manifest, f.build_steps(p))
            with self.store._transaction():
                rid = self.store.create_run(flow, p.model_dump(), project_id=project_id,
                                            requested_by=requested_by)
                # Origin is a submission fact from the actual device bundle,
                # never a caller parameter or a mutable run/store bench label.
                devices = list(self.gate.devices.devices.values())
                simulated = [getattr(d, "is_simulated", None) is True for d in devices]
                kind = ("simulated" if simulated and all(simulated) else
                        "mixed" if any(simulated) else "unknown")
                self.store.emit("laboratory.origin", {"version": "laboratory_origin_v1",
                    "kind": kind, "backend_id": "qumus_simlab.sim" if kind == "simulated" else "unverified_device_bundle",
                    "submission_event_id": self.store.last_event_id()}, run_id=rid)
                if manifest is not None:
                    self.store.update_run_meta(rid, workflow=manifest)
                    self.store.emit("job.workflow", {"workflow": manifest, "reason": "submitted"}, run_id=rid)
                f.reserve(p, self.store, rid)
            self.enqueue(rid)
        return rid

    @_accepting
    def enqueue(self, run_id: str, start_seq: int = 0, mode: str = "run") -> None:
        with self._submit_lock:
            self._require_accepting()
            self._require_unowned(run_id)
            self._cancel[run_id] = threading.Event()
            done = self._done.get(run_id)
            if done is None or done.is_set():  # keep an unset event: someone may wait on it
                self._done[run_id] = threading.Event()
            if self._threads:
                self._queued_jobs.add(run_id)
            self._q.put((run_id, start_seq, mode))

    @_accepting
    def recover(self) -> list[Any]:
        """Recover unowned interrupted work, preserving live local ownership.

        Active and already queued local jobs are excluded. A callback arriving
        during safety cleanup is revisited after its actual owner exits.
        """
        from qumus_simlab.runtime.recovery import plan_recovery

        self._threads = [t if t.is_alive() else self._spawn(i)
                         for i, t in enumerate(self._threads)]
        active = set(self._active_jobs)
        self._recovery_deferred = bool(active)
        decisions = plan_recovery(self.store, self.context_factory, gate=self.gate,
                                  exclude_run_ids=active | self._queued_jobs)
        for d in decisions:
            if d.action == "resume":
                self.store.set_run_status(d.run_id, "queued", message=f"recovered: {d.reason}")
                self.enqueue(d.run_id, d.start_seq, mode="resume")
            elif (ev := self._done.get(d.run_id)) is not None:
                ev.set()
        return decisions

    @_accepting
    def close_leftovers(self, reason: str) -> list[str]:
        """Start-up on a lab that was built fresh (the in-process testbench
        seeds a new world and ledger each time): runs left unfinished by the
        last process belong to a bench that no longer exists, so they are
        cancelled with `reason` instead of resumed. Real hardware and the lab
        process keep their state and use recover()."""
        closed = []
        for run in self.store.list_runs():
            if run["status"] in ("queued", "running", "paused", "parked", "waiting_human"):
                if (unconfirmed_energy_shutdown(run) and not guard_failed_shutdown(
                        self.store, run["id"], self.gate, unconfirmed_energy_shutdown(run))):
                    continue
                self.store.release_locks(run["id"])
                self.store.set_run_status(run["id"], "cancelled", message=reason)
                closed.append(run["id"])
        return closed

    def _spawn(self, i: int) -> threading.Thread:
        t = threading.Thread(target=self._work, daemon=True, name=f"job-{i}")
        t.start()
        return t

    def _recoverable(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown job {run_id}")
        if unconfirmed_energy_shutdown(run):
            raise ValueError(f"job {run_id} has unconfirmed energy shutdown; explicit shutdown re-verification is required")
        if run["status"] not in RECOVERABLE:
            raise ValueError(f"job {run_id} is {run['status']}; only parked jobs can be "
                             f"retried or re-verified")

    def _saved_binding_ok(self, run: dict[str, Any], stage: str) -> bool:
        try:
            saved_flow(run, store=self.store)
        except WorkflowMismatch as exc:
            reject_workflow(self.store, run["id"], exc.to_dict(), stage=stage, gate=self.gate)
            if stage != "worker":
                self._done.setdefault(run["id"], threading.Event()).set()
            return False
        return True

    @_accepting
    def retry(self, run_id: str) -> None:
        self._require_unowned(run_id)
        self._recoverable(run_id)
        if not self._saved_binding_ok(self.store.get_run(run_id), "retry"):
            return
        if self.store.get_run(run_id)["status"] == "paused":  # type: ignore[index]
            self.resume(run_id)
            return
        self.store.set_run_status(run_id, "queued", message="retry requested")
        # "resume": heaters switched off at the park come back first
        self.enqueue(run_id, first_unfinished(self.store, run_id), mode="resume")

    @_accepting
    def resume(self, run_id: str) -> dict[str, Any]:
        """Continue a paused job: decide like restart recovery, then run.
        Returns {run_id, action ("resume" | "park"), start_seq, reason}."""
        from qumus_simlab.runtime.recovery import decide, park

        self._require_unowned(run_id)
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown job {run_id}")
        if unconfirmed_energy_shutdown(run):
            raise ValueError(f"job {run_id} has unconfirmed energy shutdown; Continue cannot release its locks")
        if run["status"] != "paused":
            raise ValueError(f"job {run_id} is {run['status']}, not paused")
        d = decide(self.store, self.context_factory, run)
        if d.action == "park":
            park(self.store, self.context_factory, d, what="continue", gate=self.gate)
            if (ev := self._done.get(run_id)) is not None:
                ev.set()
        else:
            self.store.set_run_status(run_id, "queued", message=f"continue from step "
                                      f"{d.start_seq + 1} ({d.reason})")
            self.enqueue(run_id, d.start_seq, mode="resume")
        return dict(d.__dict__)

    @_accepting
    def resume_all(self) -> list[dict[str, Any]]:
        """Continue every paused job. A job the Stop has not reached yet (still
        between two safe points) simply goes on."""
        paused = self.store.list_runs("paused")
        for run in paused:
            self._require_unowned(run["id"])
        for rid, ev in list(self._cancel.items()):
            run = self.store.get_run(rid)
            if run is not None and run["status"] in ("running", "queued"):
                ev.clear()
        return [self.resume(r["id"]) for r in paused]

    @_accepting
    def reverify(self, run_id: str) -> None:
        self._require_unowned(run_id)
        run = self.store.get_run(run_id)
        if run is not None and unconfirmed_energy_shutdown(run):
            self.store.emit("job.shutdown_verification_requested", {}, run_id=run_id)
            self.enqueue(run_id, mode="reverify")
            return
        self._recoverable(run_id)
        if not self._saved_binding_ok(self.store.get_run(run_id), "reverify"):
            return
        if self.store.get_run(run_id)["status"] == "paused":  # type: ignore[index]
            raise ValueError(f"job {run_id} is paused by the Stop button; retry or Continue "
                             "resumes it")
        self.store.set_run_status(run_id, "queued", message="re-verify requested")
        self.enqueue(run_id, first_unfinished(self.store, run_id), mode="reverify")

    def _work(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            rid, start, mode = item
            owner = threading.get_ident()
            retiring = False
            with self._submit_lock:
                self._queued_jobs.discard(rid)
                self._active_jobs.setdefault(rid, set()).add(owner)
            try:
                if not self._execute_queued(rid, start, mode):
                    retiring = True
                    return
            finally:
                with self._submit_lock:
                    active = self._active_jobs[rid]
                    active.discard(owner)
                    if not active:
                        del self._active_jobs[rid]
                        run = self.store.get_run(rid)
                        # This is actual executor exit, including a crash whose
                        # safety finally produced a durable hold. Status alone
                        # while another executor owns the job is insufficient.
                        if run is not None and run["status"] in TERMINAL:
                            self._done.setdefault(rid, threading.Event()).set()
                    if not self._closing:
                        if retiring:
                            current = threading.current_thread()
                            self._retired_threads.append(current)
                            self._threads = [self._spawn(i) if thread is current else thread
                                             for i, thread in enumerate(self._threads)]
                        if self._recovery_deferred:
                            self.recover()

    def _execute_queued(self, rid: str, start: int, mode: str) -> bool:
        run = self.store.get_run(rid)
        if run is None:
            return True
        try:
            if unconfirmed_energy_shutdown(run):
                result = self.executor.reverify(rid) if mode == "reverify" else "parked"
            elif not self._saved_binding_ok(run, "worker"):
                result = "parked"
            else:
                ctx = self.context_factory(rid, run["params"], run["project_id"])
                ctx.cancel = self._cancel.setdefault(rid, threading.Event())
                if mode == "reverify":
                    result = self.executor.reverify(rid, ctx)
                else:
                    result = self.executor.execute(rid, ctx, start_seq=start, restore=mode == "resume")
        except NeedsHumanError as e:
            if unconfirmed_energy_shutdown(self.store.get_run(rid)):
                result = "parked"
            else:
                self.store.set_run_status(rid, "cancelled", message=e.message)
                result = "cancelled"
        except QumusError as e:
            if unconfirmed_energy_shutdown(self.store.get_run(rid)):
                result = "parked"
            else:
                self.store.set_run_status(rid, "failed", error=e.to_dict(), message=e.message)
                result = "failed"
        except Exception as e:  # noqa: BLE001 - never kill the worker
            if unconfirmed_energy_shutdown(self.store.get_run(rid)):
                result = "parked"
            else:
                self.store.set_run_status(rid, "failed", error={"message": repr(e)}, message=repr(e))
                result = "failed"
        except BaseException:
            # A still-running interrupted job remains unfinished for recovery.
            return False
        if result == "paused" and rid in self._abandon:
            self._abandon.discard(rid)
            self.store.set_run_status(rid, "cancelled", message="cancelled")
        if result == "queued":
            time.sleep(self._poll)
            with self._submit_lock:
                if self._closed:
                    return False
                # Only this already-owned entry may drain cancellation while
                # shutting down; no new/recovery work can be admitted.
                self._queued_jobs.add(rid)
                self._q.put((rid, start, mode))
        return True

    def status(self, run_id: str) -> JobStatus:
        with self._submit_lock:
            run = self.store.get_run(run_id)
            owned = run_id in self._active_jobs or run_id in self._queued_jobs
        if run is None:
            raise KeyError(f"unknown job {run_id}")
        return JobStatus(run_id=run_id, flow=run["flow"], status=run["status"],
                         done=run["status"] in TERMINAL and not owned, steps=self.store.get_steps(run_id),
                         result=run["result"], error=run["error"],
                         workflow=run["meta"].get("workflow"))

    def wait(self, run_id: str, timeout_s: float) -> JobStatus:
        """Block the caller (wait_job) up to `timeout_s` of wall time: how long
        an agent's tool call may hold its turn, not lab time."""
        ev = self._done.get(run_id)
        if ev is not None:
            ev.wait(timeout_s)
        return self.status(run_id)

    def cancel(self, run_id: str) -> None:
        """Stop one job at its next safe point for good: it pauses like at Stop
        and is then marked cancelled (not resumable)."""
        run = self.store.get_run(run_id)
        if run is not None and run["status"] == "paused":
            self.store.set_run_status(run_id, "cancelled", message="cancelled while paused")
            return
        self._abandon.add(run_id)
        ev = self._cancel.get(run_id)
        if ev:
            ev.set()

    def pause(self, run_id: str) -> None:
        """Stop one job at its next safe point like Stop does (resumable with
        Continue). Used by one sim-lab module's /api/stop (qumus_simlab/simlab)."""
        ev = self._cancel.get(run_id)
        if ev:
            ev.set()

    def stop_all(self) -> None:
        """Every job pauses at its next safe point (see the module doc). Heaters
        no job holds are switched off now."""
        for ev in self._cancel.values():
            ev.set()
        locked = self.store.locks()
        for dev in self.gate.devices.devices.values():
            if isinstance(dev, Heater) and dev.device_id not in locked \
                    and dev.get_target() is not None:
                try:
                    self.gate.heater(dev.device_id).off()
                except Exception as e:  # noqa: BLE001 - report, keep going
                    self.store.emit("notification", {"level": "danger", "text":
                                    f"could not switch {dev.device_id} off: {e}"})
