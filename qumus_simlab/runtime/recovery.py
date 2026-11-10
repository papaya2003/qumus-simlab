"""Restart recovery (spec 8.2).

For every run that was `running` when the process died: find its last
in-progress step and ask that step to probe the real machine state.

    consistent and done      -> mark step done, resume from the next one
    consistent and not done  -> resume from this step
    inconsistent / unknown   -> safe state, park the run, notify a human
    safe_to_resume = False   -> always wait for a human

Continue after the Stop button (runtime/jobs.py `resume`) decides the same way
for a paused run: a stop is a crash at a safe point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qumus_simlab.core.errors import QumusError
from qumus_simlab.devices.base import GasFlowController, Heater
from qumus_simlab.flows.step import ProbeResult
from qumus_simlab.flows.workflow import WorkflowMismatch
from qumus_simlab.runtime.jobs import (
    FINISHED_STEP,
    ContextFactory,
    build_run_steps,
    guard_failed_shutdown,
    reject_workflow,
    saved_flow,
    unconfirmed_energy_shutdown,
)
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.safety.gate import SafetyGate


@dataclass
class RecoveryDecision:
    run_id: str
    action: str  # "resume" | "park"
    start_seq: int
    reason: str
    workflow_error: dict[str, Any] | None = None


def decide(store: RunStore, context_factory: ContextFactory,
           run: dict[str, Any]) -> RecoveryDecision:
    """Where one interrupted run (a dead process, or a pause from the Stop
    button) can go on: probe its in-progress step, if any."""
    rid = run["id"]
    steps = store.get_steps(rid)
    in_prog = [s for s in steps if s["status"] == "in_progress"]
    done = [s["seq"] for s in steps if s["status"] in FINISHED_STEP]
    next_seq = max(done) + 1 if done else 0
    if unconfirmed_energy_shutdown(run):
        return RecoveryDecision(rid, "park", next_seq,
            "Energy shutdown is unconfirmed; retain output locks and explicitly reverify shutdown before recovery")
    # Raw procedure admission precedes context construction, probing, hooks and
    # the no-in-progress fast path, including legacy flat transfer jobs.
    try:
        saved_flow(run, store=store)
    except WorkflowMismatch as exc:
        seq = in_prog[-1]["seq"] if in_prog else next_seq
        return RecoveryDecision(rid, "park", seq, str(exc), workflow_error=exc.to_dict())
    saved = run["meta"].get("workflow")
    if saved:
        ctx = context_factory(rid, run["params"], run["project_id"])
        try:
            _, _, current_steps = build_run_steps(run, ctx.bench, store=store)
        except WorkflowMismatch as exc:
            seq = in_prog[-1]["seq"] if in_prog else next_seq
            return RecoveryDecision(rid, "park", seq, str(exc), workflow_error=exc.to_dict())
    # Retain energy locks until resumed execution or checked shutdown. Ordinary
    # placement-only stale locks retain their historical release behavior below.
    if not in_prog:
        return RecoveryDecision(rid, "resume", next_seq, "no step in progress")
    s = in_prog[-1]
    if not s["safe_to_resume"]:
        return RecoveryDecision(rid, "park", s["seq"], f"step {s['name']} is not safe to resume")
    if not saved:
        ctx = context_factory(rid, run["params"], run["project_id"])
        try:
            _, _, current_steps = build_run_steps(run, ctx.bench, store=store)
        except WorkflowMismatch as exc:
            return RecoveryDecision(rid, "park", s["seq"], str(exc), workflow_error=exc.to_dict())
    if not any(owner == rid and isinstance(ctx.gate.devices.devices.get(device), (Heater, GasFlowController))
               for device, owner in store.locks().items()):
        store.release_locks(rid)
    step = current_steps[s["seq"]]
    try:
        probe = step.probe_state(ctx)
    except QumusError as e:  # e.g. the camera is offline: cannot tell
        probe = ProbeResult.UNKNOWN
        store.emit("notification", {"level": "warn", "text": f"probe of {s['name']} "
                                    f"failed: {e.message}"}, run_id=rid)
    if probe == ProbeResult.DONE:
        step.on_success(ctx, None)
        store.finish_step(rid, s["seq"], "done", note="recovered: probe says done")
        return RecoveryDecision(rid, "resume", s["seq"] + 1, "probe: done")
    if probe == ProbeResult.NOT_DONE:
        return RecoveryDecision(rid, "resume", s["seq"], "probe: not done")
    return RecoveryDecision(rid, "park", s["seq"], "probe: unknown state")


def park(store: RunStore, context_factory: ContextFactory, d: RecoveryDecision,
         what: str = "recovery", *, gate: SafetyGate) -> None:
    """Safe state for the step that was running, then wait for a human."""
    run = store.get_run(d.run_id)
    if unconfirmed_energy_shutdown(run):
        store.set_run_status(d.run_id, "waiting_human", message=d.reason)
        store.emit("notification", {"level": "needs_human", "reason": "energy_shutdown_failed",
                                    "text": d.reason}, run_id=d.run_id)
        return
    error = d.workflow_error or _safe_state(store, context_factory, d)
    if unconfirmed_energy_shutdown(store.get_run(d.run_id)):
        return  # Checked shutdown already parked it while retaining fault locks.
    if error is not None:
        reject_workflow(store, d.run_id, error, stage=what, gate=gate)
        return
    store.set_run_status(d.run_id, "waiting_human", message=d.reason)
    store.emit("notification", {"level": "needs_human",
                                "text": f"{what} parked {d.run_id}: {d.reason}"},
               run_id=d.run_id)


def plan_recovery(store: RunStore, context_factory: ContextFactory, *,
                  gate: SafetyGate, exclude_run_ids: set[str] | None = None) -> list[RecoveryDecision]:
    decisions = [decide(store, context_factory, run)
                 for run in store.list_runs("running") + store.list_runs("queued")
                 if run["id"] not in (exclude_run_ids or set())]
    for d in decisions:
        if d.action == "park":
            park(store, context_factory, d, gate=gate)
    return decisions


def _safe_state(store: RunStore, context_factory: ContextFactory,
                d: RecoveryDecision) -> dict[str, Any] | None:
    """Put the devices of the step that was running into their safe state
    (heaters off, ...) before the run waits for a human. Device-level safe
    states never move an arm or drop what it holds; a step's before_park may
    make its one always-safe move first (lift the stamp off a chip)."""
    run = store.get_run(d.run_id)
    assert run is not None
    try:
        saved_flow(run, store=store)
    except WorkflowMismatch as exc:
        return exc.to_dict()
    ctx = context_factory(d.run_id, run["params"], run["project_id"])
    try:
        flow, _, steps = build_run_steps(run, ctx.bench, store=store)
    except WorkflowMismatch as exc:
        return exc.to_dict()
    devices = steps[d.start_seq].devices if d.start_seq < len(steps) else flow.devices
    devices = tuple(device for device in devices if device in flow.shutdown_devices)
    if d.start_seq < len(steps):
        try:
            steps[d.start_seq].before_park(ctx)
        except Exception as e:  # noqa: BLE001 - report, keep going
            store.emit("notification", {"level": "danger", "text":
                       f"safe move before parking failed during recovery: {e}"},
                       run_id=d.run_id)
    for dev in devices:
        try:
            ctx.gate.devices.get(dev).safe_state()
            ctx.gate.log(dev, "safe_state", reason="recovery")
        except Exception as e:  # noqa: BLE001 - report, keep going
            store.emit("notification", {"level": "danger", "text":
                       f"safe_state of {dev} failed during recovery: {e}"}, run_id=d.run_id)
    # The current checkpoint may only use optics/motion while an earlier Step
    # left a heater on. Energy verification covers the full flow before unlock.
    if guard_failed_shutdown(store, d.run_id, ctx.gate, flow.shutdown_devices):
        store.release_locks(d.run_id)
