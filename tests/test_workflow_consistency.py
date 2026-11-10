"""Saved hierarchy blocks process actions; trusted energy shutdown stays available."""
from __future__ import annotations

import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from qumus_simlab.core.clock import SimClock
from qumus_simlab.devices.base import DeviceBundle
from qumus_simlab.flows.flow import FLOWS, Flow
from qumus_simlab.flows.step import Observation, ProbeResult, Step, Verdict
from qumus_simlab.flows.workflow import Assembly, Molecule
from qumus_simlab.runtime.jobs import ThreadJobService
from qumus_simlab.runtime.recovery import decide, park
from qumus_simlab.runtime.store import RunStore


class Params(BaseModel):
    pass


@pytest.fixture
def admitted(monkeypatch):
    trace = []

    class Checked(Step):
        def __init__(self, name, safe=True):
            self.name, self.safe_to_resume = name, safe

        def act(self, ctx, correction):
            trace.append(("act", self.name))

        def verify(self, ctx):
            trace.append(("verify", self.name))
            return Observation(Verdict.OK)

        def probe_state(self, ctx):
            trace.append(("probe", self.name))
            return ProbeResult.DONE

        def before_park(self, ctx):
            trace.append(("before_park", self.name))

        def on_success(self, ctx, obs):
            trace.append(("on_success", self.name))

    def hierarchy(*, first="first", group="first_operation", safe=True, description="Original purpose"):
        return Assembly("consistency_fixture", description, (
            Molecule(group, "First operation", (Checked(first, safe),)),
            Molecule("second_operation", "Second operation", (Checked("second"),))))

    class OriginalFlow(Flow):
        name = "consistency_fixture"
        Params = Params

        def build_workflow(self, params):
            return hierarchy()

    monkeypatch.setitem(FLOWS, OriginalFlow.name, OriginalFlow)
    store = RunStore(":memory:", clock=SimClock())
    gate = SimpleNamespace(watchdog_tick=lambda: trace.append(("watchdog",)), devices=DeviceBundle())

    def context(run_id, params, project_id):
        return SimpleNamespace(store=store, run_id=run_id, params=params, project_id=project_id,
                               bench={}, cancel=threading.Event(), gate=gate)

    service = ThreadJobService(store, gate, context, workers=0)
    # Use real admission/persistence, not a hand-authored saved tree.
    rid = service.submit(OriginalFlow.name, {})
    service._q.get_nowait()
    service.executor._restore_heaters = lambda *args: trace.append(("restore_heaters",))
    service.executor._hold_heaters = lambda *args: trace.append(("hold_heaters",))
    assert not trace
    result = SimpleNamespace(store=store, gate=gate, factory=context, service=service,
        rid=rid, trace=trace, hierarchy=hierarchy, Checked=Checked, Original=OriginalFlow)
    yield result
    store.close()


def replace_source(h, monkeypatch, variant):
    class Updated(h.Original):
        pass

    if variant == "renamed_step":
        Updated.build_workflow = lambda self, p: h.hierarchy(first="source_updated_step")
    elif variant == "changed_molecule":
        Updated.build_workflow = lambda self, p: h.hierarchy(group="updated_operation")
    elif variant == "changed_recovery_flag":
        Updated.build_workflow = lambda self, p: h.hierarchy(safe=False)
    elif variant == "actual_steps_disagree":
        Updated.build_steps = lambda self, p: [h.Checked("second"), h.Checked("first")]
    elif variant == "hierarchy_removed":
        Updated.build_workflow = lambda self, p: None
        Updated.build_steps = lambda self, p: [h.Checked("first"), h.Checked("second")]
    elif variant == "construction_failure":
        def broken(self, p):
            raise ValueError("Do not copy private exception details into the event")
        Updated.build_steps = broken
    elif variant == "unregistered_flow":
        monkeypatch.delitem(FLOWS, h.Original.name)
        return
    elif variant == "description_only":
        Updated.build_workflow = lambda self, p: h.hierarchy(description="Clarified explanatory copy")
    else:
        raise AssertionError(variant)
    monkeypatch.setitem(FLOWS, h.Original.name, Updated)


def assert_rejected(h, before_meta, before_steps):
    run = h.store.get_run(h.rid)
    assert run["status"] == "waiting_human"
    assert run["error"]["code"] == "workflow_mismatch"
    assert "private exception" not in str(run["error"])
    assert run["meta"] == before_meta
    assert h.store.get_steps(h.rid) == before_steps
    assert not h.trace  # Only trusted heat/gas shutdown is allowed; this fixture has no outputs.
    evidence = [e for e in h.store.events_after(0, limit=10000, run_id=h.rid) if e.kind == "job.workflow_rejected"]
    assert evidence and evidence[-1].payload["error"] == run["error"]
    assert len([e for e in h.store.events_after(0, limit=10000, run_id=h.rid) if e.kind == "job.workflow"]) == 1
    assert "retained_slot" in h.store.reservations("slot")


@pytest.mark.parametrize("variant", ["renamed_step", "changed_molecule", "changed_recovery_flag",
    "actual_steps_disagree", "hierarchy_removed", "construction_failure", "unregistered_flow"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_registered_source_variants_are_blocked_before_restore_or_pause(admitted, monkeypatch, variant, cancelled):
    h = admitted
    h.store.update_run_meta(h.rid, held_heaters={"previous_heater": 300})
    h.store.reserve("slot", "retained_slot", h.rid)
    original = deepcopy(h.store.get_run(h.rid)["meta"])
    replace_source(h, monkeypatch, variant)
    ctx = h.factory(h.rid, {}, None)
    if cancelled:
        ctx.cancel.set()
    assert h.service.executor.execute(h.rid, ctx, restore=True) == "parked"
    assert_rejected(h, original, [])


@pytest.mark.parametrize("entry", ["direct_reverify", "resume_no_step", "resume_in_progress",
    "resume_unsafe", "recovery_queued", "recovery_running", "retry", "reverify"])
@pytest.mark.parametrize("variant", ["renamed_step", "actual_steps_disagree"])
def test_recovery_entry_points_cannot_bypass_the_saved_plan(admitted, monkeypatch, entry, variant):
    h = admitted
    h.store.update_run_meta(h.rid, held_heaters={"previous_heater": 300})
    h.store.reserve("slot", "retained_slot", h.rid)
    if entry in ("direct_reverify", "resume_in_progress", "resume_unsafe", "recovery_running", "reverify"):
        h.store.begin_step(h.rid, 0, "first", safe_to_resume=entry != "resume_unsafe")
    original, before_steps = deepcopy(h.store.get_run(h.rid)["meta"]), h.store.get_steps(h.rid)
    replace_source(h, monkeypatch, variant)
    if entry == "direct_reverify":
        assert h.service.executor.reverify(h.rid, h.factory(h.rid, {}, None)) == "parked"
    elif entry.startswith("resume"):
        h.store.set_run_status(h.rid, "paused")
        decision = h.service.resume(h.rid)
        assert decision["action"] == "park" and decision["workflow_error"]["code"] == "workflow_mismatch"
        assert h.service._q.empty()
    elif entry.startswith("recovery"):
        h.store.set_run_status(h.rid, "running" if entry.endswith("running") else "queued")
        decisions = h.service.recover()
        assert len(decisions) == 1 and decisions[0].action == "park"
        assert h.service._q.empty()
    else:
        h.store.set_run_status(h.rid, "parked")
        # Exercise the actual worker routes: retry restores heaters through
        # execute, whereas reverify can otherwise call verify before execute.
        worker = ThreadJobService(h.store, h.gate, h.factory, workers=1)
        worker.executor._restore_heaters = lambda *args: h.trace.append(("restore_heaters",))
        getattr(worker, entry)(h.rid)
        assert worker.wait(h.rid, 3).status == "waiting_human"
    assert_rejected(h, original, before_steps)


def test_current_step_cleanup_is_checked_again_between_decide_and_park(admitted, monkeypatch):
    h = admitted
    h.store.begin_step(h.rid, 0, "first", safe_to_resume=False)
    h.store.set_run_status(h.rid, "paused")
    h.store.reserve("slot", "retained_slot", h.rid)
    original, before_steps = deepcopy(h.store.get_run(h.rid)["meta"]), h.store.get_steps(h.rid)
    decision = decide(h.store, h.factory, h.store.get_run(h.rid))
    assert decision.action == "park" and decision.workflow_error is None
    replace_source(h, monkeypatch, "renamed_step")
    park(h.store, h.factory, decision, gate=h.gate)
    assert_rejected(h, original, before_steps)


@pytest.mark.parametrize("bad", [{"name": "partial"}, {"level": "assembly", "children": []}, "invalid"])
def test_nonempty_malformed_saved_hierarchy_is_not_legacy_compatibility(admitted, bad):
    h = admitted
    h.store.update_run_meta(h.rid, workflow=bad)
    h.store.reserve("slot", "retained_slot", h.rid)
    original = deepcopy(h.store.get_run(h.rid)["meta"])
    assert h.service.executor.execute(h.rid, h.factory(h.rid, {}, None)) == "parked"
    assert_rejected(h, original, [])


@pytest.mark.parametrize("saved", [None, {}])
def test_old_jobs_without_hierarchy_keep_flat_compatibility_without_backfill(admitted, monkeypatch, saved):
    h = admitted
    h.store.update_run_meta(h.rid, workflow=saved)
    replace_source(h, monkeypatch, "renamed_step")
    assert h.service.executor.execute(h.rid, h.factory(h.rid, {}, None)) == "succeeded"
    assert h.store.get_run(h.rid)["meta"]["workflow"] == saved
    assert [(s["seq"], s["name"]) for s in h.store.get_steps(h.rid)] == [(0, "source_updated_step"), (1, "second")]
    assert not any(e.kind == "job.workflow_rejected" for e in h.store.events_after(0, limit=10000))


def test_description_only_update_keeps_the_original_saved_evidence(admitted, monkeypatch):
    h = admitted
    original = deepcopy(h.store.get_run(h.rid)["meta"]["workflow"])
    replace_source(h, monkeypatch, "description_only")
    assert h.service.executor.execute(h.rid, h.factory(h.rid, {}, None)) == "succeeded"
    assert h.store.get_run(h.rid)["meta"]["workflow"] == original


def test_new_admission_rejects_a_declared_hierarchy_that_disagrees_with_execution(admitted, monkeypatch):
    h = admitted
    before = h.store.list_runs(), h.store.events_after(0, limit=10000)
    replace_source(h, monkeypatch, "actual_steps_disagree")
    with pytest.raises(ValueError, match="Actual execution Steps"):
        h.service.submit(h.Original.name, {})
    assert (h.store.list_runs(), h.store.events_after(0, limit=10000)) == before
    assert not h.trace


@pytest.mark.parametrize("with_unsafe_step", [False, True])
def test_known_legacy_recovery_keeps_early_decisions_without_process_context(admitted, monkeypatch, with_unsafe_step):
    h = admitted
    h.store.update_run_meta(h.rid, workflow=None)
    if with_unsafe_step:
        h.store.begin_step(h.rid, 0, "old_unsafe_step", safe_to_resume=False)
    def no_context(*args):
        raise AssertionError("Legacy no-progress and unsafe decisions must not reconstruct a flow")

    decision = decide(h.store, no_context, h.store.get_run(h.rid))
    assert decision.action == ("park" if with_unsafe_step else "resume")
    assert decision.workflow_error is None


def test_checked_tail_replan_keeps_both_manifests_and_runs_the_new_tail(admitted, monkeypatch):
    from qumus_simlab.flows.flow import Replan

    h = admitted

    class TailParams(BaseModel):
        extended: bool = False
        route: tuple[str, ...] = ("original",)

    seen = []

    class ReadNewContext(h.Checked):
        def act(self, ctx, correction):
            assert ctx.params == {"extended": True, "route": ["replacement"]}
            assert ctx.params == h.store.get_run(ctx.run_id)["params"]
            seen.append(ctx)
            super().act(ctx, None)

    class RefuseFirst(h.Checked):
        def verify(self, ctx):
            return Observation(Verdict.UNCLEAR, message="Exercise the registered tail extension")

    class TailFlow(Flow):
        name = "checked_tail_extension"
        Params = TailParams

        def build_workflow(self, params):
            steps = (RefuseFirst("first"),)
            if params.extended:
                steps += (ReadNewContext("new_tail"),)
            return Molecule("checked_operation", "A checked in-flow tail extension", steps)

        def on_step_failed(self, params, step, outcome, ctx):
            return Replan(params.model_copy(update={"extended": True, "route": ("replacement",)}),
                          "Add a reviewed replacement tail")

    monkeypatch.setitem(FLOWS, TailFlow.name, TailFlow)
    rid = h.service.submit(TailFlow.name, {})
    original = deepcopy(h.store.get_run(rid)["meta"]["workflow"])
    ctx = h.factory(rid, h.store.get_run(rid)["params"], None)
    ctx.clock, ctx.counters, ctx.history = h.store.clock, {"moves": 7}, ["original evidence"]
    original_objects = (ctx.clock, ctx.gate, ctx.cancel, ctx.counters, ctx.history)
    assert h.service.executor.execute(rid, ctx) == "succeeded"
    assert seen == [ctx]
    assert all(before is after for before, after in zip(original_objects,
        (ctx.clock, ctx.gate, ctx.cancel, ctx.counters, ctx.history), strict=True))
    assert ctx.counters == {"moves": 7} and ctx.history == ["original evidence"]
    run = h.store.get_run(rid)
    assert run["params"]["extended"] and run["meta"]["workflow"] != original
    assert [(step["name"], step["status"]) for step in h.store.get_steps(rid)] == [("first", "skipped"), ("new_tail", "done")]
    evidence = [e.payload for e in h.store.events_after(0, limit=10000, run_id=rid) if e.kind == "job.workflow"]
    assert [e["reason"] for e in evidence] == ["submitted", "replanned"]
    assert evidence[0]["workflow"] == original and evidence[1]["workflow"] == run["meta"]["workflow"]
    linked = h.store.events_after(evidence[1]["params_event_id"]-1, limit=1, run_id=rid)[0]
    assert linked.kind == "job.params_changed" and linked.payload == {
        "path": [], "value": run["params"], "note": "Add a reviewed replacement tail"}


@pytest.mark.parametrize("failure", ["unclear", "needs_human"])
@pytest.mark.parametrize("binding", ["reference", "source"])
@pytest.mark.parametrize("entry", ["retry", "reverify", "resume", "recover", "worker", "executor"])
def test_postfailure_replan_refusal_preserves_outcome_and_blocks_continuation(admitted, monkeypatch, failure, binding, entry):
    from qumus_simlab.core.errors import NeedsHumanError
    from qumus_simlab.flows.workflow import WorkflowMismatch

    h = admitted
    message = "Target material appearance is missing or ambiguous; no automatic re-press"
    original_error = NeedsHumanError(message).to_dict()
    saved_validate = h.Original.validate_saved_params

    def validate(self, raw, *, saved_run=None, store=None):
        saved_validate(self, raw, saved_run=saved_run, store=store)
        if binding == "reference" and saved_run and saved_run["meta"].get("observed_failed_endpoint"):
            raise WorkflowMismatch("transfer_reference_changed", "Saved endpoint cannot authorize continuation")

    def observed_failure(self, ctx):
        h.trace.append(("original_verify", self.name))
        ctx.store.update_run_meta(ctx.run_id, observed_failed_endpoint=True)
        if binding == "source":
            replace_source(h, monkeypatch, "actual_steps_disagree")
        if failure == "needs_human":
            raise NeedsHumanError(message)
        return Observation(Verdict.UNCLEAR, {"release_confirmed": False, "process_confirmation": False},
                           images=["retained-target.png"], message=message)

    monkeypatch.setattr(h.Original, "validate_saved_params", validate)
    monkeypatch.setattr(h.Checked, "verify", observed_failure)
    monkeypatch.setattr(h.Original, "on_step_failed", lambda *args: pytest.fail("refused replan cannot invoke a flow callback"))
    ctx = h.factory(h.rid, {}, None)
    original_manifest = deepcopy(h.store.get_run(h.rid)["meta"]["workflow"])
    assert h.service.executor.execute(h.rid, ctx) == "parked"
    run = h.store.get_run(h.rid)
    assert run["status"] == "waiting_human"
    [outcome] = run["result"]
    assert outcome["status"] == "needs_agent" and outcome["step"] == "first"
    if failure == "unclear":
        assert run["error"] == outcome
        assert outcome["observations"][0]["measurements"] == {"release_confirmed": False, "process_confirmation": False}
        assert outcome["observations"][0]["images"] == ["retained-target.png"]
    else:
        assert run["error"] == outcome["error"] == original_error
    assert message in outcome["message"]
    events = h.store.events_after(0, limit=10000, run_id=h.rid)
    [rejection] = [e.payload for e in events if e.kind == "job.workflow_rejected"]
    assert rejection["stage"] == "replan" and rejection["error"]["code"] == "workflow_mismatch"
    assert events[-1].payload["text"] == outcome["message"]
    assert h.store.get_run(h.rid)["meta"]["workflow"] == original_manifest
    assert [(s["name"], s["status"]) for s in h.store.get_steps(h.rid)] == [("first", "failed")]
    assert h.trace == [("watchdog",), ("act", "first"), ("original_verify", "first")]

    # A refused automatic proposal never grants a user retry/resume new authority.
    original_result, original_steps = run["result"], h.store.get_steps(h.rid)
    h.trace.clear()
    if binding == "reference":
        h.service.context_factory = lambda *args: pytest.fail("raw binding refusal must precede context")
    h.service.executor._restore_heaters = lambda *args: pytest.fail("continuation refusal must precede restoration")
    h.store.set_run_status(h.rid, "paused" if entry == "resume" else "running" if entry == "recover" else "parked")
    if entry == "executor":
        assert h.service.executor.execute(h.rid, ctx, restore=True) == "parked"
    elif entry == "worker":
        h.service.enqueue(h.rid)
        h.service._spawn(94)
        assert h.service._done[h.rid].wait(1)
    else:
        getattr(h.service, entry)(h.rid) if entry != "recover" else h.service.recover()
        if binding == "source" and entry in ("retry", "reverify"):
            h.service._spawn(94)
            assert h.service._done[h.rid].wait(1)
    rejected = h.store.get_run(h.rid)
    assert rejected["status"] == "waiting_human" and rejected["error"]["code"] == "workflow_mismatch"
    assert rejected["result"] == original_result and h.store.get_steps(h.rid) == original_steps
    assert not h.trace


@pytest.mark.parametrize("shutdown_failure", [False, True])
def test_postfailure_refusal_preserves_shutdown_locks_and_off_only_verification(admitted, monkeypatch, shutdown_failure):
    from qumus_simlab.devices.base import Heater
    from qumus_simlab.flows.workflow import WorkflowMismatch

    h = admitted

    class Output(Heater):
        target = 170.
        fail = shutdown_failure

        def set_target(self, celsius):
            pytest.fail("failure handling must not restore heating")

        def get_target(self):
            return self.target

        def read_temperature(self):
            pytest.fail("failure handling needs no process-temperature probe")

        def off(self):
            if self.fail:
                raise RuntimeError("private output failure")
            self.target = None

    output = Output("fixture_heater")
    h.gate.devices.add(output)
    h.gate.heater = lambda _: output
    monkeypatch.setattr(h.Original, "devices", (output.device_id,))
    saved_validate = h.Original.validate_saved_params

    def validate(self, raw, *, saved_run=None, store=None):
        saved_validate(self, raw, saved_run=saved_run, store=store)
        if saved_run and saved_run["meta"].get("observed_failed_endpoint"):
            raise WorkflowMismatch("transfer_reference_changed", "Endpoint cannot authorize continuation")

    def observed_failure(self, ctx):
        h.trace.append(("original_verify", self.name))
        ctx.store.update_run_meta(ctx.run_id, observed_failed_endpoint=True)
        return Observation(Verdict.UNCLEAR, {"release_confirmed": False}, message="Target appearance unconfirmed")

    monkeypatch.setattr(h.Original, "validate_saved_params", validate)
    monkeypatch.setattr(h.Checked, "verify", observed_failure)
    monkeypatch.setattr(h.Original, "on_step_failed", lambda *args: pytest.fail("refused replan cannot call the flow"))
    assert h.service.executor.execute(h.rid, h.factory(h.rid, {}, None)) == "parked"
    run = h.store.get_run(h.rid)
    assert run["error"]["status"] == "needs_agent"
    assert run["result"][-1]["observations"][0]["message"] == "Target appearance unconfirmed"
    [rejection] = [e.payload for e in h.store.events_after(0, limit=10000, run_id=h.rid)
                   if e.kind == "job.workflow_rejected"]
    assert rejection["error"]["code"] == "workflow_mismatch"
    assert rejection["safe_shutdown"]["failed"] == ([output.device_id] if shutdown_failure else [])
    if shutdown_failure:
        assert run["error"]["safe_shutdown_failed"] == [output.device_id]
        assert h.store.locks() == {output.device_id: h.rid}
        h.service.context_factory = lambda *args: pytest.fail("shutdown verification cannot create a process context")
        with pytest.raises(ValueError, match="unconfirmed"):
            h.service.retry(h.rid)
        output.fail = False
        h.service.reverify(h.rid)
        h.service._spawn(95)
        assert h.service._done[h.rid].wait(1)
        assert h.store.get_run(h.rid)["error"]["safe_shutdown_failed"] == []
        assert h.store.get_run(h.rid)["result"] == run["result"]
    assert output.target is None and not h.store.locks()
    assert h.store.get_steps(h.rid)[0]["status"] == "failed"
    assert h.trace == [("watchdog",), ("act", "first"), ("original_verify", "first")]


@pytest.mark.parametrize("failure", ["params", "workflow", "checkpoint"])
def test_replan_transaction_failure_preserves_live_and_durable_context(admitted, monkeypatch, failure):
    from qumus_simlab.core.errors import StopRequested
    from qumus_simlab.flows.flow import Replan
    from qumus_simlab.flows.step import StepOutcome, StepStatus

    h = admitted
    ctx = h.factory(h.rid, {}, None)
    params = h.Original.Params()
    steps = h.Original().build_steps(params)
    h.store.begin_step(h.rid, 0, steps[0].name)

    class ExpandedParams(BaseModel):
        extended: bool = True

    class Changed(h.Original):
        def on_step_failed(self, *args):
            return Replan(ExpandedParams(), "Checked tail")

    flow = Changed()
    before = deepcopy(h.store.get_run(h.rid))
    old_params = ctx.params
    emit, finish = h.store.emit, h.store.finish_step

    def fail_emit(kind, *args, **kwargs):
        result = emit(kind, *args, **kwargs)
        if kind == ("job.params_changed" if failure == "params" else "job.workflow"):
            raise StopRequested("Interrupt checked replan write")
        return result

    def fail_finish(*args, **kwargs):
        finish(*args, **kwargs)
        raise StopRequested("Interrupt skipped checkpoint write")

    if failure == "checkpoint":
        monkeypatch.setattr(h.store, "finish_step", fail_finish)
    else:
        monkeypatch.setattr(h.store, "emit", fail_emit)
    with pytest.raises(StopRequested):
        h.service.executor._replan(h.rid, ctx, flow, params, steps, 0,
            StepOutcome(StepStatus.NEEDS_AGENT, steps[0].name, 1, 0))
    assert ctx.params is old_params and ctx.params == before["params"]
    assert h.store.get_run(h.rid) == before
    assert h.store.get_steps(h.rid)[0]["status"] == "in_progress"
    assert not [e for e in h.store.events_after(0, limit=10000, run_id=h.rid)
                if e.kind == "job.params_changed"]


def test_persisted_queued_job_rejects_a_changed_registration_after_reopen(admitted, monkeypatch, tmp_path):
    h = admitted
    path = tmp_path / "queued-job.db"
    store = RunStore(path, clock=SimClock())

    def context(run_id, params, project_id):
        return SimpleNamespace(store=store, run_id=run_id, params=params, project_id=project_id,
                               bench={}, cancel=threading.Event(), gate=h.gate)

    service = ThreadJobService(store, h.gate, context, workers=0)
    rid = service.submit(h.Original.name, {})
    manifest = deepcopy(store.get_run(rid)["meta"]["workflow"])
    store.reserve("slot", "original_slot", rid)
    store.close()
    replace_source(h, monkeypatch, "actual_steps_disagree")
    store = RunStore(path, clock=SimClock())
    try:
        restored = ThreadJobService(store, h.gate, context, workers=0)
        [decision] = restored.recover()
        assert decision.action == "park" and decision.workflow_error["reason"] == "steps_changed"
        assert restored.status(rid).status == "waiting_human"
        assert restored.status(rid).workflow == manifest
        assert not store.get_steps(rid) and not h.trace
        assert "original_slot" in store.reservations("slot")
        assert any(event.kind == "job.workflow_rejected" for event in store.events_after(0, limit=10000))
    finally:
        store.close()


@pytest.mark.parametrize("entry", ["recover", "execute_restore", "direct_reverify", "replan"])
@pytest.mark.parametrize("failure", [None, "reactor_heater", "reactor_flow", "heater_no_effect"])
def test_hot_drift_shuts_down_energy_and_preserves_carrier_holding_and_evidence(admitted, monkeypatch, entry, failure):
    """Real simulated heater/gas, inserted carrier and another chip held by an arm.

    The interrupted registered workflow is an explicit structural fixture. No
    physical operation or material success is inferred from its no-op Steps.
    """
    from conftest import put_sample
    from test_reactor_devices import reactor_services, start_growth

    from qumus_simlab.core.types import Pose
    from qumus_simlab.devices.base import GasFlowController, Heater
    from qumus_simlab.flows.bench import arm_positions

    h = admitted
    s = reactor_services()
    put_sample(s, "chip_001", "mid_chip_1")
    arm = s.lab.devices.get("arm_right")
    slot = arm_positions(s.lab.bench, "arm_right")["mid_chip_1"]
    arm.move_to(Pose(x=slot.x, y=slot.y, z=2))
    arm.set_vacuum(True)
    arm.move_to(Pose(x=slot.x, y=slot.y, z=80))
    s.store.upsert_sample("chip_001", "chip", location="arm_right")
    assert s.lab.world.objects["chip_001"].held_by == "arm_right"
    start_growth(s)  # Explicit test fixture: seated/inserted carrier with active outputs.
    s.clock.sleep(5)
    assert s.gate.heater("reactor_heater").read_temperature() > 22
    assert s.gate.heater("reactor_heater").get_target() == 820
    assert s.gate.gas_flow("reactor_flow").get_target() == 200
    devices = ("reactor_heater", "reactor_flow", "reactor_insert", "arm_right")
    monkeypatch.setattr(h.Original, "devices", devices)
    rid = s.store.create_run(h.Original.name, {}, project_id=s.project_id)
    manifest = h.Original().build_workflow(Params()).manifest()
    s.store.update_run_meta(rid, workflow=manifest, held_heaters={"reactor_heater": 820})
    s.store.begin_step(rid, 0, "first", safe_to_resume=True)
    s.store.set_run_status(rid, "running")
    s.store.reserve("sample", "reactor_chip_01", rid)
    s.store.reserve("slot", "reactor_port", rid)
    assert s.store.acquire_locks(devices, rid)
    before_meta = deepcopy(s.store.get_run(rid)["meta"])
    before_steps = s.store.get_steps(rid)
    before_samples = s.store.list_samples()
    before_reservations = s.store.reservations()

    def carrier_state():
        return {name: (obj.pose.model_dump(), obj.held_by, obj.on)
                for name, obj in s.lab.world.objects.items()}

    before_carriers = deepcopy(carrier_state())
    before_arm = arm.get_pose().model_dump(), arm.vacuum_on()
    failed_device = "reactor_heater" if failure == "heater_no_effect" else failure
    if failure:
        device = s.lab.devices.get(failed_device)
        method = "off" if failed_device == "reactor_heater" else "safe_state"

        def failed_shutdown():
            raise RuntimeError("private shutdown failure details must not be exported")

        monkeypatch.setattr(device, method, (lambda: None) if failure == "heater_no_effect" else failed_shutdown)
    if entry == "replan":
        def fail_and_change_source(self, ctx):
            h.trace.append(("original_verify", self.name))
            replace_source(h, monkeypatch, "actual_steps_disagree")
            return Observation(Verdict.UNCLEAR, message="Source registration changed after this original step")

        monkeypatch.setattr(h.Checked, "verify", fail_and_change_source)
    else:
        replace_source(h, monkeypatch, "actual_steps_disagree")
    last = s.store.last_event_id()
    if entry == "recover":
        [decision] = s.jobs.recover()
        assert decision.action == "park" and decision.workflow_error["code"] == "workflow_mismatch"
    elif entry == "direct_reverify":
        assert s.jobs.executor.reverify(rid, s.jobs.context_factory(rid, {}, s.project_id)) == "parked"
    else:
        assert s.jobs.executor.execute(rid, s.jobs.context_factory(rid, {}, s.project_id),
                                       restore=entry == "execute_restore") == "parked"
    run = s.store.get_run(rid)
    assert run["status"] == "waiting_human"
    if entry == "replan":
        assert run["error"]["status"] == "needs_agent"
        assert run["result"][-1]["observations"][0]["message"] == "Source registration changed after this original step"
    else:
        assert run["error"]["code"] == "workflow_mismatch"
    assert run["meta"] == before_meta
    assert s.store.list_samples() == before_samples
    assert s.store.reservations() == before_reservations
    assert carrier_state() == before_carriers
    assert (arm.get_pose().model_dump(), arm.vacuum_on()) == before_arm
    assert arm.vacuum_on() and s.lab.world.objects["chip_001"].held_by == "arm_right"
    if entry == "replan":
        [checkpoint] = s.store.get_steps(rid)
        assert checkpoint["name"] == "first" and checkpoint["status"] == "failed"
        assert h.trace == [("act", "first"), ("original_verify", "first")]
    else:
        assert s.store.get_steps(rid) == before_steps and not h.trace
    events = s.store.events_after(last, limit=10000)
    commands = [event.payload for event in events if event.kind == "device.command"]
    assert commands and all(command["command"] in ("off", "safe_state")
        and isinstance(s.lab.devices.get(command["device"]), (Heater, GasFlowController)) for command in commands)
    [rejection] = [event.payload for event in events if event.kind == "job.workflow_rejected"]
    shutdown = rejection["safe_shutdown"]
    assert "reactor_heater" in shutdown["attempted"] and "reactor_flow" in shutdown["attempted"]
    if failure:
        assert shutdown["failed"] == [failed_device] and shutdown["failed_device_locks_held"]
        assert run["error"]["safe_shutdown_failed"] == [failed_device]
        assert s.store.locks() == {device_id: rid for device_id in devices}
        assert any(event.kind == "notification" and event.payload.get("level") == "danger"
            and event.payload.get("device") == failed_device for event in events)
        assert "private shutdown failure" not in str([event.payload for event in events])
    else:
        assert not shutdown["failed"] and rid not in s.store.locks().values()
    if failed_device != "reactor_heater":
        assert s.gate.heater("reactor_heater").get_target() is None
    else:
        assert s.gate.heater("reactor_heater").get_target() == 820
    if failed_device != "reactor_flow":
        assert s.gate.gas_flow("reactor_flow").get_target() == 0
    else:
        assert s.gate.gas_flow("reactor_flow").get_target() == 200
