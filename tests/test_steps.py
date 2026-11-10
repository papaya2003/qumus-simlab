"""Closed-loop step + correction ladder on the testbench (spec 4.2, 4.3, 11.5 #5)."""

from __future__ import annotations

from conftest import put_sample

from qumus_simlab.config import load_yaml
from qumus_simlab.core.errors import TransientError
from qumus_simlab.core.types import Pose
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.step import NOT_VERIFIABLE, Step, StepRunner, StepStatus


def run_place(s, chip="chip_001", timeout=10, place="mid_chip_1"):
    """Place a chip from the middle board (where arm_right takes it) on the stage."""
    if place and s.store.get_sample(chip) and \
            s.store.get_sample(chip)["location"].startswith("left_"):
        put_sample(s, chip, place)
    rid = s.jobs.submit("place_chip_on_transfer_stage", {"chip_id": chip})
    return s.jobs.wait(rid, timeout)


def test_place_chip_first_time(services):
    st = run_place(services)
    assert st.status == "succeeded"
    assert st.steps[0]["corrections"] == 0
    assert services.store.get_sample("chip_001")["location"] == "transfer_stage"
    assert services.store.locks() == {}


def test_askew_placement_is_detected_and_corrected(services):
    services.lab.world.faults.next_placement = {"dx": 2.5, "dy": -1.5, "drot": 7.0}
    st = run_place(services)
    assert st.status == "succeeded"
    assert st.steps[0]["corrections"] == 1
    kinds = [e.kind for e in services.store.events_after(0, limit=10_000)]
    assert "step.corrected" in kinds
    assert abs(st.steps[0]["measurements"]["offset_mm"]) <= 0.5


def test_corrections_exhausted_hands_back_to_agent(make_services):
    s = make_services(placement__big_error_prob=1.0, placement__big_error_mm=3.0)
    st = run_place(s)
    assert st.status == "parked"
    assert st.steps[0]["status"] == "failed"
    max_corr = load_yaml("verification.yaml")["place_chip_on_transfer_stage"]["max_corrections"]
    assert st.steps[0]["corrections"] == max_corr


def test_pick_failure_is_retried(make_services):
    s = make_services(pick__fail_prob=0.0)
    s.lab.world.faults.pick_fail_prob = 1.0
    real_pick = s.lab.world.pick
    calls = {"n": 0}

    def flaky(holder, nozzle):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # first pick fails, chip stays in slot
        s.lab.world.faults.pick_fail_prob = 0.0
        return real_pick(holder, nozzle)

    s.lab.world.pick = flaky
    st = run_place(s)
    # the camera sees the chip still in its slot after the lift: picked again
    # within the same attempt, no correction needed
    assert st.status == "succeeded" and st.steps[0]["corrections"] == 0
    assert calls["n"] == 2


def test_precondition_failure_moves_nothing(services):
    assert run_place(services, "chip_001").status == "succeeded"
    n_moves = sum(1 for e in services.store.events_after(0, limit=10_000)
                  if e.kind == "device.command" and e.payload["command"] == "move_to")
    st = run_place(services, "chip_002", place="mid_chip_2")  # stage already occupied
    assert st.status == "parked"
    assert "not empty" in st.steps[0]["note"]
    n_after = sum(1 for e in services.store.events_after(0, limit=10_000)
                  if e.kind == "device.command" and e.payload["command"] == "move_to")
    assert n_after == n_moves


def test_unknown_chip_is_rejected_by_params_or_precondition(services):
    st = run_place(services, "chip_999")
    assert st.status == "parked"


def _ctx(services, rid="r_test") -> FlowContext:
    services.store.create_run("place_chip_on_transfer_stage", {"chip_id": "x"}, run_id=rid)
    return FlowContext(run_id=rid, store=services.store, gate=services.gate,
                       clock=services.clock, locator=services.lab.locator,
                       bench=services.lab.bench, verification={})


class _Unverifiable(Step):
    name = "unverifiable"

    def act(self, ctx, correction):
        pass

    def verify(self, ctx):
        return NOT_VERIFIABLE("no sensor for this yet")


class _Flaky(Step):
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def act(self, ctx, correction):
        self.calls += 1
        if self.calls < 3:
            raise TransientError("serial timeout")

    def verify(self, ctx):
        return NOT_VERIFIABLE("test")


class _OutOfRange(Step):
    name = "out_of_range"
    devices = ("arm_right",)

    def act(self, ctx, correction):
        ctx.gate.arm("arm_right").move_to(Pose(x=9999, y=0, z=50))

    def verify(self, ctx):
        raise AssertionError("must not get here")


def test_not_verifiable_is_recorded(services):
    ctx = _ctx(services)
    out = StepRunner().run(_Unverifiable(), ctx, 0)
    assert out.status == StepStatus.UNVERIFIED
    assert services.store.get_steps("r_test")[0]["note"].startswith("NOT_VERIFIABLE")


def test_transient_errors_are_retried_with_backoff(services):
    ctx = _ctx(services)
    step = _Flaky()
    t0 = services.clock.now()
    out = StepRunner().run(step, ctx, 0)
    assert out.ok and step.calls == 3
    assert services.clock.now() - t0 == 3.0  # 1s + 2s simulated backoff


def test_safety_violation_inside_step_is_dangerous(services):
    ctx = _ctx(services)
    out = StepRunner().run(_OutOfRange(), ctx, 0)
    assert out.status == StepStatus.DANGEROUS
    assert out.error["category"] == "dangerous"
