"""Kill-and-resume (spec 8.2, 11.5 #5)."""

from __future__ import annotations

from conftest import put_sample

from qumus_simlab.core.types import Pose
from qumus_simlab.flows.bench import arm_positions
from qumus_simlab.runtime.recovery import plan_recovery


def _crashed_run(s):
    """A run that died inside step 0 (status running, step in progress); the
    chip waits on the middle board, where arm_right takes it."""
    if s.store.get_sample("chip_001")["location"] == "left_chip_1":
        put_sample(s, "chip_001", "mid_chip_1")
    rid = s.store.create_run("place_chip_on_transfer_stage", {"chip_id": "chip_001"})
    s.store.set_run_status(rid, "running")
    s.store.begin_step(rid, 0, "place_chip_on_transfer_stage")
    s.store.acquire_locks(["arm_right"], rid)
    return rid


def _factory(s):
    # reuse the wiring from build_services
    return s.jobs.context_factory


def test_crash_before_placement_resumes_same_step(services):
    rid = _crashed_run(services)
    [d] = plan_recovery(services.store, _factory(services), gate=services.gate)
    assert (d.action, d.start_seq) == ("resume", 0)
    assert services.store.locks() == {}  # stale locks released
    services.jobs.enqueue(rid, d.start_seq)
    assert services.jobs.wait(rid, 10).status == "succeeded"


def test_crash_after_placement_marks_step_done(services):
    put_sample(services, "chip_001", "mid_chip_1")  # where the ledger last saw it
    w = services.lab.world
    chip = w.objects["chip_001"]
    t = services.lab.bench["stations"]["transfer_stage"]
    chip.pose = Pose(x=t["x"] + 0.1, y=t["y"], rotation=0.5)  # already placed well
    rid = _crashed_run(services)
    [d] = plan_recovery(services.store, _factory(services), gate=services.gate)
    assert (d.action, d.start_seq) == ("resume", 1)
    assert services.store.get_steps(rid)[0]["status"] == "done"
    assert services.store.get_sample("chip_001")["location"] == "transfer_stage"


def test_crash_while_holding_chip_resumes_and_puts_it_down(services):
    arm = services.lab.devices.get("arm_right")
    put_sample(services, "chip_001", "mid_chip_1")
    slot = arm_positions(services.lab.bench, "arm_right")["mid_chip_1"]  # arm frame
    arm.move_to(Pose(x=slot.x, y=slot.y, z=2))
    arm.set_vacuum(True)
    arm.move_to(Pose(x=slot.x, y=slot.y, z=80))
    rid = _crashed_run(services)
    [d] = plan_recovery(services.store, _factory(services), gate=services.gate)
    assert (d.action, d.start_seq) == ("resume", 0)  # chip is on the nozzle: finish the place
    services.jobs.enqueue(rid, d.start_seq)
    assert services.jobs.wait(rid, 10).status == "succeeded"
    assert services.lab.world.objects["chip_001"].on == "transfer_stage"
    assert not arm.vacuum_on()


def test_crash_holding_with_object_still_visible_parks(services):
    arm = services.lab.devices.get("arm_right")
    arm.set_vacuum(True)  # vacuum on, but the chip is still in its slot: unclear
    rid = _crashed_run(services)
    [d] = plan_recovery(services.store, _factory(services), gate=services.gate)
    assert d.action == "park"
    assert services.store.get_run(rid)["status"] == "waiting_human"


def test_not_safe_to_resume_step_parks(services):
    rid = services.store.create_run("place_chip_on_transfer_stage", {"chip_id": "chip_001"})
    services.store.set_run_status(rid, "running")
    services.store.begin_step(rid, 0, "heating", safe_to_resume=False)
    [d] = plan_recovery(services.store, _factory(services), gate=services.gate)
    assert d.action == "park" and "not safe" in d.reason
