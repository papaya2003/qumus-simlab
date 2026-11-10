"""Fault injection on the testbench (spec 11.2, workstreams.md A2): each fault
is caught and corrected by the step, retried, or handed back with a clear
reason; and a kill at any device call resumes to the same end state."""

from __future__ import annotations

import pytest

from qumus_simlab.core.types import Pose
from qumus_simlab.flows.bench import positions
from qumus_simlab.testbench.world import TAPE_HOLDER


def run(s, flow, params, timeout=60):
    return s.jobs.wait(s.jobs.submit(flow, params), timeout)


def failed_note(st):
    return next(x["note"] for x in st.steps if x["status"] == "failed")


def events(s, kind):
    return [e for e in s.store.events_after(0, limit=100_000) if e.kind == kind]


PLACE = ("place_chip_on_transfer_stage", {"chip_id": "chip_001"})


def test_pick_failure_is_seen_and_picked_again(services):
    services.lab.faults.inject("pick_fail_next", {"count": 1})
    st = run(services, *PLACE)
    assert st.status == "succeeded"
    assert st.steps[0]["corrections"] == 0  # the camera saw it still there: picked again
    texts = [e.payload["text"] for e in events(services, "notification")]
    assert any("pick came up empty" in t for t in texts)


def test_pick_failing_again_and_again_is_corrected(services):
    services.lab.faults.inject("pick_fail_next", {"count": 3})  # 1 + PICK_RETRIES
    st = run(services, *PLACE)
    assert st.status == "succeeded"
    assert st.steps[0]["corrections"] == 1
    assert "pick failed" in st.result[0]["observations"][0]["message"]


def test_drop_in_transit_is_found_and_corrected(services):
    services.lab.faults.inject("drop_in_transit")
    st = run(services, *PLACE)
    assert st.status == "succeeded", st.steps
    first = st.result[0]["observations"][0]
    assert first["measurements"]["found_at"] == "workspace" and "dropped" in first["message"]
    assert services.lab.world.objects["chip_001"].on == "transfer_stage"


def test_serial_timeout_is_retried(services):
    services.lab.faults.inject("serial_timeout", {"device": "arm_right", "count": 2})
    st = run(services, *PLACE)
    assert st.status == "succeeded"
    texts = [e.payload["text"] for e in events(services, "notification")]
    assert sum("serial timeout" in t for t in texts) == 2


def test_persistent_timeout_parks_then_retry_after_fix(services):
    services.lab.faults.inject("serial_timeout", {"device": "arm_right", "count": 50})
    st = run(services, *PLACE)
    assert st.status == "parked" and "serial timeout" in failed_note(st)
    services.lab.faults.clear()
    services.jobs.retry(st.run_id)
    assert services.jobs.wait(st.run_id, 10).status == "succeeded"


def test_camera_offline_parks_with_reason(services):
    services.lab.faults.inject("device_offline", {"device": "overview_camera"})
    st = run(services, *PLACE)
    assert st.status == "parked" and "overview_camera offline" in failed_note(st)
    assert not events(services, "safety.violation")


def test_unknown_fault_is_refused(services):
    with pytest.raises(ValueError, match="unknown fault"):
        services.lab.faults.inject("meteor")
    assert {f["fault"] for f in services.lab.faults.describe()} >= {"crash", "drop_in_transit"}


# ----- kill at any device call, restart, recover ---------------------------

def _calls(make_services, flow, params, before=None):
    s = make_services()
    if before:
        assert run(s, *before).status == "succeeded"
    c0 = s.lab.faults.calls
    assert run(s, flow, params, 120).status == "succeeded"
    return s.lab.faults.calls - c0


def _crash_and_recover(make_services, flow, params, k, before=None):
    s = make_services()
    if before:
        assert run(s, *before).status == "succeeded"
    s.lab.faults.inject("crash", {"after_calls": k})
    rid = s.jobs.submit(flow, params)
    assert s.lab.faults.crashed.wait(30), f"no crash at call {k}"
    # the job service restarts on its own (FaultInjector.on_crash -> jobs.recover)
    return s, s.jobs.wait(rid, 120)


def _consistent(s):
    """Ledger and bench agree; nothing left on a nozzle or on a stage."""
    w = s.lab.world
    for o in w.objects.values():
        assert o.held_by is None, f"{o.id} still held by {o.held_by}"
    for sample in s.store.list_samples():
        if sample["kind"] in ("chip", "carrier"):
            p = positions(s.lab.bench)[sample["location"]]
            assert abs(w.objects[sample["id"]].pose.x - p["x"]) < 1.0 and \
                abs(w.objects[sample["id"]].pose.y - p["y"]) < 1.0, sample
    assert not w.tape.applied
    assert not events(s, "safety.violation")


EXFOLIATE = ("exfoliate_chips", {"material": "Gr"})


@pytest.mark.parametrize("flow,params,before,early,stride", [
    (*PLACE, None, 40, 1),  # every call
    (*EXFOLIATE, None, 40, 3),
    ("search_chip", {}, EXFOLIATE, 12, 97),  # ~1800 calls, mostly autofocus and scan
])
def test_kill_at_any_call_resumes(make_services, flow, params, before, early, stride):
    n = _calls(make_services, flow, params, before)
    points = sorted(set(range(1, min(n, early))) | set(range(1, n, stride)))
    for k in points:
        s, st = _crash_and_recover(make_services, flow, params, k, before)
        assert st.status == "succeeded", (k, st.steps)
        _consistent(s)


def test_kill_while_heating_waits_for_a_human_with_heater_off(make_services):
    params = {"material": "Gr", "target_temp_c": 70}
    s = make_services()
    # find a call inside heat_applied_tape
    s.lab.faults.inject("crash", {"after_calls": 10_000})
    rid = s.jobs.submit("exfoliate_chips", params)
    st = s.jobs.wait(rid, 60)
    assert st.status == "succeeded"
    heat_from = 0
    for k in range(60, 400, 20):
        s, st = _crash_and_recover(make_services, "exfoliate_chips", params, k)
        heat = [x for x in s.store.get_steps(st.run_id) if x["name"] == "heat_applied_tape"]
        if st.status == "waiting_human":
            assert heat and heat[0]["status"] == "in_progress"
            texts = [e.payload["text"] for e in events(s, "notification")]
            assert any("not safe to resume" in t for t in texts)
            assert s.lab.devices.get("exfoliation_heater").get_target() is None
            heat_from = k
            break
        assert st.status == "succeeded"
    assert heat_from, "never crashed inside the heating step"


def test_a_chip_lifted_with_the_tape_goes_back_down_with_it(services):
    # v1's heated recipe: peel with the vacuum off (the chip rides on the tape
    # while the stage heats), lay the tape again, vacuum on, cool, peel
    w = services.lab.world
    s = w.surfaces["exfoliation_stage"]
    chip = w.objects["chip_001"]
    chip.pose = chip.pose.offset(dx=sum(s.box["x"]) / 2 + s.offset[0] - chip.pose.x,
                                 dy=sum(s.box["y"]) / 2 + s.offset[1] - chip.pose.y)
    chip.on = "exfoliation_stage"
    w.tape.material = "Gr"
    w.tape_apply(22.0)
    for dy in (-4, -2, 0, 2, 4):  # rows of low strokes with the press pad
        p = chip.pose.offset(dy=dy)
        w._massage(Pose(x=p.x + 5, y=p.y, z=1.0), start=Pose(x=p.x - 5, y=p.y, z=1.0))
    assert w.press_frac(chip) > 0.9
    w.tape_peel(100.0, stage_vacuum_on=False, stage_temp_c=22.0)
    assert w.held_by(TAPE_HOLDER) is chip and not chip.flakes
    w.tape_apply(90.0)
    assert chip.held_by is None and w.on_exfoliation_stage() is chip
    w.tape_peel(100.0, stage_vacuum_on=True, stage_temp_c=60.0)
    assert chip.flakes and {f.material for f in chip.flakes} == {"Gr"}


def test_stamp_torn_tears_the_held_stamp_or_arms_the_next_one(services):
    """The fault tears the stamp on the holder, else the next one gripped; the
    one after that is whole (the flow's stamp change relies on it)."""
    w, fi = services.lab.world, services.lab.faults
    fi.inject("stamp_torn", {})
    assert w.faults.tear_next_stamp and all(st.film for st in w.stamps.values())
    with w.lock:  # stamp_01 on the holder's head, then gripped
        hx, hy = w.seat_xy()  # the slide on the holder, its dome under the head
        w.objects["stamp_01"].pose = Pose(x=hx, y=hy)
    w.transfer_grip(True)
    assert w.held_stamp().id == "stamp_01" and not w.stamps["stamp_01"].film
    assert not w.faults.tear_next_stamp
    w.transfer_grip(False)
    with w.lock:
        w.objects["stamp_01"].pose = Pose(x=hx + 30, y=hy)
        w.objects["stamp_02"].pose = Pose(x=hx, y=hy)
    w.transfer_grip(True)
    assert w.held_stamp().id == "stamp_02" and w.stamps["stamp_02"].film
    fi.inject("stamp_torn", {})  # held now: torn at once
    assert not w.stamps["stamp_02"].film and not w.faults.tear_next_stamp
