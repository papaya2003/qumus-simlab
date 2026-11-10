"""The exfoliator seen from the front (bench.yaml exfoliator, Jialei's drawing):
a fixed roll, a movable roll on an x-z gantry, the tape the line between them."""

from __future__ import annotations

from qumus_simlab.core.types import Pose


def _chip_on_stage(services):
    w = services.lab.world
    st = w.surfaces["exfoliation_stage"]
    chip = w.objects["chip_001"]
    chip.pose = chip.pose.offset(dx=sum(st.box["x"]) / 2 + st.offset[0] - chip.pose.x,
                                 dy=sum(st.box["y"]) / 2 + st.offset[1] - chip.pose.y)
    chip.on = "exfoliation_stage"
    w.tape.material = "Gr"
    services.lab.devices.get("exfoliation_vacuum").set(True)  # the chip stays down
    return w, chip, services.lab.devices.get("exfoliator")


def _press_all(w, chip):
    for dy in (-4, -2, 0, 2, 4):
        p = chip.pose.offset(dy=dy)
        w._massage(Pose(x=p.x + 5, y=p.y, z=1.0), start=Pose(x=p.x - 5, y=p.y, z=1.0))


def test_the_apply_path_lays_the_tape_and_the_peel_reaches_the_asked_angle(services):
    w, chip, tape = _chip_on_stage(services)
    assert not tape._laid()  # parked: the tape hangs high over the stage
    tape.apply()
    pos = tape.get_positions()
    assert w.tape.laid and pos["tape_z0"] < 5 and pos["tape_z1"] < 0
    _press_all(w, chip)
    assert w.press_frac(chip) > 0.9
    tape.peel(100.0, 45.0)
    assert w.tape.last_peel_angle == 45.0 and chip.flakes
    assert not w.collisions


def test_a_tape_left_high_presses_nothing_and_gives_no_flakes(services):
    w, chip, tape = _chip_on_stage(services)
    tape.apply_path = [(40, 0), (40, -30), (-70, -30)]  # never comes down to the chip
    tape.apply()
    assert w.tape.applied and not w.tape.laid
    _press_all(w, chip)
    assert w.press_frac(chip) == 0.0
    tape.peel(100.0, 30.0)
    assert not chip.flakes


def test_the_peel_angle_is_limited_by_the_gantry_travel(services):
    w, chip, tape = _chip_on_stage(services)
    tape.apply()
    _press_all(w, chip)
    tape.travel = {"x": [-60.0, 60.0], "z": [-80.0, 10.0]}  # a short x axis
    tape.peel(100.0, 10.0)
    peel = w.tape_snapshot()["last_peel"]
    assert peel["requested_deg"] == 10.0 and peel["angle_deg"] > 15.0


def test_a_roll_driven_into_the_stage_is_a_collision(services):
    w, chip, tape = _chip_on_stage(services)
    tape.apply_path = [(40, 0), (40, -60), (-40, -80)]  # far too low over the stage
    tape.apply()
    assert [c["what"] for c in w.collisions] == ["exfoliator_roll"]
