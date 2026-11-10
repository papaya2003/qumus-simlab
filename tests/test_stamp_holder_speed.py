"""Stamp holder speeds (#181): fast rail travel, v1's slow x/y alignment, z unchanged."""

from qumus_simlab.config import load_yaml
from qumus_simlab.devices.sim.devices import SimManipulator
from qumus_simlab.testbench.world import SimWorld


def _holder():
    motion = load_yaml("testbench.yaml")["motion"]
    return SimManipulator("transfer_arm", SimWorld(seed=1), {"x": 150, "y": 10, "z": 6.5}, sigma_mm=0,
                          speed_mm_s=motion["manipulator_speed_mm_s"],
                          travel_speed_mm_s=motion["manipulator_travel_speed_mm_s"],
                          fine_speed_mm_s=motion["manipulator_fine_speed_mm_s"],
                          fine_below_mm=motion["manipulator_fine_below_mm"]), motion


def test_rail_travel_alignment_and_z_use_their_own_speeds():
    holder, motion = _holder()
    # the 80 mm y leg from the hand-off to the rail corner is rail travel
    assert holder.axis_motion_duration_bound("y", 10, 90) == 80 / motion["manipulator_travel_speed_mm_s"]
    # a 10 um alignment step is v1's 0.2 mm/s (KST XY max_vel)
    assert motion["manipulator_fine_speed_mm_s"] == 0.2
    assert abs(holder.axis_motion_duration_bound("x", 40, 40.01) - 0.01 / 0.2) < 1e-12
    # z keeps the holder's own z speed
    assert holder.axis_motion_duration_bound("z", 6.5, 5.5) == 1 / motion["manipulator_speed_mm_s"]
    # an explicit speed still wins
    assert holder.axis_motion_duration_bound("y", 10, 90, speed=1.0) == 80


def test_travel_speed_stays_within_the_safety_limit():
    _, motion = _holder()
    limit = load_yaml("limits.yaml")["manipulators"]["transfer_arm"]["max_speed"]
    assert motion["manipulator_travel_speed_mm_s"] <= limit
    assert motion["manipulator_speed_mm_s"] <= limit
