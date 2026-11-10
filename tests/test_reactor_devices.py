"""Guarded mechanics, optical focus, plant history and checkpoint verification."""

from __future__ import annotations

import math

import pytest
from conftest import build_services, quiet_bench

from qumus_simlab.core.errors import DeviceUnavailable, SafetyViolation
from qumus_simlab.core.types import Pose
from qumus_simlab.lab import build_lab_side
from qumus_simlab.testbench.persist import save_lab_state


def reactor_services(image_dir=None):
    cfg = quiet_bench()
    cfg["reactor"] = {"enabled": True, "experiment_kind": "legacy_field_growth"}
    return build_services(testbench_config=cfg, image_dir=image_dir, seed=1)


def load_by_hand(s):
    """A fixture representing manual seating before testing device mechanics."""
    chip = s.lab.world.objects["reactor_chip_01"]
    chip.pose, chip.on = Pose(**s.lab.bench["reactor_module"]["geometry"]["load"]), "reactor"
    s.store.upsert_sample(chip.id, "chip", location="reactor_port")
    return chip


def start_growth(s):
    chip = load_by_hand(s)
    s.gate.stage("reactor_insert").move_to(Pose(x=0, y=160))
    for target in (142, 262, 382, 502, 622, 742, 820):
        s.gate.heater("reactor_heater").set_target(target)
    s.gate.gas_flow("reactor_flow").set_flow(200)
    s.gate.microscope("reactor_microscope").set_focus(chip.focus_offset)


def test_module_is_opt_in_and_default_world_is_preserved():
    default = build_services(testbench_config=quiet_bench())
    assert "reactor_slot" not in default.lab.devices.devices
    assert default.lab.bench["stations"]["reactor"]["placeholder"]
    s = reactor_services()
    assert len([x for x in s.store.list_samples() if x["props"].get("reactor_substrate")]) == 4
    assert s.lab.bench["stations"]["reactor"]["placeholder"] is False


def test_three_motors_move_assembly_and_insertion_independently():
    s = reactor_services()
    chip = load_by_hand(s)
    start = chip.pose.model_copy()
    t = s.clock.now()
    s.gate.stage("reactor_insert").move_to(Pose(x=0, y=160))
    assert s.clock.now() - t == pytest.approx(160 / 30)
    assert chip.pose.x == start.x and chip.pose.y == start.y + 160
    s.gate.stage("reactor_slot").move_to(Pose(x=1.5, y=-1))
    assert chip.pose.x == pytest.approx(start.x + 1.5)
    assert chip.pose.y == pytest.approx(start.y + 159)
    assert s.gate.stage("reactor_insert").get_pose().y == 160
    assert chip.pose.z == 0 and chip.pose.rotation == 0
    for device, pose in (("reactor_slot", Pose(x=0, y=0, z=0.1)),
                         ("reactor_slot", Pose(x=0, y=0, rotation=1)),
                         ("reactor_slot", Pose(x=5, y=0)),
                         ("reactor_slot", Pose(x=0, y=160)),
                         ("reactor_insert", Pose(x=1, y=160))):
        with pytest.raises(SafetyViolation):
            s.gate.stage(device).move_to(pose)


def test_loading_and_hot_motion_interlocks():
    s = reactor_services()
    with pytest.raises(SafetyViolation, match="insert"):
        s.gate.heater("reactor_heater").set_target(100)
    with pytest.raises(SafetyViolation, match="insert"):
        s.gate.gas_flow("reactor_flow").set_flow(100)
    arm = s.gate.arm("arm_reactor")
    arm.move_to(Pose(x=120, y=144, z=3))
    with pytest.raises(SafetyViolation, match="arm is low"):
        s.gate.stage("reactor_insert").move_to(Pose(x=0, y=160))
    arm.move_to(Pose(x=120, y=144, z=80))
    start_growth(s)
    s.clock.sleep(300)
    with pytest.raises(SafetyViolation, match="insertion is locked"):
        s.gate.stage("reactor_insert").move_to(Pose(x=0, y=0))
    with pytest.raises(SafetyViolation, match="not accessible"):
        arm.move_to(Pose(x=120, y=144, z=3))
    s.gate.stage("reactor_slot").move_to(Pose(x=1, y=0))


def test_actual_feedback_lags_setpoint_and_focus_is_independent(tmp_path):
    s = reactor_services(tmp_path)
    start_growth(s)
    assert s.gate.heater("reactor_heater").get_target() == 820
    assert s.gate.heater("reactor_heater").read_temperature() == pytest.approx(22)
    assert s.gate.gas_flow("reactor_flow").read_flow() == 0
    s.clock.sleep(600)
    scope = s.gate.microscope("reactor_microscope")
    frame = scope.capture()
    assert frame.meta["reactor"]["coverage"] > 0.8
    before = s.gate.stage("reactor_slot").get_pose()
    scope.set_focus(1)
    assert scope.capture().meta["reactor"]["coverage"] is None
    assert s.gate.stage("reactor_slot").get_pose() == before
    assert frame.saved_path() and frame.timestamp == s.clock.now()
    # A later capture must not rewrite the metadata attached to the saved image.
    assert frame.meta["reactor"]["phase"] != "out_of_focus"
    assert s.lab.preview("reactor_microscope")["meta"]["reactor"]["coverage"] > 0.8


def test_extra_fractional_observations_do_not_change_growth():
    a, b = reactor_services(), reactor_services()
    start_growth(a)
    start_growth(b)
    a.clock.sleep(600)
    expected = a.gate.microscope("reactor_microscope").capture().meta
    for _ in range(1200):
        b.clock.sleep(0.5)
        b.gate.microscope("reactor_microscope").capture()
    assert b.gate.microscope("reactor_microscope").capture().meta == expected


def test_restart_preserves_growth_physics_and_seed(tmp_path):
    cfg = quiet_bench()
    cfg["reactor"] = {"enabled": True, "experiment_kind": "legacy_field_growth"}
    s = reactor_services()
    start_growth(s)
    s.clock.sleep(200.5)
    s.gate.microscope("reactor_microscope").capture()
    db = tmp_path / "lab.db"
    side = build_lab_side(db, testbench_config=cfg, seed=1)
    # Use the original physical world with the file store only for this fixture.
    state = tmp_path / "lab.pkl"
    save_lab_state(s.lab, s.clock, state)
    restored = build_lab_side(db, testbench_config=cfg, seed=999, state_path=state)
    assert restored.lab.world.seed == 1
    assert side.lab is not restored.lab
    s.clock.sleep(399.5)
    restored.clock.sleep(399.5)
    left = s.gate.microscope("reactor_microscope").capture().meta
    right = restored.gate.microscope("reactor_microscope").capture().meta
    assert left == right


@pytest.mark.parametrize("value", [-1, 401, math.nan, math.inf])
def test_invalid_flow_rejected_before_command(value):
    s = reactor_services()
    s.gate.stage("reactor_insert").move_to(Pose(x=0, y=160))
    with pytest.raises(SafetyViolation):
        s.gate.gas_flow("reactor_flow").set_flow(value)
    assert s.gate.gas_flow("reactor_flow").get_target() == 0


def test_offline_reactor_does_not_break_unrelated_devices():
    s = reactor_services()
    s.lab.devices.get("reactor_heater").offline = True
    with pytest.raises(DeviceUnavailable):
        s.gate.heater("reactor_heater").read_temperature()
    s.gate.arm("arm_left").move_to(Pose(x=0, y=100, z=120))
    s.gate.arm("arm_reactor").move_to(Pose(x=120, y=144, z=120))
    s.gate.gas_flow("reactor_flow").set_flow(0)
    twin = s.lab.twin.state()
    assert twin["microscope"]["objective"] == 10
    assert twin["reactor_microscope"]["focus_z"] == 0
    s.lab.devices.get("reactor_insert").offline = True
    s.gate.gas_flow("reactor_flow").set_flow(0)
    s.gate.arm("arm_reactor").move_to(Pose(x=120, y=144, z=120))


def test_ambiguous_carrier_occupancy_is_not_accepted():
    from qumus_simlab.core.errors import PreconditionFailed

    s = reactor_services()
    load_by_hand(s)
    s.lab.world.objects["reactor_chip_02"].on = "reactor"
    with pytest.raises(PreconditionFailed, match="more than one chip"):
        s.gate.microscope("reactor_microscope").capture()


def test_layout_clearance_proportions_and_shared_arm_pose():
    s = reactor_services()
    b = s.lab.bench
    geo = b["reactor_module"]["geometry"]
    assert len({b[a]["base"]["y"] for a in ("arm_left", "arm_right", "arm_reactor")}) == 1
    assert b["arm_reactor"]["home"] == b["arm_left"]["home"] == b["arm_right"]["home"]
    normal = b["slots"]["mid_chip_1"]["size"]
    assert geo["chip_size_mm"] == [normal[0] / 4, normal[1]]
    assert geo["carrier_slot_size_mm"][0] > geo["chip_size_mm"][0]
    assert geo["load"]["x"] - 15 > b["stations"]["right_storage"]["region"]["x"][1]
    assert geo["storage_region"]["x"][0] > geo["load"]["x"] + 15
    chamber = geo["chamber_region"]
    assert geo["microscope_center"] == {a: sum(chamber[a]) / 2 for a in ("x", "y")}
    assert geo["storage_pitch_mm"] == [35, 40]
    for slot in (v for v in b["slots"].values() if v["station"] == "reactor_storage"):
        base = b["arm_reactor"]["base"]
        assert math.hypot(slot["x"] - base["x"], slot["y"] - base["y"]) < 280


def test_old_mechanics_checkpoint_is_rejected_for_resume():
    s = reactor_services()
    del s.lab.world.reactor_state["mechanics_version"]
    from qumus_simlab.core.errors import PreconditionFailed
    with pytest.raises(PreconditionFailed, match="Legacy reactor mechanics"):
        s.gate.stage("reactor_slot").get_pose()


@pytest.mark.parametrize("signature", [None, "different-layout"])
def test_incompatible_layout_checkpoint_cannot_resume(signature):
    from qumus_simlab.core.errors import PreconditionFailed

    s = reactor_services()
    s.lab.world.reactor_state["layout_signature"] = signature
    with pytest.raises(PreconditionFailed, match="layout differs"):
        s.gate.stage("reactor_slot").get_pose()


def test_equal_storage_spacing_preserves_workcell_frames_and_locator_geometry():
    from qumus_simlab.config import load_yaml

    b, old = reactor_services().lab.bench, load_yaml("bench.yaml")
    centres = []
    for station, arm in (("left_storage", "arm_left"), ("middle_storage", "arm_right"),
                         ("right_storage", "arm_reactor")):
        region = b["stations"][station]["region"]
        centres.extend([{a: sum(region[a]) / 2 for a in ("x", "y")}, b[arm]["base"]])
    assert [p["x"] for p in centres] == [130, 320, 510, 700, 890, 1080]
    assert len({math.dist(list(a.values()), list(c.values()))
                for a, c in zip(centres, centres[1:], strict=False)}) == 1
    for arm, station, region in (("arm_left", "exfoliation_stage", "tape"),
                                  ("arm_right", "transfer_stage", "microscope")):
        for axis in ("x", "y"):
            delta = b[arm]["base"][axis] - old[arm]["base"][axis]
            assert b["stations"][station][axis] - old["stations"][station][axis] == delta
            for part in ("region", "module"):
                assert b["stations"][station][part][axis] == [
                    n + delta for n in old["stations"][station][part][axis]]
            assert b["regions"][region][axis] == [n + delta for n in old["regions"][region][axis]]
    for section, point, arm in (("exfoliator", "origin", "arm_left"),
                                ("microscope", "center", "arm_right"),
                                ("transfer_arm", "origin", "arm_right")):
        for axis in ("x", "y"):
            assert b[section][point][axis] - b[arm]["base"][axis] == (
                old[section][point][axis] - old[arm]["base"][axis])
    for name in ("left_storage", "middle_storage", "right_storage"):
        for new_marker, old_marker in zip(b["qr_markers"][name], old["qr_markers"][name], strict=True):
            assert new_marker[0] - b["stations"][name]["region"]["x"][0] == (
                old_marker[0] - old["stations"][name]["region"]["x"][0])
            assert new_marker[1] == old_marker[1]
    assert b["regions"]["workspace"] == b["table"]
    geo = b["reactor_module"]["geometry"]
    assert geo["rod_length_mm"] == 190
    assert geo["load"]["y"] + geo["process_y_mm"] == geo["microscope_center"]["y"]
    assert geo["rod_length_mm"] - geo["process_y_mm"] == 30
    for arm, stations in (("arm_left", ["left_storage", "middle_storage"]),
                           ("arm_right", ["middle_storage", "right_storage"]),
                           ("arm_reactor", ["right_storage", "reactor_storage", "reactor"])):
        for slot in b["slots"].values():
            if slot["station"] in stations:
                assert math.hypot(slot["x"] - b[arm]["base"]["x"],
                                  slot["y"] - b[arm]["base"]["y"]) < b[arm]["reach_mm"]
