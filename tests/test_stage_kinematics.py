"""Actual simulated stage, optics, scene snapshots and mechanics-v2 recovery."""

from __future__ import annotations

import copy
import math
import pickle
from pathlib import Path

import pytest
from conftest import quiet_bench

from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.core.types import Pose
from qumus_simlab.testbench.lab import build_sim_lab
from qumus_simlab.testbench.persist import lab_state, load_lab_state, save_lab_state
from qumus_simlab.testbench.render import MicroscopeRenderer
from qumus_simlab.testbench.world import STAMP_HOLDER, SimFlake, SimObject, StampState


def _lab(tmp_path, *, dynamics=False):
    cfg = quiet_bench(stage__sigma_mm=0)
    cfg["microscope"]["focus_tilt_per_mm"] = 0
    cfg["stamp"]["manipulator_sigma_mm"] = 0
    cfg["motion"].update(stage_speed_mm_s=0, fine_speed_mm_s=0,
                         rotation_speed_deg_s=0, focus_speed_mm_s=0,
                         manipulator_speed_mm_s=0)
    if not dynamics:
        cfg["stamp"]["dynamics"].update(lateral_drift_mm_s=[0, 0],
                                          approach_drift_mm_per_mm=[0, 0],
                                          contact_slip_deg_s=0)
    clock = SimClock()
    lab = build_sim_lab(clock, seed=19, config=cfg)
    lab.devices.get("microscope").renderer = MicroscopeRenderer(tmp_path, size=(192, 128))
    return lab, clock


def _mount_chip(lab, offset=(2.4, -1.7), rotation=17.3):
    pivot = lab.world.surfaces["transfer_stage"].pivot()
    flake = SimFlake("geometry_flake", "Gr", 1, 0.08, -0.06, 12000,
                     outline_mm=((-0.08, -0.05), (0.09, -0.04), (0.02, 0.09)))
    chip = SimObject("geometry_chip", "chip",
                     Pose(x=pivot[0]+offset[0], y=pivot[1]+offset[1], rotation=rotation),
                     on="transfer_stage", size_mm=(4, 3), focus_offset=0.37123456789,
                     flakes=[flake])
    lab.world.add(chip)
    return chip, flake


def _under_objective(lab, chip):
    scope, stage = lab.devices.get("microscope"), lab.devices.get("transfer_stage")
    stage.move_to(Pose(x=scope.center[0]-chip.pose.x, y=scope.center[1]-chip.pose.y))
    return stage, scope


def _mount_stamp(lab, *, contact=False):
    w = lab.world
    over = lab.bench["transfer_arm"]["over_stage"]
    w.manip.x, w.manip.y = over["x"], over["y"]
    w.manip.z = 5.3 if contact else 6.5
    sx, sy = w.seat_xy()
    stamp = SimObject("geometry_stamp", "stamp", Pose(x=sx, y=sy), held_by=STAMP_HOLDER)
    w.add(stamp)
    flake = SimFlake("stamp_flake", "hBN", 2, 0, 0, 10000,
                     outline_mm=((-0.05, -0.03), (0.06, -0.02), (0, 0.07)))
    state = StampState(touch_z=5.3, apex_dx=0, apex_dy=0, melt_c=165,
                       optic_shift=0.123456789, stack=[(flake, 0.08, -0.04)])
    w.stamps[stamp.id] = state
    # Establish the initial physical time/holder reading before comparing polls.
    w.advance_transfer(lab.devices.get("microscope").clock.now())
    return stamp, state


def _rotate(point, pivot, delta_deg, translation=(0, 0)):
    r = math.radians(delta_deg)
    dx, dy = point[0]-pivot[0], point[1]-pivot[1]
    return (pivot[0]+translation[0]+dx*math.cos(r)-dy*math.sin(r),
            pivot[1]+translation[1]+dx*math.sin(r)+dy*math.cos(r))


def _physical_state(lab, clock):
    return copy.deepcopy({"objects": lab.world.objects, "stamps": lab.world.stamps,
                          "manip": lab.world.manip, "surfaces": lab.world.surfaces,
                          "dynamics": lab.world.transfer_dynamics,
                          "collisions": lab.world.collisions,
                          "physical_rng": lab.world.rng.getstate(),
                          "observation_rng": lab.world.observation_rng.getstate(),
                          "clock": clock.now()})


def test_coarse_fine_and_rotation_carry_off_centre_chip_and_flake(tmp_path):
    lab, _ = _lab(tmp_path)
    chip, flake = _mount_chip(lab)
    stage = lab.devices.get("transfer_stage")
    pivot = lab.world.surfaces["transfer_stage"].pivot()
    original_chip = chip.pose
    original_flake = chip.local_to_world(flake.x_mm, flake.y_mm)
    original_outline = [chip.local_to_world(flake.x_mm+x, flake.y_mm+y)
                        for x, y in flake.outline_mm]
    stage.move_to(Pose(x=3.25, y=42.5))
    stage.move_fine_to(0.375, -0.625)
    assert (chip.pose.x, chip.pose.y) == pytest.approx(
        (original_chip.x+3.625, original_chip.y+41.875))
    stage.rotate_to(67)
    assert stage.get_pose() == Pose(x=3.25, y=42.5, rotation=67)
    assert stage.get_sample_pose() == Pose(x=3.625, y=41.875, rotation=67)
    assert (chip.pose.x, chip.pose.y) == pytest.approx(
        _rotate((original_chip.x, original_chip.y), pivot, 67, (3.625, 41.875)))
    assert chip.pose.rotation == pytest.approx(original_chip.rotation+67)
    assert chip.local_to_world(flake.x_mm, flake.y_mm) == pytest.approx(
        _rotate(original_flake, pivot, 67, (3.625, 41.875)))
    for original, (x, y) in zip(original_outline, flake.outline_mm, strict=True):
        assert chip.local_to_world(flake.x_mm+x, flake.y_mm+y) == pytest.approx(
            _rotate(original, pivot, 67, (3.625, 41.875)))
    # A fine jog after rotation still follows world X/Y, rather than rotated axes.
    before = chip.pose
    stage.move_fine_to(0.575, -0.025)
    assert (chip.pose.x-before.x, chip.pose.y-before.y) == pytest.approx((0.2, 0.6))
    kinematics = stage.kinematics()
    assert kinematics["mechanics_version"] == 2
    assert kinematics["pivot_world_mm"] == pytest.approx((pivot[0]+3.825, pivot[1]+42.475))
    farthest = max(math.hypot(x-kinematics["pivot_world_mm"][0],
                              y-kinematics["pivot_world_mm"][1])
                   for x, y in (chip.local_to_world(x, y)
                                for x in (-2, 2) for y in (-1.5, 1.5)))
    assert kinematics["carried_radius_mm"] >= farthest


def test_focus_is_independent_of_sample_stamp_and_mechanical_gap(tmp_path):
    lab, _ = _lab(tmp_path)
    chip, _ = _mount_chip(lab)
    stage, scope = _under_objective(lab, chip)
    stamp, _ = _mount_stamp(lab)
    before = (chip.pose, stamp.pose, stage.get_pose(), stage.get_fine_position(),
              copy.deepcopy(lab.world.manip), lab.world.apex()[3],
              lab.world.manip.z-lab.world.stage_height())
    scope.set_focus(chip.focus_offset)
    chip_frame = scope.capture()
    scope.set_focus(chip_frame.meta["best_focus_z"])
    through_stamp_frame = scope.capture()
    assert through_stamp_frame.meta["sharpness_chip"] > chip_frame.meta["sharpness_chip"]
    scope.set_focus(1.0)
    assert before == (chip.pose, stamp.pose, stage.get_pose(), stage.get_fine_position(),
                      lab.world.manip, lab.world.apex()[3],
                      lab.world.manip.z-lab.world.stage_height())
    assert scope.get_focus() == 1
    assert chip_frame.meta["best_focus_z"] == pytest.approx(
        chip.focus_offset+lab.world.stamps[stamp.id].optic_shift)
    assert scope.geometry_state()["sample_pose"]["z"] == stage.get_pose().z == 0


def test_capture_calibration_pose_binding_and_twin_keep_raw_precision(tmp_path):
    lab, clock = _lab(tmp_path)
    chip, _ = _mount_chip(lab)
    stage, scope = _under_objective(lab, chip)
    stage.move_fine_to(0.000123456789, -0.000987654321)
    stage.rotate_to(0.123456789)
    scope.set_focus(0.37123456789)
    frame = scope.capture()
    metadata = frame.meta
    assert metadata["captured_at"] == frame.timestamp == clock.now()
    assert metadata["mechanics_version"] == metadata["scene"]["mechanics_version"] == 2
    assert metadata["sample_pose"] == scope.geometry_state()["sample_pose"]
    assert metadata["stage_kinematics"] == stage.kinematics()
    assert metadata["pose_binding"] == {
        "captured_at": frame.timestamp, "focus_z": scope.get_focus(), "objective": 10,
        "sample_pose": metadata["sample_pose"], "stage_kinematics": stage.kinematics()}
    calibration = metadata["calibration"]
    assert calibration["source"] == "synthetic"
    assert calibration["id"] and calibration["version"]
    assert calibration["image_size_px"] == [frame.width, frame.height] == [192, 128]
    assert calibration["mm_per_pixel"] == pytest.approx(
        [metadata["fov_mm"][0]/frame.width, metadata["fov_mm"][1]/frame.height])
    assert calibration["u_axis"] == "+world_x" and calibration["v_axis"] == "-world_y"
    twin = lab.twin.state()
    assert twin["geometry"]["stage"] == stage.kinematics()
    assert twin["geometry"]["microscope"] == scope.geometry_state()
    assert twin["geometry"]["stage"]["fine_xy_mm"][0] == 0.000123456789
    assert twin["geometry"]["microscope"]["focus_z"] == 0.37123456789
    assert twin["stages"]["transfer_stage"]["fine_xy_mm"][0] == 0
    assert twin["microscope"]["focus_z"] == 0.371
    # Calibration changes with actual objective/FOV instead of reusing an old ID.
    scope.set_objective(20)
    assert scope.capture().meta["calibration"]["version"] != calibration["version"]


def test_delayed_capture_save_uses_frozen_scene_and_pixels_after_motion(tmp_path):
    lab, _ = _lab(tmp_path)
    chip, _ = _mount_chip(lab)
    stage, scope = _under_objective(lab, chip)
    scope.set_focus(chip.focus_offset)
    frame = scope.capture()
    frozen = copy.deepcopy(frame.meta)
    assert frozen["flakes"] and frozen["on_chip"] == chip.id
    expected_path = Path(scope.renderer.render_preview("expected", frozen, immutable=True))
    expected_pixels = expected_path.read_bytes()
    stage.move_fine_to(0.7, -0.2)
    stage.rotate_to(25)
    scope.set_focus(-1)
    scope.set_objective(20)
    current = scope.preview_snapshot()
    assert Path(current.saved_path()).read_bytes() != expected_pixels
    saved = Path(frame.saved_path())
    assert saved.read_bytes() == expected_pixels
    assert {key: frame.meta[key] for key in frozen} == frozen
    assert frame.saved_path() == str(saved)
    stage.move_to(Pose(x=20, y=10, rotation=-40))
    scope.preview()
    assert saved.read_bytes() == expected_pixels
    assert {key: frame.meta[key] for key in frozen} == frozen


def test_polling_at_fixed_clock_preserves_physics_and_both_rng_streams(tmp_path):
    lab, clock = _lab(tmp_path, dynamics=True)
    chip, _ = _mount_chip(lab)
    _, scope = _under_objective(lab, chip)
    _, stamp_state = _mount_stamp(lab, contact=True)
    clock.advance(2)
    lab.world.advance_transfer(clock.now())
    assert stamp_state.front_radius_mm > 0 and stamp_state.slip_rotation_deg > 0
    assert stamp_state.drift_xy_mm != (0, 0)
    before = _physical_state(lab, clock)
    live_paths = []
    snapshot_paths = []
    for _ in range(4):
        live_paths.append(scope.preview().saved_path())
        snapshot_paths.append(scope.preview_snapshot().saved_path())
        scope.geometry_state()
        lab.twin.state()
        lab.preview("microscope")
    assert _physical_state(lab, clock) == before
    assert len(set(live_paths)) == 1
    # Identical immutable snapshots share one numbered file (render.ImageStore
    # never rewrites a numbered file and does not copy an unchanged frame).
    assert len(set(snapshot_paths)) == 1 and not set(snapshot_paths) & set(live_paths)
    assert len({Path(p).read_bytes() for p in [*live_paths, *snapshot_paths]}) == 1


def test_mechanics_v2_checkpoint_restores_pose_focus_dynamics_and_continuation(tmp_path):
    original, clock = _lab(tmp_path / "original", dynamics=True)
    chip, _ = _mount_chip(original)
    stage, scope = _under_objective(original, chip)
    stage.move_fine_to(0.123456789, -0.234567891)
    stage.rotate_to(13.123456789)
    scope.set_focus(-0.123456789)
    _, stamp = _mount_stamp(original, contact=True)
    clock.advance(3)
    original.world.advance_transfer(clock.now())
    assert stamp.front_radius_mm > 0
    original.world.rng.random()
    original.world.observation_rng.random()
    checkpoint = tmp_path / "checkpoint.pkl"
    save_lab_state(original, clock, checkpoint)
    saved_bytes = checkpoint.read_bytes()
    saved = pickle.loads(saved_bytes)
    restored, restored_clock = _lab(tmp_path / "restored", dynamics=True)
    assert load_lab_state(restored, restored_clock, checkpoint)
    actual = pickle.loads(pickle.dumps(lab_state(restored, restored_clock)))
    # Renderer dependencies remain configured; Random equality needs its full state.
    assert actual["world"].pop("_observation_rng").getstate() == saved["world"].pop(
        "_observation_rng").getstate()
    assert actual == saved
    restored_stage = restored.devices.get("transfer_stage")
    restored_scope = restored.devices.get("microscope")
    assert restored_scope.stage is restored_stage
    assert restored_scope.renderer is not None
    assert restored_stage.get_fine_position() == (0.123456789, -0.234567891)
    assert restored_stage.get_pose().rotation == 13.123456789
    assert restored_scope.get_focus() == -0.123456789
    assert restored.world.surfaces["transfer_stage"].rotation == 13.123456789
    assert restored.world.objects[chip.id].pose == chip.pose
    assert restored.world.stamps["geometry_stamp"] == original.world.stamps["geometry_stamp"]
    assert restored.world.transfer_dynamics == original.world.transfer_dynamics
    assert restored_clock.now() == clock.now()
    # The same elapsed time continues drift/slip/front growth identically.
    for lab, time in ((original, clock), (restored, restored_clock)):
        time.advance(1.25)
        lab.world.advance_transfer(time.now())
    assert restored.world.stamps == original.world.stamps
    assert restored.world.transfer_dynamics == original.world.transfer_dynamics
    assert restored.world.rng.random() == original.world.rng.random()
    assert restored.world.observation_rng.random() == original.world.observation_rng.random()
    assert checkpoint.read_bytes() == saved_bytes


@pytest.mark.parametrize("incompatibility", ["missing_version", "missing_signature", "bad_signature",
                                            "legacy_version", "changed_pivot", "changed_focus",
                                            "changed_dynamics"])
def test_incompatible_checkpoint_rejects_before_overlay_or_file_mutation(tmp_path, incompatibility):
    original, clock = _lab(tmp_path / "original", dynamics=True)
    chip, _ = _mount_chip(original)
    stage, scope = _under_objective(original, chip)
    stage.move_fine_to(0.2, -0.1)
    stage.rotate_to(30)
    scope.set_focus(0.5)
    clock.advance(12)
    saved = pickle.loads(pickle.dumps(lab_state(original, clock)))
    fresh, fresh_clock = _lab(tmp_path / "fresh", dynamics=True)
    if incompatibility == "missing_version":
        saved.pop("mechanics_version")
    elif incompatibility == "missing_signature":
        saved.pop("mechanics_signature")
    elif incompatibility == "bad_signature":
        saved["mechanics_signature"] = "different"
    elif incompatibility == "legacy_version":
        saved["mechanics_version"] = 1
    elif incompatibility == "changed_pivot":
        fresh.bench["transfer_stage"]["pivot"] = "chip_center"
    elif incompatibility == "changed_focus":
        fresh.bench["microscope"]["focal_plane_z"] += 0.1
    else:
        fresh.world.model["stamp"]["dynamics"]["contact_slip_deg_s"] += 0.001
    checkpoint = tmp_path / "incompatible.pkl"
    checkpoint.write_bytes(pickle.dumps(saved))
    saved_bytes = checkpoint.read_bytes()
    before = pickle.dumps(lab_state(fresh, fresh_clock))
    with pytest.raises(PreconditionFailed, match="incompatible stage/focus mechanics") as error:
        load_lab_state(fresh, fresh_clock, checkpoint)
    assert "History read-only" in str(error.value)
    assert "fresh" in str(error.value)
    assert pickle.dumps(lab_state(fresh, fresh_clock)) == before
    assert checkpoint.read_bytes() == saved_bytes
