"""Mechanical geometry guards reject whole swept paths before any mutation."""

from __future__ import annotations

import math

import pytest

from qumus_simlab.core.errors import SafetyViolation
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import DeviceBundle, Frame, Microscope, MotionStage, StampManipulator
from qumus_simlab.safety.gate import SafetyGate
from qumus_simlab.safety.limits import Limits


class Stage(MotionStage):
    is_simulated = True

    def __init__(self):
        super().__init__("transfer_stage")
        self.pose = Pose(x=0, y=0)
        self.fine = (0.0, 0.0)
        self.radius = 15.0
        self.top = 1.5
        self.commands = []

    def move_to(self, pose):
        self.commands.append(("coarse", pose))
        self.pose = pose

    def get_pose(self):
        return self.pose

    def move_fine_to(self, x, y):
        self.commands.append(("fine", x, y))
        self.fine = (x, y)

    def get_fine_position(self):
        return self.fine

    def kinematics(self):
        return {"carried_radius_mm": self.radius, "max_sample_surface_z": self.top}


class Stamp(StampManipulator):
    is_simulated = True

    def __init__(self):
        super().__init__("transfer_arm")
        self.positions = {"x": 150.0, "y": 10.0, "z": 6.5}
        self.commands = []

    def get_positions(self):
        return dict(self.positions)

    def move_axis_to(self, axis, value, speed=None):
        self.commands.append((axis, value))
        self.positions[axis] = value


class Scope(Microscope):
    is_simulated = True

    def __init__(self):
        super().__init__("microscope")
        self.focus = 0.0
        self.commands = []

    def get_focus(self):
        return self.focus

    def set_focus(self, z):
        self.commands.append(z)
        self.focus = z

    def get_objective(self):
        return 10

    def set_objective(self, magnification):
        self.commands.append(magnification)

    def capture(self):
        return Frame(device_id=self.device_id, timestamp=0)


@pytest.fixture
def rig():
    stage, stamp, scope = Stage(), Stamp(), Scope()
    bundle = DeviceBundle()
    for device in (stage, stamp, scope):
        bundle.add(device)
    events = []
    limits = Limits.load()
    limits.interlocks = []  # this focused fixture has no arm devices
    gate = SafetyGate(bundle, limits, sink=lambda kind, data: events.append((kind, data)))
    return gate, stage, stamp, scope, events


def test_fine_and_rotation_compose_and_log(rig):
    gate, stage, _, _, events = rig
    guarded = gate.stage("transfer_stage")
    guarded.move_to(Pose(x=2, y=3))
    guarded.move_fine_to(0.2, -0.3)
    guarded.rotate_to(45)
    assert guarded.get_fine_position() == (0.2, -0.3)
    assert guarded.get_pose() == Pose(x=2, y=3, rotation=45)
    assert guarded.get_sample_pose() == Pose(x=2.2, y=2.7, rotation=45)
    assert len(stage.commands) == 3
    assert [data["command"] for kind, data in events if kind == "device.command"] == [
        "move_to", "move_fine_to", "rotate_to"]


@pytest.mark.parametrize("operation", ["coarse", "fine", "rotation", "focus", "stamp", "speed"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_commands_reject_before_mutation(rig, operation, value):
    gate, stage, stamp, scope, events = rig
    operations = {
        "coarse": lambda: gate.stage(stage.device_id).move_to(Pose(x=value, y=0)),
        "fine": lambda: gate.stage(stage.device_id).move_fine_to(value, 0),
        "rotation": lambda: gate.stage(stage.device_id).rotate_to(value),
        "focus": lambda: gate.microscope(scope.device_id).set_focus(value),
        "stamp": lambda: gate.manipulator(stamp.device_id).move_axis_to("x", value),
        "speed": lambda: gate.manipulator(stamp.device_id).move_axis_to("x", 140, speed=value),
    }
    with pytest.raises(SafetyViolation):
        operations[operation]()
    assert not stage.commands and not stamp.commands and not scope.commands
    assert [kind for kind, _ in events] == ["safety.violation"]


@pytest.mark.parametrize("field,operation", [
    ("fine_x", lambda stage: stage.move_fine_to(0.1, 0)),
    ("fine_y", lambda stage: stage.move_fine_to(0, 0.1)),
    ("rotation", lambda stage: stage.rotate_to(0)),
    ("rotation", lambda stage: stage.move_to(Pose(x=0, y=0))),
    ("fine_x", lambda stage: stage.move_to(Pose(x=0, y=0))),
    ("fine_y", lambda stage: stage.move_to(Pose(x=0, y=0))),
])
def test_missing_axis_limits_refuse_even_nominal_commands(rig, field, operation):
    gate, stage, *_ = rig
    setattr(gate.limits.stages[stage.device_id], field, None)
    with pytest.raises(SafetyViolation, match="configured"):
        operation(gate.stage(stage.device_id))
    assert not stage.commands


def test_pure_rotation_cannot_use_lowering_exception(rig):
    gate, stage, stamp, *_ = rig
    stamp.positions.update(x=40, y=90, z=5)
    with pytest.raises(SafetyViolation, match="stamp is low"):
        gate.stage(stage.device_id).move_to(Pose(x=0, y=0, rotation=30))
    assert not stage.commands


@pytest.mark.parametrize("operation", ["rotate", "fine"])
def test_carried_radius_guards_rotation_and_fine_outside_legacy_box(rig, operation):
    gate, stage, stamp, *_ = rig
    stage.radius = 35  # mounted away from pivot: full rotating assembly radius
    stamp.positions.update(x=75, y=10, z=5)
    with pytest.raises(SafetyViolation, match="swept stage/stamp"):
        if operation == "rotate":
            gate.stage(stage.device_id).rotate_to(180)
        else:
            gate.stage(stage.device_id).move_fine_to(0.5, 0)
    assert not stage.commands


def test_stage_translation_rejects_intermediate_collision(rig):
    gate, stage, stamp, *_ = rig
    stamp.positions.update(x=40, y=50, z=5)
    # Both ends are 40 mm from stamp; the middle passes directly beneath it.
    with pytest.raises(SafetyViolation, match="swept stage/stamp"):
        gate.stage(stage.device_id).move_to(Pose(x=0, y=80))
    assert stage.pose == Pose(x=0, y=0) and not stage.commands


def test_holder_translation_rejects_intermediate_collision(rig):
    gate, stage, stamp, *_ = rig
    stamp.positions.update(x=0, y=10, z=5)
    with pytest.raises(SafetyViolation, match="swept stage/stamp"):
        gate.manipulator(stamp.device_id).move_axis_to("x", 80)
    assert stamp.positions["x"] == 0 and not stamp.commands and not stage.commands


def test_holder_sweep_rejects_objective_with_safe_endpoints(rig):
    gate, _, stamp, scope, _ = rig
    # limits.yaml puts the objective bottom 12 mm above focus zero (#204), so at
    # the lowest focus (-3) it is at 9: a holder at z 8 (top 9) is too close.
    stamp.positions.update(x=0, y=90, z=8)
    scope.focus = -3
    with pytest.raises(SafetyViolation, match="objective/stamp"):
        gate.manipulator(stamp.device_id).move_axis_to("x", 80)
    assert not stamp.commands


def test_stage_sweep_rejects_objective_when_sample_is_tall(rig):
    gate, stage, _, scope, _ = rig
    stage.pose = Pose(x=0, y=40)
    stage.top = 10  # objective bottom at 12 - 2 = 10
    scope.focus = -2  # sample/objective endpoints both separated in XY
    with pytest.raises(SafetyViolation, match="objective/sample"):
        gate.stage(stage.device_id).move_to(Pose(x=0, y=100))
    assert not stage.commands


@pytest.mark.parametrize("obstacle", ["sample", "stamp"])
def test_focus_stroke_checks_both_mechanical_clearances(rig, obstacle):
    gate, stage, stamp, scope, _ = rig
    # objective bottom = 12 + focus (limits.yaml, #204)
    if obstacle == "stamp":
        stamp.positions.update(x=40, y=90, z=8)  # stamp top 9; bottom at -3 is 9
        target = -3
    else:
        stage.pose = Pose(x=0, y=80)
        stage.top = 10  # bottom at -2 is 10
        target = -2
    before = stage.pose, stage.fine, dict(stamp.positions), scope.focus
    with pytest.raises(SafetyViolation, match=f"objective/{obstacle}"):
        gate.microscope(scope.device_id).set_focus(target)
    assert before == (stage.pose, stage.fine, stamp.positions, scope.focus)
    assert not scope.commands


def test_allowed_focus_changes_no_mechanical_sample_or_gap(rig):
    gate, stage, stamp, scope, _ = rig
    before = stage.pose, stage.fine, dict(stamp.positions)
    gate.microscope(scope.device_id).set_focus(1)
    assert before == (stage.pose, stage.fine, stamp.positions)
    assert scope.focus == 1


@pytest.mark.parametrize("device,operation", [
    ("stage", lambda gate: gate.stage("transfer_stage").move_to(Pose(x=1, y=0))),
    ("stamp", lambda gate: gate.manipulator("transfer_arm").move_axis_to("x", 140)),
    ("scope", lambda gate: gate.microscope("microscope").set_focus(1)),
])
def test_synthetic_limits_cannot_authorize_real_actuation(rig, device, operation):
    gate, stage, stamp, scope, _ = rig
    {"stage": stage, "stamp": stamp, "scope": scope}[device].is_simulated = False
    with pytest.raises(SafetyViolation, match="simulation-only"):
        operation(gate)
    assert not stage.commands and not stamp.commands and not scope.commands


def test_real_unknown_calibration_refuses_before_mutation(rig):
    gate, stage, *_ = rig
    stage.is_simulated = False
    lim = gate.limits.stages[stage.device_id]
    lim.source, lim.simulation_only = "unknown", False
    lim.calibration_id = None
    with pytest.raises(SafetyViolation, match="measured calibrated"):
        gate.stage(stage.device_id).move_to(Pose(x=1, y=0))
    assert not stage.commands


def test_missing_swept_limits_refuse_before_mutation(rig):
    gate, stage, *_ = rig
    gate.limits.geometry = None
    with pytest.raises(SafetyViolation, match="no swept geometry"):
        gate.stage(stage.device_id).move_to(Pose(x=1, y=0))
    assert not stage.commands


@pytest.mark.parametrize("operation", [
    lambda gate: gate.stage("transfer_stage").move_fine_to(1.01, 0),
    lambda gate: gate.stage("transfer_stage").rotate_to(181),
    lambda gate: gate.stage("transfer_stage").move_to(Pose(x=0, y=0, z=0.01)),
    lambda gate: gate.microscope("microscope").set_focus(3.01),
])
def test_configured_axis_bounds_reject_before_mutation(rig, operation):
    gate, stage, stamp, scope, _ = rig
    with pytest.raises(SafetyViolation):
        operation(gate)
    assert not stage.commands and not stamp.commands and not scope.commands


def test_motion_observer_restored_on_driver_failure(rig):
    gate, stage, *_ = rig
    frames, previous_calls = [], []
    def previous():
        previous_calls.append(True)

    stage.motion_observer = previous
    gate.frames = lambda: {"sample": stage.get_sample_pose().model_dump()}
    gate.sink = lambda kind, data: frames.append((kind, data))

    def failed_motion(pose):
        stage.pose = Pose(x=pose.x / 2, y=pose.y)
        stage.motion_observer()
        raise RuntimeError("driver interrupted")

    stage.move_to = failed_motion
    with pytest.raises(RuntimeError, match="driver interrupted"):
        gate.stage(stage.device_id).move_to(Pose(x=2, y=0))
    assert stage.motion_observer is previous
    assert not previous_calls
    assert [kind for kind, _ in frames] == ["device.command", "lab.frame", "lab.frame"]
    assert frames[1][1]["sample"]["x"] == 1
