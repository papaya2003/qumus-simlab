from __future__ import annotations

import pytest

from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import ErrorCategory, NeedsHumanError, SafetyViolation, StopRequested
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import DeviceBundle, Frame
from qumus_simlab.devices.sim.devices import (
    SimArm,
    SimHeater,
    SimMicroscope,
    SimStage,
    SimSwitch,
    SimTape,
)
from qumus_simlab.safety.gate import SafetyGate
from qumus_simlab.safety.limits import Limits
from qumus_simlab.testbench.world import SimWorld


@pytest.fixture
def rig():
    clock = SimClock()
    world = SimWorld()
    bundle = DeviceBundle()
    bundle.add(SimArm("arm_right", world, Pose(x=100, y=100, z=120)))
    bundle.add(SimArm("arm_unlisted", world, Pose(x=0, y=0, z=0)))
    bundle.add(SimStage("transfer_stage"))
    bundle.add(SimHeater("stamp_heater", clock))
    bundle.add(SimMicroscope("microscope", clock))
    vac, exf_heater = SimSwitch("exfoliation_vacuum"), SimHeater("exfoliation_heater", clock)
    bundle.add(vac)
    bundle.add(exf_heater)
    bundle.add(SimTape("exfoliator", world, vac, exf_heater, clock=clock))
    events: list[tuple[str, dict]] = []
    gate = SafetyGate(bundle, Limits.load(), clock, sink=lambda k, p: events.append((k, p)))
    return gate, clock, events


def test_arm_move_inside_limits_is_logged(rig):
    gate, _, events = rig
    gate.arm("arm_right").move_to(Pose(x=10, y=10, z=50))
    assert events[-1][0] == "device.command"
    assert gate.arm("arm_right").get_pose().x == 10


@pytest.mark.parametrize("offset", [0., .1, .25])
def test_microscope_exposure_timestamp_is_inside_the_shared_clock_capture_call(rig, monkeypatch, offset):
    gate, clock, events = rig
    started = clock.now()

    def capture():
        clock.sleep(.25)
        return Frame("microscope", started+offset)

    monkeypatch.setattr(gate.devices.get("microscope"), "capture", capture)
    assert gate.microscope("microscope").capture().timestamp == started+offset
    assert not any(kind == "device.observation_rejected" for kind, _ in events)


def test_microscope_repeated_timestamp_is_valid_only_while_clock_is_unchanged(rig, monkeypatch):
    gate, clock, events = rig
    frame = gate.microscope("microscope").capture()
    monkeypatch.setattr(gate.devices.get("microscope"), "capture", lambda: frame)
    assert gate.microscope("microscope").capture() is frame
    clock.sleep(.5)
    with pytest.raises(NeedsHumanError, match="stale or invalid exposure timestamp"):
        gate.microscope("microscope").capture()
    rejected = [payload for kind, payload in events if kind == "device.observation_rejected"]
    assert rejected[-1]["reason"] == "stale_capture_timestamp"
    assert rejected[-1]["frame_timestamp"] == frame.timestamp
    assert rejected[-1]["capture_started_at"] == clock.now()
    assert rejected[-1]["clock_basis"] == clock.event_clock_basis
    assert not any(kind == "safety.violation" for kind, _ in events)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf"), None, "unknown", True])
def test_microscope_invalid_capture_timestamp_is_typed_and_serializable(rig, monkeypatch, timestamp):
    import json

    gate, _, events = rig
    monkeypatch.setattr(gate.devices.get("microscope"), "capture", lambda: Frame("microscope", timestamp))
    with pytest.raises(NeedsHumanError):
        gate.microscope("microscope").capture()
    payload = next(payload for kind, payload in events if kind == "device.observation_rejected")
    assert payload["frame_timestamp"] is None
    assert payload["reason"] == "invalid_capture_timestamp"
    json.dumps(payload, allow_nan=False)


def test_microscope_future_timestamp_cannot_become_current_evidence(rig, monkeypatch):
    gate, clock, events = rig
    monkeypatch.setattr(gate.devices.get("microscope"), "capture",
                        lambda: Frame("microscope", clock.now()+.1))
    with pytest.raises(NeedsHumanError):
        gate.microscope("microscope").capture()
    assert any(kind == "device.observation_rejected" and payload["reason"] == "future_capture_timestamp"
               for kind, payload in events)


@pytest.mark.parametrize("purpose,method", [
    ("focus_capture", "capture_for_focus"),
    ("neighbourhood_capture", "capture_for_neighbourhood"),
])
def test_focus_capture_keeps_the_same_timestamp_admission(rig, monkeypatch, purpose, method):
    gate, clock, events = rig
    monkeypatch.setattr(gate.devices.get("microscope"), method,
                        lambda: Frame("microscope", clock.now()-.1))
    with pytest.raises(NeedsHumanError):
        gate.microscope("microscope").capture(purpose=purpose)
    assert any(kind == "device.observation_rejected" and payload["reason"] == "stale_capture_timestamp"
               for kind, payload in events)


def test_unknown_microscope_capture_purpose_is_not_dispatched(rig, monkeypatch):
    gate, _, events = rig
    def unexpected():
        raise AssertionError("Unknown sampling was dispatched")
    monkeypatch.setattr(gate.devices.get("microscope"), "capture", unexpected)
    with pytest.raises(ValueError, match="capture purpose"):
        gate.microscope("microscope").capture(purpose="unobserved_preview")
    assert not events


def test_microscope_capture_preserves_cancellation(rig, monkeypatch):
    gate, _, events = rig

    def capture():
        raise StopRequested("operator stopped")

    monkeypatch.setattr(gate.devices.get("microscope"), "capture", capture)
    with pytest.raises(StopRequested, match="operator stopped"):
        gate.microscope("microscope").capture()
    assert not any(kind == "device.observation_rejected" for kind, _ in events)


def test_arm_out_of_range_is_refused_and_not_executed(rig):
    gate, _, events = rig
    with pytest.raises(SafetyViolation) as ei:
        gate.arm("arm_right").move_to(Pose(x=10, y=10, z=-20))
    assert ei.value.category == ErrorCategory.DANGEROUS
    assert events[-1][0] == "safety.violation"
    assert gate.arm("arm_right").get_pose().z == 120  # did not move
    assert gate.tripped


def test_arm_speed_limit(rig):
    gate, _, _ = rig
    with pytest.raises(SafetyViolation):
        gate.arm("arm_right").move_to(Pose(x=10, y=10, z=50), speed=99_999)


def test_device_without_limits_is_refused(rig):
    gate, _, _ = rig
    with pytest.raises(SafetyViolation, match="no limits"):
        gate.arm("arm_unlisted").move_to(Pose(x=1, y=1, z=1))


def test_interlock_blocks_stage_while_arm_above(rig):
    gate, _, _ = rig
    gate.stage("transfer_stage").move_to(Pose(x=5, y=5))  # arm far away: fine
    gate.arm("arm_right").move_to(Pose(x=-25, y=204, z=10))  # over the stage (arm frame)
    with pytest.raises(SafetyViolation, match="interlock"):
        gate.stage("transfer_stage").move_to(Pose(x=6, y=6))
    gate.reset()
    gate.arm("arm_right").move_to(Pose(x=-25, y=204, z=100))  # lifted above below_z
    gate.stage("transfer_stage").move_to(Pose(x=6, y=6))


def test_arm_refuses_beyond_its_reach(rig):
    gate, _, _ = rig
    gate.arm("arm_right").move_to(Pose(x=150, y=150, z=50))  # 212 mm from the base: fine
    with pytest.raises(SafetyViolation, match="reach"):
        gate.arm("arm_right").move_to(Pose(x=250, y=250, z=50))  # in the box, 354 mm away


def test_arm_that_lost_its_position_must_home_first(rig):
    gate, _, _ = rig
    arm = gate.arm("arm_right")
    arm.move_to(Pose(x=10, y=10, z=50))
    gate.devices.get("arm_right").lose_position()  # controller reset
    assert not arm.is_homed()
    with pytest.raises(SafetyViolation, match="not homed"):
        arm.move_to(Pose(x=20, y=20, z=50))
    gate.reset()
    arm.home()
    assert arm.is_homed()
    arm.move_to(Pose(x=20, y=20, z=50))


def test_peel_angle_limits(rig):
    gate, _, _ = rig
    with pytest.raises(SafetyViolation, match="angle"):
        gate.tape("exfoliator").peel(200, angle_deg=120)


def test_heater_max_temperature(rig):
    gate, _, _ = rig
    with pytest.raises(SafetyViolation, match="max"):
        gate.heater("stamp_heater").set_target(500)


def test_heater_ramp_limit(rig):
    gate, clock, _ = rig
    h = gate.heater("stamp_heater")
    h.set_target(70)  # +48C from ambient within the first minute: allowed (60C/min)
    with pytest.raises(SafetyViolation, match="jump"):
        h.set_target(190)
    clock.advance(120)
    h.set_target(190)  # 120C over 2 minutes: allowed


def test_heater_watchdog_forces_off_after_max_on_time(rig):
    gate, clock, events = rig
    gate.heater("stamp_heater").set_target(60)
    clock.advance(7201)
    with pytest.raises(SafetyViolation, match="longer than"):
        gate.watchdog_tick()
    assert gate.devices.get("stamp_heater").get_target() is None


def test_microscope_objective_whitelist(rig):
    gate, _, _ = rig
    gate.microscope("microscope").set_objective(50)
    with pytest.raises(SafetyViolation):
        gate.microscope("microscope").set_objective(42)


def test_guarded_type_mismatch(rig):
    gate, _, _ = rig
    with pytest.raises(TypeError):
        gate.heater("arm_right")
