"""SafetyGate: the only path from flow code to devices (spec 7.2).

Flow code asks the gate for a guarded device (`gate.arm("arm_right")`) and
never touches the raw driver. Every command is checked against
config/limits.yaml and the interlocks, then logged as a `device.command` event;
what the command changed in the lab follows as a `lab.frame` event.
A refused command raises SafetyViolation (category DANGEROUS) and is logged as
`safety.violation`; the job service then puts the job in safe state and does
not auto-recover.

The same gate wraps the testbench and the real drivers, so a violation on the
testbench is a test failure.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from qumus_simlab.core.clock import Clock, RealClock
from qumus_simlab.core.errors import NeedsHumanError, SafetyViolation
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import (
    Arm,
    Camera,
    Device,
    DeviceBundle,
    Frame,
    GasFlowController,
    Heater,
    Microscope,
    MotionStage,
    StampManipulator,
    Switch,
    TapeMechanism,
)
from qumus_simlab.safety.limits import Limits

# (kind, payload) -> None. The runtime passes RunStore.emit bound to a job.
EventSink = Callable[[str, dict[str, Any]], None]


def _null_sink(kind: str, payload: dict[str, Any]) -> None:
    pass


def _in(v: float, rng: tuple[float, float]) -> bool:
    return math.isfinite(v) and rng[0] <= v <= rng[1]


def _segment_distance(point: tuple[float, float], start: tuple[float, float],
                      end: tuple[float, float]) -> float:
    """Exact closest approach along a linear axis or composed stage move."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    t = (max(0.0, min(1.0, ((point[0] - start[0]) * dx
                           + (point[1] - start[1]) * dy) / length2)) if length2 else 0.0)
    return math.hypot(start[0] + t * dx - point[0], start[1] + t * dy - point[1])


class SafetyGate:
    def __init__(
        self,
        devices: DeviceBundle,
        limits: Limits | None = None,
        clock: Clock | None = None,
        sink: EventSink | None = None,
        frames: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self.devices = devices
        #: what changed in the lab since the last frame (runtime/twin.py LabTwin.delta);
        #: logged after every command as `lab.frame`
        self.frames = frames
        self.limits = limits or Limits.load()
        self.clock = clock or RealClock()
        self.sink = sink or _null_sink
        self._heater_on_since: dict[str, float] = {}
        self._heater_last_set: dict[str, tuple[float, float]] = {}  # id -> (time, setpoint)
        self._tripped = False

    # ----- accessors -------------------------------------------------------

    def arm(self, device_id: str) -> GuardedArm:
        return GuardedArm(self, self._typed(device_id, Arm))

    def stage(self, device_id: str) -> GuardedStage:
        return GuardedStage(self, self._typed(device_id, MotionStage))

    def heater(self, device_id: str) -> GuardedHeater:
        return GuardedHeater(self, self._typed(device_id, Heater))

    def camera(self, device_id: str) -> GuardedCamera:
        return GuardedCamera(self, self._typed(device_id, Camera))

    def microscope(self, device_id: str) -> GuardedMicroscope:
        return GuardedMicroscope(self, self._typed(device_id, Microscope))

    def switch(self, device_id: str) -> GuardedSwitch:
        return GuardedSwitch(self, self._typed(device_id, Switch))

    def tape(self, device_id: str) -> GuardedTape:
        return GuardedTape(self, self._typed(device_id, TapeMechanism))

    def manipulator(self, device_id: str) -> GuardedManipulator:
        return GuardedManipulator(self, self._typed(device_id, StampManipulator))

    def gas_flow(self, device_id: str) -> GuardedGasFlow:
        return GuardedGasFlow(self, self._typed(device_id, GasFlowController))

    def _typed(self, device_id: str, cls: type[Device]) -> Any:
        dev = self.devices.get(device_id)
        if not isinstance(dev, cls):
            raise TypeError(f"{device_id} is a {type(dev).__name__}, not a {cls.__name__}")
        return dev

    # ----- checks ----------------------------------------------------------

    @property
    def tripped(self) -> bool:
        """True after any violation until reset() (job service decides when)."""
        return self._tripped

    def reset(self) -> None:
        self._tripped = False

    def refuse(self, device_id: str, command: str, reason: str, **params: Any) -> None:
        self._tripped = True
        payload = {"device": device_id, "command": command, "params": params, "reason": reason}
        self.sink("safety.violation", payload)
        raise SafetyViolation(f"{device_id}.{command} refused: {reason}", **payload)

    def log(self, device_id: str, command: str, **params: Any) -> None:
        self.sink("device.command", {"device": device_id, "command": command, "params": params})

    @contextmanager
    def command(self, device_id: str, command: str, **params: Any) -> Iterator[None]:
        """Log a checked command, run it, then log what it changed in the lab
        (`lab.frame`), also when the device raised."""
        self.log(device_id, command, **params)
        dev = self.devices.get(device_id)
        previous_observer = dev.motion_observer
        dev.motion_observer = lambda: self._frame(device_id, command)
        try:
            yield
        finally:
            dev.motion_observer = previous_observer
            self._frame(device_id, command)

    def _frame(self, device_id: str, command: str) -> None:
        if self.frames is None:
            return
        try:
            delta = self.frames()
        except Exception:  # noqa: BLE001 - a picture of the lab never stops a command
            return
        if delta:
            self.sink("lab.frame", {"device": device_id, "command": command, **delta})

    def check_interlocks(self, device_id: str, command: str) -> None:
        for il in self.limits.interlocks:
            if device_id not in il.blocks:
                continue
            arm = self.devices.get(il.arm)
            assert isinstance(arm, Arm)
            p = arm.get_pose()
            over = all(_in(getattr(p, axis), rng) for axis, rng in il.region.items())
            if over and p.z < il.below_z:
                self.refuse(device_id, command, f"interlock {il.name}: {il.arm} is above it")

    def check_calibration(self, device_id: str, command: str, limits: Any) -> None:
        """Unmeasured synthetic geometry is usable only by explicit simulators."""
        simulated = self.devices.get(device_id).is_simulated
        if limits.simulation_only or limits.source == "synthetic":
            if not simulated or not limits.simulation_only or limits.source != "synthetic":
                self.refuse(device_id, command, "synthetic limits are simulation-only")
        elif not simulated and (limits.source != "measured" or not limits.calibration_id
                                or limits.calibration_id == "unknown"):
            self.refuse(device_id, command, "real motion requires measured calibrated limits")

    def check_swept_geometry(self, device_id: str, command: str, *,
                             sample_target: Pose | None = None,
                             stamp_target: dict[str, float] | None = None,
                             focus_target: float | None = None) -> None:
        """Check one shared mechanical envelope before any actuator mutation.

        Full carried circles bound intermediate rotation, while segment distance
        bounds translation continuously. Optical apparent shifts are never used.
        """
        cfg = self.limits.geometry
        if cfg is None:
            if device_id in ("transfer_stage", "transfer_arm", "microscope"):
                self.refuse(device_id, command, "no swept geometry limits configured")
            return
        if device_id not in (cfg.stage, cfg.manipulator, cfg.microscope):
            return
        self.check_calibration(device_id, command, cfg)
        st = self.devices.devices.get(cfg.stage)
        current = st.get_sample_pose() if isinstance(st, MotionStage) else Pose(x=0, y=0)
        target = sample_target or current
        if not all(math.isfinite(v) for p in (current, target)
                   for v in (p.x, p.y, p.z, p.rotation)):
            self.refuse(device_id, command, "stage geometry reading must be finite")
        start = (cfg.stage_origin_xy[0] + current.x, cfg.stage_origin_xy[1] + current.y)
        end = (cfg.stage_origin_xy[0] + target.x, cfg.stage_origin_xy[1] + target.y)
        radius = cfg.carried_radius_mm
        sample_top = max(current.z, target.z) + cfg.sample_height_above_stage_mm
        if isinstance(st, MotionStage):
            state = st.kinematics()
            pivot = state.get("pivot_world_mm")
            if pivot is not None:
                if len(pivot) != 2 or not all(math.isfinite(v) for v in pivot):
                    self.refuse(device_id, command, "stage world pivot must be finite")
                # The adapter's calibrated layout is authoritative; composed
                # benches may translate the whole workcell from its base YAML.
                start = tuple(pivot)
                end = (start[0]+target.x-current.x, start[1]+target.y-current.y)
            carried = state.get("carried_radius_mm")
            top = state.get("max_sample_surface_z")
            if carried is not None:
                if not math.isfinite(carried) or carried < 0:
                    self.refuse(device_id, command, "carried sample radius is unknown")
                radius = max(radius, carried)
            if top is not None:
                if not math.isfinite(top):
                    self.refuse(device_id, command, "sample surface height is unknown")
                sample_top = max(sample_top, top + max(0.0, target.z - current.z))
            if not st.is_simulated and (carried is None or top is None):
                self.refuse(device_id, command, "measured carried sample envelope is required")
        mic = self.devices.devices.get(cfg.microscope)
        optical = mic.geometry_state().get("optical_axis_world_mm") if isinstance(mic, Microscope) else None
        objective_xy = tuple(optical) if optical is not None else cfg.objective_xy
        if len(objective_xy) != 2 or not all(math.isfinite(v) for v in objective_xy):
            self.refuse(device_id, command, "optical axis world position must be finite")
        focus = mic.get_focus() if isinstance(mic, Microscope) else 0.0
        if not math.isfinite(focus) or (focus_target is not None
                                        and not math.isfinite(focus_target)):
            self.refuse(device_id, command, "focus geometry reading must be finite")
        objective_bottom = cfg.objective_bottom_at_focus_zero_z + min(
            focus, focus if focus_target is None else focus_target)
        if (_segment_distance(objective_xy, start, end)
                <= radius + cfg.objective_radius_mm
                and objective_bottom - sample_top < cfg.sample_clearance_mm):
            self.refuse(device_id, command, "swept objective/sample clearance is insufficient")
        manip = self.devices.devices.get(cfg.manipulator)
        if not isinstance(manip, StampManipulator):
            return
        positions = manip.get_positions()
        origin = manip.geometry_state().get("origin_world_xy_mm", cfg.manipulator_origin_xy)
        if len(origin) != 2 or not all(math.isfinite(v) for v in origin):
            self.refuse(device_id, command, "stamp world origin must be finite")
        planned = stamp_target or positions
        if not all(math.isfinite(p[a]) for p in (positions, planned) for a in ("x", "y", "z")):
            self.refuse(device_id, command, "stamp geometry reading must be finite")
        a = (origin[0]+positions["x"], origin[1]+positions["y"])
        b = (origin[0]+planned["x"], origin[1]+planned["y"])
        stamp_top = max(positions["z"], planned["z"]) + cfg.stamp_top_above_holder_mm
        if (_segment_distance(objective_xy, a, b)
                <= cfg.stamp_radius_mm + cfg.objective_radius_mm
                and objective_bottom - stamp_top < cfg.stamp_clearance_mm):
            self.refuse(device_id, command, "swept objective/stamp clearance is insufficient")
        stage_lateral = (current.x != target.x or current.y != target.y
                         or current.rotation != target.rotation)
        stamp_lateral = a != b
        if stage_lateral or stamp_lateral:
            distance = (_segment_distance(a, start, end) if stage_lateral
                        else _segment_distance(start, a, b))
            if (distance <= radius + cfg.stamp_radius_mm
                    and min(positions["z"], planned["z"]) - max(current.z, target.z)
                    < cfg.low_stamp_gap_mm):
                self.refuse(device_id, command, "swept stage/stamp envelope is too low")

    def _stamp_low(self, il: Any, stage_z: float | None = None) -> bool:
        """The stamp hangs over the stage with less than `below_z` between the
        holder's z and the stage's z (the stage's z raises the chip). `stage_z`:
        the stage height to check against (a planned move), default its reading."""
        m = self.devices.devices.get(il.manipulator)
        if not isinstance(m, StampManipulator):
            return False  # this lab has no stacking manipulator
        p = m.get_positions()
        if stage_z is None:
            st = self.devices.devices.get(il.stage)
            stage_z = st.get_pose().z if isinstance(st, MotionStage) else 0.0
        return all(_in(p[a], r) for a, r in il.over.items()) and p["z"] - stage_z < il.below_z

    def manipulator_z_floor(self, manipulator: str) -> float | None:
        """The lowest z `manipulator` may be commanded to right now: its z limit,
        raised by however far the stage under its head is lifted beyond its
        focusing range (the same rule `move_axis_to` refuses by). None when the
        device has no limits. Flows clamp their own touch and press searches to
        it, so a search that finds no contact stops honestly instead of being
        refused."""
        lim = self.limits.manipulators.get(manipulator)
        if lim is None:
            return None
        lift = self.stage_lift_under(manipulator) - lim.stage_focus_mm
        return lim.z[0] + max(0.0, lift)

    def stage_lift_under(self, manipulator: str) -> float:
        """How far the stage under `manipulator`'s head is raised (0 when the
        head is not over a stage)."""
        m = self.devices.devices.get(manipulator)
        if not isinstance(m, StampManipulator):
            return 0.0
        p = m.get_positions()
        for il in self.limits.stamp_interlocks:
            if il.manipulator == manipulator and all(_in(p[a], r) for a, r in il.over.items()):
                st = self.devices.devices.get(il.stage)
                return max(0.0, st.get_pose().z) if isinstance(st, MotionStage) else 0.0
        return 0.0

    def check_stamp_interlocks(self, device_id: str, command: str,
                               arm_target: Pose | None = None,
                               lowers_only: bool = False,
                               stage_z: float | None = None) -> None:
        """`lowers_only`: a stage move that keeps x/y and only lowers z, which
        opens the gap to the stamp, so a blocked stage may still make it.
        `stage_z`: the highest z a stage move passes through."""
        for il in self.limits.stamp_interlocks:
            if arm_target is not None:
                if il.arm != device_id or not il.region:
                    continue
                inside = all(_in(getattr(arm_target, a), r) for a, r in il.region.items())
                if inside and arm_target.z < il.arm_below_z and self._stamp_low(il):
                    self.refuse(device_id, command, f"interlock {il.name}: the stamp is "
                                f"over the stage", pose=arm_target.model_dump())
            elif device_id in il.blocks and not lowers_only and self._stamp_low(
                    il, stage_z if device_id == il.stage else None):
                self.refuse(device_id, command, f"interlock {il.name}: the stamp is low over "
                            f"the stage")

    def watchdog_tick(self) -> None:
        """Called periodically by the job service. Enforces time-based limits."""
        now = self.clock.now()
        for hid, since in list(self._heater_on_since.items()):
            lim = self.limits.heaters[hid]
            heater = self.devices.get(hid)
            assert isinstance(heater, Heater)
            if now - since > lim.max_on_seconds:
                heater.off()
                self._heater_on_since.pop(hid, None)
                self.refuse(hid, "watchdog", f"on longer than {lim.max_on_seconds}s; forced off")
            t = heater.read_temperature()
            if t > lim.max_celsius + lim.overshoot_margin:
                heater.off()
                self._heater_on_since.pop(hid, None)
                self.refuse(hid, "watchdog", f"reading {t:.1f}C above limit; forced off")

    def check_reactor_interlocks(self, device_id: str, command: str, *,
                                 pose: Pose | None = None,
                                 temperature_c: float | None = None,
                                 flow_sccm: float | None = None) -> None:
        for il in self.limits.reactor_interlocks:
            if device_id not in (il.stage, il.insertion_stage, il.arm, il.heater, il.gas_flow):
                continue
            if il.stage not in self.devices.devices:
                continue  # the optional virtual module is not enabled
            if device_id == il.gas_flow and flow_sccm is not None and flow_sccm <= 0:
                continue  # closing flow must not depend on another sensor being online
            if device_id == il.arm and pose is not None and pose.z >= il.arm_clear_z:
                continue  # allow the arm to lift clear when process readings are unavailable
            stage = self._typed(il.stage, MotionStage)
            p = stage.get_pose()
            insertion = self._typed(il.insertion_stage, MotionStage) if il.insertion_stage else stage
            ip = insertion.get_pose()
            if device_id in (il.stage, il.insertion_stage) and pose is not None:
                arm = self._typed(il.arm, Arm)
                arm_pose = arm.get_pose()
                over = all(_in(getattr(arm_pose, a), r)
                           for a, r in il.loading_region.items())
                current = ip if device_id == il.insertion_stage else p
                if over and arm_pose.z < il.arm_clear_z and pose != current:
                    self.refuse(device_id, command,
                                f"{il.name}: arm is low over the carrier")
            if ((device_id == il.heater and temperature_c is not None)
                    or (device_id == il.gas_flow and flow_sccm is not None and flow_sccm > 0)):
                if not _in(ip.y, il.process_y):
                    self.refuse(device_id, command, f"{il.name}: insert the carrier first")
                continue
            heater = self._typed(il.heater, Heater)
            gas = self._typed(il.gas_flow, GasFlowController)
            access = (heater.read_temperature() <= il.access_max_celsius
                      and gas.read_flow() <= il.access_max_flow_sccm)
            if device_id == il.insertion_stage and pose is not None and pose != ip and not access:
                self.refuse(device_id, command, f"{il.name}: hot/flowing insertion is locked")
            if device_id == il.stage and pose is not None and not access:
                inserted = _in(ip.y, il.process_y)
                if not (inserted and (il.insertion_stage or _in(pose.y, il.process_y))
                        and math.hypot(pose.x - p.x, pose.y - p.y) <= il.max_fine_step_mm):
                    self.refuse(device_id, command,
                                f"{il.name}: hot/flowing carrier permits only fine motion")
            if device_id == il.arm and pose is not None and pose.z < il.arm_clear_z:
                over = all(_in(getattr(pose, a), r) for a, r in il.loading_region.items())
                at_load = (math.hypot(p.x, p.y) <= il.loading_tolerance_mm
                           and abs(ip.y) <= il.loading_tolerance_mm)
                if over and (not at_load or not access):
                    self.refuse(device_id, command,
                                f"{il.name}: carrier loading position is not accessible")


class _Guarded:
    def __init__(self, gate: SafetyGate, dev: Any) -> None:
        self._gate = gate
        self._dev = dev
        self.device_id: str = dev.device_id

    @property
    def motion_active(self) -> bool:
        """Current driver motion state, exposed without bypassing the gate."""
        return self._dev.motion_active

    def safe_state(self) -> None:
        with self._gate.command(self.device_id, "safe_state"):
            self._dev.safe_state()


class GuardedArm(_Guarded):
    _dev: Arm

    def move_to(self, pose: Pose, speed: float | None = None) -> None:
        g, did = self._gate, self.device_id
        lim = g.limits.arms.get(did)
        if lim is None:
            g.refuse(did, "move_to", "no limits configured", pose=pose.model_dump())
        assert lim is not None
        if not self._dev.is_homed():
            g.refuse(did, "move_to", "position unknown (not homed since a reset); home first",
                     pose=pose.model_dump())
        for axis in ("x", "y", "z", "rotation"):
            v = getattr(pose, axis)
            if not _in(v, getattr(lim, axis)):
                g.refuse(did, "move_to", f"{axis}={v} outside {getattr(lim, axis)}",
                         pose=pose.model_dump())
        if lim.reach_mm is not None and math.hypot(pose.x, pose.y) > lim.reach_mm:
            g.refuse(did, "move_to", f"({pose.x:.0f}, {pose.y:.0f}) beyond the arm's reach "
                     f"of {lim.reach_mm:.0f} mm", pose=pose.model_dump())
        if speed is not None and speed > lim.max_speed:
            g.refuse(did, "move_to", f"speed {speed} > {lim.max_speed}", speed=speed)
        for fl in lim.floors:
            if pose.z < fl.min_z and all(_in(getattr(pose, a), r) for a, r in fl.region.items()):
                g.refuse(did, "move_to", f"z={pose.z} below the floor {fl.min_z} of "
                         f"{fl.name}", pose=pose.model_dump())
        g.check_stamp_interlocks(did, "move_to", arm_target=pose)
        g.check_reactor_interlocks(did, "move_to", pose=pose)
        with g.command(did, "move_to", pose=pose.model_dump(), speed=speed):
            self._dev.move_to(pose, speed)

    def home(self) -> None:
        with self._gate.command(self.device_id, "home"):
            self._dev.home()

    def set_vacuum(self, on: bool) -> None:
        with self._gate.command(self.device_id, "set_vacuum", on=on):
            self._dev.set_vacuum(on)

    def get_pose(self) -> Pose:
        return self._dev.get_pose()

    def vacuum_on(self) -> bool:
        return self._dev.vacuum_on()

    def is_homed(self) -> bool:
        return self._dev.is_homed()

    def read_vacuum_pressure(self) -> float | None:
        return self._dev.read_vacuum_pressure()


class GuardedStage(_Guarded):
    _dev: MotionStage

    def _check_move(self, pose: Pose, command: str,
                    fine: tuple[float, float] | None = None) -> None:
        g, did = self._gate, self.device_id
        lim = g.limits.stages.get(did)
        if lim is None:
            g.refuse(did, command, "no limits configured", pose=pose.model_dump())
        assert lim is not None
        g.check_calibration(did, command, lim)
        for axis in ("x", "y", "z"):
            v = getattr(pose, axis)
            if not _in(v, getattr(lim, axis)):
                g.refuse(did, command, f"{axis}={v} outside {getattr(lim, axis)}",
                         pose=pose.model_dump())
        now = self._dev.get_pose()
        geometry_stage = g.limits.geometry is not None and did == g.limits.geometry.stage
        if (not math.isfinite(pose.rotation)
                or (lim.rotation is None and (geometry_stage or pose.rotation != 0
                                               or command == "rotate_to"))
                or (lim.rotation is not None and not _in(pose.rotation, lim.rotation))):
            g.refuse(did, command, "rotation outside configured stage limits",
                     pose=pose.model_dump())
        current_fine = self._dev.get_fine_position()
        planned_fine = current_fine if fine is None else fine
        for axis, value in zip(("fine_x", "fine_y"), planned_fine, strict=True):
            bounds = getattr(lim, axis)
            if (not math.isfinite(value) or (bounds is None and (geometry_stage or value != 0
                                                               or fine is not None))
                    or (bounds is not None and not _in(value, bounds))):
                g.refuse(did, command, f"{axis} outside configured fine limits", value=value)
        g.check_interlocks(did, command)
        g.check_reactor_interlocks(did, command, pose=pose)
        lowers = (pose.x == now.x and pose.y == now.y and pose.rotation == now.rotation
                  and planned_fine == current_fine and pose.z <= now.z)
        g.check_stamp_interlocks(did, command, lowers_only=lowers,
                                 stage_z=max(now.z, pose.z))
        composed = pose.offset(dx=planned_fine[0], dy=planned_fine[1])
        g.check_swept_geometry(did, command, sample_target=composed)

    def move_to(self, pose: Pose) -> None:
        g, did = self._gate, self.device_id
        self._check_move(pose, "move_to")
        with g.command(did, "move_to", pose=pose.model_dump()):
            self._dev.move_to(pose)

    def move_fine_to(self, x: float, y: float) -> None:
        self._check_move(self._dev.get_pose(), "move_fine_to", fine=(x, y))
        with self._gate.command(self.device_id, "move_fine_to", x=x, y=y):
            self._dev.move_fine_to(x, y)

    def rotate_to(self, angle_deg: float) -> None:
        pose = self._dev.get_pose().model_copy(update={"rotation": angle_deg})
        self._check_move(pose, "rotate_to")
        with self._gate.command(self.device_id, "rotate_to", angle_deg=angle_deg):
            self._dev.rotate_to(angle_deg)

    def get_pose(self) -> Pose:
        return self._dev.get_pose()

    def get_fine_position(self) -> tuple[float, float]:
        return self._dev.get_fine_position()

    def get_sample_pose(self) -> Pose:
        return self._dev.get_sample_pose()

    def kinematics(self) -> dict[str, Any]:
        return self._dev.kinematics()


class GuardedManipulator(_Guarded):
    _dev: StampManipulator

    def axis_motion_duration_bound(self, axis: str, start: float, target: float,
                                   speed: float | None = None) -> float | None:
        if (axis not in StampManipulator.AXES
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in (start, target))
                or (speed is not None and (not math.isfinite(speed) or speed <= 0))):
            return None
        bound = self._dev.axis_motion_duration_bound(axis, start, target, speed)
        return bound if (not isinstance(bound, bool) and isinstance(bound, (int, float))
                         and math.isfinite(bound) and bound >= 0) else None

    def move_axis_to(self, axis: str, value: float, speed: float | None = None) -> None:
        g, did = self._gate, self.device_id
        lim = g.limits.manipulators.get(did)
        if lim is None:
            g.refuse(did, "move_axis_to", "no limits configured", axis=axis, value=value)
        assert lim is not None
        g.check_calibration(did, "move_axis_to", lim)
        if axis not in StampManipulator.AXES:
            g.refuse(did, "move_axis_to", f"unknown axis {axis!r}", axis=axis, value=value)
        if not _in(value, getattr(lim, axis)):
            g.refuse(did, "move_axis_to", f"{axis}={value} outside {getattr(lim, axis)}",
                     axis=axis, value=value)
        if speed is not None and (not math.isfinite(speed) or speed <= 0 or speed > lim.max_speed):
            g.refuse(did, "move_axis_to", f"speed {speed} > {lim.max_speed}", axis=axis,
                     value=value, speed=speed)
        if axis == "z":
            # over the stage the press depth is relative to the stage's height:
            # a stage raised beyond its focusing range brings the chip closer,
            # so the z floor rises with it
            lift = g.stage_lift_under(did) - lim.stage_focus_mm
            if lift > 0 and value - lift < lim.z[0]:
                g.refuse(did, "move_axis_to", f"z={value} is below the floor {lim.z[0]} "
                         f"raised by {lift:.3f} (the stage is raised beyond its focusing "
                         f"range)", axis=axis, value=value)
        g.check_interlocks(did, "move_axis_to")
        start = dict(self._dev.get_positions())
        planned = {**start, axis: value}
        g.check_swept_geometry(did, "move_axis_to", stamp_target=planned)
        duration = self.axis_motion_duration_bound(axis, start[axis], value, speed)
        with g.command(did, "move_axis_to", axis=axis, value=value, speed=speed,
                       from_pose=start, duration_bound_s=duration):
            self._dev.move_axis_to(axis, value, speed)

    def get_positions(self) -> dict[str, float]:
        return self._dev.get_positions()


class GuardedHeater(_Guarded):
    _dev: Heater

    def set_target(self, celsius: float) -> None:
        g, did = self._gate, self.device_id
        lim = g.limits.heaters.get(did)
        if lim is None:
            g.refuse(did, "set_target", "no limits configured", celsius=celsius)
        assert lim is not None
        if not math.isfinite(celsius):
            g.refuse(did, "set_target", "temperature must be finite", celsius=str(celsius))
        if lim.min_celsius is not None and celsius < lim.min_celsius:
            g.refuse(did, "set_target", "temperature below minimum", celsius=celsius)
        if celsius > lim.max_celsius:
            g.refuse(did, "set_target", f"{celsius}C > max {lim.max_celsius}C", celsius=celsius)
        # Ramp limit: the setpoint may move at most max_ramp_c_per_min per elapsed
        # minute since the previous setpoint (at least one minute's worth), measured
        # from the current reading the first time.
        now = g.clock.now()
        prev_t, prev_sp = g._heater_last_set.get(did, (now, self._dev.read_temperature()))
        minutes = max((now - prev_t) / 60.0, 1.0)
        allowed = lim.max_ramp_c_per_min * minutes
        if celsius - prev_sp > allowed:
            g.refuse(did, "set_target",
                     f"setpoint jump {celsius - prev_sp:.0f}C exceeds {allowed:.0f}C "
                     f"({lim.max_ramp_c_per_min}C/min)", celsius=celsius)
        g.check_reactor_interlocks(did, "set_target", temperature_c=celsius)
        with g.command(did, "set_target", celsius=celsius):
            self._dev.set_target(celsius)
        g._heater_last_set[did] = (now, celsius)
        g._heater_on_since.setdefault(did, now)

    def off(self) -> None:
        g = self._gate
        with g.command(self.device_id, "off"):
            self._dev.off()
        g._heater_on_since.pop(self.device_id, None)
        g._heater_last_set.pop(self.device_id, None)

    def safe_state(self) -> None:
        self.off()

    def read_temperature(self) -> float:
        return self._dev.read_temperature()

    def get_target(self) -> float | None:
        return self._dev.get_target()


class GuardedCamera(_Guarded):
    _dev: Camera

    def capture(self) -> Frame:
        with self._gate.command(self.device_id, "capture"):
            return self._dev.capture()


class GuardedGasFlow(_Guarded):
    _dev: GasFlowController

    def set_flow(self, sccm: float) -> None:
        g, did = self._gate, self.device_id
        lim = g.limits.gas_flows.get(did)
        if lim is None or not math.isfinite(sccm) or not _in(sccm, lim.sccm):
            g.refuse(did, "set_flow", "gas flow outside configured limits",
                     sccm=sccm if math.isfinite(sccm) else str(sccm))
        g.check_reactor_interlocks(did, "set_flow", flow_sccm=sccm)
        with g.command(did, "set_flow", sccm=sccm):
            self._dev.set_flow(sccm)

    def get_target(self) -> float:
        return self._dev.get_target()

    def read_flow(self) -> float:
        return self._dev.read_flow()

    def safe_state(self) -> None:
        with self._gate.command(self.device_id, "safe_state"):
            self._dev.safe_state()


class GuardedMicroscope(_Guarded):
    _dev: Microscope

    @contextmanager
    def motion_preview_off(self):
        """Display only: no motion-preview frames while this block runs.

        The flake search shows only the photos it takes (#203); the UI's
        moving-scene preview is not drawn or saved meanwhile.
        """
        dev = self._dev
        dev.motion_preview_suppressed = getattr(dev, "motion_preview_suppressed", 0) + 1
        try:
            yield
        finally:
            dev.motion_preview_suppressed -= 1

    def focus_transition_duration_bound(self, start_z: float, target_z: float) -> float | None:
        bound = self._dev.focus_transition_duration_bound(start_z, target_z)
        return bound if isinstance(bound, (int, float)) and math.isfinite(bound) and bound >= 0 else None

    def focus_axis_calibration(self, objective: int) -> dict[str, Any] | None:
        """Return only finite, provenance-bound same-objective declarations."""
        if isinstance(objective, bool) or not isinstance(objective, int) or objective <= 0:
            return None
        calibration = self._dev.focus_axis_calibration(objective)
        if not isinstance(calibration, dict):
            return None
        if (isinstance(calibration.get("objective"), bool)
                or not isinstance(calibration.get("objective"), int)
                or calibration.get("objective") != objective
                or calibration.get("comparison_scope") != "same_objective_only"
                or not isinstance(calibration.get("calibration_id"), str)
                or calibration["calibration_id"].strip().lower() in ("", "unknown")):
            return None
        source, simulated = calibration.get("source"), calibration.get("simulation_only")
        if source == "synthetic":
            if simulated is not True or not self._dev.is_simulated:
                return None
        elif source != "measured" or simulated is not False:
            return None
        for field in ("scale_mm_per_focus_mm", "mapping_error_mm", "readback_error_mm"):
            value = calibration.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (field == "scale_mm_per_focus_mm" and value == 0)):
                return None
        for field in ("lateral_focus_slope_norm_bound_mm_per_mm", "apex_drift_clip_per_axis_mm",
                      "source_refraction_projection_norm_bound_mm_per_mm"):
            bound = calibration.get(field)
            if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))
                                      or not math.isfinite(bound) or bound < 0):
                return None
        return deepcopy(calibration)

    def capture(self, *, purpose: str = "analysis_capture") -> Frame:
        if purpose not in ("analysis_capture", "focus_capture", "neighbourhood_capture"):
            raise ValueError("Unsupported microscope capture purpose")
        params = {} if purpose == "analysis_capture" else {"purpose": purpose}
        with self._gate.command(self.device_id, "capture", **params):
            started = self._gate.clock.now()
            frame = (self._dev.capture_for_focus() if purpose == "focus_capture"
                     else self._dev.capture_for_neighbourhood() if purpose == "neighbourhood_capture"
                     else self._dev.capture())
            finished = self._gate.clock.now()
            timestamp = getattr(frame, "timestamp", None)

            def finite(value: Any) -> bool:
                try:
                    return not isinstance(value, bool) and math.isfinite(value)
                except (TypeError, ValueError):
                    return False

            if not all(finite(value) for value in (started, timestamp, finished)):
                reason = "invalid_capture_timestamp"
            elif finished < started:
                reason = "capture_clock_reversed"
            elif timestamp < started:
                reason = "stale_capture_timestamp"
            elif timestamp > finished:
                reason = "future_capture_timestamp"
            else:
                return frame
            self._gate.sink("device.observation_rejected", {
                "device": self.device_id, "command": "capture", "reason": reason,
                "capture_started_at": float(started) if finite(started) else None,
                "capture_finished_at": float(finished) if finite(finished) else None,
                "frame_timestamp": float(timestamp) if finite(timestamp) else None,
                "clock_basis": getattr(self._gate.clock, "event_clock_basis", None),
                "clock_domain": getattr(self._gate.clock, "event_clock_domain", None),
            })
            raise NeedsHumanError("Microscope capture has a stale or invalid exposure timestamp; "
                                  "the adapter must supply a current exposure in the execution Clock basis.",
                                  device=self.device_id, reason=reason, observation_rejected=True)

    def set_focus(self, z: float) -> None:
        g, did = self._gate, self.device_id
        lim = g.limits.microscopes.get(did)
        if lim is None or not _in(z, lim.focus):
            g.refuse(did, "set_focus", f"focus {z} outside limits", z=z)
        assert lim is not None
        g.check_calibration(did, "set_focus", lim)
        g.check_swept_geometry(did, "set_focus", focus_target=z)
        with g.command(did, "set_focus", z=z):
            self._dev.set_focus(z)

    def get_focus(self) -> float:
        return self._dev.get_focus()

    def get_objective(self) -> int | None:
        return self._dev.get_objective()

    def set_objective(self, magnification: int) -> None:
        g, did = self._gate, self.device_id
        lim = g.limits.microscopes.get(did)
        if lim is None or magnification not in lim.objectives:
            g.refuse(did, "set_objective", f"objective {magnification}x not allowed",
                     magnification=magnification)
        assert lim is not None
        g.check_calibration(did, "set_objective", lim)
        g.check_swept_geometry(did, "set_objective")
        with g.command(did, "set_objective", magnification=magnification):
            self._dev.set_objective(magnification)


class GuardedSwitch(_Guarded):
    _dev: Switch

    def set(self, on: bool) -> None:
        with self._gate.command(self.device_id, "set", on=on):
            self._dev.set(on)

    def get(self) -> bool:
        return self._dev.get()


class GuardedTape(_Guarded):
    _dev: TapeMechanism

    def _limits(self, command: str, **params: Any) -> Any:
        lim = self._gate.limits.tapes.get(self.device_id)
        if lim is None:
            self._gate.refuse(self.device_id, command, "no limits configured", **params)
        return lim

    def advance(self, length_mm: float) -> None:
        g, did = self._gate, self.device_id
        lim = self._limits("advance", length_mm=length_mm)
        if not 0 < length_mm <= lim.max_advance_mm:
            g.refuse(did, "advance", f"length {length_mm} outside (0, {lim.max_advance_mm}]",
                     length_mm=length_mm)
        g.check_interlocks(did, "advance")
        with g.command(did, "advance", length_mm=length_mm):
            self._dev.advance(length_mm)

    def apply(self) -> None:
        self._limits("apply")
        self._gate.check_interlocks(self.device_id, "apply")
        with self._gate.command(self.device_id, "apply"):
            self._dev.apply()

    def peel(self, speed: float, angle_deg: float | None = None) -> None:
        g, did = self._gate, self.device_id
        lim = self._limits("peel", speed=speed, angle_deg=angle_deg)
        if not _in(speed, lim.peel_speed):
            g.refuse(did, "peel", f"speed {speed} outside {lim.peel_speed}", speed=speed)
        if angle_deg is not None and not _in(angle_deg, lim.peel_angle_deg):
            g.refuse(did, "peel", f"angle {angle_deg} outside {lim.peel_angle_deg}",
                     angle_deg=angle_deg)
        g.check_interlocks(did, "peel")
        with g.command(did, "peel", speed=speed, angle_deg=angle_deg):
            self._dev.peel(speed, angle_deg)

    def is_applied(self) -> bool:
        return self._dev.is_applied()

    def remaining_mm(self) -> float | None:
        return self._dev.remaining_mm()
