"""Device interfaces (spec 12.1 devices/).

CONTRACT. The testbench (devices/sim) and the real lab drivers (devices/real,
ported from v1 common/hardware in stage 3) both implement these classes, so the
same flow code runs on both.

Rules:
- Flow code never holds a raw device. It gets guarded devices from the
  SafetyGate (qumus_simlab.safety.gate), which wrap these same interfaces.
- Methods raise qumus_simlab.core.errors.* on failure; they never return error strings.
- Units: millimetres, degrees, degrees Celsius, seconds.
- Every device has a stable `device_id` that matches config/limits.yaml.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qumus_simlab.core.types import Pose


class Device(ABC):  # noqa: B024 - base for the abstract device kinds below
    kind: str = "device"
    geometry_source: str = "unknown"
    is_simulated: bool = False
    motion_observer: Callable[[], None] | None = None
    motion_active: bool = False

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    def connect(self) -> None:  # noqa: B027 - optional hook
        """Open the connection. Each device connects on its own (spec 5.4)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Close the connection."""

    def is_connected(self) -> bool:
        return True

    def safe_state(self) -> None:  # noqa: B027 - optional hook
        """Put this device in its safe state (heater off, vacuum off, arm stopped)."""


class Arm(Device):
    """G-code robot arm with a vacuum nozzle (v1: left = exfoliation, right = search/stacking)."""

    kind = "arm"

    @abstractmethod
    def move_to(self, pose: Pose, speed: float | None = None) -> None:
        """Move the nozzle to `pose` (x, y, z in mm; rotation = nozzle angle in degrees)."""

    @abstractmethod
    def get_pose(self) -> Pose: ...

    @abstractmethod
    def home(self) -> None: ...

    @abstractmethod
    def set_vacuum(self, on: bool) -> None: ...

    @abstractmethod
    def vacuum_on(self) -> bool: ...

    def read_vacuum_pressure(self) -> float | None:
        """Vacuum pressure in kPa if the arm has a sensor, else None (spec 15.3 Q3)."""
        return None

    def is_homed(self) -> bool:
        """False after a controller reset or power loss: the arm does not know
        where it is until home() runs (a G-code arm loses its position)."""
        return True


class MotionStage(Device):
    """One stage with coarse XYZ, optional fine XY, and in-plane rotation.

    Fine translations are in the same bench-aligned axes as coarse translation.
    Rotation is about the translated stage pivot; it does not rotate the fine
    axes. ``get_pose`` is the coarse readback, including absolute rotation.
    ``get_sample_pose`` composes the fine translation without changing readbacks.
    """

    kind = "stage"

    @abstractmethod
    def move_to(self, pose: Pose) -> None: ...

    @abstractmethod
    def get_pose(self) -> Pose: ...

    def move_fine_to(self, x: float, y: float) -> None:
        from qumus_simlab.core.errors import PreconditionFailed

        raise PreconditionFailed(f"{self.device_id} has no fine translation capability")

    def get_fine_position(self) -> tuple[float, float]:
        """Fine XY in mm; a stage without fine axes has zero fine offset."""
        return (0.0, 0.0)

    def rotate_to(self, angle_deg: float) -> None:
        self.move_to(self.get_pose().model_copy(update={"rotation": angle_deg}))

    def get_sample_pose(self) -> Pose:
        pose = self.get_pose()
        x, y = self.get_fine_position()
        return pose.offset(dx=x, dy=y)

    def kinematics(self) -> dict[str, Any]:
        return {"source": self.geometry_source, "coarse_pose": self.get_pose().model_dump(),
                "fine_xy_mm": list(self.get_fine_position()),
                "sample_pose": self.get_sample_pose().model_dump()}

    def home(self) -> None:
        self.move_to(Pose(x=0, y=0, z=0))


@dataclass
class Frame:
    """One captured image plus metadata.

    `path` points to the saved image file (under the run directory) when the
    image was written to disk; `data` holds encoded bytes (PNG/JPEG) otherwise.
    `meta` is free-form (exposure, magnification, sim ground truth, ...).
    A device may write the file only on demand: then `save` is set and
    `saved_path()` writes it once (autofocus sweeps need no files).

    A current microscope capture must timestamp its exposure in the execution
    Clock's basis, between capture-call start and return. Adapters must convert
    camera timestamps explicitly into that configured basis, or fail if it is
    unknown; numeric magnitude never establishes comparability. A buffered old
    exposure cannot be returned as current evidence. Equal timestamps are valid
    when the shared simulation Clock has not advanced. Preview and historical
    Frames retain their original times and cannot establish current sensing.
    """

    device_id: str
    timestamp: float
    path: str | None = None
    data: bytes | None = None
    width: int = 0
    height: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    save: Callable[[], str] | None = field(default=None, repr=False, compare=False)

    def saved_path(self) -> str | None:
        """Path of the image file, writing it first if the device saves on demand."""
        if self.path is None and self.save is not None:
            self.path = self.save()
        return self.path


class Camera(Device):
    """Desktop camera looking down at the bench (v1: left/right YOLO cameras)."""

    kind = "camera"

    @abstractmethod
    def capture(self) -> Frame: ...

    def preview(self) -> Frame | None:
        """A picture for the UI to show, not a command: no limits, no log, may be
        a live preview or the last frame. None = this camera has no preview."""
        return None


class Microscope(Device):
    """Optical microscope camera with focus and objective (stage is a separate MotionStage)."""

    kind = "microscope"

    @abstractmethod
    def capture(self) -> Frame:
        """Return a new exposure in the configured execution Clock basis.

        SafetyGate checks that its finite timestamp falls inside this call.
        The adapter owns any explicit conversion from a hardware camera clock.
        """
        ...

    def capture_for_focus(self) -> Frame:
        """A fresh focus exposure; adapters may use a calibrated lower raster.

        The default keeps the ordinary exposure. This is still an observed,
        timestamped camera command, never a display preview or cached image.
        """
        return self.capture()

    def capture_for_neighbourhood(self) -> Frame:
        """A fresh wide-field exposure; adapters may use their full sensor.

        The caller still checks the actual field and detected neighbours. The
        default retains the ordinary camera exposure and its calibration.
        """
        return self.capture()

    def focus_transition_duration_bound(self, start_z: float, target_z: float) -> float | None:
        """Declared upper duration in execution-Clock seconds; unknown by default.

        Adapters must provide a justified bound before paired-plane transfer
        control can reserve optical motion time. This is not a focus oracle.
        """
        return None

    def focus_axis_calibration(self, objective: int) -> dict[str, Any] | None:
        """Declared axial scale and errors for comparisons at one objective.

        Unknown by default. This is separate from image/XY calibration and
        supplies no cross-objective parfocal relationship. Adapters declare
        provenance, objective, mm per focus mm, mapping and readback errors;
        the gate validates that declaration before evidence can use it. An
        optional lateral_focus_slope_norm_bound_mm_per_mm declares a static
        lateral-to-axial mapping bound; absent/None means unknown, never zero.
        """
        return None

    @abstractmethod
    def set_focus(self, z: float) -> None: ...

    @abstractmethod
    def get_focus(self) -> float: ...

    def geometry_state(self) -> dict[str, Any]:
        """Non-command geometry readback; absent geometry is explicitly unknown."""
        return {}

    def get_objective(self) -> int | None:
        """Objective in use (magnification), if the turret can report it."""
        return None

    def preview(self) -> Frame | None:
        """A picture for the UI to show, not a command (see Camera.preview)."""
        return None

    @abstractmethod
    def set_objective(self, magnification: int) -> None: ...


class Heater(Device):
    """Temperature-controlled heater (stamp heater, exfoliation heater)."""

    kind = "heater"

    @abstractmethod
    def set_target(self, celsius: float) -> None: ...

    @abstractmethod
    def get_target(self) -> float | None:
        """Current setpoint, None when off."""

    @abstractmethod
    def read_temperature(self) -> float: ...

    @abstractmethod
    def off(self) -> None: ...

    def safe_state(self) -> None:
        self.off()


class Switch(Device):
    """On/off actuator: stage vacuum, relay, fan, tape mechanism solenoid."""

    kind = "switch"

    @abstractmethod
    def set(self, on: bool) -> None: ...

    @abstractmethod
    def get(self) -> bool: ...


class GasFlowController(Device):
    """Gas-flow setpoint and measured feedback, in standard cm3/min (sccm)."""

    kind = "gas_flow"

    @abstractmethod
    def set_flow(self, sccm: float) -> None: ...

    @abstractmethod
    def get_target(self) -> float: ...

    @abstractmethod
    def read_flow(self) -> float: ...


class TapeMechanism(Device):
    """Exfoliator: white tape wound on two cylinders over the exfoliation stage.
    The lower cylinder is fixed; the upper one rides a gantry in the vertical
    x-z plane (v1: a 3D-printer head, printer_controller.py, plus reel motors).

    advance() turns the cylinders to bring fresh tape over the stage (v1
    AtomTapeRelease), apply() moves the upper cylinder right, down, left, down
    so the tape lies on whatever is on the stage, peel() lifts it off at
    `speed` (v1 G-code F value) and `angle_deg` between tape and stage (v1 had
    a fixed arc; None = the mechanism's default angle).
    """

    kind = "tape"

    @abstractmethod
    def advance(self, length_mm: float) -> None: ...

    @abstractmethod
    def apply(self) -> None: ...

    @abstractmethod
    def peel(self, speed: float, angle_deg: float | None = None) -> None: ...

    @abstractmethod
    def is_applied(self) -> bool: ...

    def remaining_mm(self) -> float | None:
        """Unused tape left on the roll, if the mechanism can tell."""
        return None


class StampManipulator(Device):
    """Stamp holder: a pure x/y/z translation stage holding the PDMS/PC stamp
    (a glass slide) by one end. x is the long travel between the load pose
    above the right storage board and over the stage under the objective; y
    aligns; z presses the stamp down to stack. No rotation.

    Axes in mm: "x", "y", "z". The stamp is held by a separate vacuum switch
    (v1 transfer arm vacuum relay).
    """

    kind = "manipulator"
    AXES = ("x", "y", "z")

    @abstractmethod
    def move_axis_to(self, axis: str, value: float, speed: float | None = None) -> None:
        """Absolute move of one axis; `speed` in mm/s (None = configured default)."""

    @abstractmethod
    def get_positions(self) -> dict[str, float]: ...

    def axis_motion_duration_bound(self, axis: str, start: float, target: float,
                                   speed: float | None = None) -> float | None:
        """Declared command duration bound, independent of arrival accuracy.

        Unknown adapters return None. This does not bound an actuator's arrival
        position or its swept path; callers still require readback acceptance.
        """
        return None

    def geometry_state(self) -> dict[str, Any]:
        return {}


class ReactionChamber(Device):
    """High-temperature vdW reaction chamber (spec 9). Stage 2; interface only.

    The hardware does not exist yet, so this interface is provisional.
    """

    kind = "reaction_chamber"

    @abstractmethod
    def set_temperature(self, celsius: float) -> None: ...

    @abstractmethod
    def read_temperature(self) -> float: ...

    @abstractmethod
    def status(self) -> dict[str, Any]: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @abstractmethod
    def open_door(self) -> None: ...

    @abstractmethod
    def close_door(self) -> None: ...


@dataclass
class DeviceBundle:
    """All devices of one lab (sim or real), keyed by device_id."""

    devices: dict[str, Device] = field(default_factory=dict)

    def add(self, device: Device) -> None:
        if device.device_id in self.devices:
            raise ValueError(f"duplicate device_id {device.device_id!r}")
        self.devices[device.device_id] = device

    def get(self, device_id: str) -> Device:
        try:
            return self.devices[device_id]
        except KeyError:
            raise KeyError(f"unknown device {device_id!r}") from None

    def all_safe_state(self) -> list[str]:
        """Put every device in safe state; return ids that failed to comply."""
        failed = []
        for dev in self.devices.values():
            try:
                dev.safe_state()
            except Exception:  # noqa: BLE001 - best effort, report all failures
                failed.append(dev.device_id)
        return failed
