"""Typed safety limits loaded from config/limits.yaml."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from qumus_simlab.config import load_yaml

Range = tuple[float, float]


class ArmFloor(BaseModel):
    """Inside `region` (arm frame x/y) the arm may not go below `min_z`, e.g. the
    nozzle pressing the tape onto a chip may not press harder than this."""

    name: str
    region: dict[str, Range]
    min_z: float


class ArmLimits(BaseModel):
    x: Range
    y: Range
    z: Range
    rotation: Range = (-180.0, 180.0)
    max_speed: float
    reach_mm: float | None = None  # radius around the arm base (its frame origin)
    floors: list[ArmFloor] = Field(default_factory=list)


class GeometryCalibration(BaseModel):
    """An explicit provenance boundary; synthetic numbers never authorize hardware."""

    model_config = ConfigDict(allow_inf_nan=False)
    source: str = "unknown"
    calibration_id: str | None = None
    simulation_only: bool = False


class StageLimits(GeometryCalibration):
    x: Range
    y: Range
    z: Range = (0.0, 0.0)
    rotation: Range | None = None
    fine_x: Range | None = None
    fine_y: Range | None = None


class HeaterLimits(BaseModel):
    min_celsius: float | None = None
    max_celsius: float
    max_ramp_c_per_min: float
    max_on_seconds: float
    overshoot_margin: float = 15.0


class GasFlowLimits(BaseModel):
    sccm: Range


class ReactorInterlock(BaseModel):
    """Chosen virtual operating envelope; not a certified physical interlock."""

    name: str
    stage: str
    insertion_stage: str | None = None
    arm: str
    heater: str
    gas_flow: str
    loading_region: dict[str, Range]
    arm_clear_z: float
    loading_tolerance_mm: float
    process_y: Range
    access_max_celsius: float
    access_max_flow_sccm: float
    max_fine_step_mm: float


class TapeLimits(BaseModel):
    peel_speed: Range
    max_advance_mm: float
    peel_angle_deg: Range = (10.0, 90.0)


class MicroscopeLimits(GeometryCalibration):
    focus: Range
    objectives: list[int]


class ManipulatorLimits(GeometryCalibration):
    x: Range
    y: Range
    z: Range
    max_speed: float  # mm/s
    #: stage z up to this under the head is focusing (chips differ in height),
    #: not a raise into the stamp: the z floor rises only with the stage above it
    stage_focus_mm: float = 0.0


class SweptGeometryLimits(GeometryCalibration):
    """Conservative mechanical envelopes in world mm, separate from optical offsets.

    The carried radius includes mounting offsets and every corner about the stage
    pivot. A full circle bounds every intermediate theta without discrete sampling.
    """

    stage: str = "transfer_stage"
    manipulator: str = "transfer_arm"
    microscope: str = "microscope"
    stage_origin_xy: tuple[float, float]
    manipulator_origin_xy: tuple[float, float]
    objective_xy: tuple[float, float]
    objective_radius_mm: float = Field(ge=0)
    objective_bottom_at_focus_zero_z: float
    sample_clearance_mm: float = Field(gt=0)
    stamp_clearance_mm: float = Field(gt=0)
    stamp_radius_mm: float = Field(ge=0)
    stamp_top_above_holder_mm: float = Field(ge=0)
    sample_height_above_stage_mm: float = Field(ge=0)
    carried_radius_mm: float = Field(gt=0)
    low_stamp_gap_mm: float = Field(gt=0)


class StampInterlock(BaseModel):
    """While `manipulator` hangs the stamp over the stage (every axis in `over`
    inside its range) and lower than `below_z`, the devices in `blocks` may not
    move (a blocked stage may still lower its z: that only opens the gap), and
    `arm` may not go below `arm_below_z` inside `region` (it would hit the stamp)."""

    name: str
    manipulator: str
    over: dict[str, Range]
    stage: str = "transfer_stage"  # the stage under the stamp (its z narrows the gap)
    below_z: float = 1e9
    blocks: list[str] = Field(default_factory=list)
    arm: str | None = None
    region: dict[str, Range] = Field(default_factory=dict)
    arm_below_z: float = 0.0


class Interlock(BaseModel):
    name: str
    arm: str
    region: dict[str, Range]
    below_z: float
    blocks: list[str]


class Limits(BaseModel):
    arms: dict[str, ArmLimits] = Field(default_factory=dict)
    stages: dict[str, StageLimits] = Field(default_factory=dict)
    heaters: dict[str, HeaterLimits] = Field(default_factory=dict)
    gas_flows: dict[str, GasFlowLimits] = Field(default_factory=dict)
    reactor_interlocks: list[ReactorInterlock] = Field(default_factory=list)
    tapes: dict[str, TapeLimits] = Field(default_factory=dict)
    microscopes: dict[str, MicroscopeLimits] = Field(default_factory=dict)
    manipulators: dict[str, ManipulatorLimits] = Field(default_factory=dict)
    interlocks: list[Interlock] = Field(default_factory=list)
    stamp_interlocks: list[StampInterlock] = Field(default_factory=list)
    geometry: SweptGeometryLimits | None = None

    @classmethod
    def load(cls, data: dict[str, Any] | None = None) -> Limits:
        return cls.model_validate(data if data is not None else load_yaml("limits.yaml"))
