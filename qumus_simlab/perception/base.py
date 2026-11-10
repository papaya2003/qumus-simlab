"""Perception interfaces used by Step.verify() (spec 4.4).

CONTRACT. Flow code asks "where is chip X?" through an ObjectLocator and gets
measurements in millimetres in an arm's frame; microscope frames go through a
MicroscopeAnalyzer (focus sharpness, chip corner, flakes). Implementations:
  - testbench: sim ground truth + noise (YOLO weights are not available yet)
  - stage 3:   v1 YOLO OBB + QR homography (agents/lab_manager_agent/tools/cv/),
               v1 segmenter + corner detection (processing_agent/segmenter/,
               molecule_corner_detection.py)
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import Frame


class Detection(BaseModel):
    label: str  # "chip", "glass_slide", "tape_carrier", ...
    pose: Pose  # centre and rotation, mm/deg in the requested frame
    confidence: float
    object_id: str | None = None  # set when the object carries a readable QR code
    #: extra findings, e.g. {"material_visible": True} for an exfoliated chip
    props: dict[str, Any] = Field(default_factory=dict)


class ObjectLocator(Protocol):
    def locate(self, frame: Frame, *, region: str, frame_of: str) -> list[Detection]:
        """Detect objects in `frame` inside named bench region `region`
        (e.g. "transfer_stage", "middle_storage", "left_chip_3"), returned in `frame_of`'s
        coordinate frame (e.g. "arm_right")."""
        ...


class CornerDetection(BaseModel):
    """A chip corner seen by the microscope, offset from the view centre."""

    corner: str  # "bl" | "br" | "tl" | "tr"
    dx_mm: float  # world/stage axes; move the stage by (-dx, -dy) to centre it
    dy_mm: float
    confidence: float


class FlakeDetection(BaseModel):
    """One flake in a microscope frame, offset from the view centre."""

    dx_mm: float
    dy_mm: float
    area_um2: float
    material: str
    thickness_label: str  # Gr: monolayer/bilayer/trilayer/fewlayer/manylayer
    confidence: float
    meta: dict[str, Any] = Field(default_factory=dict)


class ContactDetection(BaseModel):
    """Where the stacking stamp touches the chip (v1 touching_detection: the
    Newton-ring contact front), offset from the view centre."""

    dx_mm: float
    dy_mm: float
    radius_mm: float
    confidence: float
    #: False when the contact front is broken up (wrinkles, a tear running
    #: through the rings): the stamp's PC film is torn
    film_intact: bool | None = True
    #: Pixel evidence can leave film integrity unknown even with a visible front.
    uncertainty_mm: float | None = None
    image: str | None = None
    observed_at: float | None = None
    evidence_source: str = "unspecified"


class FringeDetection(BaseModel):
    """Newton rings of the thin air gap between the stamp's dome and the chip,
    seen just before contact: where their centre is (from the view centre) and
    the gap their order gives (white-light fringes fade after a few um)."""

    dx_mm: float
    dy_mm: float
    gap_mm: float
    rings: int
    confidence: float


class MicroscopeAnalyzer(Protocol):
    def plane_sharpness(self, frame: Frame, plane: str) -> float | None:
        """Quality of this captured focal plane; unknown adapters return None, never another plane's maximum."""
        ...

    def sharpness(self, frame: Frame) -> float:
        """Focus metric of the frame (v1 'blur' value; higher = sharper)."""
        ...

    def find_corner(self, frame: Frame, corner: str = "bl") -> CornerDetection | None: ...

    def chip_in_view(self, frame: Frame) -> bool:
        """Is a chip (substrate) under the objective, not the bare stage?"""
        ...

    def find_flakes(self, frame: Frame, material: str) -> list[FlakeDetection]: ...

    def contact(self, frame: Frame) -> ContactDetection | None:
        """The stamp's contact area with the chip, None if it does not touch."""
        ...

    def find_stamp_flakes(self, frame: Frame) -> list[FlakeDetection]:
        """Flakes carried by a stamp hanging over the view (seen through the PDMS)."""
        ...

    def fringes(self, frame: Frame) -> FringeDetection | None:
        """Newton rings of a dome a few um above the chip, None if none are seen
        (no stamp near, already touching, or the chip plane out of focus)."""
        ...
