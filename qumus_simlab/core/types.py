"""Shared value types used across layers."""

from __future__ import annotations

import math
import uuid

from pydantic import BaseModel, ConfigDict


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Pose(BaseModel):
    """Position in millimetres in the device's own frame, rotation in degrees."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float = 0.0
    rotation: float = 0.0

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, drot: float = 0.0) -> Pose:
        return Pose(x=self.x + dx, y=self.y + dy, z=self.z + dz, rotation=self.rotation + drot)

    def xy_distance(self, other: Pose) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b in degrees, in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d
