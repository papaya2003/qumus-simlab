"""Synthetic growth kinetics for the experimental reactor branch.

Literature motivates temperature/flux sensitivity and nucleation followed by
coalescence. The numerical optimum, dose law and defect law are chosen fixtures,
not a fit to the owner's apparatus. Randomness is sampled once per chip so
extra observations never redraw a reaction or change its future outcome.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GrowthProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = "synthetic_mos2_demo"
    material: str = "MoS2 (synthetic example)"
    optimum_temperature_c: float = Field(820.0, gt=0)
    temperature_width_c: float = Field(65.0, gt=0)
    optimum_flow_sccm: float = Field(200.0, gt=0)
    log_flow_width: float = Field(0.55, gt=0)
    optimum_time_s: float = Field(600.0, gt=0)
    activation_temperature_c: float = Field(350.0, ge=0)
    grid_size: int = Field(10, ge=4, le=40)
    source_urls: tuple[str, ...] = (
        "https://www.beilstein-journals.org/bjnano/articles/10/57",
        "https://www.mdpi.com/1996-1944/11/6/870",
    )


@dataclass
class GrowthCell:
    x: float
    y: float
    incubation_s: float
    rate_factor: float
    flux_bias: float
    dose_s: float = 0.0
    damage: float = 0.0


class GrowthModel:
    """An irreversible spatial dose model with reproducible batch variation.

    A Gaussian temperature response and log-normal flow response provide an
    operating window. Seeded cells nucleate at different accumulated doses,
    then approach full coverage. Excess dose creates an overgrowth penalty;
    overheating adds irreversible damage. The score is dimensionless and has
    no claim to a real growth probability or measured laboratory yield.
    """

    def __init__(self, seed: int, chip_id: str, profile: GrowthProfile | None = None) -> None:
        self.profile = profile or GrowthProfile()
        self.seed, self.chip_id = seed, chip_id
        self.elapsed_s = 0.0
        self.cells: list[GrowthCell] = []
        r = random.Random(f"reactor-growth:{seed}:{chip_id}")
        n = self.profile.grid_size
        for j in range(n):
            for i in range(n):
                self.cells.append(GrowthCell(
                    x=(i + 0.5) / n - 0.5, y=(j + 0.5) / n - 0.5,
                    incubation_s=r.uniform(15.0, 55.0),
                    rate_factor=math.exp(r.gauss(0.0, 0.06)),
                    flux_bias=r.gauss(0.0, 0.025)))

    def advance(self, seconds: float, temperature_c: float, flow_sccm: float,
                *, in_chamber: bool = True) -> None:
        if not all(math.isfinite(v) for v in (seconds, temperature_c, flow_sccm)):
            raise ValueError("growth inputs must be finite")
        if seconds < 0 or flow_sccm < 0:
            raise ValueError("duration and flow must be nonnegative")
        self.elapsed_s += seconds
        p = self.profile
        if not in_chamber or temperature_c < p.activation_temperature_c or seconds == 0:
            return
        thermal = (temperature_c - p.optimum_temperature_c) / p.temperature_width_c
        t_response = math.exp(-0.5 * thermal * thermal)
        heat_damage = max(0.0, thermal - 1.0) ** 2 * seconds / p.optimum_time_s * 0.15
        for cell in self.cells:
            # A fixed spatial flux gradient makes changes of view meaningful.
            local = flow_sccm * max(0.1, 1.0 + 0.22 * cell.y + cell.flux_bias)
            flow_response = (math.exp(-0.5 * (
                math.log(local / p.optimum_flow_sccm) / p.log_flow_width) ** 2)
                if local > 0 else 0.0)
            cell.dose_s += seconds * t_response * flow_response * cell.rate_factor
            cell.damage += heat_damage

    def coverage(self, cell: GrowthCell) -> float:
        dose = max(0.0, cell.dose_s - cell.incubation_s)
        return -math.expm1(-2.3 * dose / self.profile.optimum_time_s)

    def defect_fraction(self, cell: GrowthCell) -> float:
        overgrowth = max(0.0, cell.dose_s / self.profile.optimum_time_s - 1.0) * 0.6
        return -math.expm1(-(cell.damage + overgrowth))

    def measure(self, region: tuple[float, float, float, float] | None = None
                ) -> dict[str, Any]:
        """A noiseless virtual image analysis, optionally over a chip-local ROI.

        ROI coordinates are fractions of the chip width/height relative to its
        centre. This is simulated perception, not computer vision on real pixels.
        Hidden dose, optimum and cell truth are not part of this observation.
        """
        cells = self.cells if region is None else [c for c in self.cells
            if region[0] <= c.x <= region[1] and region[2] <= c.y <= region[3]]
        if not cells:
            return {"visible": False, "coverage": None, "uniformity": None,
                    "defect_fraction": None, "quality_score": None}
        coverages = [self.coverage(c) for c in cells]
        mean = sum(coverages) / len(cells)
        variance = sum((v - mean) ** 2 for v in coverages) / len(cells)
        uniformity = max(0.0, 1.0 - math.sqrt(variance) / max(mean, 1e-9)) if mean else 0.0
        defects = sum(self.defect_fraction(c) for c in cells) / len(cells)
        phase = ("no_reaction" if mean == 0 else "nucleating" if mean < 0.15 else
                 "growing" if mean < 0.8 else "continuous")
        if defects >= 0.25:
            phase = "damaged_or_overgrown"
        return {"visible": True, "coverage": mean, "uniformity": uniformity,
                "defect_fraction": defects, "quality_score": mean * uniformity * (1 - defects),
                "phase": phase, "source": "synthetic_image_analysis"}

    def snapshot(self) -> dict[str, Any]:
        return {"seed": self.seed, "chip_id": self.chip_id,
                "profile": self.profile.model_dump(mode="json"),
                "elapsed_s": self.elapsed_s, "cells": [asdict(c) for c in self.cells]}

    @classmethod
    def restore(cls, state: dict[str, Any]) -> GrowthModel:
        profile = GrowthProfile.model_validate(state["profile"])
        out = cls(int(state["seed"]), str(state["chip_id"]), profile)
        out.elapsed_s = float(state["elapsed_s"])
        cells = [GrowthCell(**c) for c in state["cells"]]
        if len(cells) != profile.grid_size ** 2:
            raise ValueError("growth snapshot has an inconsistent grid")
        out.cells = cells
        return out
