"""Qualitative two-flake local reaction: zero or one connected product.

No CVD nucleation, gas chemistry, activation energies, yield predictions or
mass conservation are inferred. An explicit synthetic exposure control drives
an overlap-bounded morphological interpolation. The reactor owns the clock,
mechanics and decisions about whether the fixture's exposure is active.
"""

from __future__ import annotations

import math
import random
from typing import Any, Literal

from pydantic import Field, model_validator

from qumus_simlab.core.polygons import area_um2, contract
from qumus_simlab.testbench.material_flakes import (
    MODEL_VERSION,
    FrozenModel,
    PairAssembly,
    stable_key,
)


class ReactionProfile(FrozenModel):
    name: str = "uncalibrated local reaction"
    product_label: str = Field("Unspecified synthetic product", min_length=1)
    exposure_to_completion_s: float = Field(60, gt=0)
    incubation_s: float = Field(0, ge=0)
    max_conversion: float = Field(1, ge=0, le=1)
    failure_probability: float = Field(0, ge=0, le=1)


class ReactionState(FrozenModel):
    version: Literal["qualitative-material-v1"] = MODEL_VERSION
    seed: int
    assembly: PairAssembly
    profile: ReactionProfile
    elapsed_s: float = Field(0, ge=0)
    active_s: float = Field(0, ge=0)
    failure_reason: str | None = None

    @model_validator(mode="after")
    def coherent(self) -> ReactionState:
        if self.active_s > self.elapsed_s:
            raise ValueError("active exposure cannot exceed elapsed time")
        if self.failure_reason is not None and not self.failure_reason.strip():
            raise ValueError("failure reason must be nonempty")
        return self


class LocalReaction:
    """One assembly's persisted state; observe() is a pure, deterministic read.

    Exactly two identities enter. One stable product identity appears after
    positive conversion; failures may leave no product or retain a partial one.
    A different pair/pose requires a new instance, never a hidden state reset.
    """

    def __init__(self, assembly: PairAssembly, *, seed: int, profile: ReactionProfile | None = None):
        p = profile or ReactionProfile()
        failure = random.Random(stable_key(seed, assembly.id, "failure")).random() < p.failure_probability
        self._state = ReactionState(seed=seed, assembly=assembly, profile=p,
                                    failure_reason="seeded_fixture_failure" if failure else None)

    @property
    def state(self) -> ReactionState:
        return self._state

    def advance(self, seconds: float, *, active: bool) -> None:
        if not math.isfinite(seconds) or seconds < 0 or type(active) is not bool:
            raise ValueError("duration must be finite and nonnegative; active must be boolean")
        s = self.state
        # Exposure is frozen after a terminal failure; elapsed time still passes.
        elapsed = s.elapsed_s + seconds
        exposure = s.active_s + (seconds if active and s.failure_reason is None else 0)
        self._state = ReactionState(**{**s.model_dump(), "elapsed_s": elapsed, "active_s": exposure})

    def fail(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure requires an explicit reason")
        if self.state.failure_reason is None:
            self._state = ReactionState(**{**self.state.model_dump(), "failure_reason": reason})

    def observe(self) -> dict[str, Any]:
        s = self.state
        assembly, p = s.assembly, s.profile
        overlap = assembly.overlap_mm
        overlap_area = area_um2(overlap)
        conversion = min(1, max(0, s.active_s - p.incubation_s) / p.exposure_to_completion_s)
        conversion *= p.max_conversion
        if not overlap:
            conversion = 0
        # A fixed convex combination lies inside the overlap. It is sampled from
        # identity on each pure read, not a mutable RNG or a camera-dependent key.
        anchor = None
        if overlap:
            rng = random.Random(stable_key(s.seed, assembly.model_dump(mode="json"), "anchor"))
            weights = [rng.uniform(0.2, 1) for _ in overlap]
            total = sum(weights)
            anchor = (sum(w * x for w, (x, _) in zip(weights, overlap, strict=True)) / total,
                      sum(w * y for w, (_, y) in zip(weights, overlap, strict=True)) / total)
        outline = contract(overlap, conversion, anchor)
        source_ids = [placed.flake.id for placed in assembly.inputs]
        product = None
        if outline and area_um2(outline) > 1e-8:
            product = {
                "id": "synthetic_product_" + stable_key(assembly.model_dump(mode="json"), s.seed),
                "material": p.product_label, "layers": None, "outline_mm": outline,
                "area_um2": area_um2(outline), "source_flake_ids": source_ids,
                "assembly_id": assembly.id, "chip_id": assembly.target_chip_id,
            }
        status = ("failed" if s.failure_reason else "no_overlap" if not overlap else
                  "unreacted" if conversion == 0 else "complete" if conversion == 1 else "partial")
        return {
            "model_version": MODEL_VERSION, "source": "qualitative_synthetic_model",
            "status": status, "failure_reason": s.failure_reason, "elapsed_s": s.elapsed_s,
            "active_s": s.active_s, "assembly_id": assembly.id, "chip_id": assembly.target_chip_id,
            "inputs": [{"id": placed.flake.id, "material": placed.flake.material,
                        "layers": placed.flake.layers, "source_chip_id": placed.flake.source_chip_id,
                        "batch_id": placed.flake.batch_id, "source_outline_mm": placed.flake.outline_mm,
                        "target_center_mm": placed.target_center_mm, "rotation_deg": placed.rotation_deg,
                        "outline_mm": placed.outline_mm, "area_um2": placed.flake.area_um2}
                       for placed in assembly.inputs],
            "overlap_mm": overlap, "overlap_area_um2": overlap_area,
            "conversion_of_overlap": conversion, "product": product,
            "remaining_overlap_area_um2": overlap_area * (1 - conversion),
            "profile": p.model_dump(mode="json"),
        }

    def snapshot(self) -> dict[str, Any]:
        return self.state.model_dump(mode="json")

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> LocalReaction:
        state = ReactionState.model_validate(snapshot)
        out = cls.__new__(cls)
        out._state = state
        return out
