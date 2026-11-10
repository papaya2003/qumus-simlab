"""Ledger-side cardinality and lineage checks, independent of material morphology.

The material backend owns overlap geometry and reaction kinetics. This boundary
checks that an observation concerns exactly the prepared pair and cannot turn
it into several products or a product larger than either source flake.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from qumus_simlab.runtime.store import RunStore


class PairPreparation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    preparation_id: str = Field(min_length=1)
    chip_id: str = Field(min_length=1)
    input_flake_ids: tuple[str, str]
    preparation_source: Literal["synthetic_fixture", "operator_report", "stacking_job"]

    @model_validator(mode="after")
    def distinct_inputs(self) -> PairPreparation:
        if not all(self.input_flake_ids) or len(set(self.input_flake_ids)) != 2:
            raise ValueError("a reactor preparation requires exactly two distinct flake IDs")
        return self


class ReactionProduct(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    id: str = Field(min_length=1)
    material: str = Field(min_length=1)
    area_um2: float = Field(gt=0, strict=True)
    source_flake_ids: tuple[str, str]
    # Serialization owned by the material model; never regenerated here.
    material_state: dict[str, Any]


class PairObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    experiment_kind: Literal["two_flake_reaction"] = "two_flake_reaction"
    preparation_id: str = Field(min_length=1)
    input_flake_ids: tuple[str, str]
    visible: StrictBool
    status: Literal["failed", "no_overlap", "unreacted", "partial", "complete"]
    failure_reason: str | None = None
    reaction_fraction: float | None = Field(None, ge=0, le=1, strict=True)
    product: ReactionProduct | None = None
    model_id: str = Field(min_length=1)
    source: Literal["synthetic_material_model"] = "synthetic_material_model"


def prepared_pair(store: RunStore, chip_id: str, *, run_id: str | None = None,
                  allow_consumed: bool = False) -> PairPreparation:
    """Require the actual ledger occupants, not only a list supplied by an agent."""
    chip = store.get_sample(chip_id)
    if not chip or chip["kind"] != "chip" or not chip["props"].get("reactor_substrate"):
        raise ValueError("choose a dedicated reactor substrate")
    manifest = chip["props"].get("reactor_pair")
    if not manifest:
        raise ValueError("reactor chip has no verified two-flake preparation manifest")
    pair = PairPreparation.model_validate(manifest)
    if pair.chip_id != chip_id:
        raise ValueError("preparation belongs to another carrier chip")
    reserved = store.reserved_samples(exclude_run=run_id)
    for fid in pair.input_flake_ids:
        flake = store.get_sample(fid)
        if not flake or flake["kind"] != "flake" or flake["location"] != chip_id:
            raise ValueError("both prepared input flakes must be on this carrier chip")
        if fid in reserved:
            raise ValueError("an input flake is reserved by another job")
        consumed = flake["props"].get("reactor_consumed_by")
        if consumed and (not allow_consumed or consumed != pair.preparation_id):
            raise ValueError("an input flake was already consumed")
        area = flake["props"].get("area_um2")
        if isinstance(area, bool) or not isinstance(area, (int, float)) \
                or not math.isfinite(area) or area <= 0:
            raise ValueError("input flakes need finite positive measured or synthetic areas")
    products = [s for s in store.list_samples(kind="flake", location=chip_id)
                if s["props"].get("reactor_preparation_id") == pair.preparation_id
                and s["props"].get("source_flake_ids") == list(pair.input_flake_ids)]
    if len(products) > 1:
        raise ValueError("a preparation cannot produce more than one product flake")
    product_ids = {s["id"] for s in products} if allow_consumed else set()
    occupants = {s["id"] for s in store.list_samples(kind="flake", location=chip_id)
                 if s["id"] not in product_ids}
    if occupants != set(pair.input_flake_ids):
        raise ValueError("reactor carrier must contain exactly the two prepared input flakes")
    return pair


def validate_pair_observation(store: RunStore, chip_id: str, value: dict[str, Any],
                              *, run_id: str | None = None,
                              require_product: bool = False) -> PairObservation:
    pair = prepared_pair(store, chip_id, run_id=run_id, allow_consumed=True)
    obs = PairObservation.model_validate(value)
    if obs.preparation_id != pair.preparation_id or obs.input_flake_ids != pair.input_flake_ids:
        raise ValueError("observation does not belong to the current ordered flake pair")
    if require_product and (not obs.visible or obs.product is None
                            or obs.status != "complete" or obs.failure_reason
                            or obs.reaction_fraction is None or obs.reaction_fraction <= 0):
        raise ValueError("no visible verified reaction product for this preparation")
    product = obs.product
    if product is not None:
        if product.source_flake_ids != pair.input_flake_ids or product.id in pair.input_flake_ids:
            raise ValueError("product lineage must identify both distinct source flakes")
        areas = [store.get_sample(fid)["props"].get("geometry_area_um2",
                 store.get_sample(fid)["props"]["area_um2"]) for fid in pair.input_flake_ids]
        if any(isinstance(a, bool) or not isinstance(a, (int, float))
               or not math.isfinite(a) or a <= 0 for a in areas):
            raise ValueError("input geometry areas must be finite and positive")
        minimum = min(areas)
        if product.area_um2 > minimum + 1e-9:
            raise ValueError("product cannot be larger than either input flake")
        products = [s for s in store.list_samples(kind="flake")
                    if s["props"].get("reactor_preparation_id") == pair.preparation_id]
        if any(s["id"] != product.id for s in products):
            raise ValueError("this preparation already has a different product identity")
        existing = store.get_sample(product.id)
        if existing and (existing["kind"] != "flake" or existing["location"] != chip_id
                         or existing["props"].get("reactor_preparation_id") != pair.preparation_id
                         or existing["props"].get("source_flake_ids") != list(pair.input_flake_ids)):
            raise ValueError("product ID is already used by another sample or preparation")
    return obs


def record_pair_product(store: RunStore, chip_id: str, value: dict[str, Any], *,
                        reaction_job_id: str, project_id: str) -> str:
    """Atomically check and persist one product, across SQLite connections."""
    with store._transaction():
        return _record_pair_product(store, chip_id, value, reaction_job_id=reaction_job_id,
                                    project_id=project_id)


def _record_pair_product(store: RunStore, chip_id: str, value: dict[str, Any], *,
                         reaction_job_id: str, project_id: str) -> str:
    """Requires an active store transaction, including all acceptance reads."""
    obs = validate_pair_observation(store, chip_id, value, require_product=True)
    run = store.get_run(reaction_job_id)
    if not run or run["flow"] != "reactor_reaction" or run["status"] != "succeeded" \
            or run["project_id"] != project_id or run["params"].get("chip_id") != chip_id:
        raise ValueError("product needs this project's successful reaction job for this chip")
    product = obs.product
    assert product is not None
    existing = store.get_sample(product.id)
    if existing and existing["props"].get("reactor_evidence_job_id") != reaction_job_id:
        raise ValueError("the existing product belongs to a different reaction job")
    # Reuse the saved, accepted observation; do not fabricate a new microscope view.
    result = store.get_sample(chip_id)["props"].get("reactor_result") or {}
    images = result.get("images") or []
    photo = images[-1] if result.get("reaction_job_id") == reaction_job_id and images else None
    store.upsert_sample(product.id, "flake", location=chip_id, status="reactor_product",
                        props={"material": product.material, "area_um2": product.area_um2,
                               "chip_id": chip_id,
                               "images": {"reactor": photo} if photo else {},
                               "source_flake_ids": list(obs.input_flake_ids),
                               "reactor_preparation_id": obs.preparation_id,
                               "reactor_evidence_job_id": reaction_job_id,
                               "material_state": product.material_state,
                               "image_source": "simulated"})
    for fid in obs.input_flake_ids:
        store.upsert_sample(fid, "flake", status="reactor_consumed",
                            props={"reactor_consumed_by": obs.preparation_id})
    return product.id
