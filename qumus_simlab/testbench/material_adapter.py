"""Small integration helpers for the existing software world and microscope.

No device calls, ledger writes, clock access or reactor mechanics live here.
Legacy area-only flakes get a deterministic display outline, never provenance
invented as a new physical deposition. Fresh deposits store their geometry.
"""

from __future__ import annotations

import math
import random
from typing import Any

from qumus_simlab.core.polygons import Polygon, rectangle, transform
from qumus_simlab.testbench.material_flakes import (
    MODEL_VERSION,
    CleavageProfile,
    Flake,
    Placement,
    cleave,
    flake_outline,
    project_outline,
    stable_key,
)


def cleavage_candidates(*, seed: int, chip_id: str, size_mm: tuple[float, float],
                        segment: int, uses: int, material: str, count: int,
                        cells: list[tuple[int, int]], cell_mm: float,
                        classes: list[dict], weights: list[float], config: dict,
                        rng: random.Random, first_index: int) -> list[dict]:
    """Keep existing recipe/count controls; add bounded, persistent morphology."""
    if "morphology" not in config and min(size_mm) >= 1:
        return _compatible_candidates(
            seed=seed, chip_id=chip_id, size_mm=size_mm, segment=segment, uses=uses,
            material=material, count=count, cells=cells, cell_mm=cell_mm,
            classes=classes, weights=weights, config=config, rng=rng, first_index=first_index)
    values, probabilities = [], []
    for cls, weight in zip(classes, weights, strict=True):
        layers = cls["layers"]
        group = list(range(layers[0], layers[1] + 1)) if isinstance(layers, list) else [int(layers)]
        values.extend(group)
        probabilities.extend([weight / len(group)] * len(group))
    params = {"median_area_um2": config.get("area_um2_median", 250),
              "log_area_sigma": config.get("area_log_sigma", 1),
              **config.get("morphology", {}),
              "layer_values": values, "layer_weights": probabilities}
    profile = CleavageProfile.model_validate(params)
    regions = tuple(rectangle(cell_mm, cell_mm,
                              ((i + 0.5) * cell_mm - size_mm[0] / 2,
                               (j + 0.5) * cell_mm - size_mm[1] / 2)) for i, j in cells)
    batch_id = f"tape_segment_{segment}_use_{uses}"
    flakes = cleave(seed=seed, chip_id=chip_id, batch_id=batch_id, material=material, count=count,
                    chip_size_mm=size_mm, profile=profile, contact_regions=regions)
    return [{"id": f.id, "material": f.material, "layers": f.layers,
             "x_mm": f.center_mm[0], "y_mm": f.center_mm[1], "area_um2": f.area_um2,
             "outline_mm": transform(f.outline_mm, xy=(-f.center_mm[0], -f.center_mm[1])),
             "origin": {"source_chip_id": chip_id, "batch_id": batch_id,
                        "model_version": MODEL_VERSION, "seed": seed,
                        "population_mode": "clustered_contact_regions", "source_outline_mm": f.outline_mm}}
            for f in flakes]


def sim_outline(flake: Any) -> Polygon:
    """Offsets from a SimFlake's centre in its current carrier's frame."""
    return getattr(flake, "outline_mm", ()) or flake_outline(
        stable_key("legacy-display", flake.id, flake.material, flake.layers), flake.area_um2)


def microscope_flake(flake: Any, center_mm: tuple[float, float], fov_mm: tuple[float, float],
                     *, rotation_deg: float = 0) -> dict | None:
    """Camera-centred mm, +y up. Includes partially visible outlines."""
    outline = transform(sim_outline(flake), xy=center_mm, rotation_deg=rotation_deg)
    projected = project_outline(outline, center_mm=(0, 0), fov_mm=fov_mm)
    if not projected["visible"]:
        return None
    return {"sim_id": flake.id, "material": flake.material, "layers": flake.layers,
            "area_um2": flake.area_um2, "dx_mm": center_mm[0], "dy_mm": center_mm[1],
            "outline_mm": outline, "visible_area_um2": projected["visible_area_um2"],
            "geometry_source": MODEL_VERSION if getattr(flake, "outline_mm", ()) else "legacy_display",
            "catalog_id": getattr(flake, "origin", {}).get("catalog_id"),
            "source_flake_ids": getattr(flake, "origin", {}).get("source_flake_ids", []),
            "material_role": getattr(flake, "origin", {}).get("material_role", "flake")}


def catalog_placement(flake: Any, catalog_id: str) -> Placement:
    """Map physical state to a ledger identity without losing source geometry.

    Call after explicit selection/transfer, with the real catalog ID. Legacy
    samples lacking provenance need a new declared fixture; do not infer origin.
    """
    origin = getattr(flake, "origin", {})
    if not all(k in origin for k in ("source_chip_id", "batch_id", "source_outline_mm")):
        raise ValueError("material preparation needs explicit source geometry and provenance")
    source = Flake(id=catalog_id, material=flake.material, layers=flake.layers,
                   source_chip_id=origin["source_chip_id"], batch_id=origin["batch_id"],
                   outline_mm=origin["source_outline_mm"])
    current = sim_outline(flake)
    if len(current) != len(source.outline_mm):
        raise ValueError("transferred outline no longer matches source geometry")
    a, b = source.outline_mm[:2]
    c, d = current[:2]
    angle = math.degrees(math.atan2(d[1] - c[1], d[0] - c[0]) - math.atan2(b[1] - a[1], b[0] - a[0]))
    placement = Placement(flake=source, target_center_mm=(flake.x_mm, flake.y_mm), rotation_deg=angle)
    expected = transform(current, xy=(flake.x_mm, flake.y_mm))
    if any(math.dist(x, y) > 1e-9 for x, y in zip(expected, placement.outline_mm, strict=True)):
        raise ValueError("transferred outline is not a rigid transform of its source")
    if not math.isclose(source.area_um2, flake.area_um2, rel_tol=1e-8):
        raise ValueError("physical area disagrees with source geometry")
    return placement


def prepare_catalog_pair(chip: Any, samples: list[dict], *, preparation_id: str,
                         preparation_source: str = "stacking_job") -> tuple[dict, dict[str, dict]]:
    """Bind two catalog identities to flakes already physically on the target.

    Caller owns guarded transfer, reservations and the ledger transaction. This
    reads physical state; it never moves, creates or relabels physical flakes.
    Samples are ordered bottom then top. Measured area is not replaced by truth.
    """
    from qumus_simlab.testbench.material_flakes import PairAssembly

    if not preparation_id.strip() or preparation_source not in ("stacking_job", "operator_report"):
        raise ValueError("a real preparation needs an ID and an explicit preparation source")
    if len(samples) != 2 or len({s["id"] for s in samples}) != 2:
        raise ValueError("preparation requires exactly two distinct catalog samples")
    physical = {f.id: f for f in chip.flakes}
    sim_ids = [s["props"].get("sim_id") for s in samples]
    if len(chip.flakes) != 2 or len(set(sim_ids)) != 2 or set(sim_ids) != set(physical):
        raise ValueError("the carrier must physically contain exactly the selected catalog flakes")
    placements, updates = [], {}
    for sample, sim_id in zip(samples, sim_ids, strict=True):
        props, flake = sample["props"], physical[sim_id]
        if sample["kind"] != "flake" or props.get("material") != flake.material:
            raise ValueError("catalog material must match the transferred physical flake")
        if props.get("reactor_consumed_by") or sample.get("status") == "reactor_consumed":
            raise ValueError("a consumed source cannot be prepared again")
        measured = props.get("area_um2")
        if isinstance(measured, bool) or not isinstance(measured, (int, float)) \
                or not math.isfinite(measured) or measured <= 0:
            raise ValueError("catalog sample needs a finite positive measured area")
        placed = catalog_placement(flake, sample["id"])
        placements.append(placed)
        updates[sample["id"]] = {"material_placement": placed.model_dump(mode="json"),
                                 "geometry_area_um2": placed.flake.area_um2,
                                 "measured_area_um2": measured, "sim_id": sim_id}
    # Validate whole-chip containment before returning any prospective writes.
    PairAssembly(id=preparation_id, target_chip_id=chip.id,
                 chip_size_mm=chip.size_mm, inputs=placements)
    manifest = {"schema_version": 1, "preparation_id": preparation_id, "chip_id": chip.id,
                "input_flake_ids": [s["id"] for s in samples], "preparation_source": preparation_source}
    return manifest, updates


def sync_pair_flakes(chip: Any, model: Any, samples: list[dict], *, allow_fixture: bool = False) -> None:
    """Project persisted reaction geometry into ordinary physical flakes.

    A first real binding must match transferred geometry. Only declared fixture
    sources may be initialized from placements. A terminal product moved away
    later is never recreated by looking at its original carrier. No ledger writes.
    Caller must hold the world's lock, including while checkpointing.
    """
    from dataclasses import replace

    from qumus_simlab.testbench.world import SimFlake

    assembly, obs = model.state.assembly, model.observe()
    if assembly.target_chip_id != chip.id:
        raise ValueError("material model belongs to a different carrier")
    signature = stable_key(assembly.model_dump(mode="json"), obs["status"], obs["product"])
    prior = getattr(chip, "material_binding", {})
    if prior and prior.get("preparation_id") != assembly.id:
        raise ValueError("carrier is already bound to a different material preparation")
    by_id = {s["id"]: s for s in samples}
    if set(by_id) != {p.flake.id for p in assembly.inputs}:
        raise ValueError("physical binding requires the prepared source identities")
    source_ids = [by_id[p.flake.id]["props"].get("sim_id", p.flake.id) for p in assembly.inputs]
    if len(set(source_ids)) != 2:
        raise ValueError("both catalog sources cannot refer to the same physical flake")
    physical = {f.id: f for f in chip.flakes}
    product = obs["product"]
    allowed = set(source_ids) | ({product["id"]} if product else set())
    if len(physical) != len(chip.flakes) or set(physical) - allowed:
        raise ValueError("carrier has unexpected physical material")
    converted = obs["conversion_of_overlap"] == 1
    if prior.get("signature") == signature and converted:
        # A completed product may subsequently move through a real transfer.
        # Its source carrier must never regenerate it from saved provenance.
        if set(physical) - {product["id"]}:
            raise ValueError("consumed reactants cannot remain on a completed carrier")
        return
    sources = []
    for placed, sim_id in zip(assembly.inputs, source_ids, strict=True):
        actual = physical.get(sim_id)
        if actual is None and not (allow_fixture and not prior):
            # A stored completed model can already have only its product.
            if converted and product and product["id"] in physical:
                continue
            raise ValueError("prepared physical source is missing from the carrier")
        if actual is not None:
            current = transform(sim_outline(actual), xy=(actual.x_mm, actual.y_mm))
            if (actual.material != placed.flake.material
                    or not math.isclose(actual.area_um2, placed.flake.area_um2, rel_tol=1e-8)
                    or len(current) != len(placed.outline_mm)
                    or any(math.dist(a, b) > 1e-8 for a, b in zip(current, placed.outline_mm, strict=True))):
                raise ValueError("prepared physical geometry no longer matches its source placement")
        else:
            cx, cy = placed.target_center_mm
            actual = SimFlake(sim_id, placed.flake.material, placed.flake.layers, cx, cy,
                              placed.flake.area_um2, transform(placed.outline_mm, xy=(-cx, -cy)))
        origin = {**getattr(actual, "origin", {}), "source_chip_id": placed.flake.source_chip_id,
                  "batch_id": placed.flake.batch_id, "source_outline_mm": placed.flake.outline_mm,
                  "catalog_id": placed.flake.id, "reactor_preparation_id": assembly.id,
                  "material_role": "reactant"}
        sources.append(replace(actual, origin=origin))
    if prior.get("signature") == signature:
        return  # validated unchanged sources; observation itself changes nothing
    flakes = [] if converted else sources
    if product:
        outline = product["outline_mm"]
        cx, cy = (sum(x for x, _ in outline) / len(outline), sum(y for _, y in outline) / len(outline))
        flakes.append(SimFlake(product["id"], product["material"], None, cx, cy, product["area_um2"],
                              transform(outline, xy=(-cx, -cy)), {
                                  "catalog_id": product["id"], "source_chip_id": chip.id,
                                  "batch_id": assembly.id, "source_outline_mm": outline,
                                  "source_flake_ids": product["source_flake_ids"],
                                  "reactor_preparation_id": assembly.id, "material_role": "product",
                                  "reaction_status": obs["status"], "thickness_known": False}))
    chip.flakes = flakes
    chip.material_binding = {"preparation_id": assembly.id, "signature": signature,
                             "status": obs["status"], "product_id": product["id"] if product else None}


def _compatible_candidates(*, seed: int, chip_id: str, size_mm: tuple[float, float],
                           segment: int, uses: int, material: str, count: int,
                           cells: list[tuple[int, int]], cell_mm: float, classes: list[dict],
                           weights: list[float], config: dict, rng: random.Random,
                           first_index: int) -> list[dict]:
    """Preserve old scalar draws for existing scripted fixtures, adding geometry.

    Explicit exfoliation.morphology opts into clustered/contact-contained
    generation. Compatibility mode retains the existing edge-clamped centres;
    it is not the improved spatial distribution. Area is now bounded.
    """
    result = []
    width, height = size_mm
    for i in range(count):
        layers = rng.choices(classes, weights=weights)[0]["layers"]
        nl = rng.randint(layers[0], layers[1]) if isinstance(layers, list) else int(layers)
        area = config.get("area_um2_median", 250) * math.exp(rng.gauss(0, config.get("area_log_sigma", 1)))
        if nl >= 10:
            area *= 4
        ci, cj = rng.choice(cells)
        fx = min(max((ci + rng.random()) * cell_mm - width / 2, -width / 2 + 0.4), width / 2 - 0.4)
        fy = min(max((cj + rng.random()) * cell_mm - height / 2, -height / 2 + 0.4), height / 2 - 0.4)
        area = round(min(12000, max(12, area)), 1)
        fid = f"simflake_{first_index + i + 1:05d}"
        outline = flake_outline(stable_key(seed, chip_id, segment, uses, fid), area)
        result.append({"id": fid, "material": material, "layers": nl, "x_mm": fx, "y_mm": fy,
                       "area_um2": area, "outline_mm": outline,
                       "origin": {"source_chip_id": chip_id, "batch_id": f"tape_segment_{segment}_use_{uses}",
                                  "model_version": MODEL_VERSION, "seed": seed,
                                  "population_mode": "compatible_scalar_draws",
                                  "source_outline_mm": transform(outline, xy=(fx, fy))}})
    return result
