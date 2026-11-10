"""Finite fresh Gr-only source policy for the separately admitted primary case.

Only public records reach selection. No saved-world reuse, seed search, shape
repair, controller change or outcome-driven replacement is supported here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import monotonic

from qumus_simlab.core.polygons import transform
from qumus_simlab.flows.bench import slot_station
from qumus_simlab.flows.flow import get_flow
from qumus_simlab.records.geometry import (
    _pickup_outline,
    contact_bridge_geometry,
    current_pickup_evidence,
    resolve_direct_contact_contract,
)

SOURCE_POLICY = {
    "version": "fresh-ideal8-direct-gr-source-v1",
    "seed": 8,
    "simulation_condition": "ideal_stacking_v1",
    "chip_order": [f"chip_{i:03d}" for i in range(1, 5)],
    "max_rounds": 4,
    "max_jobs": 12,
    "wall_cap_s": 1800,
    "recipe": {
        "material": "Gr",
        "num_chips": 1,
        "prepare_mother_tape": "always",
        "target_temp_c": 50,
        "dwell_s": 30,
        "num_massages": 2,
        "press_depth_mm": 0.2,
        "peel_speed": 200,
        "peel_angle_deg": 30,
    },
    "search": {"material": "Gr", "min_area_um2": 50, "max_revisits": 20},
    "extra_revisit_max": 20,
    "order": "area descending then ID ascending",
    "stop": "first qualified fit",
}


class DirectGrSourceInfeasible(AssertionError):
    pass


def eligible_blank(s, chip_id: str, preserved_chip_ids=()) -> bool:
    chip = s.store.get_sample(chip_id)
    return bool(
        chip
        and chip["kind"] == "chip"
        and chip["status"] in ("bare", "available")
        and chip_id not in preserved_chip_ids
        and chip_id not in s.store.reserved_samples()
        and not any(
            chip["props"].get(k)
            for k in ("location_uncertain", "reactor_substrate", "owner_preserved", "preserved")
        )
        and slot_station(s.lab.bench, chip["location"]) == "left_storage"
    )


def ranked_candidates(s, chip_id: str, *, statuses=("catalogued",)) -> list[dict]:
    return sorted(
        (
            f
            for f in s.store.list_samples(kind="flake", location=chip_id)
            if f["status"] in statuses
            and f["props"].get("material") == "Gr"
            and not f["props"].get("location_uncertain")
            and isinstance(f["props"].get("area_um2"), (int, float))
            and math.isfinite(f["props"]["area_um2"])
            and f["props"]["area_um2"] >= 50
        ),
        key=lambda f: (-f["props"]["area_um2"], f["id"]),
    )


def qualified_choice(s, chip_id: str, intent: dict):
    for f in ranked_candidates(s, chip_id):
        evidence = current_pickup_evidence(s.store, f)
        ref = f["props"].get("tracking_references", {}).get("20x", {})
        g = f["props"].get("optical_geometry") or {}
        if (
            not evidence
            or evidence.get("clear") is not True
            or ref.get("meta", {}).get("objective") != 20
            or ref.get("job_id") != evidence["job_id"]
            or g.get("job_id") != evidence["job_id"]
            or not Path(ref.get("image", "")).is_file()
        ):
            continue
        try:
            _pickup_outline(s.store, f)
            c = resolve_direct_contact_contract(s.store, [f["id"]], intent)
            shape = transform(c.source_outline_mm, rotation_deg=c.outline_rotation_deg)
            if (
                contact_bridge_geometry(
                    shape, intent["target_contact"], intent["bridge_requirement"]
                )["status"]
                == "pass"
            ):
                return c
        except (ValueError, TypeError, KeyError, OSError):
            continue
    return None


def extra_revisit_candidates(s, chip_id: str) -> list[str]:
    # Current rejections and complete native20 observations are terminal for
    # this policy, even when they are too small or their neighbourhood is bad.
    result = []
    for f in ranked_candidates(s, chip_id, statuses=("candidate", "catalogued")):
        # Normal ScanChip discoveries remain candidate until a revisit.
        if current_pickup_evidence(s.store, f) is not None:
            continue
        g = f["props"].get("optical_geometry") or {}
        ref = f["props"].get("tracking_references", {}).get("20x") or {}
        run = s.store.get_run(ref.get("job_id", ""))
        if g.get("outline_mm") or (run and run["status"] == "succeeded"):
            continue
        result.append(f["id"])
    return result[: SOURCE_POLICY["extra_revisit_max"]]


def build_direct_gr_source(s, intent: dict, folder: Path, *, preserved_chip_ids=()):
    """Execute <=4 stock-limited rounds; caller owns fresh ideal8 setup and admission."""
    from tests.test_stacking import run

    from qumus_simlab.config import load_yaml

    folder.mkdir(parents=True, exist_ok=True)
    deadline, jobs = monotonic() + SOURCE_POLICY["wall_cap_s"], []
    manifest = {
        "policy": SOURCE_POLICY,
        "target_intent": intent,
        "preserved_chip_ids": list(preserved_chip_ids),
        "jobs": jobs,
        "rounds": [],
        "status": "building",
        "selected": None,
        "before_transfer": True,
    }

    def save():
        (folder / "source-policy-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def submit(flow, params):
        remaining = deadline - monotonic()
        if remaining <= 0 or len(jobs) >= SOURCE_POLICY["max_jobs"]:
            raise DirectGrSourceInfeasible("finite source job/time budget exhausted; no transfer")
        status = run(s, flow, params, timeout=min(900, remaining))
        jobs.append(
            {
                "id": status.run_id,
                "flow": flow,
                "params": s.store.get_run(status.run_id)["params"],
                "status": status.status,
            }
        )
        save()
        if status.status != "succeeded":
            raise DirectGrSourceInfeasible(
                f"source job {status.run_id} ended {status.status}; no retry/transfer"
            )

    def choose(chip_id):
        if monotonic() >= deadline:
            raise DirectGrSourceInfeasible(
                "source wall budget exhausted before prospective qualification; no transfer"
            )
        chosen = qualified_choice(s, chip_id, intent)
        if monotonic() >= deadline:
            raise DirectGrSourceInfeasible(
                "source wall budget exhausted during prospective qualification; no transfer"
            )
        return chosen

    try:
        save()
        for chip_id in SOURCE_POLICY["chip_order"]:
            if not eligible_blank(s, chip_id, preserved_chip_ids):
                manifest["rounds"].append(
                    {"chip_id": chip_id, "status": "unavailable_public_stock"}
                )
                save()
                continue
            if deadline - monotonic() <= 0:
                break
            advance = load_yaml("verification.yaml")["tape"]["advance_mm"]
            tape = s.store.get_sample("tape")
            if not tape or tape["props"].get("roll_left_mm", 0) < advance:
                manifest["stock_stop"] = "insufficient public tape"
                break
            params = {**SOURCE_POLICY["recipe"], "chip_ids": [chip_id]}
            flow = get_flow("exfoliate_chips")
            flow._bench = s.lab.bench
            try:
                flow.resolve_params(flow.Params.model_validate(params), s.store)
            except ValueError as exc:
                manifest["stock_stop"] = str(exc)
                break
            manifest["rounds"].append({"chip_id": chip_id, "status": "started"})
            save()
            submit("exfoliate_chips", params)
            submit("search_chip", {**SOURCE_POLICY["search"], "chip_id": chip_id})
            chosen = choose(chip_id)
            if chosen is None:
                ids = extra_revisit_candidates(s, chip_id)
                if ids:
                    submit(
                        "revisit_chip",
                        {
                            **SOURCE_POLICY["search"],
                            "chip_id": chip_id,
                            "max_revisits": len(ids),
                            "flake_ids": ids,
                        },
                    )
                    chosen = choose(chip_id)
            if chosen is not None:
                manifest.update(
                    status="qualified_frozen_before_transfer",
                    selected=chosen.source_flake_id,
                    direct_contact_contract=chosen.model_dump(mode="json"),
                )
                save()
                return chosen, manifest
            manifest["rounds"][-1]["status"] = "no_qualified_fixed_orientation_fit"
            save()
        raise DirectGrSourceInfeasible(
            "no qualified Gr bridges the fixed contacts within actual stock and finite policy; no transfer"
        )
    except BaseException as exc:
        manifest.update(
            status="infeasible_or_unfinished_no_transfer",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        save()
        raise
