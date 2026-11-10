"""Advisory revisit order from public coarse pixels; never source admission."""
from __future__ import annotations

import hashlib
from pathlib import Path

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from qumus_simlab.core.regions import _geos, as_region


def coarse_region(sample, image_hashes):
    evidence = sample.get("props", {}).get("coarse_geometry_10x") or {}
    try:
        image = evidence["image"]
        path = Path(image["path"])
        if str(path) not in image_hashes:
            image_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if (evidence["source"] != "v1_lab_rgb" or evidence["objective"] != 10
                or evidence["frame"] != "flake_center_chip_axes" or evidence["units"] != "mm"
                or image_hashes[str(path)] != image["sha256"] or not evidence["calibration"]
                or evidence["calibration"]["image_size_px"] != image["size_px"]):
            return None
        return _geos(as_region(evidence["region_mm"]))
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _centred(shape):
    return affinity.translate(shape, -shape.centroid.x, -shape.centroid.y)


def _potential(outer, inner):
    """Centred 15degree grid: a hit is observed potential; a miss is unknown."""
    if outer.area < inner.area:
        return False
    outer, inner = _centred(outer), _centred(inner)
    return any(affinity.rotate(outer, angle, origin=(0, 0)).covers(inner)
               for angle in range(0, 360, 15))


def rank_revisits(candidates, material, quota, max_n, target, alternatives):
    """Role-aware coarse ordering with an uncertainty slot, no hard10X rejection.

    Every region and all eight target polygons retain their exact relative
    geometry. This finite centred search supplies no robust-fit certificate or
    infeasibility claim. Other-material comparisons are bounded to eight public
    observed sources in stable-ID order; an untested source remains unknown.
    """
    contacts = target.get("contacts", [])
    if len(contacts) != 8:
        raise ValueError("Coarse device priority needs all eight registered contact regions")
    required = _centred(unary_union([Polygon(c["points_mm"]) for c in contacts]))
    hashes = {}
    other = []
    for sample in sorted(alternatives, key=lambda s: s["id"]):
        shape = sample.get("_observed_priority_region")
        shape = _geos(as_region(shape)) if shape is not None else coarse_region(sample, hashes)
        if shape is not None and not shape.is_empty:
            other.append(shape)
        if len(other) == 8:
            break
    roles = [r for r in quota if r.material is None or r.material == material]
    reasons = {}
    for sample in candidates:
        props = sample.get("props", {})
        label = props.get("labels", {}).get("10x") or props.get("thickness_label")
        relevant = [r for r in roles if not r.thickness_labels or label in r.thickness_labels]
        role_match = bool(relevant)
        area_match = any(props.get("area_um2", 0) >= r.min_area_um2 for r in relevant)
        shape = coarse_region(sample, hashes)
        target_hit = pair_hit = False
        if shape is not None and not shape.is_empty:
            if material == "Gr":
                target_hit = _potential(shape, required)
                pair_hit = target_hit and any(_potential(hbn, shape) for hbn in other)
            else:
                viable = [gr for gr in other if _potential(gr, required)]
                pair_hit = any(_potential(shape, gr) for gr in viable)
                target_hit = _potential(shape, required)
        if not role_match:
            tier, reason = 4, "coarse_label_provisional_mismatch"
        elif not area_match:
            tier, reason = 4, "coarse_area_below_role_preference"
        elif pair_hit:
            tier, reason = 0, "observed_target_and_pair_potential"
        elif target_hit:
            tier, reason = 1, "observed_eight_contact_potential_pair_unknown"
        elif shape is None:
            tier, reason = 2, "coarse_geometry_unavailable_pending"
        else:
            tier, reason = 3, "centred_finite_screen_miss_pending"
        reasons[sample["id"]] = {"tier": tier, "reason": reason, "role_label_match": role_match,
            "coarse_geometry_available": shape is not None, "target_potential": target_hit,
            "pair_potential": pair_hit, "admission": "pending20X_geometry_and100X_classification"}
    ordered = sorted(candidates, key=lambda s: (reasons[s["id"]]["tier"], s["id"]))
    uncertain = [s for s in ordered if reasons[s["id"]]["tier"] >= 2]
    # Preserve per-role slots, reserving one available slot for uncertainty so
    # missing10X geometry or an incomplete finite screen cannot starve forever.
    capacity = max_n - 1 if max_n > 1 and uncertain else max_n
    chosen = []
    taken = set()
    for role in roles:
        count = 0
        for sample in ordered:
            label = sample["props"].get("labels", {}).get("10x") or sample["props"].get("thickness_label")
            if (sample["id"] not in taken and (not role.thickness_labels or label in role.thickness_labels)
                    and sample["props"].get("area_um2", 0) >= role.min_area_um2):
                if len(chosen) >= capacity or count >= role.slots:
                    break
                chosen.append(sample)
                taken.add(sample["id"])
                count += 1
    if uncertain and max_n > 1:
        first = next((s for s in uncertain if s["id"] not in taken), None)
        if first is not None:
            chosen.append(first)
            taken.add(first["id"])
    for sample in ordered:
        if len(chosen) >= max_n:
            break
        if sample["id"] not in taken:
            chosen.append(sample)
            taken.add(sample["id"])
    return chosen[:max_n], {"version": "observed-coarse-device-priority-v1",
        "scope": "advisory15degree_centred_potential_only_no_rejection_or_admission",
        "other_material_sources_checked": len(other), "selected_ids": [s["id"] for s in chosen[:max_n]],
        "candidates": reasons}
