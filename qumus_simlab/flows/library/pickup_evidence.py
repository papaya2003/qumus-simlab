"""Shared image evidence for selection and exceptional source revisits."""
from __future__ import annotations

import math

from qumus_simlab.flows.library.primitives import MICROSCOPE, analyzer, autofocus


def _observed_shape(detection):
    """Observed region in exposure axes; never use inventory or hidden outlines."""
    from shapely.affinity import translate
    from shapely.geometry import MultiPolygon, Polygon

    from qumus_simlab.core.regions import as_region

    meta = detection.meta
    if not meta.get("outline_complete"):
        raise ValueError("incomplete_observed_boundary")
    region = as_region(meta.get("region_mm", meta.get("outline_mm", [])))
    shape = MultiPolygon([Polygon(c.exterior, c.holes) for c in region.components])
    if shape.is_empty:
        raise ValueError("empty_observed_boundary")
    return translate(shape, detection.dx_mm, detection.dy_mm)


def _neighbour_shape(detection):
    """Observed pixels of another flake, complete or cut by the frame edge.

    A cut neighbour's unseen part lies outside the image. The caller has
    already checked that the target plus the margin lies inside the image,
    so that unseen part is farther than the margin from the target."""
    from shapely.affinity import translate
    from shapely.geometry import MultiPolygon, Polygon

    from qumus_simlab.core.regions import as_region

    meta = detection.meta
    if meta.get("outline_complete"):
        return _observed_shape(detection)
    data = meta.get("region_mm") or meta.get("visible_region_mm")
    if not data:
        raise ValueError("unobserved_neighbour_boundary")
    region = as_region(data)
    shape = MultiPolygon([Polygon(c.exterior, c.holes) for c in region.components])
    if shape.is_empty:
        raise ValueError("empty_observed_boundary")
    return translate(shape, detection.dx_mm, detection.dy_mm)


def _local_clearance(detections, fov, margin):
    """Edge-to-edge clearance; no contour dilation or stamp-wide exclusion."""
    try:
        target = _observed_shape(detections[0])
    except (ValueError, TypeError, AttributeError):
        return "observed_boundary_unavailable", []
    x0, y0, x1, y1 = target.bounds
    if (x0-margin < -fov[0]/2 or x1+margin > fov[0]/2
            or y0-margin < -fov[1]/2 or y1+margin > fov[1]/2):
        return "local_neighbourhood_outside_image", []
    try:
        others = [_neighbour_shape(d) for d in detections[1:]]
    except (ValueError, TypeError, AttributeError):
        return "observed_boundary_unavailable", []
    return None, [target.distance(other) for other in others]


def capture_neighbourhood(ctx, flake_id: str, chip_id: str, material: str, *, local: bool = True) -> dict:
    scope = ctx.gate.microscope(MICROSCOPE)
    previous = scope.get_objective()
    policy = ctx.verification.get("pickup", {})
    radius = policy.get("local_clearance_mm", 0.020) if local else policy.get("inspection_radius_mm", 0.85)
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Pickup neighbourhood distance must be positive and finite")
    objective = 20 if local else 5
    counts, distances, images, diagnostics = [], [], [], []
    try:
        scope.set_objective(objective)
        autofocus(ctx, objective)
        fov = [float(v) * 10/objective for v in ctx.bench["microscope"]["fov_mm_10x"]]
        for _ in range(3):
            frame = scope.capture(purpose="analysis_capture" if local else "neighbourhood_capture")
            an = analyzer(ctx)
            # The 20X view is centred on the chosen flake, as v1's revisit
            # capture is, so the RGB recognizer takes v1's revisit area
            # ceiling (10% of the frame) and still lists every neighbour.
            # The scan ceiling (100000 px, 3300 um2 at 20X) dropped the
            # largest monolayers, which then read as "target_not_centred".
            detections = (an.find_flakes(frame, material, centered=True)
                          if local and getattr(an, "recognition_backend", None) == "v1_lab_rgb"
                          else an.find_flakes(frame, material))
            detections.sort(key=lambda d: math.hypot(d.dx_mm, d.dy_mm))
            image = frame.saved_path()
            if image:
                images.append(image)
            # Native captures have their own calibrated sensor field. The
            # nominal bench field must not certify pixels outside that image.
            fov = frame.meta.get("fov_mm", fov)
            valid_fov = (isinstance(fov, (list, tuple)) and len(fov) == 2
                         and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in fov))
            reason = ("invalid_capture_field" if not valid_fov else
                      "target_not_detected" if not detections else
                      "target_not_centred" if math.hypot(detections[0].dx_mm, detections[0].dy_mm) > 0.05 else
                      "inspection_radius_outside_image" if not local and (min(fov) / 2 < radius
                      or abs(detections[0].dx_mm) + radius > fov[0] / 2
                      or abs(detections[0].dy_mm) + radius > fov[1] / 2) else None)
            neighbours = []
            if reason is None:
                if local:
                    reason, neighbours = _local_clearance(detections, fov, radius)
                else:
                    neighbours = [math.hypot(d.dx_mm, d.dy_mm) for d in detections[1:]]
            diagnostics.append({"reason": reason, "detections": len(detections),
                                "image_size_px": [frame.width, frame.height], "fov_mm": fov})
            if reason:
                counts.append(-1)
                continue
            counts.append(sum(d <= radius for d in neighbours))
            distances.extend(neighbours)
        return {"flake_id": flake_id, "source_chip_id": chip_id,
            "clear": len(counts) == 3 and all(n == 0 for n in counts) and bool(images),
            "neighbours_per_view": counts,
            **({"policy": "local_flake_edge_clearance_v1", "required_edge_clearance_mm": radius,
                "distance_basis": "observed_region_edge_to_edge"} if local else
               {"policy": "legacy_stamp_radius", "clear_radius_mm": radius,
                "distance_basis": "view_centre_to_other_centroid"}),
            "assessment_status": "unknown" if any(n < 0 for n in counts) else
                                 "occupied" if any(n > 0 for n in counts) else "clear",
            "detection_scope": "Requested material classifier; not a universal debris detector",
            "nearest_other_mm": min(distances) if distances else None,
            "source": "focused_optical_neighbourhood", "timestamp": ctx.clock.now(),
            "images": images, "objective": objective, "fov_mm": fov, "job_id": ctx.run_id,
            "views": diagnostics,
            "scope": "simulated" if ctx.store.bench == "sim" else "camera"}
    finally:
        scope.set_objective(previous)
        autofocus(ctx, previous)
