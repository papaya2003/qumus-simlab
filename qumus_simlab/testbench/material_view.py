"""Pure pair observation and synthetic rendering for reactor integration.

The caller supplies chip-local view centre, FOV, chip-to-camera rotation and
observed focus readiness. Stored outlines always remain in chip coordinates.
Nothing here starts exposure, writes a ledger, consumes a source or moves hardware.
"""

from __future__ import annotations

import io
import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from qumus_simlab.core.polygons import Point, Polygon, transform
from qumus_simlab.testbench.material_flakes import PairAssembly, Placement, project_outline
from qumus_simlab.testbench.material_reaction import LocalReaction


def assembly_from_manifest(manifest: dict, samples: list[dict], *, chip_size_mm: Point) -> PairAssembly:
    """Bridge the reactor v1 preparation manifest and explicit material placements.

    Each input sample's props.material_placement is Placement.model_dump(mode="json").
    The Placement.flake.id must be the catalog/ledger ID, not a substituted sim ID.
    Reactor still owns kind/location/reservation/consumption validation.
    """
    if manifest.get("schema_version") != 1:
        raise ValueError("only reactor_pair schema_version 1 is supported")
    ids = manifest.get("input_flake_ids")
    if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("manifest must name exactly two distinct flake IDs, bottom then top")
    if len(samples) != 2 or {s["id"] for s in samples} != set(ids):
        raise ValueError("sample identities must exactly match the preparation")
    by_id = {s["id"]: s for s in samples}
    placements = []
    for fid in ids:
        sample = by_id[fid]
        props = sample["props"]
        placed = Placement.model_validate(props["material_placement"])
        if placed.flake.id != fid or placed.flake.material != props["material"]:
            raise ValueError("material placement identity must match the ledger")
        # Catalog area is an optical estimate. The transfer binding records
        # exact physical geometry separately; historical fixtures used area_um2.
        area = float(props.get("geometry_area_um2", props["area_um2"]))
        if not math.isfinite(area) or area <= 0 or not math.isclose(area, placed.flake.area_um2, rel_tol=1e-6):
            raise ValueError("ledger geometry area must match polygon area")
        placements.append(placed)
    return PairAssembly(id=manifest["preparation_id"], target_chip_id=manifest["chip_id"],
                        chip_size_mm=chip_size_mm, inputs=placements)


def camera_outline(outline: Polygon, center_mm: Point, rotation_deg: float) -> Polygon:
    """Camera-centred mm: translate in the chip frame, then rotate into optics."""
    return transform(transform(outline, xy=(-center_mm[0], -center_mm[1])), rotation_deg=rotation_deg)


def pair_sensor_rect(fov_mm: Point, image_size: tuple[int, int] = (640, 420)) -> tuple[int, int, int, int]:
    """Sensor x/y/width/height, letterboxed outside header and scale annotations."""
    width, height = image_size
    if width < 32 or height <= 112 or not all(math.isfinite(v) and v > 0 for v in fov_mm):
        raise ValueError("pair image needs positive FOV and space for the annotation margins")
    scale = min((width - 24) / fov_mm[0], (height - 112) / fov_mm[1])
    fw, fh = (max(1, math.floor(length * scale)) for length in fov_mm)
    return (width - fw) // 2, 64 + (height - 112 - fh) // 2, fw, fh


def _project_sensor(outline: Polygon, fov_mm: Point, image_size: tuple[int, int]) -> dict:
    x, y, width, height = pair_sensor_rect(fov_mm, image_size)
    projected = project_outline(outline, center_mm=(0, 0), fov_mm=fov_mm, image_size=image_size)
    # Keep calibrated sensor coordinates unchanged for acquisition/motion.
    # The annotated PNG has a separate, explicit display transform.
    return {**projected, "display_outline_px": tuple(
        (x + px * width / image_size[0], y + py * height / image_size[1])
        for px, py in projected["outline_px"])}


def observe_pair(reaction: LocalReaction, *, center_mm: Point = (0, 0),
                 fov_mm: Point = (0.13, 0.086), in_focus: bool = True,
                 camera_rotation_deg: float = 0,
                 image_size: tuple[int, int] = (640, 420)) -> dict[str, Any]:
    """Read-only render payload plus reactor contract; never advance on capture."""
    if type(in_focus) is not bool:
        raise ValueError("in_focus must be boolean")

    def project(outline: Polygon) -> dict:
        camera = camera_outline(outline, center_mm, camera_rotation_deg)
        return {**_project_sensor(camera, fov_mm, image_size),
                "camera_outline_mm": camera}

    obs = reaction.observe()
    inputs = [{**f, **project(f["outline_mm"])} for f in obs["inputs"]]
    overlap = project(obs["overlap_mm"])
    product = obs["product"]
    product_view = None
    if product:
        product_view = {**product, **project(product["outline_mm"]),
                        "geometry": {"schema_version": 1, "frame": "chip_local_mm",
                                     "outline_mm": product["outline_mm"]},
                        "state": {"status": obs["status"], "conversion_of_overlap": obs["conversion_of_overlap"],
                                  "model_version": obs["model_version"]}}
    visible = in_focus and (overlap["visible"] if obs["overlap_area_um2"] > 0
                            else all(f["visible"] for f in inputs))
    product_visible = visible and product_view is not None and product_view["fully_visible"]
    # Failed/partial morphology remains inspectable in render_product, but only
    # completed, focused observations offer a product for ledger acceptance.
    verified = product_visible and obs["status"] == "complete"
    return {**obs, "experiment_kind": "two_flake_reaction", "preparation_id": obs["assembly_id"],
            "input_flake_ids": [f["id"] for f in inputs], "visible": visible,
            "reason": None if visible else "out_of_focus" if not in_focus else "overlap_outside_field",
            "view_center_mm": center_mm, "fov_mm": fov_mm, "inputs": inputs,
            "frame_size_px": image_size, "sensor_rect_px": pair_sensor_rect(fov_mm, image_size),
            "sensor_size_px": image_size, "outline_px_frame": "sensor",
            "display_outline_px_frame": "annotated_image",
            "camera_rotation_deg": camera_rotation_deg, "in_focus": in_focus,
            "overlap_view": overlap, "render_product": product_view,
            "product": product_view if verified else None,
            "conversion_of_overlap": obs["conversion_of_overlap"] if visible else None,
            "reaction_fraction": obs["conversion_of_overlap"] if visible else None,
            "product_verified": bool(verified),
            "input_geometry_role": "provenance_only" if obs["conversion_of_overlap"] == 1 else "reactant"}


def render_pair(view: dict, *, size: tuple[int, int] | None = None) -> bytes:
    """Full scientific field plus separate annotations; no photo realism claim.

    Only the sensor layer receives specimen pixels. Clipping therefore cannot
    paint into annotations, and no annotation can cover a specimen at an edge.
    """
    size = size or tuple(view.get("frame_size_px", (640, 420)))
    x0, y0, width, height = pair_sensor_rect(view["fov_mm"], size)
    im = Image.new("RGB", size, "#182235")
    sensor = Image.new("RGB", (width, height), "#333d58")
    draw = ImageDraw.Draw(sensor)
    font = ImageFont.load_default(size=14)
    colors = ("#518eb5", "#ab88ba")
    # An unavailable optical image must not show a sharp model-truth specimen.
    focused = view.get("in_focus", view.get("reason") != "out_of_focus")
    consumed = view.get("input_geometry_role") == "provenance_only" or view["status"] == "complete"
    visible_inputs = () if not focused or consumed else view["inputs"]

    def project(f: dict) -> dict:
        camera = camera_outline(f["outline_mm"], view["view_center_mm"], view.get("camera_rotation_deg", 0))
        projected = _project_sensor(camera, view["fov_mm"], size)
        return {**projected, "outline_px": tuple((px - x0, py - y0)
                                                 for px, py in projected["display_outline_px"])}

    for f, color in zip(visible_inputs, colors, strict=False):
        projected = project(f)
        draw.polygon(projected["outline_px"], fill=color, outline="#d2dfed")
    product = view["render_product"]
    if product and focused:
        projected = project(product)
        draw.polygon(projected["outline_px"], fill="#dfbb64", outline="#fff0bc")
    im.paste(sensor, (x0, y0))
    header = Image.new("RGB", (size[0], 64), "#182235")
    draw = ImageDraw.Draw(header)
    draw.text((12, 6), "SYNTHETIC | Two-flake local reaction", fill="#ffffff", font=font)
    caption = (f"{view['status']} | synthetic overlap conversion {view['conversion_of_overlap']:.0%}"
               if view["visible"] and view.get("conversion_of_overlap") is not None
               else "Observation unavailable | conversion unmeasured")
    draw.text((12, 24), caption, fill="#d9e2ee", font=font)
    if not view["visible"]:
        draw.text((12, 44), f"Not verified: {view['reason']}", fill="#ffd29a", font=font)
    im.paste(header, (0, 0))
    footer = Image.new("RGB", (size[0], 48), "#182235")
    draw = ImageDraw.Draw(footer)
    scale_um = view["fov_mm"][0] * 1000 / 5
    draw.line((x0, 32, x0 + width / 5, 32), fill="white", width=3)
    draw.text((x0, 10), f"{scale_um:g} um", fill="white", font=font)
    im.paste(footer, (0, size[1] - 48))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()
