"""Synthetic optical assumptions and electrode geometry.

This illustrative six-contact pattern is a testbench fixture, not a mask design
or a calibrated photograph. Both the stock image and the moving microscope
project exactly these polygons, so the image follows the chip's pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

WHITE_LIGHT_PROFILE = "synthetic-white-light-air-gap-v2"
WHITE_LIGHT_MAX_GAP_MM = .008
WHITE_LIGHT_TABLE_SIZE = 2049
LEGACY_STAMP_STRUCTURE_PROFILE = "synthetic-glass-pdms-pc-paper-v1"
STAMP_STRUCTURE_PROFILE = "synthetic-glass-pdms-pc-paper-v2"
# Illustrative white illumination and Gaussian RGB responses in wavenumber.
# These centres/widths, contrast, display gamma and opacity are NOT measured
# source/camera/PC properties. No PC thickness or multilayer reflectance is fit.
_RGB_BANDS_NM = ((610., 45.), (540., 40.), (460., 35.))
# Fringe coherence relative to those bands: 2x longer, so rings first show at
# about a 2.8 um apex gap (1.4 um at 1x). INFERRED, not measured: the owner
# judged the 1x rings to appear too late against the real stacking scope
# (2026-11-04, after seeing 1x/3x/4x renders, chose 2x).
WHITE_LIGHT_COHERENCE_SCALE = 2.0


@dataclass(frozen=True)
class StampStructure:
    """Exposure-frozen illustrative support geometry, never optical metrology.

    Cap radius is its circular plan-view footprint, not air-gap curvature.
    The diamond aperture's radius is its inradius (centre-to-edge distance),
    giving axis-aligned vertices at sqrt(2) times that distance. Aperture,
    cap centre and observed first-contact centre remain distinct fields even
    when this synthetic fixture makes the first two concentric.
    """

    glass_outline_mm: tuple[tuple[float, float], ...]
    cap_center_mm: tuple[float, float]
    cap_radius_mm: float
    aperture_center_mm: tuple[float, float]
    aperture_radius_mm: float
    aperture_shape: str = "circle"
    aperture_radius_semantics: str = "radius"

    @classmethod
    def from_packet(cls, packet: dict[str, Any]) -> StampStructure:
        profile = packet.get("profile_id")
        if (profile not in (LEGACY_STAMP_STRUCTURE_PROFILE, STAMP_STRUCTURE_PROFILE)
                or packet.get("source") != "synthetic"
                or packet.get("frame") != "camera_center_axes" or packet.get("units") != "mm"):
            raise ValueError("unsupported synthetic stamp-structure profile/frame")
        if profile == LEGACY_STAMP_STRUCTURE_PROFILE:
            if "aperture_shape" in packet or "aperture_radius_semantics" in packet:
                raise ValueError("legacy stamp structure has an implicit circular aperture")
            shape, semantics = "circle", "radius"
        else:
            shape, semantics = packet.get("aperture_shape"), packet.get("aperture_radius_semantics")
            if shape != "diamond" or semantics != "inradius":
                raise ValueError("stamp structure v2 requires an explicit diamond aperture and inradius")

        def number(value):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError("stamp-structure coordinates must be finite")
            return float(value)

        def point(value):
            if not isinstance(value, list | tuple) or len(value) != 2:
                raise ValueError("stamp-structure point must have two coordinates")
            return tuple(number(v) for v in value)

        outline = packet.get("glass_outline_mm")
        if not isinstance(outline, list | tuple) or len(outline) != 4:
            raise ValueError("stamp-structure glass needs four corners")
        corners = tuple(point(p) for p in outline)
        area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(corners, (*corners[1:], corners[0]), strict=True))
        if abs(area) < 1e-12:
            raise ValueError("stamp-structure glass outline is degenerate")
        cap_radius, aperture_radius = number(packet.get("cap_radius_mm")), number(packet.get("aperture_radius_mm"))
        if cap_radius <= 0 or aperture_radius <= 0:
            raise ValueError("stamp-structure radii must be positive")
        return cls(corners, point(packet.get("cap_center_mm")), cap_radius,
                   point(packet.get("aperture_center_mm")), aperture_radius, shape, semantics)


def stamp_structure_packet(bench: dict[str, Any], config: dict[str, Any],
                           head_xy: tuple[float, float], camera_xy: tuple[float, float]) -> dict[str, Any] | None:
    """Project declared slide/cap/aperture geometry using the actual holder pose.

    No default is inferred for legacy configurations without this explicit
    synthetic profile. These plan dimensions do not change contact mechanics.
    """
    if not config:
        return None
    slide = bench["stamp_slide"]
    width, height = slide["size"]
    hx, hy = head_xy[0]-camera_xy[0], head_xy[1]-camera_xy[1]
    left, right = hx-slide["dome_inset_mm"], hx-slide["dome_inset_mm"]+width
    dx, dy = config["aperture_offset_mm"]
    packet = {"profile_id": config["profile_id"], "source": "synthetic",
              "frame": "camera_center_axes", "units": "mm",
              "glass_outline_mm": [[left, hy-height/2], [right, hy-height/2],
                                   [right, hy+height/2], [left, hy+height/2]],
              "cap_center_mm": [hx, hy], "cap_radius_mm": config["cap_radius_mm"],
              "aperture_center_mm": [hx+dx, hy+dy], "aperture_radius_mm": config["aperture_radius_mm"]}
    if config.get("profile_id") == STAMP_STRUCTURE_PROFILE:
        packet.update(aperture_shape=config["aperture_shape"],
                      aperture_radius_semantics=config["aperture_radius_semantics"])
    StampStructure.from_packet(packet)
    return packet


def white_light_response(gap_mm: float) -> tuple[float, float, float, float]:
    """Premultiplied RGB/opacity for a synthetic broadband air-gap reflection.

    Gaussian integration of cos(4*pi*gap/lambda + pi) in wavenumber gives
    smooth phase and coherence decay, without finite-spectrum alias revivals.
    Geometry/time come from the caller; this pure function has no animation.
    """
    if not math.isfinite(gap_mm) or gap_mm < 0:
        raise ValueError("air-gap response requires a finite nonnegative gap")
    phase = 4*math.pi*gap_mm*1e6  # mm -> nm
    envelopes = [math.exp(-.5*(phase*width/WHITE_LIGHT_COHERENCE_SCALE/centre**2)**2)
                 for centre, width in _RGB_BANDS_NM]
    alpha = .45*max(envelopes)
    colors = [255*(.65+.35*envelope*math.cos(phase/centre+math.pi))**(1/2.2)
              for (centre, _), envelope in zip(_RGB_BANDS_NM, envelopes, strict=True)]
    return *(color*alpha for color in colors), alpha


@lru_cache(maxsize=1)
def white_light_table() -> tuple[tuple[float, float, float, float], ...]:
    """One bounded, immutable profile table; interpolate before RGB8 output."""
    return tuple(white_light_response(WHITE_LIGHT_MAX_GAP_MM*i/(WHITE_LIGHT_TABLE_SIZE-1))
                 for i in range(WHITE_LIGHT_TABLE_SIZE))


def electrode_pattern() -> list[dict]:
    shapes = []
    for sign in (-1, 1):
        for y in (-0.035, 0, 0.035):
            # Contact pad -> tapered lead -> inner contact. Keep an optical
            # channel between the opposite contact tips.
            pts = [(sign*.014, y-.002), (sign*.052, y-.006),
                   (sign*.070, y-.006), (sign*.070, y+.006),
                   (sign*.052, y+.006), (sign*.014, y+.002)]
            shapes.append({'kind': 'contact', 'points': pts})
    shapes.append({'kind': 'gate', 'points': [(-.009,-.057),(.009,-.057),(.009,.057),(-.009,.057)]})
    for index, shape in enumerate(shapes):
        shape["id"] = f"{shape['kind']}_{index}"
    return shapes


def electrode_geometry() -> dict:
    """Dimensions of the centre pair, derived from the same synthetic polygons."""
    pattern = electrode_pattern()
    left, right = pattern[1], pattern[4]
    lx, rx = max(x for x, _ in left["points"]), min(x for x, _ in right["points"])
    tip = [y for x, y in right["points"] if x == rx]
    return {"source": "simulated_design", "reference": "testbench/optics.py:electrode_pattern centre pair",
            "units": "um", "contact_gap_um": (rx - lx) * 1000,
            "contact_span_um": (max(tip) - min(tip)) * 1000,
            "contact_depth_um": (max(x for x, _ in right["points"]) - rx) * 1000,
            "contact_regions": [{"region_id": "centre_pair", "units": "mm", "frame": "chip_center_axes",
                "anchor_mm": [0, 0], "contact_pair": [left["id"], right["id"]],
                "polygons": [{"id": p["id"], "kind": p["kind"], "points_mm": p["points"]} for p in (left, right)]}]}
