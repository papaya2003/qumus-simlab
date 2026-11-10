"""Synthetic contact-interior evidence from calibrated original RGB exposures.

The observed front bounds the domain; it never fills that domain. Every supported
cell must fit a contact observation and exclude the entire conservative unchanged
hypothesis, including fractional registration and unknown carried hBN projection.
No world, controller, renderer state, hidden layer count, mask, gap or radius is read.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Literal, TypedDict

import numpy as np

from qumus_simlab.core.materials import _HBN_BINS_NM, normalize_thickness_label
from qumus_simlab.core.optical_profiles import (
    CAMERA_RESPONSE_PROFILE,
    CHIP_EDGE_ILLUMINATION_MM,
    CHIP_MAX_DOWNSAMPLE,
    CONTACT_INTERIOR_PROFILE,
    HBN_LABEL_NM_PER_LAYER,
    STAMP_PSF_PROFILE,
    SYNTHETIC_CONTACT_PHOTOMETRY,
    SYNTHETIC_HBN_SLAB,
    SYNTHETIC_OFF_CHIP_RGB,
    flake_tint,
)
from qumus_simlab.perception.tracking import (
    PixelCalibration,
    _whole_outline_cell_coverage,
    _whole_region_cell_coverage,
    contact_front_domain,
    load_pixels,
)

LIMITS = {"contact_rgb": 9., "contact_chroma": 4.5, "unchanged_rgb": 10., "unchanged_chroma": 4.2}


_FRONT_ANGULAR_SUPPORT = .9


class SourceProjectionCertificate(TypedDict):
    method: str
    status: Literal["admitted", "bound_failed", "insufficient_observation"]
    admitted: bool
    reason: str | None
    observed_front_angular_coverage: float
    required_front_angular_coverage: float
    elapsed_s: float
    z_total_variation_mm: float
    z_endpoint_delta_mm: float
    apex_displacement_norm_bound_mm: float
    source_branch_distance_bound_mm: float | None
    dome_radius_mm: float
    source_branch_stays_eligible: bool | None
    projection_displacement_bound_mm: float | None
    native_pixel_bound_mm: float
    scope: str


@dataclass(frozen=True)
class InteriorCalibration:
    """Carried-stamp forward law; observed source labels are nm bands.

    Native source-chip RGB recognition uses a separate image-formation profile.
    Its observed classes do not change this existing slab-alpha/flake-tint law.
    """
    labels_nm: tuple[str, ...] = ()
    profile: str = CONTACT_INTERIOR_PROFILE
    camera_response: str = CAMERA_RESPONSE_PROFILE
    contact: str = SYNTHETIC_CONTACT_PHOTOMETRY.calibration_id
    carried_hbn: str = SYNTHETIC_HBN_SLAB.calibration_id
    raster: str = STAMP_PSF_PROFILE
    source_mode: str = "synthetic_complete_outline_flat_tint_v1"

    def generators(self):
        expected = InteriorCalibration(self.labels_nm)
        if self != expected:
            raise ValueError("Unsupported contact-interior calibration/profile")
        bands = []
        for label in self.labels_nm:
            if normalize_thickness_label("hBN", label) != label:
                raise ValueError("Unknown carried hBN nanometre classification")
            band = next((b for name, *b in _HBN_BINS_NM if name == label), None)
            if band is None or not math.isfinite(band[1]):
                raise ValueError("Unbounded carried hBN thickness is unobservable")
            bands.append(band)
        layers = [n for n in range(1, math.ceil(max((hi for _, hi in bands), default=0)/HBN_LABEL_NM_PER_LAYER)+1)
                  if any(lo <= n*HBN_LABEL_NM_PER_LAYER < hi for lo, hi in bands)]
        return [(0., (0, 0, 0)), *[(SYNTHETIC_HBN_SLAB.projection_weight(n), flake_tint("hBN", n)) for n in layers]]

    def snapshot(self):
        self.generators()
        return {**asdict(self), "labels_nm": list(self.labels_nm)}


def _projections(values):
    return np.concatenate((values, np.stack((values[:, 0]-values[:, 1],
                                            values[:, 2]-values[:, 1]), axis=1)), axis=1)


def _joint_mixture_feasible(target, points, errors) -> bool:
    """True when some convex mixture of `points` lies within `errors` of
    `target` in every projection at once. Solver failure counts as feasible,
    so an unresolved cell is never presented as excluded."""
    from scipy.optimize import linprog

    n = len(points)
    upper = np.vstack((points.T, -points.T))
    bound = np.concatenate((target+errors, -(target-errors)))
    result = linprog(np.zeros(n), A_ub=upper, b_ub=bound, A_eq=np.ones((1, n)), b_eq=[1.],
                     bounds=[(0, None)]*n, method="highs")
    return result.status != 2


def contact_interior_evidence(reference, current, calibration: PixelCalibration,
                              outline_px, *, margin_mm: float, optical: InteriorCalibration):
    """Strict complete-body evidence; unknown cells and pockets remain excluded."""
    generators = optical.generators()
    signed = (calibration.mm_per_pixel_x, calibration.mm_per_pixel_y)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v == 0 for v in signed):
        raise ValueError("Contact interior needs finite nonzero signed native pixel scales")
    sx, sy = (abs(value) for value in signed)
    if any(v <= 0 for v in (sx, sy)):
        raise ValueError("Contact interior needs finite nonzero signed native pixel scales")
    before, after = load_pixels(reference), load_pixels(current)
    if before.size != after.size or min(before.size) < 64:
        raise ValueError("Contact interior requires identical finite native rasters")
    width, height = before.size
    base = {"method": CONTACT_INTERIOR_PROFILE, "optical_calibration": optical.snapshot(),
            "covered": False, "found": False, "coverage_is_transfer_proof": False,
            "holes_filled": False, "required_margin_mm": margin_mm,
            "registration_guard_mm": math.hypot(sx, sy),
            "scope": "Synthetic optical interior evidence only; no physical transfer or real calibration"}
    front, observed_bounds = contact_front_domain(after)
    base["front"] = {**asdict(front), "centroid_px": list(front.centroid_px) if front.centroid_px else None,
                     "boundary_diagnostics": list(front.boundary_diagnostics) if front.boundary_diagnostics else None}
    base["observed_front_cell_bounds_px"] = list(observed_bounds) if observed_bounds else None
    if (not front.has_contact or front.failure_reason == "boundary_clipped"
            or front.film_intact is False or front.centroid_px is None or front.radius_px is None):
        return {**base, "reason": "No complete supported observed contact front", "supported_pixels": 0,
                "unknown_pixels": width*height}
    radius = front.radius_px-front.interface_uncertainty_px-.5
    if not math.isfinite(radius) or radius <= 0:
        return {**base, "reason": "Observed front uncertainty leaves no positive interior", "supported_pixels": 0,
                "unknown_pixels": width*height}
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    yy, xx = np.mgrid[:height, :width]
    gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
    gamma = max(float(np.max(gain[1:-1, 1:-1]/gain[1+dy:height-1+dy, 1+dx:width-1+dx]))
                for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    q = 1/(1-max(alpha for alpha, _ in generators))
    optics = SYNTHETIC_CONTACT_PHOTOMETRY
    alpha = (1-optics.front_opacity)*(1-optics.interior_opacity)
    offset = (np.array(optics.front_color)*optics.front_opacity*(1-optics.interior_opacity)
              +np.array(optics.interior_color)*optics.interior_opacity)
    gmax = float(gain.max())
    bounds = {"contact_rgb": 1.72*gmax+gmax+max(1., alpha*q*gamma)+3*(1+alpha*q*gamma),
              "contact_chroma": 1.72*gmax+gmax*(1+alpha*q)+1+alpha*q*gamma,
              "unchanged_rgb": max(gmax, q*gmax)+max(1., q*gamma)+3*(1+q*gamma),
              "unchanged_chroma": gmax*(1+q)+1+q*gamma}
    base["camera_bounds"] = {"gain_ratio_max": gamma, "inverse_projection_max": q,
                             "derived_dn": bounds, "limits_dn": LIMITS}
    if any(not math.isfinite(value) or value > LIMITS[key] for key, value in bounds.items()):
        return {**base, "reason": "Optical uncertainty exceeds the frozen recognition bounds",
                "supported_pixels": 0, "unknown_pixels": width*height}
    cx, cy = front.centroid_px
    domain = ((xx-cx)**2+(yy-cy)**2 <= radius**2) & (xx >= 8) & (xx < width-8) & (yy >= 30) & (yy < height-32)
    observed_domain = domain.copy()
    # Every admitted neighbour must satisfy the unclipped camera model. A
    # clipped reference alternative cannot be silently dropped from the null.
    clipped = (np.min(a, axis=2) <= 0) | (np.max(a, axis=2) >= 255)
    padded = np.pad(clipped, ((1, 1), (1, 1)), constant_values=True)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            domain &= ~padded[1+dy:1+dy+height, 1+dx:1+dx+width]
    domain &= (np.min(b, axis=2) > 0) & (np.max(b, axis=2) < 255)
    indices = np.flatnonzero(domain)
    iy, ix = indices//width, indices % width
    current_rgb, g = b[iy, ix], gain[iy, ix, None]
    target = _projections(current_rgb)
    contact = np.zeros(len(indices), dtype=bool)
    baselines = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            r = a[iy+dy, ix+dx]*(gain[iy, ix]/gain[iy+dy, ix+dx])[:, None]
            baselines.append(_projections(r))
            prediction = _projections(alpha*r+g*offset)
            # Sufficient consistency subset. It need not recover every contact
            # pixel; zero local carried support is allowed by the public hull.
            contact |= ((np.max(abs(target[:, :3]-prediction[:, :3]), axis=1) <= LIMITS["contact_rgb"])
                        & (np.max(abs(target[:, 3:]-prediction[:, 3:]), axis=1) <= LIMITS["contact_chroma"]))
    pooled = np.stack(baselines)
    ref_lo, ref_hi = np.min(pooled, axis=0), np.max(pooled, axis=0)
    underlying_lo, underlying_hi = ref_lo.copy(), ref_hi.copy()
    profiles = []
    for opacity, color in generators:
        projected = _projections(g*np.array(color))
        profiles.append((opacity, projected))
        underlying_lo = np.minimum(underlying_lo, (ref_lo-opacity*projected)/(1-opacity))
        underlying_hi = np.maximum(underlying_hi, (ref_hi-opacity*projected)/(1-opacity))
    null_lo, null_hi = underlying_lo.copy(), underlying_hi.copy()
    for opacity, color in profiles:
        null_lo = np.minimum(null_lo, (1-opacity)*underlying_lo+opacity*color)
        null_hi = np.maximum(null_hi, (1-opacity)*underlying_hi+opacity*color)
    errors = np.array([LIMITS["unchanged_rgb"]]*3+[LIMITS["unchanged_chroma"]]*2)
    # Inside all slabs means only 'unchanged not excluded'. The full independent
    # profile hull and ALL fractional registration mixtures are conservatively
    # included. No per-channel values are presented as a joint witness.
    unchanged = np.all((target >= null_lo-errors) & (target <= null_hi+errors), axis=1)
    # Edge refinement (#136, owner decision A). The slabs above bound each
    # channel separately, so beside any reference edge a contact colour lies
    # inside the box although no single mixture of the unchanged alternatives
    # reproduces all channels at once. A box-ambiguous, contact-consistent cell
    # stays 'unchanged not excluded' unless the joint test rules out EVERY
    # convex (fractional registration) mixture of every alternative.
    ambiguous = np.flatnonzero(contact & unchanged)
    edge_resolved = 0
    for k in ambiguous:
        alternatives = [(1-o2)*(ref[k]-o1*p1[k])/(1-o1)+o2*p2[k]
                        for ref in baselines for o1, p1 in profiles for o2, p2 in profiles]
        if not _joint_mixture_feasible(target[k], np.stack(alternatives), errors):
            unchanged[k] = False
            edge_resolved += 1
    supported = contact & ~unchanged
    mask = np.zeros(width*height, dtype=np.uint8)
    mask[indices[supported]] = 1
    cells = set(np.flatnonzero(mask).tolist())
    coverage_check = _whole_region_cell_coverage if isinstance(outline_px, dict) else _whole_outline_cell_coverage
    # Owner decision B (#136, 2026-11-02): a neighbouring flake inside the
    # margin may be picked up with the source. The source body (plus the
    # registration guard) still needs strictly supported cells. The margin band
    # admits contact-consistent cells that only the unchanged hypothesis keeps
    # undecided (neighbour edges); model-unsupported cells (bubbles, debris,
    # no contact) still block, and holes are never filled.
    guard = base["registration_guard_mm"]
    body = coverage_check(cells, width, height, outline_px, sx, sy, guard)
    consistent = mask.copy()
    consistent[indices[contact]] = 1
    band = coverage_check(set(np.flatnonzero(consistent).tolist()), width, height, outline_px, sx, sy,
                          margin_mm+guard)
    coverage = {**band, "covered": bool(body["covered"] and band["covered"]),
                "reason": body["reason"] if not body["covered"] else band["reason"],
                "margin_rule": "strict_body_contact_consistent_margin_v1",
                "body_coverage": {k: v for k, v in body.items() if not k.startswith("target_")}}
    return {**base, **coverage, "found": True, "required_margin_mm": margin_mm,
            "effective_required_margin_mm": margin_mm+base["registration_guard_mm"],
            "supported_pixels": int(supported.sum()), "front_domain_pixels": int(observed_domain.sum()),
            "unknown_inside_front_pixels": int(observed_domain.sum())-int(supported.sum()),
            "ineligible_camera_pixels": int(observed_domain.sum())-len(indices),
            "ambiguous_contact_unchanged_pixels": int((contact & unchanged).sum()),
            "unchanged_rule": "channel_slabs_then_joint_mixture_v1",
            "edge_resolved_pixels": edge_resolved,
            "model_unsupported_pixels": int((~contact).sum()),
            "supported_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}


#: Non-contact pockets smaller than this many native pixels are not resolved.
CONTACT_POCKET_MIN_PX = 4


def _outline_rings(outline_px):
    if isinstance(outline_px, dict):
        return [component["exterior"] for component in outline_px["components"]]
    return [outline_px]


REGION_COVERAGE_PROFILE = "observed-contact-region-outline-margin-v1"


def contact_region_coverage(reference, current, calibration: PixelCalibration, outline_px, *, margin_mm: float,
                            unknown=None):
    """Coverage from the observed contact REGION, of any shape (#136, D13).

    Owner decision (Jialei, 2026-11-02): a real front need not be a circle
    (a tall neighbour, an imperfect PDMS dome), so no circle is fitted. Each
    pixel is classified against this exposure's own contact colour (declared
    transmission, offset learned from the dominant changed colour) and the
    unchanged reference, over all 3x3 registration shifts. A pixel is contact
    only when it matches the contact colour within half its own contact
    change and better than any unchanged colour; white-light fringes and
    bubbles therefore stay non-contact. Covered means every pocket of
    non-contact pixels inside the outline plus margin is smaller than
    CONTACT_POCKET_MIN_PX, and that whole region is in the observed support.
    No frozen absolute colour tolerance and no front shape are assumed.
    `unknown` marks camera cells whose contact state cannot be observed (an
    opaque load at release); they are never counted as contact or pocket.
    """
    from PIL import Image, ImageDraw
    from scipy import ndimage

    signed = (calibration.mm_per_pixel_x, calibration.mm_per_pixel_y)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v == 0 for v in signed):
        raise ValueError("Contact coverage needs finite nonzero signed native pixel scales")
    sx, sy = (abs(value) for value in signed)
    before, after = load_pixels(reference), load_pixels(current)
    if before.size != after.size or min(before.size) < 64:
        raise ValueError("Contact coverage requires identical finite native rasters")
    width, height = before.size
    guard = math.hypot(sx, sy)
    need = margin_mm+guard
    base = {"method": REGION_COVERAGE_PROFILE, "covered": False, "found": False,
            "coverage_is_transfer_proof": False, "holes_filled": False, "required_margin_mm": margin_mm,
            "registration_guard_mm": guard, "effective_required_margin_mm": need,
            "scope": "Synthetic relative contact-region evidence; no physical transfer"}
    front, observed_bounds = contact_front_domain(after)
    base["front"] = {**asdict(front), "centroid_px": list(front.centroid_px) if front.centroid_px else None,
                     "boundary_diagnostics": list(front.boundary_diagnostics) if front.boundary_diagnostics else None}
    base["observed_front_cell_bounds_px"] = list(observed_bounds) if observed_bounds else None
    rings = _outline_rings(outline_px)
    points = [point for ring in rings for point in ring]
    xs, ys = [x for x, _ in points], [y for _, y in points]
    region_box = [min(xs)-need/sx, min(ys)-need/sy, max(xs)+need/sx, max(ys)+need/sy]
    inside_view = (region_box[0] >= 8 and region_box[2] < width-8
                   and region_box[1] >= 30 and region_box[3] < height-32)
    base.update(region_box_px=region_box, region_inside_observed_support=inside_view)
    if not inside_view:
        return {**base, "reason": "Outline and margin are not inside the observed camera support"}
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    yy, xx = np.mgrid[:height, :width]
    support = ((xx >= 8) & (xx < width-8) & (yy >= 30) & (yy < height-32)
               & (np.min(a, axis=2) > 0) & (np.max(a, axis=2) < 255)
               & (np.min(b, axis=2) > 0) & (np.max(b, axis=2) < 255))
    # Declared contact photometry: the front disc, then the interior disc, are
    # composited over the reference, so contact = alpha*reference + offset.
    photometry = SYNTHETIC_CONTACT_PHOTOMETRY
    slope = (1-photometry.front_opacity)*(1-photometry.interior_opacity)
    # Declared camera vignette scales the added contact colour per pixel.
    gain = ((1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2))[..., None]
    offsets = (b-slope*a)/gain
    changed = support & (np.max(abs(b-a), axis=2) > 4)
    if changed.sum() < 64:
        return {**base, "reason": "No observed contact region"}
    # Dominant changed colour = the flat contact interior; fringes spread out.
    samples = offsets[changed]
    bins, counts = np.unique(np.round(samples/2).astype(int), axis=0, return_counts=True)
    mode = bins[np.argmax(counts)]*2.
    near = np.max(abs(samples-mode), axis=1) <= 4
    offset = np.median(samples[near], axis=0)
    noise = float(np.max(1.4826*np.median(abs(samples[near]-offset), axis=0)))
    if near.sum() < 64:
        return {**base, "reason": "No observed contact region"}
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for ring in rings:
        draw.polygon([tuple(p) for p in ring], fill=1, outline=1)
    body = np.asarray(canvas, dtype=bool)
    region = ndimage.distance_transform_edt(~body, sampling=(sy, sx)) <= need
    pad_a = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    shifted = [pad_a[1+dy:1+dy+height, 1+dx:1+dx+width] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    to_unchanged = np.min([np.max(abs(b-ref), axis=2) for ref in shifted], axis=0)

    def classify(segments):
        to_contact, change = np.full((height, width), np.inf), np.zeros((height, width))
        for ref in shifted:
            observed = (b-slope*ref)/gain
            for start_offset, end_offset in segments:
                step = end_offset-start_offset
                length = float(step@step)
                t = np.clip(((observed-start_offset)@step)/length, 0, 1)[..., None] if length else 0.
                pred = slope*ref+gain*(start_offset+t*step)
                closer = np.max(abs(b-pred), axis=2)
                better = closer < to_contact
                change = np.where(better, np.max(abs(pred-ref), axis=2), change)
                to_contact = np.minimum(to_contact, closer)
        return (support & (to_contact < to_unchanged) & (to_contact <= np.maximum(.5*change, 3*noise+1))), change

    contact, change = classify([(offset, offset)])
    # A carried layer inside the contact composites its own constant offset
    # (and a blurred edge mixing both). Such colours are learned only inside
    # holes of the main contact region, so fringes outside a front can never
    # be learned as contact; an unchanged bubble matches no changed colour.
    holes = ndimage.binary_fill_holes(contact) & ~contact & support
    pool = holes & (np.max(abs(b-a), axis=2) > 4)
    carried = []
    while len(carried) < 3 and pool.sum() >= 16:
        samples = offsets[pool]
        bins, counts = np.unique(np.round(samples/2).astype(int), axis=0, return_counts=True)
        if counts.max() < 16:
            break
        centre = bins[np.argmax(counts)]*2.
        members = np.max(abs(samples-centre), axis=1) <= 4
        carried.append(np.median(samples[members], axis=0))
        pool[pool] = ~members
    if carried:
        segments = [(offset, offset)]+[(offset, o) for o in carried]+[(o, o) for o in carried]
        inner, inner_change = classify(segments)
        contact = contact | (holes & inner)
        change = np.where(holes & inner, inner_change, change)
    base["carried_contact_offsets_dn"] = [o.tolist() for o in carried]
    separation = float(np.median(change[region]))
    base.update(found=True, contact_offset_dn=offset.tolist(), contact_fit_noise_dn=noise,
                contact_separation_dn=separation, contact_pixels=int(contact.sum()))
    if separation < 3*noise+2:
        return {**base, "reason": "Contact colour change is not separable from unchanged"}
    labels, _ = ndimage.label(contact)
    touching = np.unique(labels[region & contact])
    domain = np.isin(labels, touching[touching > 0])
    if domain.any():
        rows, cols = np.nonzero(domain)
        base["contact_region_bounds_px"] = [int(cols.min()), int(rows.min()), int(cols.max())+1, int(rows.max())+1]
    pocket = region & ~contact
    if unknown is not None:
        pocket &= ~unknown
        base["unknown_pixels_in_region"] = int((region & unknown).sum())
    base["contact_mask_sha256"] = hashlib.sha256(np.packbits(contact).tobytes()).hexdigest()
    labels, count = ndimage.label(pocket)
    sizes = ndimage.sum(pocket, labels, range(1, count+1)) if count else np.array([])
    # A carried flake beside the source (a co-picked neighbour, Jialei
    # 2026-11-03) lies in a hole of the contact, off the source outline, and
    # its colour CHANGED from the reference. A bubble or a lagging front
    # stays unchanged, and a pocket on the outline always blocks.
    touches_body = ndimage.binary_dilation(body, iterations=2)
    changed_floor = 3*noise+4
    carried_pockets = [i for i in range(1, count+1) if sizes[i-1] >= CONTACT_POCKET_MIN_PX
                       and holes[labels == i].all() and not touches_body[labels == i].any()
                       and float(np.median(to_unchanged[labels == i])) > changed_floor]
    blocking = int((sizes >= CONTACT_POCKET_MIN_PX).sum())-len(carried_pockets)
    base.update(region_pixels=int(region.sum()), pocket_pixels=int(pocket.sum()),
                blocking_pockets=blocking, carried_pockets=len(carried_pockets),
                largest_pocket_px=int(sizes.max()) if count else 0,
                pocket_min_px=CONTACT_POCKET_MIN_PX,
                region_contact_fraction=float(contact[region].mean()))
    if blocking:
        reason = ("Contact region does not yet contain the outline and margin"
                  if base["region_contact_fraction"] < .5 else
                  "A non-contact pocket lies inside the outline or margin")
        return {**base, "reason": reason}
    return {**base, "covered": True, "reason": None}


def parked_chip_support(source, calibration: PixelCalibration):
    """A finite chip-support domain from an unobstructed named-profile exposure.

    Under the declared finite-footprint renderer, a disjoint chip draws NONE
    of its flakes, electrodes or residue, and returns the OFF_CHIP field. An
    observed pixel outside that entire null therefore proves that SOME chip
    footprint point intersects the field. It does not locate that point at the
    observed pixel. The caller binds the unobstructed parked path and profile.
    """
    values = (calibration.mm_per_pixel_x, calibration.mm_per_pixel_y)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v == 0 for v in values):
        raise ValueError("Chip support needs finite nonzero signed native scales")
    sx, sy = map(abs, values)
    image = load_pixels(source)
    width, height = image.size
    if min(image.size) < 64:
        raise ValueError("Chip support has no observable native interior")
    rgb = np.asarray(image, dtype=float)
    yy, xx = np.mgrid[:height, :width]
    gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
    null = np.floor(gain[:, :, None]*np.array(SYNTHETIC_OFF_CHIP_RGB))
    eligible = (np.any((rgb < null-3) | (rgb > null+3), axis=2)
                & (np.min(rgb, axis=2) > 0) & (np.max(rgb, axis=2) < 255)
                & (xx >= 8) & (xx < width-8) & (yy >= 30) & (yy < height-32))
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return {"found": False, "reason": "Parked field does not exclude the full disjoint-chip optical null"}
    chosen = min(indices.tolist(), key=lambda i: (i % width-width/2)**2+(i//width-height/2)**2)
    # k<=7, but native/coarse ratios are not assumed exactly7. Bilinear
    # support is bounded by TWO actual coarse cells, plus the declared4um
    # illuminated chip edge. No14px shortcut or exact corner is inferred.
    coarse = (max(8, width//CHIP_MAX_DOWNSAMPLE), max(8, height//CHIP_MAX_DOWNSAMPLE))
    support = (2*width*sx/coarse[0]+CHIP_EDGE_ILLUMINATION_MM,
               2*height*sy/coarse[1]+CHIP_EDGE_ILLUMINATION_MM)
    box = [-width*sx/2-support[0], -height*sy/2-support[1],
           width*sx/2+support[0], height*sy/2+support[1]]
    return {"found": True, "method": "synthetic-parked-chip-intersection-domain-v1",
            "null_excluding_pixel_px": [chosen % width, chosen//width],
            "possible_chip_point_box_mm": box, "finite_raster_support_mm": list(support),
            "max_downsample": CHIP_MAX_DOWNSAMPLE, "coarse_native_shape_px": list(coarse),
            "illuminated_edge_support_mm": CHIP_EDGE_ILLUMINATION_MM,
            "scope": "Conservative full-FOV source-chip support; no precise feature or apex metrology"}


def source_projection_certificate(*, axis, elapsed_s, z_variation_mm, z_delta_mm,
                                  source_box_mm, front, front_bounds_px, size_px,
                                  calibration: PixelCalibration, cap_center_mm=None,
                                  region_domain: bool = False) -> SourceProjectionCertificate:
    """Conservative public trajectory/domain bound, with no apex or gap input.

    The caller proves a sole stationary chip and the complete actual Z-only
    command/encoder prefix. Drift is bounded BEFORE excluding the discontinuous
    dome-branch boundary. Only then is the within-branch height bound used.
    """
    keys = ("apex_lateral_drift_speed_norm_bound_mm_s", "apex_approach_drift_norm_bound_mm_per_mm",
            "apex_drift_clip_per_axis_mm", "dome_curvature_radius_mm",
            "lateral_focus_slope_norm_bound_mm_per_mm", "source_refraction_projection_norm_bound_mm_per_mm")
    values = [axis.get(key) for key in keys]
    if (axis.get("source") != "synthetic" or axis.get("simulation_only") is not True
            or axis.get("apex_lateral_drift_model") != "componentwise-clipped-synthetic-stamp-v1"
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in [*values, elapsed_s, z_variation_mm, z_delta_mm])
            or values[3] <= 0 or z_variation_mm+1e-12 < z_delta_mm):
        raise ValueError("Missing finite declared synthetic trajectory calibration")
    scales = (calibration.mm_per_pixel_x, calibration.mm_per_pixel_y)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v == 0 for v in scales):
        raise ValueError("Invalid signed native projection calibration")
    sx, sy = scales
    speed, approach, clip, dome_radius, slope, projection = values
    displacement = min(2*math.sqrt(2)*clip, speed*elapsed_s+approach*z_variation_mm)
    angular = front.get("angular_coverage")
    if type(angular) not in (int, float) or not math.isfinite(angular) or not 0 <= angular <= 1:
        raise ValueError("Observed front angular support must be a finite fraction")
    # Region coverage (#136, D13) bounds the domain by the observed contact
    # region itself, of any shape, instead of a fitted circular front.
    if region_domain:
        front = {**front, "has_contact": True, "failure_reason": None, "film_intact": None,
                 "angular_coverage": 1., "interface_uncertainty_px": 0.}
        angular = 1.
    if (front.get("has_contact") is not True
            or front.get("failure_reason") == "boundary_clipped"
            or front.get("film_intact") is False
            or not front_bounds_px or len(front_bounds_px) != 4 or not source_box_mm or len(source_box_mm) != 4
            or not isinstance(size_px, (tuple, list)) or len(size_px) != 2
            or any(type(v) is not int or v <= 0 for v in size_px)
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for v in [*front_bounds_px, *source_box_mm])
            or front_bounds_px[0] >= front_bounds_px[2] or front_bounds_px[1] >= front_bounds_px[3]
            or source_box_mm[0] >= source_box_mm[2] or source_box_mm[1] >= source_box_mm[3]):
        raise ValueError("No complete observed front/source domain for the surface-branch certificate")
    width, height = size_px
    fc = [((x-width/2)*sx, (y-height/2)*sy)
          for x in (front_bounds_px[0], front_bounds_px[2]) for y in (front_bounds_px[1], front_bounds_px[3])]
    sc = [(x, y) for x in (source_box_mm[0], source_box_mm[2]) for y in (source_box_mm[1], source_box_mm[3])]
    uncertainty = front.get("interface_uncertainty_px")
    if type(uncertainty) not in (int, float) or not math.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("Unknown observed front support uncertainty")
    common = {"method": "synthetic-id-bounded-source-projection-v1", "elapsed_s": elapsed_s,
              "z_total_variation_mm": z_variation_mm, "z_endpoint_delta_mm": z_delta_mm,
              "apex_displacement_norm_bound_mm": displacement, "dome_radius_mm": dome_radius,
              "native_pixel_bound_mm": min(abs(sx), abs(sy)),
              "observed_front_angular_coverage": angular,
              "required_front_angular_coverage": _FRONT_ANGULAR_SUPPORT, "region_domain": region_domain,
              "scope": "Declared synthetic model bound; no precise apex or physical metrology"}
    if angular < _FRONT_ANGULAR_SUPPORT:
        return {**common, "status": "insufficient_observation", "admitted": False,
                "reason": "Observed contact-front angular support is insufficient for the unchanged positive certificate",
                "source_branch_distance_bound_mm": None, "source_branch_stays_eligible": None,
                "projection_displacement_bound_mm": None}
    allowance = math.hypot(sx, sy)+uncertainty*max(abs(sx), abs(sy))
    domain_bound = max(math.dist(a, b) for a in fc for b in sc)+allowance+displacement
    projected = projection*(z_delta_mm+(1+slope)*displacement)
    cap = {}
    eligible = domain_bound < dome_radius
    if axis.get("stamp_discovery") is not None:
        from qumus_simlab.records.discovery import validate_policy

        policy = validate_policy(axis["stamp_discovery"])
        if (axis.get("cap_support_radius_mm") != policy["cap_radius_mm"]
                or not isinstance(cap_center_mm, (list, tuple)) or len(cap_center_mm) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in cap_center_mm)):
            raise ValueError("Truncated-cap source projection needs its original nominal-head/camera context")
        cap_distance = max(math.dist(cap_center_mm, point) for point in sc)+allowance+displacement
        cap = {"cap_support_radius_mm": policy["cap_radius_mm"],
               "cap_center_camera_mm": list(cap_center_mm), "cap_support_distance_bound_mm": cap_distance,
               "cap_support_stays_eligible": cap_distance < policy["cap_radius_mm"]}
        eligible = eligible and cap["cap_support_stays_eligible"]
    admitted = eligible and projected <= min(abs(sx), abs(sy))
    return {**common, **cap, "status": "admitted" if admitted else "bound_failed", "admitted": admitted,
            "reason": None if admitted else "Source registration/domain exceeds the fixed native-pixel certificate",
            "source_branch_distance_bound_mm": domain_bound,
            "source_branch_stays_eligible": eligible,
            "projection_displacement_bound_mm": projected}


# Release has an opaque-cell exemption and conditional convex support prior;
# pickup's optical whole-interior contract above is deliberately unchanged.
def release_load_transport(*, axis, elapsed_s, z_variation_mm, noise_scale,
                           footprint_size_mm, camera_size_mm):
    from statistics import NormalDist

    if (not isinstance(axis, dict) or not isinstance(footprint_size_mm, (tuple, list))
            or not isinstance(camera_size_mm, (tuple, list)) or len(footprint_size_mm) != 2 or len(camera_size_mm) != 2):
        raise ValueError("Unknown release transport coordinate declaration")
    keys = ("apex_lateral_drift_speed_norm_bound_mm_s", "apex_approach_drift_norm_bound_mm_per_mm",
            "apex_drift_clip_per_axis_mm", "contact_slip_rate_bound_deg_s")
    values = [axis.get(k) for k in keys]
    numbers = [elapsed_s, z_variation_mm, noise_scale, *values, *footprint_size_mm, *camera_size_mm]
    if (axis.get("source") != "synthetic" or axis.get("simulation_only") is not True
            or axis.get("apex_lateral_drift_model") != "componentwise-clipped-synthetic-stamp-v1"
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in numbers)
            or not 0 <= noise_scale <= 1 or min(*footprint_size_mm, *camera_size_mm) <= 0):
        raise ValueError("Unsupported finite release transport declarations")
    speed, approach, clip, slip = values
    e0 = math.sqrt(2)*NormalDist().inv_cdf(1-.001/(4*2))* .002*noise_scale
    drift = min(2*math.sqrt(2)*clip, speed*elapsed_s+approach*z_variation_mm)
    angle = min(math.pi, math.radians(slip*elapsed_s))
    rbase = math.hypot((footprint_size_mm[0]+camera_size_mm[0])/2,
                       (footprint_size_mm[1]+camera_size_mm[1])/2)
    radius = rbase+e0+drift
    return {"elapsed_s": elapsed_s, "z_total_variation_mm": z_variation_mm,
            "centroid_envelope_mm": e0, "drift_bound_mm": drift, "rotation_bound_rad": angle,
            "rotation_domain_radius_mm": radius,
            "load_envelope_mm": e0+drift+2*radius*math.sin(angle/2),
            "alpha_centroid_only": .001, "Kmax": 2, "N": 1}


def _release_opaque_masks(loaded, size, calibration, reference_envelope_mm, current_envelope_mm, *,
                          opaque_hbn=False, unknown_flakes=()):
    """Camera cells whose contact state is unknown during release.

    Gr is always opaque. With `opaque_hbn` (a carried hBN class the camera cannot
    see through within its frozen limits) the loaded hBN and every recorded
    co-picked carried flake are unknown too, never inferred through.
    """
    from PIL import Image, ImageDraw, ImageFilter
    from scipy.ndimage import distance_transform_edt

    width, height = size
    sx, sy = calibration.mm_per_pixel_x, calibration.mm_per_pixel_y
    image = Image.new("1", size)
    draw = ImageDraw.Draw(image)
    for body in loaded:
        if body["material"] == "Gr" or opaque_hbn:
            if "region_mm" not in body:
                draw.polygon([(width/2+x/sx, height/2+y/sy) for x, y in body["outline_mm"]], fill=1)
    mask = np.array(image, dtype=bool, copy=True)
    # Native observations retain holes/components. Mark every camera cell
    # intersecting opaque material as unknown, rather than fabricating a hull
    # or clearing another component through a hole in an overlapping ring.
    native_gr = [body for body in loaded if (body["material"] == "Gr" or opaque_hbn) and "region_mm" in body]
    if opaque_hbn:
        native_gr += [body for body in unknown_flakes if "region_mm" in body]
    if native_gr:
        import shapely

        from qumus_simlab.core.regions import _geos, as_region

        yy, xx = np.mgrid[:height, :width]
        cx, cy = (xx-width/2)*sx, (yy-height/2)*sy
        cells = shapely.box(cx-abs(sx)/2, cy-abs(sy)/2, cx+abs(sx)/2, cy+abs(sy)/2)
        for body in native_gr:
            mask |= shapely.intersects(_geos(as_region(body["region_mm"])), cells)
    def expanded(envelope):
        grow = distance_transform_edt(~mask, sampling=(abs(sy), abs(sx))) <= envelope
        return np.asarray(Image.fromarray(grow).filter(ImageFilter.MaxFilter(27)), dtype=bool)
    reference, current = expanded(reference_envelope_mm), expanded(current_envelope_mm)
    union = current | np.asarray(Image.fromarray(reference).filter(ImageFilter.MaxFilter(5)), dtype=bool)
    return reference, current, union

def _release_strict_cells(reference, current, optical, unknown):
    from PIL import Image, ImageFilter
    W, H = reference.size
    front, bounds = contact_front_domain(current)
    info = {'has_contact': front.has_contact, 'angular_coverage': front.angular_coverage, 'reason': front.failure_reason, 'film_intact': front.film_intact, 'radius_px': front.radius_px, 'interface_uncertainty_px': front.interface_uncertainty_px, 'front_bounds_px': bounds}
    zeros = np.zeros((H, W), bool)
    if not front.has_contact or front.failure_reason is not None or front.film_intact is False or (front.angular_coverage < 1.0) or (front.centroid_px is None) or (front.radius_px is None):
        return (zeros, info)
    radius = front.radius_px - front.interface_uncertainty_px - 0.5
    if radius <= 0:
        return (zeros, info)
    before = np.asarray(reference, dtype=float)
    after = np.asarray(current, dtype=float)
    yy, xx = np.mgrid[:H, :W]
    gain = (1 - 0.14 * ((xx - W / 2) / (W / 2)) ** 2) * (1 - 0.14 * ((yy - H / 2) / (H / 2)) ** 2)
    cx, cy = front.centroid_px
    domain = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2) & (xx >= 8) & (xx < W - 8) & (yy >= 30) & (yy < H - 32) & ~unknown
    clipped = (before.min(axis=2) <= 0) | (before.max(axis=2) >= 255)
    clipped = Image.fromarray(clipped).filter(ImageFilter.MaxFilter(5))
    domain &= ~np.asarray(clipped, dtype=bool)
    domain &= (after.min(axis=2) > 0) & (after.max(axis=2) < 255)
    iy, ix = np.where(domain)
    if not len(ix):
        return (zeros, info)
    pooled = []
    target = _projections(after[iy, ix])
    g = gain[iy, ix, None]
    generators = optical.generators()
    photo = SYNTHETIC_CONTACT_PHOTOMETRY
    ai = (1 - photo.front_opacity) * (1 - photo.interior_opacity)
    ci = np.array(photo.front_color) * photo.front_opacity * (1 - photo.interior_opacity) + np.array(photo.interior_color) * photo.interior_opacity
    contact = np.zeros(len(ix), bool)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            raw = before[iy + dy, ix + dx] * (gain[iy, ix] / gain[iy + dy, ix + dx])[:, None]
            pooled.append(_projections(raw))
            prediction = _projections(ai * raw + g * ci)
            contact |= (abs(target[:, :3] - prediction[:, :3]).max(axis=1) <= LIMITS['contact_rgb']) & (abs(target[:, 3:] - prediction[:, 3:]).max(axis=1) <= LIMITS['contact_chroma'])
    pooled = np.stack(pooled)
    lo = pooled.min(axis=0)
    hi = pooled.max(axis=0)
    underlying_lo = lo.copy()
    underlying_hi = hi.copy()
    profiles = []
    for opacity, color in generators:
        color = _projections(g * np.array(color))
        profiles.append((opacity, color))
        underlying_lo = np.minimum(underlying_lo, (lo - opacity * color) / (1 - opacity))
        underlying_hi = np.maximum(underlying_hi, (hi - opacity * color) / (1 - opacity))
    null_lo = underlying_lo.copy()
    null_hi = underlying_hi.copy()
    af = 1 - photo.front_opacity
    foffset = _projections(g * np.array(photo.front_color) * photo.front_opacity)
    f_lo = af * underlying_lo + foffset
    f_hi = af * underlying_hi + foffset
    front_lo = f_lo.copy()
    front_hi = f_hi.copy()
    for opacity, color in profiles:
        null_lo = np.minimum(null_lo, (1 - opacity) * underlying_lo + opacity * color)
        null_hi = np.maximum(null_hi, (1 - opacity) * underlying_hi + opacity * color)
        front_lo = np.minimum(front_lo, (1 - opacity) * f_lo + opacity * color)
        front_hi = np.maximum(front_hi, (1 - opacity) * f_hi + opacity * color)
    errors = np.array([LIMITS['unchanged_rgb']] * 3 + [LIMITS['unchanged_chroma']] * 2)
    unchanged = np.all((target >= null_lo - errors) & (target <= null_hi + errors), axis=1)
    annulus = np.all((target >= front_lo - errors) & (target <= front_hi + errors), axis=1)
    supported = contact & ~unchanged & ~annulus
    result = zeros.copy()
    result[iy[supported], ix[supported]] = True
    info.update(supported_pixels=int(supported.sum()), comparison_pixels=len(ix), unchanged_pixels=int(unchanged.sum()), front_null_pixels=int(annulus.sum()))
    return (result, info)


def _release_gain():
    width, height = 480, 320
    yy, xx = np.mgrid[:height, :width]
    gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
    gamma = max(float(np.max(gain[2:-2, 2:-2]/gain[2+dy:height-2+dy, 2+dx:width-2+dx]))
                for dy in range(-2, 3) for dx in range(-2, 3))
    return gain, gamma


def _release_camera_bounds(optical, gain, gamma):
    q = 1/(1-max(alpha for alpha, _ in optical.generators()))
    alpha = (1-SYNTHETIC_CONTACT_PHOTOMETRY.front_opacity)*(1-SYNTHETIC_CONTACT_PHOTOMETRY.interior_opacity)
    gmax = float(gain.max())
    bounds = {"contact_rgb": 1.72*gmax+gmax+max(1., alpha*q*gamma)+3*(1+alpha*q*gamma),
              "contact_chroma": 1.72*gmax+gmax*(1+alpha*q)+1+alpha*q*gamma,
              "unchanged_rgb": max(gmax, q*gmax)+max(1., q*gamma)+3*(1+q*gamma),
              "unchanged_chroma": gmax*(1+q)+1+q*gamma}
    front_alpha = 1-SYNTHETIC_CONTACT_PHOTOMETRY.front_opacity
    bounds["front_only_rgb"] = gmax+gmax+max(1., front_alpha*q*gamma)+3*(1+front_alpha*q*gamma)
    bounds["front_only_chroma"] = gmax+gmax*(1+front_alpha*q)+1+front_alpha*q*gamma
    return q, bounds


def _release_bounds_within_limits(bounds):
    return not (any(bounds[k] > LIMITS[k] for k in LIMITS)
                or bounds["front_only_rgb"] > LIMITS["unchanged_rgb"]
                or bounds["front_only_chroma"] > LIMITS["unchanged_chroma"])


def release_carried_hbn_opaque(optical: InteriorCalibration, loaded_count: int) -> bool:
    """True when the camera cannot see through the frozen carried hBN class.

    Same principle as the pickup region rule (D13): the frozen recognition
    LIMITS are not widened. The 480x320 fixed5x release raster is declared, so
    this depends only on the frozen class. Thin classes keep the see-through
    law unchanged.
    """
    if loaded_count != 2 or not optical.labels_nm:
        return False
    gain, gamma = _release_gain()
    return not _release_bounds_within_limits(_release_camera_bounds(optical, gain, gamma)[1])


def _json_native(value):
    """The recorded form: the release record is re-derived and compared after storage."""
    if isinstance(value, dict):
        return {k: _json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _release_region_support(reference, current, calibration, vertices, masks, hashes, envelopes, profile,
                            current_transport, unknown_flakes, q, gamma, bounds, native):
    """Release support for an opaque carried hBN load (D13 principle, #136).

    The pickup region rule Jialei approved (2026-11-02) applied at release:
    the observed contact REGION, of any shape and possibly clipped by the
    frame on its far side, must contain the whole load hull plus the existing
    release margin (envelope + 175 um collar + erosion guard) with no
    non-contact pocket. Only the opaque-unknown load cells are exempt. The
    region rule classifies each pixel against this exposure's own contact
    colour, so pre-made electrode metal under the stamp is judged by its own
    reference colour rather than by frozen absolute limits.
    """
    from scipy.spatial import ConvexHull

    sx, sy = calibration.mm_per_pixel_x, calibration.mm_per_pixel_y
    points = np.asarray(vertices, dtype=float)
    hull = [(240+x/sx, 160+y/sy) for x, y in points[ConvexHull(points).vertices]]
    collar, erosion = .1752, .05963091755805144
    region = _json_native(contact_region_coverage(reference, current, calibration, hull,
                                                  margin_mm=envelopes[1]+collar+erosion, unknown=masks[2]))
    front = dict(region["front"])
    centre, radius = front.get("centroid_px"), front.get("radius_px")
    # A far-side clipped front is admissible (D13); the surface-branch bound
    # then uses the whole fitted circle, never the clipped visible box.
    front["front_bounds_px"] = ([float(centre[0]-radius), float(centre[1]-radius), float(centre[0]+radius), float(centre[1]+radius)]
                                if front.get("has_contact") and centre and radius else None)
    front["front_bounds_from_fitted_circle"] = True
    admitted = bool(region.get("covered")) and front["front_bounds_px"] is not None and front.get("film_intact") is not False
    result = {"method": profile["id"], "conditional_modeled_support": admitted,
        "optical_whole_interior_observed": False, "integrity": "unknown", "transfer_confirmed": None,
        "recognition_tail_budget_centroid_only": .001, "Kmax": 2, "N": 1,
        "reference_opaque_mask_sha256": hashes[0], "current_opaque_mask_sha256": hashes[1],
        "opaque_unknown_mask_sha256": hashes[2], "opaque_unknown_pixels": int(masks[2].sum()),
        "supported_mask_sha256": region.get("contact_mask_sha256"), "supported_pixels": region.get("contact_pixels", 0),
        "front": front, "transport": current_transport, "collar_mm": collar, "erosion_mm": erosion,
        "required_clearance_mm": envelopes[1]+collar+erosion, "region_coverage": region,
        "camera_bounds": {"gain_ratio_max": gamma, "inverse_projection_max": q, "derived_dn": bounds},
        "carried_hbn_unknown": {
            "rule": "carried hBN outside the frozen see-through limits is opaque-unknown; the observed contact "
                    "region around it must contain the load hull plus the release margin with no pocket",
            "co_picked_unknown_flakes": len([b for b in unknown_flakes if "region_mm" in b])},
        "reason": "Conditional SIM support" if admitted else region.get("reason") or "Contact region does not cover the load"}
    if native:
        from qumus_simlab.perception.measurement import owner_error_budget_policy

        result.update(owner_error_budget=owner_error_budget_policy(), recognition_tail_budget_applies=False,
            transport_uncertainty_scope="Existing numerical envelope retained as an engineering guard; no native Gaussian recognition law claimed")
    return result


def release_support_evidence(reference, current, calibration: PixelCalibration, loaded, *,
                             optical: InteriorCalibration, reference_transport, current_transport,
                             profile, claimed_mask_sha256=None, unknown_flakes=()):
    """Conditional SIM support from original pixels and independently observed load.

    Contours/transport must be authenticated by the caller. No supplied opacity
    mask is trusted: both-frame masks are reconstructed before any admission.
    """
    from scipy.spatial import ConvexHull, QhullError

    from qumus_simlab.core.optical_profiles import RELEASE_SUPPORT_ASSUMPTIONS
    from qumus_simlab.core.polygons import convex_polygon
    from qumus_simlab.core.regions import as_region, region_vertices

    if profile != RELEASE_SUPPORT_ASSUMPTIONS:
        raise ValueError("Unsupported conditional release support profile")
    sx, sy = calibration.mm_per_pixel_x, calibration.mm_per_pixel_y
    if any(type(v) not in (int, float) or not math.isfinite(v) or v == 0 for v in (sx, sy)):
        raise ValueError("Release needs finite signed native pixel calibration")
    if sx <= 0 or sy >= 0:
        raise ValueError("Unsupported release camera orientation")
    before, after = load_pixels(reference), load_pixels(current)
    if before.size != after.size or before.size != (480, 320):
        raise ValueError("Release requires the declared native raster")
    if not math.isclose(sx, 2.6/480, abs_tol=1e-12) or not math.isclose(sy, -1.72/320, abs_tol=1e-12):
        raise ValueError("Release requires the declared fixed5x scale")
    if ([b.get("material") for b in loaded] not in (["Gr"], ["hBN", "Gr"])
            or len({b.get("flake_id") for b in loaded}) != len(loaded)):
        raise ValueError("Unsupported or ambiguous release load")
    native = any("region_mm" in body for body in loaded)
    if native and not all("region_mm" in body for body in loaded):
        raise ValueError("Release cannot mix native region and legacy outline observations")
    vertices = []
    for body in loaded:
        if native:
            region = as_region(body["region_mm"])
            if not region.components:
                raise ValueError("Release requires a nonempty complete native material region")
            vertices.extend(region_vertices(region))
        else:
            vertices.extend(convex_polygon(body["outline_mm"]))
    if ((len(loaded) == 1 and optical.labels_nm)
            or len(loaded) == 2 and (not optical.labels_nm or loaded[0].get("thickness_label") not in optical.labels_nm)):
        raise ValueError("Release differs from its frozen observed hBN classifications")
    optical.generators()
    envelopes = [t["load_envelope_mm"] for t in (reference_transport, current_transport)]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in envelopes):
        raise ValueError("Invalid release load envelope")
    width, height = before.size
    yy, xx = np.mgrid[:height, :width]
    gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
    gamma = max(float(np.max(gain[2:-2, 2:-2]/gain[2+dy:height-2+dy, 2+dx:width-2+dx]))
                for dy in range(-2, 3) for dx in range(-2, 3))
    opaque_hbn = release_carried_hbn_opaque(optical, len(loaded))
    if opaque_hbn:
        optical = InteriorCalibration()
    q, bounds = _release_camera_bounds(optical, gain, gamma)
    if not _release_bounds_within_limits(bounds):
        raise ValueError("Release camera uncertainty exceeds its unchanged recognition limits")
    masks = _release_opaque_masks(loaded, before.size, calibration, *envelopes, opaque_hbn=opaque_hbn,
                                  unknown_flakes=unknown_flakes if opaque_hbn else ())
    hashes = [hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks]
    if claimed_mask_sha256 is not None and claimed_mask_sha256 != hashes[2]:
        raise ValueError("Opacity mask differs from the independently reconstructed original load")
    if opaque_hbn:
        return _release_region_support(reference, current, calibration, vertices, masks, hashes, envelopes,
                                       profile, current_transport, unknown_flakes, q, gamma, bounds, native)
    cells, front = _release_strict_cells(before, after, optical, masks[2])
    result = {"method": profile["id"], "conditional_modeled_support": False,
        "optical_whole_interior_observed": False, "integrity": "unknown", "transfer_confirmed": None,
        "recognition_tail_budget_centroid_only": .001, "Kmax": 2, "N": 1,
        "reference_opaque_mask_sha256": hashes[0], "current_opaque_mask_sha256": hashes[1],
        "opaque_unknown_mask_sha256": hashes[2], "opaque_unknown_pixels": int(masks[2].sum()),
        "supported_mask_sha256": hashlib.sha256(cells.tobytes()).hexdigest(),
        "supported_pixels": int(cells.sum()), "front": front,
        "transport": current_transport, "collar_mm": .1752, "erosion_mm": .05963091755805144,
        "camera_bounds": {"gain_ratio_max": gamma, "inverse_projection_max": q, "derived_dn": bounds}}
    if native:
        from qumus_simlab.perception.measurement import owner_error_budget_policy

        result.update(owner_error_budget=owner_error_budget_policy(), recognition_tail_budget_applies=False,
            transport_uncertainty_scope="Existing numerical envelope retained as an engineering guard; no native Gaussian recognition law claimed")
    if cells.sum() < 3:
        return {**result, "reason": "Insufficient strict observed interior"}
    iy, ix = np.where(cells)
    points = np.column_stack(((ix-240)*sx, (iy-160)*sy))
    try:
        eq = ConvexHull(points).equations
    except QhullError:
        return {**result, "reason": "Degenerate strict support hull"}
    eq = eq/np.linalg.norm(eq[:, :2], axis=1)[:, None]
    outlines = np.asarray(vertices)
    clearance = float(np.min(-(outlines@eq[:, :2].T+eq[:, 2])))
    required = envelopes[1]+.1752+.05963091755805144
    yy, xx = np.mgrid[:before.height, :before.width]
    eroded = np.ones((before.height, before.width), dtype=bool)
    for a, b, c in eq:
        eroded &= a*(xx-240)*sx+b*(yy-160)*sy+c <= -.05963091755805144
    missing = eroded & ~masks[2] & ~cells
    admitted = bool(clearance >= required and not missing.any())
    return {**result, "conditional_modeled_support": admitted,
            "minimum_hull_clearance_mm": clearance, "required_clearance_mm": required,
            "visible_unsupported_in_eroded_hull": int(missing.sum()),
            "reason": "Conditional SIM support" if admitted else "Containment or visible-cell support unavailable"}
