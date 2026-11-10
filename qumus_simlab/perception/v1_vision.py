"""v1 camera-image algorithms, ported to run on captured pixels.

Sources (v1 original, read-only):
  - focus:   common/hardware/camera_controller.py  ``blur_score_bgr`` (Tenengrad)
  - corner:  processing_agent/workflows/search/utils/edge_follow.py
             (``detect_edge_and_motion``) and corner_coordinate_adjustment.py
             (``compute_corner_xy_from_detector``)
  - refind:  processing_agent/workflows/search/molecule_flows/
             molecule_flake_refind_accurate.py (legacy centre-patch template
             matching over angles on Canny edges)

Constants and decision rules follow v1. Inputs are RGB uint8 arrays (the v2
frame convention); they are converted to BGR before any v1 code path that
reads colour channels. Nothing here reads simulator metadata.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- focus: v1 blur_score -------------------------------------------------

def blur_score(rgb: np.ndarray, shrink: int = 2) -> float:
    """v1 ``blur_score_bgr``: log of the mean Sobel gradient energy (higher = sharper)."""
    rgb = np.asarray(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    if shrink and shrink > 1:
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (max(1, w // shrink), max(1, h // shrink)),
                          interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.log(float(np.mean(gx * gx + gy * gy)) + 1e-12))


# --- chip edge / corner: v1 edge_follow ------------------------------------

AREA_INVALID_HIGH = 0.80
AREA_INVALID_LOW = 0.15
BOUNDARY_NEAR_BORDER_FRAC = 0.06
TOUCH_TOL_PX = 6
HALF_DOMINANCE_MIN = 0.55
HALF_DOMINANCE_GAP = 0.08
EDGE_BAND_PX = 20
MIN_FIT_PTS = 120
RGB_REGION_MIN_AREA = 400
RGB_BLACK_SUM_MAX = 60.0
RGB_CHIP_SCORE_MIN = 20.0
DEFAULT_DIR = (1.0, 0.0)
DETECT_DOWNSCALE = 0.5
RGB_ROI_MARGIN_PX = 36
MIN_COMPONENT_AREA = 200
MIN_FULLRES_NONZERO = 200

#: v1 names the corner of the screen the chip's bbox touches; the chip corner in
#: view is the opposite one (bbox in the top-right of the screen = the chip's
#: bottom-left corner is visible).
_CHIP_CORNER_OF_SCREEN = {"TL": "br", "TR": "bl", "BR": "tl", "BL": "tr"}


@dataclass
class EdgeResult:
    ok: bool
    reason: str
    chip_ratio: float
    chip_mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    motion_vec: tuple[float, float]
    motion_label: str


def _unit(vx: float, vy: float) -> tuple[float, float]:
    n = float(np.hypot(vx, vy))
    return (0.0, 0.0) if n < 1e-9 else (vx / n, vy / n)


def _largest_component(mask255):
    cnts, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_COMPONENT_AREA:
        return None, None
    filled = np.zeros_like(mask255)
    cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
    x, y, w, h = cv2.boundingRect(cnt)
    return filled, (x, y, x + w, y + h)


def _boundary_points(mask255: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(mask255, cv2.MORPH_GRADIENT, k)
    ys, xs = np.where(grad > 0)
    return np.stack([xs, ys], axis=1).astype(np.int32) if len(xs) else np.zeros((0, 2), np.int32)


def _fit_line(points_xy: np.ndarray):
    if points_xy.shape[0] < MIN_FIT_PTS:
        return None
    vx, vy, x0, y0 = (float(np.asarray(v).reshape(-1)[0]) for v in
                      cv2.fitLine(points_xy.astype(np.float32).reshape(-1, 1, 2),
                                  cv2.DIST_L2, 0, 0.01, 0.01))
    n = float(np.hypot(vx, vy))
    return None if n < 1e-9 else ((x0, y0), (vx / n, vy / n))


def _select_edge_points(boundary_xy, bbox, edge):
    x0, y0, x1, y1 = bbox
    x, y = boundary_xy[:, 0], boundary_xy[:, 1]
    mask = {"top": np.abs(y - y0) <= EDGE_BAND_PX, "bottom": np.abs(y - y1) <= EDGE_BAND_PX,
            "left": np.abs(x - x0) <= EDGE_BAND_PX, "right": np.abs(x - x1) <= EDGE_BAND_PX
            }.get(edge, np.ones_like(x, dtype=bool))
    return boundary_xy[mask]


def _enforce_sign(vec, want):
    dx, dy = vec
    if (want == "RIGHT" and dx < 0) or (want == "LEFT" and dx > 0) \
            or (want == "UP" and dy > 0) or (want == "DOWN" and dy < 0):
        dx, dy = -dx, -dy
    return _unit(dx, dy)


def _fallback_axis(want):
    return {"RIGHT": (1.0, 0.0), "LEFT": (-1.0, 0.0), "UP": (0.0, -1.0),
            "DOWN": (0.0, 1.0)}.get(want, _unit(*DEFAULT_DIR))


def _screen_corner(bbox, w, h):
    x0, y0, x1, y1 = bbox
    touched = {k for k, v in {"L": x0 <= TOUCH_TOL_PX, "R": x1 >= w - 1 - TOUCH_TOL_PX,
                              "T": y0 <= TOUCH_TOL_PX, "B": y1 >= h - 1 - TOUCH_TOL_PX}.items() if v}
    return {frozenset("LT"): "TL", frozenset("RT"): "TR", frozenset("RB"): "BR",
            frozenset("LB"): "BL"}.get(frozenset(touched))


def _classify(chip_filled, bbox, h, w) -> EdgeResult:
    """v1 ``_classify_from_chip_mask``: corner, edge or failure, and the jog direction."""
    def fail(reason, ratio):
        return EdgeResult(False, reason, ratio, chip_filled, bbox, _unit(*DEFAULT_DIR), "DEFAULT")

    chip_ratio = float((chip_filled > 0).mean())
    if chip_ratio > AREA_INVALID_HIGH or chip_ratio < AREA_INVALID_LOW:
        return fail(f"area invalid ({chip_ratio:.3f})", chip_ratio)
    bpts = _boundary_points(chip_filled)
    if bpts.shape[0] < MIN_FIT_PTS:
        return fail("too few boundary pts", chip_ratio)
    bx, by = float(bpts[:, 0].mean()), float(bpts[:, 1].mean())
    mx, my = BOUNDARY_NEAR_BORDER_FRAC * w, BOUNDARY_NEAR_BORDER_FRAC * h
    if bx < mx or bx > w - mx or by < my or by > h - my:
        return fail("boundary near image border", chip_ratio)

    corner_name, corner_reason, override_k1 = _screen_corner(bbox, w, h), None, None
    if corner_name is not None:
        hm, wm = h // 2, w // 2
        fr = {"TL": float((chip_filled[:hm, :wm] > 0).mean()), "TR": float((chip_filled[:hm, wm:] > 0).mean()),
              "BR": float((chip_filled[hm:, wm:] > 0).mean()), "BL": float((chip_filled[hm:, :wm] > 0).mean())}
        # v1: the opposite screen quadrant must not dominate, and the touched one must hold chip.
        spill, side, reason = {"TL": ("BL", "LEFT", "BR"), "TR": ("TL", "TOP", "BL"),
                               "BR": ("TR", "RIGHT", "TL"), "BL": ("BR", "BOTTOM", "TR")}[corner_name]
        if fr[spill] > 0.5:
            override_k1, corner_name = side, None
        elif fr[corner_name] >= 0.08:
            corner_reason = reason
        else:
            corner_name = None
    if corner_name is not None:
        want, edge = {"TL": ("LEFT", "bottom"), "TR": ("UP", "left"),
                      "BR": ("RIGHT", "top"), "BL": ("DOWN", "right")}[corner_name]
        fit = _fit_line(_select_edge_points(bpts, bbox, edge))
        motion = _enforce_sign(fit[1], want) if fit is not None else _fallback_axis(want)
        return EdgeResult(True, f"CORNER {corner_reason}", chip_ratio, chip_filled, bbox, motion, want)

    ys, xs = np.where(chip_filled > 0)
    if len(xs) < 200:
        return fail("chip too small", chip_ratio)
    fr = {"TOP": float((ys < h * 0.5).mean()), "BOTTOM": float((ys >= h * 0.5).mean()),
          "LEFT": float((xs < w * 0.5).mean()), "RIGHT": float((xs >= w * 0.5).mean())}
    (k1, v1), (_, v2) = sorted(fr.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if override_k1 is None and (v1 < HALF_DOMINANCE_MIN or (v1 - v2) < HALF_DOMINANCE_GAP):
        return fail("ambiguous halves", chip_ratio)
    k1 = override_k1 or k1
    want, edge = {"BOTTOM": ("RIGHT", "top"), "TOP": ("LEFT", "bottom"),
                  "RIGHT": ("UP", "left"), "LEFT": ("DOWN", "right")}[k1]
    fit = _fit_line(_select_edge_points(bpts, bbox, edge)) or _fit_line(bpts)
    if fit is not None:
        motion = _enforce_sign(fit[1], want)
        if (want in ("RIGHT", "LEFT") and abs(motion[0]) < 0.3) or \
                (want in ("UP", "DOWN") and abs(motion[1]) < 0.3):
            motion = _fallback_axis(want)
    else:
        motion = _fallback_axis(want)
    return EdgeResult(True, f"EDGE {k1}", chip_ratio, chip_filled, bbox, motion, want)


def _rgb_chip_mask(bgr, thr0, bbox):
    """v1 ``_build_chip_mask_by_rgb_regions_roi``: keep blue-dominant Otsu regions."""
    h, w = thr0.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    rx0, ry0 = max(0, x0 - RGB_ROI_MARGIN_PX), max(0, y0 - RGB_ROI_MARGIN_PX)
    rx1, ry1 = min(w, x1 + RGB_ROI_MARGIN_PX), min(h, y1 + RGB_ROI_MARGIN_PX)
    roi, thr = bgr[ry0:ry1, rx0:rx1], thr0[ry0:ry1, rx0:rx1]
    for side_val in (0, 255):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(thr == side_val), connectivity=8)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) < RGB_REGION_MIN_AREA:
                continue
            region = labels == i
            b, g, r = (float(v) for v in roi[region].mean(axis=0))
            if r + g + b >= RGB_BLACK_SUM_MAX and b - 0.5 * (r + g) > RGB_CHIP_SCORE_MIN:
                out[ry0:ry1, rx0:rx1][region] = 255
    return out


def detect_edge_and_motion(rgb: np.ndarray) -> EdgeResult:
    """v1 ``detect_edge_and_motion`` on a half-size image; the mask is returned full size."""
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    full_h, full_w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(2, round(full_w * DETECT_DOWNSCALE)), max(2, round(full_h * DETECT_DOWNSCALE))),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, thr0 = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # v1 _choose_chip_mask: the brighter Otsu side is the chip.
    white = thr0 == 255
    if white.sum() >= 50 and (~white).sum() >= 50 and gray[white].mean() < gray[~white].mean():
        thr0 = cv2.bitwise_not(thr0)
    filled, bbox = _largest_component(thr0.copy())
    if filled is None:
        res = EdgeResult(False, "no chip component", 0.0, thr0, None, _unit(*DEFAULT_DIR), "DEFAULT")
    else:
        res = _classify(filled, bbox, h, w)
        if res.reason.startswith(("EDGE ", "CORNER ")):
            rgb_filled, rgb_bbox = _largest_component(_rgb_chip_mask(small, thr0, bbox))
            if rgb_filled is not None and np.count_nonzero(rgb_filled) >= MIN_COMPONENT_AREA:
                alt = _classify(rgb_filled, rgb_bbox, h, w)
                if alt.reason.startswith(("EDGE ", "CORNER ")):
                    res = alt
    if (h, w) == (full_h, full_w):
        return res
    sx, sy = full_w / w, full_h / h
    mask = cv2.resize(res.chip_mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    bbox = None if res.bbox is None else tuple(
        int(np.clip(round(v * s), 0, lim - 1)) for v, s, lim in
        zip(res.bbox, (sx, sy, sx, sy), (full_w, full_h, full_w, full_h), strict=True))
    if np.count_nonzero(mask) < MIN_FULLRES_NONZERO:
        return EdgeResult(False, "chip too small after upscale", res.chip_ratio, mask, bbox,
                          _unit(*DEFAULT_DIR), "DEFAULT")
    return EdgeResult(res.ok, res.reason, res.chip_ratio, mask, bbox, res.motion_vec, res.motion_label)


@dataclass
class CornerPixel:
    chip_corner: str          # "bl", "br", "tl" or "tr" of the chip
    xy_px: tuple[float, float]  # sub-pixel intersection of the two fitted edges
    reason: str


def corner_from_detector(res: EdgeResult, shape: tuple[int, int]) -> CornerPixel | None:
    """v1 ``compute_corner_xy_from_detector``: intersect the two fitted chip edges."""
    if res is None or res.bbox is None or res.chip_mask is None:
        return None
    height, width = shape[:2]
    screen = _screen_corner(res.bbox, width, height)
    if screen is None:
        return None
    edge1, edge2 = {"TL": ("bottom", "right"), "TR": ("bottom", "left"),
                    "BR": ("top", "left"), "BL": ("top", "right")}[screen]
    boundary = _boundary_points(res.chip_mask)
    if len(boundary) < MIN_FIT_PTS:
        return None
    fit1 = _fit_line(_select_edge_points(boundary, res.bbox, edge1))
    fit2 = _fit_line(_select_edge_points(boundary, res.bbox, edge2))
    if fit1 is None or fit2 is None:
        return None
    (x1, y1), (dx1, dy1) = fit1
    (x2, y2), (dx2, dy2) = fit2
    a = np.array([[dx1, -dx2], [dy1, -dy2]], dtype=np.float64)
    if abs(float(np.linalg.det(a))) < 1e-6:
        return None
    t, _ = np.linalg.solve(a, np.array([x2 - x1, y2 - y1], dtype=np.float64))
    return CornerPixel(_CHIP_CORNER_OF_SCREEN[screen], (float(x1 + t * dx1), float(y1 + t * dy1)), res.reason)


# --- flake refind: v1 accurate template matching ----------------------------

#: molecule_flake_refind_accurate constants (legacy centre-patch path).
CENTER_PATCH_RATIO = 0.2
SEARCH_RANGE_DEG = 15.0
DOWNSAMPLE = 0.5


@dataclass
class RefindMatch:
    centre_px: tuple[float, float]  # where the reference image centre lies in the current image
    angle_deg: float                # rotation applied to the reference to match (cv2 sense)
    score: float                    # TM_CCOEFF_NORMED peak


def _edges(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(np.ascontiguousarray(rgb), None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
    return cv2.Canny(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), 50, 150)


def _centre_patch(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = int(h * CENTER_PATCH_RATIO), int(w * CENTER_PATCH_RATIO)
    return img[h // 2 - ph // 2:h // 2 + ph // 2, w // 2 - pw // 2:w // 2 + pw // 2]


def _search_once(initial: np.ndarray, target: np.ndarray, angles) -> tuple[float, float, tuple[int, int]]:
    best = None
    h, w = initial.shape[:2]
    for angle in angles:
        mat = cv2.getRotationMatrix2D((w // 2, h // 2), float(angle), 1.0)
        template = _centre_patch(cv2.warpAffine(initial, mat, (w, h), flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REFLECT))
        if template.size == 0 or not np.any(template):
            continue  # no edges in the patch: nothing to match (v1 had no such guard)
        _, value, _, loc = cv2.minMaxLoc(cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED))
        if best is None or value > best[0]:
            best = (float(value), float(angle), (int(loc[0]), int(loc[1])))
    if best is None:
        raise ValueError("Template matching produced no valid result")
    return best


def refind_template(reference_rgb: np.ndarray, current_rgb: np.ndarray, *,
                    init_angle_deg: float = 0.0) -> RefindMatch:
    """v1 accurate refind (legacy centre patch): find the reference centre in a new image.

    Both images are halved and reduced to Canny edges; the reference's centre
    20 % patch is rotated over +-15 deg in 3 deg steps, then +-2 deg in 0.5 deg
    steps around the best, and matched by normalised cross-correlation. As in
    v1 the location is integer at half size, so it is quantised to 2 px.
    """
    initial, current = _edges(reference_rgb), _edges(current_rgb)
    coarse = _search_once(initial, current, np.arange(init_angle_deg - SEARCH_RANGE_DEG,
                                                      init_angle_deg + SEARCH_RANGE_DEG + 1.0, 3.0))
    score, angle, (lx, ly) = _search_once(initial, current, np.arange(coarse[1] - 2.0, coarse[1] + 2.1, 0.5))
    ph, pw = _centre_patch(initial).shape[:2]
    return RefindMatch((lx / DOWNSAMPLE + pw / DOWNSAMPLE / 2.0, ly / DOWNSAMPLE + ph / DOWNSAMPLE / 2.0),
                       angle, score)


# ------------------------------------------------- v1 rough refind (corner)
# v1 molecule_flake_refind_rough: ORB-match the search-time bottom-left corner
# image against the new one, keep the RANSAC homography inliers, and take the
# rotation from estimateAffinePartial2D on them (#204). Constants are v1's.
ORB_NFEATURES = 2000
RATIO_TEST = 0.75
RANSAC_REPROJ_THRESH = 3.0
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 5
MIN_KEYPOINTS = 12
AFFINE_RANSAC_REPROJ_THRESH = 2.0
AFFINE_MAX_ITERS = 5000
AFFINE_CONFIDENCE = 0.99


def _gray_variants(bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return [gray, clahe.apply(gray), cv2.equalizeHist(gray)]


def _orb_features(bgr: np.ndarray, detector) -> tuple[list, np.ndarray | None]:
    best_kp, best_des = [], None
    for gray in _gray_variants(bgr):
        kp, des = detector.detectAndCompute(gray, None)
        if des is not None and len(kp) > len(best_kp):
            best_kp, best_des = kp, des
    return best_kp, best_des


@dataclass(frozen=True)
class CornerRotation:
    found: bool
    theta_cw_rad: float | None
    good_matches: int
    inliers: int
    loss_reason: str | None = None


def corner_rotation(reference_bgr: np.ndarray, current_bgr: np.ndarray) -> CornerRotation:
    """theta_cw of the current BL-corner image against the reference (v1 rough refind)."""
    detector = cv2.ORB_create(nfeatures=ORB_NFEATURES, fastThreshold=5, edgeThreshold=15, patchSize=31)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    kp_new, des_new = _orb_features(current_bgr, detector)
    kp_ref, des_ref = _orb_features(reference_bgr, detector)
    if des_new is None or des_ref is None or len(kp_new) < MIN_KEYPOINTS or len(kp_ref) < MIN_KEYPOINTS:
        return CornerRotation(False, None, 0, 0, "too few keypoints/descriptors")
    good = [m for pair in matcher.knnMatch(des_new, des_ref, k=2) if len(pair) == 2
            for m, n in [pair] if m.distance < RATIO_TEST*n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        return CornerRotation(False, None, len(good), 0, "match too weak")
    pts_new = np.float32([kp_new[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts_ref.reshape(-1, 1, 2), pts_new.reshape(-1, 1, 2), cv2.RANSAC,
                                 ransacReprojThreshold=RANSAC_REPROJ_THRESH)
    if H is None or mask is None:
        return CornerRotation(False, None, len(good), 0, "findHomography failed")
    inliers = int(mask.sum())
    keep = mask.ravel().astype(bool)
    if inliers < MIN_INLIERS or inliers < 6:  # v1 also needs 6 points for the affine angle
        return CornerRotation(False, None, len(good), inliers, "too few inliers")
    M, _ = cv2.estimateAffinePartial2D(pts_ref[keep], pts_new[keep], method=cv2.RANSAC,
        ransacReprojThreshold=AFFINE_RANSAC_REPROJ_THRESH, maxIters=AFFINE_MAX_ITERS,
        confidence=AFFINE_CONFIDENCE)
    if M is None:
        return CornerRotation(False, None, len(good), inliers, "estimateAffinePartial2D failed")
    return CornerRotation(True, float(np.arctan2(M[1, 0], M[0, 0])), len(good), inliers)
