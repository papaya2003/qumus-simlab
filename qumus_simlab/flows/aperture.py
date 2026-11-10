"""Observed aperture alignment for the explicitly synthetic paper-mask profile.

The scan prior plans clear-height travel. Only saved pixels, signed camera
calibration and captured encoder poses determine the fitted opening. No hidden
stamp/aperture/apex geometry is read. Reported uncertainty is a raster residual
heuristic, not measured laboratory metrology.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from qumus_simlab.core.errors import NeedsHumanError, PreconditionFailed
from qumus_simlab.core.run_paths import resolve_run_path
from qumus_simlab.flows.step import Observation, ProbeResult, Step, Verdict
from qumus_simlab.records.geometry import geometry_digest

PROFILE = "synthetic-paper-aperture-fit-v2"
LEGACY_PROFILE = "synthetic-paper-aperture-fit-v1"
STRUCTURE_PROFILE = "synthetic-glass-pdms-pc-paper-v2"
LEGACY_STRUCTURE_PROFILE = "synthetic-glass-pdms-pc-paper-v1"


@dataclass(frozen=True)
class ApertureScan:
    """Finite virtual search policy, not measured aperture dimensions."""
    offset_mm: float = 4.7
    max_extent_mm: float = 5.5
    speed_mm_s: float = 1.0
    max_seconds: float = 180.0
    max_captures: int = 9
    max_residual_mm: float = .012
    center_tolerance_mm: float = .025
    shape: str = "diamond"

    def __post_init__(self):
        values = (self.offset_mm, self.max_extent_mm, self.speed_mm_s, self.max_seconds,
                  self.max_residual_mm, self.center_tolerance_mm)
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                or self.offset_mm >= self.max_extent_mm or self.speed_mm_s > 2
                or self.max_seconds > 900 or type(self.max_captures) is not int
                or not 1 <= self.max_captures <= 9 or self.shape not in ("diamond", "circle")):
            raise ValueError("Aperture scan requires finite bounded policy values")


def _axis_point(u, v, calibration):
    scales = calibration.get("mm_per_pixel")
    if (not isinstance(scales, (list, tuple)) or len(scales) != 2
            or any(type(s) not in (int, float) or not math.isfinite(s) or s <= 0 for s in scales)):
        raise ValueError("Aperture observation requires positive millimetres-per-pixel calibration")
    result, axes = [0., 0.], []
    for offset, scale, label in zip((u, v), scales,
            (calibration.get("u_axis"), calibration.get("v_axis")), strict=True):
        if label not in ("+world_x", "-world_x", "+world_y", "-world_y"):
            raise ValueError("Aperture camera axes are not registered")
        index = 0 if label.endswith("x") else 1
        axes.append(index)
        result[index] += offset*scale*(1 if label.startswith("+") else -1)
    if len(set(axes)) != 2:
        raise ValueError("Aperture camera axes are not independent")
    return tuple(result)


def _paper(pixel):
    # Profile-bound observed chromaticity, invariant to the synthetic vignette.
    # It classifies visible opaque paper; it carries no geometric centre/radius.
    r, g, b = pixel
    return g > 80 and -.01 <= (r-g)/g <= .075 and .045 <= (g-b)/g <= .14


#: Paper/transmission regions smaller than this are specks on the sample, not
#: the aperture: the boundary of a 5 mm paper opening spans hundreds of pixels.
MIN_REGION_PX = 64


def _drop_small_regions(mask, width, height, min_px=MIN_REGION_PX):
    """Flip 4-connected regions below `min_px` pixels to the surrounding class."""
    mask = list(mask)
    seen = bytearray(width*height)
    for start in range(width*height):
        if seen[start]:
            continue
        value, stack, region = mask[start], [start], []
        seen[start] = 1
        while stack:
            index = stack.pop()
            region.append(index)
            x, y = index % width, index // width
            for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if 0 <= nx < width and 0 <= ny < height:
                    neighbour = ny*width+nx
                    if not seen[neighbour] and mask[neighbour] == value:
                        seen[neighbour] = 1
                        stack.append(neighbour)
        if len(region) < min_px:
            for index in region:
                mask[index] = not value
    return mask


def aperture_edges(record):
    """Return actual paper/transmission boundary pixels in camera millimetres."""
    path = resolve_run_path(record["image"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Aperture source pixels changed")
    calibration = record["calibration"]
    if (calibration.get("source") != "synthetic"
            or calibration.get("photometry_profile", {}).get("stamp_structure") not in
                (STRUCTURE_PROFILE, LEGACY_STRUCTURE_PROFILE)):
        raise ValueError("Aperture paper classifier requires its explicit synthetic profile")
    with Image.open(path) as source:
        pixels = source.convert("RGB")
    width, height = pixels.size
    if calibration.get("image_size_px") != [width, height] or min(width, height) < 64:
        raise ValueError("Aperture image size/calibration changed")
    data = list(pixels.get_flattened_data())
    mask = _drop_small_regions([_paper(p) for p in data], width, height)
    points = []
    # Ignore camera labels/scale bars and image-border cut edges.
    for y in range(30, height-32):
        for x in range(8, width-8):
            here = mask[y*width+x]
            if x+1 < width-8 and here != mask[y*width+x+1]:
                points.append(_axis_point(x+.5-width/2, y-height/2, calibration))
            if y+1 < height-32 and here != mask[(y+1)*width+x]:
                points.append(_axis_point(x-width/2, y+.5-height/2, calibration))
    if len(points) < 24:
        raise ValueError("No complete observable aperture edge segment in this exposure")
    return points[::max(1, math.ceil(len(points)/400))]


def _solve(matrix, rhs):
    augmented = [list(row)+[value] for row, value in zip(matrix, rhs, strict=True)]
    scale = max(abs(v) for row in matrix for v in row)
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) <= max(scale, 1.)*1e-9:
            raise ValueError("Aperture arcs do not determine a stable circle")
        augmented[column] = [v/divisor for v in augmented[column]]
        for row in range(3):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [a-factor*b for a, b in zip(augmented[row], augmented[column], strict=True)]
    return [row[-1] for row in augmented]


def fit_aperture(records, baseline_xy, *, max_residual_mm=.012, shape="diamond"):
    """Fit four observed boundaries; circle support is for frozen legacy replay."""
    if (len(baseline_xy) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in baseline_xy)
            or not math.isfinite(max_residual_mm) or max_residual_mm <= 0):
        raise ValueError("Aperture fit requires finite millimetre coordinates and residual policy")
    if len(records) != 4:
        raise ValueError("Four independent opposing aperture boundary exposures are required")
    if len({tuple(r["holder_xy_mm"]) for r in records}) != 4:
        raise ValueError("Aperture arcs require distinct encoder positions")
    calibration, camera = records[0]["calibration"], records[0]["camera_world_mm"]
    if shape not in ("diamond", "circle") or calibration.get("photometry_profile", {}).get("stamp_structure") != (
            STRUCTURE_PROFILE if shape == "diamond" else LEGACY_STRUCTURE_PROFILE):
        raise ValueError("Observed aperture shape/profile is not supported")
    points, groups = [], []
    for record in records:
        if record["calibration"] != calibration or record["camera_world_mm"] != camera:
            raise ValueError("Aperture camera or calibration changed during the scan")
        hx, hy = record["holder_xy_mm"]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (hx, hy)):
            raise ValueError("Aperture encoder coordinates are not finite millimetres")
        group = [(x-hx+baseline_xy[0], y-hy+baseline_xy[1]) for x, y in aperture_edges(record)]
        groups.append(group)
        points.extend(group)
    if shape == "diamond":
        return _fit_diamond(groups, calibration, max_residual_mm)
    origin = tuple(statistics.mean(p[i] for p in points) for i in (0, 1))
    rows = [(2*(x-origin[0]), 2*(y-origin[1]), 1.) for x, y in points]
    squares = [(x-origin[0])**2+(y-origin[1])**2 for x, y in points]
    matrix = [[sum(r[i]*r[j] for r in rows) for j in range(3)] for i in range(3)]
    rhs = [sum(r[i]*q for r, q in zip(rows, squares, strict=True)) for i in range(3)]
    cx, cy, constant = _solve(matrix, rhs)
    radius2 = constant+cx*cx+cy*cy
    if radius2 <= 0 or not math.isfinite(radius2):
        raise ValueError("Observed aperture circle is invalid")
    center, radius = (cx+origin[0], cy+origin[1]), math.sqrt(radius2)
    residuals = [abs(math.dist(p, center)-radius) for p in points]
    maximum = max(residuals)
    if maximum > max_residual_mm:
        raise ValueError("Aperture edges are occluded, ambiguous or inconsistent with one circle")
    quadrants = {(x >= center[0], y >= center[1]) for x, y in points}
    if len(quadrants) != 4:
        raise ValueError("Observed aperture arcs do not span opposing sides")
    scales = calibration["mm_per_pixel"]
    resolution = math.hypot(*scales)
    return {"center_offset_mm": list(center), "radius_mm": radius,
            "max_residual_mm": maximum, "rms_residual_mm": math.sqrt(statistics.mean(v*v for v in residuals)),
            "uncertainty_mm": maximum+resolution, "edge_points": len(points),
            "arc_points": list(map(len, groups)), "frame": "baseline_holder_camera_parallel_axes",
            "units": "mm", "uncertainty_scope": "Raster residual plus one pixel diagonal; heuristic, not physical metrology"}


def _fit_diamond(groups, calibration, max_residual_mm):
    """Fit four straight edges, pairing their observed opposing normals."""
    center_seed = tuple(statistics.mean(p[i] for group in groups for p in group) for i in (0, 1))
    lines = []
    for group in groups:
        mean = tuple(statistics.mean(p[i] for p in group) for i in (0, 1))
        xx = statistics.mean((x-mean[0])**2 for x, y in group)
        yy = statistics.mean((y-mean[1])**2 for x, y in group)
        xy = statistics.mean((x-mean[0])*(y-mean[1]) for x, y in group)
        angle = .5*math.atan2(2*xy, xx-yy)
        normal = (-math.sin(angle), math.cos(angle))
        if sum(n*(m-c) for n, m, c in zip(normal, mean, center_seed, strict=True)) < 0:
            normal = tuple(-n for n in normal)
        if abs(abs(normal[0])-abs(normal[1])) > .08:
            raise ValueError("Observed paper edges do not describe the declared diamond orientation")
        lines.append({"normal": normal, "mean": mean, "points": group})
    remaining, pairs = set(range(4)), []
    while remaining:
        i = min(remaining)
        remaining.remove(i)
        j = min(remaining, key=lambda k: sum(a*b for a, b in zip(lines[i]["normal"], lines[k]["normal"], strict=True)))
        remaining.remove(j)
        ni, nj = lines[i]["normal"], lines[j]["normal"]
        if sum(a*b for a, b in zip(ni, nj, strict=True)) > -.998:
            raise ValueError("Observed diamond edges do not have independent opposing sides")
        norm = math.hypot(ni[0]-nj[0], ni[1]-nj[1])
        normal = tuple((a-b)/norm for a, b in zip(ni, nj, strict=True))
        intercepts = [sum(n*m for n, m in zip(normal, lines[k]["mean"], strict=True)) for k in (i, j)]
        residuals = [abs(sum(n*p for n, p in zip(normal, point, strict=True))-intercept)
                     for k, intercept in zip((i, j), intercepts, strict=True) for point in lines[k]["points"]]
        pairs.append({"normal": normal, "intercepts": intercepts, "residuals": residuals})
    a, b = pairs
    if abs(sum(x*y for x, y in zip(a["normal"], b["normal"], strict=True))) > .05:
        raise ValueError("Observed diamond edge pairs are not perpendicular")
    determinant = a["normal"][0]*b["normal"][1]-a["normal"][1]*b["normal"][0]
    mids = [statistics.mean(pair["intercepts"]) for pair in pairs]
    center = ((mids[0]*b["normal"][1]-mids[1]*a["normal"][1])/determinant,
              (a["normal"][0]*mids[1]-b["normal"][0]*mids[0])/determinant)
    widths = [pair["intercepts"][0]-pair["intercepts"][1] for pair in pairs]
    residuals = a["residuals"]+b["residuals"]
    maximum = max(residuals)
    resolution = math.hypot(*calibration["mm_per_pixel"])
    uncertainty = maximum+resolution
    if maximum > max_residual_mm or min(widths) <= 0 or abs(widths[0]-widths[1]) > 2*uncertainty:
        raise ValueError("Aperture edges are occluded, ambiguous or inconsistent with one square diamond")
    return {"shape": "diamond", "center_offset_mm": list(center), "inradius_mm": statistics.mean(widths)/2,
            "opposing_widths_mm": widths, "edge_lines": [{"normal": list(pair["normal"]),
                "intercepts_mm": pair["intercepts"]} for pair in pairs],
            "max_residual_mm": maximum, "rms_residual_mm": math.sqrt(statistics.mean(v*v for v in residuals)),
            "uncertainty_mm": uncertainty, "edge_points": sum(map(len, groups)),
            "arc_points": list(map(len, groups)), "frame": "baseline_holder_camera_parallel_axes", "units": "mm",
            "uncertainty_scope": "Observed edge residual plus one pixel diagonal; heuristic, not physical metrology"}


def validate_aperture_alignment(store, run_id, body):
    """Recheck immutable exposure/encoder receipts and raster fits on resume."""
    if body.get("profile") not in (PROFILE, LEGACY_PROFILE) or body.get("run_id") != run_id or body.get("status") != "observed":
        raise ValueError("Aperture alignment is incomplete or belongs to another process")
    shape = "diamond" if body["profile"] == PROFILE else "circle"
    policy = ApertureScan(**{**body["config"], "shape": shape})
    receipts = store.events_after(body["receipt_event_id"]-1, limit=1)
    original_body = {k: v for k, v in body.items() if k not in ("receipt_event_id", "receipt_sha256")}
    if (not receipts or receipts[0].id != body["receipt_event_id"]
            or receipts[0].kind != "stack.aperture_alignment" or receipts[0].run_id != run_id
            or receipts[0].payload != original_body or geometry_digest(original_body) != body["receipt_sha256"]):
        raise ValueError("Aperture alignment receipt changed")
    records = body["frames"]
    if len(records) != 9:
        raise ValueError("Aperture alignment lost required exposure evidence")
    previous = 0
    for record in records:
        captures = store.events_after(record["event_id"]-1, limit=1)
        capture = captures[0] if captures else None
        preview = capture.payload.get("microscope_preview", {}) if capture else {}
        if (not capture or capture.id != record["event_id"] or capture.kind != "lab.frame" or capture.run_id != run_id
                or capture.payload.get("device") != "microscope" or capture.payload.get("command") != "capture"
                or geometry_digest(capture.payload) != record["event_sha256"]
                or capture.ts != record.get("capture_event_ts")
                or preview.get("image") != record["image"] or preview.get("image_sha256") != record["sha256"]
                or preview.get("timestamp") != record["captured_at"] or preview.get("calibration") != record["calibration"]
                or preview.get("objective") != 5 or preview.get("phase") != body.get("step_name", "center_stamp_aperture")
                or not previous < capture.id < record["encoder_event_id"]
                or not body["started_at"] <= record["captured_at"] <= body["ended_at"]):
            raise ValueError("Original aperture capture event changed or is out of order")
        events = store.events_after(record["encoder_event_id"]-1, limit=1)
        original = {k: v for k, v in record.items() if k not in ("encoder_event_id", "encoder_event_sha256")}
        if (not events or events[0].id != record["encoder_event_id"]
                or events[0].kind != "stack.aperture_exposure" or events[0].run_id != run_id
                or events[0].payload != original or geometry_digest(original) != record["encoder_event_sha256"]
                or hashlib.sha256(resolve_run_path(record["image"]).read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError("Aperture image/encoder exposure evidence changed")
        previous = record["encoder_event_id"]
    first = fit_aperture(records[:4], body["baseline_holder_xy_mm"], max_residual_mm=policy.max_residual_mm, shape=shape)
    second = fit_aperture(records[4:8], body["centered_holder_xy_mm"], max_residual_mm=policy.max_residual_mm, shape=shape)
    target = [v-d for v, d in zip(body["baseline_holder_xy_mm"], first["center_offset_mm"], strict=True)]
    if (first != body["initial_fit"] or second != body["confirmation_fit"]
            or target != body["centered_holder_xy_mm"]
            or math.hypot(*second["center_offset_mm"])+second["uncertainty_mm"] > policy.center_tolerance_mm):
        raise ValueError("Aperture raster fit no longer proves the recorded alignment")
    focus = store.events_after(body["wafer_focus_event_id"]-1, limit=1)
    if (not focus or focus[0].id != body["wafer_focus_event_id"] or focus[0].kind != "microscope.focus"
            or focus[0].run_id != run_id or focus[0].payload.get("objective") != 5
            or focus[0].payload.get("plane") != "chip" or focus[0].payload.get("peak_observable") is not True
            or focus[0].payload.get("holder_pose") != records[-1]["holder_pose"]
            or geometry_digest(focus[0].payload) != body.get("wafer_focus_event_sha256")
            or focus[0].ts != body.get("wafer_focus_event_ts")
            or body["final_frame_event_id"] != records[-1]["event_id"]
            or not previous < focus[0].id < body["receipt_event_id"]
            or not records[-1]["captured_at"] <= focus[0].ts <= body["ended_at"]):
        raise ValueError("Aperture wafer-refocus observation changed")
    return body


class CenterStampAperture(Step):
    """A finite clear-height pixel scan, followed by independent edge confirmation."""
    devices = ("transfer_arm", "microscope")
    safe_to_resume = False
    pause_inside = True
    max_corrections = 0

    def __init__(self, stamp_id, name="center_stamp_aperture", *, config=None):
        self.stamp_id, self.name = stamp_id, name
        self.config = config or ApertureScan()

    def preconditions(self, ctx):
        from qumus_simlab.records.geometry import DirectContactContract

        direct = False
        if ctx.params.get("release_policy") == "native20-observed-loaded-support-parked-release-v3" and ctx.params.get("direct_contact_contract"):
            try:
                contract = DirectContactContract.model_validate(ctx.params["direct_contact_contract"])
                direct = (ctx.params.get("flake_ids") == [contract.source_flake_id]
                          and ctx.params.get("materials") == {contract.source_flake_id: "Gr"}
                          and ctx.params.get("stamp_film") == "PC")
            except (ValueError, TypeError):
                pass
        if ctx.store.bench != "sim" or not (ctx.params.get("device_design") or direct):
            raise PreconditionFailed("Observed aperture centering requires the explicit virtual design/direct release process")
        phases = (ctx.store.get_run(ctx.run_id) or {}).get("meta", {}).get("normal_transfer", {}).get("phases", {})
        if any(name.startswith("v1_pick_up_") for name in phases):
            raise PreconditionFailed("Aperture scanning cannot run after any loaded/unknown pickup attempt")
        if ctx.verification["transfer_sequence"].get("holder_xy_mapping") != "synthetic-parallel-world-axes-v1":
            raise PreconditionFailed("Aperture scanning requires its registered holder/camera axis mapping")
        if ctx.params.get("stamp_id") != self.stamp_id:
            raise PreconditionFailed("Aperture scan must use the selected stamp identity")
        if self.config.shape != "diamond":
            raise PreconditionFailed("New aperture scans require the observed diamond opening; circles are legacy replay only")
        pose = ctx.gate.manipulator("transfer_arm").get_positions()
        safe = ctx.bench["transfer_arm"]["safe_z"]
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in pose.values())
                or pose["z"] < safe-ctx.verification["manipulator"]["tolerance_mm"]):
            raise PreconditionFailed("Aperture scanning requires the actual declared clear height")
        if ctx.gate.microscope("microscope").get_objective() != 5:
            raise PreconditionFailed("Aperture scanning requires the observed 5x objective")

    def _guard(self, ctx):
        ctx.check_cancel()
        ctx.gate.watchdog_tick()
        if ctx.clock.now()-self.started > self.config.max_seconds:
            raise NeedsHumanError("Finite aperture scan time exhausted")

    def _move(self, ctx, xy):
        self._guard(ctx)
        arm = ctx.gate.manipulator("transfer_arm")
        tolerance = ctx.verification["manipulator"]["tolerance_mm"]
        for axis, target in zip(("x", "y"), xy, strict=True):
            self._guard(ctx)
            pose = arm.get_positions()
            if pose["z"] < ctx.bench["transfer_arm"]["safe_z"]-tolerance:
                raise NeedsHumanError("Aperture clear-height readback changed")
            if abs(target-self.origin[0 if axis == "x" else 1]) > self.config.max_extent_mm:
                raise NeedsHumanError("Aperture scan exceeds its finite search extent")
            if abs(pose[axis]-target) > 1e-8:
                bound = arm.axis_motion_duration_bound(axis, pose[axis], target, self.config.speed_mm_s)
                if bound is None or not math.isfinite(bound) or ctx.clock.now()-self.started+bound > self.config.max_seconds:
                    raise NeedsHumanError("Aperture rail move lacks its finite duration allowance")
                arm.move_axis_to(axis, target, speed=self.config.speed_mm_s)
            if abs(arm.get_positions()[axis]-target) > tolerance:
                raise NeedsHumanError("Aperture rail encoder did not reach its request")

    def _capture(self, ctx):
        from qumus_simlab.flows.library.primitives import microscope_frame

        self._guard(ctx)
        if len(self.frames) >= self.config.max_captures:
            raise NeedsHumanError("Finite aperture capture count exhausted")
        arm = ctx.gate.manipulator("transfer_arm")
        # Registered adapter geometry is a read-only Gate input, as in the
        # swept-clearance check. Commands continue through the guarded wrapper.
        geometry = ctx.gate.devices.get("microscope").geometry_state
        before, camera = arm.get_positions(), geometry()["optical_axis_world_mm"]
        cursor = ctx.store.last_event_id()
        frame = microscope_frame(ctx, phase=self.name)
        if (before != arm.get_positions() or camera != geometry()["optical_axis_world_mm"]
                or frame.meta.get("objective") != 5):
            raise NeedsHumanError("Aperture exposure encoder/camera context changed")
        path = frame.saved_path()
        if not path:
            raise NeedsHumanError("Aperture alignment requires saved exposure pixels")
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        events = [e for e in ctx.store.events_after(cursor, limit=32, run_id=ctx.run_id)
                  if e.kind == "lab.frame" and e.payload.get("microscope_preview", {}).get("phase") == self.name
                  and e.payload["microscope_preview"].get("image_sha256") == sha]
        if len(events) != 1:
            raise NeedsHumanError("Aperture exposure lacks a unique immutable capture event")
        record = {"image": path, "sha256": sha, "captured_at": frame.timestamp,
            "calibration": deepcopy(frame.meta["calibration"]), "holder_xy_mm": [before["x"], before["y"]],
            "holder_pose": before, "camera_world_mm": list(camera), "event_id": events[0].id,
            "capture_event_ts": events[0].ts,
            "event_sha256": geometry_digest(events[0].payload)}
        encoder_event = ctx.store.emit("stack.aperture_exposure", record, run_id=ctx.run_id)
        record = {**record, "encoder_event_id": encoder_event,
                  "encoder_event_sha256": geometry_digest(record)}
        self.frames.append(record)
        return record

    def _scan(self, ctx, baseline):
        records = []
        offset = self.config.offset_mm/math.sqrt(2)
        for dx, dy in ((offset, offset), (-offset, -offset), (offset, -offset), (-offset, offset)):
            self._move(ctx, (baseline[0]+dx, baseline[1]+dy))
            records.append(self._capture(ctx))
        return fit_aperture(records, baseline, max_residual_mm=self.config.max_residual_mm, shape="diamond")

    def act(self, ctx, correction):
        from qumus_simlab.flows.library.primitives import autofocus

        self.preconditions(ctx)
        self.started, self.frames = ctx.clock.now(), []
        pose = ctx.gate.manipulator("transfer_arm").get_positions()
        self.origin = (pose["x"], pose["y"])
        state = deepcopy(ctx.store.get_run(ctx.run_id)["meta"].get("normal_transfer") or {})
        if state.get("aperture_alignment"):
            raise NeedsHumanError("An existing/interrupted aperture scan requires review, not blind repetition")
        body = {"profile": PROFILE, "step_name": self.name, "aperture_shape": "diamond",
                "stamp_id": self.stamp_id, "run_id": ctx.run_id,
                "baseline_holder_xy_mm": list(self.origin),
                "config": asdict(self.config), "started_at": self.started, "status": "in_progress"}
        state["aperture_alignment"] = body
        ctx.store.update_run_meta(ctx.run_id, normal_transfer=state)
        try:
            # Preflight every proposed rail envelope through the same Gate before
            # the first lateral command; no safety limits are changed.
            limits = ctx.gate.limits.manipulators["transfer_arm"]
            for axis, value in zip(("x", "y"), self.origin, strict=True):
                if not getattr(limits, axis)[0] <= value-self.config.max_extent_mm <= value+self.config.max_extent_mm <= getattr(limits, axis)[1]:
                    raise NeedsHumanError("The bounded aperture sweep does not fit actual axis limits")
            for dx, dy in ((self.config.max_extent_mm, 0), (-self.config.max_extent_mm, 0),
                           (0, self.config.max_extent_mm), (0, -self.config.max_extent_mm)):
                ctx.gate.check_swept_geometry("transfer_arm", "move_axis_to",
                    stamp_target={**pose, "x": pose["x"]+dx, "y": pose["y"]+dy})
            first = self._scan(ctx, self.origin)
            target = tuple(v-d for v, d in zip(self.origin, first["center_offset_mm"], strict=True))
            self._move(ctx, target)
            second = self._scan(ctx, target)
            if math.hypot(*second["center_offset_mm"])+second["uncertainty_mm"] > self.config.center_tolerance_mm:
                raise NeedsHumanError("Fresh aperture edges do not confirm centering within the declared raster tolerance")
            if abs(first["inradius_mm"]-second["inradius_mm"]) > first["uncertainty_mm"]+second["uncertainty_mm"]:
                raise NeedsHumanError("Independent aperture edge-distance observations disagree")
            self._move(ctx, target)
            final = self._capture(ctx)
            # Centre-view transmission alone is never centering evidence.
            autofocus(ctx, 5, attempts=1, half_range=.15, plane="chip")
            focus = ctx.store.get_run(ctx.run_id)["meta"].get("last_focus_observation") or {}
            if not focus.get("peak_observable"):
                raise NeedsHumanError("Wafer refocus after aperture centering is not optically observable")
            self._guard(ctx)
            focus_id = ctx.store.get_run(ctx.run_id)["meta"]["last_focus_event_id"]
            focus_event = ctx.store.events_after(focus_id-1, limit=1)[0]
            body.update(status="observed", initial_fit=first, confirmation_fit=second,
                        centered_holder_xy_mm=list(target), final_frame_event_id=final["event_id"],
                        wafer_focus_event_id=focus_id, wafer_focus_event_sha256=geometry_digest(focus_event.payload),
                        wafer_focus_event_ts=focus_event.ts,
                        ended_at=ctx.clock.now())
        except ValueError as exc:
            body.update(status="unresolved", reason=str(exc))
            raise NeedsHumanError(str(exc)) from exc
        finally:
            body["frames"] = self.frames
            current = deepcopy(ctx.store.get_run(ctx.run_id)["meta"].get("normal_transfer") or {})
            current["aperture_alignment"] = body
            ctx.store.update_run_meta(ctx.run_id, normal_transfer=current)
        receipt_event = ctx.store.emit("stack.aperture_alignment", body, run_id=ctx.run_id)
        body = {**body, "receipt_event_id": receipt_event, "receipt_sha256": geometry_digest(body)}
        current = deepcopy(ctx.store.get_run(ctx.run_id)["meta"].get("normal_transfer") or {})
        current["aperture_alignment"] = body
        ctx.store.update_run_meta(ctx.run_id, normal_transfer=current)
        self.result = body

    def verify(self, ctx):
        body = getattr(self, "result", None)
        if not body or body.get("status") != "observed":
            return Observation(Verdict.UNCLEAR, message="Aperture centering has no observed completion")
        try:
            validate_aperture_alignment(ctx.store, ctx.run_id, body)
        except (ValueError, OSError) as exc:
            raise NeedsHumanError("Aperture evidence changed before verification") from exc
        pose = ctx.gate.manipulator("transfer_arm").get_positions()
        if math.dist((pose["x"], pose["y"]), body["centered_holder_xy_mm"]) > ctx.verification["manipulator"]["tolerance_mm"]:
            raise NeedsHumanError("Aperture centered encoder pose changed")
        return Observation(Verdict.OK, body, images=[r["image"] for r in body["frames"]],
                           message="Diamond aperture centered from independent saved edges; wafer refocus observed")

    def probe_state(self, ctx):
        return ProbeResult.UNKNOWN
