"""Chip search (graphene path, part 2), ported from v1
processing_agent/workflows/search/ (AssemblySearchChip, MoleculePickupChipToTransferStage,
MoleculeCornerDetection, MoleculeAutoSearch, MoleculeStoreSearchSiliconChip).

    transfer_stage_to_load, place_chip_on_transfer_stage, chip_to_microscope,
    find_chip_corner, measure_chip_rotation, scan_chip (10x snake scan), revisit_candidates (20x/100x),
    transfer_stage_to_load, store_chip_from_transfer_stage

`revisit_chip` is the same without the scan: it puts a searched chip back
under the microscope and photographs, at 20x/100x, candidates the ledger
already has but that were never revisited (found again from their position
relative to the chip corner).

What changed from v1:
- placement on the transfer stage is measured and corrected (spec 4.1);
- the corner search knows where the corner should be (from the measured chip
  pose) and spirals out if it is not in view, instead of failing;
- every frame is focus-checked; blurred tiles are refocused and, if still
  blurred, rescanned as a correction;
- candidates and catalogued flakes go to the sample ledger (kind "flake",
  location = chip id) with stage coordinates relative to the chip corner, so
  stacking can find them again after the chip is moved;
- which candidates are revisited follows the device plan: each flake role has
  its own slots (as v1 kept 15 graphene targets and 5 graphite gates apart),
  so one abundant kind cannot take every slot; without a plan v1's quotas.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.core.materials import normalize_material_code
from qumus_simlab.core.types import Pose, angle_diff
from qumus_simlab.flows.bench import is_slot, reachable, slot_station
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.flow import Flow, register_flow
from qumus_simlab.flows.library.place_chip import PlaceChipOnTransferStage
from qumus_simlab.flows.library.primitives import (
    MICROSCOPE,
    analyzer,
    autofocus,
    lift,
    microscope_frame,
    min_sharpness,
    observed_outline_pixels,
    publish_frame,
    require_sample,
)
from qumus_simlab.flows.library.transfer import MoveObject
from qumus_simlab.flows.step import Correction, Observation, ProbeResult, Step, Verdict
from qumus_simlab.perception.base import FlakeDetection

ARM = "arm_right"
CAMERA = "overview_camera"
STAGE = "transfer_stage"
STAGE_VACUUM = "transfer_stage_vacuum"

#: v1 MoleculeAutoSearch preferred labels: graphene targets first, then graphite gates.
#: Fills the revisit slots the quotas leave free.
PREFERRED = {"Gr": [["monolayer"], ["bilayer"], ["manylayer"]],
             "hBN": [["10-20", "20-30"], ["0-10", "30-40"]]}


class RevisitRole(BaseModel):
    """Revisit slots kept for one kind of flake (a device-plan role)."""

    role: str
    material: str | None = None  # None = any material (the chip's)
    thickness_labels: list[str] = Field(default_factory=list)  # 10x labels; empty = any
    min_area_um2: float = Field(0.0, ge=0)
    slots: int = Field(ge=0)


def default_quota(material: str) -> list[RevisitRole]:
    """v1 _select_revisit_targets without a device plan: graphene 15 monolayer
    targets plus 5 manylayer (graphite gates); hBN 4 per thickness bin so one
    bin cannot take the others' slots."""
    if material == "Gr":
        return [RevisitRole(role="graphene_target", thickness_labels=["monolayer"], slots=15),
                RevisitRole(role="graphite_gate", thickness_labels=["manylayer"], slots=5)]
    if material == "hBN":
        return [RevisitRole(role=f"hbn_{b}", thickness_labels=[b], slots=4)
                for b in ("0-10", "10-20", "20-30", "30-40")]
    return []


def pick_revisits(cands: list[dict[str, Any]], material: str,
                  quota: list[RevisitRole] | None, max_n: int) -> list[dict[str, Any]]:
    """Which candidates to revisit, at most `max_n`: each role's slots first
    (largest flakes of its labels and minimum area), then free slots to the
    other flakes the roles would take, then PREFERRED labels, largest first."""
    roles = [r for r in (quota if quota is not None else default_quota(material))
             if r.material is None or normalize_material_code(r.material) == material]

    def label(c: dict[str, Any]) -> str | None:
        return c["props"].get("labels", {}).get("10x") or c["props"].get("thickness_label")

    def fits(c: dict[str, Any], r: RevisitRole) -> bool:
        return ((not r.thickness_labels or label(c) in r.thickness_labels)
                and c["props"].get("area_um2", 0) >= r.min_area_um2)

    by_area = sorted(cands, key=lambda c: -c["props"].get("area_um2", 0))
    chosen: list[dict[str, Any]] = []
    taken: set[str] = set()

    def take(pool: list[dict[str, Any]], n: int) -> None:
        for c in pool:
            if n <= 0 or len(chosen) >= max_n:
                return
            if c["id"] not in taken:
                chosen.append(c)
                taken.add(c["id"])
                n -= 1

    for r in roles:
        take([c for c in by_area if fits(c, r)], r.slots)
    take([c for c in by_area if any(fits(c, r) for r in roles)], max_n)
    for labels in PREFERRED.get(material, []):
        take([c for c in by_area if label(c) in labels], max_n)
    return chosen


def _stage_pose(ctx: FlowContext, name: str) -> Pose:
    return Pose(**ctx.bench["transfer_stage"][name])


def _chip_geometry(ctx: FlowContext, chip_id: str) -> tuple[Pose, tuple[float, float]]:
    """Measured chip pose on the transfer stage (world frame, stage at load) and size."""
    chip = ctx.store.get_sample(chip_id) or {"props": {}}
    st = ctx.bench["stations"][STAGE]
    pose = chip["props"].get("pose_on_stage") or {"x": st["x"], "y": st["y"]}
    size = chip["props"].get("size_mm") or [10.0, 10.0]
    return Pose(**pose), (float(size[0]), float(size[1]))


def _chip_rotation(ctx: FlowContext, chip_id: str) -> float:
    """Chip rotation on the stage: from two corners under the microscope when
    measured for this placement, else the overview camera's (~0.2 deg noise,
    i.e. tens of um over a chip)."""
    chip = ctx.store.get_sample(chip_id) or {"props": {}}
    rot = chip["props"].get("rotation_deg")
    return float(rot) if rot is not None else _chip_geometry(ctx, chip_id)[0].rotation


def _local_to_stage(corner_stage: tuple[float, float], rotation: float,
                    lx: float, ly: float) -> tuple[float, float]:
    """Stage position that puts chip-local point (lx, ly) (mm from the BL corner,
    chip axes) under the objective. Moving the stage by +v moves the chip by +v,
    so a point offset by v from the corner needs the stage at corner - v."""
    r = math.radians(rotation)
    vx, vy = lx * math.cos(r) - ly * math.sin(r), lx * math.sin(r) + ly * math.cos(r)
    return corner_stage[0] - vx, corner_stage[1] - vy


def _stage_to_local(corner_stage: tuple[float, float], rotation: float,
                    sx: float, sy: float) -> tuple[float, float]:
    """Chip-local point (mm from the BL corner) under the objective with the
    stage at (sx, sy); inverse of _local_to_stage."""
    vx, vy = corner_stage[0] - sx, corner_stage[1] - sy
    r = math.radians(rotation)
    return vx * math.cos(r) + vy * math.sin(r), -vx * math.sin(r) + vy * math.cos(r)


def _move_stage(ctx: FlowContext, x: float, y: float, z: float | None = None) -> None:
    """Translate to composed sample XY without changing fine axes or theta."""
    ctx.check_cancel()
    st = ctx.gate.stage(STAGE)
    coarse = st.get_pose()
    fx, fy = st.get_fine_position()
    st.move_to(Pose(x=x - fx, y=y - fy, z=coarse.z if z is None else z,
                    rotation=coarse.rotation))


class MoveTransferStage(Step):
    """Move the transfer stage to a named position; encoder readback checked."""

    devices = (STAGE, STAGE_VACUUM, ARM)

    def __init__(self, to: str, vacuum: bool | None = None, name: str | None = None) -> None:
        self.to, self.vacuum = to, vacuum
        self.name = name or f"transfer_stage_to_{to}"

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        p = _stage_pose(ctx, self.to)
        return {"x": p.x, "y": p.y, "tolerance_mm": ctx.verification[STAGE]["tolerance_mm"]}

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        lift(ctx, ARM)
        if self.vacuum is not None:
            ctx.gate.switch(STAGE_VACUUM).set(self.vacuum)
        p = _stage_pose(ctx, self.to)
        _move_stage(ctx, p.x, p.y, p.z)

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        p = ctx.gate.stage(STAGE).get_sample_pose()
        err = math.hypot(p.x - exp["x"], p.y - exp["y"])
        m = {"x": p.x, "y": p.y, "error_mm": round(err, 4)}
        if err <= exp["tolerance_mm"]:
            return Observation(Verdict.OK, m, exp)
        return Observation(Verdict.CORRECTABLE, m, exp, message=f"stage off by {err:.3f} mm")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return Correction("move again")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE  # moving again is harmless


class ChipToMicroscope(MoveTransferStage):
    """Stage vacuum on, carry the chip under the microscope; the microscope must see it."""

    def __init__(self, chip_id: str) -> None:
        super().__init__("microscope", vacuum=True, name="chip_to_microscope")
        self.chip_id = chip_id
        self.devices = (STAGE, STAGE_VACUUM, ARM, MICROSCOPE)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        super().act(ctx, correction)
        m = ctx.gate.microscope(MICROSCOPE)
        m.set_objective(ctx.bench["microscope"]["scan_objective"])  # the stage z: nominal focus

    def verify(self, ctx: FlowContext) -> Observation:
        obs = super().verify(ctx)
        if obs.verdict != Verdict.OK:
            return obs
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        obs.measurements["chip_in_view"] = analyzer(ctx).chip_in_view(frame)
        if not obs.measurements["chip_in_view"]:
            return Observation(Verdict.UNCLEAR, obs.measurements, obs.expected,
                               message="no chip under the microscope")
        return obs


class FindChipCorner(Step):
    """Find the chip's bottom-left corner under the microscope (v1
    MoleculeCornerDetection). The corner gives the chip's position in stage
    coordinates; flake positions are stored relative to it."""

    name = "find_chip_corner"
    devices = (STAGE, MICROSCOPE)

    def __init__(self, chip_id: str) -> None:
        self.chip_id = chip_id
        self._nominal: tuple[float, float] | None = None
        self._tries = 0

    def _cfg(self, ctx: FlowContext) -> dict[str, Any]:
        return ctx.verification["chip_corner"]

    def nominal(self, ctx: FlowContext) -> tuple[float, float]:
        """Where the corner should be (stage coordinates), from the measured placement."""
        if self._nominal is None:
            pose, (w, h) = _chip_geometry(ctx, self.chip_id)
            mic = ctx.bench["microscope"]["center"]
            load = _stage_pose(ctx, "load")
            r = math.radians(pose.rotation)
            cx = pose.x + (-w / 2) * math.cos(r) - (-h / 2) * math.sin(r)
            cy = pose.y + (-w / 2) * math.sin(r) + (-h / 2) * math.cos(r)
            # The chip pose was measured with the stage at load; moving the stage
            # by v moves the corner by v, and it must end up under the objective.
            self._nominal = (load.x + mic["x"] - cx, load.y + mic["y"] - cy)
        return self._nominal

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        nx, ny = self.nominal(ctx)
        return {"corner_stage_xy": [round(nx, 3), round(ny, 3)],
                "tolerance_mm": self._cfg(ctx)["tolerance_mm"]}

    def preconditions(self, ctx: FlowContext) -> None:
        self.max_corrections = self._cfg(ctx)["max_corrections"]
        require_sample(ctx, self.chip_id, "chip")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        nx, ny = self.nominal(ctx)
        ox, oy = correction.params["offset"] if correction else (0.0, 0.0)
        obj = ctx.bench["microscope"]["scan_objective"]
        ctx.gate.microscope(MICROSCOPE).set_objective(obj)
        # The observed corner edge supplies focus contrast even on a textureless chip.
        _move_stage(ctx, nx + ox, ny + oy)
        autofocus(ctx, obj)

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        det = analyzer(ctx).find_corner(frame, "bl")
        s = ctx.gate.stage(STAGE).get_sample_pose()
        if det is None:
            return Observation(Verdict.CORRECTABLE, {"found": False, "stage_xy": [s.x, s.y]},
                               exp, message="corner not in view")
        cx, cy = s.x - det.dx_mm, s.y - det.dy_mm
        nx, ny = self.nominal(ctx)
        off = math.hypot(cx - nx, cy - ny)
        m = {"found": True, "corner_stage_xy": [round(cx, 4), round(cy, 4)],
             "offset_from_nominal_mm": round(off, 3)}
        if off <= exp["tolerance_mm"]:
            path = frame.saved_path()
            return Observation(Verdict.OK, m, exp, images=[path] if path else [])
        return Observation(Verdict.UNCLEAR, m, exp,
                           message=f"corner {off:.2f} mm from where the placement puts it")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        # Every offset is relative to the same nominal corner, so grow only
        # after a complete symmetric four-direction ring (v1 had no search).
        self._tries += 1
        k = self._tries
        step = self._cfg(ctx)["search_step_mm"] * ((k - 1) // 4 + 1)
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][(k - 1) % 4]
        return Correction("look around for the corner", {"offset": (dx * step, dy * step)})

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        props: dict[str, Any] = {}
        if obs:
            props["corner_stage_xy"] = obs.measurements["corner_stage_xy"]
            props["corner_image"] = obs.images[0] if obs.images else None
        ctx.store.upsert_sample(self.chip_id, "chip", props=props)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE  # only looks; redo


class MeasureChipRotation(Step):
    """Find the bottom-right corner too: the two corners give the chip's
    rotation to ~0.03 deg (the overview camera's 0.2 deg would put a flake
    7 mm from the corner 25 um off). v1 relied on image matching instead."""

    devices = (STAGE, MICROSCOPE)

    def __init__(self, chip_id: str, name: str = "measure_chip_rotation") -> None:
        self.chip_id, self.name = chip_id, name
        self._tries = 0

    def _br_nominal(self, ctx: FlowContext) -> tuple[float, float]:
        chip = require_sample(ctx, self.chip_id, "chip")
        bl = chip["props"]["corner_stage_xy"]
        pose, (w, _) = _chip_geometry(ctx, self.chip_id)
        return _local_to_stage((bl[0], bl[1]), pose.rotation, w, 0.0)

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        pose, (w, _) = _chip_geometry(ctx, self.chip_id)
        return {"camera_rotation_deg": round(pose.rotation, 3), "max_difference_deg": 3.0,
                "edge_mm": w, "edge_tolerance_mm": 0.3}

    def preconditions(self, ctx: FlowContext) -> None:
        self.max_corrections = ctx.verification["chip_corner"]["max_corrections"]
        chip = require_sample(ctx, self.chip_id, "chip")
        if not chip["props"].get("corner_stage_xy"):
            raise PreconditionFailed(f"{self.chip_id} has no corner position (find it first)")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        x, y = self._br_nominal(ctx)
        ox, oy = correction.params["offset"] if correction else (0.0, 0.0)
        obj = ctx.bench["microscope"]["scan_objective"]
        ctx.gate.microscope(MICROSCOPE).set_objective(obj)
        _move_stage(ctx, x + ox + 0.4, y + oy - 0.3)  # focus just inside the corner
        autofocus(ctx, obj)
        _move_stage(ctx, x + ox, y + oy)

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        det = analyzer(ctx).find_corner(frame, "br")
        if det is None:
            return Observation(Verdict.CORRECTABLE, {"found": False}, exp,
                               message="bottom-right corner not in view")
        s = ctx.gate.stage(STAGE).get_sample_pose()
        bl = (ctx.store.get_sample(self.chip_id) or {"props": {}})["props"]["corner_stage_xy"]
        vx, vy = (s.x - det.dx_mm) - bl[0], (s.y - det.dy_mm) - bl[1]
        rot = math.degrees(math.atan2(-vy, -vx))  # stage moves opposite to the chip
        edge = math.hypot(vx, vy)
        m = {"found": True, "rotation_deg": round(rot, 4), "edge_mm": round(edge, 4)}
        if (abs(angle_diff(rot, exp["camera_rotation_deg"])) <= exp["max_difference_deg"]
                and abs(edge - exp["edge_mm"]) <= exp["edge_tolerance_mm"]):
            return Observation(Verdict.OK, m, exp)
        return Observation(Verdict.UNCLEAR, m, exp, message=f"corners give {rot:.2f} deg and a "
                           f"{edge:.2f} mm edge; does not fit the chip")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        self._tries += 1
        k = self._tries
        step = ctx.verification["chip_corner"]["search_step_mm"] * ((k + 1) // 2)
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][(k - 1) % 4]
        return Correction("look around for the corner", {"offset": (dx * step, dy * step)})

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        if obs is not None:
            ctx.store.upsert_sample(self.chip_id, "chip",
                                    props={"rotation_deg": obs.measurements["rotation_deg"]})

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


def _last_verified(store: Any, run_id: str, name: str) -> Any:
    """The run's last OK step.verified event of this step, or None."""
    found, after = None, 0
    while batch := store.events_after(after, limit=500, run_id=run_id):
        for e in batch:
            if (e.kind == "step.verified" and e.payload.get("name") == name
                    and e.payload.get("verdict") == "ok"):
                found = e
        after = batch[-1].id
    return found


class MeasureCornerRotation(Step):
    """v1 rough refind's angle (molecule_flake_refind_rough): ORB-match the
    bottom-left corner photo taken when the chip was searched against the one
    just taken by find_corner, and add the change to the search-time rotation
    the flake positions were recorded with. No second corner, no stage move."""

    devices = ()
    max_corrections = 0

    def __init__(self, chip_id: str, flake_id: str, name: str) -> None:
        self.chip_id, self.flake_id, self.name = chip_id, flake_id, name

    def _images(self, ctx: FlowContext) -> tuple[str, str, float, str]:
        flake = ctx.store.get_sample(self.flake_id) or {"props": {}}
        job = (flake["props"].get("tracking_references", {}).get("20x") or {}).get("job_id")
        if not job:
            raise PreconditionFailed(f"{self.flake_id} has no search run to take the corner photo from")
        ref, rot = _last_verified(ctx.store, job, "find_chip_corner"), _last_verified(
            ctx.store, job, "measure_chip_rotation")
        cur = _last_verified(ctx.store, ctx.run_id, f"find_corner_{self.chip_id}")
        if ref is None or rot is None or cur is None:
            raise PreconditionFailed(f"{self.chip_id} needs its search-time and current corner photos")
        ref_img, cur_img = (ref.payload.get("images") or [None])[0], (cur.payload.get("images") or [None])[0]
        r = rot.payload.get("measurements", {}).get("rotation_deg")
        if not ref_img or not cur_img or not Path(ref_img).is_file() or not Path(cur_img).is_file() or r is None:
            raise PreconditionFailed(f"{self.chip_id} corner photos or search rotation are missing")
        return ref_img, cur_img, float(r), job

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        from qumus_simlab.perception import v1_vision
        return {"min_good_matches": v1_vision.MIN_GOOD_MATCHES, "min_inliers": v1_vision.MIN_INLIERS}

    def preconditions(self, ctx: FlowContext) -> None:
        require_sample(ctx, self.chip_id, "chip")
        self._images(ctx)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        pass  # the stage stays at the corner find_corner just photographed

    def verify(self, ctx: FlowContext) -> Observation:
        import cv2

        from qumus_simlab.perception import v1_vision
        exp = self.expected(ctx)
        ref_img, cur_img, search_rot, job = self._images(ctx)
        res = v1_vision.corner_rotation(cv2.imread(ref_img), cv2.imread(cur_img))
        m: dict[str, Any] = {"found": res.found, "good_matches": res.good_matches,
                             "inliers": res.inliers, "search_rotation_deg": search_rot,
                             "search_run": job, "reference_image": ref_img}
        if not res.found:
            return Observation(Verdict.UNCLEAR, m, exp, images=[cur_img],
                               message=f"corner photos do not match ({res.loss_reason})")
        # theta_cw is clockwise on screen (image v points to -world_y), i.e.
        # minus the chip's world counter-clockwise rotation change.
        m["theta_cw_deg"] = round(math.degrees(res.theta_cw_rad), 4)
        m["rotation_deg"] = round((search_rot - m["theta_cw_deg"] + 180) % 360 - 180, 4)
        return Observation(Verdict.OK, m, exp, images=[cur_img])

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        if obs is not None:
            ctx.store.upsert_sample(self.chip_id, "chip",
                                    props={"rotation_deg": obs.measurements["rotation_deg"]})

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


def _centre_edge_clearance_mm(candidate: dict[str, Any], size: tuple[float, float]) -> float:
    """Distance from a detection's centre to the nearest side of its chip
    (negative when the centre is off the chip)."""
    x, y = candidate["rel_corner_mm"]
    return min(x, y, size[0] - x, size[1] - y)


def _source_edge_clearance_mm(ctx: FlowContext) -> float:
    """The pickup rule's chip-edge clearance (same default as the pair search)."""
    return float(ctx.verification.get("pickup", {}).get("source_chip_edge_clearance_mm", 1.0))


def _snake(nx: int, ny: int) -> list[tuple[int, int]]:
    out = []
    for j in range(ny):
        cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
        out += [(i, j) for i in cols]
    return out


def _no_motion_preview(microscope):
    """The flake search shows only its photos (#203); devices without the
    switch (test doubles, other adapters) simply keep their previews."""
    off = getattr(microscope, "motion_preview_off", None)
    return off() if callable(off) else nullcontext()


def _plane_through(points: list[list[float]]) -> list[float]:
    """z = a + b*x + c*y through (x, y, z) points; flat or sloped by what they allow."""
    pts = [tuple(p) for p in points]
    zs = [p[2] for p in pts]
    mean = sum(zs) / len(zs)
    if len(pts) < 3:
        if len(pts) == 2 and math.dist(pts[0][:2], pts[1][:2]) > 0:
            (x0, y0, z0), (x1, y1, z1) = pts
            d2 = (x1-x0)**2 + (y1-y0)**2
            b, c = (z1-z0)*(x1-x0)/d2, (z1-z0)*(y1-y0)/d2
            return [z0 - b*x0 - c*y0, b, c]
        return [mean, 0.0, 0.0]
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = pts[:3]
    d = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
    if abs(d) < 1e-12:
        return _plane_through([list(pts[0]), list(max(pts[1:], key=lambda p: math.dist(p[:2], pts[0][:2])))])
    b = ((z1-z0)*(y2-y0) - (z2-z0)*(y1-y0)) / d
    c = ((x1-x0)*(z2-z0) - (x2-x0)*(z1-z0)) / d
    return [z0 - b*x0 - c*y0, b, c]


class ScanChip(Step):
    """10x snake scan over the whole chip; flake candidates into the ledger
    (v1 MoleculeAutoSearch phases 1-2)."""

    name = "scan_chip"
    devices = (STAGE, MICROSCOPE)

    def __init__(self, chip_id: str, material: str, min_area_um2: float) -> None:
        self.chip_id, self.material, self.min_area = chip_id, material, min_area_um2
        self._tiles: set[tuple[int, int]] = set()
        self._blurred: list[tuple[int, int]] = []
        self._found: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._near_edge: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._scan: dict[str, Any] = {}

    def _progress(self, ctx: FlowContext, phase: str, **fields: Any) -> None:
        """Persist acquisition progress, never experimental completion.

        A restarted Step reacquires the whole grid. Prior attempt records stay
        in events, but cannot become fresh coverage from a saved estimate.
        """
        self._scan.update(phase=phase, updated_at=ctx.clock.now(),
                          covered_tiles=[list(t) for t in sorted(self._tiles)],
                          blurred_tiles=[list(t) for t in self._blurred], **fields)
        ctx.store.update_run_meta(ctx.run_id, microscope_scan=self._scan)
        ctx.store.emit("microscope.scan_progress", {
            "version": 1, "step": self.name, "chip_id": self.chip_id,
            "attempt": self._scan["attempt"], "phase": phase,
            "tiles_scanned": len(self._tiles), "tiles_total": self._scan["tiles_total"],
            "blurred_tiles": len(self._blurred), "verification": False,
            **fields,
        }, run_id=ctx.run_id)

    def _grid(self, ctx: FlowContext) -> tuple[list[float], list[float]]:
        """Tile centres in chip-local mm from the BL corner, all on the chip."""
        _, (w, h) = _chip_geometry(ctx, self.chip_id)
        fw, fh = ctx.bench["microscope"]["fov_mm_10x"]
        scale = 10.0 / ctx.bench["microscope"]["scan_objective"]
        fw, fh = fw * scale, fh * scale
        ov = ctx.verification["scan"]["overlap"]

        def axis(length: float, fov: float) -> list[float]:
            n = max(1, math.ceil((length - fov) / (fov * (1 - ov))) + 1)
            if n == 1:
                return [length / 2]
            return [fov / 2 + k * (length - fov) / (n - 1) for k in range(n)]

        return axis(w, fw), axis(h, fh)

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        xs, ys = self._grid(ctx)
        return {"tiles": len(xs) * len(ys),
                "max_blurred_tiles": ctx.verification["scan"]["max_blurred_tiles"],
                "min_sharpness": min_sharpness(ctx, ctx.bench["microscope"]["scan_objective"])}

    def preconditions(self, ctx: FlowContext) -> None:
        self.max_corrections = ctx.verification["scan"]["max_corrections"]
        chip = require_sample(ctx, self.chip_id, "chip")
        if not chip["props"].get("corner_stage_xy"):
            raise PreconditionFailed(f"{self.chip_id} has no corner position (find it first)")

    def _corner(self, ctx: FlowContext) -> tuple[tuple[float, float], float]:
        chip = ctx.store.get_sample(self.chip_id) or {"props": {}}
        c = chip["props"]["corner_stage_xy"]
        return (float(c[0]), float(c[1])), _chip_rotation(ctx, self.chip_id)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        obj = ctx.bench["microscope"]["scan_objective"]
        xs, ys = self._grid(ctx)
        corner, rot = self._corner(ctx)
        binding = {"chip_id": self.chip_id, "material": self.material,
                   "corner_stage_xy": list(corner), "rotation_deg": rot,
                   "grid_x_mm": xs, "grid_y_mm": ys, "objective": obj,
                   "min_area_um2": self.min_area}
        if correction:
            if self._scan.get("binding") != binding:
                raise PreconditionFailed("scan geometry changed; reacquire the whole chip before correcting tiles")
            tiles = [(int(t[0]), int(t[1])) for t in correction.params["tiles"]]
        else:
            tiles = _snake(len(xs), len(ys))
            self._tiles, self._found, self._near_edge = set(), {}, {}
            previous = (ctx.store.get_run(ctx.run_id) or {}).get("meta", {}).get("microscope_scan", {})
            self._scan = {
                "version": 1, "attempt": int(previous.get("attempt", 0)) + 1,
                "tiles_total": len(tiles), "started_at": ctx.clock.now(),
                "restart_policy": "full_reacquisition", "captures": {},
                "binding": binding,
            }
        self._blurred = []
        self._progress(ctx, "reacquiring" if correction else "started")
        m = ctx.gate.microscope(MICROSCOPE)
        m.set_objective(obj)
        an, need = analyzer(ctx), min_sharpness(ctx, obj)
        chip_plane = ctx.verification["scan"].get("focus_policy", "chip_plane") == "chip_plane"
        with _no_motion_preview(m):
            if chip_plane and not (correction and self._scan.get("focus_plane_fit")):
                self._scan["focus_plane_fit"] = self._fit_chip_plane(ctx, obj, xs, ys, corner, rot)
            self._acquire(ctx, tiles, xs, ys, corner, rot, obj, an, need, chip_plane)
        self._progress(ctx, "acquisition_complete")

    def _fit_chip_plane(self, ctx: FlowContext, obj: int, xs: list[float], ys: list[float],
                        corner: tuple[float, float], rot: float) -> dict[str, Any]:
        """Autofocus once per chip, at three corner fields, and fit a plane.

        Owner decision 2026-11-04 (#203): focus only when a chip is first
        looked at, then assume its surface is a flat plane for the whole scan
        (no per-field refocus). LIMITATION for the real lab: this assumes a
        flat, rigidly held chip. A warped chip, a chip that shifts on the
        vacuum, or focus drift during a long scan will blur fields; v1 itself
        used the camera's continuous hardware AF per field instead.
        """
        nx, ny = len(xs), len(ys)
        picks = list(dict.fromkeys([(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]))[:3]
        lim = ctx.gate.limits.microscopes[MICROSCOPE].focus
        points = []
        for i, j in picks:
            ctx.check_cancel()
            _move_stage(ctx, *_local_to_stage(corner, rot, xs[i], ys[j]))
            autofocus(ctx, obj)
            z = ctx.gate.microscope(MICROSCOPE).get_focus()
            points.append([xs[i], ys[j], float(z)])
        coef = _plane_through(points)
        fit = {"policy": "chip_plane_three_corner_v1", "objective": obj, "points": points,
               "coefficients": coef, "focus_limits": list(lim),
               "assumption": "flat rigid chip; no per-field refocus (owner 2026-11-04)"}
        ctx.store.emit("microscope.scan_focus_plane", {"version": 1, "chip_id": self.chip_id,
                       "verification": False, **fit}, run_id=ctx.run_id)
        return fit

    def _plane_focus(self, ctx: FlowContext, x: float, y: float) -> float:
        fit = self._scan["focus_plane_fit"]
        a, b, c = fit["coefficients"]
        lo, hi = fit["focus_limits"]
        return min(max(a + b*x + c*y, lo), hi)

    def _acquire(self, ctx: FlowContext, tiles, xs, ys, corner, rot, obj, an, need,
                 chip_plane: bool) -> None:
        m = ctx.gate.microscope(MICROSCOPE)
        # v1 RGB recognition of a tile can run in worker processes while the
        # stage moves on; results are collected in tile order, unchanged.
        depth = int(getattr(an, "prefetch_depth", 0) or 0)
        queued: deque = deque()

        _, size = _chip_geometry(ctx, self.chip_id)
        edge_mm = _source_edge_clearance_mm(ctx)

        def collect() -> None:
            (ti, tj), tile_frame, tile_capture, tsx, tsy = queued.popleft()
            seen = an.find_flakes(tile_frame, self.material)
            if seen and chip_plane:
                # Only fields with material are saved and shown (#203);
                # a blank field is checked in memory and dropped. The
                # per_tile policy published every field when it took it.
                publish_frame(ctx, tile_frame, purpose="analysis", phase="scan")
                tile_capture.update(image=tile_frame.saved_path(),
                                    image_available=tile_frame.saved_path() is not None)
            found = [self._candidate(f, tsx, tsy, corner, rot, capture=tile_capture)
                     for f in seen if f.area_um2 >= self.min_area]
            # A source flake closer than this to the chip edge is never picked
            # (Jialei 2026-11-03, the pickup rule); with the centre alone this
            # drops only what the pair search would refuse anyway. It keeps the
            # lit chip edge, which v1 RGB recognition reads as 40+ hBN (#199),
            # out of the revisit list.
            kept, near = [], []
            for c in found:
                d = _centre_edge_clearance_mm(c, size)
                if d < edge_mm:
                    near.append({**c, "centre_edge_clearance_mm": round(d, 4)})
                else:
                    kept.append(c)
            self._found[(ti, tj)], self._near_edge[(ti, tj)] = kept, near
            self._progress(ctx, "acquired", tile=[ti, tj], capture=tile_capture,
                           detections=len(self._found[(ti, tj)]), near_chip_edge=len(near))

        row = None
        for i, j in tiles:
            ctx.check_cancel()
            x, y = _local_to_stage(corner, rot, xs[i], ys[j])
            self._progress(ctx, "moving", tile=[i, j], target_stage_xy=[x, y])
            _move_stage(ctx, x, y)
            if chip_plane:
                m.set_focus(self._plane_focus(ctx, xs[i], ys[j]))
                frame = m.capture()
                sharpness = an.sharpness(frame)
                tries = [(sharpness, frame)]
            else:
                if row != j and ctx.verification["scan"]["refocus_every_row"]:
                    autofocus(ctx, obj)
                    row = j
                frame, sharpness, tries = self._per_tile_focus(ctx, m, an, obj, need, xs[i], ys[j])
            keep_best = getattr(an, "keeps_best_focus_frame", False)
            kept_best = sharpness < need and keep_best
            if kept_best:
                # v1 wait_focus_and_snap: when no try reaches min_blur, keep and
                # segment the best-scoring frame (a bare tile has no contrast).
                sharpness, frame = max(tries, key=lambda t: t[0])
            image = None if chip_plane else frame.saved_path()
            # This is encoder readback captured with the frame, not the twin's
            # hidden object position. Legacy adapters lack capture readback.
            pose = (frame.meta.get("stage_kinematics") or {}).get("sample_pose") or {}
            sx, sy = float(pose.get("x", x)), float(pose.get("y", y))
            capture = {"image": image, "image_available": image is not None,
                       "timestamp": frame.timestamp, "objective": frame.meta.get("objective", obj),
                       "focus_z": frame.meta.get("focus_z", frame.meta.get("focus")),
                       "focus_plane": frame.meta.get("focus_plane"),
                       "fov_mm": frame.meta.get("fov_mm"),
                       "calibration": frame.meta.get("calibration"),
                       "mechanics_version": frame.meta.get("mechanics_version"),
                       "stage_xy": [sx, sy],
                       "pose_source": "capture_readback" if pose else "settled_command",
                       "sharpness": sharpness, "purpose": "analysis", "verification": False}
            if chip_plane:
                capture["focus_source"] = "chip_plane"
            if kept_best:
                capture["focus_verdict"] = "below_min_blur_kept_best_frame"
            self._scan["captures"][f"{i},{j}"] = capture
            if sharpness < need and not kept_best:
                if chip_plane:
                    publish_frame(ctx, frame, purpose="analysis", phase="scan")
                    capture.update(image=frame.saved_path(), image_available=frame.saved_path() is not None)
                self._tiles.discard((i, j))
                self._found.pop((i, j), None)
                self._near_edge.pop((i, j), None)
                self._blurred.append((i, j))
                self._progress(ctx, "blurred", tile=[i, j], capture=capture)
                continue
            self._tiles.add((i, j))
            if depth:
                an.prefetch_flakes(frame, self.material)
            queued.append(((i, j), frame, capture, sx, sy))
            while len(queued) > depth:
                collect()
        while queued:
            collect()

    def _per_tile_focus(self, ctx: FlowContext, m, an, obj: int, need: float, lx: float, ly: float):
        """Legacy focus policy: refocus a field whenever it scores under the limit."""
        frame = microscope_frame(ctx, purpose="analysis", phase="scan")
        sharpness = an.sharpness(frame)
        tries = [(sharpness, frame)]
        keep_best = getattr(an, "keeps_best_focus_frame", False)
        if sharpness < need:
            z0 = m.get_focus() if keep_best else None
            autofocus(ctx, obj)
            # A keep-best camera whose sweep left the focus where it was
            # would only re-take the same exposure.
            if not (keep_best and m.get_focus() == z0):
                frame = microscope_frame(ctx, purpose="analysis", phase="scan_refocus")
                sharpness = an.sharpness(frame)
                tries.append((sharpness, frame))
        if sharpness < need:
            # A bare tile can give the contrast sweep nothing to lock onto
            # (a518 seed 10 on the 16 x 26 mm chip). Try the focus plane
            # fitted to this scan's in-focus tiles; the frame must still
            # meet the same sharpness limit.
            z = self._planned_focus(lx, ly)
            if z is not None:
                lim = ctx.gate.limits.microscopes[MICROSCOPE].focus
                m.set_focus(min(max(z, lim[0]), lim[1]))
                frame = microscope_frame(ctx, purpose="analysis", phase="scan_focus_plane")
                sharpness = an.sharpness(frame)
                tries.append((sharpness, frame))
        return frame, sharpness, tries

    def _planned_focus(self, x: float, y: float) -> float | None:
        """Focus Z at chip-local (x, y) from a plane through this scan's sharp tiles.

        Uses only captures this scan acquired in focus (their recorded focus
        readback and grid position). None until three non-collinear tiles exist.
        """
        xs, ys = self._scan["binding"]["grid_x_mm"], self._scan["binding"]["grid_y_mm"]
        pts = []
        for key, cap in self._scan["captures"].items():
            i, j = (int(v) for v in key.split(","))
            z = cap.get("focus_z")
            if ((i, j) in self._tiles and isinstance(z, (int, float)) and math.isfinite(z)
                    and not cap.get("focus_verdict")):
                pts.append((xs[i], ys[j], float(z)))
        if len(pts) < 3:
            return None
        # least squares z = a + b*x + c*y (normal equations, 3 x 3)
        def total(f: Callable[[tuple[float, float, float]], float]) -> float:
            return sum(f(p) for p in pts)

        sx, sy = total(lambda p: p[0]), total(lambda p: p[1])
        sxy = total(lambda p: p[0]*p[1])
        a = [[float(len(pts)), sx, sy],
             [sx, total(lambda p: p[0]*p[0]), sxy],
             [sy, sxy, total(lambda p: p[1]*p[1])]]
        rhs = [total(lambda p: p[2]), total(lambda p: p[0]*p[2]), total(lambda p: p[1]*p[2])]

        def det(mx: list[list[float]]) -> float:
            return (mx[0][0]*(mx[1][1]*mx[2][2]-mx[1][2]*mx[2][1])
                    - mx[0][1]*(mx[1][0]*mx[2][2]-mx[1][2]*mx[2][0])
                    + mx[0][2]*(mx[1][0]*mx[2][1]-mx[1][1]*mx[2][0]))

        d = det(a)
        if abs(d) < 1e-9:
            return None
        coef = []
        for k in range(3):
            mk = [row[:] for row in a]
            for r in range(3):
                mk[r][k] = rhs[r]
            coef.append(det(mk) / d)
        return coef[0] + coef[1]*x + coef[2]*y

    @staticmethod
    def _merge(ctx: FlowContext) -> float:
        return float(ctx.verification["scan"].get("merge_mm", 0.03))

    @staticmethod
    def _candidate(f: FlakeDetection, sx: float, sy: float, corner: tuple[float, float],
                   rot: float, *, capture=None) -> dict[str, Any]:
        stage = (sx - f.dx_mm, sy - f.dy_mm)
        lx, ly = _stage_to_local(corner, rot, *stage)
        candidate = {"stage_xy": [round(stage[0], 4), round(stage[1], 4)],
                "rel_corner_mm": [round(lx, 4), round(ly, 4)], "area_um2": f.area_um2,
                "thickness_label": f.thickness_label, "confidence": f.confidence,
                "sim_id": f.meta.get("sim_id")}
        chain = f.meta.get("recognition_chain") or {}
        if (capture and f.meta.get("geometry_source") == "v1_lab_rgb"
                and f.meta.get("outline_complete") and "region_mm" in f.meta
                and chain.get("image", {}).get("objective") == 10
                and chain["image"].get("path") == capture.get("image")
                and chain.get("calibration") == capture.get("calibration")):
            from qumus_simlab.core.regions import region_dict, transform_region

            calibration = chain["calibration"]
            candidate["coarse_geometry_10x"] = {
                "source": "v1_lab_rgb", "objective": 10, "units": "mm",
                "frame": "flake_center_chip_axes", "image": deepcopy(chain["image"]),
                "calibration": deepcopy(calibration), "capture_timestamp": capture["timestamp"],
                "centroid_px": list(chain["derivation"]["centroid_px"]),
                "observed_chip_rotation_deg": rot,
                "selected_label_ids": list(chain["selected_label_ids"]),
                "region_mm": region_dict(transform_region(f.meta["region_mm"], rotation_deg=-rot)),
                "scope": "coarse_observed_priority_only_not_source_admission"}
        return candidate

    def candidates(self, merge_mm: float = 0.03) -> list[dict[str, Any]]:
        """All tiles' detections, duplicates from tile overlaps merged (bigger kept)."""
        out: list[dict[str, Any]] = []
        for tile in sorted(self._found, key=lambda t: (t[1], t[0])):
            for c in self._found[tile]:
                dup = next((o for o in out
                            if math.dist(o["stage_xy"], c["stage_xy"]) < merge_mm), None)
                if dup is None:
                    out.append(dict(c))
                elif c["area_um2"] > dup["area_um2"]:
                    dup.update(c)
        return out

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        m = {"tiles_scanned": len(self._tiles), "blurred_tiles": len(self._blurred),
             "candidates": len(self.candidates(self._merge(ctx))),
             "near_chip_edge_detections": sum(map(len, self._near_edge.values()))}
        if m["tiles_scanned"] >= exp["tiles"] and m["blurred_tiles"] <= exp["max_blurred_tiles"]:
            return Observation(Verdict.OK, m, exp)
        return Observation(Verdict.CORRECTABLE, m, exp,
                           message=f"{len(self._blurred)} tile(s) out of focus")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        if not self._blurred:
            return None
        return Correction("rescan blurred tiles", {"tiles": [list(t) for t in self._blurred]})

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        near = [{"tile": list(t), **{k: c[k] for k in ("stage_xy", "rel_corner_mm", "area_um2",
                                                         "thickness_label", "centre_edge_clearance_mm")}}
                for t in sorted(self._near_edge) for c in self._near_edge[t]]
        if near:
            ctx.store.emit("microscope.scan_near_chip_edge", {
                "version": 1, "step": self.name, "chip_id": self.chip_id,
                "rule": "pickup.source_chip_edge_clearance_mm",
                "required_mm": _source_edge_clearance_mm(ctx),
                "reason": "not a candidate: its centre is closer to the chip edge than a source may be",
                "detections": near}, run_id=ctx.run_id)
        for n, c in enumerate(self.candidates(self._merge(ctx)), start=1):
            ctx.store.upsert_sample(
                f"{self.chip_id}_f{n:03d}", "flake", location=self.chip_id, status="candidate",
                props={"material": self.material, "chip_id": self.chip_id,
                       "thickness_label": c["thickness_label"],
                       "labels": {"10x": c["thickness_label"]}, "area_um2": c["area_um2"],
                       "stage_xy": c["stage_xy"], "rel_corner_mm": c["rel_corner_mm"],
                       "confidence": c["confidence"], "images": ({"10x": c["coarse_geometry_10x"]["image"]["path"]}
                           if "coarse_geometry_10x" in c else {}), "sim_id": c["sim_id"],
                       **({"coarse_geometry_10x": {**c["coarse_geometry_10x"],
                           "source_chip_id": self.chip_id, "scan_job_id": ctx.run_id}}
                           if "coarse_geometry_10x" in c else {})})

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE  # only looks; redo the scan


def _rgb_revisit_evidence(ctx, an, frame, detection, chip_id, flake_id, capture_event_id, *,
                          capture_stage_xy=None, stage_readback_event_id=None):
    """Persist actual RGB20X masks once;100X classification cannot replace them."""
    if frame.meta["objective"] != 20:
        if frame.meta["objective"] != 100:
            raise PreconditionFailed("RGB revisit classification requires100X")
        chain = an.persist_detection_masks(frame, detection)
        event_id = ctx.store.emit("microscope.rgb_classification", {"version": 1, "flake_id": flake_id,
            "source_chip_id": chip_id, "material": detection.material, "thickness_label": detection.thickness_label,
            "objective": 100,
            "capture_event_id": capture_event_id, "recognition_chain": deepcopy(chain),
            "geometry_use": "classification_only"}, run_id=ctx.run_id)
        return {"recognition_100x": chain, "classification_event_100x": event_id}
    import numpy as np
    from PIL import Image

    from qumus_simlab.core.regions import region_dict, transform_region
    from qumus_simlab.perception.measurement import owner_error_budget_policy
    from qumus_simlab.perception.rgb_material import component_geometry
    from qumus_simlab.records.recognition import validate_recognition_chain

    chain = an.persist_detection_masks(frame, detection)
    validate_recognition_chain(chain, image_path=frame.saved_path())
    from qumus_simlab.core.run_paths import resolve_run_path

    with Image.open(resolve_run_path(chain["geometry_mask"]["path"])) as image:
        geometry = component_geometry(np.asarray(image) != 0, chain["calibration"])
    rotation = _chip_rotation(ctx, chip_id)
    if not math.isfinite(rotation):
        raise PreconditionFailed("RGB geometry requires finite observed chip rotation")
    rotation_row = ctx.store._x("SELECT id,payload_json FROM event WHERE run_id=? AND kind='step.verified' "
        "AND json_extract(payload_json,'$.name')='measure_chip_rotation' AND id<? ORDER BY id DESC LIMIT 1",
        (ctx.run_id, capture_event_id)).fetchone()
    rotation_body = json.loads(rotation_row["payload_json"]) if rotation_row else {}
    if (not rotation_row or rotation_body.get("verdict") != "ok"
            or rotation_body.get("measurements", {}).get("rotation_deg") != rotation):
        raise PreconditionFailed("RGB geometry lacks its original observed chip rotation receipt")
    if (capture_stage_xy is None or len(capture_stage_xy) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in capture_stage_xy)):
        raise PreconditionFailed("RGB20 geometry lacks its original public stage readback")
    readback = ctx.store._x("SELECT kind,run_id,payload_json FROM event WHERE id=?",
                            (stage_readback_event_id,)).fetchone()
    public = json.loads(readback["payload_json"]) if readback else {}
    capture = ctx.store._x("SELECT kind,run_id,payload_json FROM event WHERE id=?", (capture_event_id,)).fetchone()
    captured = json.loads(capture["payload_json"]) if capture else {}
    if (type(stage_readback_event_id) is not int or not rotation_row["id"] < stage_readback_event_id < capture_event_id
            or not readback or readback["run_id"] != ctx.run_id
            or readback["kind"] != "microscope.source_stage_readback" or public.get("source_chip_id") != chip_id
            or public.get("flake_id") != flake_id or public.get("objective") != 20
            or public.get("stage_xy") != list(capture_stage_xy)
            or not capture or capture["run_id"] != ctx.run_id or capture["kind"] != "device.command"
            or captured.get("device") != MICROSCOPE or captured.get("command") != "capture"):
        raise PreconditionFailed("RGB20 source anchor lacks its original readback/capture journal")
    policy = owner_error_budget_policy()
    source_anchor = {"version": "original20-mask-centroid-stage-v1", "objective": 20,
        "capture_event_id": capture_event_id, "image_sha256": chain["image"]["sha256"],
        "stage_readback_event_id": stage_readback_event_id,
        "capture_stage_xy": list(capture_stage_xy), "centroid_px": geometry["centroid_px"],
        "detection_offset_mm": [geometry["dx_mm"], geometry["dy_mm"]],
        "stage_xy": [capture_stage_xy[0]-geometry["dx_mm"], capture_stage_xy[1]-geometry["dy_mm"]]}
    observation = {"region_mm": region_dict(transform_region(geometry["region_mm"], rotation_deg=-rotation)),
        "owner_error_budget": policy, "boundary_uncertainty_mm": .001,
        "boundary_uncertainty_provenance": policy, "source_anchor20": source_anchor,
        "geometry_policy": "original20-selected-pixel-union-v1"} if geometry["status"] == "supported" else {}
    event_id = ctx.store.emit("microscope.rgb_recognition", {"version": 1, "flake_id": flake_id,
        "source_chip_id": chip_id, "material": detection.material, "capture_timestamp": frame.timestamp,
        "thickness_label": detection.thickness_label,
        "capture_event_id": capture_event_id, "chip_rotation_event_id": rotation_row["id"],
        "chip_rotation_deg": rotation, "recognition_chain": deepcopy(chain), **observation}, run_id=ctx.run_id)
    result = {"recognition_20x": chain, "recognition_event_id": event_id,
        "mask_20x": chain["original_mask"]["path"], "geometry_mask_20x": chain["geometry_mask"]["path"],
        "recognition_status": chain["derivation"]["status"], "recognition_reason": chain["derivation"]["unsupported_reason"],
        "optical_geometry": None}
    if observation:
        # Exact selected pixel cells retain holes. Unlike v1's external-only
        # contour adapter, this representation never fills an interior void.
        result["optical_geometry"] = {**observation,
            "units": "mm", "frame": "flake_center_chip_axes", "source": "rgb_optical_outline",
            "source_chip_id": chip_id, "image": frame.saved_path(), "job_id": ctx.run_id,
            "chip_rotation_deg": rotation, "recognition_event_id": event_id, "recognition_chain": deepcopy(chain)}
        result["source_anchor20"] = source_anchor
        result["tracking_references"] = {"20x": {"image": frame.saved_path(),
            "polygon_px": [point for ring in chain["derivation"]["rings_px"] for point in ring],
            "timestamp": frame.timestamp, "job_id": ctx.run_id,
            "meta": {key: frame.meta.get(key) for key in ("objective", "focus_z", "focus", "focus_plane", "fov_mm", "calibration")}}}
    return result


def _native_rgb_revisit_evidence(ctx, an, frame, detection, chip_id, flake_id, capture_event_id):
    """Capture actual native evidence without granting an unqualified source admission."""
    from qumus_simlab.perception.measurement import measure_recognition_boundary

    objective = frame.meta["objective"]
    if objective not in (20, 100):
        raise PreconditionFailed("Native material evidence requires 20X or 100X")
    rotation = _chip_rotation(ctx, chip_id)
    row = ctx.store._x("SELECT id,payload_json FROM event WHERE run_id=? AND kind='step.verified' "
        "AND json_extract(payload_json,'$.name')='measure_chip_rotation' AND id<? ORDER BY id DESC LIMIT 1",
        (ctx.run_id, capture_event_id)).fetchone()
    step = json.loads(row["payload_json"]) if row else {}
    if (not row or not math.isfinite(rotation) or step.get("verdict") != "ok"
            or step.get("measurements", {}).get("rotation_deg") != rotation):
        raise PreconditionFailed("Native measurement lacks its original observed chip rotation")
    capture = ctx.store._x("SELECT id,run_id,kind,payload_json FROM event WHERE id=?", (capture_event_id,)).fetchone()
    capture_payload = json.loads(capture["payload_json"]) if capture else {}
    if (not capture or capture["run_id"] != ctx.run_id or capture["kind"] != "device.command"
            or capture_payload.get("device") != MICROSCOPE or capture_payload.get("command") != "capture"):
        raise PreconditionFailed("Native measurement lacks its original same-job capture")
    chain = an.persist_detection_masks(frame, detection)
    measurement = measure_recognition_boundary(chain)
    rotation_evidence = {"version": "observed-chip-rotation-v1", "source_chip_id": chip_id,
        "native100_image_sha256": chain["image"]["sha256"] if objective == 100 else None,
        "rotation_event": {"id": row["id"], "run_id": ctx.run_id, "kind": "step.verified", "payload": step},
        "capture_event": {"id": capture["id"], "run_id": capture["run_id"], "kind": capture["kind"],
                          "payload": capture_payload}}
    event_id = ctx.store.emit("microscope.rgb_measurement", {"version": 1,
        "source_chip_id": chip_id, "flake_id": flake_id, "material": detection.material,
        "thickness_label": detection.thickness_label,
        "objective": objective, "capture_event_id": capture_event_id,
        "chip_rotation_evidence": rotation_evidence, "measurement": measurement,
        "admission_status": "blocked", "reason": "physical_boundary_qualification_pending"}, run_id=ctx.run_id)
    result = {f"measurement_{objective}x": measurement, f"measurement_event_{objective}x": event_id,
        f"native_rotation_{objective}x": rotation_evidence, f"recognition_{objective}x": chain,
        "recognition_status": measurement["measurement"]["status"],
        "recognition_reason": measurement["measurement"]["reason"], "optical_geometry": None,
        "native_source_admission": "blocked: physical boundary qualification pending"}
    rings = chain.get("derivation", {}).get("rings_px", [])
    if rings:
        # Registration uses the original image ROI, never this outline as a
        # replacement for the separately measured complete material region.
        result["tracking_references"] = {f"{objective}x": {
            "image": frame.saved_path(), "polygon_px": [point for ring in rings for point in ring],
            "timestamp": frame.timestamp, "job_id": ctx.run_id,
            "meta": {key: frame.meta.get(key) for key in
                     ("objective", "focus_z", "focus", "focus_plane", "fov_mm", "calibration")}}}
    if objective == 20:
        result.update(mask_20x=chain.get("original_mask", {}).get("path"),
                      geometry_mask_20x=chain.get("geometry_mask", {}).get("path"))
    return result


class RevisitCandidates(Step):
    """Go back to the best candidates at 20x then 100x, re-centre, refocus,
    photograph and classify again (v1 MoleculeAutoSearch phases 3-4)."""

    name = "revisit_candidates"
    devices = (STAGE, MICROSCOPE)

    def __init__(self, chip_id: str, material: str, max_revisits: int,
                 quota: list[RevisitRole] | None = None, *, only_new: bool = False,
                 flake_ids: list[str] | None = None) -> None:
        """`only_new` (revisit_chip): only candidates never revisited, found
        again from their corner-relative position, since the chip was put down
        again; `flake_ids` limits it to those flakes."""
        self.chip_id, self.material, self.max_revisits = chip_id, material, max_revisits
        self.quota, self.only_new, self.flake_ids = quota, only_new, flake_ids
        self._results: dict[str, dict[str, Any]] = {}

    def _targets(self, ctx: FlowContext) -> list[dict[str, Any]]:
        ok = ("candidate",) if self.only_new and not self.flake_ids else ("candidate", "catalogued", "lost")
        cands = [s for s in ctx.store.list_samples(kind="flake", location=self.chip_id)
                 if s["status"] in ok and (not self.flake_ids or s["id"] in self.flake_ids)
                 and not (self.only_new and not self.flake_ids and any(
                     (s["props"].get("images") or {}).get(key) for key in ("20x", "100x")))]
        if self.flake_ids:
            # Explicit requests select the eligible candidates themselves; a
            # device-oriented automatic quota must not silently discard them.
            by_id = {s["id"]: s for s in cands}
            return [by_id[fid] for fid in dict.fromkeys(self.flake_ids) if fid in by_id][:self.max_revisits]
        project_id = getattr(ctx, "project_id", None)
        project = ctx.store.get_project(project_id) if project_id else None
        plan = (project or {}).get("state", {}).get("device_plan") or {}
        target_chip = ctx.store.get_sample(plan.get("target_chip_id", "")) if plan else None
        target = (target_chip or {}).get("props", {}).get("device_design_target")
        if self.quota is not None and isinstance(target, dict) and len(target.get("contacts", [])) == 8:
            from qumus_simlab.flows.library.revisit_priority import rank_revisits
            from qumus_simlab.records.geometry import _pickup_region

            alternatives = []
            for sample in ctx.store.list_samples(kind="flake"):
                if (sample["props"].get("material") not in ("hBN", "Gr")
                        or sample["props"]["material"] == self.material
                        or sample["status"] not in ("candidate", "catalogued")):
                    continue
                observed = sample["props"].get("optical_geometry") or {}
                if observed.get("source") == "rgb_optical_outline":
                    try:
                        sample = {**sample, "_observed_priority_region":
                            _pickup_region(ctx.store, sample).model_dump(mode="json")}
                    except (ValueError, KeyError, TypeError, OSError):
                        pass  # Unqualified/missing observations remain coarse or unknown.
                alternatives.append(sample)
            selected, self._revisit_priority = rank_revisits(cands, self.material, self.quota,
                self.max_revisits, target, alternatives)
            return selected
        self._revisit_priority = None
        return pick_revisits(cands, self.material, self.quota, self.max_revisits)

    def _position(self, ctx: FlowContext, flake: dict[str, Any]) -> tuple[float, float]:
        if not self.only_new:
            x, y = flake["props"]["stage_xy"]
            return float(x), float(y)
        chip = ctx.store.get_sample(self.chip_id) or {"props": {}}
        corner = chip["props"].get("corner_stage_xy")
        if not corner or not flake["props"].get("rel_corner_mm"):
            raise PreconditionFailed(f"{flake['id']}: no chip corner or corner-relative "
                                     "position to find it again")
        lx, ly = flake["props"]["rel_corner_mm"]
        return _local_to_stage((corner[0], corner[1]), _chip_rotation(ctx, self.chip_id), lx, ly)

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        checkpoint = self._read_checkpoint(ctx)
        if checkpoint and checkpoint["phase"] != "closed":
            self._validate_checkpoint(ctx, checkpoint)
            count = len(checkpoint["binding"]["targets"])
        else:
            count = len(self._targets(ctx))
        return {"revisits": count, "objectives": ctx.bench["microscope"]["revisit_objectives"]}

    def preconditions(self, ctx: FlowContext) -> None:
        self.max_corrections = ctx.verification["revisit"]["max_corrections"]

    _checkpoint_key = "revisit_candidates_checkpoint"
    _checkpoint_event = "microscope.revisit_checkpoint"

    @staticmethod
    def _sample_input(sample):
        if not isinstance(sample, dict):
            raise PreconditionFailed("Revisit source sample is missing")
        return {key: deepcopy(sample.get(key)) for key in ("id", "kind", "location", "status", "props")}

    @staticmethod
    def _json_snapshot(value):
        # Match the store representation without rounding or reordering progress.
        return json.loads(json.dumps(value, allow_nan=False))

    def _input_binding(self, ctx, targets):
        binding = {"params": deepcopy(ctx.params), "chip": self._sample_input(ctx.store.get_sample(self.chip_id)),
                "chip_id": self.chip_id, "material": self.material, "max_revisits": self.max_revisits,
                "only_new": self.only_new, "flake_ids": self.flake_ids,
                "quota": [role.model_dump(mode="json") for role in self.quota] if self.quota is not None else None,
                "targets": targets, "microscope": deepcopy(ctx.bench["microscope"]),
                "verification": {key: deepcopy(ctx.verification.get(key)) for key in ("focus", "revisit", "pickup")},
                "backend": getattr(analyzer(ctx), "recognition_backend", "legacy")}
        return self._json_snapshot(binding)

    @staticmethod
    def _checkpoint_digest(checkpoint):
        return hashlib.sha256(json.dumps({k: v for k, v in checkpoint.items() if k != "receipt_event_id"},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def _read_checkpoint(self, ctx):
        checkpoint = ctx.store.get_run(ctx.run_id)["meta"].get(self._checkpoint_key)
        if checkpoint is None:
            return None
        try:
            receipt = ctx.store._x("SELECT id,payload_json FROM event WHERE run_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                                   (ctx.run_id, self._checkpoint_event)).fetchone()
            expected = {"version": 1, "step": self.name, "checkpoint_sha256": self._checkpoint_digest(checkpoint)}
            if (checkpoint["version"] != 1 or checkpoint["run_id"] != ctx.run_id or checkpoint["step"] != self.name
                    or receipt is None or receipt["id"] != checkpoint["receipt_event_id"]
                    or json.loads(receipt["payload_json"]) != expected):
                raise ValueError("original checkpoint receipt differs")
            return checkpoint
        except (KeyError, TypeError, ValueError) as exc:
            raise PreconditionFailed("Revisit checkpoint does not match its original same-job receipt") from exc

    def _save_checkpoint(self, ctx, checkpoint):
        checkpoint = self._json_snapshot(checkpoint)
        with ctx.store._transaction():
            receipt = ctx.store.emit(self._checkpoint_event, {"version": 1, "step": self.name,
                "checkpoint_sha256": self._checkpoint_digest(checkpoint)}, run_id=ctx.run_id)
            checkpoint["receipt_event_id"] = receipt
            ctx.store.update_run_meta(ctx.run_id, **{self._checkpoint_key: checkpoint})
        return checkpoint

    def _validate_checkpoint(self, ctx, checkpoint):
        try:
            binding = checkpoint["binding"]
            targets = binding["targets"]
            ids = [target["id"] for target in targets]
            completed = checkpoint["completed"]
            if (checkpoint["phase"] not in ("acquiring", "published")
                    or self._input_binding(ctx, targets) != binding or len(ids) != len(set(ids))
                    or list(completed) != ids[:len(completed)]):
                raise ValueError("input binding or completed prefix changed")
            if checkpoint["phase"] == "published":
                current = [self._sample_input(ctx.store.get_sample(fid)) for fid in ids]
                if current != checkpoint["published"] or len(completed) != len(ids) or checkpoint["current"] is not None:
                    raise ValueError("published sample outputs changed")
            elif [self._sample_input(flake) for flake in self._targets(ctx)] != targets:
                raise ValueError("ordered candidate inputs changed")
            anchor = checkpoint["current"]
            if anchor is not None and (len(completed) >= len(ids) or anchor["candidate_id"] != ids[len(completed)]
                    or not math.isfinite(anchor["focus_z"]) or type(anchor["objective"]) is not int
                    or anchor["target_xy"] != list(self._position(ctx, targets[len(completed)]))):
                raise ValueError("incomplete candidate anchor changed")
        except (KeyError, TypeError, ValueError) as exc:
            raise PreconditionFailed("Revisit checkpoint inputs, progress or publication changed") from exc

    def _correction_event(self, ctx, correction):
        if correction is None:
            return None
        receipt = ctx.store._x("SELECT id,payload_json FROM event WHERE run_id=? AND kind='step.corrected' "
            "AND json_extract(payload_json,'$.name')=? ORDER BY id DESC LIMIT 1", (ctx.run_id, self.name)).fetchone()
        payload = json.loads(receipt["payload_json"]) if receipt else {}
        if payload.get("action") != correction.action or payload.get("params") != self._json_snapshot(correction.params):
            raise PreconditionFailed("Revisit correction has no original same-step receipt")
        return receipt["id"]

    def _open_checkpoint(self, ctx, correction):
        checkpoint = self._read_checkpoint(ctx)
        correction_event = self._correction_event(ctx, correction)
        if (checkpoint and checkpoint["phase"] != "closed"
                and (correction is None or checkpoint["correction_event_id"] == correction_event)):
            self._validate_checkpoint(ctx, checkpoint)
            return checkpoint
        paused = ctx.store.get_run(ctx.run_id)["meta"].get("paused", {})
        if checkpoint is None:
            start = ctx.store._x("SELECT MIN(id) AS id FROM event WHERE run_id=? AND kind='step.started' "
                "AND json_extract(payload_json,'$.name')=?", (ctx.run_id, self.name)).fetchone()["id"]
            progressed = start is not None and ctx.store._x(
                "SELECT id FROM event WHERE run_id=? AND id>? AND kind='device.command' LIMIT 1",
                (ctx.run_id, start)).fetchone() is not None
            unknown_pause = start is None and paused.get("step") == self.name and paused.get("in_progress")
            if progressed or unknown_pause:
                raise PreconditionFailed("Interrupted revisit has no original candidate checkpoint")
        targets = [self._sample_input(flake) for flake in self._targets(ctx)]
        return self._save_checkpoint(ctx, {"version": 1, "run_id": ctx.run_id, "step": self.name,
            "phase": "acquiring", "binding": self._input_binding(ctx, targets), "completed": {}, "current": None,
            "correction_event_id": correction_event,
            **({"selection_priority": self._revisit_priority} if getattr(self, "_revisit_priority", None) else {})})

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        # The flake search shows only the photos it takes (#203).
        with _no_motion_preview(ctx.gate.microscope(MICROSCOPE)):
            self._revisit(ctx, correction)

    def _revisit(self, ctx: FlowContext, correction: Correction | None) -> None:
        checkpoint = self._open_checkpoint(ctx, correction)
        self._results = deepcopy(checkpoint["completed"])
        if checkpoint["phase"] == "published":
            return
        m = ctx.gate.microscope(MICROSCOPE)
        for flake in checkpoint["binding"]["targets"]:
            if flake["id"] in self._results:
                continue
            ctx.check_cancel()
            anchor = checkpoint["current"]
            if anchor is None:
                # These public readbacks do not issue a command/check cancellation.
                anchor = {"candidate_id": flake["id"], "objective": m.get_objective(),
                    "focus_z": m.get_focus(), "target_xy": list(self._position(ctx, flake)),
                    "captured_at": ctx.clock.now(), "event_cursor": ctx.store.last_event_id()}
                if type(anchor["objective"]) is not int or not math.isfinite(anchor["focus_z"]):
                    raise PreconditionFailed("Revisit entry optical readback is unavailable")
                checkpoint["current"] = anchor
                checkpoint = self._save_checkpoint(ctx, checkpoint)
            else:
                ctx.check_cancel()
                m.set_objective(anchor["objective"])
                ctx.check_cancel()
                m.set_focus(anchor["focus_z"])
            ctx.check_cancel()
            result = self._acquire_candidate(ctx, flake)
            checkpoint["completed"][flake["id"]] = result
            checkpoint["current"] = None
            checkpoint = self._save_checkpoint(ctx, checkpoint)
            self._results = deepcopy(checkpoint["completed"])

    def _acquire_candidate(self, ctx, flake):
        m, an = ctx.gate.microscope(MICROSCOPE), analyzer(ctx)
        radius = ctx.verification["revisit"]["refind_radius_um"] / 1000.0
        x, y = self._position(ctx, flake)
        res: dict[str, Any] = {"found": True, "labels": {}, "images": {}}
        if getattr(an, "recognition_backend", None) == "v1_lab_rgb":
            res["recognition_backend"] = "v1_lab_rgb"
        rgb = res.get("recognition_backend") == "v1_lab_rgb"
        for obj in ctx.bench["microscope"]["revisit_objectives"]:
            m.set_objective(obj)
            _move_stage(ctx, x, y)
            sharp = autofocus(ctx, obj)
            capture_stage_xy = stage_readback_event_id = None
            if rgb and obj == 20:
                stage = ctx.gate.stage(STAGE)
                coarse, fine = stage.get_pose(), stage.get_fine_position()
                capture_stage_xy = [coarse.x+fine[0], coarse.y+fine[1]]
                stage_readback_event_id = ctx.store.emit("microscope.source_stage_readback", {
                    "version": 1, "source_chip_id": self.chip_id, "flake_id": flake["id"],
                    "objective": 20, "coarse_xy": [coarse.x, coarse.y], "fine_xy": list(fine),
                    "stage_xy": capture_stage_xy}, run_id=ctx.run_id)
            capture_cursor = ctx.store.last_event_id()
            frame = m.capture()
            if capture_stage_xy is not None:
                coarse, fine = stage.get_pose(), stage.get_fine_position()
                if [coarse.x+fine[0], coarse.y+fine[1]] != capture_stage_xy:
                    raise PreconditionFailed("Stage moved during the original20 RGB capture")
            if rgb:
                try:
                    # v1 names a graphite-gate capture graphite_<mag>x and its
                    # colour recovery then accepts only manylayer (livesegment.py:600-606).
                    graphite = (self.material == "Gr"
                                and ((flake.get("props") or {}).get("labels") or {}).get("10x") == "manylayer")
                    selected = (an.revisit_target(frame, self.material, accepted_labels=("manylayer",))
                                if graphite else an.revisit_target(frame, self.material))
                    dets = [selected] if math.hypot(selected.dx_mm, selected.dy_mm) <= radius else []
                except ValueError as exc:
                    if obj == 100 and res["found"] and "20x" in res["labels"]:
                        # v1 uses 100X only to classify. A 100X frame the
                        # recognizer cannot read (a large flake leaves no
                        # four-side substrate) keeps the 20X entry and its photo.
                        res["images"]["100x"] = frame.saved_path()
                        res["classification_100x"] = {"status": "unclassified", "reason": str(exc)}
                        break
                    res.update(found=False, lost_at=obj, recognition_status="unsupported", recognition_reason=str(exc))
                    break
            else:
                dets = [d for d in an.find_flakes(frame, self.material)
                        if math.hypot(d.dx_mm, d.dy_mm) <= radius]
            if not dets:
                res.update(found=False, lost_at=obj)
                break
            d = min(dets, key=lambda d: math.hypot(d.dx_mm, d.dy_mm))
            x, y = x - d.dx_mm, y - d.dy_mm  # re-centre on the flake
            res["labels"][f"{obj}x"] = d.thickness_label
            res["images"][f"{obj}x"] = frame.saved_path()
            if rgb:
                captures = [e for e in ctx.store.events_after(capture_cursor, limit=32, run_id=ctx.run_id)
                    if e.kind == "device.command" and e.payload.get("device") == MICROSCOPE and e.payload.get("command") == "capture"]
                if len(captures) != 1:
                    raise PreconditionFailed("RGB recognition lost its unique original capture command")
                evidence = _rgb_revisit_evidence(ctx, an, frame, d, self.chip_id, flake["id"], captures[0].id,
                    capture_stage_xy=capture_stage_xy, stage_readback_event_id=stage_readback_event_id)
                if "tracking_references" in evidence:
                    res.setdefault("tracking_references", {}).update(evidence.pop("tracking_references"))
                res.update(evidence)
                if obj == 20 and res.get("optical_geometry") is None:
                    res.update(found=False, lost_at=obj)
                    break
            elif d.meta.get("outline_complete") and d.meta.get("geometry_source") not in (None, "legacy_display") and not frame.meta.get("replayed_from"):
                from qumus_simlab.core.polygons import convex_polygon, transform

                res["optical_geometry"] = {
                    "outline_mm": transform(convex_polygon(d.meta["outline_mm"]),
                        rotation_deg=-_chip_rotation(ctx, self.chip_id)),
                    "units": "mm", "frame": "flake_center_chip_axes",
                    "source": "simulated_optical_outline" if ctx.store.bench == "sim" else "calibrated_optical_outline",
                    "source_chip_id": self.chip_id, "image": frame.saved_path(),
                    # Image evidence is digested by content (stacking._digest).
                    # A run without an image store saves no file, so no hash.
                    "image_sha256": (hashlib.sha256(Path(frame.saved_path()).read_bytes()).hexdigest()
                                     if frame.saved_path() else None),
                    "job_id": ctx.run_id}
                res.setdefault("tracking_references", {})[f"{obj}x"] = {
                    "image": frame.saved_path(), "polygon_px": observed_outline_pixels(frame, d),
                    "timestamp": frame.timestamp, "job_id": ctx.run_id,
                    "meta": {k: frame.meta.get(k) for k in
                             ("objective", "focus_z", "focus", "focus_plane", "fov_mm", "calibration")},
                }
            res["sharpness"] = sharp
            if not rgb or obj == 20:
                res["stage_xy"] = (list(res["source_anchor20"]["stage_xy"]) if rgb
                                   else [round(x, 4), round(y, 4)])
        if res["found"]:
            if rgb:
                x, y = res["source_anchor20"]["stage_xy"]
            _move_stage(ctx, x, y)
            from qumus_simlab.flows.library.pickup_evidence import capture_neighbourhood

            res["pickup_assessment"] = capture_neighbourhood(ctx, flake["id"], self.chip_id, self.material)
            if res["pickup_assessment"]["images"]:
                res["images"]["context"] = res["pickup_assessment"]["images"][0]
        return res

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        found = sum(1 for r in self._results.values() if r["found"])
        m = {"revisited": len(self._results), "found_again": found,
             "lost": len(self._results) - found}
        if self._results and not found:
            return Observation(Verdict.UNCLEAR, m, exp,
                               message="none of the candidates could be found again")
        return Observation(Verdict.OK, m, exp)

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        with ctx.store._transaction():
            checkpoint = self._read_checkpoint(ctx)
            if checkpoint is None:
                raise PreconditionFailed("Revisit publication has no original candidate checkpoint")
            self._validate_checkpoint(ctx, checkpoint)
            if (checkpoint["current"] is not None or self._results != checkpoint["completed"]
                    or len(self._results) != len(checkpoint["binding"]["targets"])):
                raise PreconditionFailed("Incomplete revisit results cannot be published")
            self._publish_results(ctx)
            checkpoint["phase"] = "published"
            checkpoint["published"] = [self._sample_input(ctx.store.get_sample(target["id"]))
                                       for target in checkpoint["binding"]["targets"]]
            self._save_checkpoint(ctx, checkpoint)

    def on_give_up(self, ctx: FlowContext, obs: Observation) -> None:
        checkpoint = self._read_checkpoint(ctx)
        if checkpoint is not None:
            checkpoint["phase"] = "closed"
            self._save_checkpoint(ctx, checkpoint)

    def _publish_results(self, ctx: FlowContext) -> None:
        chip = ctx.store.get_sample(self.chip_id) or {"props": {}}
        corner = chip["props"].get("corner_stage_xy")
        rot = _chip_rotation(ctx, self.chip_id)
        for fid, r in self._results.items():
            flake = ctx.store.get_sample(fid) or {"props": {}}
            labels = {**flake["props"].get("labels", {}), **r["labels"]}
            props: dict[str, Any] = {"labels": labels,
                                     "images": {**flake["props"].get("images", {}), **r["images"]},
                                     "tracking_references": r.get("tracking_references", {}),
                                     "image_source": "simulated" if ctx.store.bench == "sim" else "camera"}
            for key in ("recognition_20x", "recognition_100x", "classification_event_100x", "recognition_event_id",
                        "mask_20x", "geometry_mask_20x", "recognition_status", "recognition_reason"):
                if key in r:
                    props[key] = r[key]
            if r.get("recognition_backend") == "v1_lab_rgb":
                props.update(optical_geometry=r.get("optical_geometry"), pickup_assessment=r.get("pickup_assessment"))
                for key in ("recognition_20x", "recognition_100x", "classification_event_100x",
                            "classification_100x", "recognition_event_id", "mask_20x", "geometry_mask_20x"):
                    props[key] = r.get(key)
                # Keep diagnostic evidence discoverable after publication and
                # clear an older native anchor if this revisit did not produce it.
                for key in ("measurement_20x", "measurement_100x", "measurement_event_20x",
                            "measurement_event_100x", "native_rotation_20x", "native_rotation_100x",
                            "native_registration", "native_source_anchor", "native_source_anchor_event_id",
                            "native_source_admission"):
                    props[key] = r.get(key)
            if r["found"]:
                props["optical_geometry"] = r.get("optical_geometry")
                props["pickup_assessment"] = r.get("pickup_assessment")
                props.update(thickness_label=labels.get("100x") or labels.get("20x"),
                             stage_xy=r["stage_xy"])
                if corner:
                    props["stage_from_corner_mm"] = [round(r["stage_xy"][0] - corner[0], 4),
                                                     round(r["stage_xy"][1] - corner[1], 4)]
                    lx, ly = _stage_to_local((corner[0], corner[1]), rot, *r["stage_xy"])
                    props["rel_corner_mm"] = [round(lx, 4), round(ly, 4)]  # RGB retains the original20 anchor.
            ctx.store.upsert_sample(fid, "flake", status="catalogued" if r["found"] else "lost",
                                    props=props)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


class SearchParams(BaseModel):
    chip_id: str | None = Field(None, description="Exfoliated chip; None = most recent one")
    material: str | None = Field(None, description="Default: the chip's ledger material")
    min_area_um2: float = Field(50.0, ge=0)
    max_revisits: int = Field(20, ge=0, le=50)
    revisit_quota: list[RevisitRole] | None = Field(
        None, description="Slots per flake role (the search_chip tool fills them from the "
        "device plan); None = v1's quotas")
    home_slot: str | None = None  # filled in at submit

    @field_validator("material")
    @classmethod
    def _norm(cls, v: str | None) -> str | None:
        return normalize_material_code(v)


@register_flow
class SearchChipFlow(Flow):
    name = "search_chip"
    description = ("Put an exfoliated chip under the microscope, find its corner, scan it at "
                   "10x, revisit the best flake candidates at 20x/100x, store it back. Flakes "
                   "go to the sample ledger.")
    Params = SearchParams
    devices = (ARM, CAMERA, STAGE, STAGE_VACUUM, MICROSCOPE)

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        assert isinstance(params, SearchParams)
        p = params.model_copy()
        bench = self.bench
        places = {n for n in reachable(bench, ARM) if is_slot(bench, n)}
        if p.chip_id:
            chip = store.get_sample(p.chip_id)
            if chip is None or chip["kind"] != "chip":
                raise ValueError(f"{p.chip_id} is not a known chip")
        else:
            booked = store.reserved_samples()  # chips a queued job already plans on
            chips = [s for s in store.list_samples(kind="chip")
                     if s["status"] == "exfoliated" and s["location"] in places
                     and not s["props"].get("reactor_substrate")
                     and s["id"] not in booked]
            if not chips:
                raise ValueError(f"no exfoliated chip on a board {ARM} reaches")
            chip = max(chips, key=lambda s: s["updated_at"] or 0)
        p.chip_id = chip["id"]
        if chip["props"].get("reactor_substrate"):
            raise ValueError(f"{p.chip_id} belongs to the dedicated reactor workflow")
        if chip["location"] not in places:
            raise ValueError(f"{p.chip_id} is at {chip['location']}, not on a board {ARM} "
                             f"reaches ({', '.join(bench[ARM]['reach'])})")
        p.home_slot = chip["location"]
        have = normalize_material_code(chip["props"].get("material"))
        if not have and not p.material:
            raise ValueError(f"{p.chip_id} has no material on record; give material")
        if p.material and have and p.material != have:
            raise ValueError(f"{p.chip_id} carries {have}, not {p.material}")
        p.material = p.material or have
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        # the chip's slot stays its own while the chip is away under the microscope
        assert isinstance(params, SearchParams) and params.chip_id and params.home_slot
        store.reserve("slot", params.home_slot, run_id, self.name,
                      note=f"{params.chip_id} comes back here")
        store.reserve("sample", params.chip_id, run_id, self.name, note="being searched")

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, SearchParams)
        p = params
        assert p.chip_id and p.material and p.home_slot
        return [
            MoveTransferStage("load", vacuum=False),
            PlaceChipOnTransferStage(p.chip_id),
            ChipToMicroscope(p.chip_id),
            FindChipCorner(p.chip_id),
            MeasureChipRotation(p.chip_id),
            ScanChip(p.chip_id, p.material, p.min_area_um2),
            RevisitCandidates(p.chip_id, p.material, p.max_revisits, p.revisit_quota),
            MoveTransferStage("load"),
            MoveObject(p.chip_id, "chip", arm=ARM, camera=CAMERA, dest=p.home_slot,
                       cfg_key="store_chip", name="store_chip_from_transfer_stage",
                       release_switch=STAGE_VACUUM, status="searched",
                       fallback=(slot_station(self.bench, p.home_slot) or "", "chip"),
                       param_path=["home_slot"]),
        ]


class RevisitParams(SearchParams):
    flake_ids: list[str] = Field(default_factory=list,
                                 description="Only these candidates; empty = by quota")


def prior_scan_evidence(store: Any, chip_id: str, material: str) -> dict[str, Any] | None:
    """Find a verified scan of the current ledger specimen, including zero yield.

    Use existing observation events, never searched labels, candidate counts or
    simulator truth. A later preparation submission, replacement, material change
    or reset invalidates old scans. Legacy records missing this evidence require
    a full scan; no new provenance is invented for them.
    """
    boundary, previous = 0, None
    for row in store._x("SELECT id,kind,payload_json FROM event WHERE kind IN "
                        "('sample.updated','sample.removed') AND json_extract(payload_json,'$.id')=? "
                        "ORDER BY id", (chip_id,)):
        value = json.loads(row["payload_json"])
        if row["kind"] == "sample.removed":
            boundary, previous = row["id"], None
            continue
        state = (value.get("kind"), value.get("status"),
                 normalize_material_code(value.get("props", {}).get("material")))
        if (previous is None or state[0] != previous[0] or state[2] != previous[2]
                or state[1] in ("bare", "available", "exfoliated") and state[1] != previous[1]):
            boundary = row["id"]
        previous = state
    # Submission itself invalidates prior evidence even if a failed preparation
    # never reached its ledger update. Its physical specimen may be uncertain.
    preparation = store._x("SELECT MAX(e.id) AS id FROM event e JOIN run r ON r.id=e.run_id "
        "WHERE r.flow='exfoliate_chips' AND e.kind IN ('job.submitted','job.params_changed') "
        "AND EXISTS (SELECT 1 FROM json_each(r.params_json,'$.chip_ids') WHERE value=?)", (chip_id,)).fetchone()
    boundary = max(boundary, preparation["id"] or 0)
    for row in store._x("SELECT e.id,e.run_id,e.payload_json,r.params_json FROM event e "
            "JOIN run r ON r.id=e.run_id WHERE e.kind='step.finished' AND e.id>? "
            "AND json_extract(e.payload_json,'$.name')='scan_chip' "
            "AND json_extract(e.payload_json,'$.status')='done' AND r.flow='search_chip' "
            "AND json_extract(r.params_json,'$.chip_id')=? ORDER BY e.id DESC", (boundary, chip_id)):
        params = json.loads(row["params_json"])
        if normalize_material_code(params.get("material")) != material:
            continue
        seq = json.loads(row["payload_json"]).get("seq")
        verified = store._x("SELECT id,payload_json FROM event WHERE run_id=? AND kind='step.verified' "
            "AND id>? AND id<? AND json_extract(payload_json,'$.name')='scan_chip' "
            "AND json_extract(payload_json,'$.seq')=? AND json_extract(payload_json,'$.verdict')='ok' "
            "ORDER BY id DESC LIMIT 1", (row["run_id"], boundary, row["id"], seq)).fetchone()
        if not verified:
            continue
        observation = json.loads(verified["payload_json"])
        measured, expected = observation.get("measurements") or {}, observation.get("expected") or {}
        counts = [measured.get("tiles_scanned"), measured.get("blurred_tiles"), measured.get("candidates"),
                  expected.get("tiles"), expected.get("max_blurred_tiles")]
        if (not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts)
                or counts[3] == 0 or counts[0] < counts[3] or counts[1] > counts[4]):
            continue
        return {"job_id": row["run_id"], "verified_event_id": verified["id"], "finished_event_id": row["id"],
                "specimen_boundary_event_id": boundary, "tiles_scanned": counts[0], "candidates": counts[2]}
    return None


@register_flow
class RevisitChipFlow(SearchChipFlow):
    name = "revisit_chip"
    description = ("Put a searched chip back under the microscope and photograph at 20x/100x "
                   "candidates the ledger has but never revisited (no rescan). Flakes are "
                   "found again from their position relative to the chip corner.")
    Params = RevisitParams

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        assert isinstance(params, RevisitParams)
        if not params.chip_id:
            raise ValueError("revisit_chip needs the chip_id of a searched chip")
        p = super().resolve_params(params, store)
        assert isinstance(p, RevisitParams)
        if prior_scan_evidence(store, p.chip_id, p.material) is None:
            raise ValueError(f"{p.chip_id} has no verified microscope scan for its current specimen. "
                "revisit_only cannot discover new flakes. First call search_chip with this chip_id and "
                "revisit_only=false for a full scan; do not exfoliate another chip on the basis of an empty revisit. "
                "A searched label or legacy candidates without scan provenance are insufficient. "
                "No revisit job was created.")
        for fid in p.flake_ids:
            f = store.get_sample(fid)
            if f is None or f["kind"] != "flake" or f["location"] != p.chip_id:
                raise ValueError(f"{fid} is not a flake on {p.chip_id}")
        return p

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, RevisitParams)
        p = params
        assert p.chip_id and p.material and p.home_slot
        return [
            MoveTransferStage("load", vacuum=False),
            PlaceChipOnTransferStage(p.chip_id),
            ChipToMicroscope(p.chip_id),
            FindChipCorner(p.chip_id),
            MeasureChipRotation(p.chip_id),
            RevisitCandidates(p.chip_id, p.material, p.max_revisits, p.revisit_quota,
                              only_new=True, flake_ids=p.flake_ids or None),
            MoveTransferStage("load"),
            MoveObject(p.chip_id, "chip", arm=ARM, camera=CAMERA, dest=p.home_slot,
                       cfg_key="store_chip", name="store_chip_from_transfer_stage",
                       release_switch=STAGE_VACUUM, status="searched",
                       fallback=(slot_station(self.bench, p.home_slot) or "", "chip"),
                       param_path=["home_slot"]),
        ]
