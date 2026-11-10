"""Sim perception: reads ground truth from the frame and adds noise.

SimLocator stands in for the overview camera's YOLO OBB detector calibrated by
the storage boards' QR markers (the deployed YOLO weights are not available);
SimMicroscopeAnalyzer stands in for v1's segmenter, corner detection and focus
metric.
"""

from __future__ import annotations

import math
import random
from typing import Any

from qumus_simlab.core.errors import TransientError
from qumus_simlab.core.polygons import area_um2, intersection, rectangle
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.bench import to_arm
from qumus_simlab.perception.base import (
    ContactDetection,
    CornerDetection,
    Detection,
    FlakeDetection,
    FringeDetection,
)
from qumus_simlab.testbench.material_adapter import (
    MODEL_VERSION,
    microscope_flake,
    prepare_catalog_pair,
)
from qumus_simlab.testbench.world import STAMP_HOLDER, TAPE_HOLDER, SimWorld

GR_LABELS = ["monolayer", "bilayer", "trilayer", "fewlayer", "manylayer"]
HBN_LABELS = ["0-10", "10-20", "20-30", "30-40", ">40"]


def thickness_label(material: str, layers: int | None) -> str:
    """Thickness class names as v1 uses them (segmenter/material.py, autosearch)."""
    if layers is None:
        return "unknown"
    if material == "hBN":
        nm = layers * 0.33
        return HBN_LABELS[min(int(nm // 10), 4)]
    if layers <= 3:
        return GR_LABELS[layers - 1]
    return "fewlayer" if layers < 10 else "manylayer"


class SimLocator:
    """The overview camera's detector: objects in a world region, returned in
    the frame of the arm that will act on them (`frame_of`, or "world").

    Error model (testbench.yaml locator): a per-object bias that stays the same
    from frame to frame until the object moves (lighting, the edge the detector
    locks on to, local calibration error: averaging frames cannot remove it),
    Gaussian jitter per frame with a heavier tail, objects missed in a frame, phantom detections, bare/exfoliated chips read
    the wrong way round. Objects carry no QR code, so detections carry no
    object id: the flow knows an object by where it is (the slot). The storage
    boards' corner QR markers calibrate the camera; while a board's markers are
    unreadable (fault `qr_unreadable`) a look at that board fails (transient).
    """

    def __init__(self, world: SimWorld, regions: dict[str, dict[str, Any]],
                 bench: dict[str, Any] | None = None) -> None:
        self.world = world
        self.regions = regions
        self.bench = bench or {}
        #: board station -> number of further looks that cannot read its QR markers
        self.qr_unreadable: dict[str, int] = {}
        loc = world.model.get("locator", {})
        self.tail_prob = float(loc.get("tail_prob", 0.0))
        self.tail_sigma = float(loc.get("tail_sigma_mm", 0.0))
        self.miss_prob = float(loc.get("miss_prob", 0.0))
        self.false_positive_prob = float(loc.get("false_positive_prob", 0.0))
        self.confuse_prob = float(loc.get("confuse_material_prob", 0.0))
        self.bias_sigma = float(loc.get("bias_sigma_mm", 0.0))
        self.bias_sigma_deg = float(loc.get("bias_sigma_deg", 0.0))
        self.noise_scale = float(world.model.get("microscope", {}).get("observation_noise_scale", 1.0))
        #: the storage boards the last look covered (chaos runs aim QR faults at them)
        self.last_boards: list[str] = []
        #: object id -> (where it was when the bias was drawn, (dx, dy, drot), draws)
        self._bias: dict[str, tuple[tuple[float, float, float], tuple[float, float, float],
                                    int]] = {}

    def _bias_of(self, o: dict[str, Any]) -> tuple[float, float, float]:
        """The object's frame-to-frame constant error; drawn anew once it moved.
        Own random stream, so it does not shift the world's other draws."""
        if self.bias_sigma <= 0 and self.bias_sigma_deg <= 0:
            return (0.0, 0.0, 0.0)
        here = (o["x"], o["y"], o["rotation"])
        old = self._bias.get(o["id"])
        if old is not None and abs(old[0][0] - here[0]) < 0.05 and \
                abs(old[0][1] - here[1]) < 0.05 and abs(old[0][2] - here[2]) < 0.2:
            return old[1]
        n = old[2] + 1 if old else 0
        r = random.Random(f"locator-bias:{self.world.seed}:{o['id']}:{n}")
        b = (r.gauss(0, self.bias_sigma), r.gauss(0, self.bias_sigma),
             r.gauss(0, self.bias_sigma_deg))
        self._bias[o["id"]] = (here, b, n)
        return b

    def _boards_in(self, box: dict[str, Any]) -> list[str]:
        out = []
        for name, st in self.bench.get("stations", {}).items():
            if st.get("kind") != "storage":
                continue
            r = st["region"]
            if box["x"][0] < r["x"][1] and r["x"][0] < box["x"][1] and \
                    box["y"][0] < r["y"][1] and r["y"][0] < box["y"][1]:
                out.append(name)
        return out

    def _to_frame(self, pose: Pose, frame_of: str) -> Pose:
        if frame_of == "world" or frame_of not in self.bench:
            return pose
        return to_arm(self.bench, frame_of, pose)

    def locate(self, frame: Frame, *, region: str, frame_of: str = "arm_right") -> list[Detection]:
        try:
            box = self.regions[region]
        except KeyError:
            raise KeyError(f"unknown region {region!r}; known: {sorted(self.regions)}") from None
        self.last_boards = self._boards_in(box)
        for board in self.last_boards:
            if self.qr_unreadable.get(board, 0) > 0:
                self.qr_unreadable[board] -= 1
                raise TransientError(f"overview camera: QR markers of {board} not readable "
                                     f"(no calibration for that board in this frame)",
                                     board=board)
        f, r = self.world.faults, self.world.observation_rng
        out = []
        if region == "tape":
            tape = frame.meta.get("sim_tape", {})
            if tape.get("material"):
                out.append(Detection(label="tape_crystal", pose=self._to_frame(
                    Pose(x=_mid(box["x"]), y=_mid(box["y"])), frame_of), confidence=0.9,
                                     props={"uses": tape.get("uses", 0)}))
        for o in frame.meta.get("sim_objects", []):
            on_tape = o["held_by"] == TAPE_HOLDER
            # the "tape" region sees only what sticks to the tape; others see free
            # objects and a stamp on the manipulator head (it is in plain view)
            visible = o["held_by"] in (None, STAMP_HOLDER)
            if (region == "tape") != on_tape or (not on_tape and not visible):
                continue
            if not (box["x"][0] <= o["x"] <= box["x"][1] and box["y"][0] <= o["y"] <= box["y"][1]):
                continue
            if r.random() < self.miss_prob:
                continue
            sx = f.locator_sigma_mm
            bx, by, brot = self._bias_of(o)
            ex, ey = bx + r.gauss(0, sx), by + r.gauss(0, sx)
            if r.random() < self.tail_prob:
                ex, ey = ex + r.gauss(0, self.tail_sigma), ey + r.gauss(0, self.tail_sigma)
            pose = Pose(x=o["x"] + ex, y=o["y"] + ey, z=o["z"],
                        rotation=o["rotation"] + brot + r.gauss(0, f.locator_sigma_deg))
            props: dict[str, Any] = {"on_tape": on_tape,
                                     "on_manipulator": o["held_by"] == STAMP_HOLDER}
            if o["kind"] == "chip":
                # v1 scan_materials_tool / YOLO classes: bare vs exfoliated chip
                seen = o.get("n_flakes", 0) > 0
                if r.random() < self.confuse_prob:
                    seen = not seen
                props["material_visible"] = seen
            confidence = r.uniform(0.75, 0.98)
            if self.noise_scale != 1:
                confidence = 0.865 + self.noise_scale * (confidence - 0.865)
            out.append(Detection(label=o["kind"], pose=self._to_frame(pose, frame_of),
                                 confidence=round(confidence, 2), props=props))
        if region not in ("tape", "workspace") and r.random() < self.false_positive_prob:
            # glare or tape residue read as a chip somewhere in the region
            ph = Pose(x=r.uniform(*box["x"]), y=r.uniform(*box["y"]),
                      rotation=r.uniform(-10, 10))
            out.append(Detection(label="chip", pose=self._to_frame(ph, frame_of),
                                 confidence=round(r.uniform(0.5, 0.7), 2),
                                 props={"on_tape": False, "on_manipulator": False,
                                        "material_visible": False}))
        return out


def _mid(rng: Any) -> float:
    return (rng[0] + rng[1]) / 2


class SimMicroscopeAnalyzer:
    def __init__(self, world: SimWorld) -> None:
        self.world = world
        self.cfg = world.model.get("microscope", {})

    @property
    def rng(self) -> random.Random:
        return self.world.observation_rng

    @property
    def noise_scale(self) -> float:
        return float(self.cfg.get("observation_noise_scale", 1.0))

    def sharpness(self, frame: Frame) -> float:
        return float(frame.meta.get("sharpness", 0.0))

    def plane_sharpness(self, frame: Frame, plane: str) -> float | None:
        """Synthetic analyzer observation only; no live World query or hardware calibration."""
        if plane not in ("chip", "stamp"):
            return None
        value = frame.meta.get(f"sharpness_{plane}")
        return (float(value) if not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) else None)

    @staticmethod
    def _plane(frame: Frame, plane: str) -> float:
        """Sharpness of one focal plane (the chip surface or the stamp film)."""
        return float(frame.meta.get(f"sharpness_{plane}", frame.meta.get("sharpness", 0.0)))

    def chip_in_view(self, frame: Frame) -> bool:
        return frame.meta.get("on_chip") is not None

    def metal_region(self, frame: Frame) -> dict:
        from qumus_simlab.core.polygons import convex_polygon

        region = frame.meta.get("metal_region") or {}
        if self._plane(frame, "chip") < 6 or region.get("chip_id") != frame.meta.get("on_chip"):
            raise ValueError("Metal geometry requires a focused view of the coated chip.")
        outline = convex_polygon(region.get("outline_mm", []))
        fw, fh = frame.meta["fov_mm"]
        if any(abs(x) > fw / 2 or abs(y) > fh / 2 for x, y in outline):
            raise ValueError("Metal region is clipped; acquire a wider view or calibrated tiles.")
        return {**region, "outline_mm": outline, "units": "mm", "frame": "camera"}

    def reactor_preparation(self, frame: Frame, samples: list[dict], *,
                            preparation_id: str) -> tuple[dict, dict[str, dict]]:
        """Verify transferred catalog sources through a fresh focused micrograph.

        This is the simulator's geometry verifier, not an empirical segmentation
        calibration. Missing, clipped, moved or out-of-focus evidence is refused.
        """
        if frame.device_id != "microscope" or self._plane(frame, "chip") < 6:
            raise ValueError("Preparation requires a focused ordinary-microscope observation")
        with self.world.lock:
            chip = self.world.objects.get(frame.meta.get("on_chip"))
            if chip is None or chip.on != "transfer_stage" or chip.held_by is not None:
                raise ValueError("Preparation carrier must be physically on the observed transfer stage")
            if frame.meta.get("chip_pose") != chip.pose.model_dump():
                raise ValueError("Carrier moved after the preparation image; capture a fresh image")
            center, fov = frame.meta.get("view_center_world_mm"), frame.meta.get("fov_mm")
            if not center or not fov:
                raise ValueError("Preparation image lacks camera geometry")
            observed = {f["sim_id"]: f for f in frame.meta.get("flakes", [])}
            for flake in chip.flakes:
                seen = observed.get(flake.id)
                wx, wy = chip.local_to_world(flake.x_mm, flake.y_mm)
                expected = microscope_flake(flake, (wx - center[0], wy - center[1]), fov,
                                             rotation_deg=chip.pose.rotation)
                if seen is None or expected is None or seen.get("chip") != chip.id \
                        or not math.isclose(expected["visible_area_um2"], flake.area_um2, rel_tol=1e-8):
                    raise ValueError("Both complete input outlines must be visible for preparation")
                a, b = seen.get("outline_mm", ()), expected["outline_mm"]
                if len(a) != len(b) or any(math.dist(p, q) > 1e-8 for p, q in zip(a, b, strict=True)):
                    raise ValueError("Physical flake geometry changed after the preparation image")
            return prepare_catalog_pair(chip, samples, preparation_id=preparation_id)

    def find_corner(self, frame: Frame, corner: str = "bl") -> CornerDetection | None:
        if self._plane(frame, "chip") < 3.0:
            return None
        s = self.cfg.get("corner_sigma_mm", 0.003)
        for c in frame.meta.get("chip_corners", []):
            if c["corner"] == corner:
                return CornerDetection(corner=corner, dx_mm=c["dx_mm"] + self.rng.gauss(0, s),
                                       dy_mm=c["dy_mm"] + self.rng.gauss(0, s), confidence=0.9)
        return None

    def fringes(self, frame: Frame) -> FringeDetection | None:
        fr = frame.meta.get("fringes")
        if fr is None or self._plane(frame, "chip") < 3.0:
            return None  # no rings, or the interface is out of focus
        s = self.cfg.get("fringe_gap_sigma_mm", 0.0003)  # about half a fringe order
        return FringeDetection(dx_mm=fr["dx_mm"] + self.rng.gauss(0, 0.003 * self.noise_scale),
                               dy_mm=fr["dy_mm"] + self.rng.gauss(0, 0.003 * self.noise_scale),
                               gap_mm=max(0.0, fr["gap_mm"] + self.rng.gauss(0, s)),
                               rings=fr["rings"], confidence=0.8)

    def contact(self, frame: Frame) -> ContactDetection | None:
        c = frame.meta.get("contact")
        s = self.cfg.get("contact_sigma_mm", 0.003)
        if c is None or c["radius_mm"] < self.cfg.get("contact_min_radius_mm", 0.02):
            return None  # Newton rings too small to see yet
        if self._plane(frame, "chip") < 3.0:
            return None  # the interface is out of focus: no rings to see
        return ContactDetection(dx_mm=c["dx_mm"] + self.rng.gauss(0, s),
                                dy_mm=c["dy_mm"] + self.rng.gauss(0, s),
                                radius_mm=max(0.0, c["radius_mm"] + self.rng.gauss(0, s)),
                                confidence=0.9, film_intact=c.get("film", True))

    def find_stamp_flakes(self, frame: Frame) -> list[FlakeDetection]:
        sharp = self._plane(frame, "stamp")
        out = []
        for fl in frame.meta.get("stamp_flakes", []):
            if sharp < 3.0 or (sharp < 6.0 and self.rng.random() < 0.5 * self.noise_scale):
                continue  # the film is out of focus: flakes on it are a blur
            out.append(FlakeDetection(
                dx_mm=fl["dx_mm"] + self.rng.gauss(0, 0.002 * self.noise_scale),
                dy_mm=fl["dy_mm"] + self.rng.gauss(0, 0.002 * self.noise_scale), area_um2=fl["area_um2"],
                material=fl["material"], thickness_label=thickness_label(fl["material"],
                                                                         fl["layers"]),
                confidence=0.7, meta={"sim_id": fl["sim_id"],
                    "recognition_profile": "synthetic-complete-outline-centroid-gaussian-v1",
                    "centroid_sigma_mm": .002*self.noise_scale,
                    "outline_mm": [[x - fl["dx_mm"], y - fl["dy_mm"]] for x, y in fl.get("outline_mm", [])],
                    "geometry_source": fl.get("geometry_source"),
                    "outline_complete": abs(fl.get("visible_area_um2", 0) - fl["area_um2"]) <= max(1e-6, fl["area_um2"] * 1e-8)}))
        return out

    def discovery_obstacles(self, frame: Frame) -> dict[str, Any]:
        """One conditional SIM observation from this exact immutable exposure.

        This extends the existing segmenter boundary, never a real-adapter or
        invisible-material absence claim. Incomplete/unknown support refuses.
        """
        import hashlib
        import json
        from pathlib import Path

        import numpy as np

        from qumus_simlab.core.optical_profiles import SYNTHETIC_OFF_CHIP_RGB
        from qumus_simlab.perception.tracking import load_pixels
        from qumus_simlab.records.discovery import OBSTACLE_PROFILE, validate_policy
        from qumus_simlab.testbench.render import supported_photometry_profile

        calibration = frame.meta.get("calibration", {})
        try:
            policy = validate_policy(calibration.get("stamp_discovery"))
        except ValueError as exc:
            raise ValueError("Discovery exposure lacks its declared conditional SIM profile") from exc
        if (calibration.get("source") != "synthetic" or frame.meta.get("objective") != 5
                or not supported_photometry_profile(calibration.get("photometry_profile"))
                or calibration.get("u_axis") != "+world_x" or calibration.get("v_axis") != "-world_y"
                or calibration.get("mm_per_pixel") != [2.6/480, 1.72/320]
                or frame.meta.get("fov_mm") != [2.6, 1.72]
                or frame.meta.get("replayed_from") or self._plane(frame, "chip") < 6
                or frame.meta.get("stamp") or frame.meta.get("electrodes")
                or not isinstance(frame.meta.get("on_chip"), str)):
            raise ValueError("Discovery map needs unobstructed sharp native5x source support")
        image = load_pixels(frame)
        if list(image.size) != calibration.get("image_size_px") or image.size != (480, 320):
            raise ValueError("Discovery exposure changed its fixed native raster")
        raw = frame.data or Path(frame.saved_path()).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        # One stable observation per exposure; never repeatedly sample noise
        # and select a favourable obstacle position from the same pixels.
        rng = random.Random(f"{OBSTACLE_PROFILE}:{digest}")
        flakes = frame.meta.get("flakes")
        if not isinstance(flakes, list) or len(flakes) > policy["max_obstacles_per_frame"]:
            raise ValueError("Discovery material classes/support are unknown or overflowed")
        sigma = .001*self.noise_scale
        if self.noise_scale != 1:
            raise ValueError("Discovery policy has no admission for changed observation noise")
        obstacles = []
        for fl in flakes:
            if (fl.get("material") not in ("Gr", "hBN", "WTe2", "ReSe2")
                    or not fl.get("outline_mm") or fl.get("geometry_source") != MODEL_VERSION):
                raise ValueError("Discovery contains unsupported or unknown material outlines")
            outline = [tuple(p) for p in fl["outline_mm"]]
            complete = abs(fl.get("visible_area_um2", 0)-fl.get("area_um2", 0)) <= max(1e-6, fl.get("area_um2", 0)*1e-8)
            if not complete:
                # A flake cut by the frame edge is seen only up to that edge:
                # the obstacle is its in-frame part. The clear domain must lie
                # inside the union of the maps' supported rectangles, so any
                # material beyond this frame is seen in another map or the
                # domain cannot reach it (records.discovery).
                inside = intersection(outline, rectangle(*frame.meta["fov_mm"]))
                if len(inside) >= 3 and area_um2(inside) >= area_um2(outline)*(1-1e-8):
                    raise ValueError("Discovery outline is incomplete without crossing the frame edge")
                if len(inside) < 3:
                    continue
                outline = inside
            dx, dy = rng.gauss(0, sigma), rng.gauss(0, sigma)
            obstacles.append({"material": fl["material"], "outline_mm":
                [[x+dx, y+dy] for x, y in outline], "complete": complete,
                **({} if complete else {"clipped_at": "frame_edge"})})
        width, height = image.size
        rgb = np.asarray(image, dtype=float)
        yy, xx = np.mgrid[:height, :width]
        gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
        off_chip = np.floor(gain[:, :, None]*np.array(SYNTHETIC_OFF_CHIP_RGB))
        saturated = (rgb.min(axis=2) <= 0) | (rgb.max(axis=2) >= 255)
        if saturated[30:height-32, 8:width-8].any():
            raise ValueError("Discovery map contains unknown saturated camera support")
        unknown = np.all(np.abs(rgb-off_chip) <= 3, axis=2)
        runs = []
        for y in range(30, height-32):
            xs = np.flatnonzero(unknown[y, 8:width-8])+8
            if not len(xs):
                continue
            starts = [int(xs[0])]
            ends = []
            for left, right in zip(xs, xs[1:], strict=False):
                if right != left+1:
                    ends.append(int(left)+1)
                    starts.append(int(right))
            ends.append(int(xs[-1])+1)
            runs.extend([[y, a, b] for a, b in zip(starts, ends, strict=True)])
        return {"profile": OBSTACLE_PROFILE, "source": "synthetic", "conditional_visible_completeness": True,
                "image_sha256": digest, "captured_at": frame.timestamp, "chip_id": frame.meta["on_chip"],
                "calibration": json.loads(json.dumps(calibration)), "obstacles": obstacles,
                "supported_roi_px": [8, 30, width-8, height-32],
                "unknown_runs_px": runs, "unknown_mask_sha256": hashlib.sha256(np.packbits(unknown).tobytes()).hexdigest(),
                "unknown_outside_roi": True, "centroid_sigma_mm": sigma,
                "scope": "Visible complete-outline SIM recognition only; invisible material and integrity are unknown"}

    def find_flakes(self, frame: Frame, material: str) -> list[FlakeDetection]:
        sharp = self._plane(frame, "chip")
        objective = frame.meta.get("objective", 10)
        p_wrong = self.cfg.get("misclassify_100x" if objective >= 50 else "misclassify_10x", 0.1)
        out = []
        for fl in frame.meta.get("flakes", []):
            if fl["material"] != material:
                continue
            if sharp < 3.0 or (sharp < 6.0 and self.rng.random() < 0.5 * self.noise_scale):
                continue  # blurred: missed
            # the segmenter errs the same way on the same flake at the same
            # magnification (contrast, edges): its misreading and area error are
            # fixed per (flake, objective), not drawn anew every frame
            k = random.Random(f"segmenter:{self.world.seed}:{fl['sim_id']}:{objective}")
            layers = fl["layers"]
            if k.random() < p_wrong and layers is not None:
                layers = max(1, layers + k.choice([-1, 1]))
            area_err = k.uniform(0.9, 1.1)
            if self.noise_scale != 1:
                area_err = 1.0 + self.noise_scale * (area_err - 1.0)
            px = 0.0005 * 10 / objective * self.noise_scale  # grows with pixel size
            out.append(FlakeDetection(
                dx_mm=fl["dx_mm"] + self.rng.gauss(0, px), dy_mm=fl["dy_mm"] + self.rng.gauss(0, px),
                area_um2=round(fl["area_um2"] * area_err, 1), material=material,
                thickness_label=thickness_label(material, layers),
                confidence=round(min(0.99, 0.5 + sharp / 20), 2),
                meta={"sim_id": fl["sim_id"], "sim_layers": fl["layers"],
                      "outline_mm": [[x - fl["dx_mm"], y - fl["dy_mm"]] for x, y in fl.get("outline_mm", [])],
                      "geometry_source": fl.get("geometry_source"),
                      "outline_complete": abs(fl.get("visible_area_um2", 0) - fl["area_um2"]) <= max(1e-6, fl["area_um2"] * 1e-8)}))
        return out
