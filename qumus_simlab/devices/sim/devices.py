"""Simulated devices backed by a SimWorld.

Each sim device implements the same interface as the real driver will
(devices/base.py) and changes or reads the simulated world. Frames carry the
ground truth in `meta` and the sim perception (testbench/locator.py) turns it
into noisy measurements; pictures are drawn or replayed only when a step asks
for the file (Frame.saved_path, testbench/render.py).
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any

from qumus_simlab.core.clock import Clock
from qumus_simlab.core.errors import (
    DeviceUnavailable,
    NeedsHumanError,
    TransientError,
)
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import (
    Arm,
    Camera,
    Frame,
    Heater,
    Microscope,
    MotionStage,
    StampManipulator,
    Switch,
    TapeMechanism,
)
from qumus_simlab.flows.bench import exfoliator_rolls, tape_height
from qumus_simlab.testbench.material_adapter import microscope_flake
from qumus_simlab.testbench.world import SimWorld


def _motion(clock: Clock | None, duration: float, update: Any, owner: Any) -> None:
    """Integrate a device stroke on the existing clock, exposing actual state.

    A command still has one policy/safety admission. Intermediate frames use its
    normal observer so replay has the same physical trajectory as live preview.
    """
    n = min(1024, max(1, math.ceil(duration / 0.25)))
    owner.motion_active, owner.last_motion_duration_s = True, duration
    try:
        for i in range(1, n + 1):
            if clock is not None and duration > 0:
                clock.sleep(duration / n)
            update(i / n)
            if owner.motion_observer is not None:
                owner.motion_observer()
    finally:
        owner.motion_active = False


def _lerp(start: float, end: float, f: float) -> float:
    """Stroke position; the completed stroke lands exactly on its target.

    start+(end-start)*1 can differ from end in the last bit, and exact
    readback comparisons (focus/stage context bindings) then see a change
    that no command made (a518 seed 8 release support).
    """
    return end if f >= 1 else start+(end-start)*f


class _Offline:
    offline: bool = False
    injector: Any = None  # testbench.faults.FaultInjector (serial timeouts, crashes)
    geometry_source: str = "synthetic"
    is_simulated: bool = True

    def _check_online(self: Any) -> None:
        if self.offline:
            raise DeviceUnavailable(f"{self.device_id} offline")
        if self.injector is not None:
            self.injector.before_call(self.device_id)


class SimArm(_Offline, Arm):
    """G-code arm in its own frame: world = base + pose (+ lost steps).

    Moves take time on the clock (distance / speed + settling), so heaters keep
    heating or cooling while an arm travels. `lose_position()` models a
    controller reset: until home() runs the arm does not know where it is (the
    SafetyGate refuses moves). `step_offset` models lost steps: the arm thinks
    it is at the commanded pose but is off by this much until it homes again.
    """

    def __init__(self, device_id: str, world: SimWorld, home: Pose,
                 base: tuple[float, float] = (0.0, 0.0), clock: Clock | None = None,
                 speed_mm_s: float = 0.0, settle_s: float = 0.0) -> None:
        super().__init__(device_id)
        self.world = world
        self._home = home
        self._pose = home
        self._vacuum = False
        self.base = base
        self.clock, self.speed, self.settle = clock, speed_mm_s, settle_s
        self.homed = True
        self.step_offset: tuple[float, float] = (0.0, 0.0)

    def world_pose(self, pose: Pose | None = None) -> Pose:
        p = pose or self._pose
        return p.offset(dx=self.base[0] + self.step_offset[0],
                        dy=self.base[1] + self.step_offset[1])

    def _travel(self, a: Pose, b: Pose) -> None:
        if self.clock is not None and self.speed > 0:
            d = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
            self.clock.sleep(d / self.speed + self.settle)

    def move_to(self, pose: Pose, speed: float | None = None) -> None:
        self._check_online()
        self._travel(self._pose, pose)
        start = self.world_pose()
        self._pose = pose
        self.world.carry(self.device_id, self.world_pose(), start=start)

    def get_pose(self) -> Pose:
        return self._pose

    def home(self) -> None:
        self._check_online()
        # homing: straight up first, then to the home pose; re-zeroes lost steps
        up = Pose(x=self._pose.x, y=self._pose.y, z=max(self._pose.z, self._home.z),
                  rotation=self._pose.rotation)
        self._travel(self._pose, up)
        self._pose = up
        self.world.carry(self.device_id, self.world_pose())
        self.step_offset = (0.0, 0.0)
        self.homed = True
        self.move_to(self._home)

    def is_homed(self) -> bool:
        return self.homed

    def lose_position(self) -> None:
        self.homed = False

    def set_vacuum(self, on: bool) -> None:
        self._check_online()
        if on and not self._vacuum:
            self.world.pick(self.device_id, self.world_pose())
        elif not on and self._vacuum:
            self.world.release(self.device_id, self.world_pose())
        self._vacuum = on

    def vacuum_on(self) -> bool:
        return self._vacuum

    def safe_state(self) -> None:
        # Stop where it is; do not drop a held object mid-air.
        pass


class SimStage(_Offline, MotionStage):
    """One coarse/fine/rotary stage carrying its entire mounted assembly.

    Fine XY stays bench-aligned. Rotation is about the translated surface
    centre, preserving each chip's actual mounting offset, not its centroid.
    All geometry is an explicitly synthetic fixture, not real calibration.
    """

    def __init__(self, device_id: str, world: SimWorld | None = None,
                 surface: str | None = None, clock: Clock | None = None,
                 speed_mm_s: float = 0.0, fine_speed_mm_s: float = 0.5,
                 rotation_speed_deg_s: float = 10.0) -> None:
        super().__init__(device_id)
        self.world, self.surface = world, surface
        self.clock, self.speed = clock, speed_mm_s
        self.fine_speed, self.rotation_speed = fine_speed_mm_s, rotation_speed_deg_s
        self._pose = Pose(x=0, y=0, z=0)
        self._fine = (0.0, 0.0)
        self._err = (0.0, 0.0)  # where the stage really is, minus the commanded pose

    #: a move shorter than this is a jog: it keeps the stage's present error
    JOG_MM = 0.05

    def move_to(self, pose: Pose) -> None:
        self._check_online()
        a = self._pose
        duration = (max(abs(pose.x-a.x), abs(pose.y-a.y), abs(pose.z-a.z)) / self.speed
                    if self.speed > 0 else 0.0)
        if self.rotation_speed > 0:
            duration = max(duration, abs(pose.rotation-a.rotation) / self.rotation_speed)
        old_err, err = self._err, self._err
        if self.world is not None and self.surface:
            # The encoder reads back the commanded pose; the real position is a
            # little off, which the microscope then sees. A long move lands
            # within the stage sigma (repeatability); a short corrective jog
            # keeps that error and adds only a little, so re-centring converges.
            r, s = self.world.rng, self.world.faults.stage_sigma_mm
            moved = math.hypot(pose.x - a.x, pose.y - a.y)
            if moved > self.JOG_MM:
                err = (r.gauss(0, s), r.gauss(0, s))
            elif moved > 0:  # a z-only move (focusing) keeps x/y as they are
                err = (self._err[0] + r.gauss(0, s / 5), self._err[1] + r.gauss(0, s / 5))

        def update(f: float) -> None:
            self._pose = Pose(**{k: _lerp(getattr(a, k), getattr(pose, k), f)
                                 for k in ("x", "y", "z", "rotation")})
            self._err = tuple(_lerp(v, end, f) for v, end in zip(old_err, err, strict=True))
            self._apply_surface()

        _motion(self.clock, duration, update, self)

    def get_pose(self) -> Pose:
        return self._pose

    def move_fine_to(self, x: float, y: float) -> None:
        self._check_online()
        if not all(math.isfinite(v) for v in (x, y)):
            raise ValueError("fine translation must be finite")
        a, b = self._fine, (x, y)
        duration = max(abs(x-a[0]), abs(y-a[1])) / self.fine_speed if self.fine_speed > 0 else 0.0

        def update(f: float) -> None:
            self._fine = tuple(_lerp(v, end, f) for v, end in zip(a, b, strict=True))
            self._apply_surface()

        _motion(self.clock, duration, update, self)

    def get_fine_position(self) -> tuple[float, float]:
        return self._fine

    def _apply_surface(self) -> None:
        if self.world is not None and self.surface:
            p = self.get_sample_pose()
            self.world.move_surface(self.surface, p.x+self._err[0], p.y+self._err[1],
                                    p.z, p.rotation)
            if self.clock is not None:
                self.world.advance_transfer(self.clock.now())

    def kinematics(self) -> dict[str, Any]:
        out = {**super().kinematics(), "mechanics_version": 2,
               "order": "coarse_translation,fine_translation,pivot_rotation"}
        if self.world is not None and self.surface in self.world.surfaces:
            sf = self.world.surfaces[self.surface]
            px, py = sf.pivot()
            carried = [o for o in self.world.objects.values()
                       if o.on == self.surface and o.held_by is None]
            radius = max([math.hypot(*o.size_mm)/2 + math.hypot(o.pose.x-px, o.pose.y-py)
                          for o in carried] or [0.0])
            tilt = abs(self.world.model.get("microscope", {}).get("focus_tilt_per_mm", 0.0))
            top = max([sf.height + o.focus_offset + tilt*sum(o.size_mm)/2 for o in carried]
                      or [sf.height])
            out.update(pivot_world_mm=[px, py], carried_radius_mm=radius,
                       max_sample_surface_z=top)
        return out


class SimCamera(_Offline, Camera):
    """The overview camera over the whole table. Returns a Frame whose meta
    carries the world snapshot; with a renderer it also draws the top-down view
    to an image file on Frame.saved_path(). `drop_next` frames fail to arrive
    (a USB camera dropping frames)."""

    def __init__(self, device_id: str, world: SimWorld, clock: Clock,
                 renderer: Any = None, arms: Any = None) -> None:
        super().__init__(device_id)
        self.world, self.clock = world, clock
        self.renderer = renderer  # testbench.camera_render.BenchCameraRenderer
        self.arms = arms  # () -> {arm_id: {x, y, z, vacuum}} (world) for position markers
        self.drop_next = 0

    def capture(self) -> Frame:
        self._check_online()
        if self.drop_next > 0:
            self.drop_next -= 1
            raise TransientError(f"{self.device_id}: frame dropped (no image from the camera)",
                                 device=self.device_id)
        meta = {"sim_objects": self.world.snapshot(), "sim_tape": self.world.tape_snapshot()}
        frame = Frame(device_id=self.device_id, timestamp=self.clock.now(), meta=meta)
        if self.renderer is not None:
            arms = self.arms() if self.arms else {}
            frame.save = lambda: self.renderer.render(self.device_id, meta["sim_objects"],
                                                      meta["sim_tape"], arms)
        return frame

    def preview(self) -> Frame | None:
        """The table as it is now, drawn to one live file (no fault, no log)."""
        if self.offline or self.renderer is None:
            return None
        meta = {"sim_objects": self.world.snapshot(), "sim_tape": self.world.tape_snapshot()}
        arms = self.arms() if self.arms else {}
        path = self.renderer.render(self.device_id, meta["sim_objects"], meta["sim_tape"],
                                    arms, fixed=True)
        return Frame(device_id=self.device_id, timestamp=self.clock.now(), path=path)


class SimMicroscope(_Offline, Microscope):
    """Independent vertical microscope focus over a moving sample assembly.

    Mechanical focus moves the optical plane, never the sample or stamp. The
    additional apparent shift through a stamp remains an optical parameter.
    """

    def __init__(self, device_id: str, clock: Clock, world: SimWorld | None = None,
                 center: tuple[float, float] = (0.0, 0.0),
                 fov_mm_10x: tuple[float, float] = (1.3, 0.86),
                 focal_plane: float = 0.0, renderer: Any = None,
                 stage: MotionStage | None = None, focus_speed_mm_s: float = 1.0,
                 lateral_focus_slope_norm_bound_mm_per_mm: float | None = None,
                 apex_drift_clip_per_axis_mm: float | None = None,
                 source_refraction_projection_norm_bound_mm_per_mm: float | None = None,
                 apex_lateral_drift_speed_norm_bound_mm_s: float | None = None,
                 apex_approach_drift_norm_bound_mm_per_mm: float | None = None,
                 dome_curvature_radius_mm: float | None = None,
                 contact_slip_rate_bound_deg_s: float | None = None,
                 stamp_discovery_profile: dict[str, Any] | None = None) -> None:
        super().__init__(device_id)
        self.clock, self.world = clock, world
        self.renderer = renderer  # testbench.render.MicroscopeRenderer
        self.center, self.fov_10x = center, fov_mm_10x
        self._focus, self.nominal_focus = 0.0, focal_plane
        self.stage, self.focus_speed = stage, focus_speed_mm_s
        self.lateral_focus_slope_norm_bound = lateral_focus_slope_norm_bound_mm_per_mm
        self.apex_drift_clip_per_axis_mm = apex_drift_clip_per_axis_mm
        self.source_refraction_projection_norm_bound = source_refraction_projection_norm_bound_mm_per_mm
        self.apex_lateral_drift_speed_norm_bound = apex_lateral_drift_speed_norm_bound_mm_s
        self.apex_approach_drift_norm_bound = apex_approach_drift_norm_bound_mm_per_mm
        self.declared_dome_curvature_radius_mm = dome_curvature_radius_mm
        self.contact_slip_rate_bound_deg_s = contact_slip_rate_bound_deg_s
        if stamp_discovery_profile is not None:
            from qumus_simlab.records.discovery import validate_generator_config

            if world is None or validate_generator_config(world.model) != stamp_discovery_profile:
                raise ValueError("Discovery calibration requires its matching simulated generator configuration")
        self.stamp_discovery_profile = deepcopy(stamp_discovery_profile)
        self._objective = 10

    def fov_mm(self) -> tuple[float, float]:
        return tuple(self._exposure_geometry()["fov_mm"])

    def _exposure_geometry(self, purpose: str = "analysis_capture") -> dict[str, Any]:
        k = 10.0 / self._objective
        fov = (self.fov_10x[0] * k, self.fov_10x[1] * k)
        profile = getattr(self.renderer, "exposure_geometry", None)
        if callable(profile):
            return profile(self._objective, fov, purpose=purpose)
        width, height = self.renderer.size if self.renderer is not None else (640, 424)
        return {"image_size_px": [width, height], "fov_mm": list(fov),
                "mm_per_pixel": [fov[0]/width, fov[1]/height]}

    def _exposure_calibration(self, purpose: str = "analysis_capture") -> dict[str, Any]:
        """Copy the optical assumptions of this exposure, outside mechanics."""
        geometry = self._exposure_geometry(purpose)
        fw, fh = geometry["fov_mm"]
        width, height = geometry["image_size_px"]
        calibration = {"id": "synthetic-microscope-optics",
            "version": f"v2:{self._objective}:{width}x{height}:{fw}:{fh}",
            "source": "synthetic", "frame": "microscope_image", "image_size_px": [width, height],
            "mm_per_pixel": [fw/width, fh/height], "u_axis": "+world_x", "v_axis": "-world_y"}
        if geometry.get("material_capture_profile"):
            calibration["material_capture_profile"] = geometry["material_capture_profile"]
        if geometry.get("neighbourhood_capture_profile"):
            calibration["neighbourhood_capture_profile"] = geometry["neighbourhood_capture_profile"]
        if geometry.get("camera_profile"):
            calibration["camera_profile"] = geometry["camera_profile"]
        profile = getattr(self.renderer, "photometry_profile", None)
        if isinstance(profile, dict):
            calibration["photometry_profile"] = deepcopy(profile)
        if self.world is not None and (structure := self.world.model.get("stamp", {}).get("structure")):
            calibration["stamp_structure_model"] = {
                "source": "synthetic", "definition": deepcopy(structure),
                "slide": deepcopy(self.world.stamp_geometry["bench"]["stamp_slide"])}
        calibration["observation_condition"] = self._exposure_observation_condition()
        if self.stamp_discovery_profile is not None:
            calibration["stamp_discovery"] = deepcopy(self.stamp_discovery_profile)
        if self.world is not None:
            from qumus_simlab.core.optical_profiles import RELEASE_SUPPORT_ASSUMPTIONS
            calibration["release_support_profile"] = deepcopy(RELEASE_SUPPORT_ASSUMPTIONS)
        return calibration

    def _exposure_observation_condition(self) -> dict[str, Any]:
        """Freeze the shared SIM condition; rendering never reads live config."""
        config = self.world.model if self.world is not None else {}
        condition = {"observation_noise_scale": float(
            config.get("microscope", {}).get("observation_noise_scale", 1.0))}
        if "simulation_condition" in config:
            condition["simulation_condition"] = config["simulation_condition"]
        return condition

    def capture(self) -> Frame:
        return self._capture("analysis_capture")

    def capture_for_focus(self) -> Frame:
        return self._capture("focus_capture")

    def capture_for_neighbourhood(self) -> Frame:
        return self._capture("neighbourhood_capture")

    def _capture(self, purpose: str) -> Frame:
        self._check_online()
        timestamp = self.clock.now()
        geometry = self._exposure_geometry(purpose)
        fw, fh = geometry["fov_mm"]
        width, height = geometry["image_size_px"]
        meta: dict[str, Any] = {"focus": self._focus, "focus_z": self._focus,
                                "mechanics_version": 2, "geometry_source": "synthetic",
                                "objective": self._objective,
                                "captured_at": timestamp, "purpose": purpose,
                                "replayed_from": None,
                                "fov_mm": [fw, fh],
                                "calibration": self._exposure_calibration(purpose),
                                **self._exposure_observation_condition()}
        if geometry.get("material_capture_profile"):
            meta["material_capture_profile"] = geometry["material_capture_profile"]
        if self.world is not None:
            meta.update(self._view(fov_mm=(fw, fh)))
        meta["pose_binding"] = {"captured_at": timestamp, "focus_z": self._focus,
                                "objective": self._objective,
                                "sample_pose": meta.get("sample_pose"),
                                "stage_kinematics": meta.get("stage_kinematics")}
        frame = Frame(device_id=self.device_id, timestamp=timestamp, meta=meta,
                      width=width, height=height)
        if self.renderer is not None:
            renderer = self.renderer
            pixels = renderer.render_bytes(self.device_id, meta)
            frame.data = pixels
        if self.renderer is not None and self.renderer.writer is not None:
            def save() -> str:
                return renderer.writer.write(self.device_id, pixels)
            frame.save = save
        return frame

    def preview(self) -> Frame | None:
        """Current synthetic scene in one ephemeral UI image, never evidence."""
        return self._preview(immutable=False)

    def preview_snapshot(self) -> Frame | None:
        """Immutable saved motion frame for lab.frame/replay, never verification."""
        return self._preview(immutable=True)

    def preview_signature(self) -> str:
        """Optical changes only, quantized to raster resolution; no image write."""
        return self.preview_state()[0]

    def preview_state(self) -> tuple[str, bool]:
        """(signature, detailed) of the live picture; no image write.

        Detailed while white-light rings or a contact are in view: then every
        optical change is a new picture. During a far stamp approach the gap
        and the film's focus are left out (focus only in coarse steps), so the
        display holds one still picture while simulated time runs (#205).
        """
        from qumus_simlab.testbench.optics import WHITE_LIGHT_MAX_GAP_MM

        scene = self._view() if self.world is not None else {}
        gap = scene.get("optical_gap")
        detailed = isinstance(gap, dict) and (bool(gap.get("has_contact"))
                                              or gap["signed_apex_gap_mm"] < WHITE_LIGHT_MAX_GAP_MM)
        far = isinstance(gap, dict) and not detailed and scene.get("contact") is None
        geometry = self._exposure_geometry("motion_preview")
        width, height = geometry["image_size_px"]
        fw, fh = geometry["fov_mm"]
        quantum = min(fw/width, fh/height)/2

        def quantize(value: Any, key: str = "") -> Any:
            if isinstance(value, dict):
                return {k: quantize(v, k) for k, v in value.items()}
            if isinstance(value, list | tuple):
                return [quantize(v, key) for v in value]
            if isinstance(value, float):
                step = ((2.0 if far and key == "sharpness_stamp" else 0.05)
                        if key.startswith("sharpness") else quantum)
                return round(value/step)
            return value

        data = {k: scene.get(k) for k in ("on_chip", "chip_corners", "flakes", "stamp_flakes",
                "stamp", "contact", "fringes", "electrodes", "metal_region",
                "sharpness_chip", "sharpness_stamp", "specimen_texture", "stamp_texture_seed")}
        data.update(objective=self._objective, fov_mm=list(self.fov_mm()),
                    photometry_profile=self._exposure_calibration().get("photometry_profile"),
                    **self._exposure_observation_condition())
        # Optical phase can change below one spatial pixel. Keep this frozen
        # packet exact instead of rounding it by the image-position quantum.
        signature = {"raster": quantize(data), "optical_gap": "beyond_rings" if far else gap,
                     "stamp_structure": scene.get("stamp_structure")}
        return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(), detailed

    def _preview(self, *, immutable: bool) -> Frame | None:
        if self.offline:
            return None
        timestamp = self.clock.now()
        calibration = self._exposure_calibration("motion_preview")
        meta = {"objective": self._objective, "focus_z": self._focus,
                "fov_mm": list(self.fov_mm()), "purpose": "motion_preview",
                "verification": False, "source": "synthetic_scene", "immutable": immutable,
                "captured_at": timestamp, "calibration": calibration,
                **self._exposure_observation_condition()}
        if self.world is not None:
            meta.update(self._view())
        path = (self.renderer.render_preview(self.device_id, meta, immutable=immutable)
                if self.renderer is not None and self.renderer.writer is not None else None)
        return Frame(device_id=self.device_id, timestamp=timestamp, path=path, meta=meta,
                     width=calibration["image_size_px"][0], height=calibration["image_size_px"][1])

    def _view(self, *, fov_mm: tuple[float, float] | None = None) -> dict[str, Any]:
        w = self.world
        assert w is not None
        w.advance_transfer(self.clock.now())
        cx, cy = self.center
        fw, fh = self.fov_mm() if fov_mm is None else fov_mm
        mcfg = w.model.get("microscope", {})
        dof = mcfg.get("depth_of_focus_10x", 0.15) * 10.0 / self._objective
        focal = self.nominal_focus + self._focus
        out: dict[str, Any] = {"on_chip": None, "chip_corners": [], "flakes": [],
                               "sharpness": 0.5, "sharpness_chip": 0.5,
                               "sharpness_stamp": 0.5, "view_center_world_mm": [cx, cy]}
        scfg = w.model.get("stamp", {})
        with w.lock:
            # A stamp over the stage lies in the light path: through the glass
            # slide and the PDMS dome the chip (and the film) look optic_shift
            # higher, so the focus that was sharp before is not any more, and
            # the picture loses a little contrast
            ap = w.apex()
            shift = ap[1].optic_shift if ap is not None else 0.0
            peak = 10.0 * (scfg.get("through_stamp_contrast", 1.0) if ap is not None else 1.0)
            for o in w.objects.values():
                if o.kind != "chip" or o.held_by is not None or o.on != "transfer_stage":
                    continue
                # chip-local coordinates of the view centre
                r = math.radians(o.pose.rotation)
                dx, dy = cx - o.pose.x, cy - o.pose.y
                lx = dx * math.cos(r) + dy * math.sin(r)
                ly = -dx * math.sin(r) + dy * math.cos(r)
                hw, hh = o.size_mm[0] / 2, o.size_mm[1] / 2
                reach = max(fw, fh)
                if abs(lx) > hw + reach or abs(ly) > hh + reach:
                    continue
                out["specimen_texture"] = {"profile": "chip-local-procedural-v1",
                    "seed": f"{w.seed}:{o.id}", "chip_id": o.id,
                    # what the chip was exfoliated with: its debris (#182)
                    "material": getattr(o, "material", None),
                    "pose": o.pose.model_dump(), "size_mm": list(o.size_mm),
                    "view_center_world_mm": [cx, cy],
                    # Jialei 2026-11-04: tape residue is not simulated where
                    # the profile turns it off (too hard to model faithfully).
                    **({} if w.model.get("microscope", {}).get("residue", True) else {"residue": False})}
                if abs(lx) <= hw and abs(ly) <= hh:
                    out["on_chip"] = o.id
                    out["chip_pose"] = o.pose.model_dump()
                    out["electrodes"] = [
                        {**shape, "points": [(wx-cx, wy-cy) for wx, wy in
                                             (o.local_to_world(x, y) for x, y in shape["points"])]}
                        for shape in getattr(o, "electrodes", [])]
                for disc in getattr(o, "pc_films", []):
                    # a released PC film disc left on the chip (#180)
                    wx, wy = o.local_to_world(disc["x_mm"], disc["y_mm"])
                    if abs(wx-cx) <= fw/2+disc["radius_mm"] and abs(wy-cy) <= fh/2+disc["radius_mm"]:
                        out.setdefault("pc_films", []).append({"chip": o.id, "dx_mm": wx-cx, "dy_mm": wy-cy,
                                                               "radius_mm": disc["radius_mm"]})
                coating = w.sample_properties(o.id).get("metal_coating")
                patch = w.model.get("reactor", {}).get("metal_patch_outline_mm")
                if coating and patch:
                    out["metal_region"] = {"chip_id": o.id, "metal": coating["metal"],
                        "outline_mm": [[wx - cx, wy - cy] for wx, wy in (o.local_to_world(x, y) for x, y in patch)],
                        "source": "simulated_optical_outline"}
                if abs(lx) <= hw + fw / 2 and abs(ly) <= hh + fh / 2:
                    # some chip surface in view: the focus metric sees its texture
                    px, py = max(-hw, min(hw, lx)), max(-hh, min(hh, ly))
                    height = (w.stage_height() + o.focus_offset
                              + mcfg.get("focus_tilt_per_mm", 0.0) * (px + py))
                    seen = height + shift
                    out["sharpness_chip"] = round(
                        peak * math.exp(-((seen - focal) / dof) ** 2) + 0.5, 3)
                    # ground truth, for tests only: the stage z that focuses here,
                    # and the focal plane that would (v1compare's lens model)
                    out["best_stage_z"] = w.stage_height() + focal - seen
                    out["best_focus"] = seen
                    out["best_focus_z"] = seen-self.nominal_focus
                for name, (kx, ky) in {"bl": (-hw, -hh), "br": (hw, -hh), "tl": (-hw, hh),
                                       "tr": (hw, hh)}.items():
                    wx, wy = o.local_to_world(kx, ky)
                    if abs(wx - cx) <= fw / 2 and abs(wy - cy) <= fh / 2:
                        out["chip_corners"].append({"chip": o.id, "corner": name,
                                                    "dx_mm": wx - cx, "dy_mm": wy - cy,
                                                    "chip_rotation": o.pose.rotation})
                for f in o.flakes:
                    wx, wy = o.local_to_world(f.x_mm, f.y_mm)
                    visible = microscope_flake(f, (wx - cx, wy - cy), (fw, fh),
                                                rotation_deg=o.pose.rotation)
                    if visible is not None:
                        level = w.stack_levels.get((o.id, f.id), 0)
                        out["flakes"].append({**visible, "chip": o.id, **({"stack_level": level} if level else {})})
            out["stamp_flakes"] = []
            if ap is not None:  # a stamp hangs over the stage, seen through
                stamp_object, st, (ax, ay), depth = ap
                out["stamp_texture_seed"] = f"{w.seed}:{stamp_object.id}"
                out["stamp"] = {"film": st.film}
                from qumus_simlab.testbench.optics import (
                    WHITE_LIGHT_PROFILE,
                    stamp_structure_packet,
                )

                structure = stamp_structure_packet(w.stamp_geometry["bench"], scfg.get("structure", {}),
                                                   w.head_xy(), (cx, cy))
                if structure is not None:
                    out["stamp_structure"] = structure

                # Renderer-only SIM truth from this same Clock projection. It
                # is not a camera measurement or a control/gap certificate.
                out["optical_gap"] = {"signed_apex_gap_mm": -depth,
                    "dome_radius_mm": scfg.get("dome_radius_mm", 1.5),
                    "centre_dx_mm": ax-cx, "centre_dy_mm": ay-cy,
                    "current_contact_radius_mm": st.front_radius_mm,
                    "has_contact": depth > 0, "profile_id": WHITE_LIGHT_PROFILE}
                # The film is its own plane: its height is the holder z above
                # the dome's touch height (seen optic_shift higher, like the
                # chip). It is sharp only when the holder brings it into the
                # focal plane (dof is tens of um at 20x), with the chip moved
                # down out of the way first.
                film = w.manip.z - st.touch_z + shift
                out["sharpness_stamp"] = round(
                    8.0 * math.exp(-((film - focal) / dof) ** 2) + 0.5, 3)
                out["best_stamp_focus_z"] = film-self.nominal_focus
                # ground truth, for tests only: the holder z that focuses the film
                out["best_holder_z"] = st.touch_z + focal - shift
                if depth > 0:
                    out["contact"] = {"dx_mm": ax - cx, "dy_mm": ay - cy,
                                      "radius_mm": st.front_radius_mm, "film": st.film}
                    if st.pinned_radius_mm > 0 and st.pinned_xy is not None:
                        # molten film pinned to the chip; on lift the PDMS
                        # separates inside it at the shrinking front (#180)
                        out["contact"]["pinned"] = {"dx_mm": st.pinned_xy[0]-cx, "dy_mm": st.pinned_xy[1]-cy,
                                                    "radius_mm": st.pinned_radius_mm}
                elif -depth < scfg.get("fringe_max_gap_mm", 0.0):
                    # a few um of air between the dome and the chip: Newton
                    # rings around the nearest point, one per half wavelength
                    lam = scfg.get("fringe_wavelength_mm", 0.00055)
                    out["fringes"] = {"dx_mm": ax - cx, "dy_mm": ay - cy, "gap_mm": -depth,
                                      "rings": int(2 * -depth / lam) + 1}
                for fl, ox, oy in st.stack:
                    fx, fy = ax + ox - cx, ay + oy - cy
                    visible = microscope_flake(fl, (fx, fy), (fw, fh))
                    if visible is not None:
                        out["stamp_flakes"].append(visible)
        # a focus metric sees whatever plane is sharp
        out["sharpness"] = max(out["sharpness_chip"], out["sharpness_stamp"])
        out["focus_plane"] = ("chip" if out["sharpness_chip"] >= 3.0 else
                              "stamp" if out["sharpness_stamp"] >= 3.0 else None)
        geometry = self.geometry_state()
        out["sample_pose"] = geometry.get("sample_pose")
        out["stage_kinematics"] = self.stage.kinematics() if self.stage is not None else None
        out["scene"] = {"version": 1, "source": "synthetic", "frame": "world",
                        "units": {"position": "mm", "rotation": "deg"},
                        "mechanics_version": 2, **geometry}
        if ap is not None:
            gains = scfg.get("dynamics", {}).get("refraction_xy_per_gap_mm", [0.0, 0.0])
            gap = max(0.0, -ap[3])
            rx, ry = float(gains[0])*gap, float(gains[1])*gap
            out["scene"]["refractive_shift_xy_mm"] = [rx, ry]
            if "specimen_texture" in out:
                out["specimen_texture"]["apparent_shift_xy_mm"] = [rx, ry]
            # Apply the optical map to the rendered source layer, not to any
            # mechanical pose or carried-stack coordinates.
            for feature in [*out["flakes"], *out["chip_corners"]]:
                feature["dx_mm"] += rx
                feature["dy_mm"] += ry
                if feature.get("outline_mm"):
                    feature["outline_mm"] = [(x+rx, y+ry) for x, y in feature["outline_mm"]]
            for shape in out.get("electrodes", []):
                shape["points"] = [(x+rx, y+ry) for x, y in shape["points"]]
        return out

    def geometry_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": "synthetic", "focus_z": self._focus,
                               "nominal_focus": self.nominal_focus,
                               "optical_axis_world_mm": list(self.center)}
        if self.world is None:
            return out
        w = self.world
        with w.lock:
            sf = w.surfaces.get("transfer_stage")
            if sf is not None:
                px, py = sf.pivot()
                out["sample_pose"] = {"x": px, "y": py, "z": sf.height, "rotation": sf.rotation}
                out["sample_surface_z"] = sf.height
            if self.stage is not None:
                out["max_sample_surface_z"] = self.stage.kinematics().get("max_sample_surface_z")
            if w.stamp_geometry:
                hx, hy = w.head_xy()
                out["stamp_holder_xyz"] = {"x": hx, "y": hy, "z": w.manip.z}
            else:
                out["stamp_holder_xyz"] = None
            out["stamp_axes_xyz_mm"] = {"x": w.manip.x, "y": w.manip.y, "z": w.manip.z}
            out["stamp_axes_frame"] = "transfer_arm"
            ap = w.apex()
            out["stamp"] = None if ap is None else {
                "id": ap[0].id, "apex_world_xy_mm": list(ap[2]),
                "film_plane_z": w.manip.z-ap[1].touch_z,
                "optic_shift_mm": ap[1].optic_shift, "contact_depth_mm": ap[3],
                "mechanical_drift_xy_mm": list(ap[1].drift_xy_mm),
                "slip_rotation_deg": ap[1].slip_rotation_deg,
                "front_radius_mm": ap[1].front_radius_mm}
        return out

    def focus_plane(self) -> str | None:
        """Which surface is in focus now: "chip", "stamp" or None (for the UI)."""
        return self._view().get("focus_plane") if self.world is not None else None

    def get_objective(self) -> int:
        return self._objective

    def set_focus(self, z: float) -> None:
        self._check_online()
        if not math.isfinite(z):
            raise ValueError("focus Z must be finite")
        old = self._focus
        duration = abs(z-old)/self.focus_speed if self.focus_speed > 0 else 0.0
        _motion(self.clock, duration, lambda f: setattr(self, "_focus", _lerp(old, z, f)),
                self)

    def focus_transition_duration_bound(self, start_z: float, target_z: float) -> float | None:
        """Bound the simulated constant-speed focus drive, in its shared Clock."""
        if (not all(math.isfinite(v) for v in (start_z, target_z, self.focus_speed))
                or self.focus_speed <= 0):
            return None
        return abs(target_z-start_z)/self.focus_speed

    def focus_axis_calibration(self, objective: int) -> dict[str, Any] | None:
        """Exact declared synthetic axis, not a laboratory calibration."""
        if isinstance(objective, bool) or not isinstance(objective, int) or objective <= 0:
            return None
        return {"source": "synthetic", "simulation_only": True,
                "calibration_id": "synthetic-microscope-focus-axis-v1",
                "objective": objective, "comparison_scope": "same_objective_only",
                "scale_mm_per_focus_mm": 1.0, "mapping_error_mm": 0.0,
                "readback_error_mm": 0.0,
                "lateral_focus_slope_norm_bound_mm_per_mm": self.lateral_focus_slope_norm_bound,
                "apex_lateral_drift_model": "componentwise-clipped-synthetic-stamp-v1",
                "apex_drift_clip_per_axis_mm": self.apex_drift_clip_per_axis_mm,
                "source_refraction_projection_norm_bound_mm_per_mm": self.source_refraction_projection_norm_bound,
                "apex_lateral_drift_speed_norm_bound_mm_s": self.apex_lateral_drift_speed_norm_bound,
                "apex_approach_drift_norm_bound_mm_per_mm": self.apex_approach_drift_norm_bound,
                "dome_curvature_radius_mm": self.declared_dome_curvature_radius_mm,
                "contact_slip_rate_bound_deg_s": self.contact_slip_rate_bound_deg_s,
                **({"stamp_discovery": deepcopy(self.stamp_discovery_profile),
                    "cap_support_radius_mm": self.stamp_discovery_profile["cap_radius_mm"]}
                   if self.stamp_discovery_profile is not None else {})}

    def get_focus(self) -> float:
        return self._focus

    def move_focal_plane(self, z: float) -> None:
        """v1compare's optical-model input, in world focal-plane millimetres."""
        self._check_online()
        self._focus = z-self.nominal_focus

    def set_objective(self, magnification: int) -> None:
        self._check_online()
        self._objective = magnification


class SimHeater(_Offline, Heater):
    """First-order thermal model: approaches setpoint with time constant tau.

    `cooling` (optional) returns True while a fan blows on the heater, which
    shortens the cooling time constant.
    """

    def __init__(self, device_id: str, clock: Clock, ambient: float = 22.0,
                 tau_s: float = 60.0, fan_tau_s: float | None = None,
                 cooling: Any = None) -> None:
        super().__init__(device_id)
        self.clock, self.ambient, self.tau = clock, ambient, tau_s
        self.fan_tau = fan_tau_s
        self.cooling = cooling
        self._target: float | None = None
        self._temp = ambient
        self._t = clock.now()
        #: sensor fault: reading frozen at this value (testbench fault injection)
        self.stuck_at: float | None = None

    def _update(self) -> None:
        now = self.clock.now()
        dt = max(0.0, now - self._t)
        self._t = now
        goal = self._target if self._target is not None else self.ambient
        tau = self.tau
        if goal < self._temp:  # passive cooling is slow; a fan helps
            tau = self.fan_tau if (self.fan_tau and self.cooling and self.cooling()) \
                else self.tau * 3
        k = 1.0 - math.exp(-dt / tau)
        self._temp += (goal - self._temp) * k

    def set_target(self, celsius: float) -> None:
        self._check_online()
        self._update()
        self._target = celsius

    def get_target(self) -> float | None:
        return self._target

    def read_temperature(self) -> float:
        self._check_online()
        self._update()
        return self.stuck_at if self.stuck_at is not None else self._temp

    def true_temperature(self) -> float:
        self._update()
        return self._temp

    def off(self) -> None:
        self._update()
        self._target = None


class SimSwitch(_Offline, Switch):
    """`on_change(really_on)` tells the world (the manipulator's stamp vacuum);
    `hold_in_safe_state` keeps a vacuum that holds something on in safe state."""

    def __init__(self, device_id: str, on_change: Any = None,
                 hold_in_safe_state: bool = False) -> None:
        super().__init__(device_id)
        self._on = False
        #: actuator fault: the switch reports what was commanded but really stays here
        self.stuck_at: bool | None = None
        self.on_change, self.hold_in_safe_state = on_change, hold_in_safe_state
        self.before_change: Any = None

    def set(self, on: bool) -> None:
        self._check_online()
        if self.before_change is not None:
            self.before_change()
        self._on = on
        if self.on_change is not None:
            self.on_change(self.really_on())

    def get(self) -> bool:
        return self._on

    def really_on(self) -> bool:
        return self._on if self.stuck_at is None else self.stuck_at

    def safe_state(self) -> None:
        if not self.hold_in_safe_state:
            if self.before_change is not None:
                self.before_change()
            self._on = False
            if self.on_change is not None:
                self.on_change(self.really_on())


class SimManipulator(_Offline, StampManipulator):
    """Stamp holder: x/y/z translation stage. Each axis lands within `sigma_mm`
    of the command; a move takes |distance| / speed on the clock (`speed`
    argument, else the default for that move; 0 = instant).

    Defaults: z moves use `speed_mm_s`. x/y moves use `travel_speed_mm_s` for
    rail travel and `fine_speed_mm_s` for moves up to `fine_below_mm` (an
    alignment step); either falls back to `speed_mm_s` when not given."""

    def __init__(self, device_id: str, world: SimWorld, home: dict[str, float],
                 sigma_mm: float = 0.0005, clock: Clock | None = None,
                 speed_mm_s: float = 0.0, travel_speed_mm_s: float | None = None,
                 fine_speed_mm_s: float | None = None, fine_below_mm: float = 0.0) -> None:
        super().__init__(device_id)
        self.world, self.sigma = world, sigma_mm
        self.clock, self.speed = clock, speed_mm_s
        self.travel_speed = speed_mm_s if travel_speed_mm_s is None else travel_speed_mm_s
        self.fine_speed = self.travel_speed if fine_speed_mm_s is None else fine_speed_mm_s
        self.fine_below = fine_below_mm
        for axis, v in home.items():
            setattr(world.manip, axis, float(v))

    def default_speed(self, axis: str, distance: float) -> float:
        """The speed an axis move of `distance` mm uses when none is commanded."""
        if axis == "z":
            return self.speed
        return self.fine_speed if abs(distance) <= self.fine_below else self.travel_speed

    def axis_motion_duration_bound(self, axis: str, start: float, target: float,
                                   speed: float | None = None) -> float | None:
        # The synthetic command clock uses requested travel, not the Gaussian
        # arrival. No finite arrival-position bound is implied by this duration.
        if axis in self.AXES and all(math.isfinite(x) for x in (start, target)):
            v = speed if speed is not None else self.default_speed(axis, target-start)
        else:
            v = speed if speed is not None else self.speed
        if (axis not in self.AXES or any(not math.isfinite(x) for x in (start, target, v))
                or v < 0):
            return None
        return abs(target-start)/v if v > 0 else 0.

    def move_axis_to(self, axis: str, value: float, speed: float | None = None) -> None:
        self._check_online()
        if axis not in self.AXES:
            raise ValueError(f"unknown axis {axis!r}")
        old = getattr(self.world.manip, axis)
        v = speed or self.default_speed(axis, value-old)
        target = value + self.world.rng.gauss(0, self.sigma)
        duration = abs(value-old)/v if v > 0 else 0.0
        def update(f: float) -> None:
            self.world.manip_move(axis, _lerp(old, target, f))
            if self.clock is not None:
                self.world.advance_transfer(self.clock.now())

        _motion(self.clock, duration, update, self)

    def get_positions(self) -> dict[str, float]:
        m = self.world.manip
        return {"x": m.x, "y": m.y, "z": m.z}

    def geometry_state(self) -> dict[str, Any]:
        origin = self.world.stamp_geometry.get("bench", {}).get("transfer_arm", {}).get("origin")
        return ({"source": "synthetic", "origin_world_xy_mm": [origin["x"], origin["y"]]}
                if origin is not None else {})

    def safe_state(self) -> None:
        # Stop where it is: lifting or swinging the stamp away is a process step.
        pass


class SimTape(_Offline, TapeMechanism):
    """Exfoliator over the exfoliation stage: white tape on two cylinders, the
    upper one on an x-z gantry (bench.yaml exfoliator). Needs the stage vacuum
    switch and heater to know whether the object is held down and how hot it is.
    Gantry moves take time on the clock (mm/min feed, like the v1 G-code)."""

    def __init__(self, device_id: str, world: SimWorld, vacuum: SimSwitch,
                 heater: SimHeater, clock: Clock | None = None,
                 geometry: dict[str, Any] | None = None) -> None:
        super().__init__(device_id)
        self.world, self.vacuum, self.heater, self.clock = world, vacuum, heater, clock
        g = geometry or {}
        self.geom = {"exfoliator": g} if "origin" in g else None
        self.apply_path = [tuple(p) for p in g.get("apply_path", [[0, -60]])]
        self.default_angle = float(g.get("peel_angle_deg", 30.0))
        self.travel = g.get("travel", {"x": [-1e9, 1e9], "z": [-1e9, 1e9]})
        self.reach = float(g.get("press_reach_mm", 2.0))
        self.run = float(g.get("peel_run_mm", 80.0))
        self.radius = float(g.get("roller", {}).get("diameter_mm", 12.0)) / 2
        self.width = float(g.get("tape_width_mm", 25.0))
        self.feed = 1000.0  # mm/min for apply moves (v1 F1000)
        world.tape.roll_left_mm = float(g.get("roll_length_mm", world.tape.roll_left_mm))

    def _rolls(self) -> dict[str, tuple[float, float]] | None:
        t = self.world.tape
        return exfoliator_rolls(self.geom, t.gantry_x, t.gantry_z) if self.geom else None

    def _gantry_to(self, x: float, z: float, feed: float) -> None:
        t = self.world.tape
        if self.clock is not None:
            self.clock.sleep(math.hypot(x - t.gantry_x, z - t.gantry_z) / feed * 60.0)
        t.gantry_x, t.gantry_z = x, z
        rolls, stage = self._rolls(), self.world.surfaces.get("exfoliation_stage")
        if rolls is not None and stage is not None:
            mx, mz = rolls["movable"]
            sx = stage.box["x"]
            if mz - self.radius < 0 and sx[0] - self.radius < mx < sx[1] + self.radius:
                # the roll came down onto the stage (a bad apply path)
                self.world.collisions.append({"what": "exfoliator_roll", "by": self.device_id,
                                              "gantry": [round(x, 2), round(z, 2)]})

    def _laid(self) -> bool:
        """Does the tape span the object on the stage in x, cross it in y, and
        hang within reach of the press pad all along?"""
        rolls, obj = self._rolls(), self.world.on_exfoliation_stage()
        if rolls is None:
            return True  # no geometry configured: the old "apply = on" model
        if obj is None:
            return False
        w, h = obj.size_mm
        y0 = float(self.geom["exfoliator"]["origin"]["y"])
        if abs(obj.pose.y - y0) >= h / 2 + self.width / 2:
            return False  # the tape runs past the object
        heights = [tape_height(rolls, obj.pose.x - w / 2), tape_height(rolls, obj.pose.x + w / 2)]
        if any(z is None for z in heights):
            return False  # the tape does not span the object
        return max(heights) <= self.reach  # type: ignore[type-var]

    def advance(self, length_mm: float) -> None:
        self._check_online()
        if self.world.tape.applied:
            raise NeedsHumanError("tape is still applied; peel before advancing")
        if self.world.tape.roll_left_mm < length_mm:
            raise NeedsHumanError(f"{self.device_id}: tape roll empty "
                                  f"({self.world.tape.roll_left_mm:.0f} mm left); put on a "
                                  f"new roll of white tape", device=self.device_id)
        if self.clock is not None:
            self.clock.sleep(length_mm / 20.0)  # reel motors, ~20 mm/s
        self.world.tape_advance(length_mm)

    def apply(self) -> None:
        self._check_online()
        for x, z in self.apply_path:  # right, down, left, down (Jialei, 2026-10-21)
            self._gantry_to(x, z, self.feed)
        y0 = float(self.geom["exfoliator"]["origin"]["y"]) if self.geom else None
        self.world.tape_apply(self.heater.true_temperature(), laid=self._laid(),
                              band=(y0 - self.width / 2, y0 + self.width / 2)
                              if y0 is not None else None)

    def peel(self, speed: float, angle_deg: float | None = None) -> None:
        """Lift the movable roll up and back to the left so the tape leaves the
        chip centre at `angle_deg` (tape vs stage, on the peeled side). The
        target is clipped to the cross positioner's travel; the angle the roll
        really reaches is what the world gets (and get_positions reports)."""
        self._check_online()
        asked = self.default_angle if angle_deg is None else float(angle_deg)
        angle, t = asked, self.world.tape
        if t.applied:
            if self.geom is not None:
                ex = self.geom["exfoliator"]
                obj = self.world.on_exfoliation_stage()
                cx = obj.pose.x if obj is not None else float(ex["origin"]["x"]) - 40.0
                r = math.radians(asked)
                gx = cx - self.run * math.cos(r) - float(ex["origin"]["x"])
                gz = self.run * math.sin(r) - float(ex.get("park_height_mm", 78.0))
                gx = min(max(gx, self.travel["x"][0]), self.travel["x"][1])
                gz = min(max(gz, self.travel["z"][0]), self.travel["z"][1])
                self._gantry_to(gx, gz, speed)
                mx, mz = self._rolls()["movable"]  # type: ignore[index]
                angle = round(math.degrees(math.atan2(mz, cx - mx)), 2)
            else:  # no geometry: lift along the asked direction
                r = math.radians(asked)
                self._gantry_to(t.gantry_x + 80 * math.cos(r), t.gantry_z + 80 * math.sin(r),
                                speed)
        self.world.tape_peel(speed, self.vacuum.really_on(), self.heater.true_temperature(),
                             angle, requested_deg=asked)
        self._gantry_to(0.0, 0.0, self.feed)

    def is_applied(self) -> bool:
        return self.world.tape.applied

    def remaining_mm(self) -> float:
        return self.world.tape.roll_left_mm

    def tape_remaining(self) -> float:
        """Same as remaining_mm (UI read)."""
        return self.world.tape.roll_left_mm

    def get_positions(self) -> dict[str, float]:
        """Gantry position, the tape line in the front view (x and height above
        the stage top of both roll axes), how much of the object is pressed on,
        and the last peel angle reached (UI read; floats only)."""
        t = self.world.tape
        out = {"gantry_x": t.gantry_x, "gantry_z": t.gantry_z,
               "peel_angle_deg": t.last_peel_angle if t.last_peel_angle is not None
               else self.default_angle,
               "pressed_frac": round(self.world.press_frac(), 3)}
        rolls = self._rolls()
        if rolls is not None:
            (x0, z0), (x1, z1) = rolls["movable"], rolls["fixed"]
            out.update(tape_x0=x0, tape_z0=z0, tape_x1=x1, tape_z1=z1)
        return out

    def note_temperature(self) -> None:
        """Called by the lab's tick so the tape remembers the hottest stage temperature."""
        self.world.tape_note_temperature(self.heater.true_temperature())

    def safe_state(self) -> None:
        # Leave the tape where it is: peeling is a process step, not a safe action.
        pass
