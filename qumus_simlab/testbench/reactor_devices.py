"""Opt-in virtual reactor devices and their shared, clock-driven plant.

The whole reactor assembly translates in XY. A separate insertion axis moves
the rod relative to the chamber. The microscope has independent Z focus.
Thermal and flow feedback lag their setpoints. Numerical geometry,
dynamics and interlocks are synthetic fixtures described in docs/reactor-model.md.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from qumus_simlab.core.clock import Clock
from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import Frame, GasFlowController, Heater, Microscope, MotionStage
from qumus_simlab.devices.sim.devices import _Offline
from qumus_simlab.testbench.reactor_model import GrowthModel, GrowthProfile
from qumus_simlab.testbench.render import _Writer
from qumus_simlab.testbench.world import SimObject, SimWorld


class ReactorPlant:
    """A shared plant whose physical state belongs to the persisted SimWorld.

    Growth integration uses fixed one-second intervals anchored to the most
    recent control change. Read-only observations commit whole intervals only.
    They never split an interval or redraw stochastic state. Actual control
    changes flush the remainder and begin a new thermal/flow segment.
    """

    def __init__(self, world: SimWorld, clock: Clock, config: dict[str, Any],
                 image_dir: Path | None = None, *, store: Any = None) -> None:
        self.world, self.clock, self.config = world, clock, config
        layout = {k: config[k] for k in ("geometry", "bench_layout")}
        from qumus_simlab.testbench.material_metal_model import MaterialMetalBackend

        self.material_metal_backend = MaterialMetalBackend(store, config.get("material_metal", {}), lambda: world.seed)
        self.pair_backend = None
        if config.get("experiment_kind") == "two_flake_reaction":
            from qumus_simlab.testbench.reactor_pair_backend import PairMaterialBackend

            self.pair_backend = PairMaterialBackend(store, config.get("reaction", {}), lambda: world.seed)
            layout["experiment_kind"] = config["experiment_kind"]
            layout["reaction"] = config.get("reaction", {})
        self.layout_signature = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
        # One optical fixture was added without changing reactor mechanics.
        # Retain the exact old hash for checkpoint admission, not a generally
        # geometry-insensitive signature. All other layout/reaction keys stay in it.
        self.pre_metal_patch_layout_signature = None
        if "metal_patch_outline_mm" in layout["geometry"]:
            legacy_layout = {**layout, "geometry": {k: v for k, v in layout["geometry"].items()
                                                   if k != "metal_patch_outline_mm"}}
            self.pre_metal_patch_layout_signature = hashlib.sha256(
                json.dumps(legacy_layout, sort_keys=True).encode()).hexdigest()
        self.profile = GrowthProfile.model_validate(config["growth"])
        self.writer = _Writer(image_dir) if image_dir is not None else None
        ambient = float(config["thermal"]["ambient_c"])
        if not world.reactor_state:
            world.reactor_state = {
                "mechanics_version": 2,
                "layout_signature": self.layout_signature,
                "epoch": clock.now(), "integrated_s": 0.0,
                "temperature_start_c": ambient, "flow_start_sccm": 0.0,
                "temperature_target_c": None, "flow_target_sccm": 0.0,
                "slot": {"x": 0.0, "y": 0.0, "z": 0.0, "rotation": 0.0},
                "insertion": {"x": 0.0, "y": 0.0, "z": 0.0, "rotation": 0.0},
                "focus_z": 0.0, "objective": 10, "models": {}, "pair_models": {}, "last_image": None,
            }

    @property
    def state(self) -> dict[str, Any]:
        # Loading a checkpoint replaces world state; never cache that dictionary.
        if self.world.reactor_state.get("mechanics_version") != 2:
            raise PreconditionFailed("Legacy reactor mechanics cannot resume as three motors; "
                                     "preserve the run for replay and start a new simulation")
        if self.world.reactor_state.get("layout_signature") != self.layout_signature:
            raise PreconditionFailed("Reactor layout differs from the saved checkpoint; "
                                     "preserve the run for replay and start a new simulation")
        return self.world.reactor_state

    def chip(self) -> SimObject | None:
        chips = [o for o in self.world.objects.values()
                 if o.on == "reactor" and o.held_by is None]
        if len(chips) > 1:
            raise PreconditionFailed("more than one chip is seated on the reactor carrier")
        return chips[0] if chips else None

    def _values(self, elapsed: float) -> tuple[float, float]:
        s, th = self.state, self.config["thermal"]
        heating = s["temperature_target_c"] is not None
        target = float(s["temperature_target_c"] if heating else th["ambient_c"])
        tau = float(th["heating_tau_s"] if heating else th["cooling_tau_s"])
        temperature = target + (s["temperature_start_c"] - target) * math.exp(-elapsed / tau)
        flow = s["flow_target_sccm"] + (s["flow_start_sccm"] - s["flow_target_sccm"]) * math.exp(
            -elapsed / float(th["flow_tau_s"]))
        return temperature, flow

    def readings(self) -> tuple[float, float]:
        with self.world.lock:
            return self._values(max(0.0, self.clock.now() - self.state["epoch"]))

    def sync(self, *, flush: bool = False) -> None:
        with self.world.lock:
            age = self.clock.now() - self.state["epoch"]
            if age < -1e-6:
                raise PreconditionFailed("reactor clock predates its saved physical state")
            end = max(0.0, age if flush else math.floor(age))
            done = self.state["integrated_s"]
            chip = self.chip()
            model = None
            material_metal = chip is not None and self.material_metal_backend.matches(chip)
            if chip is not None:
                if material_metal:
                    model = self.material_metal_backend.model(chip, self.state.setdefault("material_metal_models", {}))
                elif self.pair_backend:
                    model = self.pair_backend.model(chip, self.state.setdefault("pair_models", {}))
                else:
                    models = self.state["models"]
                    if chip.id not in models:
                        models[chip.id] = GrowthModel(self.world.seed, chip.id, self.profile)
                    model = models[chip.id]
            else:
                self.state["integrated_s"] = end
                return
            inside = self.state["insertion"]["y"] >= self.config["geometry"]["chamber_entry_y_mm"]
            while done < end - 1e-9:
                dt = min(1.0, end - done)
                if model is not None:
                    temperature, flow = self._values(done + dt / 2)
                    if material_metal:
                        self.material_metal_backend.advance(model, dt, temperature, flow, inside)
                    elif self.pair_backend:
                        active = self.pair_backend.active(temperature, flow, inside=inside)
                        model.advance(dt, active=active)
                    else:
                        model.advance(dt, temperature, flow, in_chamber=inside)
                done += dt
            if self.pair_backend and not material_metal:
                self.state["pair_models"][chip.id] = model.snapshot()
                self.pair_backend.sync_world(chip, model)
            self.state["integrated_s"] = done

    def control_boundary(self) -> None:
        """Preserve accumulated history before changing a physical control."""
        with self.world.lock:
            self.sync(flush=True)
            temperature, flow = self.readings()
            self.state.update(epoch=self.clock.now(), integrated_s=0.0,
                              temperature_start_c=temperature, flow_start_sccm=flow)

    def observation(self) -> tuple[dict[str, Any], list[dict[str, float]]]:
        with self.world.lock:
            self.sync()
            chip = self.chip()
            temperature, flow = self.readings()
            meta: dict[str, Any] = {
                "source": "synthetic_reactor", "timestamp": self.clock.now(),
                "temperature_c": temperature, "flow_sccm": flow,
                "focus_z": self.state["focus_z"], "objective": self.state["objective"],
                "frame_size_px": [640, 420],
                "chip_id": chip.id if chip else None,
            }
            if chip is None:
                if self.pair_backend:
                    meta["experiment_kind"] = "two_flake_reaction"
                return {**meta, "visible": False, "reason": "carrier_empty"}, []
            center = self.config["geometry"]["microscope_center"]
            fov = self.config["geometry"]["fov_mm_10x"]
            scale = 10 / self.state["objective"]
            fw, fh = fov[0] * scale, fov[1] * scale
            sx, sy = chip.size_mm
            cx, cy = center["x"] - chip.pose.x, center["y"] - chip.pose.y
            if self.material_metal_backend.matches(chip):
                theta = math.radians(chip.pose.rotation)
                cx, cy = cx * math.cos(theta) + cy * math.sin(theta), -cx * math.sin(theta) + cy * math.cos(theta)
                sharpness = math.exp(-0.5 * ((self.state["focus_z"] - chip.focus_offset) / 0.07) ** 2)
                model = self.material_metal_backend.model(chip, self.state.setdefault("material_metal_models", {}))
                observation, view = self.material_metal_backend.observation(model, center=(cx, cy), fov=(fw, fh),
                    in_focus=sharpness >= 0.6, rotation=chip.pose.rotation)
                return {**meta, "experiment_kind": "material_metal", "material_metal_observation": observation,
                    "material_view": view, "visible": observation["visible"], "reason": observation["reason"],
                    "sharpness": sharpness, "fov_mm": [fw, fh], "view_center_mm": [cx, cy]}, []
            if self.pair_backend:
                theta = math.radians(chip.pose.rotation)
                cx, cy = cx * math.cos(theta) + cy * math.sin(theta), -cx * math.sin(theta) + cy * math.cos(theta)
                sharpness = math.exp(-0.5 * ((self.state["focus_z"] - chip.focus_offset) / 0.07) ** 2)
                model = self.pair_backend.model(chip, self.state["pair_models"])
                pair, view = self.pair_backend.observation(model, center=(cx, cy), fov=(fw, fh),
                                                          in_focus=sharpness >= 0.6,
                                                          camera_rotation_deg=chip.pose.rotation)
                return {**meta, "experiment_kind": "two_flake_reaction", "pair_observation": pair,
                        "material_view": view, "visible": pair["visible"], "reason": view["reason"],
                        "sharpness": sharpness, "fov_mm": [fw, fh], "view_center_mm": [cx, cy]}, []
            roi = ((cx - fw / 2) / sx, (cx + fw / 2) / sx,
                   (cy - fh / 2) / sy, (cy + fh / 2) / sy)
            meta.update(fov_mm=[fw, fh], view_center_mm=[cx, cy])
            model = self.state["models"][chip.id]
            measured = model.measure(roi)
            sharpness = math.exp(-0.5 * ((self.state["focus_z"] - chip.focus_offset) / 0.07) ** 2)
            meta.update(measured, sharpness=sharpness)
            if not measured["visible"]:
                meta["reason"] = "chip_outside_field"
            elif sharpness < 0.6:
                meta.update(coverage=None, uniformity=None, defect_fraction=None,
                            quality_score=None, phase="out_of_focus", reason="out_of_focus")
            pixels = [{"x": (c.x * sx - cx) / fw + 0.5,
                       "y": 0.5 - (c.y * sy - cy) / fh,
                       "angle": i * 2.39996,
                       "coverage": model.coverage(c), "defects": model.defect_fraction(c)}
                      for i, c in enumerate(model.cells)
                      if roi[0] <= c.x <= roi[1] and roi[2] <= c.y <= roi[3]]
            return meta, pixels

    def frame(self) -> Frame:
        meta, cells = self.observation()
        frame = Frame("reactor_microscope", self.clock.now(), width=640, height=420,
                      meta={"reactor": meta, "objective": meta["objective"],
                            "focus": meta["focus_z"]})
        if self.writer is not None:
            def save() -> str:
                assert self.writer is not None
                if meta.get("experiment_kind") == "material_metal":
                    from qumus_simlab.testbench.material_metal_model import render_material_metal

                    png = render_material_metal(meta["material_view"])
                elif self.pair_backend:
                    empty = {"status": "carrier_empty", "inputs": [], "render_product": None,
                             "visible": False, "reason": "carrier_empty", "conversion_of_overlap": 0,
                             "view_center_mm": [0, 0], "fov_mm": self.config["geometry"]["fov_mm_10x"]}
                    png = self.pair_backend.render(meta.get("material_view") or empty)
                else:
                    png = _render(meta, cells)
                filename = self.writer.write("reactor_microscope", png)
                self.state["last_image"] = {"path": filename, "timestamp": frame.timestamp,
                                            "meta": frame.meta}
                return filename
            frame.save = save
        return frame


def _render(meta: dict[str, Any], cells: list[dict[str, float]]) -> bytes:
    im = Image.new("RGB", (640, 420), "#78629a" if meta["visible"] else "#34343f")
    draw = ImageDraw.Draw(im)
    fw = (meta.get("fov_mm") or [2.6])[0]
    for c in cells:
        if c["coverage"] <= 0:
            continue
        x, y = c["x"] * 640, c["y"] * 420
        radius = 0.08 / fw * 640 * math.sqrt(c["coverage"])
        angle = c["angle"]
        points = [(x + radius * math.cos(angle + k * math.tau / 3),
                   y + radius * math.sin(angle + k * math.tau / 3)) for k in range(3)]
        draw.polygon(points, fill="#bfbd79" if c["defects"] < 0.25 else "#65513c",
                     outline="#dddba2")
    blur = max(0.0, 1.0 - float(meta.get("sharpness", 0))) * 6
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=16)
    draw.text((12, 10), f"{meta['objective']}X  Reactor microscope", font=font, fill="white")
    draw.text((12, 390), "SIMULATED / SYNTHETIC GROWTH", font=font, fill="white")
    bar = 0.2 / fw * 640
    draw.line((616 - bar, 386, 616, 386), fill="white", width=4)
    draw.text((616, 365), "200 um", anchor="ra", font=font, fill="white")
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


class SimReactorStage(_Offline, MotionStage):
    def __init__(self, plant: ReactorPlant, *, insertion: bool = False) -> None:
        super().__init__("reactor_insert" if insertion else "reactor_slot")
        self.plant = plant
        self.insertion = insertion
        self.key = "insertion" if insertion else "slot"

    def move_to(self, pose: Pose) -> None:
        self._check_online()
        if pose.z != 0 or pose.rotation != 0 or (self.insertion and pose.x != 0):
            raise PreconditionFailed("reactor assembly has only XY; insertion has only Y")
        with self.plant.world.lock:
            before = self.get_pose()
            self.plant.control_boundary()
            speed_key = "insertion_speed_mm_s" if self.insertion else "assembly_speed_mm_s"
            speed = float(self.plant.config["motion"][speed_key])
            # Record elapsed laboratory time so replay can interpolate command -> frame.
            # Segment integration preserves exposure history along the actual path.
            distance = math.hypot(pose.x - before.x, pose.y - before.y)
            count = max(1, math.ceil(distance))
            for index in range(1, count + 1):
                self.plant.clock.sleep(distance / speed / count)
                self.plant.control_boundary()
                fraction = index / count
                current = Pose(x=before.x + (pose.x - before.x) * fraction,
                               y=before.y + (pose.y - before.y) * fraction)
                self.plant.state[self.key] = current.model_dump()
                stage, insert = self.plant.state["slot"], self.plant.state["insertion"]
                self.plant.world.move_surface("reactor", stage["x"], stage["y"] + insert["y"], 0)

    def get_pose(self) -> Pose:
        self._check_online()
        return Pose(**self.plant.state[self.key])


class SimReactorHeater(_Offline, Heater):
    def __init__(self, plant: ReactorPlant) -> None:
        super().__init__("reactor_heater")
        self.plant = plant

    def set_target(self, celsius: float) -> None:
        self._check_online()
        with self.plant.world.lock:
            self.plant.control_boundary()
            self.plant.state["temperature_target_c"] = celsius

    def get_target(self) -> float | None:
        return self.plant.state["temperature_target_c"]

    def read_temperature(self) -> float:
        self._check_online()
        return self.plant.readings()[0]

    def off(self) -> None:
        self._check_online()
        with self.plant.world.lock:
            self.plant.control_boundary()
            self.plant.state["temperature_target_c"] = None


class SimGasFlow(_Offline, GasFlowController):
    def __init__(self, plant: ReactorPlant) -> None:
        super().__init__("reactor_flow")
        self.plant = plant

    def set_flow(self, sccm: float) -> None:
        self._check_online()
        with self.plant.world.lock:
            self.plant.control_boundary()
            self.plant.state["flow_target_sccm"] = sccm

    def get_target(self) -> float:
        return self.plant.state["flow_target_sccm"]

    def read_flow(self) -> float:
        self._check_online()
        return self.plant.readings()[1]

    def safe_state(self) -> None:
        # Chosen simulation shutdown; the physical procedure remains an open question.
        self.set_flow(0.0)


class SimReactorMicroscope(_Offline, Microscope):
    def __init__(self, plant: ReactorPlant) -> None:
        super().__init__("reactor_microscope")
        self.plant = plant

    def set_focus(self, z: float) -> None:
        self._check_online()
        self.plant.state["focus_z"] = z

    def get_focus(self) -> float:
        return self.plant.state["focus_z"]

    def set_objective(self, magnification: int) -> None:
        self._check_online()
        self.plant.state["objective"] = magnification

    def get_objective(self) -> int:
        return self.plant.state["objective"]

    def capture(self) -> Frame:
        self._check_online()
        return self.plant.frame()

    def preview(self) -> Frame | None:
        if self.offline or not (last := self.plant.state["last_image"]):
            return None
        return Frame(self.device_id, last["timestamp"], path=last["path"], meta=last["meta"])
