"""Persistent qualitative material/metal interface model and synthetic optics.

Controls advance exposure; camera captures only project the current state. Both
input flakes retain their identity and geometry. hBN never reacts or disappears.
No chemical product composition, calibrated conversion or yield is predicted.
"""
from __future__ import annotations

import io
import math
import random
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont

from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.core.polygons import transform
from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation
from qumus_simlab.testbench.material_adapter import sim_outline
from qumus_simlab.testbench.material_flakes import project_outline, stable_key
from qumus_simlab.testbench.material_view import camera_outline, pair_sensor_rect


class MaterialMetalBackend:
    def __init__(self, store, config, seed):
        self.store, self.config, self.seed = store, MaterialMetalSimulation.model_validate(config), seed

    def matches(self, chip):
        row = self.store.get_sample(chip.id) if self.store else None
        return bool(row and row["props"].get("material_metal_preparation", {}).get("scope") == "device_simulation")

    def model(self, chip, states):
        receipt = self.store.get_sample(chip.id)["props"]["material_metal_preparation"]
        if receipt["model_assumptions"] != self.config.model_dump():
            raise PreconditionFailed("Material-metal assumptions changed after preparation; preserve the recorded profile")
        ids = receipt["top_first_flake_ids"]
        rows = [self.store.get_sample(fid) for fid in ids]
        sim_ids = [s["props"].get("sim_id") for s in rows]
        physical = {f.id: f for f in chip.flakes}
        if len(chip.flakes) != 2 or [f.id for f in chip.flakes] != sim_ids:
            raise PreconditionFailed("Material-metal physical layers differ from the prepared inputs")
        inputs = []
        for row, sid in zip(rows, sim_ids, strict=True):
            flake = physical[sid]
            outline = transform(sim_outline(flake), xy=(flake.x_mm, flake.y_mm))
            from qumus_simlab.testbench.material_flakes import Placement

            placed = Placement.model_validate(receipt["placements"][row["id"]])
            if row["location"] != chip.id or row["status"] != "stacked" or flake.material != row["props"]["material"] or len(outline) != len(placed.outline_mm) or any(math.dist(a, b) > 1e-8 for a, b in zip(outline, placed.outline_mm, strict=True)):
                raise PreconditionFailed("Material-metal geometry, identity or ledger position changed")
            inputs.append({"id": row["id"], "sim_id": sid, "material": flake.material, "outline_mm": outline,
                           "role": "nonreactive_protection" if row["id"] == ids[0] else "reactive_interface"})
        if chip.id not in states:
            succeeds = random.Random(stable_key(self.seed(), chip.id, sim_ids, "material-metal" )).random() < self.config.success_probability
            states[chip.id] = {"preparation_job_id": receipt["preparation_job_id"], "active_seconds": 0.0,
                "qualitative_progress": 0.0, "trajectory": "progressing" if succeeds else "stalled",
                "model_assumptions": self.config.model_dump()}
        state = states[chip.id]
        if state["preparation_job_id"] != receipt["preparation_job_id"] or state["model_assumptions"] != receipt["model_assumptions"]:
            raise PreconditionFailed("Saved material-metal exposure belongs to another preparation/profile")
        return state, receipt, inputs

    def advance(self, model, dt, temperature, flow, inside):
        state, _, _ = model
        if inside and temperature >= self.config.activation_temperature_c and flow >= self.config.activation_flow_sccm:
            state["active_seconds"] += dt
            limit = 1.0 if state["trajectory"] == "progressing" else self.config.unsuccessful_max_progress
            state["qualitative_progress"] = limit * (1 - math.exp(-state["active_seconds"] / self.config.characteristic_time_s))

    def observation(self, model, *, center, fov, in_focus, rotation):
        state, prep, inputs = model
        projected = []
        for layer in inputs:
            camera = camera_outline(layer["outline_mm"], center, rotation)
            projected.append({**layer, **project_outline(camera, center_mm=(0, 0), fov_mm=fov)})
        visible = bool(in_focus and self.config.observation_available and all(f["visible"] for f in projected))
        full = visible and all(f["fully_visible"] for f in projected)
        reason = None if visible else "observation_unavailable" if not self.config.observation_available else "out_of_focus" if not in_focus else "layers_outside_field"
        obs = {k: deepcopy(prep[k]) for k in ("protocol_id", "preparation_job_id", "requirements_event_id", "layer_order", "top_first_flake_ids", "model_assumptions")}
        obs.update(scope="device_simulation", chip_id=prep["chip_id"], visible=visible, fully_visible=full,
            status=state["trajectory"], qualitative_progress=state["qualitative_progress"] if full else None,
            active_seconds=state["active_seconds"] if full else None, product_identity=None, chemistry_verified=False,
            reason=reason, observable="qualitative_interface_progress", protective_layer_preserved=True)
        view = {"inputs": projected, "view_center_mm": center, "fov_mm": fov, "camera_rotation_deg": rotation,
                "in_focus": in_focus and self.config.observation_available, "observation": obs}
        return obs, view

    def restore_world(self, world):
        # Validate preserved geometry. Never reconstruct missing physical layers.
        with world.lock:
            for chip in world.objects.values():
                if self.matches(chip):
                    self.model(chip, world.reactor_state.setdefault("material_metal_models", {}))


def render_material_metal(view):
    obs, fov, size = view["observation"], view["fov_mm"], (640, 420)
    x, y, w, h = pair_sensor_rect(fov, size)
    image = Image.new("RGB", size, "#182235")
    sensor = Image.new("RGB", (w, h), "#585765")
    draw = ImageDraw.Draw(sensor)
    if view["in_focus"]:
        for layer in reversed(view["inputs"]):
            camera = camera_outline(layer["outline_mm"], view["view_center_mm"], view["camera_rotation_deg"])
            points = project_outline(camera, center_mm=(0, 0), fov_mm=fov, image_size=(w, h))["outline_px"]
            if layer["role"] == "nonreactive_protection":
                draw.polygon(points, outline="#b9dbff", width=2)
            else:
                progress = obs.get("qualitative_progress") or 0
                draw.polygon(points, fill=(int(70+120*progress), int(95+65*progress), 110), outline="#dae5eb")
    image.paste(sensor, (x, y))
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=14)
    draw.text((12, 6), "SIMULATION | Material-metal interface (uncalibrated)", fill="white", font=font)
    p = obs.get("qualitative_progress")
    caption = f"{obs['status']} | qualitative progress {p:.0%}" if p is not None else f"Not measurable: {obs.get('reason') or 'clipped layers'}"
    draw.text((12, 26), caption, fill="#dbe7f3", font=font)
    draw.text((12, 46), "hBN preserved | no chemical product identity inferred", fill="#dbe7f3", font=font)
    draw.text((12, 380), f"Field {fov[0]*1000:g} x {fov[1]*1000:g} um", fill="white", font=font)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
