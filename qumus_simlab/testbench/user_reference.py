"""SIM stand-in for the photo a person takes of an electrode chip beforehand.

v1 aligns the stack to a pre-made electrode chip by matching a 20x photo of
the device site that the user supplied (assembly_device: bottom_electrode_image,
molecule_flake_refind_accurate). The testbench plays that person once, when the
bench is set up: the target chip is laid on the transfer stage at a hand-placed
angle, the device site is centred under the microscope at 20x and focused, and
the photo is kept as the chip's user reference. The chip and the microscope are
then put back exactly as they were; no device command or ledger event is made
and no shared random stream is drawn (#204).
"""
from __future__ import annotations

import math
import random
from typing import Any

from qumus_simlab.core.types import Pose

USER_REFERENCE_OBJECTIVE = 20
USER_PLACEMENT_SIGMA_DEG = 1.0  # same hand/arm placement spread as the SIM transfer stage


def photograph_site(lab: Any, chip_id: str, site_mm: tuple[float, float], *,
                    objective: int = USER_REFERENCE_OBJECTIVE) -> dict[str, Any] | None:
    """A focused photo centred on chip-local (bottom-left axes) `site_mm`."""
    world = lab.world
    mic = lab.devices.get("microscope")
    chip = world.objects.get(chip_id)
    if chip is None or mic is None or getattr(mic, "renderer", None) is None:
        return None
    if getattr(mic.renderer, "writer", None) is None:
        return None
    rotation = random.Random(f"user-reference:{world.seed}:{chip_id}").gauss(0., USER_PLACEMENT_SIGMA_DEG)
    sx, sy = site_mm[0]-chip.size_mm[0]/2, site_mm[1]-chip.size_mm[1]/2  # chip-centre axes
    cx, cy = mic.center
    r = math.radians(rotation)
    saved = (chip.pose, chip.on, chip.held_by, mic._objective, mic._focus)
    others = [(o, o.on) for o in world.objects.values()
              if o is not chip and o.on == "transfer_stage" and o.held_by is None]
    try:
        for o, _ in others:
            o.on = "_user_reference_aside"
        chip.pose = Pose(x=cx-(sx*math.cos(r)-sy*math.sin(r)), y=cy-(sx*math.sin(r)+sy*math.cos(r)),
                         rotation=rotation)
        chip.on, chip.held_by = "transfer_stage", None
        mic._objective = objective
        best = mic._view(fov_mm=mic.fov_mm()).get("best_focus_z")
        if best is not None:
            mic._focus = best
        frame = mic.capture()
        path = frame.saved_path()
    finally:
        chip.pose, chip.on, chip.held_by, mic._objective, mic._focus = saved
        for o, on in others:
            o.on = on
    if not path:
        return None
    return {"image": path, "objective": objective, "site_mm": [float(site_mm[0]), float(site_mm[1])],
            "frame": "chip_bottom_left_axes", "source": "simulated_user_photo",
            "calibration": frame.meta.get("calibration"),
            "scope": "SIM stand-in for the person's own 20x photo of the device site (v1 bottom_electrode_image)"}


def add_user_references(lab: Any, store: Any) -> None:
    """Give every seeded design target chip its user 20x site photo."""
    from qumus_simlab.records.device_design import design_anchor_mm

    for sample in store.list_samples(kind="chip"):
        props = sample.get("props") or {}
        if not props.get("device_design_target") or props.get("user_reference_20x"):
            continue
        reference = photograph_site(lab, sample["id"], design_anchor_mm(props["device_design_target"]))
        if reference is not None:
            store.upsert_sample(sample["id"], "chip", props={"user_reference_20x": reference})
