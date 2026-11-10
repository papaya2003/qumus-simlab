"""LabTwin: what the lab looks like now, from device readings, for the UI.

After every device command the SafetyGate asks the twin what changed
(`delta()`) and logs it as a `lab.frame` event, so the page (live or replayed)
moves the bench drawing to where the devices really are instead of guessing
from the commands. Devices are read, never commanded, and only through reads
that are not device calls (no temperature reads: they would count as calls on
a serial line).

    arms         {id: {x, y, z, rotation (world frame), vacuum, homed}}
    stages       {id: {x, y, z}} (stage coordinates, transfer_stage: 0 = arm drop-off)
    manipulator  {x, y, z, head: {x, y} (world), vacuum,
                  over_stage (the head is over the transfer stage),
                  gap_mm (holder z minus the transfer stage z: the stamp-low gap),
                  stamp (id of the stamp it holds, from the objects, or None)}
    exfoliator   {gantry_x, gantry_z, roller_x (world), applied, roll_left_mm,
                  tape {x0, z0, x1, z1} (front view: world x, height above the stage
                  top, movable then fixed roll), pressed_frac, peel_angle_deg,
                  vacuum (the stage vacuum switch, or None),
                  rolls {movable {x, z}, fixed {x, z}} (the same two axes as tape)}
    microscope   {objective, focus_plane ("chip" / "stamp" / None when the
                  microscope can tell, else absent)}
    objects      {id: {kind, x, y, rotation, held_by, on, slot, n_flakes, size_mm, ...}}

Objects come from a callable: the testbench passes its world (ground truth);
on real hardware it will be the sample ledger plus the last measured
positions. `delta()` returns only the parts that changed since the previous
call (objects one by one; `gone` lists removed objects), or None.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from qumus_simlab.devices.base import Arm, DeviceBundle, Microscope, MotionStage, StampManipulator
from qumus_simlab.flows.bench import arm_base, exfoliator_roller_x, manipulator_head, over_stage

ObjectSource = Callable[[], list[dict[str, Any]]]


def _r(v: Any) -> Any:
    """Round floats so sub-micron noise does not make a frame."""
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, dict):
        return {k: _r(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_r(x) for x in v]
    return v


class LabTwin:
    def __init__(self, devices: DeviceBundle, bench: dict[str, Any],
                 objects: ObjectSource | None = None) -> None:
        self.devices, self.bench, self.objects = devices, bench, objects
        self._last: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._preview_key: str | None = None
        self._preview_at: float = float("-inf")
        self._was_moving = False
        self._moving_duration = 0.0

    def state(self) -> dict[str, Any]:
        out: dict[str, Any] = {"arms": {}, "stages": {}}
        devs = self.devices.devices
        for did, d in devs.items():
            if isinstance(d, Arm):
                p = d.get_pose()
                bx, by = arm_base(self.bench, did)
                out["arms"][did] = {"x": p.x + bx, "y": p.y + by, "z": p.z,
                                    "rotation": p.rotation, "vacuum": d.vacuum_on(),
                                    "homed": d.is_homed()}
            elif isinstance(d, MotionStage):
                p = d.get_pose()
                out["stages"][did] = {"x": p.x, "y": p.y, "z": p.z, "rotation": p.rotation,
                                       "fine_xy_mm": list(d.get_fine_position())}
            elif isinstance(d, StampManipulator):
                pos = d.get_positions()
                hx, hy = manipulator_head(self.bench, pos)
                vac = devs.get(f"{did}_vacuum")
                over = "microscope" in self.bench and over_stage(self.bench, pos)
                stage = devs.get("transfer_stage")
                gap = (pos.get("z", 0.0) - stage.get_pose().z
                       if isinstance(stage, MotionStage) else None)
                out["manipulator"] = {**pos, "head": {"x": hx, "y": hy},
                                      "vacuum": bool(vac.get()) if vac is not None
                                      and hasattr(vac, "get") else None,
                                      "over_stage": over, "gap_mm": gap, "stamp": None}
                self._holder = did
            elif isinstance(d, Microscope):
                if did == "reactor_microscope":
                    out["reactor_microscope"] = {"focus_z": d.get_focus(),
                                                  "objective": d.get_objective()}
                    continue
                out["microscope"] = {"objective": d.get_objective(), "focus_z": d.get_focus()}
                plane = getattr(d, "focus_plane", None)
                if callable(plane):
                    out["microscope"]["focus_plane"] = plane()
            elif did == "exfoliator" and hasattr(d, "is_applied"):
                g = d.get_positions() if hasattr(d, "get_positions") else {}
                gx = float(g.get("gantry_x", 0.0))
                out["exfoliator"] = {"gantry_x": gx, "gantry_z": float(g.get("gantry_z", 0.0)),
                                     "roller_x": exfoliator_roller_x(self.bench, gx)
                                     if "exfoliator" in self.bench else None,
                                     "applied": bool(d.is_applied()),
                                     "roll_left_mm": d.remaining_mm()}
                if "tape_x0" in g:  # front view: the tape between the two roll axes
                    out["exfoliator"]["tape"] = {k[5:]: float(g[k]) for k in
                                                 ("tape_x0", "tape_z0", "tape_x1", "tape_z1")}
                    tp = out["exfoliator"]["tape"]
                    out["exfoliator"]["rolls"] = {"movable": {"x": tp["x0"], "z": tp["z0"]},
                                                  "fixed": {"x": tp["x1"], "z": tp["z1"]}}
                if "pressed_frac" in g:
                    out["exfoliator"]["pressed_frac"] = float(g["pressed_frac"])
                if "peel_angle_deg" in g:
                    out["exfoliator"]["peel_angle_deg"] = float(g["peel_angle_deg"])
                vac = devs.get("exfoliation_vacuum")
                out["exfoliator"]["vacuum"] = (bool(vac.get()) if vac is not None
                                               and hasattr(vac, "get") else None)
        if self.objects is not None:
            out["objects"] = {o["id"]: {k: v for k, v in o.items() if k != "id"}
                              for o in self.objects()}
            if "manipulator" in out:
                holder = getattr(self, "_holder", None)
                out["manipulator"]["stamp"] = next(
                    (k for k, o in out["objects"].items()
                     if o.get("kind") == "stamp" and o.get("held_by") == holder), None)
        display = _r(out)
        stage, microscope = devs.get("transfer_stage"), devs.get("microscope")
        # Preserve raw precision separately from the old millimetre drawing
        # fields. Display rounding is never a calibration or controller input.
        display["geometry"] = {"mechanics_version": 2,
            "source": "synthetic" if getattr(stage, "is_simulated", False) else "device_readback",
            "stage": stage.kinematics() if isinstance(stage, MotionStage) else None,
            "microscope": microscope.geometry_state() if isinstance(microscope, Microscope) else None}
        return display

    def delta(self) -> dict[str, Any] | None:
        with self._lock:
            now = self.state()
            last, out = self._last, {}
            for key, val in now.items():
                if key in ("arms", "stages", "objects"):
                    prev = last.get(key, {})
                    changed = {k: v for k, v in val.items() if prev.get(k) != v}
                    if changed:
                        out[key] = changed
                    if key == "objects" and (gone := [k for k in prev if k not in val]):
                        out["gone"] = gone
                elif last.get(key) != val:
                    out[key] = val
            self._last = now
            microscope = self.devices.devices.get("microscope")
            moving = any(getattr(d, "motion_active", False) for d in self.devices.devices.values())
            final_stroke = self._was_moving and not moving and self._moving_duration >= 0.5
            if moving:
                self._moving_duration = max(getattr(d, "last_motion_duration_s", 0.0)
                    for d in self.devices.devices.values() if getattr(d, "motion_active", False))
            self._was_moving = moving
            if getattr(microscope, "motion_preview_suppressed", 0):
                # The flake search shows only its photos (#203), no motion.
                self._preview_key = None
            elif final_stroke or any(k in out for k in ("geometry", "objects", "stages", "manipulator", "microscope")):
                snapshot = getattr(microscope, "preview_snapshot", None)
                state = getattr(microscope, "preview_state", None)
                signature = getattr(microscope, "preview_signature", None)
                key, detailed = (state() if callable(state) else
                                 (signature() if callable(signature) else None, False))
                now_ts = microscope.clock.now() if hasattr(microscope, "clock") else 0.0
                objective_changed = (last.get("microscope", {}).get("objective")
                                     != now.get("microscope", {}).get("objective"))
                # Near contact (rings, contact front, peel) every motion step
                # is a frame; elsewhere at most one per half simulated second.
                save = (key != self._preview_key and (detailed or now_ts-self._preview_at >= 0.5
                                                      or objective_changed or final_stroke))
                frame = snapshot() if save and callable(snapshot) else None
                if frame is not None:
                    self._preview_key, self._preview_at = key, now_ts
                    out["microscope_preview"] = {"image": frame.saved_path(),
                        "objective": frame.meta.get("objective"),
                        "focus_plane": frame.meta.get("focus_plane"),
                        "calibration": deepcopy(frame.meta.get("calibration")),
                        "focus_z": frame.meta.get("focus_z"), "timestamp": frame.timestamp,
                        "purpose": "motion_preview", "verification": False,
                        "source": "synthetic_scene", "immutable": True,
                        # a stamp over the stage: these frames make the stacking video
                        "stamp_in_view": frame.meta.get("stamp") is not None, "detailed": detailed,
                        "state_binding": "containing_lab.frame", "mechanics_version": 2}
            return out or None
