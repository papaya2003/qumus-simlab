"""Keep the simulated physical world across lab-process restarts.

Real hardware stays where it is when the job process dies; the simulator lives
inside that process, so it writes its physical state (objects, flakes, stamps,
tape, every device's plain state, simulated time) to a file before every device
command. A restarted lab process loads it, so recovery probes the same bench
the dead process left behind. A kill during a command loses only that command,
as if it never reached the hardware.

The file is written and read only by our own lab process (runs/, not in git).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.testbench.world import STAMP_HOLDER, SimFlake

_WORLD_FIELDS = ("objects", "faults", "tape", "_flake_seq", "stamps", "manip", "collisions",
                 "reactor_state", "seed", "_observation_rng", "_legacy_observation_rng", "material_morphology",
                 "transfer_dynamics", "_move_counts")
# The observer is a runtime callback even when a completed command restores None.
# Restore already refuses to overwrite it; do not serialize incidental instance keys.
_TRANSIENT_MOTION_FIELDS = frozenset({"motion_active", "last_motion_duration_s", "motion_observer"})
_MICROSCOPE_MECHANICS_FIELDS = ("best_focus_sigma", "focus_tilt_per_mm", "depth_of_focus_10x")


def mechanics_signature(lab: Any) -> str:
    """Bind stage/optics/contact continuation to its configured physical model.

    Detector noise and replay photo paths are not mechanical calibration. The
    entire stamp model is physical/optical behavior, including contact surfaces,
    refraction, dynamics and release; changing it cannot silently resume a bench.
    """
    microscope = lab.world.model.get("microscope", {})
    data = {"version": 2, "stage": lab.bench.get("transfer_stage"),
            "station": lab.bench.get("stations", {}).get("transfer_stage"),
            "microscope": lab.bench.get("microscope"), "holder": lab.bench.get("transfer_arm"),
            "microscope_model": {k: microscope.get(k) for k in _MICROSCOPE_MECHANICS_FIELDS},
            "stamp_model": lab.world.model.get("stamp", {})}
    if "geometry_profile" in lab.bench:
        from qumus_simlab.testbench.bench_geometry import geometry_binding

        data["bench_geometry"] = geometry_binding(lab.bench)
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _plain(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, float, str, BaseModel)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _plain(x) for k, x in v.items())
    return False


def lab_state(lab: Any, clock: Any) -> dict[str, Any]:
    w = lab.world
    w.advance_transfer(clock.now())
    with w.lock:
        world = {k: getattr(w, k) for k in _WORLD_FIELDS}
        world["rng"] = w.rng.getstate()
        # motorised surfaces carry what rests on them: keep where each one is
        world["surface_offsets"] = {n: (*sf.offset, sf.height, sf.rotation) for n, sf in w.surfaces.items()}
        devices = {dev_id: {k: v for k, v in vars(d).items()
                           if k not in _TRANSIENT_MOTION_FIELDS and _plain(v)}
                   for dev_id, d in lab.devices.devices.items()}
        # The overview camera's per-object constant error (and how many were
        # drawn) lives on the locator: keep it, or a resumed bench measures the
        # same chips with fresh errors.
        loc = getattr(lab, "locator", None)
        locator = ({"bias": dict(loc._bias), "qr_unreadable": dict(loc.qr_unreadable)}
                   if loc is not None and hasattr(loc, "_bias") else None)
        return {"world": world, "devices": devices, "locator": locator, "mechanics_version": 2,
                "mechanics_signature": mechanics_signature(lab),
                "clock": clock.now() if isinstance(clock, SimClock) else None}


def save_lab_state(lab: Any, clock: Any, path: str | Path) -> None:
    path = Path(path)
    data = pickle.dumps(lab_state(lab, clock), protocol=pickle.HIGHEST_PROTOCOL)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)  # atomic: a kill leaves the old or the new file, never half


def _contact_history_compatibility(world: dict[str, Any]) -> None:
    """Validate only the loaded copy, before overlaying any live state.

    Dataclass class defaults can hide a missing legacy instance field. A
    quiescent legacy bench can acquire known-empty bookkeeping in memory;
    an active/unknown contact cannot acquire an invented coverage history.
    """
    stamps = world.get("stamps", {})
    epoch = (world.get("transfer_dynamics") or {}).get("epoch")
    legacy = [stamp for stamp in stamps.values() if "contact_history" not in vars(stamp)]
    if legacy or (epoch is not None and "contact_history" not in epoch):
        quiescent = (epoch is None
            and not any(obj.held_by == STAMP_HOLDER for obj in world.get("objects", {}).values())
            and all(vars(stamp).get("stack") == [] and vars(stamp).get("front_radius_mm") == 0
                    and vars(stamp).get("max_depth") == 0
                    for stamp in stamps.values()))
        if not quiescent:
            raise PreconditionFailed("Saved active contact lacks whole-outline contact history; "
                                     "restore is unknown and requires read-only review")
    def valid(history, stack):
        def footprint(witness):
            return (isinstance(witness, dict) and isinstance(witness.get("apex_xy"), (tuple, list))
                    and len(witness["apex_xy"]) == 2
                    and all(type(x) in (int, float) and math.isfinite(x) for x in witness["apex_xy"])
                    and type(witness.get("radius_mm")) in (int, float)
                    and math.isfinite(witness["radius_mm"]) and witness["radius_mm"] > 0)
        if history is not None and (not isinstance(history, dict)
                or set(history) != {"version", "chip_id", "pickup", "release"}
                or type(history["version"]) is not int or history["version"] != 1
                or history.get("chip_id") not in world.get("objects", {})
                or world["objects"][history["chip_id"]].kind != "chip"
                or not isinstance(history.get("pickup"), dict)
                or any(not footprint(witness) or not isinstance(witness.get("flake"), SimFlake)
                       or witness["flake"].id != fid for fid, witness in history["pickup"].items())
                or (history.get("release") is not None
                    and (not footprint(history["release"]) or history["release"].get("stack") != stack))):
            raise PreconditionFailed("Saved contact history has an incompatible geometry binding")
    legacy_ids = {id(stamp) for stamp in legacy}
    for stamp in stamps.values():
        if id(stamp) not in legacy_ids:
            valid(stamp.contact_history, stamp.stack)
    if epoch is not None:
        valid(epoch.get("contact_history"), epoch.get("stack"))
    for stamp in legacy:
        stamp.contact_history = None


def load_lab_state(lab: Any, clock: Any, path: str | Path) -> bool:
    """Overlay a saved state on a freshly built lab; False if there is none."""
    path = Path(path)
    if not path.exists():
        return False
    state = pickle.loads(path.read_bytes())  # noqa: S301 - our own file, see module doc
    if state.get("mechanics_version") != 2 or state.get("mechanics_signature") != mechanics_signature(lab):
        raise PreconditionFailed(
            "Saved lab checkpoint uses incompatible stage/focus mechanics. It cannot be resumed with "
            "independent microscope focus and coarse/fine/rotation mechanics. The original checkpoint "
            "has not been changed. Open its saved History read-only, or explicitly select a fresh "
            "simulation workspace; do not reset or overwrite the old database.")
    # Validate the entire reactor boundary before changing even one live object,
    # device, random stream or clock. Never migrate an old bench by overlaying it
    # onto a different layout; that can separate physical objects from the ledger.
    saved_devices = {name for name in state["devices"] if name.startswith("reactor_") or name == "arm_reactor"}
    current_devices = {name for name in lab.devices.devices if name.startswith("reactor_") or name == "arm_reactor"}
    saved_reactor = state["world"].get("reactor_state") or {}
    current_reactor = lab.world.reactor_state
    mismatch = saved_devices != current_devices or bool(saved_reactor) != bool(current_reactor)
    if saved_reactor and current_reactor:
        plant = lab.devices.get("reactor_microscope").plant
        legacy_signature = plant.pre_metal_patch_layout_signature
        signature_matches = saved_reactor.get("layout_signature") == current_reactor.get("layout_signature")
        signature_matches |= (legacy_signature is not None
                              and saved_reactor.get("layout_signature") == legacy_signature)
        mismatch |= (saved_reactor.get("mechanics_version") != current_reactor.get("mechanics_version")
                     or not signature_matches)
    if mismatch:
        raise PreconditionFailed("Saved lab checkpoint is incompatible with this reactor configuration/layout. "
                                 "Use a fresh explicitly named simulation database or the compatible original "
                                 "configuration. The checkpoint and existing ledger have not been overlaid.")
    _contact_history_compatibility(state["world"])
    if saved_reactor:
        # Only after every boundary check passes: upgrade the loaded dictionary,
        # preserving all physical state and the original checkpoint file.
        saved_reactor["layout_signature"] = current_reactor["layout_signature"]
    w = lab.world
    with w.lock:
        for k in _WORLD_FIELDS:
            if k in state["world"]:  # a file from an older version may lack a field
                setattr(w, k, state["world"][k])
        w._transfer_projection_cache = None
        # A pre-upgrade checkpoint used one shared noise stream. Preserve its
        # continuation semantics explicitly, without rewriting the saved file.
        if "_observation_rng" not in state["world"]:
            w._legacy_observation_rng = True
        if "material_morphology" not in state["world"]:
            w.material_morphology = None
        w.rng.setstate(state["world"]["rng"])
        for n, off in state["world"].get("surface_offsets", {}).items():
            if n in w.surfaces:
                w.surfaces[n].offset = (off[0], off[1])
                w.surfaces[n].height = off[2] if len(off) > 2 else 0.0
                w.surfaces[n].rotation = off[3] if len(off) > 3 else 0.0
        for dev_id, attrs in state["devices"].items():
            if dev_id in lab.devices.devices:
                # Legacy transfer heaters had no fan capability. Preserve their
                # physical temperature/time/constants, while binding the newly
                # modelled fan to today's explicit synthetic fan-rate setting.
                if (dev_id == "stamp_heater" and "transfer_stage_fan" not in state["devices"]
                        and attrs.get("fan_tau") is None):
                    attrs = {k: v for k, v in attrs.items() if k not in ("fan_tau", "cooling")}
                # Renderers are configured dependencies, not physical state.
                # Old image-free snapshots could serialize renderer=None.
                vars(lab.devices.devices[dev_id]).update(
                    {k: v for k, v in attrs.items()
                     if k not in _TRANSIENT_MOTION_FIELDS
                     and k not in ("renderer", "stage", "motion_observer")})
        # A checkpoint can be taken by another job while an actuator is between
        # observer samples. Restore its actual pose, never a running controller:
        # no suspended Python stroke resumes in the new laboratory process.
        for device in lab.devices.devices.values():
            device.motion_active = False
            device.last_motion_duration_s = 0.0
        loc = getattr(lab, "locator", None)
        if state.get("locator") and loc is not None and hasattr(loc, "_bias"):
            loc._bias = dict(state["locator"]["bias"])
            loc.qr_unreadable = dict(state["locator"]["qr_unreadable"])
    if state.get("clock") is not None and isinstance(clock, SimClock):
        clock.advance(state["clock"] - clock.now())
    return True
