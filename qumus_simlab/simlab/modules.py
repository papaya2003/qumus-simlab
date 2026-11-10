"""The three sim-lab modules, `exfoliation`, `stacking` and `reactor`, over one simulated bench.

Both modules share one `LabSide` (app.build_lab_side): one simulated world, one
run store, one SafetyGate and one job queue, exactly like one lab table with two
machines on it. Each module only offers its own devices and its own work:

    exfoliation   arm_left, exfoliator (tape), exfoliation heater / fan / vacuum
    stacking      arm_right, transfer stage (+ vacuum, fan), microscope,
                  stamp heater, stamp manipulator (transfer_arm + vacuum)
    reactor       arm_reactor, reactor slot + insert stages, reactor heater,
                  gas flow, reactor microscope (the vdW reaction chamber)

The simulated bench has one overview camera over the whole table; both modules
list it (read only) where the real lab has one desk camera per module.

Actions (contract spec 2026-11-08): `read_all`, `home_query`, `capture`,
`go_to_point`, `heat`, `heat_off`, local-only `jog` and `set_switch`, plus the
module's whole flows as job actions ([Q] sim lab only: on real hardware these
become qumus-lab level-4 actions after L3). Every device command goes through
the existing SafetyGate. A device that does not answer during a motion command
(serial timeout) is an unknown outcome: the module locks.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

from qumus_simlab.core.errors import QumusError, SafetyViolation, TransientError
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import (
    Arm,
    Camera,
    GasFlowController,
    Heater,
    Microscope,
    MotionStage,
    StampManipulator,
    Switch,
    TapeMechanism,
)
from qumus_simlab.flows.bench import arm_base, positions, regions
from qumus_simlab.simlab.service import Action, CommandContext, ContractService, UnknownOutcome

PORTS = {"exfoliation": 8781, "stacking": 8782, "reactor": 8783}

MODULE_DEVICES: dict[str, tuple[str, ...]] = {
    "exfoliation": ("arm_left", "exfoliator", "exfoliation_heater", "exfoliation_fan",
                    "exfoliation_vacuum", "overview_camera"),
    "stacking": ("arm_right", "transfer_stage", "transfer_stage_vacuum", "transfer_stage_fan",
                 "microscope", "stamp_heater", "transfer_arm", "transfer_arm_vacuum",
                 "overview_camera"),
    "reactor": ("arm_reactor", "reactor_slot", "reactor_insert", "reactor_heater", "reactor_flow",
                "reactor_microscope", "overview_camera"),
}
MODULE_ARM = {"exfoliation": "arm_left", "stacking": "arm_right", "reactor": "arm_reactor"}

#: which module runs which flow (agent tools keep submitting flows by name)
FLOW_MODULE: dict[str, str] = {
    "exfoliate_chips": "exfoliation",
    "place_chip_on_transfer_stage": "stacking",
    "search_chip": "stacking",
    "revisit_chip": "stacking",
    "stack_device": "stacking",
    "cool_experiment_stages": "stacking",
    "inspect_bench": "stacking",
    # stamp-based preparation for the reactor runs on the stacking machine
    "prepare_reactor_pair": "stacking",
    "prepare_material_metal": "stacking",
    "assess_reactor_inputs": "reactor",
    "load_reactor_chip": "reactor",
    "position_reactor_slot": "reactor",
    "reactor_growth": "reactor",
    "reactor_reaction": "reactor",
    "reactor_material_metal": "reactor",
    "verify_reactor_state": "reactor",
    "hold_reactor_observation": "reactor",
    "observe_reactor": "reactor",
    "inspect_reactor_product": "reactor",
    "shutdown_reactor": "reactor",
    "unload_reactor_chip": "reactor",
}

#: [Q] local-only jog caps (spec: 0.5 mm arm/stage, 0.1 mm stamp)
JOG_CAP_MM = {"arm": 0.5, "stage": 0.5, "manipulator": 0.1}
SWITCH_AUTO_OFF_MAX_S = 10.0
HEAT_TIMEOUT_MAX_S = 3600.0
ACTIVE = ("queued", "running")


def _kind(dev: Any) -> str:
    for cls, name in ((Arm, "arm"), (MotionStage, "stage"), (StampManipulator, "manipulator"),
                      (Microscope, "microscope"), (Camera, "camera"), (Heater, "heater"),
                      (TapeMechanism, "tape"), (Switch, "switch"), (GasFlowController, "flow")):
        if isinstance(dev, cls):
            return name
    return "device"


FIELDS: dict[str, dict[str, str]] = {
    "arm": {"x": "mm", "y": "mm", "z": "mm", "rotation": "deg", "vacuum": "bool",
            "homed": "bool"},
    "stage": {"x": "mm", "y": "mm", "z": "mm", "rotation": "deg"},
    "manipulator": {"x": "mm", "y": "mm", "z": "mm"},
    "microscope": {"objective": "x", "focus": "mm"},
    "camera": {"online": "bool"},
    "heater": {"temperature": "C", "target": "C"},
    "switch": {"on": "bool"},
    "tape": {"applied": "bool", "remaining": "mm"},
    "flow": {"flow": "sccm", "target": "sccm"},
}


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


class SimModule:
    """Builds one ContractService over the shared lab side."""

    def __init__(self, module: str, side: Any, *, motion_delay_s: float = 0.0,
                 wall: Callable[[], float] | None = None, records: Any = None) -> None:
        self.module, self.side = module, side
        if records is None:
            from qumus_simlab.simlab.records import LabRecords

            records = LabRecords(side.store, getattr(side, "image_dir", None),
                                 getattr(side, "image_roots", ()))
        self.records = records
        self.lab, self.gate, self.jobs, self.store = side.lab, side.gate, side.jobs, side.store
        self.bench = self.lab.bench
        self.device_ids = tuple(d for d in MODULE_DEVICES[module] if d in self.lab.devices.devices)
        self.arm_id = MODULE_ARM[module]
        self.motion_delay_s = motion_delay_s
        self.flows = tuple(f for f, m in FLOW_MODULE.items() if m == module)
        self._timers: dict[str, threading.Timer] = {}
        kwargs: dict[str, Any] = {} if wall is None else {"wall": wall}
        # a busy simulation can delay the heartbeat reply past the 3 s contract
        # lease; QUMUS_SIMLAB_LEASE_S lengthens it for the SIM lab only
        try:
            kwargs["lease_s"] = max(3.0, float(os.environ.get("QUMUS_SIMLAB_LEASE_S", "") or 3.0))
        except ValueError:
            pass
        self.service = ContractService(
            f"qumus-sim-{module}", module, self._actions(),
            devices={d: {"kind": _kind(self.lab.devices.get(d)),
                         "fields": FIELDS.get(_kind(self.lab.devices.get(d)), {})}
                     for d in self.device_ids},
            read_devices=self.read_devices, on_stop=self.on_stop,
            active_jobs=self.active_jobs, demo=True,
            # the testbench applies device faults (serial timeouts) only to lab
            # work, which it recognises by the "job-" thread name, not to UI reads
            worker_prefix="job-contract", **kwargs)

    # -- hooks ----------------------------------------------------------------
    def _dev(self, device_id: str) -> Any:
        if device_id not in self.device_ids:
            raise ValueError(f"{device_id!r} is not a device of the {self.module} module; "
                             f"devices: {', '.join(self.device_ids)}")
        return self.lab.devices.get(device_id)

    def read_device(self, device_id: str) -> dict[str, Any]:
        dev = self.lab.devices.get(device_id)
        kind = _kind(dev)
        out: dict[str, Any] = {"kind": kind, "offline": bool(getattr(dev, "offline", False))}
        try:
            if kind == "arm":
                p = dev.get_pose()
                out.update(x=p.x, y=p.y, z=p.z, rotation=p.rotation, vacuum=dev.vacuum_on(),
                           homed=dev.is_homed())
            elif kind == "stage":
                p = dev.get_pose()
                out.update(x=p.x, y=p.y, z=p.z, rotation=p.rotation)
            elif kind == "manipulator":
                out.update(dev.get_positions())
            elif kind == "microscope":
                out.update(objective=dev.get_objective(), focus=dev.get_focus())
            elif kind == "camera":
                out.update(online=not out["offline"])
            elif kind == "heater":
                out.update(temperature=None if out["offline"] else dev.read_temperature(),
                           target=dev.get_target())
            elif kind == "switch":
                out.update(on=dev.get())
            elif kind == "tape":
                out.update(applied=dev.is_applied(), remaining=dev.remaining_mm())
            elif kind == "flow":
                out.update(flow=None if out["offline"] else dev.read_flow(),
                           target=dev.get_target())
        except Exception as e:  # noqa: BLE001 - one bad device never hides the others
            out["error"] = str(e)
        return out

    def read_devices(self) -> dict[str, dict[str, Any]]:
        return {d: self.read_device(d) for d in self.device_ids}

    def active_jobs(self) -> list[str]:
        return [r["id"] for r in self.store.list_runs()
                if r["flow"] in self.flows and r["status"] in ACTIVE]

    def on_stop(self) -> bool:
        """Pause this module's jobs at their next safe point (resumable, like
        the UI's Stop) and switch off this module's heaters no job holds."""
        for rid in self.active_jobs():
            self.jobs.pause(rid)
        for t in self._timers.values():
            t.cancel()
        self._timers.clear()
        locked = self.store.locks()
        ok = True
        for d in self.device_ids:
            dev = self.lab.devices.get(d)
            if isinstance(dev, Heater) and d not in locked and dev.get_target() is not None:
                try:
                    self.gate.heater(d).off()
                except Exception:  # noqa: BLE001 - reported as an unconfirmed stop
                    ok = False
        return ok

    # -- actions --------------------------------------------------------------
    def _actions(self) -> list[Action]:
        devs = list(self.device_ids)
        heaters = [d for d in devs if _kind(self.lab.devices.get(d)) == "heater"]
        switches = [d for d in devs if _kind(self.lab.devices.get(d)) == "switch"]
        cameras = [d for d in devs if _kind(self.lab.devices.get(d)) in ("camera", "microscope")]
        jog_devs = [d for d in devs
                    if _kind(self.lab.devices.get(d)) in ("arm", "stage", "manipulator")]
        acts = [
            Action("read_all", self._read_all, any_client=True,
                   description="Read every device of this module (no motion)."),
            Action("home_query", self._home_query, any_client=True,
                   description="Ask whether arms know their position (query only, never homes)."),
            Action("capture", self._capture, exclusive=True,
                   params=_schema({"camera": {"type": "string", "enum": cameras}}, ["camera"]),
                   validate=self._need({"camera": str}),
                   description="Take one picture; returns the image id/path."),
            Action("go_to_point", self._go_to_point, motion=True, sync=False,
                   params=_schema({"point": {"type": "string", "enum": self.points()}}, ["point"]),
                   validate=self._validate_point,
                   description=f"Move {self.arm_id} at its safe height to a named point "
                   "(config, never raw coordinates)."),
            Action("heat", self._heat, exclusive=True,
                   params=_schema({"device": {"type": "string", "enum": heaters},
                                   "target_c": {"type": "number"},
                                   "timeout_s": {"type": "number", "exclusiveMinimum": 0,
                                                 "maximum": HEAT_TIMEOUT_MAX_S}},
                                  ["device", "target_c", "timeout_s"]),
                   validate=self._validate_heat,
                   description="Set a heater; it switches itself off after timeout_s."),
            Action("heat_off", self._heat_off, any_client=True,
                   params=_schema({"device": {"type": "string", "enum": heaters}}, ["device"]),
                   validate=self._need({"device": str}),
                   description="Switch a heater off (any client)."),
            Action("set_switch", self._set_switch, local_only=True, exclusive=True,
                   params=_schema({"device": {"type": "string", "enum": switches},
                                   "on": {"type": "boolean"},
                                   "auto_off_s": {"type": "number", "exclusiveMinimum": 0,
                                                  "maximum": SWITCH_AUTO_OFF_MAX_S}},
                                  ["device", "on"]),
                   validate=self._validate_switch,
                   description="Debug page / probe scripts only: raw switch, auto-off <= 10 s."),
            Action("jog", self._jog, motion=True, local_only=True, sync=False,
                   params=_schema({"device": {"type": "string", "enum": jog_devs},
                                   "axis": {"type": "string", "enum": ["x", "y", "z"]},
                                   "mm": {"type": "number"}}, ["device", "axis", "mm"]),
                   validate=self._validate_jog,
                   description="Debug page / probe scripts only: small relative move "
                   "(0.5 mm arm/stage, 0.1 mm stamp)."),
        ]
        acts += [self._flow_action(f) for f in self.flows]
        acts += self._job_actions()
        acts += self._record_actions()
        return acts

    # records [Q] ---------------------------------------------------------------
    def _record_actions(self) -> list[Action]:
        def put(ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
            return {"put": self.records.put(p)}

        def status(ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
            run_id = p.get("run_id")
            run = self.store.get_run(run_id) if isinstance(run_id, str) else None
            if run is None:
                raise KeyError(f"unknown job {run_id}")
            return self.records.scrub({
                "run_id": run_id, "flow": run["flow"], "status": run["status"],
                "done": run["status"] in ("succeeded", "failed", "cancelled"),
                "result": run["result"], "error": run["error"],
                "workflow": (run.get("meta") or {}).get("workflow"),
                "steps": self.store.get_steps(run_id),
                "module": FLOW_MODULE.get(run["flow"])})

        return [
            Action("put_records", put, any_client=True, params={
                "type": "object", "properties": {
                    "project": {}, "samples": {"type": "array"},
                    "removed_samples": {"type": "array"},
                    "reservations": {"type": "array"}, "instructions": {"type": "array"}}},
                description="[Q] The agent's own records the flows read (project state, "
                "samples it edited, reservations, UI instructions); upserted as given, except "
                "that a stale placement (location, pose_on_stage, corner_stage_xy, "
                "rotation_deg) keeps the lab's (docs/interface.md)."),
            Action("job_status", status, any_client=True,
                   params=_schema({"run_id": {"type": "string"}}, ["run_id"]),
                   description="[Q] One job's status, steps, result and error (any module's job)."),
        ]

    @staticmethod
    def _need(types: dict[str, type]) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def check(p: dict[str, Any]) -> dict[str, Any]:
            for k, t in types.items():
                if not isinstance(p.get(k), t) or (t in (int, float) and isinstance(p.get(k), bool)):
                    raise ValueError(f"parameter {k!r} must be {t.__name__}")
            extra = set(p) - set(types)
            if extra:
                raise ValueError(f"unknown parameters: {', '.join(sorted(extra))}")
            return p
        return check

    def _read_all(self, ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
        return {"devices": self.read_devices()}

    def _home_query(self, ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
        return {"homed": {d: self.lab.devices.get(d).is_homed() for d in self.device_ids
                          if callable(getattr(self.lab.devices.get(d), "is_homed", None))}}

    def _capture(self, ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
        dev = self._dev(p["camera"])
        guarded = (self.gate.microscope(p["camera"]) if isinstance(dev, Microscope)
                   else self.gate.camera(p["camera"]))
        try:
            frame = guarded.capture()
        except QumusError as e:
            raise ValueError(str(e)) from e
        path = frame.saved_path()
        return {"camera": p["camera"], "image_id": path.rsplit("/", 1)[-1] if path else None,
                "path": path, "timestamp": frame.timestamp}

    # named points ------------------------------------------------------------
    def points(self) -> list[str]:
        reach = set(self.bench[self.arm_id].get("reach", []))
        places = set(self.bench[self.arm_id].get("places", []))
        slots = self.bench.get("slots", {})
        out = ["home"]
        for name in positions(self.bench):
            station = slots.get(name, {}).get("station", name)
            if station in reach or name in places:
                out.append(name)
        return out

    def _validate_point(self, p: dict[str, Any]) -> dict[str, Any]:
        self._need({"point": str})(p)
        if p["point"] not in self.points():
            raise ValueError(f"unknown point {p['point']!r} for {self.arm_id}")
        return p

    def _point_pose(self, point: str) -> Pose:
        b = self.bench[self.arm_id]
        if point == "home":
            return Pose(**b["home"])
        w = positions(self.bench)[point]
        bx, by = arm_base(self.bench, self.arm_id)
        return Pose(x=w["x"] - bx, y=w["y"] - by, z=float(b["safe_z"]), rotation=0.0)

    def _motion(self, fn: Callable[[], None]) -> None:
        """One guarded motion segment: a device that does not answer is an
        unknown outcome (the module locks); a safety refusal is a plain failure."""
        try:
            fn()
        except TransientError as e:
            raise UnknownOutcome(str(e)) from e
        except SafetyViolation as e:
            raise ValueError(f"refused by SafetyGate: {e}") from e

    def _go_to_point(self, ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
        arm = self.gate.arm(self.arm_id)
        target = self._point_pose(p["point"])
        safe_z = float(self.bench[self.arm_id]["safe_z"])
        now = arm.get_pose()
        ctx.sleep(self.motion_delay_s)
        if now.z < safe_z:  # lift first, never drag across the table
            self._motion(lambda: arm.move_to(Pose(x=now.x, y=now.y, z=safe_z,
                                                  rotation=now.rotation)))
            ctx.sleep(self.motion_delay_s)
        self._motion(lambda: arm.move_to(target))
        end = arm.get_pose()
        return {"arm": self.arm_id, "point": p["point"],
                "pose": {"x": end.x, "y": end.y, "z": end.z, "rotation": end.rotation}}

    # heaters and switches ----------------------------------------------------
    def _validate_heat(self, p: dict[str, Any]) -> dict[str, Any]:
        p = {**p, "target_c": float(p["target_c"]) if isinstance(p.get("target_c"), int | float)
             and not isinstance(p.get("target_c"), bool) else p.get("target_c"),
             "timeout_s": float(p["timeout_s"]) if isinstance(p.get("timeout_s"), int | float)
             and not isinstance(p.get("timeout_s"), bool) else p.get("timeout_s")}
        self._need({"device": str, "target_c": float, "timeout_s": float})(p)
        if not 0 < p["timeout_s"] <= HEAT_TIMEOUT_MAX_S:
            raise ValueError(f"timeout_s must be in (0, {HEAT_TIMEOUT_MAX_S:.0f}] (mandatory)")
        if _kind(self._dev(p["device"])) != "heater":
            raise ValueError(f"{p['device']} is not a heater")
        return p

    def _off_later(self, device_id: str, seconds: float, off: Callable[[], None]) -> None:
        old = self._timers.pop(device_id, None)
        if old is not None:
            old.cancel()

        def fire() -> None:
            try:
                off()
                self.service.emit("warn", "auto off", device=device_id, after_s=seconds)
            except Exception as e:  # noqa: BLE001 - report
                self.service.emit("error", "auto off failed", device=device_id, error=str(e))
        t = threading.Timer(seconds, fire)
        t.daemon = True
        self._timers[device_id] = t
        t.start()

    def _heat(self, ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
        heater = self.gate.heater(p["device"])
        try:
            heater.set_target(p["target_c"])
        except SafetyViolation as e:
            raise ValueError(f"refused by SafetyGate: {e}") from e
        self._off_later(p["device"], p["timeout_s"], heater.off)
        return {"device": p["device"], "target_c": p["target_c"], "off_after_s": p["timeout_s"]}

    def _heat_off(self, ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
        if _kind(self._dev(p["device"])) != "heater":
            raise ValueError(f"{p['device']} is not a heater")
        t = self._timers.pop(p["device"], None)
        if t is not None:
            t.cancel()
        self.gate.heater(p["device"]).off()
        return {"device": p["device"], "target_c": None}

    def _validate_switch(self, p: dict[str, Any]) -> dict[str, Any]:
        p = {"auto_off_s": SWITCH_AUTO_OFF_MAX_S, **p}
        if isinstance(p["auto_off_s"], int) and not isinstance(p["auto_off_s"], bool):
            p["auto_off_s"] = float(p["auto_off_s"])
        self._need({"device": str, "on": bool, "auto_off_s": float})(p)
        if not 0 < p["auto_off_s"] <= SWITCH_AUTO_OFF_MAX_S:
            raise ValueError(f"auto_off_s must be in (0, {SWITCH_AUTO_OFF_MAX_S:.0f}]")
        if _kind(self._dev(p["device"])) != "switch":
            raise ValueError(f"{p['device']} is not a switch")
        return p

    def _set_switch(self, ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
        sw = self.gate.switch(p["device"])
        sw.set(p["on"])
        if p["on"]:
            self._off_later(p["device"], p["auto_off_s"], lambda: sw.set(False))
        return {"device": p["device"], "on": p["on"],
                "auto_off_s": p["auto_off_s"] if p["on"] else None}

    # jog -----------------------------------------------------------------------
    def _validate_jog(self, p: dict[str, Any]) -> dict[str, Any]:
        if isinstance(p.get("mm"), int) and not isinstance(p.get("mm"), bool):
            p = {**p, "mm": float(p["mm"])}
        self._need({"device": str, "axis": str, "mm": float})(p)
        kind = _kind(self._dev(p["device"]))
        cap = JOG_CAP_MM.get(kind)
        if cap is None:
            raise ValueError(f"{p['device']} cannot jog")
        if p["axis"] not in ("x", "y", "z"):
            raise ValueError("axis must be x, y or z")
        if not 0 < abs(p["mm"]) <= cap:
            raise ValueError(f"jog of {p['mm']} mm is outside the {cap} mm cap for a {kind}")
        return p

    def _jog(self, ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
        dev, axis, mm = self.lab.devices.get(p["device"]), p["axis"], p["mm"]
        kind = _kind(dev)
        ctx.sleep(self.motion_delay_s)
        if kind == "arm":
            arm = self.gate.arm(p["device"])
            pose = arm.get_pose()
            self._motion(lambda: arm.move_to(pose.model_copy(update={axis: getattr(pose, axis) + mm})))
            new = arm.get_pose()
            return {"device": p["device"], axis: getattr(new, axis)}
        if kind == "stage":
            stage = self.gate.stage(p["device"])
            pose = stage.get_pose()
            self._motion(lambda: stage.move_to(pose.model_copy(update={axis: getattr(pose, axis) + mm})))
            return {"device": p["device"], axis: getattr(stage.get_pose(), axis)}
        man = self.gate.manipulator(p["device"])
        start = man.get_positions()[axis]
        self._motion(lambda: man.move_axis_to(axis, start + mm))
        return {"device": p["device"], axis: man.get_positions()[axis]}

    # whole flows as job actions [Q] -----------------------------------------
    def _flow_action(self, flow: str) -> Action:
        from qumus_simlab.flows.flow import get_flow

        try:
            schema = get_flow(flow).Params.model_json_schema()
        except Exception:  # noqa: BLE001 - describe stays available
            schema = {"type": "object"}
        props = dict(schema.get("properties", {}))
        props["job"] = {"type": "object", "description": "job bookkeeping: project_id, "
                        "requested_by (not a flow parameter)"}
        schema = {**schema, "properties": props}

        def run(ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
            job = p.pop("job", None) or {}
            if not isinstance(job, dict):
                raise ValueError("'job' must be an object")
            rid = self.jobs.submit(flow, p, project_id=job.get("project_id"),
                                   requested_by=job.get("requested_by") or ctx.client)
            return {"run_id": rid, "flow": flow}
        return Action(flow, run, params=schema, motion=True, queue=True, job=True,
                      description=f"Queue the {flow} job on this module; follow it by "
                      "run_id (store) or the events.")

    def _own_run(self, run_id: Any) -> str:
        if not isinstance(run_id, str):
            raise ValueError("run_id must be a string")
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown job {run_id}")
        if FLOW_MODULE.get(run["flow"], "stacking") != self.module:
            raise ValueError(f"job {run_id} ({run['flow']}) belongs to the "
                             f"{FLOW_MODULE.get(run['flow'])} module")
        return run_id

    def _job_actions(self) -> list[Action]:
        rid = _schema({"run_id": {"type": "string"}}, ["run_id"])

        def on_run(fn: Callable[[str], Any]) -> Callable[[CommandContext, dict[str, Any]], dict]:
            def run(ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
                run_id = self._own_run(p.get("run_id"))
                out = fn(run_id)
                return {"run_id": run_id, **({"result": out} if out is not None else {})}
            return run

        acts = [
            Action("job_cancel", on_run(self.jobs.cancel), params=rid, any_client=True,
                   description="Stop one job for good at its next safe point."),
            Action("job_retry", on_run(self.jobs.retry), params=rid, motion=True, queue=True),
            Action("job_reverify", on_run(self.jobs.reverify), params=rid, motion=True,
                   queue=True),
            Action("job_resume", on_run(self.jobs.resume), params=rid, motion=True, queue=True),
        ]

        def review(ctx: CommandContext, p: dict[str, Any]) -> dict[str, Any]:
            run_id = self._own_run(p.get("run_id"))
            return {"run_id": run_id, "result": self.jobs.review_waiting(
                run_id, project_id=p["project_id"], failure_event_id=int(p["failure_event_id"]),
                reviewer=p["reviewer"], reason=p["reason"])}
        acts.append(Action("job_review_waiting", review, any_client=True, params=_schema(
            {"run_id": {"type": "string"}, "project_id": {"type": "string"},
             "failure_event_id": {"type": "integer"}, "reviewer": {"type": "string"},
             "reason": {"type": "string"}},
            ["run_id", "project_id", "failure_event_id", "reviewer", "reason"])))
        if self.module == "stacking":  # [default] lab-wide job controls live on the last module
            acts.append(Action("job_resume_all", lambda ctx, p: {"result": self.jobs.resume_all()},
                               motion=True, queue=True))
            acts.append(Action(
                "complete_experiment",
                lambda ctx, p: {"result": self.jobs.complete_experiment(p["project_id"],
                                                                        p["summary"])},
                motion=True, queue=True, params=_schema(
                    {"project_id": {"type": "string"}, "summary": {"type": "string"}},
                    ["project_id", "summary"])))
        return acts


def build_modules(side: Any, *, motion_delay_s: float = 0.0,
                  wall: Callable[[], float] | None = None) -> dict[str, SimModule]:
    from qumus_simlab.simlab.records import LabRecords

    records = LabRecords(side.store, getattr(side, "image_dir", None),
                         getattr(side, "image_roots", ()))
    return {m: SimModule(m, side, motion_delay_s=motion_delay_s, wall=wall, records=records)
            for m in PORTS}


__all__ = ["FLOW_MODULE", "MODULE_DEVICES", "PORTS", "SimModule", "build_modules", "regions"]
