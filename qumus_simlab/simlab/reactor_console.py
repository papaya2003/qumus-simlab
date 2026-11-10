"""The simulated vdW reactor in the real reactor's own Lab Console API (under /lab/).

Jialei, 2026-11-10: no translator. The brain speaks the real reactor's Lab Console
agent API (reactor repo shc0507/dobot_mg400, ADR 0010, docs/agent-cli.en.md and
docs/orchestrator.en.md), and the simulated reactor window (8783) imitates that API
so sim and real stay symmetric. Spec: notes/vdw-reactor-20261110/reactor-native-spec.md
(section 3). Written from the reactor's public docs; no reactor code is copied (AGPL-3.0).

The contract v1 routes of the 8783 window are unchanged; everything here lives
under /lab/:

    GET  /lab/api/console/state | state/<device> | describe/<device> | events
    POST /lab/api/console/heartbeat | control/<device>/take|release | release-page
    POST /lab/api/console/command/<device> {action, params}  -> 202 | 409 | 422
    POST /lab/api/console/stop/<device>                       (no control needed)
    POST /lab/api/dashboard/motion | dashboard/heat
    POST /lab/api/heartbeat | renew | release                 (orchestrator lease)
    POST /lab/api/runs {recipe, params, borrow: true}  GET /lab/api/state  GET /lab/api/runs/<id>
    POST /lab/api/runs/<id>/abort
    GET  /lab/sim/microscope/api/camera/microscope/frame.jpg  (a generated picture)
    POST /lab/sim/handover {slot, chip_id | null}             (sim only: the person's hand)
    GET  /lab/sim/world                                       (sim only: slots, crucible, furnace)

A small world of its own: tray slots A1..A4, one crucible, a furnace whose
temperature follows the started sequence in fast sim time (``time_scale`` sim
seconds per real second), a thermal camera whose ``crucible`` region reads the
furnace temperature. ``crucible-load`` is refused at 100 C or more,
``crucible-unload`` waits until the crucible region is below 60 C.
POST requests need ``X-MG400-Client`` (the page identity, ``agent-<session>``).
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

DEVICES = ("robot", "stage", "thermal", "vision", "furnace", "microscope")
SLOTS = ("A1", "A2", "A3", "A4")
CLIENT_HEADER = "X-MG400-Client"
AMBIENT_C = 25.0
MAX_C = 1000.0  # [default] sim host limit, the real limit is the reactor host's config
MAX_RAMP = 100.0  # [default] deg C per minute
COOL_C_PER_MIN = 30.0  # [default] free cooling of the sim furnace
LOAD_MAX_C = 100.0
UNLOAD_MAX_C = 60.0
TIME_SCALE = 600.0  # [default] sim seconds per real second (10 sim minutes per second)

#: devices each catalog recipe uses (the run borrows them from the caller's page)
RECIPE_DEVICES: dict[str, tuple[str, ...]] = {
    "slot-state": ("robot", "vision"),
    "crucible-load": ("robot", "stage", "furnace"),
    "crucible-unload": ("robot", "stage", "furnace", "thermal"),
    "microscope-record": ("microscope",),
}
RECIPE_PARAMS: dict[str, dict[str, str]] = {
    "slot-state": {"slot": "string"},
    "crucible-load": {"from_slot": "string"},
    "crucible-unload": {"to_slot": "string"},
    "microscope-record": {"chip_id": "string", "seconds": "number", "run_name": "string",
                          "mode": "string"},
}

_STEP_SCHEMA = {"type": "object", "required": ["temperature_c", "ramp_rate_c_per_min", "hold_minutes"],
                "properties": {"temperature_c": {"type": "number", "minimum": 0, "maximum": MAX_C},
                               "ramp_rate_c_per_min": {"type": "number", "exclusiveMinimum": 0,
                                                       "maximum": MAX_RAMP},
                               "hold_minutes": {"type": "number", "minimum": 0},
                               "gas": {"type": "object"}}}
COMMANDS: dict[str, dict[str, Any]] = {
    "robot": {"enable": {}, "disable": {}, "clear_alarm": {}},
    "stage": {"home": {}, "goto": {"motor": "string", "name": "string"}},
    "thermal": {"connect": {}, "shutter": {}},
    "vision": {"observe_slots": {"slots": "array"}},
    "furnace": {"start_sequence": {"steps": {"type": "array", "items": _STEP_SCHEMA},
                                   "end_gas": {"type": "object"},
                                   "end_gas_until_c": {"type": "number"}},
                "acknowledge": {}},
    "microscope": {"snapshot": {}},
}


class Refused(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


class SimReactorConsole:
    """World + console state. Thread safe; time advances on every read."""

    def __init__(self, *, time_scale: float = TIME_SCALE,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        self.time_scale = time_scale
        self.clock, self.wall = clock, wall
        self.lock = threading.RLock()
        self.slots: dict[str, str | None] = dict.fromkeys(SLOTS)
        self.crucible: str | None = None
        self.holders: dict[str, str] = {}
        self.last_command: dict[str, dict[str, Any] | None] = dict.fromkeys(DEVICES)
        self.events: list[dict[str, Any]] = []
        self.seq = 0
        self.furnace: dict[str, Any] = {"pv": AMBIENT_C, "sv": AMBIENT_C, "mode": "STOPPED",
                                        "gas": [0.0, 0.0, 0.0], "running": False,
                                        "phase": "idle", "index": 0, "hold_left": None,
                                        "steps": [], "end_gas": None, "end_gas_until_c": None}
        self._last = clock()
        self.orchestrator_holder: str | None = None
        self.orchestrator_last: dict[str, Any] | None = None
        self.active_run: dict[str, Any] | None = None
        self.runs: dict[str, dict[str, Any]] = {}
        self._abort: set[str] = set()
        self._hold: set[str] = set()
        self.snapshots = 0

    # ------------------------------------------------------------------ basics
    def _event(self, service: str, message: str, level: str = "info") -> None:
        # like the real console: seq counts per service, ts_unix orders, newest first
        seq = 1 + max((e["seq"] for e in self.events if e["service"] == service), default=0)
        self.events.insert(0, {"seq": seq, "service": service, "level": level,
                               "message": message, "ts_unix": self.wall()})
        del self.events[500:]

    def _next_id(self) -> str:
        self.seq += 1
        return str(self.seq)

    def sim_sleep(self, sim_seconds: float) -> None:
        """Wait `sim_seconds` of sim time (real wait = sim / time_scale)."""
        time.sleep(max(0.0, sim_seconds) / self.time_scale)

    # ----------------------------------------------------------------- furnace
    def _advance(self) -> None:
        now = self.clock()
        dt = (now - self._last) * self.time_scale
        self._last = now
        f = self.furnace
        while dt > 1e-9:
            if f["running"] and f["phase"] == "ramping":
                step = f["steps"][f["index"]]
                rate = step["ramp_rate_c_per_min"] / 60.0
                gap = step["temperature_c"] - f["pv"]
                need = abs(gap) / rate if rate > 0 else 0.0
                if need > dt:
                    f["pv"] += rate * dt * (1 if gap > 0 else -1)
                    dt = 0.0
                else:
                    f["pv"] = float(step["temperature_c"])
                    dt -= need
                    f["phase"], f["hold_left"] = "holding", step["hold_minutes"] * 60.0
            elif f["running"] and f["phase"] == "holding":
                if f["hold_left"] > dt:
                    f["hold_left"] -= dt
                    dt = 0.0
                    continue
                dt -= f["hold_left"]
                f["hold_left"] = None
                if f["index"] + 1 < len(f["steps"]):
                    f["index"] += 1
                    self._enter_step()
                else:
                    f["running"], f["mode"], f["sv"] = False, "STOPPED", AMBIENT_C
                    if f["end_gas"]:
                        f["phase"] = "cooling"
                        f["gas"] = _gas_list(f["end_gas"])
                    else:
                        f["phase"], f["gas"] = "complete", [0.0, 0.0, 0.0]
                    self._event("furnace", "Sequence finished")
            else:  # free cooling
                f["pv"] = max(AMBIENT_C, f["pv"] - COOL_C_PER_MIN / 60.0 * dt)
                dt = 0.0
                if f["phase"] == "cooling" and f["pv"] < (f["end_gas_until_c"] or 0):
                    f["phase"], f["gas"] = "complete", [0.0, 0.0, 0.0]
                    self._event("furnace", "Cooling gas plan finished")

    def _enter_step(self) -> None:
        f = self.furnace
        step = f["steps"][f["index"]]
        f["phase"], f["sv"] = "ramping", float(step["temperature_c"])
        f["gas"] = _gas_list(step.get("gas") or {})

    def _start_sequence(self, params: dict[str, Any]) -> str:
        steps = params.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
            raise Refused(422, "A sequence needs 1 to 32 steps.")
        clean = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise Refused(422, f"Step {i + 1} must be an object.")
            try:
                t, r, h = (float(step["temperature_c"]), float(step["ramp_rate_c_per_min"]),
                           float(step["hold_minutes"]))
            except (KeyError, TypeError, ValueError):
                raise Refused(422, f"Step {i + 1} needs temperature_c, ramp_rate_c_per_min "
                                   "and hold_minutes.") from None
            if not 0 <= t <= MAX_C:
                raise Refused(422, f"Temperature must be between 0 and {MAX_C:g} C.")
            if not 0 < r <= MAX_RAMP:
                raise Refused(422, f"Ramp rate must be above 0 and at most {MAX_RAMP:g} C/min.")
            if h < 0:
                raise Refused(422, "Hold time must not be negative.")
            gas = _check_gas(step.get("gas"), f"Step {i + 1} gas")
            clean.append({"temperature_c": t, "ramp_rate_c_per_min": r, "hold_minutes": h,
                          **({"gas": gas} if gas else {})})
        end_gas = _check_gas(params.get("end_gas"), "end_gas")
        until = params.get("end_gas_until_c")
        if any("gas" in s for s in clean) and (end_gas is None or until is None):
            raise Refused(422, "A sequence with gas needs end_gas and end_gas_until_c.")
        f = self.furnace
        if f["running"] or f["phase"] == "cooling":
            raise Refused(409, "Stop the running sequence or cooling gas plan first.")
        f.update(steps=clean, index=0, running=True, mode="RUN", end_gas=end_gas,
                 end_gas_until_c=float(until) if until is not None else None)
        self._enter_step()
        self._event("furnace", "Sequence started")
        return "Sequence started."

    def _furnace_stop(self) -> None:
        f = self.furnace
        f.update(running=False, mode="STOPPED", phase="stopped", sv=AMBIENT_C,
                 gas=[0.0, 0.0, 0.0], hold_left=None)
        self._event("furnace", "Furnace stopped: heat and gas off")

    # ------------------------------------------------------------------- reads
    def device_state(self, device: str) -> dict[str, Any]:
        with self.lock:
            self._advance()
            f = self.furnace
            base = {"service": device, "demo": True, "connected": True, "locked": False,
                    "fault": None, "busy": False, "ts_unix": self.wall(),
                    "last_command": self.last_command[device]}
            if device == "furnace":
                remaining = f["hold_left"] if f["phase"] == "holding" else None
                base.update(busy=f["running"] or f["phase"] == "cooling", pv_c=round(f["pv"], 2),
                            target_c=f["sv"], pv=round(f["pv"], 2), sv=f["sv"],
                            control_mode=f["mode"], flow_pct=list(f["gas"]),
                            flow={"connected": True, "gas_channels": [
                                {"channel_index": i, "channel_number": i + 1, "available": True,
                                 "target_percent": g} for i, g in enumerate(f["gas"])]},
                            sequence={"running": f["running"], "steps": f["steps"],
                                      "current_index": f["index"], "phase": f["phase"],
                                      "hold_remaining_seconds": remaining,
                                      "end_gas": f["end_gas"],
                                      "end_gas_until_c": f["end_gas_until_c"]})
            elif device == "thermal":
                base.update(state="streaming", fps=25.0)
            elif device == "stage":
                inside = self.crucible is not None
                base["automation"] = {"configured": True, "error": None, "poses": ["load", "furnace"],
                                      "at_poses": {"furnace": {"matched": inside},
                                                   "load": {"matched": not inside}}}
            elif device == "robot":
                base.update(vacuum={"held": None}, enabled=True)
            elif device == "vision":
                base["calibrated_cameras"] = [4]
            elif device == "microscope":
                base.update(snapshots=self.snapshots, cameras={"microscope": True, "side": False})
            run = self.active_run
            if run and not run["closed"] and device in run["devices"]:
                base["busy"] = True
            return base

    def control_list(self, page: str | None) -> list[dict[str, Any]]:
        return [{"device": d, "holder": h, "mine": h == page} for d, h in self.holders.items()]

    def overview(self, page: str | None) -> dict[str, Any]:
        with self.lock:
            run = self.active_run
            active = run is not None and not run["closed"]
            services = [{"id": d, "label": d.capitalize(), "connected": True, "configured": True,
                         "demo": True, "busy": False, "fault": None, "error": None, "phase": "ready",
                         "locked": False, "lease_holder": None}
                        for d in ("robot", "stage", "thermal", "vision", "furnace", "microscope")]
            return {"console": {"commit": "qumus-simlab", "demo": True},
                    "control": self.control_list(page),
                    "orchestrator": {"connected": True, "busy": active, "demo": True,
                                     "lease": {"holder": self.orchestrator_holder}},
                    "services": services,
                    "run": dict(run) if active else None}

    def world(self) -> dict[str, Any]:
        with self.lock:
            self._advance()
            return {"slots": dict(self.slots), "crucible": self.crucible,
                    "furnace_c": round(self.furnace["pv"], 2), "phase": self.furnace["phase"]}

    # ----------------------------------------------------------------- control
    def take(self, page: str, device: str) -> dict[str, Any]:
        with self.lock:
            holder = self.holders.get(device)
            if holder not in (None, page):
                raise Refused(409, f"{device} is held by another page.")
            if self._borrowed(device):
                raise Refused(409, f"{device} is borrowed by a running recipe.")
            self.holders[device] = page
            return {"device": device, "holder": page, "mine": True}

    def release(self, page: str, device: str) -> str:
        with self.lock:
            if self.holders.get(device) == page and not self._borrowed(device):
                del self.holders[device]
                return "released"
            return "not-held"

    def release_page(self, page: str) -> dict[str, str]:
        with self.lock:
            return {d: self.release(page, d) for d, h in list(self.holders.items()) if h == page}

    def _borrowed(self, device: str) -> bool:
        run = self.active_run
        return bool(run and not run["closed"] and device in run["devices"])

    def command(self, page: str, device: str, action: str, params: dict[str, Any]) -> str:
        with self.lock:
            self._advance()
            if self.holders.get(device) != page:
                raise Refused(409, "Take control of this device first.")
            if self._borrowed(device):
                raise Refused(409, "The device is busy with a recipe run.")
            if action not in COMMANDS[device]:
                raise Refused(422, f"Unknown {device} command {action!r}.")
            command_id = self._next_id()
            result, message = "ok", "done"
            try:
                if device == "furnace" and action == "start_sequence":
                    message = self._start_sequence(params)
                elif device == "microscope" and action == "snapshot":
                    self.snapshots += 1
                    message = "Snapshot taken."
            except Refused as exc:
                if exc.status == 409:
                    raise
                result, message = "rejected", exc.message
            self.last_command[device] = {"id": command_id, "action": action, "result": result,
                                         "message": message, "client": page,
                                         "finished_at": self.wall()}
            self._event(device, f"{action}: {result}")
            return command_id

    def stop(self, device: str) -> str:
        with self.lock:
            self._advance()
            if device == "furnace":
                self._furnace_stop()
            else:
                self._event(device, "Stop")
            run = self.active_run
            if run and not run["closed"] and device in run["devices"]:
                self._hold.add(run["run_id"])
            return "confirmed"

    def motion_stop(self) -> None:
        with self.lock:
            run = self.active_run
            if run and not run["closed"]:
                self._hold.add(run["run_id"])
            self._event("console", "STOP ALL MOTION")

    # ------------------------------------------------------------ orchestrator
    def lease(self, page: str, verb: str) -> dict[str, Any]:
        with self.lock:
            if verb == "heartbeat":
                if self.orchestrator_holder not in (None, page):
                    raise Refused(409, "Another client holds the orchestrator lease.")
                self.orchestrator_holder = page
            elif verb == "renew":
                if self.orchestrator_holder != page:
                    raise Refused(409, "The orchestrator lease expired.")
            else:
                if self.orchestrator_holder == page:
                    self.orchestrator_holder = None
                    run = self.active_run
                    if run and not run["closed"]:
                        self._hold.add(run["run_id"])  # lease release holds the run
                return {"result": "released"}
            return {"lease": {"holder": page}}

    def start_run(self, page: str, body: dict[str, Any]) -> str:
        with self.lock:
            if self.orchestrator_holder != page:
                raise Refused(409, "Acquire the orchestrator lease first.")
            recipe, params = body.get("recipe"), body.get("params") or {}
            if recipe not in RECIPE_DEVICES:
                raise Refused(422, f"Unknown recipe {recipe!r}.")
            if not isinstance(params, dict):
                raise Refused(422, "params must be an object.")
            missing_params = [k for k in RECIPE_PARAMS[recipe]
                              if k not in params and k not in ("run_name", "mode")]
            if missing_params:
                raise Refused(422, f"Missing recipe parameter(s): {', '.join(missing_params)}.")
            if self.active_run and not self.active_run["closed"]:
                raise Refused(409, "Another run is active.")
            devices = RECIPE_DEVICES[recipe]
            if body.get("borrow") is not True:
                raise Refused(409, "The simulated console only runs recipes with borrow: true.")
            missing = [d for d in devices if self.holders.get(d) != page]
            if missing:
                raise Refused(409, "Take control of these devices first: " + ", ".join(missing) + ".")
            command_id = self._next_id()
            run_id = f"run-{command_id}"
            self.active_run = {"run_id": run_id, "recipe": recipe, "status": "running",
                               "closed": False, "detail": None, "step": 0,
                               "devices": list(devices)}
            self.runs[run_id] = {"run_id": run_id, "recipe": recipe, "params": params,
                                 "status": "running", "steps": []}
            self.orchestrator_last = {"id": command_id, "action": "start", "result": "ok",
                                      "message": f"Run {run_id} started."}
            self._event("orchestrator", f"Run {run_id} ({recipe}) started")
        threading.Thread(target=self._run, args=(run_id, recipe, dict(params)),
                         name=f"sim-reactor-{run_id}", daemon=True).start()
        return command_id

    def abort(self, run_id: str) -> None:
        with self.lock:
            run = self.active_run
            if not run or run["run_id"] != run_id or run["closed"]:
                raise Refused(409, "That run is not active.")
            self._abort.add(run_id)
            if run["status"] == "held":
                self._close(run_id, "aborted", run.get("detail") or "Aborted.")

    def orchestrator_state(self) -> dict[str, Any]:
        with self.lock:
            run = self.active_run
            return {"service": "orchestrator", "demo": True, "connected": True,
                    "busy": bool(run and not run["closed"]),
                    "locked": bool(run and run["status"] == "held" and not run["closed"]),
                    "lease": {"holder": self.orchestrator_holder},
                    "last_command": self.orchestrator_last,
                    "active_run": {k: v for k, v in run.items() if k != "devices"} if run else None}

    def run_record(self, run_id: str) -> dict[str, Any] | None:
        with self.lock:
            run = self.runs.get(run_id)
            return {**run, "steps": [dict(s) for s in run["steps"]]} if run else None

    def _close(self, run_id: str, status: str, detail: str | None) -> None:
        run = self.active_run
        if run and run["run_id"] == run_id and not run["closed"]:
            run.update(status=status, closed=True, detail=detail)
            self.runs[run_id]["status"] = status
            self._event("orchestrator", f"Run {run_id} {status}" + (f": {detail}" if detail else ""))

    def _interrupted(self, run_id: str) -> str | None:
        with self.lock:
            if run_id in self._abort:
                return "abort"
            if run_id in self._hold:
                return "hold"
            return None

    def _wait(self, run_id: str, sim_seconds: float, until: Callable[[], bool] | None = None,
              limit_sim_s: float | None = None) -> str | None:
        """Sim-time wait that notices abort/hold; with `until`, wait for it (up to limit)."""
        waited = 0.0
        tick = max(1.0, self.time_scale * 0.02)  # about 20 ms real per tick
        while True:
            if (why := self._interrupted(run_id)) is not None:
                return why
            if until is None and waited >= sim_seconds:
                return None
            if until is not None:
                with self.lock:
                    self._advance()
                    done = until()
                if done:
                    return None
                if limit_sim_s is not None and waited >= limit_sim_s:
                    return "timeout"
            self.sim_sleep(tick)
            waited += tick

    def _run(self, run_id: str, recipe: str, params: dict[str, Any]) -> None:
        skill = {"slot-state": "vision.slot_state", "crucible-load": "crucible.load",
                 "crucible-unload": "crucible.unload",
                 "microscope-record": "microscope.record"}[recipe]
        error, result = None, None
        try:
            why, error, result = self._skill(run_id, recipe, params)
        except Exception as exc:  # noqa: BLE001 - a sim fault becomes a failed step
            why, error = None, f"sim error: {exc}"
        with self.lock:
            self.runs[run_id]["steps"].append({
                "skill": skill, "status": "succeeded" if error is None and why is None else "failed",
                "result_json": _json(result) if result is not None else None,
                "error": error or ({"abort": "Aborted.", "hold": "Held: motion stopped.",
                                    "timeout": "Step timeout."}.get(why) if why else None)})
            if why == "hold" and run_id not in self._abort:
                run = self.active_run
                if run and run["run_id"] == run_id:
                    run.update(status="held", detail="Held: motion stopped or lease lost.")
                    self._event("orchestrator", f"Run {run_id} held")
                return  # an abort closes it
            if why is not None or error is not None:
                self._close(run_id, "aborted", error or {"abort": "Aborted.", "timeout": "Step timeout.",
                                                         "hold": "Aborted."}[why])
            else:
                self._close(run_id, "completed", None)

    def _skill(self, run_id: str, recipe: str, params: dict[str, Any]
               ) -> tuple[str | None, str | None, Any]:
        """-> (interruption, error, result)."""
        if recipe == "slot-state":
            slot = str(params.get("slot"))
            if (why := self._wait(run_id, 20.0)) is not None:
                return why, None, None
            with self.lock:
                if slot not in self.slots:
                    return None, f"Unknown tray slot {slot!r}.", None
                if self.slots[slot] is None:
                    return None, None, {"state": "empty", "center_xy_mm": None, "angle_deg": None,
                                        "reason": None}
                return None, None, {"state": "present", "center_xy_mm": [120.0 + 15 * SLOTS.index(slot), -40.0],
                                    "angle_deg": 0.0, "reason": None}
        if recipe == "crucible-load":
            slot = str(params.get("from_slot"))
            with self.lock:
                self._advance()
                if self.slots.get(slot) is None:
                    return None, "No chip is present in the requested source slot.", None
                if self.crucible is not None:
                    return None, "The crucible already holds a chip.", None
                if self.furnace["pv"] >= LOAD_MAX_C:
                    return None, (f"The furnace is at {self.furnace['pv']:.0f} C; loading needs "
                                  f"below {LOAD_MAX_C:.0f} C."), None
            if (why := self._wait(run_id, 90.0)) is not None:
                return why, None, None
            with self.lock:
                self.crucible, self.slots[slot] = self.slots[slot], None
            return None, None, {"from_slot": slot, "inserted": True}
        if recipe == "crucible-unload":
            slot = str(params.get("to_slot"))
            with self.lock:
                if slot not in self.slots:
                    return None, f"Unknown tray slot {slot!r}.", None
                if self.slots[slot] is not None:
                    return None, "The target slot is not empty.", None
                if self.crucible is None:
                    return None, "The crucible is empty.", None
            why = self._wait(run_id, 0.0, until=lambda: self.furnace["pv"] < UNLOAD_MAX_C,
                             limit_sim_s=4 * 3600.0)
            if why is not None:
                return why, None, None
            if (why := self._wait(run_id, 90.0)) is not None:
                return why, None, None
            with self.lock:
                self.slots[slot], self.crucible = self.crucible, None
            return None, None, {"to_slot": slot, "crucible_max_c": round(self.furnace["pv"], 1)}
        if recipe == "microscope-record":
            try:
                seconds = float(params.get("seconds"))
            except (TypeError, ValueError):
                return None, "seconds must be a number.", None
            if not 0 < seconds <= 3600:
                return None, "seconds must be above 0 and at most 3600.", None
            if (why := self._wait(run_id, seconds)) is not None:
                return why, None, None
            n = len(self.runs)
            return None, None, {"recording_id": f"rec-{n}", "artifact_id": f"art-{n}",
                                "status": "completed", "chip_id": params.get("chip_id"),
                                "mode": params.get("mode") or "time_lapse"}
        return None, f"Recipe {recipe} is not simulated.", None

    # --------------------------------------------------------------- sim only
    def handover(self, slot: Any, chip_id: Any) -> dict[str, Any]:
        with self.lock:
            if slot not in self.slots:
                raise Refused(422, f"Unknown tray slot {slot!r}; known: {', '.join(SLOTS)}.")
            if chip_id is not None and (not isinstance(chip_id, str) or not chip_id):
                raise Refused(422, "chip_id must be a nonempty string or null.")
            if chip_id is not None and self.slots[slot] is not None:
                raise Refused(409, f"Tray slot {slot} is occupied.")
            self.slots[slot] = chip_id
            self._event("sim", f"Handover: slot {slot} <- {chip_id}")
            return {"slots": dict(self.slots)}

    def frame(self) -> bytes:
        """A generated microscope picture: the chip (if loaded) tinted by temperature."""
        from PIL import Image, ImageDraw

        with self.lock:
            self._advance()
            pv, chip = self.furnace["pv"], self.crucible
        heat = min(1.0, max(0.0, (pv - AMBIENT_C) / 900.0))
        img = Image.new("RGB", (640, 480), (int(30 + 200 * heat), int(30 + 60 * heat), 30))
        draw = ImageDraw.Draw(img)
        if chip:
            draw.rectangle((220, 170, 420, 310), fill=(150, 150, 170), outline=(240, 240, 240))
            draw.rectangle((300, 220, 350, 260), fill=(90, 120, 200))
        draw.text((12, 12), f"sim reactor microscope  {pv:.0f} C  {chip or 'empty crucible'}",
                  fill=(255, 255, 255))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue()


def _gas_list(gas: dict[str, Any]) -> list[float]:
    out = [0.0, 0.0, 0.0]
    for channel, percent in (gas or {}).items():
        out[int(channel)] = float(percent)
    return out


def _check_gas(value: Any, what: str) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise Refused(422, f"{what} must map channel '0'..'2' to percent.")
    out = {}
    for channel, percent in value.items():
        if str(channel) not in ("0", "1", "2"):
            raise Refused(422, f"{what}: channel must be 0, 1 or 2.")
        try:
            p = float(percent)
        except (TypeError, ValueError):
            raise Refused(422, f"{what}: percent must be a number.") from None
        if not 0 <= p <= 100:
            raise Refused(422, f"{what}: percent of full scale must be 0..100.")
        out[str(channel)] = p
    return out


def _json(value: Any) -> str:
    import json

    return json.dumps(value)


def console_routes(sim: SimReactorConsole | None = None) -> Callable[[Any, Any], None]:
    """extra(app, is_loopback) for create_app: mount the console under /lab/."""
    sim = sim or SimReactorConsole()

    def add(app: Any, loopback: Callable[[Request], bool]) -> None:
        router = APIRouter(prefix="/lab")
        app.state.reactor_console = sim

        def page_of(request: Request) -> str | None:
            return request.headers.get(CLIENT_HEADER)

        async def posted(request: Request) -> tuple[str, dict[str, Any]]:
            page = page_of(request)
            if not page:
                raise Refused(400, f"Send the page identity in {CLIENT_HEADER}.")
            raw = await request.body()
            if not raw:
                return page, {}
            try:
                body = await request.json()
            except ValueError:
                raise Refused(400, "The body must be JSON.") from None
            if not isinstance(body, dict):
                raise Refused(400, "The body must be a JSON object.")
            return page, body

        def device_of(device: str) -> str:
            if device not in DEVICES:
                raise Refused(404, "Unknown console device.")
            return device

        def guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
            import functools

            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await fn(*args, **kwargs)
                except Refused as exc:
                    body = {"error": exc.message}
                    if exc.status == 409 and fn.__name__ == "command":
                        body["outcome"] = "refused"
                    return JSONResponse(body, status_code=exc.status)
            return wrapper

        @router.get("/api/console/state")
        @guarded
        async def console_state(request: Request) -> Any:
            return sim.overview(page_of(request))

        @router.get("/api/console/state/{device}")
        @guarded
        async def one_state(device: str, request: Request) -> Any:
            device_of(device)
            holder = sim.holders.get(device)
            return {"state": sim.device_state(device),
                    "control": {"holder": holder, "mine": holder == page_of(request)}}

        @router.get("/api/console/describe/{device}")
        @guarded
        async def describe(device: str) -> Any:
            device_of(device)
            return {"commands": {name: {"params": {"type": "object", "properties": spec}}
                                 for name, spec in COMMANDS[device].items()}}

        @router.get("/api/console/events")
        @guarded
        async def events() -> Any:
            with sim.lock:
                return {"events": list(sim.events), "errors": {}}

        @router.post("/api/console/heartbeat")
        @guarded
        async def heartbeat(request: Request) -> Any:
            page, _ = await posted(request)
            with sim.lock:
                return {"control": sim.control_list(page)}

        @router.post("/api/console/control/{device}/{verb}")
        @guarded
        async def control(device: str, verb: str, request: Request) -> Any:
            page, _ = await posted(request)
            device_of(device)
            if verb == "take":
                return {"control": sim.take(page, device)}
            if verb == "release":
                return {"result": sim.release(page, device)}
            raise Refused(404, "Unknown control verb.")

        @router.post("/api/console/release-page")
        @guarded
        async def release_page(request: Request) -> Any:
            page, _ = await posted(request)
            return {"results": sim.release_page(page)}

        @router.post("/api/console/command/{device}")
        @guarded
        async def command(device: str, request: Request) -> Any:
            page, body = await posted(request)
            device_of(device)
            action, params = body.get("action"), body.get("params") or {}
            if not isinstance(action, str) or not isinstance(params, dict):
                raise Refused(422, "Send {action, params}.")
            command_id = sim.command(page, device, action, params)
            return JSONResponse({"outcome": "accepted", "command_id": command_id}, status_code=202)

        @router.post("/api/console/stop/{device}")
        @guarded
        async def stop(device: str, request: Request) -> Any:
            await posted(request)
            return {"outcome": sim.stop(device_of(device))}

        @router.post("/api/dashboard/motion")
        @guarded
        async def motion(request: Request) -> Any:
            await posted(request)
            sim.motion_stop()
            for device in ("robot", "stage", "microscope"):
                sim.stop(device)
            return JSONResponse({"operation_id": sim._next_id()}, status_code=202)

        @router.post("/api/dashboard/heat")
        @guarded
        async def heat(request: Request) -> Any:
            _, body = await posted(request)
            if body.get("confirmed") is not True:
                raise Refused(422, "Send confirmed: true.")
            sim.stop("furnace")
            return JSONResponse({"operation_id": sim._next_id()}, status_code=202)

        @router.post("/api/{verb}")
        @guarded
        async def orchestrator_lease(verb: str, request: Request) -> Any:
            page, body = await posted(request)
            if verb in ("heartbeat", "renew", "release"):
                return sim.lease(page, verb)
            if verb == "runs":
                command_id = sim.start_run(page, body)
                return JSONResponse({"command_id": command_id}, status_code=202)
            raise Refused(404, "not found")

        @router.get("/api/state")
        @guarded
        async def orchestrator_state() -> Any:
            return sim.orchestrator_state()

        @router.get("/api/recipes")
        @guarded
        async def recipes() -> Any:
            return {"recipes": [{"id": r, "params": RECIPE_PARAMS[r]} for r in RECIPE_DEVICES]}

        @router.get("/api/runs/{run_id}")
        @guarded
        async def run(run_id: str) -> Any:
            record = sim.run_record(run_id)
            if record is None:
                raise Refused(404, "Unknown run.")
            return record

        @router.post("/api/runs/{run_id}/abort")
        @guarded
        async def abort(run_id: str, request: Request) -> Any:
            await posted(request)
            sim.abort(run_id)
            return JSONResponse({"command_id": sim._next_id()}, status_code=202)

        @router.get("/sim/microscope/api/camera/microscope/frame.jpg")
        async def frame() -> Any:
            return Response(sim.frame(), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

        @router.post("/sim/handover")
        @guarded
        async def handover(request: Request) -> Any:
            _, body = await posted(request)
            return sim.handover(body.get("slot"), body.get("chip_id"))

        @router.get("/sim/world")
        async def world() -> Any:
            return sim.world()

        app.include_router(router)

    return add
