"""Fault injection for the testbench (spec 11.2, workstreams.md A2).

One entry point, `FaultInjector.inject(name, params)`, used by tests, the UI
fault menu and the CLI. `FAULTS` lists what can be injected, with English
descriptions for the UI.

Device-level faults (serial timeout, crash) act in `before_call`, which every
sim device runs before a command or a reading (devices/sim/devices.py).

In a separate lab process (runtime/labproc.py) `crash` really kills the
process and `hang` freezes it; the supervisor restarts it.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from qumus_simlab.core.errors import TransientError


class SimulatedCrash(BaseException):  # noqa: N818 - it is not an error the code handles
    """The process died here. A BaseException so no `except Exception` in flow
    or job code catches it: the step stays `in_progress`, nothing is cleaned
    up, exactly like a kill -9 between two device calls."""


#: name -> (description, params with defaults). Shown in the UI fault menu.
FAULTS: dict[str, tuple[str, dict[str, Any]]] = {
    "next_placement": ("The next object put down lands off target by dx/dy mm and drot deg "
                       "({} clears it).",
                       {"dx": 2.0, "dy": 0.0, "drot": 6.0}),
    "pick_fail_next": ("The next N picks come up empty.", {"count": 1}),
    "pick_fail_prob": ("Each pick fails with this probability.", {"value": 0.2}),
    "big_error_prob": ("Each placement is off by a few mm with this probability.",
                       {"value": 0.2}),
    "drop_in_transit": ("The next carried object falls off the nozzle on the way.", {}),
    "pickup_fail_next": ("The next N stamp pick-ups leave the flake on the chip.",
                         {"count": 1}),
    "release_fail_next": ("The next N drop-downs keep the stack on the stamp "
                          "(PC film does not release).", {"count": 1}),
    "device_offline": ("A device stops answering until set back online.",
                       {"device": "overview_camera", "offline": True}),
    "camera_frame_drop": ("The overview camera drops its next N frames (USB hiccup; "
                          "transient, retried).", {"count": 2}),
    "qr_unreadable": ("The QR markers of a storage board cannot be read for the next N "
                      "looks at it (glare, a hand in the way; transient).",
                      {"board": "middle_storage", "looks": 3}),
    "chip_removed": ("Someone takes the object out of a slot or stage without telling the "
                     "system (the ledger still lists it there).", {"place": "left_chip_2"}),
    "arm_step_loss": ("An arm loses steps: it thinks it is where it was told but sits "
                      "dx/dy mm off until it homes again.",
                      {"device": "arm_left", "dx": 1.5, "dy": 0.0}),
    "arm_reset": ("An arm's controller resets: it no longer knows where it is and must "
                  "home before it moves again.", {"device": "arm_right"}),
    "stamp_torn": ("The PC film of the stamp on the manipulator (or the next one loaded) "
                   "tears: it can no longer pick anything up.", {}),
    "tape_roll_empty": ("Set how much white tape is left on the roll (0 = empty; a full "
                        "roll is 3000 mm).", {"left_mm": 0.0}),
    "perception": ("Overview-camera detector error rates: missed objects, phantom chips, "
                   "bare/exfoliated chips confused, heavy-tailed position error "
                   "(null = config value).",
                   {"miss_prob": 0.1, "false_positive_prob": 0.05,
                    "confuse_material_prob": 0.1, "tail_prob": 0.3}),
    "serial_timeout": ("The next N calls to a device time out (transient).",
                       {"device": "arm_right", "count": 1}),
    "sensor_stuck": ("A heater's temperature reading freezes at a value (null clears).",
                     {"device": "exfoliation_heater", "value": 23.0}),
    "vacuum_stuck": ("A vacuum switch stays at a state whatever it is told (null clears).",
                     {"device": "exfoliation_vacuum", "value": False}),
    "stuck_at": ("A heater reading or vacuum switch freezes at a value (null clears); "
                 "general form of sensor_stuck / vacuum_stuck.",
                 {"device": "exfoliation_heater", "value": 22.0}),
    "crash": ("The job process dies before the Nth device call from now; the job "
              "service then restarts and recovers (kill -9 and restart).",
              {"after_calls": 1}),
    "hang": ("The lab process freezes (no heartbeat, jobs stall); the supervisor kills "
             "and restarts it. Only with a separate lab process.", {}),
}

#: faults that need a separate lab process
PROCESS_FAULTS = {"hang"}

#: perception fault parameter -> SimLocator attribute
_LOCATOR_ATTR = {"miss_prob": "miss_prob", "false_positive_prob": "false_positive_prob",
                 "confuse_material_prob": "confuse_prob", "tail_prob": "tail_prob"}


class FaultInjector:
    def __init__(self, lab: Any) -> None:
        self.lab = lab
        loc = getattr(lab, "locator", None)
        #: perception error rates from config, restored by clear()
        self._perception = {k: getattr(loc, _LOCATOR_ATTR[k]) for k in _LOCATOR_ATTR} \
            if loc is not None else {}
        #: pick / placement error probabilities from config, restored by clear()
        f = getattr(getattr(lab, "world", None), "faults", None)
        self._probs = {"pick_fail_prob": f.pick_fail_prob,
                       "big_error_prob": f.big_error_prob} if f is not None else {}

        self.calls = 0
        self.crash_at: int | None = None
        self.timeouts: dict[str, int] = {}
        self.crashed = threading.Event()  # set when a SimulatedCrash was raised
        #: called (on its own thread) after a crash: the app restarts the job
        #: service there, as a supervisor would restart the killed process
        self.on_crash: Any = None
        #: separate lab process: crash = really exit; persist() saves the sim
        #: world before every job device call (testbench/persist.py)
        self.hard_kill = False
        self.persist: Any = None
        self.hung = threading.Event()
        self._lock = threading.Lock()

    # ----- called by every sim device ---------------------------------------

    def before_call(self, device_id: str) -> None:
        # Only calls made by job workers count: a UI poll reading a heater
        # must neither crash the "process" nor eat a serial timeout.
        if not threading.current_thread().name.startswith("job-"):
            return
        if self.hung.is_set():
            threading.Event().wait()  # frozen until the supervisor kills the process
        with self._lock:
            if self.persist is not None:
                self.persist()
            self.calls += 1
            if self.crash_at is not None and self.calls >= self.crash_at:
                self.crash_at = None
                if self.hard_kill:
                    os._exit(137)  # like kill -9: no cleanup, no state save
                self.crashed.set()
                if self.on_crash is not None:
                    threading.Timer(0.02, self.on_crash).start()
                raise SimulatedCrash(f"simulated crash before a call to {device_id}")
            if self.timeouts.get(device_id, 0) > 0:
                self.timeouts[device_id] -= 1
                raise TransientError(f"{device_id}: serial timeout (no reply)",
                                     device=device_id)

    # ----- the one entry point ---------------------------------------------

    def inject(self, fault: str, params: dict[str, Any] | None = None) -> None:
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; known: {sorted(FAULTS)}")
        if fault in PROCESS_FAULTS and not self.hard_kill:
            raise ValueError(f"{fault} needs a separate lab process (serve --lab-process)")
        p = {**FAULTS[fault][1], **(params or {})}
        w, devices = self.lab.world, self.lab.devices
        if fault == "next_placement":  # explicit {} clears it (UI toggle)
            given = FAULTS[fault][1] if params is None else params
            w.faults.next_placement = {k: float(v) for k, v in given.items()
                                       if k in ("dx", "dy", "drot")}
        elif fault == "pick_fail_next":
            w.faults.pick_fail_next = int(p["count"])
        elif fault == "pick_fail_prob":
            w.faults.pick_fail_prob = float(p["value"])
        elif fault == "big_error_prob":
            w.faults.big_error_prob = float(p["value"])
        elif fault == "pickup_fail_next":
            w.faults.pickup_fail_next = int(p["count"])
        elif fault == "release_fail_next":
            w.faults.release_fail_next = int(p["count"])
        elif fault == "drop_in_transit":
            w.faults.drop_next = True
        elif fault == "device_offline":
            devices.get(str(p["device"])).offline = bool(p["offline"])
        elif fault == "serial_timeout":
            devices.get(str(p["device"]))  # unknown device -> KeyError now, not later
            self.timeouts[str(p["device"])] = int(p["count"])
        elif fault == "sensor_stuck":
            dev = devices.get(str(p["device"]))
            dev.stuck_at = None if p["value"] is None else float(p["value"])
        elif fault == "vacuum_stuck":
            dev = devices.get(str(p["device"]))
            dev.stuck_at = None if p["value"] is None else bool(p["value"])
        elif fault == "stuck_at":
            dev = devices.get(str(p["device"]))
            if not hasattr(dev, "stuck_at"):
                raise ValueError(f"{dev.device_id} cannot be stuck")
            dev.stuck_at = p["value"]
        elif fault == "camera_frame_drop":
            devices.get("overview_camera").drop_next = int(p["count"])
        elif fault == "qr_unreadable":
            board = str(p["board"])
            if board not in self.lab.bench.get("qr_markers", {}):
                raise ValueError(f"{board} has no QR markers; boards: "
                                 f"{sorted(self.lab.bench.get('qr_markers', {}))}")
            self.lab.locator.qr_unreadable[board] = int(p["looks"])
        elif fault == "chip_removed":
            self._remove_at(str(p["place"]))
        elif fault == "arm_step_loss":
            arm = devices.get(str(p["device"]))
            if not hasattr(arm, "step_offset"):
                raise ValueError(f"{arm.device_id} is not an arm")
            arm.step_offset = (float(p["dx"]), float(p["dy"]))
        elif fault == "arm_reset":
            arm = devices.get(str(p["device"]))
            if not hasattr(arm, "lose_position"):
                raise ValueError(f"{arm.device_id} is not an arm")
            arm.lose_position()
        elif fault == "stamp_torn":
            with w.lock:
                held = w.held_stamp()
                if held is not None and held.id in w.stamps:
                    w.stamps[held.id].film = False
                else:
                    w.faults.tear_next_stamp = True
        elif fault == "tape_roll_empty":
            with w.lock:
                w.tape.roll_left_mm = max(0.0, float(p["left_mm"]))
        elif fault == "perception":
            loc = self.lab.locator
            for k, attr in _LOCATOR_ATTR.items():
                if k in p:
                    v = self._perception.get(k) if p[k] is None else float(p[k])
                    setattr(loc, attr, v)
        elif fault == "hang":
            self.hung.set()
        elif fault == "crash":
            self.crashed.clear()
            self.crash_at = self.calls + max(1, int(p["after_calls"]))

    def clear(self) -> None:
        """Stop every fault that is still going on or waiting to happen, and put
        the error probabilities back to their config values.

        What a fault already did to the bench stays, as it would in the lab: a
        dropped chip stays where it fell, a torn stamp stays torn, used tape is
        used, an arm whose controller reset still has to home (the system can
        see that one). Lost steps are undone, because nothing could see them."""
        f = self.lab.world.faults
        for k, v in self._probs.items():
            setattr(f, k, v)
        f.next_placement, f.pick_fail_next, f.drop_next = {}, 0, False
        f.pickup_fail_next = f.release_fail_next = 0
        self.crash_at, self.timeouts = None, {}
        f.tear_next_stamp = False
        loc = getattr(self.lab, "locator", None)
        if loc is not None:
            loc.qr_unreadable.clear()
            for k, v in self._perception.items():
                setattr(loc, _LOCATOR_ATTR[k], v)
        for d in self.lab.devices.devices.values():
            if hasattr(d, "offline"):
                d.offline = False
            if hasattr(d, "stuck_at"):
                d.stuck_at = None
            if hasattr(d, "drop_next"):
                d.drop_next = 0
            if hasattr(d, "step_offset"):
                d.step_offset = (0.0, 0.0)

    def _remove_at(self, place: str) -> None:
        """Take whatever free object lies at a named place off the bench."""
        from qumus_simlab.flows.bench import regions

        box = regions(self.lab.bench).get(place)
        if box is None:
            raise ValueError(f"unknown place {place!r}")
        w = self.lab.world
        with w.lock:
            for oid, o in list(w.objects.items()):
                if o.held_by is None and box["x"][0] <= o.pose.x <= box["x"][1] \
                        and box["y"][0] <= o.pose.y <= box["y"][1]:
                    del w.objects[oid]
                    return
        raise ValueError(f"nothing lies at {place}")

    def describe(self) -> list[dict[str, Any]]:
        return [{"fault": k, "description": d, "params": p} for k, (d, p) in FAULTS.items()
                if self.hard_kill or k not in PROCESS_FAULTS]
