"""Contract 2 (papaya2003/lab-contract spec/contract-2.md) next to contract v1.

The three windows run in one process over one simulated bench, so the sim lab is one contract 2
**station**. Every window port (and every remote listener) also serves the station entry:

    GET /api/station                       -> services exfoliation/, stacking/, reactor/
    GET /<module>/api/describe | state | events | commands/<id> | jobs/<run_id>
    POST /<module>/api/lease               (renew_only: true to renew)    DELETE /<module>/api/lease
    POST /<module>/api/command             {command_id, action, params, deadline_s?}
    POST /<module>/api/stop

The v1 routes at /api/... stay unchanged until the brain has moved (spec section 20). This module
is an adapter: the window's ContractService still does the work (lease, busy, locked, stop), and
the adapter adds what contract 2 requires on top: client command ids with lookup and duplicate
handling, nested params, error codes, boot_id, deadlines and readings under `readings`.

Notes for the sim lab:
- Units are added to params from their names (`*_c` -> C, `*_s` -> s, `*mm` -> mm, ...).
- Windows with heaters say `on_expiry: stop_motion` and `orphan_timeout_s`: a heater set by
  `heat` always switches itself off after its mandatory `timeout_s` (at most 3600 s).
- Jobs (`job: true`) finish their command when queued; `deadline_s` then bounds the job: when it
  passes, the adapter cancels the job (it stops at its next safe point). For jobs the default is
  the maximum (48 h), so a caller that sends no deadline is never cut short by a tighter one.
- The records feed stays at `/api/records` (v1 shape, sim sample ids). The contract 2 `records`
  profile with brain-issued sample ids comes with the brain migration, so it is not listed yet.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from qumus_simlab.simlab.service import Action, CommandRefused, ContractService

CONTRACT = 2
CONTRACT_MINOR = 0
CLIENT_HEADER = "X-Lab-Client"
ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
RETRYABLE = {"busy", "lease_held", "device_offline"}
STATUS = {"bad_request": 400, "unknown_action": 400, "bad_params": 400, "local_only": 403,
          "forbidden": 403, "not_found": 404, "busy": 409, "locked": 409, "maintenance": 409,
          "lease_held": 409, "lease_lost": 409, "lease_required": 409, "id_reused": 409,
          "bad_content_type": 415, "device_offline": 503, "internal": 500}
#: param name ending -> contract 2 unit
UNIT_SUFFIXES = (("_c", "C"), ("_s", "s"), ("_mm", "mm"), ("mm", "mm"), ("_um", "um"),
                 ("_deg", "deg"), ("_sccm", "sccm"), ("_percent", "percent"))
# jobs: the default is the maximum, so a job without deadline_s is only bounded by 48 h
JOB_DEADLINE_S = (48 * 3600.0, 48 * 3600.0)
MOTION_DEADLINE_S = (600.0, 3600.0)
HEAT_DEADLINE_S = (3600.0, 3600.0)
JOB_STATE = {"queued": "queued", "running": "running", "succeeded": "succeeded", "failed": "failed",
             "cancelled": "cancelled", "paused": "held", "parked": "held", "waiting_human": "held"}
KEEP = 500


class Refusal(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message

    def response(self) -> JSONResponse:
        return JSONResponse({"error": {"code": self.code, "message": self.message,
                                       "retry": self.code in RETRYABLE}},
                            status_code=STATUS[self.code])


def _v1_code(e: CommandRefused) -> str:
    """Map a contract v1 refusal (status + text) to a contract 2 error code."""
    text = e.message.lower()
    if e.status == 403:
        return "local_only"
    if e.status == 422 or e.status == 400:
        return "unknown_action" if text.startswith("unknown action") else "bad_params"
    if e.status == 409:
        if "locked" in text:
            return "locked"
        if "needs the control lease" in text:
            return "lease_required"
        if "lease held" in text:
            return "lease_held"
        if "no lease to renew" in text:
            return "lease_lost"
        return "busy"
    return "internal"


def _text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, dict):
        return str(value.get("message") or value.get("error") or json.dumps(value))
    return str(value)


def _unit(name: str) -> str | None:
    for suffix, unit in UNIT_SUFFIXES:
        if name.endswith(suffix):
            return unit
    return None


class Window:
    """Contract 2 face of one ContractService (one sim lab module)."""

    def __init__(self, name: str, service: ContractService, station: Station,
                 module: Any = None) -> None:
        self.name, self.service, self.station, self.module = name, service, station, module
        self.lock = threading.RLock()
        self.records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.bodies: dict[str, str] = {}

    # -- describe ----------------------------------------------------------------------------
    def command_spec(self, action: Action) -> dict[str, Any]:
        params = json.loads(json.dumps(action.params or {}))
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        params["additionalProperties"] = False
        for pname, prop in params["properties"].items():
            unit = _unit(pname)
            if unit and isinstance(prop, dict) and prop.get("type") in ("number", "integer"):
                prop.setdefault("unit", unit)
        heats = action.name == "heat"
        motion = action.motion or heats
        hazard = (["heat"] if heats else ["motion"] if motion else [])
        spec: dict[str, Any] = {
            "summary": action.description or action.name,
            "params": params,
            "lease": not action.any_client,
            "motion": motion,
            "hazard": hazard,
            "local_only": action.local_only,
            "allowed_when_locked": action.any_client,
            "job": action.job,
        }
        if motion:
            default, maximum = (JOB_DEADLINE_S if action.job or action.queue else
                                HEAT_DEADLINE_S if heats else MOTION_DEADLINE_S)
            spec["deadline_s"] = {"default": default, "maximum": maximum}
        return spec

    def describe(self) -> dict[str, Any]:
        svc = self.service
        lease: dict[str, Any] = {"duration_s": svc.lease_s, "on_expiry": "stop_motion"}
        if "heat" in svc.actions:
            from qumus_simlab.simlab.modules import HEAT_TIMEOUT_MAX_S

            lease["orphan_timeout_s"] = HEAT_TIMEOUT_MAX_S
        return {"service": self.name, "station": self.station.name, "contract": CONTRACT,
                "contract_minor": CONTRACT_MINOR, "version": "qumus-simlab", "demo": svc.demo,
                "boot_id": self.station.boot_id, "lease": lease,
                "profiles": ["jobs"] if self.module is not None else [],
                "commands": {n: self.command_spec(a) for n, a in svc.actions.items()}}

    # -- state -------------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        v1 = self.service.state()
        self._sync_finished()
        with self.lock:
            last = next(reversed(self.records.values()), None)
            running = [cid for cid, r in self.records.items() if r["result"] is None]
        fault = v1.get("fault")
        lease = v1.get("lease") or {"holder": None, "remaining_s": 0.0}
        return {
            "service": self.name, "boot_id": self.station.boot_id, "seq": v1["seq"],
            "ts_unix": v1["ts_unix"], "demo": v1["demo"], "connected": v1["connected"],
            "busy": v1["busy"], "locked": v1["locked"], "maintenance": None,
            "fault": None if not fault else {"code": "fault", "message": str(fault),
                                             "since_unix": v1["ts_unix"]},
            "lease": {"holder": lease["holder"], "remaining_s": lease["remaining_s"]},
            "running": running,
            "last_command": (None if last is None else
                             {"id": last["id"], "action": last["action"], "result": last["result"]}),
            "readings": {"devices": v1.get("devices", {}), "jobs": v1.get("jobs", [])},
        }

    # -- events ------------------------------------------------------------------------------
    @staticmethod
    def _kind(e: dict[str, Any]) -> str:
        msg = e["message"]
        if "lease" in msg:
            return "lease"
        if msg.startswith("command") or msg.startswith("stop"):
            return "command"
        if msg in ("locked", "unlocked by a person"):
            return "fault"
        if msg.startswith("job") or "job" in (e.get("fields") or {}):
            return "job"
        return "log"

    def events(self, since: int, limit: int) -> dict[str, Any]:
        page = self.service.events(since, limit)
        level = {"warn": "warning"}
        out = [{"seq": e["seq"], "ts_unix": e["ts_unix"], "level": level.get(e["level"], e["level"]),
                "kind": self._kind(e), "message": e["message"], "fields": e.get("fields") or {}}
               for e in page["events"]]
        return {"boot_id": self.station.boot_id, "events": out, "next": page["next"]}

    # -- lease -------------------------------------------------------------------------------
    def lease(self, client: str, renew_only: bool) -> dict[str, Any]:
        try:
            out = self.service.renew(client) if renew_only else self.service.heartbeat(client)
        except CommandRefused as e:
            code = _v1_code(e)
            if renew_only:
                code = "lease_lost"
            raise Refusal(code, e.message) from None
        view = out["lease"] or {"holder": None, "remaining_s": 0.0}
        return {"holder": view["holder"], "remaining_s": view["remaining_s"],
                "duration_s": self.service.lease_s}

    def release(self, client: str) -> dict[str, Any]:
        lease = self.service.state().get("lease")
        if lease is None or lease["holder"] != client:
            raise Refusal("lease_lost" if lease is None else "lease_held",
                          "you do not hold the lease")
        self.service.release(client)
        return {"holder": None, "remaining_s": 0.0, "duration_s": self.service.lease_s}

    # -- commands ----------------------------------------------------------------------------
    def command(self, client: str, body: dict[str, Any], loopback: bool) -> tuple[int, dict]:
        extra = sorted(set(body) - {"command_id", "action", "params", "deadline_s"})
        if extra:
            raise Refusal("bad_params", f"unknown fields {extra}; params go under 'params'")
        cid, action, params = body.get("command_id"), body.get("action"), body.get("params")
        if not isinstance(cid, str) or not ID_RE.match(cid):
            raise Refusal("bad_request", "command_id is required: 1-64 chars of [A-Za-z0-9._:-]")
        if not isinstance(action, str):
            raise Refusal("bad_request", "action is required")
        if not isinstance(params, dict):
            raise Refusal("bad_params", "params must be an object ({} when none)")
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if cid in self.bodies:
                if self.bodies[cid] != digest:
                    raise Refusal("id_reused", f"command_id {cid} was used with another body")
                rec = self.records[cid]
                duplicate = {"command_id": cid, "accepted_unix": rec["accepted_unix"], "duplicate": True}
            else:
                duplicate = None
                # Claim the id now, in the same lock as the check: a retry that
                # arrives while this request is still being handled is a
                # duplicate, never a second run (the record below replaces this).
                self._remember(cid, digest, {
                    "id": cid, "action": str(action), "client": client, "params": params,
                    "result": None, "message": "", "accepted_unix": time.time(),
                    "finished_unix": None, "data": {}, "boot_id": self.station.boot_id})
        if duplicate is not None:
            return 200, {**duplicate, "seq": self.service.state()["seq"]}
        act = self.service.actions.get(action)
        if act is None:
            return self._reject(cid, digest, client, action, params,
                                Refusal("unknown_action", f"no command {action!r}"))
        spec = self.command_spec(act)
        known = set(spec["params"]["properties"])
        unknown = sorted(set(params) - known)
        if unknown:
            return self._reject(cid, digest, client, action, params,
                                Refusal("bad_params", f"unknown params {unknown}"))
        deadline_s = body.get("deadline_s")
        if deadline_s is not None:
            limit = spec.get("deadline_s", {}).get("maximum")
            if (not isinstance(deadline_s, int | float) or isinstance(deadline_s, bool)
                    or deadline_s <= 0 or (limit is not None and deadline_s > limit)):
                return self._reject(cid, digest, client, action, params,
                                    Refusal("bad_params", f"deadline_s must be > 0 and <= {limit}"))
        elif "deadline_s" in spec:
            deadline_s = spec["deadline_s"]["default"]
        now = time.time()
        record = {"id": cid, "action": action, "client": client, "params": params, "result": None,
                  "message": "", "accepted_unix": now, "finished_unix": None, "data": {},
                  "boot_id": self.station.boot_id,
                  # v1 events after this seq carry this command's end (see _sync_finished)
                  "_since": self._v1_seq()}
        with self.lock:
            self._remember(cid, digest, record)
        try:
            out = self.service.command(client, {"action": action, "command_id": cid, **params},
                                       loopback=loopback)
        except CommandRefused as e:
            self._finish(record, "rejected", e.message)
            raise Refusal(_v1_code(e), e.message) from None
        if act.sync:
            last = (out.get("state") or {}).get("last_command") or {}
            result = {"ok": "ok", "stopped": "stopped"}.get(last.get("result"), "failed")
            if result == "failed" and "unknown outcome" in (last.get("message") or ""):
                result = "unknown"
            self._finish(record, result, last.get("message") or "", out.get("result") or {})
            run_id = (out.get("result") or {}).get("run_id") if isinstance(out.get("result"), dict) \
                else None
            if act.job and run_id and deadline_s:
                self._job_deadline(run_id, float(deadline_s))
        elif deadline_s:
            self._deadline(record, float(deadline_s))
        return 202, {"command_id": cid, "accepted_unix": now, "seq": self.service.state()["seq"]}

    def _v1_seq(self) -> int:
        with self.service._lock:  # noqa: SLF001 - same package
            return self.service._seq  # noqa: SLF001

    def _reject(self, cid: str, digest: str, client: str, action: Any, params: Any,
                refusal: Refusal) -> tuple[int, dict]:
        now = time.time()
        with self.lock:
            self._remember(cid, digest, {
                "id": cid, "action": str(action), "client": client, "params": params,
                "result": "rejected", "message": refusal.message, "accepted_unix": now,
                "finished_unix": now, "data": {}, "boot_id": self.station.boot_id})
        raise refusal

    def _remember(self, cid: str, digest: str, record: dict[str, Any]) -> None:
        self.records[cid] = record
        self.bodies[cid] = digest
        while len(self.records) > KEEP:
            old, _ = self.records.popitem(last=False)
            self.bodies.pop(old, None)

    def _finish(self, record: dict[str, Any], result: str, message: str,
                data: dict[str, Any] | None = None) -> None:
        with self.lock:
            if record["result"] is None:
                record.update(result=result, message=message or "", data=data or {},
                              finished_unix=time.time())

    def _deadline(self, record: dict[str, Any], seconds: float) -> None:
        def fire() -> None:
            self._sync_finished()
            if record["result"] is None:
                record["expiring"] = True
                self.service.stop(wait_s=2.0, reason=f"deadline of {record['id']} passed",
                                  client=f"deadline:{record['id']}")
                self._sync_finished()
        t = threading.Timer(seconds, fire)
        t.daemon = True
        t.start()

    def _job_deadline(self, run_id: str, seconds: float) -> None:
        def fire() -> None:
            runs = self.module.store.run_statuses() if self.module is not None else []
            run = next((r for r in runs if r["id"] == run_id), None)
            if run is not None and run["status"] in ("queued", "running"):
                self.service.emit("warn", "job deadline passed: cancelling", job=run_id)
                try:
                    self.module.jobs.cancel(run_id)
                except Exception as e:  # noqa: BLE001 - report
                    self.service.emit("error", "job deadline cancel failed", job=run_id,
                                      error=str(e))
        t = threading.Timer(seconds, fire)
        t.daemon = True
        t.start()

    def _sync_finished(self) -> None:
        """Finish async records from the window's v1 events ('command done/stopped/failed')."""
        with self.lock:
            open_recs = {cid: r for cid, r in self.records.items() if r["result"] is None}
        if not open_recs:
            return
        since = min(r["_since"] for r in open_recs.values())
        while True:
            page = self.service.events(since, 2000)
            for e in page["events"]:
                f = e.get("fields") or {}
                rec = open_recs.get(f.get("command_id"))
                if rec is None or e["message"] not in ("command done", "command stopped",
                                                       "command failed"):
                    continue
                verb = e["message"].removeprefix("command ")
                result = {"done": "ok", "stopped": "stopped", "failed": "failed"}[verb]
                if result == "stopped" and rec.get("expiring"):
                    result = "expired"
                if result == "failed" and "unknown outcome" in str(f.get("error", "")):
                    result = "unknown"
                self._finish(rec, result, str(f.get("error") or ""), f.get("result") or {})
            if not page["events"] or page["next"] <= since:
                break
            since = page["next"]

    def command_record(self, cid: str) -> dict[str, Any]:
        self._sync_finished()
        with self.lock:
            rec = self.records.get(cid)
            if rec is None:
                raise Refusal("not_found", f"no command {cid} in this boot")
            return {k: v for k, v in rec.items() if not k.startswith("_") and k != "expiring"}

    def stop(self, client: str, wait_s: float) -> dict[str, Any]:
        out = self.service.stop(wait_s, client=client)
        return {"stop": out["stop"], "seq": self.service.state()["seq"]}

    # -- jobs profile ------------------------------------------------------------------------
    def job(self, run_id: str) -> dict[str, Any]:
        store = self.module.store if self.module is not None else None
        run = store.get_run(run_id) if store is not None else None
        if run is None:
            raise Refusal("not_found", f"no run {run_id}")
        steps = store.get_steps(run_id) or []
        done = [s for s in steps if s.get("status") in ("done", "skipped")]
        current = steps[len(done)] if len(done) < len(steps) else (steps[-1] if steps else {})
        state = JOB_STATE.get(run["status"], "held")
        return {"run_id": run_id, "action": run["flow"], "state": state, "step": len(done),
                "steps_total": len(steps), "step_name": str(current.get("name") or ""),
                "message": _text(run.get("error")), "safe_to_resume": state == "held",
                "started_unix": run.get("created_at"),
                "finished_unix": (run.get("updated_at")
                                  if state in ("succeeded", "failed", "cancelled") else None),
                "sample_ids": [], "image_ids": [],
                "result": run.get("result") if isinstance(run.get("result"), dict) else None}


class Station:
    """The sim lab as one contract 2 station: the three windows behind one entry."""

    def __init__(self, windows: dict[str, tuple[ContractService, Any]], *,
                 name: str = "qumus-simlab") -> None:
        self.name = name
        self.boot_id = time.strftime("b-%Y%m%dT%H%M%S-") + secrets.token_hex(3)
        self.windows = {n: Window(n, svc, self, mod) for n, (svc, mod) in windows.items()}

    def entry(self) -> dict[str, Any]:
        demo = all(w.service.demo for w in self.windows.values())
        return {"station": self.name, "contract": CONTRACT, "contract_minor": CONTRACT_MINOR,
                "demo": demo, "boot_id": self.boot_id,
                "services": [{"name": n, "path": f"{n}/", "kind": "device"} for n in self.windows],
                "auth": "none", "maintenance": None}

    def routes(self, app: FastAPI, loopback: Any) -> None:
        """Add the contract 2 routes to one window app (every port serves the same station)."""
        station = self

        def window(name: str) -> Window:
            w = station.windows.get(name)
            if w is None:
                raise Refusal("not_found", f"no service {name}")
            return w

        def client_of(request: Request) -> str:
            client = request.headers.get(CLIENT_HEADER)
            if not client:
                raise Refusal("bad_request", f"{CLIENT_HEADER} header is required")
            return client

        async def json_body(request: Request) -> dict[str, Any]:
            ctype = (request.headers.get("content-type") or "").split(";")[0].strip()
            if ctype != "application/json":
                raise Refusal("bad_content_type", "POST bodies are application/json")
            raw = await request.body()
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                raise Refusal("bad_request", "body is not JSON") from None
            if not isinstance(body, dict):
                raise Refusal("bad_request", "body must be a JSON object")
            return body

        async def guarded(fn: Any) -> Any:
            from qumus_simlab.simlab.service import _in_thread

            try:
                return await fn(_in_thread)
            except Refusal as r:
                return r.response()
            except Exception as e:  # noqa: BLE001 - always a contract 2 error body
                return Refusal("internal", f"{type(e).__name__}: {e}").response()

        @app.get("/api/station")
        def entry() -> Any:
            return station.entry()

        @app.get("/{name}/api/describe")
        async def describe(name: str) -> Any:
            return await guarded(lambda run: run(lambda: window(name).describe()))

        @app.get("/{name}/api/state")
        async def state(name: str) -> Any:
            return await guarded(lambda run: run(lambda: window(name).state()))

        @app.get("/{name}/api/events")
        async def events(name: str, since: int = 0, limit: int = 500) -> Any:
            return await guarded(lambda run: run(
                lambda: window(name).events(max(0, since), min(max(1, limit), 2000))))

        @app.get("/{name}/api/commands/{cid}")
        async def record(name: str, cid: str) -> Any:
            return await guarded(lambda run: run(lambda: window(name).command_record(cid)))

        @app.get("/{name}/api/jobs/{run_id}")
        async def job(name: str, run_id: str) -> Any:
            return await guarded(lambda run: run(lambda: window(name).job(run_id)))

        @app.post("/{name}/api/lease")
        async def take(name: str, request: Request) -> Any:
            async def go(run: Any) -> Any:
                client = client_of(request)
                body = await json_body(request)
                return await run(lambda: window(name).lease(client, bool(body.get("renew_only"))))
            return await guarded(go)

        @app.delete("/{name}/api/lease")
        async def give_back(name: str, request: Request) -> Any:
            async def go(run: Any) -> Any:
                client = client_of(request)
                return await run(lambda: window(name).release(client))
            return await guarded(go)

        @app.post("/{name}/api/command")
        async def command(name: str, request: Request) -> Any:
            async def go(run: Any) -> Any:
                client = client_of(request)
                body = await json_body(request)
                status, out = await run(lambda: window(name).command(client, body, loopback(request)))
                return JSONResponse(out, status_code=status)
            return await guarded(go)

        @app.post("/{name}/api/stop")
        async def stop(name: str, request: Request) -> Any:
            async def go(run: Any) -> Any:
                body = await json_body(request)
                wait_s = body.get("wait_s", 0)
                if not isinstance(wait_s, int | float) or isinstance(wait_s, bool) \
                        or not 0 <= wait_s <= 15:
                    raise Refusal("bad_params", "wait_s must be 0..15")
                client = request.headers.get(CLIENT_HEADER) or "anonymous"
                return await run(lambda: window(name).stop(client, float(wait_s)))
            return await guarded(go)

        # anything else under a window's api/ answers with a contract 2 error body (registered
        # last, per window name, so it shadows neither the routes above nor /lab/ or /api/)
        for wname in station.windows:
            @app.api_route(f"/{wname}/api/{{rest:path}}", methods=["GET", "POST", "PUT", "DELETE"])
            def missing(rest: str, request: Request) -> Any:
                return Refusal("not_found", f"no route {request.url.path}").response()


__all__ = ["CLIENT_HEADER", "CONTRACT", "Station", "Window"]
