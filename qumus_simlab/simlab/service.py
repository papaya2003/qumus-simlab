"""Instrument service contract v1, server side (shared spec 2026-11-08, wire format
of reactor ADR 0001 plus the Qumus rules marked [Q]).

One `ContractService` is one lab module (one window). It knows nothing about
devices: a module gives it a list of `Action`s and three hooks (read the
devices, stop, list the module's running jobs). The service keeps the rules
every window shares:

- one control lease (default 3 s, renewed by heartbeat/renew); commands that
  are not `any_client` need it; the lease running out during motion stops it;
- `busy` while a motion command runs (or the module still has jobs running);
  a second motion command gets 409;
- an unknown outcome of a motion command (a device timeout) sets `locked`;
  only a person clears it with /api/unlock from the lab computer, never a retry;
- /api/stop works for any client, needs no lease and can be repeated;
- every step is an event with a sequence number (/api/events?since=seq).

`create_app(service)` puts it on HTTP (FastAPI). The service itself is plain
Python and thread safe, so tests can drive it without a server.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

CONTRACT_VERSION = "1"
DEFAULT_CLIENT_HEADER = "X-Qumus-Client"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class CommandRefused(Exception):
    """A command the service will not accept; `status` is the HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


class UnknownOutcome(Exception):
    """A motion command whose result is not known (no reply, timeout)."""


@dataclass
class CommandContext:
    """What an action's code gets while it runs."""

    command_id: str
    client: str
    cancelled: threading.Event
    emit: Callable[..., None]

    def check(self) -> None:
        """Raise `Stopped` when a stop arrived (call between motion segments)."""
        if self.cancelled.is_set():
            raise Stopped("stopped")

    def sleep(self, seconds: float) -> None:
        """Wait, but wake up and raise `Stopped` on a stop."""
        if self.cancelled.wait(max(0.0, seconds)):
            raise Stopped("stopped")


class Stopped(Exception):
    """Raised inside an action when /api/stop (or lease expiry) cancelled it."""


@dataclass
class Action:
    """One command a module offers.

    `run(ctx, params)` does the work and returns a result dict (or None).
    `validate(params)` returns cleaned params or raises ValueError (-> 422).
    `sync`: finished inside the POST (reads, job submission); otherwise the
    command runs on its own thread and the module is `busy` until it ends.
    `queue` [Q]: a job submission; accepted while the module still has jobs
    running (they are queued in the module's job queue), refused only while a
    single motion command is running.
    """

    name: str
    run: Callable[[CommandContext, dict[str, Any]], dict[str, Any] | None]
    params: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    validate: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    description: str = ""
    motion: bool = False
    local_only: bool = False
    any_client: bool = False
    sync: bool = True
    queue: bool = False
    exclusive: bool = False
    job: bool = False

    @property
    def needs_idle(self) -> bool:
        """Refused (409) while the module is busy."""
        return (self.motion or self.exclusive) and not self.queue

    def describe(self) -> dict[str, Any]:
        out = {"params": self.params, "any_client": self.any_client, "motion": self.motion,
               "local_only": self.local_only, "description": self.description}
        if self.job:
            out["job"] = True  # [Q] queues a whole job; follow it by run_id
        return out


@dataclass
class _Lease:
    holder: str
    expires: float


class ContractService:
    """One module window. Hooks:

    `read_devices()` -> {device_id: {field: value}} for /api/state;
    `on_stop()` -> stop everything the module is doing (jobs, heaters it owns);
                   returns True when it confirmed the stop;
    `active_jobs()` -> ids of the module's jobs still running or queued.
    """

    def __init__(self, service: str, module: str, actions: list[Action], *,
                 devices: dict[str, dict[str, Any]] | None = None,
                 read_devices: Callable[[], dict[str, dict[str, Any]]] | None = None,
                 on_stop: Callable[[], bool] | None = None,
                 active_jobs: Callable[[], list[str]] | None = None,
                 demo: bool = True, lease_s: float = 3.0,
                 client_header: str = DEFAULT_CLIENT_HEADER,
                 wall: Callable[[], float] = time.time, max_events: int = 10_000,
                 worker_prefix: str = "contract") -> None:
        self.service, self.module, self.demo = service, module, demo
        self.actions = {a.name: a for a in actions}
        self.devices = devices or {}
        self.lease_s, self.client_header, self.wall = lease_s, client_header, wall
        #: lease expiry and stop waits: monotonic, so an NTP step or a laptop
        #: sleep/wake neither lapses a live lease nor keeps a dead one (an injected
        #: test clock drives both)
        self.clock = time.monotonic if wall is time.time else wall
        self.worker_prefix = worker_prefix
        self._read_devices = read_devices or (lambda: {})
        self._on_stop = on_stop or (lambda: True)
        self._active_jobs = active_jobs or (lambda: [])
        self._lock = threading.RLock()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._seq = 0
        self._lease: _Lease | None = None
        self._worker: threading.Thread | None = None
        self._worker_action: Action | None = None
        self._cancel = threading.Event()
        self.locked = False
        self.fault: str | None = None
        self.last_command: dict[str, Any] | None = None
        self._watchdog: threading.Thread | None = None
        self._closed = threading.Event()
        self.emit("info", "service started", module=module, demo=demo)

    # -- events ---------------------------------------------------------------
    def emit(self, level: str, message: str, **fields: Any) -> int:
        with self._lock:
            self._seq += 1
            self._events.append({"seq": self._seq, "ts_unix": self.wall(), "level": level,
                                 "message": message, "fields": fields})
            return self._seq

    def events(self, since: int = 0, limit: int = 500) -> dict[str, Any]:
        with self._lock:
            out = [e for e in self._events if e["seq"] > since][:limit]
        return {"events": out, "next": out[-1]["seq"] if out else since}

    # -- describe / state -----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {"service": self.service, "contract": CONTRACT_VERSION, "module": self.module,
                "demo": self.demo, "client_header": self.client_header,
                "lease_s": self.lease_s, "devices": self.devices,
                "commands": {n: a.describe() for n, a in self.actions.items()}}

    def _lease_view(self) -> dict[str, Any] | None:
        lease = self._lease
        if lease is None:
            return None
        return {"holder": lease.holder, "remaining_s": round(max(0.0, lease.expires - self.clock()), 3)}

    def _motion_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def busy(self) -> bool:
        return self._motion_running() or bool(self._active_jobs())

    def state(self) -> dict[str, Any]:
        self.tick()
        try:
            devices, connected = self._read_devices(), True
        except Exception as e:  # noqa: BLE001 - a state read never fails the page
            devices, connected = {}, False
            self.fault = self.fault or f"device read failed: {e}"
        # job lookups read the lab store: keep them outside the lock heartbeats need
        jobs = list(self._active_jobs())
        with self._lock:
            return {"service": self.service, "module": self.module, "seq": self._seq,
                    "ts_unix": self.wall(), "demo": self.demo, "connected": connected,
                    "busy": self._motion_running() or bool(jobs), "locked": self.locked,
                    "fault": self.fault, "lease": self._lease_view(),
                    "last_command": self.last_command, "jobs": jobs, "devices": devices}

    # -- lease ----------------------------------------------------------------
    def _lease_valid(self) -> _Lease | None:
        lease = self._lease
        return lease if lease is not None and lease.expires > self.clock() else None

    def heartbeat(self, client: str) -> dict[str, Any]:
        """Take the lease when it is free (or already ours) and renew it."""
        with self._lock:
            lease = self._lease_valid()
            if lease is not None and lease.holder != client:
                raise CommandRefused(409, f"lease held by {lease.holder}")
            if lease is None:
                self.emit("info", "lease taken", client=client)
            self._lease = _Lease(client, self.clock() + self.lease_s)
            return {"lease": self._lease_view()}

    def renew(self, client: str) -> dict[str, Any]:
        with self._lock:
            lease = self._lease_valid()
            if lease is None or lease.holder != client:
                raise CommandRefused(409, "no lease to renew; send a heartbeat first")
            lease.expires = self.clock() + self.lease_s
            return {"lease": self._lease_view()}

    def release(self, client: str) -> dict[str, Any]:
        with self._lock:
            lease = self._lease_valid()
            if lease is not None and lease.holder == client:
                self._lease = None
                self.emit("info", "lease released", client=client)
            return {"lease": self._lease_view()}

    def tick(self) -> None:
        """Expire the lease; motion without a living lease is stopped."""
        with self._lock:
            lease = self._lease
            if lease is None or lease.expires > self.clock():
                return
            self._lease = None
            moving = self.busy()
            self.emit("warn", "lease expired", client=lease.holder, stopping=moving)
        if moving:
            self.stop(wait_s=0.0, reason="lease expired")

    def start_watchdog(self, period_s: float = 0.2) -> None:
        def loop() -> None:
            while not self._closed.wait(period_s):
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 - the watchdog must keep running
                    pass
        self._watchdog = threading.Thread(target=loop, daemon=True,
                                          name=f"{self.service}-lease-watchdog")
        self._watchdog.start()

    def close(self) -> None:
        self._closed.set()

    # -- commands -------------------------------------------------------------
    def command(self, client: str, body: Any, *, loopback: bool = True) -> dict[str, Any]:
        """Accept a command or raise CommandRefused. Returns the 202 body."""
        self.tick()
        if not isinstance(body, dict) or not isinstance(body.get("action"), str):
            raise CommandRefused(422, "body must be a JSON object with an 'action'")
        name = body["action"]
        action = self.actions.get(name)
        if action is None:
            raise CommandRefused(422, f"unknown action {name!r}")
        if action.local_only and not loopback:
            raise CommandRefused(403, f"{name} is local only (lab computer)")
        command_id = str(body.get("command_id") or uuid.uuid4().hex[:12])
        params = {k: v for k, v in body.items() if k not in ("action", "command_id")}
        with self._lock:
            if not action.any_client:
                if self.locked:
                    raise CommandRefused(409, "locked: a person must check the module and unlock it")
                lease = self._lease_valid()
                if lease is None:
                    raise CommandRefused(409, "this command needs the control lease (heartbeat first)")
                if lease.holder != client:
                    raise CommandRefused(409, f"lease held by {lease.holder}")
                if self._motion_running():
                    raise CommandRefused(409, "busy: a motion command is running")
                if action.needs_idle and self.busy():
                    raise CommandRefused(409, "busy: module jobs are running")
            if action.validate is not None:
                try:
                    params = action.validate(params)
                except ValueError as e:
                    raise CommandRefused(422, str(e)) from e
            self.emit("info", "command accepted", command_id=command_id, action=name,
                      client=client)
            ctx = CommandContext(command_id, client, threading.Event(), self.emit)
            if not action.sync:
                self._cancel = ctx.cancelled
                self._worker_action = action
                self.last_command = self._last(command_id, name, client, "running", None)
                self._worker = threading.Thread(target=self._execute, args=(action, ctx, params),
                                                daemon=True,
                                                name=f"{self.worker_prefix}-{self.service}-{name}")
                self._worker.start()
                return {"command_id": command_id, "state": self._state_light()}
        # synchronous: run now, outside the service lock (reads may take a moment)
        record = self._execute(action, ctx, params)
        state = self._state_light()
        state["last_command"] = record  # this command's own end, even if another ended since
        result = record.get("data")
        out = {"command_id": command_id, "state": state, "result": result}
        if isinstance(result, dict) and "run_id" in result:
            out["run_id"] = result["run_id"]
        return out

    def _last(self, command_id: str, action: str, client: str, result: str,
              message: str | None) -> dict[str, Any]:
        return {"id": command_id, "action": action, "client": client, "result": result,
                "message": message,
                "finished_at": None if result == "running" else self.wall()}

    def _execute(self, action: Action, ctx: CommandContext,
                 params: dict[str, Any]) -> dict[str, Any]:
        """Run the action; returns its last_command record."""
        name, cid, client = action.name, ctx.command_id, ctx.client
        if not action.sync:
            self.emit("info", "command started", command_id=cid, action=name)
        result: dict[str, Any] | None = None
        try:
            result = action.run(ctx, params) or {}
        except Stopped:
            outcome, message, level = "stopped", "stopped before it finished", "warn"
        except (UnknownOutcome, TimeoutError) as e:
            outcome, message, level = "failed", f"unknown outcome: {e}", "error"
            if action.motion:
                with self._lock:
                    self.locked = True
                    self.fault = f"{name}: unknown outcome ({e})"
                self.emit("error", "locked", command_id=cid, action=name, reason=str(e))
        except ValueError as e:
            if action.sync:  # a sync refusal is the caller's bad request (422)
                with self._lock:
                    self.last_command = self._last(cid, name, client, "rejected", str(e))
                self.emit("warn", "command rejected", command_id=cid, action=name, error=str(e))
                raise CommandRefused(422, str(e)) from e
            outcome, message, level = "failed", str(e), "error"
        except KeyError as e:
            msg = str(e.args[0]) if e.args else str(e)
            if action.sync:
                with self._lock:
                    self.last_command = self._last(cid, name, client, "rejected", msg)
                self.emit("warn", "command rejected", command_id=cid, action=name, error=msg)
                raise CommandRefused(422, msg) from e
            outcome, message, level = "failed", msg, "error"
        except Exception as e:  # noqa: BLE001 - reported, never kills the service
            outcome, message, level = "failed", f"{type(e).__name__}: {e}", "error"
        else:
            outcome, message, level = "ok", None, "info"
        record = self._last(cid, name, client, outcome, message)
        if outcome == "ok" and result:
            record["data"] = result  # [Q] the action's answer (e.g. run_id, image id)
        with self._lock:
            self.last_command = record
            if self._worker_action is action and not action.sync:
                self._worker_action = None
        verb = {"ok": "done", "stopped": "stopped", "failed": "failed"}[outcome]
        self.emit(level, f"command {verb}", command_id=cid, action=name,
                  **({"error": message} if message else {}),
                  **({"result": result} if result and outcome == "ok" else {}))
        return record

    def _state_light(self) -> dict[str, Any]:
        """State without device reads (fast; for command replies)."""
        busy = self.busy()
        with self._lock:
            return {"service": self.service, "seq": self._seq, "busy": busy,
                    "locked": self.locked, "fault": self.fault, "lease": self._lease_view(),
                    "last_command": self.last_command}

    # -- stop / unlock --------------------------------------------------------
    def stop(self, wait_s: float = 2.0, *, reason: str = "stop requested",
             client: str | None = None) -> dict[str, Any]:
        """Any client, no lease, idempotent."""
        self.emit("warn", reason, client=client)
        self._cancel.set()
        try:
            hook_ok = bool(self._on_stop())
        except Exception as e:  # noqa: BLE001 - report, the stop still counts as sent
            hook_ok = False
            self.emit("error", "stop hook failed", error=str(e))
        deadline = self.clock() + max(0.0, wait_s)
        while self._motion_running() and self.clock() < deadline:
            time.sleep(0.02)
        idle = not self._motion_running()
        status = "unconfirmed" if not hook_ok else "confirmed" if idle else "pending"
        self.emit("info" if status == "confirmed" else "warn", "stop " + status)
        return {"stop": status, "state": self._state_light()}

    def unlock(self, client: str, *, loopback: bool) -> dict[str, Any]:
        """[Q] A person clears `locked` from the lab computer; never automatic."""
        if not loopback:
            raise CommandRefused(403, "unlock is local only (lab computer)")
        with self._lock:
            was = self.locked
            self.locked, self.fault = False, None
        if was:
            self.emit("warn", "unlocked by a person", client=client)
        return {"locked": False, "state": self._state_light()}


def create_app(service: ContractService, *, loopback_hosts: frozenset[str] = LOOPBACK_HOSTS,
               extra: Callable[[FastAPI, Callable[[Request], bool]], None] | None = None) -> Any:
    """The HTTP face of one service (FastAPI). Bind it to 127.0.0.1.
    `extra(app, is_loopback)` adds module-specific routes (sim lab: records,
    images, the [Q-sim] bench views)."""
    app = FastAPI(title=f"Qumus {service.service}", docs_url=None, redoc_url=None)
    app.state.contract = service

    def client_of(request: Request) -> str:
        return request.headers.get(service.client_header) or "anonymous"

    def loopback(request: Request) -> bool:
        return request.client is not None and request.client.host in loopback_hosts

    def refused(e: CommandRefused) -> JSONResponse:
        return JSONResponse({"error": e.message, "state": service._state_light()},
                            status_code=e.status)

    async def body_of(request: Request) -> Any:
        raw = await request.body()
        if not raw:
            return {}
        try:
            return await request.json()
        except ValueError:
            return None

    @app.get("/api/describe")
    def describe() -> dict[str, Any]:
        return service.describe()

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return service.state()

    @app.get("/api/events")
    def events(since: int = 0, limit: int = 500) -> dict[str, Any]:
        return service.events(since, min(max(1, limit), 2000))

    @app.post("/api/command")
    async def command(request: Request) -> Any:
        body = await body_of(request)
        try:
            out = await _in_thread(service.command, client_of(request), body,
                                   loopback=loopback(request))
        except CommandRefused as e:
            return refused(e)
        return JSONResponse(out, status_code=202)

    @app.post("/api/stop")
    async def stop(request: Request) -> Any:
        body = await body_of(request) or {}
        wait_s = float(body.get("wait_s", 2.0)) if isinstance(body, dict) else 2.0
        return await _in_thread(service.stop, min(max(0.0, wait_s), 30.0),
                                client=client_of(request))

    @app.post("/api/heartbeat")
    def heartbeat(request: Request) -> Any:
        try:
            return service.heartbeat(client_of(request))
        except CommandRefused as e:
            return refused(e)

    @app.post("/api/renew")
    def renew(request: Request) -> Any:
        try:
            return service.renew(client_of(request))
        except CommandRefused as e:
            return refused(e)

    @app.post("/api/release")
    def release(request: Request) -> Any:
        return service.release(client_of(request))

    @app.post("/api/unlock")
    def unlock(request: Request) -> Any:
        try:
            return service.unlock(client_of(request), loopback=loopback(request))
        except CommandRefused as e:
            return refused(e)

    if extra is not None:
        extra(app, loopback)
    return app


async def _in_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run blocking service code off the event loop."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))
