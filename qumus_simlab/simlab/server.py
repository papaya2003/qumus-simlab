"""Run the three sim-lab windows as HTTP services on loopback.

    python -m qumus_simlab serve --db runs/simlab/qumus.db --seed 3
        exfoliation  http://127.0.0.1:8781/api/describe
        stacking     http://127.0.0.1:8782/api/describe
        reactor      http://127.0.0.1:8783/api/describe

Every window serves contract v1 plus the [Q] records/images extensions
(simlab/records.py) and the [Q-sim] bench views the agent's live page draws
(`/api/sim/*`: layout, bench state, camera previews, testbench faults). The
bench views exist only in the simulation; a real station never serves them.
Port 0 picks free ports (tests).

Same computer or another computer (Jialei 2026-11-10): the windows bind to
127.0.0.1 by default. For an agent on another computer either
- tunnel: `--remote-ports` adds a second listener per window (default
  28781/28782/28783, loopback) where EVERY request counts as remote, so an
  SSH tunnel ends there and local-only actions (jog, raw switch, unlock,
  fault injection) stay refused even though the tunnel looks like loopback; or
- network: `--host 0.0.0.0` (or a LAN address); requests from other
  addresses are remote and get the same refusals.
The agent only needs the URL of each window (its stations.yaml).
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Request

from qumus_simlab.simlab.modules import PORTS, SimModule, build_modules
from qumus_simlab.simlab.service import create_app


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def device_state(dev: Any) -> dict[str, Any]:
    """Read-only view of one device for the agent's lab panel (never commands)."""
    out: dict[str, Any] = {"id": dev.device_id, "kind": type(dev).__name__.removeprefix("Sim"),
                           "offline": bool(getattr(dev, "offline", False))}
    if getattr(dev, "stuck_at", None) is not None:
        out["stuck_at"] = dev.stuck_at
    try:
        if hasattr(dev, "get_pose"):
            out["pose"] = dev.get_pose().model_dump()  # an arm: in its own frame
        if hasattr(dev, "world_pose"):
            out["world_pose"] = dev.world_pose().model_dump()
        if hasattr(dev, "is_homed"):
            out["homed"] = bool(dev.is_homed())
        if hasattr(dev, "remaining_mm"):
            out["roll_left_mm"] = dev.remaining_mm()
        if hasattr(dev, "focus_plane"):
            out["focus_plane"] = dev.focus_plane()
        if hasattr(dev, "get_objective"):
            out["objective"] = dev.get_objective()
        if hasattr(dev, "get_positions"):
            out["positions"] = {k: round(v, 2) for k, v in dev.get_positions().items()}
        if hasattr(dev, "vacuum_on"):
            out["vacuum"] = dev.vacuum_on()
        if hasattr(dev, "read_temperature") and not out["offline"]:
            out["temperature"] = round(dev.read_temperature(), 1)
            out["target"] = dev.get_target()
        if hasattr(dev, "read_flow") and not out["offline"]:
            out["flow_sccm"] = round(dev.read_flow(), 1)
            out["flow_target_sccm"] = dev.get_target()
        if hasattr(dev, "get") and callable(dev.get) and not hasattr(dev, "get_pose"):
            out["on"] = bool(dev.get())
    except Exception as e:  # noqa: BLE001 - a status read must never break the page
        out["error"] = str(e)
    return out


def bench_state(side: Any) -> dict[str, Any]:
    """The lab half of the agent page's bench picture (qumus-v2 ui/app.py
    bench_state without observed_scene and busy, which the agent adds)."""
    lab, store = side.lab, side.store
    w = lab.world
    f = w.faults
    objects = w.snapshot()
    status = {x["id"]: x["status"] for x in store.list_samples(kind="stamp")}
    for o in objects:
        if o["kind"] == "stamp":
            o["status"] = status.get(o["id"], "fresh")
    return {
        "devices": [device_state(d) for d in lab.devices.devices.values()],
        "objects": objects,
        "twin": lab.twin.state() if hasattr(lab, "twin") else {},
        "stamp": w.stamp_snapshot() if hasattr(w, "stamp_snapshot") else None,
        "faults": {"pick_fail_prob": f.pick_fail_prob, "big_error_prob": f.big_error_prob,
                   "next_placement": f.next_placement,
                   "armed": {"pick_fail_next": f.pick_fail_next,
                             "drop_in_transit": int(bool(f.drop_next)),
                             "pickup_fail_next": getattr(f, "pickup_fail_next", 0),
                             "release_fail_next": getattr(f, "release_fail_next", 0),
                             "stamp_torn_next": int(bool(getattr(f, "tear_next_stamp",
                                                                 False)))}},
    }


def sim_routes(side: Any, records: Any, info: dict[str, Any] | None = None) -> Any:
    """The [Q] records/images routes and the [Q-sim] bench views of one window."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    from qumus_simlab.flows.bench import positions, regions

    cache: dict[str, Any] = {}

    def add(app: Any, loopback: Any) -> None:
        @app.get("/api/records")
        def records_feed(since: int = 0, limit: int = 500) -> dict[str, Any]:
            return records.changes(since, limit)

        @app.get("/api/images/{image_id:path}/meta")
        def image_meta(image_id: str) -> dict[str, Any]:
            try:
                return records.image_meta(image_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=f"no image {image_id}") from e

        @app.get("/api/images/{image_id:path}")
        def image(image_id: str) -> Any:
            try:
                f = records.image_path(image_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=f"no image {image_id}") from e
            return FileResponse(f, headers={"Cache-Control": "no-cache"})

        @app.get("/api/sim/layout")
        def layout() -> dict[str, Any]:
            if "layout" not in cache:
                bench = side.lab.bench
                cache["layout"] = records.scrub({
                    "bench": bench,
                    "layout": {**bench, "regions": regions(bench),
                               "positions": positions(bench)}})
            return cache["layout"]

        @app.get("/api/sim/bench_state")
        def sim_bench_state() -> dict[str, Any]:
            return records.scrub(bench_state(side))

        @app.get("/api/sim/preview/{camera_id}")
        def preview(camera_id: str) -> dict[str, Any]:
            lab = side.lab
            known = (camera_id in lab.bench.get("cameras", {}) or camera_id == "microscope"
                     or (camera_id == "reactor_microscope" and "reactor_module" in lab.bench))
            if not known:
                raise HTTPException(status_code=404, detail=f"unknown camera {camera_id}")
            p = lab.preview(camera_id) if hasattr(lab, "preview") else None
            if p is None:
                return {"camera": camera_id, "image_id": None, "timestamp": None, "meta": {}}
            iid = records.image_id(p.get("path")) if p.get("path") else None
            version = None
            if iid is not None:
                try:
                    version = records.image_path(iid).stat().st_mtime_ns
                except (ValueError, OSError):
                    iid = None
            return {"camera": camera_id, "image_id": iid, "v": version,
                    "timestamp": p.get("timestamp"),
                    "meta": records.scrub(p.get("meta") or {})}

        @app.post("/api/sim/checkpoint")
        async def checkpoint(request: Request) -> Any:
            """Save point of the whole sim lab (store + world), local only; 409 while a
            job is open. The agent saves its own store at the same moment."""
            from types import SimpleNamespace

            from qumus_simlab.testbench.checkpoint import open_jobs, save_checkpoint

            if not loopback(request):
                raise HTTPException(status_code=403, detail="save points are local only")
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("dir"), str):
                raise HTTPException(status_code=400, detail="body: {dir, label}")
            if busy := open_jobs(side.store):
                raise HTTPException(status_code=409, detail=f"jobs still open: {', '.join(busy)}")
            roots = [str(r) for r in records.roots]
            extra = {**(info or {}), "image_roots": roots, "lab_id": records.lab_id,
                     "records_head": records.changes(0, 1)["head"]}
            try:
                path = save_checkpoint(SimpleNamespace(store=side.store, lab=side.lab,
                                                       clock=side.clock),
                                       Path(body["dir"]), label=str(body.get("label") or ""),
                                       extra=extra)
            except RuntimeError as e:
                raise HTTPException(status_code=409, detail=str(e)) from e
            return {"ok": True, "dir": str(path), "lab_id": records.lab_id,
                    "records_head": extra["records_head"]}

        @app.get("/api/sim/faults")
        def faults() -> Any:
            return side.lab.faults.describe()

        @app.post("/api/sim/faults")
        async def inject(request: Request) -> Any:
            if not loopback(request):
                raise HTTPException(status_code=403, detail="fault injection is local only")
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("fault"), str):
                raise HTTPException(status_code=400, detail="body: {fault, params}")
            try:
                side.lab.faults.inject(body["fault"], body.get("params") or {})
            except (ValueError, KeyError, TypeError) as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            side.store.emit("fault.injected", {"fault": body["fault"],
                                               "params": body.get("params") or {}})
            return {"ok": True}

    return add


class SimLabServer:
    """The three windows on their own uvicorn servers (threads) over one lab side."""

    def __init__(self, side: Any, *, host: str = "127.0.0.1",
                 ports: dict[str, int | None] | None = None, motion_delay_s: float = 0.0,
                 remote_ports: dict[str, int] | None = None,
                 info: dict[str, Any] | None = None) -> None:
        self.side, self.host = side, host
        #: how this lab was started (seed, target bundle): kept with save points
        self.info = dict(info or {})
        #: extra listeners (loopback) that treat every request as remote: SSH tunnels end here
        self.remote_ports = dict(remote_ports or {})
        self.modules: dict[str, SimModule] = build_modules(side, motion_delay_s=motion_delay_s)
        wanted = {**PORTS, **{k: v for k, v in (ports or {}).items() if v is not None}}
        self.ports = {m: (wanted[m] or _free_port(host)) for m in self.modules}
        self._servers: dict[str, Any] = {}
        self._threads: list[threading.Thread] = []

    @property
    def records(self) -> Any:
        return next(iter(self.modules.values())).records

    def url(self, module: str) -> str:
        return f"http://{self.host}:{self.ports[module]}/"

    @property
    def urls(self) -> dict[str, str]:
        return {m: self.url(m) for m in self.modules}

    def start(self, timeout_s: float = 10.0) -> SimLabServer:
        import uvicorn

        extra = sim_routes(self.side, self.records, self.info)
        # the reactor window also speaks the real reactor's Lab Console API under /lab/
        # (reactor_console.py); one simulated reactor for its normal and remote listener
        from qumus_simlab.simlab.reactor_console import SimReactorConsole, console_routes

        self.reactor_console = SimReactorConsole()
        lab_routes = console_routes(self.reactor_console)

        def reactor_extra(app: Any, loopback: Any) -> None:
            extra(app, loopback)
            lab_routes(app, loopback)

        listeners = [(m, self.host, self.ports[m], None) for m in self.modules]
        listeners += [(m, "127.0.0.1", p, frozenset()) for m, p in self.remote_ports.items()
                      if m in self.modules]
        for module in self.modules.values():
            module.service.start_watchdog()
        for module, host, port, loop_hosts in listeners:
            mod = self.modules[module]
            kwargs = {} if loop_hosts is None else {"loopback_hosts": loop_hosts}
            add = reactor_extra if module == "reactor" else extra
            config = uvicorn.Config(create_app(mod.service, extra=add, **kwargs), host=host,
                                    port=port, log_level="warning",
                                    access_log=False, lifespan="off")
            server = uvicorn.Server(config)
            server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
            name = f"simlab-{module}" + ("-remote" if loop_hosts is not None else "")
            t = threading.Thread(target=server.run, daemon=True, name=name)
            t.start()
            self._servers[name], self._threads = server, [*self._threads, t]
        deadline = time.monotonic() + timeout_s
        while not all(s.started for s in self._servers.values()):
            if time.monotonic() > deadline or any(not t.is_alive() for t in self._threads):
                self.close()
                raise RuntimeError(f"sim lab services did not start on {self.urls} "
                                   "(port in use?)")
            time.sleep(0.02)
        return self

    def close(self) -> None:
        for mod in self.modules.values():
            mod.service.close()
        for s in self._servers.values():
            s.should_exit = True
        for t in self._threads:
            t.join(timeout=5)


def build_side(db: str | Path, seed: int | None, target_bundle: str | Path | None = None,
               resume: str | Path | None = None, accept_changed: tuple[str, ...] = ()) -> Any:
    """The one simulated bench: the v1-linked Hall-bar laboratory, optionally
    with an imported electrode target (the operator's GDS bundle).

    `resume`: a save point (POST /api/sim/checkpoint) to start from; its seed and
    target bundle are used, and `db` must be `<new folder>/qumus.db`."""
    from qumus_simlab.config import simulation_profile
    from qumus_simlab.lab import build_lab_side

    db = Path(db).resolve()
    if db.exists():
        raise SystemExit(f"{db} exists: the sim lab always starts from a fresh bench; "
                         "give a new --db path")
    db.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {}
    if resume is not None:
        from qumus_simlab.testbench.checkpoint import StaleCheckpoint, restore_checkpoint

        if db.name != "qumus.db":
            raise SystemExit("with --resume, --db must be <new folder>/qumus.db")
        try:
            saved = restore_checkpoint(resume, db.parent, accept_changed=accept_changed)
        except StaleCheckpoint as e:
            raise SystemExit(f"cannot resume: {e}; pass --accept-changed <file> only for a "
                             "change proven not to alter the search") from None
        if seed is not None and saved.get("seed") is not None and seed != saved["seed"]:
            raise SystemExit(f"the save point was made with seed {saved['seed']}")
        seed = saved.get("seed", seed)
        target_bundle = saved.get("target_bundle") or target_bundle
    config = simulation_profile("hallbar-linked")
    if target_bundle is not None:
        from qumus_simlab.testbench.hallbar import (
            load_imported_hallbar_target,
            with_hallbar_target,
        )

        bundle = Path(target_bundle).resolve()
        load_imported_hallbar_target(bundle, workspace=db.parent)
        config = with_hallbar_target(config, target=load_imported_hallbar_target(bundle))
    side = build_lab_side(db, seed=seed, testbench_config=config,
                          state_path=saved.get("state_path"))
    if saved:
        side.image_roots = tuple(Path(r) for r in saved.get("image_roots") or ())
        side.store.emit("notification", {"level": "info", "text": (
            f"Sim lab resumed from save point {saved['resumed']['checkpoint']} (lab event "
            f"{saved['last_event_id']}, code {saved.get('git_commit')}).")})
    side.info = {"seed": seed, "target_bundle": str(Path(target_bundle).resolve())
                 if target_bundle else None, "resumed": saved.get("resumed")}
    return side


def main(args: Any) -> None:
    side = build_side(args.db, args.seed, getattr(args, "target_bundle", None),
                      getattr(args, "resume", None), tuple(getattr(args, "accept_changed", ())))
    remote = {}
    if getattr(args, "remote_ports", False):
        remote = {m: p + 20000 for m, p in PORTS.items()}
    server = SimLabServer(side, host=args.host, motion_delay_s=args.motion_delay,
                          ports={"exfoliation": args.exfoliation_port,
                                 "stacking": args.stacking_port,
                                 "reactor": args.reactor_port},
                          remote_ports=remote, info=getattr(side, "info", None)).start()
    for module, url in server.urls.items():
        print(f"{module:12s} {url}api/describe", flush=True)
        if module == "reactor":
            print(f"{'':12s} {url}lab/  (reactor Lab Console API)", flush=True)
    for module, port in remote.items():
        print(f"{module:12s} http://127.0.0.1:{port}/  (remote: SSH tunnel endpoint)", flush=True)
    print(f"Simulated lab (demo: true), Hall-bar bench, seed {side.info['seed']}, "
          f"records in {Path(args.db).resolve()}. Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        side.jobs.shutdown(10.0) if hasattr(side.jobs, "shutdown") else None
        side.lab.devices.all_safe_state()
        side.store.close()
