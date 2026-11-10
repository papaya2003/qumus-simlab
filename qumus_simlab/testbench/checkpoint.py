"""Save points for long testbench recordings (save once after the search, resume later).

A full hallbar_device recording spends most of its wall time exfoliating and
searching chips (about 30-60 min per v1-size chip). When a fix only touches
what happens after the search (pairing, design, stacking), that part is the
same every time. A save point keeps the whole simulated lab and run store at
the moment the search is over, so a later recording can start from it:

- `CheckpointWriter` wraps the tool gateway. Just before the picker reads the
  flake candidates (`get_flake_candidates`), with no job open, it saves:
  the run store (SQLite backup), the model session store if any, the sim
  world (testbench/persist.py, every random stream included), how far the
  recorder timeline had got, and which source files decided the search.
- `restore_checkpoint` copies one into a fresh workspace. The new run's
  `build_services(..., state_path=...)` overlays the saved world.

Only the test harness uses this. The program's operations are unchanged: the
resumed run calls the same tools in the same order as an unbroken run would
from that point.

Photos (and the recorder timeline) from before the save point stay in the
original workspace: records point at the photos by absolute path and some
records are digested with those paths, so nothing is rewritten. Keep the original workspace while a resumed one is
in use; `export_recording(..., extra_roots=[...])` copies those photos into the
recording, which is self-contained afterwards.

A save point is only valid for code that would have made the same search. The
files in GUARDED decide exfoliation, search, recognition and the search camera;
`restore_checkpoint` refuses when any of them changed since the save, unless
the caller accepts the listed changes explicitly (e.g. a speedup proven to
give identical pictures and detections). The accepted list is kept with the
resumed run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qumus_simlab.testbench.persist import save_lab_state

VERSION = 1
REPO = Path(__file__).resolve().parents[2]
#: Source files whose change can alter what happened before the save point:
#: exfoliation, search and revisit flows, recognition, the sim world and its
#: flake population, the search camera, and the sim settings they read.
GUARDED = (
    "qumus_simlab/flows/library/exfoliation.py",
    "qumus_simlab/flows/library/search.py",
    "qumus_simlab/flows/library/place_chip.py",
    "qumus_simlab/flows/library/transfer.py",
    "qumus_simlab/flows/library/revisit_priority.py",
    "qumus_simlab/perception/_v1_rgb_core.py",
    "qumus_simlab/perception/_v1_rgb_materials.py",
    "qumus_simlab/perception/_v1_rgb_revisit.py",
    "qumus_simlab/perception/rgb_material.py",
    "qumus_simlab/perception/observations.py",
    "qumus_simlab/perception/v1_vision.py",
    "qumus_simlab/testbench/world.py",
    "qumus_simlab/testbench/material_flakes.py",
    "qumus_simlab/testbench/material_view.py",
    "qumus_simlab/testbench/material_adapter.py",
    "qumus_simlab/testbench/camera_render.py",
    "qumus_simlab/testbench/optics.py",
    "qumus_simlab/testbench/render.py",
    "qumus_simlab/testbench/lab.py",
    "qumus_simlab/devices/sim/devices.py",
    "qumus_simlab/config/testbench.yaml",
    "qumus_simlab/config/material-morphology.yaml",
)
#: Saved just before this tool: the search is over and pairing starts.
SAVE_BEFORE = frozenset({"get_flake_candidates"})
#: Job states that mean a flow may still move hardware or write records.
OPEN_JOB = frozenset({"queued", "running", "paused", "parked", "waiting_human"})
SESSIONS_SUFFIX = ".openai-sessions.db"


class StaleCheckpoint(RuntimeError):
    """The code that made the search changed since the save point."""

    def __init__(self, changed: list[str]) -> None:
        self.changed = changed
        super().__init__("source files that decide the search changed since this save point: "
                         + ", ".join(changed))


def guarded_fingerprint(root: Path = REPO, files: Iterable[str] = GUARDED) -> dict[str, str]:
    out = {}
    for rel in files:
        p = root / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"
    return out


def changed_files(saved: dict[str, str], now: dict[str, str]) -> list[str]:
    return sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=REPO).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _backup_sqlite(src: sqlite3.Connection, dst_path: Path) -> None:
    dst = sqlite3.connect(dst_path)
    try:
        src.backup(dst)
    finally:
        dst.close()


def open_jobs(store: Any) -> list[str]:
    return [r["id"] for r in store.run_statuses() if r["status"] in OPEN_JOB]


def save_checkpoint(services: Any, dest: Path, *, label: str = "", extra: dict[str, Any] | None = None
                    ) -> Path:
    """Write one save point of an idle in-process testbench run to `dest`."""
    store, lab = services.store, services.lab
    if getattr(services, "supervisor", None) is not None or not hasattr(lab, "world"):
        raise RuntimeError("save points need the in-process testbench (no separate lab process)")
    if store.path == ":memory:":
        raise RuntimeError("save points need a file run store")
    if busy := open_jobs(store):
        raise RuntimeError(f"jobs still open: {', '.join(busy)}")
    work = Path(store.path).resolve().parent
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with store._lock:  # no event lands between the ledger copy and the world copy
        last_id = store.last_event_id()
        _backup_sqlite(store._conn, tmp / "qumus.db")
        save_lab_state(lab, services.clock, tmp / "lab_state.pkl")
        sim_now = services.clock.now()
    sessions = Path(store.path).with_suffix(SESSIONS_SUFFIX)
    if sessions.exists():
        src = sqlite3.connect(sessions)
        try:
            _backup_sqlite(src, tmp / "qumus.openai-sessions.db")
        finally:
            src.close()
    # The recorder timeline only grows (and can be hundreds of MB): note its
    # length now and copy the prefix when resuming, from the kept workspace.
    timeline = work / "timeline.jsonl"
    meta = {"version": VERSION, "label": label, "workspace": str(work),
            "last_event_id": last_id,
            "timeline_bytes": timeline.stat().st_size if timeline.exists() else 0, "sim_time": sim_now, "saved_wall": time.time(),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "git_commit": _git_commit(),
            "agents": getattr(getattr(services, "runtime", None), "mode", None),
            "guarded": guarded_fingerprint(), **(extra or {})}
    (tmp / "checkpoint.json").write_text(json.dumps(meta, indent=1, default=str))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


class CheckpointWriter:
    """Save a point before each `SAVE_BEFORE` tool call while no job is open.

    The latest `keep` save points stay in `root` (older ones are removed: each
    holds a copy of the run store, about 200 MB for a full hallbar run). Never raises into the run: a failed save
    is printed and the run goes on.
    """

    def __init__(self, services: Any, root: str | Path, *, keep: int = 1,
                 extra: dict[str, Any] | None = None, log: Any = print) -> None:
        self.services, self.root, self.keep = services, Path(root), keep
        self.extra, self.log = extra or {}, log
        self.saved: list[Path] = []
        gateway = services.gateway
        inner = gateway.call

        async def call(agent: str, tool: str, args: dict[str, Any],
                       tool_use_id: str | None = None) -> Any:
            if tool in SAVE_BEFORE:
                self.maybe_save(f"before {agent}:{tool}")
            return await inner(agent, tool, args, tool_use_id)

        gateway.call = call

    def maybe_save(self, label: str) -> Path | None:
        store = self.services.store
        if open_jobs(store):
            return None
        last = store.last_event_id()
        name = f"cp{len(self.saved) + 1:02d}-ev{last}"
        try:
            t0 = time.time()
            path = save_checkpoint(self.services, self.root / name, label=label, extra=self.extra)
        except Exception as e:  # noqa: BLE001 - a save point must never stop the run
            self.log(f"save point {name} failed: {type(e).__name__}: {e}")
            return None
        self.saved.append(path)
        self.log(f"save point {path} ({time.time() - t0:.1f} s)")
        for old in self.saved[:-self.keep] if self.keep else []:
            shutil.rmtree(old, ignore_errors=True)
        self.saved = self.saved[-self.keep:] if self.keep else self.saved
        return path


def find_checkpoint(path: str | Path) -> Path:
    """A save point folder, or the newest one of a workspace (its `checkpoints/`)."""
    path = Path(path)
    if (path / "checkpoint.json").exists():
        return path
    found = sorted((p for p in (path / "checkpoints").glob("cp*") if (p / "checkpoint.json").exists()),
                   key=lambda p: json.loads((p / "checkpoint.json").read_text())["last_event_id"])
    if not found:
        raise FileNotFoundError(f"no save point in {path}")
    return found[-1]


def _copy_timeline(src: Path, dst: Path, size: int, last_id: int) -> None:
    """The source timeline's lines up to the save point (its first `size` bytes)."""
    if not size or not src.exists():
        return
    with src.open("rb") as f, dst.open("wb") as out:
        for line in f.read(size).splitlines(keepends=True):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line cut at the save moment
                continue
            if row.get("id", 0) <= last_id:
                out.write(line if line.endswith(b"\n") else line + b"\n")


def restore_checkpoint(checkpoint: str | Path, work: str | Path, *,
                       accept_changed: Iterable[str] = ()) -> dict[str, Any]:
    """Copy a save point into the fresh workspace `work`.

    Returns the save point's metadata plus `state_path` (pass it to
    build_services) and `photo_root` (the original workspace; pass it to
    export_recording as an extra root). Raises StaleCheckpoint when a guarded
    file changed and is not in `accept_changed` (use "all" to accept any).
    """
    cp = find_checkpoint(checkpoint)
    meta = json.loads((cp / "checkpoint.json").read_text())
    if meta.get("version") != VERSION:
        raise RuntimeError(f"save point version {meta.get('version')} is not supported")
    changed = changed_files(meta["guarded"], guarded_fingerprint())
    accepted = set(accept_changed)
    refused = [f for f in changed if "all" not in accepted and f not in accepted]
    if refused:
        raise StaleCheckpoint(refused)
    photo_root = Path(meta["workspace"])
    if not photo_root.is_dir():
        raise FileNotFoundError(f"the save point's photos are in {photo_root}, which is gone")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if (work / "qumus.db").exists():
        raise RuntimeError(f"{work} already has a run store; use a fresh workspace")
    shutil.copy2(cp / "qumus.db", work / "qumus.db")
    if (cp / "qumus.openai-sessions.db").exists():
        shutil.copy2(cp / "qumus.openai-sessions.db", (work / "qumus.db").with_suffix(SESSIONS_SUFFIX))
    shutil.copy2(cp / "lab_state.pkl", work / "lab_state.pkl")
    _copy_timeline(photo_root / "timeline.jsonl", work / "timeline.jsonl",
                   meta.get("timeline_bytes", 0), meta["last_event_id"])
    resumed = {"checkpoint": str(cp.resolve()), "changed_accepted": changed, **{
        k: meta.get(k) for k in ("workspace", "last_event_id", "saved_at", "git_commit", "label")}}
    (work / "resumed_from.json").write_text(json.dumps(resumed, indent=1))
    return {**meta, "resumed": resumed, "state_path": work / "lab_state.pkl",
            "photo_root": photo_root, "changed": changed}


def resume_scenario(scenario: dict[str, Any], *, from_agent: str = "picker",
                    say: str = "") -> dict[str, Any]:
    """The scripted scenario from its first delegation to `from_agent` on.

    The first turn's actions up to there (plan, exfoliation, search) are what
    the save point already holds; everything after it is unchanged.
    """
    turns = scenario.get("turns") or []
    if not turns:
        raise ValueError("scenario has no turns")
    actions = turns[0]["actions"]
    start = next((i for i, a in enumerate(actions)
                  if isinstance(a, dict) and (a.get("delegate") or {}).get("to") == from_agent), None)
    if start is None:
        raise ValueError(f"scenario never delegates to {from_agent}")
    head = [{"say": {"agent": "orchestrator", "text": say}}] if say else []
    first = {**turns[0], "actions": head + actions[start:]}
    return {**scenario, "name": f"{scenario.get('name', 'scenario')}_resumed",
            "turns": [first, *turns[1:]]}
