"""Run store: one SQLite file, the single source of truth (spec 2, 8.1).

CONTRACT. Tables:
  project        project state and the plan book (orchestrator's tools write it)
  plan_revision  every plan change with author and note (UI shows diffs)
  run            one job (a flow execution): status, params, result
  step           one closed-loop step of a run: checkpoint for resume
  instruction    what the human sent from the UI (messages, vetoes, stop)
  event          append-only timeline that the UI tails (see runtime/events.py)
  event_clock    comparable writer clock basis, saved atomically with new events
  device_lock    which run holds which device
  sample         "ledger of the physical world": where every chip/flake/stamp is
  heartbeat      liveness of long-running processes (lab process; supervisor reads it)
  meta           store-wide settings; `bench` = "sim" (testbench) or "real" (lab hardware);
                 `sessions` = where each `serve` began (a JSON list of event ids, oldest
                 first: a session is the events after its id up to the next one's)
  experience     what the lab learned (records/experience.py): claim, status, evidence
  experience_vote  one real run that supported or contradicted an experience

Runs carry the store's bench and a free-form `meta` (purpose, target, conditions
at submit; records/). Experience only ever comes from real runs (spec 10).

The SDK session transcript is only conversation history; anything needed to
resume lives here.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qumus_simlab.core.clock import Clock, RealClock
from qumus_simlab.core.types import new_id
from qumus_simlab.runtime.events import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS project (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL, updated_at REAL,
    state_json TEXT NOT NULL DEFAULT '{}', plan_json TEXT NOT NULL DEFAULT '[]',
    plan_revision INTEGER NOT NULL DEFAULT 0, session_id TEXT
);
CREATE TABLE IF NOT EXISTS plan_revision (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, revision INTEGER, ts REAL,
    author TEXT, plan_json TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS run (
    id TEXT PRIMARY KEY, project_id TEXT, flow TEXT NOT NULL, params_json TEXT,
    status TEXT NOT NULL, created_at REAL, updated_at REAL,
    result_json TEXT, error_json TEXT, requested_by TEXT
);
CREATE TABLE IF NOT EXISTS step (
    run_id TEXT NOT NULL, seq INTEGER NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL,
    expected_json TEXT, measurements_json TEXT, corrections INTEGER DEFAULT 0,
    safe_to_resume INTEGER DEFAULT 1, verifiable INTEGER DEFAULT 1, note TEXT,
    started_at REAL, finished_at REAL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS instruction (
    id TEXT PRIMARY KEY, ts REAL, project_id TEXT, source TEXT, kind TEXT, text TEXT,
    status TEXT NOT NULL, note TEXT
);
CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT NOT NULL, run_id TEXT,
    agent TEXT, payload_json TEXT
);
CREATE TABLE IF NOT EXISTS event_clock (
    event_id INTEGER PRIMARY KEY REFERENCES event(id),
    clock_basis TEXT NOT NULL, clock_domain TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS event_clock_domain ON event_clock(clock_domain, event_id);
CREATE TABLE IF NOT EXISTS device_lock (
    device_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, acquired_at REAL
);
CREATE TABLE IF NOT EXISTS sample (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, location TEXT, status TEXT,
    props_json TEXT, updated_at REAL
);
CREATE TABLE IF NOT EXISTS heartbeat (
    name TEXT PRIMARY KEY, pid INTEGER, ts REAL, status TEXT, info_json TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS reservation (
    kind TEXT NOT NULL, key TEXT NOT NULL, run_id TEXT NOT NULL, flow TEXT, note TEXT,
    created_at REAL, PRIMARY KEY (kind, key)
);
CREATE TABLE IF NOT EXISTS experience (
    id TEXT PRIMARY KEY, created_at REAL, updated_at REAL, topic TEXT NOT NULL,
    scope_json TEXT, claim TEXT NOT NULL, recipe_json TEXT, expect_json TEXT,
    source TEXT NOT NULL, status TEXT NOT NULL, detail_json TEXT, retired_reason TEXT,
    sandbox INTEGER NOT NULL DEFAULT 0, author TEXT
);
CREATE TABLE IF NOT EXISTS experience_vote (
    experience_id TEXT NOT NULL, run_id TEXT NOT NULL, verdict TEXT NOT NULL, ts REAL,
    author TEXT, batch TEXT, note TEXT, PRIMARY KEY (experience_id, run_id)
);
CREATE INDEX IF NOT EXISTS event_run ON event(run_id);
"""

RUN_STATUSES = {"queued", "running", "waiting_human", "parked", "paused", "succeeded",
                "failed", "cancelled"}
STEP_STATUSES = {"in_progress", "done", "failed", "skipped"}
INSTRUCTION_KINDS = {"message", "plan_change", "veto", "stop", "continue"}
BENCHES = {"sim", "real"}
EXPERIENCE_STATUSES = {"trial", "confirmed", "retired"}
EXPERIENCE_SOURCES = {"review", "human", "prior"}
#: sample props a `sample.updated` event carries (what the Lab page's inventory and the
#: flake library show). Flake photos (`images`: {"20x": path, "100x": path}; a device's is
#: a list) ride along, so a recording keeps every flake's photos. Never `sim_id`: that is
#: the testbench's ground truth.
SAMPLE_EVENT_PROPS = ("image_source", "image_description", "material", "film", "stack", "layers_top_first", "materials", "device",
                      "uses", "roll_left_mm", "location_uncertain", "thickness_label",
                      "electrodes", "electrode_geometry", "film_reports", "material_reviews", "simulation_identity",
                      "chip_id", "labels", "area_um2",
                      "confidence", "stage_xy", "rel_corner_mm", "images", "residue",
                      "recognition_20x", "recognition_100x", "recognition_event_id",
                      "recognition_status", "recognition_reason", "mask_20x", "geometry_mask_20x",
                      "reactor_substrate", "size_mm", "preparation", "reactor_home_slot",
                      "reactor_result", "reactor_state_uncertain", "reactor_pair",
                      "support", "support_history", "metal_coating", "metal_geometry", "optical_geometry", "pickup_assessment", "material_metal_preparation", "material_metal_observation",
                      "source_flake_ids", "reactor_preparation_id", "reactor_evidence_job_id", "reactor_pickup_assessment",
                      "measured_area_um2", "geometry_area_um2",
                      "reactor_consumed_by", "material_placement", "material_state",
                      "user_reference_20x")
#: columns added after the first release; old database files get them on open
_RUN_COLUMNS = {"bench": "TEXT", "meta_json": "TEXT"}


def _j(v: Any) -> str:
    return json.dumps(v, default=str)


def _uj(s: str | None, default: Any = None) -> Any:
    return json.loads(s) if s else default


class _Rows:
    """A statement's rows, read while the store lock is held. The connection is
    shared by threads and sqlite3 reuses a cached statement for the same SQL, so
    a cursor fetched after the lock is released can return another thread's rows
    (or none)."""

    __slots__ = ("_rows", "lastrowid", "rowcount")

    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._rows = cur.fetchall()
        self.lastrowid, self.rowcount = cur.lastrowid, cur.rowcount

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._rows)


class PlanConflict(ValueError):
    """A Task Plan was changed after the caller read it."""


class RunStore:
    """`bench`: "sim" for the testbench, "real" for lab hardware. A new
    database records it once; opening it later with a different bench raises,
    so testbench runs can never mix into a real lab's records. Default "sim".

    `experience_sandbox` (tests only, never persisted): experience may cite sim
    runs; such rows are marked sandbox and never shown as lab experience."""

    def __init__(self, path: str | Path = ":memory:", clock: Clock | None = None, *,
                 bench: str | None = None, experience_sandbox: bool = False) -> None:
        self.path = str(path)
        self.clock = clock or RealClock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self.bench = self._init_bench(bench)
        self.experience_sandbox = experience_sandbox

    def _migrate(self) -> None:
        cols = {r["name"] for r in self._x("PRAGMA table_info(run)").fetchall()}
        for col, typ in _RUN_COLUMNS.items():
            if col not in cols:
                self._x(f"ALTER TABLE run ADD COLUMN {col} {typ}")
        with self._transaction():
            cols = {r["name"] for r in self._x("PRAGMA table_info(project)")}
            if "plan_version" not in cols:
                self._x("ALTER TABLE project ADD COLUMN plan_version INTEGER NOT NULL DEFAULT 0")
            # Refuse ambiguous legacy history rather than deleting or rewriting evidence.
            self._x("CREATE UNIQUE INDEX IF NOT EXISTS plan_revision_unique "
                    "ON plan_revision(project_id, revision)")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Lock before reading, across threads and independent SQLite connections."""
        with self._lock:
            if self._conn.in_transaction:
                # A compound checked write may call another atomic store API.
                # Savepoints preserve rollback boundaries without releasing the
                # outer connection/thread lock or committing a partial plan.
                self._conn.execute("SAVEPOINT nested_write")
                try:
                    yield
                    self._conn.execute("RELEASE SAVEPOINT nested_write")
                except BaseException:
                    self._conn.execute("ROLLBACK TO SAVEPOINT nested_write")
                    self._conn.execute("RELEASE SAVEPOINT nested_write")
                    raise
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def _init_bench(self, bench: str | None) -> str:
        if bench is not None and bench not in BENCHES:
            raise ValueError(f"bench must be one of {sorted(BENCHES)}")
        with self._lock:
            stored = self.get_meta("bench")
            if stored is None:
                stored = bench or "sim"
                self._x("INSERT OR IGNORE INTO meta (key, value) VALUES ('bench', ?)", (stored,))
                stored = self.get_meta("bench") or stored
        if bench is not None and bench != stored:
            raise ValueError(f"{self.path} is a {stored!r} run store; refusing to open it as "
                             f"{bench!r} (testbench and lab records never mix)")
        return stored

    def get_meta(self, key: str) -> str | None:
        r = self._x("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r["value"] if r else None

    def set_meta(self, key: str, value: str) -> None:
        self._x("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def sessions(self) -> list[int]:
        """Where each session (one `serve`) began: the id of the last event
        before it, oldest first (see begin_session)."""
        try:
            return sorted({int(x) for x in json.loads(self.get_meta("sessions") or "[]")})
        except (TypeError, ValueError):
            return []

    def begin_session(self, after_id: int) -> None:
        """A new session starts after event `after_id` (the UI's "this run";
        earlier sessions are its History)."""
        with self._lock:
            ids = self.sessions()
            if after_id not in ids:
                self.set_meta("sessions", json.dumps(sorted([*ids, int(after_id)])))

    def close(self) -> None:
        self._conn.close()

    def _x(self, sql: str, args: Iterable[Any] = ()) -> _Rows:
        with self._lock:
            return _Rows(self._conn.execute(sql, tuple(args)))

    # ----- events ----------------------------------------------------------

    def emit(self, kind: str, payload: dict[str, Any] | None = None, *,
             run_id: str | None = None, agent: str | None = None) -> int:
        # Preserve the public event envelope and historical rows. Unknown custom
        # clocks remain untagged unless they explicitly declare a comparable basis.
        basis = getattr(self.clock, "event_clock_basis", None)
        domain = getattr(self.clock, "event_clock_domain", basis)
        with self._transaction():
            cur = self._x("INSERT INTO event (ts, kind, run_id, agent, payload_json) VALUES (?,?,?,?,?)",
                          (self.clock.now(), kind, run_id, agent, _j(payload or {})))
            event_id = int(cur.lastrowid or 0)
            if all(isinstance(v, str) and 0 < len(v) <= 256 for v in (basis, domain)):
                self._x("INSERT INTO event_clock (event_id, clock_basis, clock_domain) VALUES (?,?,?)",
                        (event_id, basis, domain))
            return event_id

    def spent_usd(self) -> float:
        """Recorded model cost (OpenAI token-rate estimates, legacy SDK totals).

        Per-response usage survives interrupted turns; skip duplicate turn totals.
        """
        rows = self._x("SELECT kind, payload_json FROM event "
                       "WHERE kind IN ('agent.turn_end', 'agent.usage')").fetchall()
        total = 0.0
        for row in rows:
            p = _uj(row["payload_json"], {})
            if row["kind"] == "agent.turn_end" and p.get("cost_accounted_by_usage"):
                continue
            total += float(p.get("cost_usd") or 0)
        return total

    def events_after(self, after_id: int = 0, limit: int = 500,
                     run_id: str | None = None) -> list[Event]:
        sql = "SELECT * FROM event WHERE id > ?"
        args: list[Any] = [after_id]
        if run_id:
            sql += " AND run_id = ?"
            args.append(run_id)
        sql += " ORDER BY id LIMIT ?"
        args.append(limit)
        return [Event(id=r["id"], ts=r["ts"], kind=r["kind"], run_id=r["run_id"],
                      agent=r["agent"], payload=_uj(r["payload_json"], {}))
                for r in self._x(sql, args).fetchall()]

    def run_ids_since(self, after_id: int) -> set[str]:
        """The jobs that wrote an event after `after_id` (this session's jobs)."""
        rows = self._x("SELECT DISTINCT run_id FROM event WHERE id > ? AND run_id IS NOT NULL",
                       (after_id,)).fetchall()
        return {r["run_id"] for r in rows}

    def last_event_id(self) -> int:
        row = self._x("SELECT MAX(id) AS m FROM event").fetchone()
        return int(row["m"] or 0)

    def event_clock_readings(self, through_event_id: int) -> dict[str, float]:
        """Saved comparable readings at a prefix, selecting one epoch per domain."""
        rows = self._x("SELECT c.clock_basis, e.ts FROM event_clock c JOIN event e ON e.id=c.event_id "
                       "JOIN (SELECT MAX(event_id) id FROM event_clock WHERE event_id<=? "
                       "GROUP BY clock_domain) latest ON latest.id=c.event_id",
                       (through_event_id,))
        return {r["clock_basis"]: r["ts"] for r in rows}

    def instrument_observations(self) -> list[dict[str, Any]]:
        """Stored frames, with elapsed age only for a comparable reader clock.

        Unknown/retired clock bases and future timestamps have unknown age.
        Reading a saved frame never establishes fresh sensing or readiness.
        """
        through = self.last_event_id()
        rows = self._x("SELECT e.*, c.clock_basis FROM event e LEFT JOIN event_clock c ON c.event_id=e.id "
                       "JOIN (SELECT MAX(id) id FROM event WHERE kind='lab.frame' AND id<=? "
                       "GROUP BY json_extract(payload_json, '$.device')) latest ON e.id=latest.id "
                       "ORDER BY e.id", (through,)).fetchall()
        readings = self.event_clock_readings(through)
        caller_basis = getattr(self.clock, "event_clock_basis", None)
        now = self.clock.now() if isinstance(caller_basis, str) and caller_basis in readings else None
        observations = []
        for row in rows:
            basis = row["clock_basis"]
            age = now-row["ts"] if now is not None and basis == caller_basis else None
            if age is not None and (not math.isfinite(age) or age < 0):
                age = None
            observations.append({"event_id": row["id"], "observed_at": row["ts"],
                "age_s": age, "age_basis": "comparable_reader_clock" if age is not None else "unknown",
                "clock_basis": basis, "fresh_measurement": False,
                "frame": _uj(row["payload_json"], {})})
        return observations

    # ----- projects and plan book -------------------------------------------

    def create_project(self, name: str, project_id: str | None = None) -> str:
        pid = project_id or new_id("prj")
        now = self.clock.now()
        self._x("INSERT INTO project (id, name, created_at, updated_at) VALUES (?,?,?,?)",
                (pid, name, now, now))
        return pid

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        r = self._x("SELECT * FROM project WHERE id = ?", (project_id,)).fetchone()
        if r is None:
            return None
        return {"id": r["id"], "name": r["name"], "state": _uj(r["state_json"], {}),
                "plan": _uj(r["plan_json"], []), "plan_revision": r["plan_revision"],
                "plan_version": r["plan_version"],
                "session_id": r["session_id"], "updated_at": r["updated_at"]}

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self._x("SELECT id FROM project ORDER BY updated_at DESC").fetchall()
        return [p for r in rows if (p := self.get_project(r["id"])) is not None]

    def update_project_state(self, project_id: str, **fields: Any) -> None:
        proj = self.get_project(project_id)
        if proj is None:
            raise KeyError(project_id)
        state = {**proj["state"], **fields}
        self._x("UPDATE project SET state_json = ?, updated_at = ? WHERE id = ?",
                (_j(state), self.clock.now(), project_id))

    def set_session_id(self, project_id: str, session_id: str) -> None:
        self._x("UPDATE project SET session_id = ? WHERE id = ?", (session_id, project_id))

    def set_plan(self, project_id: str, plan: list[dict[str, Any]], *, author: str,
                 note: str = "", expected_version: int | None = None,
                 preserve_progress_ids: set[str] | None = None) -> int:
        plan = [dict(s) for s in plan]
        ids = [s.get("id") for s in plan]
        if len(ids) != len(set(ids)):
            raise ValueError("Task Plan step IDs must be unique")
        with self._transaction():
            proj = self.get_project(project_id)
            if proj is None:
                raise KeyError(project_id)
            if expected_version is not None and proj["plan_version"] != expected_version:
                raise PlanConflict("Task Plan changed; read get_project_context and retry "
                                   "with its plan_version")
            if preserve_progress_ids is not None:
                before = {(s.get("id"), s.get("owner")): s for s in proj["plan"]}
                for step in plan:
                    old = before.get((step.get("id"), step.get("owner")))
                    if step.get("status", "pending") == "pending" and old and (
                            step["id"] in preserve_progress_ids
                            or old.get("title", "").lower() == step.get("title", "").lower()) and (
                            (old.get("sample_outcomes") or []) == (step.get("sample_outcomes") or [])):
                        step["status"] = old.get("status") or "pending"
            rev = proj["plan_revision"] + 1
            version = proj["plan_version"] + 1
            now = self.clock.now()
            self._x("UPDATE project SET plan_json = ?, plan_revision = ?, plan_version = ?, "
                    "updated_at = ? WHERE id = ?", (_j(plan), rev, version, now, project_id))
            self._x("INSERT INTO plan_revision (project_id, revision, ts, author, plan_json, note)"
                    " VALUES (?,?,?,?,?,?)", (project_id, rev, now, author, _j(plan), note))
            self.emit("plan.updated", {"project_id": project_id, "revision": rev, "plan": plan,
                                       "version": version, "note": note}, agent=author)
        return rev

    def set_step_status(self, project_id: str, step_id: str, status: str, *,
                        agent: str | None = None, expected_version: int | None = None) -> bool:
        """Change one plan step's status in place (no new revision: the plan's
        content did not change). The latest revision's record is updated too,
        so the history shows progress. False if there is no such step."""
        with self._transaction():
            proj = self.get_project(project_id)
            if proj is None:
                raise KeyError(project_id)
            if expected_version is not None and proj["plan_version"] != expected_version:
                raise PlanConflict("Task Plan changed before the progress update")
            plan = proj["plan"]
            step = next((s for s in plan if s.get("id") == step_id), None)
            if step is None or step.get("status") == status:
                return False
            step["status"] = status
            rev = proj["plan_revision"]
            version = proj["plan_version"] + 1
            self._x("UPDATE project SET plan_json = ?, plan_version = ?, updated_at = ? "
                    "WHERE id = ?", (_j(plan), version, self.clock.now(), project_id))
            changed = self._x("UPDATE plan_revision SET plan_json = ? WHERE project_id = ? "
                              "AND revision = ?", (_j(plan), project_id, rev))
            if changed.rowcount != 1:
                raise PlanConflict("Task Plan revision missing; progress was not changed")
            self.emit("plan.step", {"project_id": project_id, "revision": rev, "step_id": step_id,
                                    "version": version, "status": status, "plan": plan}, agent=agent)
        return True

    def plan_history(self, project_id: str) -> list[dict[str, Any]]:
        rows = self._x("SELECT * FROM plan_revision WHERE project_id = ? ORDER BY revision",
                       (project_id,)).fetchall()
        return [{"revision": r["revision"], "ts": r["ts"], "author": r["author"],
                 "plan": _uj(r["plan_json"], []), "note": r["note"]} for r in rows]

    # ----- runs (jobs) -----------------------------------------------------

    def create_run(self, flow: str, params: dict[str, Any], *, project_id: str | None = None,
                   requested_by: str | None = None, run_id: str | None = None) -> str:
        rid = run_id or new_id("run")
        now = self.clock.now()
        self._x("INSERT INTO run (id, project_id, flow, params_json, status, created_at, "
                "updated_at, requested_by, bench, meta_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (rid, project_id, flow, _j(params), "queued", now, now, requested_by,
                 self.bench, _j({})))
        self.emit("job.submitted", {"flow": flow, "params": params}, run_id=rid,
                  agent=requested_by)
        return rid

    def set_run_status(self, run_id: str, status: str, *, result: Any = None,
                       error: Any = None, message: str = "") -> None:
        if status not in RUN_STATUSES:
            raise ValueError(status)
        self._x("UPDATE run SET status = ?, updated_at = ?, result_json = COALESCE(?, result_json),"
                " error_json = COALESCE(?, error_json) WHERE id = ?",
                (status, self.clock.now(), _j(result) if result is not None else None,
                 _j(error) if error is not None else None, run_id))
        if status in ("succeeded", "failed", "cancelled"):
            self.release_reservations(run_id)  # the run is over: its bookings lapse
        self.emit("job.status", {"status": status, "message": message}, run_id=run_id)

    def update_run_params(self, run_id: str, path: list[Any], value: Any) -> None:
        """Set one entry of the run's resolved params (`path`: keys / list indexes),
        e.g. a store slot re-chosen at run time, so that a resume rebuilds the
        same steps (jobs.execute builds them from the stored params)."""
        with self._lock:
            r = self._x("SELECT params_json FROM run WHERE id = ?", (run_id,)).fetchone()
            if r is None:
                raise KeyError(run_id)
            params = _uj(r["params_json"], {})
            node = params
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
            self._x("UPDATE run SET params_json = ?, updated_at = ? WHERE id = ?",
                    (_j(params), self.clock.now(), run_id))
        self.emit("job.params_changed", {"path": [str(k) for k in path], "value": value},
                  run_id=run_id)

    def set_run_params(self, run_id: str, params: dict[str, Any], note: str = "") -> int:
        """Replace the run's resolved params in one write: a flow that changes its
        plan at run time (Flow.on_step_failed) records the new plan atomically,
        so a resume rebuilds exactly the steps that were planned.

        Return the existing change event's ID for a checked workflow link.
        """
        with self._lock:
            if self._x("SELECT id FROM run WHERE id = ?", (run_id,)).fetchone() is None:
                raise KeyError(run_id)
            self._x("UPDATE run SET params_json = ?, updated_at = ? WHERE id = ?",
                    (_j(params), self.clock.now(), run_id))
        return self.emit("job.params_changed", {"path": [], "value": params, "note": note},
                         run_id=run_id)

    # ----- reservations -----------------------------------------------------
    # A queued job books what it will need (a storage slot it will put a chip
    # in, a bare chip or stamp it will use) at submit, so a second job
    # submitted while it is queued or running does not plan on the same thing.
    # Released when the run ends (set_run_status) or when the step that used
    # the booking is done.

    def reserve(self, kind: str, key: str, run_id: str, flow: str = "", note: str = "") -> None:
        """`kind`: "slot" (a bench slot name) or "sample" (a sample id)."""
        self._x("INSERT INTO reservation (kind, key, run_id, flow, note, created_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(kind, key) DO UPDATE SET run_id=excluded.run_id,"
                " flow=excluded.flow, note=excluded.note, created_at=excluded.created_at",
                (kind, key, run_id, flow, note, self.clock.now()))

    def reservations(self, kind: str | None = None, exclude_run: str | None = None
                     ) -> dict[str, dict[str, Any]]:
        """{key: {kind, run_id, flow, note}}, without `exclude_run`'s own bookings."""
        rows = self._x("SELECT * FROM reservation").fetchall()
        return {r["key"]: {"kind": r["kind"], "run_id": r["run_id"], "flow": r["flow"],
                           "note": r["note"]}
                for r in rows if (kind is None or r["kind"] == kind)
                and r["run_id"] != exclude_run}

    def release_reservations(self, run_id: str, kind: str | None = None,
                             keys: list[str] | None = None) -> None:
        sql, args = "DELETE FROM reservation WHERE run_id = ?", [run_id]
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        if keys is not None:
            sql += f" AND key IN ({','.join('?' * len(keys))})"
            args += keys
        self._x(sql, tuple(args))

    def occupied_slots(self, exclude_run: str | None = None) -> set[str]:
        """Slots a sample sits in or a queued/running job has booked."""
        used = {x["location"] for x in self.list_samples() if x["location"]}
        return used | set(self.reservations("slot", exclude_run))

    def reserved_samples(self, exclude_run: str | None = None) -> set[str]:
        return set(self.reservations("sample", exclude_run))

    def update_run_meta(self, run_id: str, **fields: Any) -> None:
        """Merge fields into the run's meta (purpose, target, conditions, review)."""
        with self._lock:
            r = self._x("SELECT meta_json FROM run WHERE id = ?", (run_id,)).fetchone()
            if r is None:
                raise KeyError(run_id)
            meta = {**_uj(r["meta_json"], {}), **fields}
            self._x("UPDATE run SET meta_json = ? WHERE id = ?", (_j(meta), run_id))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        r = self._x("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        return self._run_row(r) if r else None

    def list_runs(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._x("SELECT * FROM run WHERE status = ? ORDER BY created_at", (status,))
        else:
            rows = self._x("SELECT * FROM run ORDER BY created_at")
        return [self._run_row(r) for r in rows.fetchall()]

    def run_statuses(self) -> list[dict[str, Any]]:
        """id, project, flow and status of every run, without parsing params/meta
        (a search run's meta is megabytes; busy checks only need the status)."""
        rows = self._x("SELECT id, project_id, flow, status FROM run ORDER BY created_at")
        return [dict(r) for r in rows.fetchall()]

    @staticmethod
    def _run_row(r: sqlite3.Row) -> dict[str, Any]:
        return {"id": r["id"], "project_id": r["project_id"], "flow": r["flow"],
                "params": _uj(r["params_json"], {}), "status": r["status"],
                "created_at": r["created_at"], "updated_at": r["updated_at"],
                "result": _uj(r["result_json"]), "error": _uj(r["error_json"]),
                "requested_by": r["requested_by"], "bench": r["bench"] or "sim",
                "meta": _uj(r["meta_json"], {}) or {}}

    # ----- steps (checkpoints) ---------------------------------------------

    def begin_step(self, run_id: str, seq: int, name: str, *, expected: Any = None,
                   safe_to_resume: bool = True, verifiable: bool = True) -> None:
        self._x("INSERT INTO step (run_id, seq, name, status, expected_json, safe_to_resume, "
                "verifiable, started_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id, seq) DO UPDATE SET status='in_progress', "
                "started_at=excluded.started_at",
                (run_id, seq, name, "in_progress", _j(expected), int(safe_to_resume),
                 int(verifiable), self.clock.now()))
        self.emit("step.started", {"seq": seq, "name": name, "expected": expected},
                  run_id=run_id)

    def finish_step(self, run_id: str, seq: int, status: str, *, measurements: Any = None,
                    corrections: int = 0, note: str = "") -> None:
        if status not in STEP_STATUSES:
            raise ValueError(status)
        self._x("UPDATE step SET status = ?, measurements_json = ?, corrections = ?, note = ?, "
                "finished_at = ? WHERE run_id = ? AND seq = ?",
                (status, _j(measurements), corrections, note, self.clock.now(), run_id, seq))
        r = self._x("SELECT name FROM step WHERE run_id = ? AND seq = ?", (run_id, seq)).fetchone()
        self.emit("step.finished", {"seq": seq, "name": r["name"] if r else "", "status": status,
                                    "corrections": corrections, "note": note}, run_id=run_id)

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._x("SELECT * FROM step WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
        return [{"seq": r["seq"], "name": r["name"], "status": r["status"],
                 "expected": _uj(r["expected_json"]), "measurements": _uj(r["measurements_json"]),
                 "corrections": r["corrections"], "safe_to_resume": bool(r["safe_to_resume"]),
                 "verifiable": bool(r["verifiable"]), "note": r["note"],
                 "started_at": r["started_at"], "finished_at": r["finished_at"]} for r in rows]

    # ----- device locks ----------------------------------------------------

    def acquire_locks(self, device_ids: Iterable[str], run_id: str) -> bool:
        """Atomically lock all devices for run_id, or none. Re-entrant per run."""
        ids = list(device_ids)
        with self._lock:
            self._x("BEGIN IMMEDIATE")
            try:
                for d in ids:
                    r = self._x("SELECT run_id FROM device_lock WHERE device_id = ?",
                                (d,)).fetchone()
                    if r and r["run_id"] != run_id:
                        self._x("ROLLBACK")
                        return False
                now = self.clock.now()
                for d in ids:
                    self._x("INSERT OR REPLACE INTO device_lock VALUES (?,?,?)", (d, run_id, now))
                self._x("COMMIT")
                return True
            except Exception:
                self._x("ROLLBACK")
                raise

    def release_locks(self, run_id: str) -> None:
        self._x("DELETE FROM device_lock WHERE run_id = ?", (run_id,))

    def locks(self) -> dict[str, str]:
        return {r["device_id"]: r["run_id"]
                for r in self._x("SELECT * FROM device_lock").fetchall()}

    # ----- samples (physical-world ledger) ---------------------------------

    def upsert_sample(self, sample_id: str, kind: str, *, location: str | None = None,
                      status: str | None = None, props: dict[str, Any] | None = None) -> None:
        with self._lock:  # read-merge-write as one: concurrent updates keep each other's props
            self._upsert_sample(sample_id, kind, location=location, status=status, props=props)

    def _upsert_sample(self, sample_id: str, kind: str, *, location: str | None,
                       status: str | None, props: dict[str, Any] | None) -> None:
        old = self.get_sample(sample_id)
        merged = {**(old["props"] if old else {}), **(props or {})}
        self._x("INSERT INTO sample (id, kind, location, status, props_json, updated_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, "
                "location=COALESCE(excluded.location, location), "
                "status=COALESCE(excluded.status, status), props_json=excluded.props_json, "
                "updated_at=excluded.updated_at",
                (sample_id, kind, location, status, _j(merged), self.clock.now()))
        # the ledger as events, so a replay can rebuild the inventory (Lab page);
        # only the fields the inventory shows, not every flake measurement
        self.emit("sample.updated", {
            "id": sample_id, "kind": kind,
            "location": location if location is not None else (old or {}).get("location"),
            "status": status if status is not None else (old or {}).get("status"),
            "props": {k: merged[k] for k in SAMPLE_EVENT_PROPS if merged.get(k) is not None}})

    def remove_sample(self, sample_id: str, *, reason: str = "", emit: bool = True) -> None:
        """Take a sample out of the ledger (its history stays in the events).
        `emit=False` when a `sample.updated` of the same id follows at once."""
        old = self.get_sample(sample_id)
        if old is None:
            return
        self._x("DELETE FROM sample WHERE id = ?", (sample_id,))
        if emit:
            self.emit("sample.removed", {"id": sample_id, "kind": old["kind"], "reason": reason})

    def get_sample(self, sample_id: str) -> dict[str, Any] | None:
        r = self._x("SELECT * FROM sample WHERE id = ?", (sample_id,)).fetchone()
        if r is None:
            return None
        return {"id": r["id"], "kind": r["kind"], "location": r["location"],
                "status": r["status"], "props": _uj(r["props_json"], {}),
                "updated_at": r["updated_at"]}

    def list_samples(self, kind: str | None = None,
                     location: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT id FROM sample WHERE 1=1", []
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if location:
            sql += " AND location = ?"
            args.append(location)
        rows = self._x(sql + " ORDER BY id", args).fetchall()
        return [s for r in rows if (s := self.get_sample(r["id"])) is not None]

    # ----- human instructions (from the UI) --------------------------------

    def add_instruction(self, text: str, *, kind: str = "message", source: str = "ui",
                        project_id: str | None = None) -> str:
        if kind not in INSTRUCTION_KINDS:
            raise ValueError(kind)
        iid = new_id("ins")
        self._x("INSERT INTO instruction (id, ts, project_id, source, kind, text, status) "
                "VALUES (?,?,?,?,?,?,?)",
                (iid, self.clock.now(), project_id, source, kind, text, "pending"))
        self.emit("user.message", {"text": text, "kind": kind, "instruction_id": iid},
                  agent="user")
        return iid

    def pending_instructions(self) -> list[dict[str, Any]]:
        rows = self._x("SELECT * FROM instruction WHERE status = 'pending' ORDER BY ts, rowid")
        return [dict(r) for r in rows.fetchall()]

    def get_instruction(self, instruction_id: str) -> dict[str, Any] | None:
        r = self._x("SELECT * FROM instruction WHERE id = ?", (instruction_id,)).fetchone()
        return dict(r) if r else None

    def mark_instruction(self, instruction_id: str, status: str, note: str = "") -> None:
        self._x("UPDATE instruction SET status = ?, note = ? WHERE id = ?",
                (status, note, instruction_id))

    # ----- process heartbeats (spec 5.4) -----------------------------------

    def beat(self, name: str, *, pid: int, ts: float, status: str = "running",
             info: dict[str, Any] | None = None) -> None:
        """`ts` is wall-clock time (a heartbeat measures real liveness, so the
        caller passes RealClock time even when the store's clock is simulated)."""
        self._x("INSERT INTO heartbeat (name, pid, ts, status, info_json) VALUES (?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET pid=excluded.pid, ts=excluded.ts, "
                "status=excluded.status, info_json=excluded.info_json",
                (name, pid, ts, status, _j(info or {})))

    def get_heartbeat(self, name: str) -> dict[str, Any] | None:
        r = self._x("SELECT * FROM heartbeat WHERE name = ?", (name,)).fetchone()
        if r is None:
            return None
        return {"name": r["name"], "pid": r["pid"], "ts": r["ts"], "status": r["status"],
                "info": _uj(r["info_json"], {})}

    # ----- experience (records/experience.py holds the rules) ----------------

    def insert_experience(self, *, topic: str, claim: str, source: str, status: str,
                          scope: dict[str, Any] | None = None,
                          recipe: dict[str, Any] | None = None,
                          expect: dict[str, Any] | None = None,
                          detail: dict[str, Any] | None = None, author: str | None = None,
                          sandbox: bool = False, experience_id: str | None = None) -> str:
        if source not in EXPERIENCE_SOURCES or status not in EXPERIENCE_STATUSES:
            raise ValueError(f"bad experience source/status {source!r}/{status!r}")
        eid = experience_id or new_id("exp")
        now = self.clock.now()
        self._x("INSERT INTO experience (id, created_at, updated_at, topic, scope_json, claim, "
                "recipe_json, expect_json, source, status, detail_json, retired_reason, sandbox, "
                "author) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (eid, now, now, topic, _j(scope or {}), claim, _j(recipe) if recipe else None,
                 _j(expect) if expect else None, source, status, _j(detail or {}), "",
                 int(sandbox), author))
        return eid

    def set_experience_status(self, experience_id: str, status: str, reason: str = "") -> None:
        if status not in EXPERIENCE_STATUSES:
            raise ValueError(status)
        self._x("UPDATE experience SET status = ?, updated_at = ?, retired_reason = "
                "CASE WHEN ? = 'retired' THEN ? ELSE retired_reason END WHERE id = ?",
                (status, self.clock.now(), status, reason, experience_id))

    def get_experience(self, experience_id: str) -> dict[str, Any] | None:
        r = self._x("SELECT * FROM experience WHERE id = ?", (experience_id,)).fetchone()
        return self._experience_row(r) if r else None

    def list_experiences(self, *, sandbox: bool | None = False) -> list[dict[str, Any]]:
        """sandbox=False: lab experience only; True: sandbox rows only; None: all."""
        sql, args = "SELECT * FROM experience", []
        if sandbox is not None:
            sql += " WHERE sandbox = ?"
            args.append(int(sandbox))
        rows = self._x(sql + " ORDER BY created_at, rowid", args).fetchall()
        return [self._experience_row(r) for r in rows]

    def add_vote(self, experience_id: str, run_id: str, verdict: str, *, author: str,
                 batch: str, note: str = "") -> bool:
        """One vote per run and experience; False if this run already voted."""
        if verdict not in ("support", "against"):
            raise ValueError(verdict)
        cur = self._x("INSERT OR IGNORE INTO experience_vote (experience_id, run_id, verdict, ts,"
                      " author, batch, note) VALUES (?,?,?,?,?,?,?)",
                      (experience_id, run_id, verdict, self.clock.now(), author, batch, note))
        if cur.rowcount:
            self._x("UPDATE experience SET updated_at = ? WHERE id = ?",
                    (self.clock.now(), experience_id))
        return bool(cur.rowcount)

    def votes(self, experience_id: str) -> list[dict[str, Any]]:
        rows = self._x("SELECT * FROM experience_vote WHERE experience_id = ? ORDER BY ts, rowid",
                       (experience_id,)).fetchall()
        return [dict(r) for r in rows]

    def _experience_row(self, r: sqlite3.Row) -> dict[str, Any]:
        votes = self.votes(r["id"])
        return {"id": r["id"], "created_at": r["created_at"], "updated_at": r["updated_at"],
                "topic": r["topic"], "scope": _uj(r["scope_json"], {}), "claim": r["claim"],
                "recipe": _uj(r["recipe_json"]), "expect": _uj(r["expect_json"]),
                "source": r["source"], "status": r["status"],
                "detail": _uj(r["detail_json"], {}), "retired_reason": r["retired_reason"] or "",
                "sandbox": bool(r["sandbox"]), "author": r["author"],
                "support": [v["run_id"] for v in votes if v["verdict"] == "support"],
                "against": [v["run_id"] for v in votes if v["verdict"] == "against"],
                "votes": votes}
