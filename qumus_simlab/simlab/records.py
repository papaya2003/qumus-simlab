"""The lab's records as seen from outside: change feed, brain write-back, images.

[Q] contract extensions agreed with the agent (notes/agent-brain-20261110/
brain-simlab-interface.md). The sim lab and the agent each keep their OWN run
store; these three pieces are the only way records cross between them:

``GET /api/records?since=<cursor>``
    Every change the lab makes to its `sample`, `reservation`, `run`, `step`
    and `event` tables, in commit order. A change log table filled by SQLite
    triggers gives each change a sequence number; ``since`` is the last
    ``next`` the client saw. ``row`` is the row as the RunStore API returns it
    (``null``: deleted or released). Image files are never referred to by a
    local path: every string that is a path to a file in the lab's image
    folder becomes the string ``"qumus-image:<image_id>"`` (row shapes and
    types stay the same), fetchable at ``GET /api/images/<image_id>``.

``put_records`` (command; any client, not motion, allowed while busy)
    The agent's own writes the flows read: project state, samples it edits
    (designs, vetoes, reports), reservations, UI instructions. Upserted as
    given (last write wins); no lab events are emitted for them, but the
    changed rows do come back through the feed (harmless echo). Strings
    ``"qumus-image:<id>"`` are turned back into the lab's own file paths.
    One exception, the placement guard: the lab owns where a sample is
    (``location`` and the props in ``PLACEMENT_PROPS``). A pushed value that
    the sample held EARLIER but not now (a stale copy, e.g. from a save point)
    or a placement prop missing from the pushed row keeps the lab's current
    value; a value the sample never had (the brain's own measurement) is taken.
    ``"placement_from_brain": true`` on a sample row takes its placement as
    given (e.g. a chip carried back by hand). Likewise a picture path outside the
    lab's folders (a brain-local path not sent as ``"qumus-image:<id>"``) keeps
    the lab's own path at the same place in the row. The receipt lists what was
    kept.

``GET /api/images/<image_id>`` (+ ``/meta``)
    The picture bytes. An image id is the picture's path relative to the
    lab's image folder (it may contain ``/``).
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

CHANGE_LOG_SQL = """
CREATE TABLE IF NOT EXISTS change_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, tbl TEXT NOT NULL, key TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS cl_sample_i AFTER INSERT ON sample
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('sample', NEW.id); END;
CREATE TRIGGER IF NOT EXISTS cl_sample_u AFTER UPDATE ON sample
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('sample', NEW.id); END;
CREATE TRIGGER IF NOT EXISTS cl_sample_d AFTER DELETE ON sample
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('sample', OLD.id); END;
CREATE TRIGGER IF NOT EXISTS cl_res_i AFTER INSERT ON reservation
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('reservation', NEW.kind || ':' || NEW.key); END;
CREATE TRIGGER IF NOT EXISTS cl_res_u AFTER UPDATE ON reservation
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('reservation', NEW.kind || ':' || NEW.key); END;
CREATE TRIGGER IF NOT EXISTS cl_res_d AFTER DELETE ON reservation
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('reservation', OLD.kind || ':' || OLD.key); END;
CREATE TRIGGER IF NOT EXISTS cl_run_i AFTER INSERT ON run
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('run', NEW.id); END;
CREATE TRIGGER IF NOT EXISTS cl_run_u AFTER UPDATE ON run
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('run', NEW.id); END;
CREATE TRIGGER IF NOT EXISTS cl_run_d AFTER DELETE ON run
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('run', OLD.id); END;
CREATE TRIGGER IF NOT EXISTS cl_step_i AFTER INSERT ON step
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('step', NEW.run_id || ':' || NEW.seq); END;
CREATE TRIGGER IF NOT EXISTS cl_step_u AFTER UPDATE ON step
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('step', NEW.run_id || ':' || NEW.seq); END;
CREATE TRIGGER IF NOT EXISTS cl_step_d AFTER DELETE ON step
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('step', OLD.run_id || ':' || OLD.seq); END;
CREATE TRIGGER IF NOT EXISTS cl_event_i AFTER INSERT ON event
    BEGIN INSERT INTO change_log (tbl, key) VALUES ('event', CAST(NEW.id AS TEXT)); END;
"""

#: props of a sample that say where it physically is; the lab owns them
PLACEMENT_PROPS = ("pose_on_stage", "corner_stage_xy", "rotation_deg")

#: every placement a sample has held (stale-copy detection for put_records)
PLACEMENT_SQL = """
CREATE TABLE IF NOT EXISTS placement_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL, location TEXT,
    pose_on_stage TEXT, corner_stage_xy TEXT, rotation_deg TEXT
);
CREATE INDEX IF NOT EXISTS placement_history_sample ON placement_history (sample_id);
CREATE TRIGGER IF NOT EXISTS ph_sample_i AFTER INSERT ON sample
    BEGIN INSERT INTO placement_history (sample_id, location, pose_on_stage, corner_stage_xy,
        rotation_deg) VALUES (NEW.id, NEW.location, NEW.props_json -> '$.pose_on_stage',
        NEW.props_json -> '$.corner_stage_xy', NEW.props_json -> '$.rotation_deg'); END;
CREATE TRIGGER IF NOT EXISTS ph_sample_u AFTER UPDATE ON sample
    WHEN NEW.location IS NOT OLD.location
      OR (NEW.props_json -> '$.pose_on_stage') IS NOT (OLD.props_json -> '$.pose_on_stage')
      OR (NEW.props_json -> '$.corner_stage_xy') IS NOT (OLD.props_json -> '$.corner_stage_xy')
      OR (NEW.props_json -> '$.rotation_deg') IS NOT (OLD.props_json -> '$.rotation_deg')
    BEGIN INSERT INTO placement_history (sample_id, location, pose_on_stage, corner_stage_xy,
        rotation_deg) VALUES (NEW.id, NEW.location, NEW.props_json -> '$.pose_on_stage',
        NEW.props_json -> '$.corner_stage_xy', NEW.props_json -> '$.rotation_deg'); END;
"""

TABLES = ("sample", "reservation", "run", "step", "event")
IMAGE_PREFIX = "qumus-image:"
MAX_PAGE = 2000
#: run.meta values larger than this stay in the lab (see _run_for_feed)
FEED_META_VALUE_MAX = 64 * 1024


def install_change_log(store: Any) -> None:
    """Create the change log and its triggers, and log the rows already there
    (a fresh lab seeds its ledger before the windows exist)."""
    with store._transaction():
        fresh = store._x("SELECT name FROM sqlite_master WHERE type='table' "
                         "AND name='change_log'").fetchone() is None
        no_history = store._x("SELECT name FROM sqlite_master WHERE type='table' "
                              "AND name='placement_history'").fetchone() is None
        for stmt in _statements(CHANGE_LOG_SQL + PLACEMENT_SQL):
            store._x(stmt)
        if no_history:  # a store from before the guard (older save point): start from now
            store._x("INSERT INTO placement_history (sample_id, location, pose_on_stage, "
                     "corner_stage_xy, rotation_deg) SELECT id, location, "
                     "props_json -> '$.pose_on_stage', props_json -> '$.corner_stage_xy', "
                     "props_json -> '$.rotation_deg' FROM sample")
        if store.get_meta("lab_id") is None:
            import uuid

            store.set_meta("lab_id", uuid.uuid4().hex)
        if fresh:
            store._x("INSERT INTO change_log (tbl, key) SELECT 'sample', id FROM sample")
            store._x("INSERT INTO change_log (tbl, key) "
                     "SELECT 'reservation', kind || ':' || key FROM reservation")
            store._x("INSERT INTO change_log (tbl, key) SELECT 'run', id FROM run")
            store._x("INSERT INTO change_log (tbl, key) "
                     "SELECT 'step', run_id || ':' || seq FROM step")
            store._x("INSERT INTO change_log (tbl, key) "
                     "SELECT 'event', CAST(id AS TEXT) FROM event ORDER BY id")


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def _statements(sql: str) -> list[str]:
    """Split the script into statements (triggers contain ';' inside BEGIN..END)."""
    out, buf = [], []
    for line in sql.strip().splitlines():
        buf.append(line)
        joined = "\n".join(buf).strip()
        if joined.endswith("END;") or (joined.endswith(";") and "BEGIN" not in joined):
            out.append(joined)
            buf = []
    return out


class LabRecords:
    """Change feed, write-back and images of one simulated lab."""

    def __init__(self, store: Any, image_dir: str | Path | None,
                 extra_roots: tuple[str | Path, ...] | list[str | Path] = ()) -> None:
        self.store = store
        self.image_dir = Path(image_dir).resolve() if image_dir is not None else None
        #: picture folders of earlier workspaces a resumed lab still points at
        #: (save points: records keep absolute paths). Same id in every root.
        self.roots = ([self.image_dir] if self.image_dir is not None else []) + [
            Path(r).resolve() for r in extra_roots]
        install_change_log(store)
        #: unique per lab store: a client that sees it change starts again from since=0
        self.lab_id = store.get_meta("lab_id")

    # -- images ---------------------------------------------------------------
    def image_id(self, path: str | Path | None) -> str | None:
        """The id of a picture the lab saved (None: not one of the lab's files)."""
        if not path or not self.roots:
            return None
        try:
            resolved = Path(path).resolve()
        except OSError:
            return None
        for root in self.roots:
            try:
                return resolved.relative_to(root).as_posix()
            except ValueError:
                continue
        return None

    def image_path(self, image_id: str) -> Path:
        """The file of an image id. ValueError: malformed; FileNotFoundError: missing."""
        if (not self.roots or not image_id or image_id.startswith("/")
                or "\\" in image_id or any(p in ("", ".", "..") for p in image_id.split("/"))):
            raise ValueError(f"malformed image id {image_id!r}")
        for root in self.roots:
            f = (root / image_id).resolve()
            try:
                f.relative_to(root)
            except ValueError as e:
                raise ValueError(f"malformed image id {image_id!r}") from e
            if f.is_file():
                return f
        raise FileNotFoundError(image_id)

    def image_meta(self, image_id: str) -> dict[str, Any]:
        f = self.image_path(image_id)
        meta: dict[str, Any] = {"image_id": image_id, "bytes": f.stat().st_size,
                                "content_type": mimetypes.guess_type(f.name)[0]
                                or "application/octet-stream",
                                "version": f.stat().st_mtime_ns}
        sidecar = f.with_suffix(f.suffix + ".json")
        if sidecar.is_file():
            try:
                meta.update(json.loads(sidecar.read_text()))
            except (OSError, ValueError):
                pass
        try:
            from PIL import Image

            with Image.open(f) as im:
                meta.setdefault("width", im.width)
                meta.setdefault("height", im.height)
        except Exception:  # noqa: BLE001 - meta is best effort
            pass
        return meta

    def scrub(self, value: Any) -> Any:
        """Replace every lab image path inside a JSON value by "qumus-image:<id>"."""
        if isinstance(value, str):
            if len(value) < 4096 and any(value.startswith(str(r)) for r in self.roots):
                iid = self.image_id(value)
                if iid is not None:
                    try:
                        self.image_path(iid)
                    except (ValueError, FileNotFoundError):
                        return value
                    return IMAGE_PREFIX + iid
            return value
        if isinstance(value, dict):
            return {k: self.scrub(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self.scrub(v) for v in value]
        return value

    def unscrub(self, value: Any) -> Any:
        """The reverse of `scrub` for rows the agent sends back."""
        if isinstance(value, str):
            if value.startswith(IMAGE_PREFIX) and self.roots:
                iid = value[len(IMAGE_PREFIX):]
                try:
                    return str(self.image_path(iid))
                except (ValueError, FileNotFoundError):
                    return value
            return value
        if isinstance(value, dict):
            return {k: self.unscrub(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self.unscrub(v) for v in value]
        return value

    # -- change feed ----------------------------------------------------------
    def changes(self, since: int = 0, limit: int = 500) -> dict[str, Any]:
        limit = min(max(1, int(limit)), MAX_PAGE)
        rows = self.store._x("SELECT seq, tbl, key FROM change_log WHERE seq > ? "
                             "ORDER BY seq LIMIT ?", (int(since), limit)).fetchall()
        # Every entry carries the row as it is NOW, so of several changes to one row
        # in a page only the last is sent (a stamp or chip rewritten every frame
        # during a transfer otherwise sends hundreds of copies of the same row).
        last = {(r["tbl"], r["key"]): r["seq"] for r in rows}
        out = []
        for r in rows:
            if last[(r["tbl"], r["key"])] != r["seq"]:
                continue
            out.append({"seq": r["seq"], "table": r["tbl"], "key": r["key"],
                        "row": self.scrub(self._row(r["tbl"], r["key"]))})
        head = self.store._x("SELECT MAX(seq) AS m FROM change_log").fetchone()
        return {"changes": out, "next": rows[-1]["seq"] if rows else int(since),
                "head": int(head["m"] or 0), "lab_id": self.lab_id}

    def _run_for_feed(self, run_id: str) -> dict[str, Any] | None:
        """The run row without its large flow-internal meta values.

        Flows keep megabytes of working state in run.meta (a search's checkpoint,
        a transfer's normal_transfer record) and rewrite it every frame. The brain
        reads only small meta keys (purpose, target, workflow, review), so values
        over FEED_META_VALUE_MAX bytes stay in the lab; their keys are listed
        under meta["omitted_from_feed"]. SQLite drops them before Python parses."""
        st = self.store
        r = st._x("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        if r is None:
            return None
        row = dict(r)
        meta = row.get("meta_json")
        if meta and len(meta) > FEED_META_VALUE_MAX:
            big = [k for (k,) in st._x("SELECT key FROM json_each(?) WHERE length(CAST(value AS TEXT)) > ?",
                                        (meta, FEED_META_VALUE_MAX)).fetchall()]
            if big:
                paths = ['$."' + str(k).replace('"', '\\"') + '"' for k in big]
                stripped = st._x(f"SELECT json_remove(?, {', '.join('?' * len(paths))})",
                                 (meta, *paths)).fetchone()[0]
                row["meta_json"] = json.dumps({**json.loads(stripped), "omitted_from_feed": big})
        return st._run_row(row)

    def _row(self, table: str, key: str) -> dict[str, Any] | None:
        st = self.store
        if table == "sample":
            return st.get_sample(key)
        if table == "run":
            return self._run_for_feed(key)
        if table == "reservation":
            kind, _, k = key.partition(":")
            r = st._x("SELECT * FROM reservation WHERE kind = ? AND key = ?", (kind, k)).fetchone()
            return None if r is None else {"kind": r["kind"], "key": r["key"],
                                           "run_id": r["run_id"], "flow": r["flow"],
                                           "note": r["note"], "created_at": r["created_at"]}
        if table == "step":
            run_id, _, seq = key.rpartition(":")
            for s in st.get_steps(run_id):
                if str(s["seq"]) == seq:
                    return {"run_id": run_id, **s}
            return None
        if table == "event":
            r = st._x("SELECT e.*, c.clock_basis, c.clock_domain FROM event e "
                      "LEFT JOIN event_clock c ON c.event_id = e.id WHERE e.id = ?",
                      (int(key),)).fetchone()
            if r is None:
                return None
            row = {"id": r["id"], "ts": r["ts"], "kind": r["kind"], "run_id": r["run_id"],
                   "agent": r["agent"], "payload": json.loads(r["payload_json"] or "{}")}
            if r["clock_basis"]:
                row["clock"] = {"basis": r["clock_basis"], "domain": r["clock_domain"]}
            return row
        raise ValueError(table)

    # -- write-back -----------------------------------------------------------
    def put(self, body: dict[str, Any]) -> dict[str, Any]:
        """Upsert the agent's records as given. ValueError on a malformed body."""
        body = self.unscrub(body)
        projects = body.get("project")
        if isinstance(projects, dict):
            projects = [projects]
        projects = projects or []
        samples = body.get("samples") or []
        reservations = body.get("reservations") or []
        instructions = body.get("instructions") or []
        removed = body.get("removed_samples") or []
        for name, v in (("project", projects), ("samples", samples),
                        ("reservations", reservations), ("instructions", instructions),
                        ("removed_samples", removed)):
            if not isinstance(v, list):
                raise ValueError(f"{name} must be a list")
        st, now = self.store, self.store.clock.now()
        with st._transaction():
            for p in projects:
                if not isinstance(p, dict) or not isinstance(p.get("id"), str):
                    raise ValueError("each project needs an 'id'")
                st._x("INSERT INTO project (id, name, created_at, updated_at, state_json) "
                      "VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                      "name=excluded.name, updated_at=excluded.updated_at, "
                      "state_json=excluded.state_json",
                      (p["id"], str(p.get("name") or p["id"]), now, now,
                       json.dumps(p.get("state") or {}, default=str)))
            kept: dict[str, list[str]] = {}
            for s in samples:
                if not isinstance(s, dict) or not isinstance(s.get("id"), str) \
                        or not isinstance(s.get("kind"), str):
                    raise ValueError("each sample needs 'id' and 'kind'")
                s = self._guard_placement(s, kept)
                st._x("INSERT INTO sample (id, kind, location, status, props_json, updated_at) "
                      "VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, "
                      "location=excluded.location, status=excluded.status, "
                      "props_json=excluded.props_json, updated_at=excluded.updated_at",
                      (s["id"], s["kind"], s.get("location"), s.get("status"),
                       json.dumps(s.get("props") or {}, default=str), now))
            for sid in removed:
                st._x("DELETE FROM sample WHERE id = ?", (str(sid),))
            for r in reservations:
                if not isinstance(r, dict) or not all(isinstance(r.get(k), str)
                                                      for k in ("kind", "key")):
                    raise ValueError("each reservation needs 'kind' and 'key'")
                if r.get("released") or r.get("run_id") is None:
                    st._x("DELETE FROM reservation WHERE kind = ? AND key = ?",
                          (r["kind"], r["key"]))
                    continue
                st._x("INSERT INTO reservation (kind, key, run_id, flow, note, created_at) "
                      "VALUES (?,?,?,?,?,?) ON CONFLICT(kind, key) DO UPDATE SET "
                      "run_id=excluded.run_id, flow=excluded.flow, note=excluded.note",
                      (r["kind"], r["key"], str(r["run_id"]), r.get("flow") or "",
                       r.get("note") or "", now))
            for i in instructions:
                if not isinstance(i, dict) or not isinstance(i.get("id"), str):
                    raise ValueError("each instruction needs an 'id'")
                st._x("INSERT INTO instruction (id, ts, project_id, source, kind, text, status, note) "
                      "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                      "status=excluded.status, note=excluded.note, text=excluded.text",
                      (i["id"], i.get("ts", now), i.get("project_id"), i.get("source", "ui"),
                       i.get("kind", "message"), i.get("text", ""), i.get("status", "pending"),
                       i.get("note")))
        return {"projects": len(projects), "samples": len(samples), "removed_samples": len(removed),
                "reservations": len(reservations), "instructions": len(instructions),
                "placement_kept": kept}

    def _guard_placement(self, pushed: dict[str, Any], kept: dict[str, list[str]]) -> dict[str, Any]:
        """Keep the lab's placement where the pushed row carries a stale one (see module doc)."""
        pushed = self._keep_lab_paths(pushed, kept)
        if pushed.get("placement_from_brain"):
            return pushed
        st = self.store
        cur = st._x("SELECT location, props_json -> '$.pose_on_stage' AS pose_on_stage, "
                    "props_json -> '$.corner_stage_xy' AS corner_stage_xy, "
                    "props_json -> '$.rotation_deg' AS rotation_deg FROM sample WHERE id = ?",
                    (pushed["id"],)).fetchone()
        if cur is None:
            return pushed  # a new sample: the brain's as given
        history = st._x("SELECT location, pose_on_stage, corner_stage_xy, rotation_deg "
                        "FROM placement_history WHERE sample_id = ?", (pushed["id"],)).fetchall()
        props = dict(pushed.get("props") or {})
        out = dict(pushed)
        names: list[str] = []
        loc = pushed.get("location")
        if loc != cur["location"] and (loc is None or any(h["location"] == loc for h in history)):
            out["location"] = cur["location"]
            names.append("location")
        for key in PLACEMENT_PROPS:
            lab_value = _loads(cur[key])
            if props.get(key) is not None:  # missing or null: a stale row, keep the lab's
                value = props[key]
                if _same(value, lab_value):
                    continue
                if not any(h[key] is not None and _same(value, _loads(h[key])) for h in history):
                    continue  # a value the lab never had: the brain's own measurement
            elif cur[key] is None or lab_value is None:
                continue
            if cur[key] is None:
                props.pop(key, None)
            else:
                props[key] = lab_value
            names.append(key)
        if names:
            out["props"] = props
            kept.setdefault(pushed["id"], []).extend(names)
        return out

    def _foreign_path(self, value: Any) -> bool:
        """A local file path that is not one of the lab's pictures (e.g. the brain's copy)."""
        if not isinstance(value, str) or not value or len(value) > 4096 \
                or value.startswith(IMAGE_PREFIX):
            return False
        try:
            path = Path(value)
            return path.is_absolute() and self.image_id(path) is None
        except (OSError, ValueError):
            return False

    def _keep_lab_paths(self, pushed: dict[str, Any], kept: dict[str, list[str]]) -> dict[str, Any]:
        """Picture paths outside the lab's folders (a brain-local path that was not sent as
        "qumus-image:<id>") keep the lab's own path at the same place in the row."""
        row = self.store.get_sample(pushed["id"])
        if row is None or not isinstance(pushed.get("props"), dict):
            return pushed
        fixed: list[str] = []

        def walk(new: Any, old: Any, where: str) -> Any:
            if isinstance(new, dict) and isinstance(old, dict):
                return {k: walk(v, old.get(k), f"{where}.{k}" if where else k)
                        for k, v in new.items()}
            if isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
                return [walk(v, o, f"{where}[{i}]") for i, (v, o) in enumerate(zip(new, old, strict=True))]
            if self._foreign_path(new) and isinstance(old, str) and self.image_id(old) is not None:
                fixed.append(where)
                return old
            return new

        props = walk(pushed["props"], row["props"], "")
        if not fixed:
            return pushed
        kept.setdefault(pushed["id"], []).extend(f"props.{w}" for w in fixed)
        return {**pushed, "props": props}
