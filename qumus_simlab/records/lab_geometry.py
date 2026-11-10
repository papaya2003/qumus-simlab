"""Typed laboratory assertions and prefix-safe views over RunStore events.

An assertion records what its source said, not a motion permission. Historical,
synthetic, reported and measured facts never acquire one another's authority.
There is no separate state store; bounded views disclose incomplete histories.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qumus_simlab.core.types import new_id
from qumus_simlab.runtime.store import RunStore


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["measured", "operator_report", "historical_constant", "estimate", "synthetic"]
    reference: str = Field(min_length=1, max_length=1024)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    event_ids: list[int] = Field(default_factory=list, max_length=32)
    instruction_id: str | None = None
    reporter: str | None = None


class Uncertainty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1)
    units: str = Field(min_length=1)
    bounds: dict[str, tuple[float, float]]

    @field_validator("bounds")
    @classmethod
    def finite_bounds(cls, values: dict) -> dict:
        if any(not math.isfinite(a) or not math.isfinite(b) or a > b for a, b in values.values()):
            raise ValueError("uncertainty bounds must be finite and ordered")
        return values


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evidence numbers must be finite; use null for unknown")
    if isinstance(value, dict):
        for v in value.values():
            _finite(v)
    if isinstance(value, list | tuple):
        for v in value:
            _finite(v)


class LabEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    assertion_id: str = Field(default_factory=lambda: new_id("fact"))
    subject: str = Field(min_length=1, max_length=256)
    property: str = Field(min_length=1, max_length=128)
    bench: Literal["sim", "real"]
    value: Any
    units: str = Field(min_length=1, max_length=128)
    frame: str = Field(min_length=1, max_length=256)
    source: Source
    observed_at: float | None = None
    timebase: Literal["store_clock", "utc", "unknown"] = "unknown"
    calibration_id: str | None = None
    calibration_version: str | None = None
    uncertainty: Uncertainty | None = None
    status: Literal["asserted", "uncertain", "retracted"] = "asserted"
    unknown_reason: str | None = None
    supersedes: list[str] = Field(default_factory=list, max_length=1000)
    conflicts_with: list[str] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def coherent(self) -> LabEvidence:
        _finite(self.value)
        if self.observed_at is not None and not math.isfinite(self.observed_at):
            raise ValueError("observation time must be finite")
        if (self.observed_at is None) != (self.timebase == "unknown"):
            raise ValueError("unknown time must have null observation time")
        if self.value is None and not self.unknown_reason:
            raise ValueError("a null fact needs an unknown reason")
        if self.source.kind == "operator_report" and not (
                self.source.instruction_id or self.source.reporter):
            raise ValueError("operator reports need the cited instruction or named UI reporter")
        if (self.calibration_id is None) != (self.calibration_version is None):
            raise ValueError("calibration identity and version must be supplied together")
        if self.assertion_id in [*self.supersedes, *self.conflicts_with]:
            raise ValueError("an assertion cannot replace or conflict with itself")
        if len(json.dumps(self.model_dump(), allow_nan=False)) > 65536:
            raise ValueError("laboratory assertion exceeds the 64 KiB bound")
        return self


def reduce_evidence(events: list, *, complete: bool = True, limit: int | None = 100,
                    assertion_limit: int | None = 8) -> dict:
    """Reduce only the supplied prefix. Missing dependencies remain unresolved."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    seen: dict[str, dict] = {}
    replaced_by_group: dict[tuple[str, str, str], set[str]] = {}
    malformed = 0
    for event in sorted(events, key=lambda e: e.id):
        try:
            fact = LabEvidence.model_validate(event.payload)
        except ValueError:
            malformed += 1
            continue
        item = {**fact.model_dump(mode="json"), "event_id": event.id, "recorded_at": event.ts}
        # A duplicate id is not a new authority, even in malformed external logs.
        if fact.assertion_id in seen:
            malformed += 1
            continue
        key = (fact.bench, fact.subject, fact.property)
        valid = {ref for ref in fact.supersedes if ref in seen
                 and (seen[ref]["bench"], seen[ref]["subject"], seen[ref]["property"]) == key}
        replaced_by_group.setdefault(key, set()).update(valid)
        seen[fact.assertion_id] = item
        groups.setdefault(key, []).append(item)
    current = []
    for (bench, subject, prop), facts in groups.items():
        replaced = replaced_by_group.get((bench, subject, prop), set())
        active = [f for f in facts if f["assertion_id"] not in replaced and f["status"] != "retracted"]
        missing = sorted({ref for f in facts for ref in [*f["supersedes"], *f["conflicts_with"]]
                          if ref not in seen})
        values = {json.dumps([f["value"], f["units"], f["frame"]], sort_keys=True) for f in active}
        explicit = any(ref in {a["assertion_id"] for a in active}
                       for f in active for ref in f["conflicts_with"])
        state = ("conflicted" if len(values) > 1 or explicit else
                 "uncertain" if missing or not complete or any(f["status"] == "uncertain" for f in active)
                 else "unknown" if not active or active[0]["value"] is None else "recorded")
        shown = active if assertion_limit is None else active[-assertion_limit:]
        current.append({"bench": bench, "subject": subject, "property": prop, "status": state,
                        "assertions": shown, "omitted_assertions": len(active)-len(shown),
                        "missing_references": missing[:32],
                        "omitted_missing_references": max(0, len(missing)-32)})
    current.sort(key=lambda r: max((f["event_id"] for f in r["assertions"]), default=0), reverse=True)
    return {"schema_version": 1, "facts": current[:limit], "complete_history": complete,
            "omitted_properties": max(0, len(current)-(limit if limit is not None else len(current))),
            "malformed_assertions": malformed,
            "fresh_measurement": False, "readiness": "not evaluated"}


def _evidence_events(store: RunStore, through: int, *, subject: str | None = None,
                     page_size: int = 500, clock_bases: dict[int, str] | None = None) -> list:
    from qumus_simlab.runtime.events import Event

    sql = ("SELECT e.*, c.clock_basis FROM event e LEFT JOIN event_clock c ON c.event_id=e.id "
           "WHERE e.kind='lab.evidence' AND e.id<=? AND e.id>?")
    args: list[Any] = [through]
    if subject is not None:
        sql += " AND json_extract(payload_json, '$.subject')=?"
    events, after = [], 0
    while True:
        page_args = [*args, after]
        if subject is not None:
            page_args.append(subject)
        rows = store._x(sql+" ORDER BY e.id LIMIT ?", (*page_args, page_size)).fetchall()
        if clock_bases is not None:
            clock_bases.update({r["id"]: r["clock_basis"] for r in rows if r["clock_basis"] is not None})
        events.extend(Event(id=r["id"], ts=r["ts"], kind=r["kind"], run_id=r["run_id"], agent=r["agent"],
                            payload=json.loads(r["payload_json"])) for r in rows)
        if len(rows) < page_size:
            break
        after = rows[-1]["id"]
    return events


def evidence_view(store: RunStore, *, through_event_id: int | None = None,
                  subject: str | None = None, limit: int = 50, scan_limit: int = 500) -> dict:
    """Reduce the full prefix in pages, then bound returned properties/assertions."""
    if not 1 <= limit <= 100 or not 1 <= scan_limit <= 1000:
        raise ValueError("evidence view bounds are 1..100 properties and 1..1000 assertions")
    through = store.last_event_id() if through_event_id is None else through_event_id
    events = _evidence_events(store, through, subject=subject, page_size=scan_limit)
    return {**reduce_evidence(events, limit=limit),
            "through_event_id": through, "assertions_scanned": len(events),
            "source": "stored lab.evidence events"}


def project_observed_scene(events: list, *, through_event_id: int, as_of: float | None,
                           clock_readings: dict[str, float] | None = None,
                           evidence_clock_bases: dict[int, str] | None = None,
                           as_of_clock_basis: str | None = None,
                           stale_after_s: float = 300, object_limit: int = 200,
                           subject_limit: int = 100, unresolved_limit: int = 100) -> dict:
    """Pure observed/prior scene from an evidence prefix, without hidden truth.

    Unknown identities stay anonymous. Historical local coordinates are useful
    in their declared frame without becoming calibrated current world poses.
    Freshness uses explicitly comparable saved clock readings, not readiness.
    """
    if (through_event_id < 0 or not math.isfinite(stale_after_s) or stale_after_s <= 0
            or (as_of is not None and not math.isfinite(as_of))) or not all(
            1 <= n <= 1000 for n in (object_limit, subject_limit, unresolved_limit)):
        raise ValueError("invalid observed-scene bounds")
    objects, subjects, unresolved, frames, named_poses = {}, {}, [], {}, {}

    def display_value(value: Any) -> Any:
        encoded = json.dumps(value, ensure_ascii=False)
        return value if len(encoded) <= 4096 else {"display_omitted": True,
            "serialized_characters": len(encoded), "preview": encoded[:512]}

    def observation_value(value: Any) -> Any:
        # The scene consumes observed label/pose/confidence only, including in
        # unresolved rows. Detector extras cannot promote hidden identity.
        if not isinstance(value, dict):
            return None
        if "detections" in value:
            detections = value["detections"]
            return {"detections": [observation_value(d) for d in detections]
                    if isinstance(detections, list) else None}
        confidence = value.get("confidence")
        return {"label": value.get("label") if isinstance(value.get("label"), str) else "object",
                "pose": point(value.get("pose")),
                "confidence": confidence if isinstance(confidence, int | float)
                and not isinstance(confidence, bool) and 0 <= confidence <= 1 else None}

    def metadata(f: dict, row: dict) -> dict:
        writer_basis = (evidence_clock_bases or {}).get(f["event_id"])
        basis = ("utc" if f["timebase"] == "utc" and writer_basis is not None else writer_basis
                 if f["timebase"] == "store_clock" else None)
        reference = (clock_readings or {}).get(basis)
        age = (reference-f["observed_at"] if reference is not None
               and f["observed_at"] is not None else None)
        freshness = ("unknown" if age is None or age < 0 else "stale" if age > stale_after_s else "recent")
        return {"source": f["source"], "frame": f["frame"], "observed_at": f["observed_at"],
                "timebase": f["timebase"], "freshness": {"status": freshness, "age_s": age,
                    "as_of": reference, "clock_basis": basis, "stale_after_s": stale_after_s},
                "calibration_id": f["calibration_id"], "calibration_version": f["calibration_version"],
                "uncertainty": f["uncertainty"], "status": row["status"],
                "assertion_ids": [a["assertion_id"] for a in row["assertions"]][-8:],
                "event_ids": [a["event_id"] for a in row["assertions"]][-8:],
                "omitted_assertions": max(0, len(row["assertions"])-8)}

    def problem(row: dict, f: dict, reason: str, value: Any = None) -> None:
        shown = f["value"] if value is None else value
        if row["property"] == "overview_observation":
            shown = observation_value(shown)
        item = {"subject": row["subject"], "property": row["property"],
            "status": row["status"], "reason": reason, "evidence": metadata(f, row),
            "value": display_value(shown)}
        if row["status"] == "conflicted":
            item["conflicting_values"] = [{"value": display_value(observation_value(a["value"])
                if row["property"] == "overview_observation" else a["value"]), "source": a["source"],
                                           "frame": a["frame"]} for a in row["assertions"]][-8:]
        unresolved.append(item)

    def point(value: Any) -> dict | None:
        if isinstance(value, dict):
            p = {k: value.get(k) for k in ("x", "y", "z", "rotation")}
        elif isinstance(value, list | tuple) and len(value) in (2, 3):
            p = {"x": value[0], "y": value[1], "z": value[2] if len(value) == 3 else None,
                 "rotation": None}
        else:
            return None
        if not all(isinstance(p[k], int | float) and not isinstance(p[k], bool)
                   and math.isfinite(p[k]) for k in ("x", "y")):
            return None
        if any(v is not None and (not isinstance(v, int | float) or isinstance(v, bool)
                                   or not math.isfinite(v)) for v in (p["z"], p["rotation"])):
            return None
        return p

    # Conflict equality for this consumer must use the same observed fields as
    # rendering. Hidden detector extras cannot create or resolve a conflict.
    # Keep raw assertion history intact, including its original validation.
    from qumus_simlab.runtime.events import Event

    prefix = []
    for event in events:
        if event.id > through_event_id:
            continue
        try:
            fact = LabEvidence.model_validate(event.payload)
        except ValueError:
            prefix.append(event)
            continue
        if fact.property == "overview_observation" and isinstance(fact.value, dict):
            event = Event(id=event.id, ts=event.ts, kind="lab.evidence",
                          payload={**event.payload, "value": observation_value(fact.value)})
        prefix.append(event)
    rows = reduce_evidence(prefix, limit=None, assertion_limit=None)["facts"]

    transforms = {}
    for row in rows:
        if row["property"] != "frame_transform" or row["status"] != "recorded" or not row["assertions"]:
            continue
        f = row["assertions"][-1]
        v = f["value"]
        if (isinstance(v, dict) and v.get("to_frame") == "world"
                and isinstance(v.get("from_frame"), str) and v["from_frame"] == f["frame"]
                and v["from_frame"] not in ("unknown", "world")
                and f["source"]["kind"] in ("measured", "synthetic")
                and f["calibration_id"] not in (None, "unknown", "unavailable")
                and f["calibration_version"] not in (None, "unknown", "unavailable")
                and metadata(f, row)["freshness"]["status"] == "recent"
                and f["units"] in ("mm", "mm,deg")):
            offset = v.get("translation_mm")
            angle = v.get("rotation_deg")
            if (isinstance(offset, list) and len(offset) == 3
                    and all(isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)
                            for x in [*offset, angle])):
                transforms.setdefault(v["from_frame"], []).append((offset, angle, f, metadata(f, row)))

    def in_world(p: dict, f: dict) -> tuple[dict | None, dict | None]:
        if f["frame"] == "world":
            return p, None
        options = transforms.get(f["frame"], [])
        if len(options) != 1:
            return None, None
        offset, theta, transform, provenance = options[0]
        r = math.radians(theta)
        return {"x": offset[0]+p["x"]*math.cos(r)-p["y"]*math.sin(r),
                "y": offset[1]+p["x"]*math.sin(r)+p["y"]*math.cos(r),
                "z": None if p["z"] is None else offset[2]+p["z"],
                "rotation": None if p["rotation"] is None else p["rotation"]+theta}, {
                    "assertion_id": transform["assertion_id"], "calibration_id": transform["calibration_id"],
                    "calibration_version": transform["calibration_version"], "evidence": provenance,
                    "uncertainty_propagation": "not evaluated"}

    for row in rows:
        if not row["assertions"]:
            continue
        f = row["assertions"][-1]
        prop, value, ev = row["property"], f["value"], metadata(f, row)
        if prop in ("material_identity", "contact_support") and f["source"]["kind"] == "operator_report":
            subject = subjects.setdefault(row["subject"], {"subject": row["subject"], "attributes": {}, "pose_key": None})
            subject["attributes"][prop] = {"value": value if row["status"] == "recorded" else None,
                "status": row["status"], "evidence": ev,
                "reported_values": [a["value"] for a in row["assertions"]][-8:]}
            if row["status"] != "recorded":
                problem(row, f, row["status"])
            continue
        if prop == "overview_observation" and isinstance(value, dict):
            if row["status"] != "recorded":
                problem(row, f, row["status"])
                continue
            if ev["freshness"]["status"] != "recent":
                problem(row, f, ev["freshness"]["status"]+"_observation_time")
                continue
            if f["source"]["kind"] not in ("measured", "synthetic") or f["units"] not in ("mm", "mm,deg"):
                problem(row, f, "unsupported_pose_source_or_units")
                continue
            detections = value.get("detections")
            if not isinstance(detections, list):
                problem(row, f, "invalid_detections")
                continue
            for index, detection in enumerate(detections):
                p = point(detection.get("pose")) if isinstance(detection, dict) else None
                mapped, binding = in_world(p, f) if p is not None else (None, None)
                if mapped is None:
                    problem(row, f, "unknown_frame" if p is not None else "invalid_pose", detection)
                    continue
                key = f"obs:{f['assertion_id']}:{index}"
                reported = observation_value(detection)
                label = reported["label"] or "object"
                kind = label if label in ("chip", "carrier", "stamp", "device") else "object"
                objects[key] = {**mapped, "frame": "world", "kind": kind, "label": label, "subject": None,
                    "identity_status": "unidentified", "evidence": {**ev, "transform": binding},
                    "attributes": {}, "reported_confidence": reported["confidence"]}
            continue
        spatial = prop in ("pose", "position", "seeded_xy", "software_waypoint", "microscope_XYL_waypoint") or prop.startswith("QR_")
        if spatial:
            points = (value if prop.startswith("QR_") and isinstance(value, list) else [value])
            parsed = [(i, point(v)) for i, v in enumerate(points)]
            if f["source"]["kind"] == "historical_constant":
                frame = frames.setdefault(f["frame"], {"id": f["frame"], "label": f["frame"], "objects": {}})
                for index, p in parsed:
                    if p is not None and f["units"] in ("mm", "mm,deg"):
                        key = f"prior:{f['assertion_id']}:{index}"
                        original = points[index]
                        if prop == "microscope_XYL_waypoint" and isinstance(original, list) and len(original) == 3:
                            p = {**p, "z": None, "linear_mm": original[2]}
                        frame["objects"][key] = {**p, "frame": f["frame"], "kind": "prior", "label": row["subject"]+(f" [{index+1}]" if len(parsed)>1 else ""),
                            "subject": row["subject"], "identity_status": "historical", "evidence": ev,
                            "original_coordinates": original, "attributes": {}}
            if row["status"] != "recorded":
                problem(row, f, row["status"])
                continue
            if f["source"]["kind"] == "historical_constant":
                problem(row, f, "historical_prior_in_original_frame")
                continue
            if f["source"]["kind"] not in ("measured", "synthetic", "operator_report"):
                problem(row, f, "unsupported_pose_source")
                continue
            if ev["freshness"]["status"] != "recent":
                problem(row, f, ev["freshness"]["status"]+"_observation_time")
                continue
            p = parsed[0][1] if len(parsed) == 1 else None
            mapped, binding = in_world(p, f) if p is not None and f["units"] in ("mm", "mm,deg") else (None, None)
            if mapped is None:
                problem(row, f, "unknown_frame_or_units")
                continue
            key = row["subject"]
            candidate = {**mapped, "frame": "world", "kind": "object", "label": key, "subject": key,
                "identity_status": "declared", "evidence": {**ev, "transform": binding}, "attributes": {}}
            named_poses.setdefault(key, []).append((candidate, row, f))
            continue
        problem(row, f, f["unknown_reason"] or ("frame_transform_fact" if prop == "frame_transform" else "unlocated_fact"))

    for key, candidates in named_poses.items():
        subject = subjects.setdefault(key, {"subject": key, "attributes": {}, "pose_key": None})
        poses = {json.dumps({axis: obj[axis] for axis in ("x", "y", "z", "rotation")}, sort_keys=True)
                 for obj, _, _ in candidates}
        if len(poses) > 1:
            for _, row, f in candidates:
                problem(row, f, "conflicting_subject_poses")
            continue
        objects[key] = candidates[0][0]  # rows are newest first
        subject["pose_key"] = key

    for obj in objects.values():
        if obj["subject"] in subjects:
            obj["attributes"] = subjects[obj["subject"]]["attributes"]
    historical = []
    for frame in frames.values():
        points = list(frame["objects"].values())
        if not points:
            continue
        frame["bounds"] = {axis: [min(p[axis] for p in points)-20, max(p[axis] for p in points)+20] for axis in ("x", "y")}
        frame["count"] = len(points)
        frame["omitted_objects"] = max(0, len(points)-object_limit)
        frame["objects"] = dict(list(frame["objects"].items())[:object_limit])
        historical.append(frame)
    return {"schema_version": 1, "through_event_id": through_event_id,
        "as_of": {"value": as_of, "timebase": "utc" if as_of_clock_basis == "utc" else
                  "store_clock" if as_of_clock_basis is not None else "unknown",
                  "basis": "event_prefix", "clock_basis": as_of_clock_basis},
        "source": "stored_lab_evidence", "uses_simulator_truth": False,
        "objects": dict(list(objects.items())[:object_limit]), "subjects": list(subjects.values())[:subject_limit],
        "unresolved": unresolved[:unresolved_limit], "historical_frames": historical[:20],
        "counts": {"objects": len(objects), "subjects": len(subjects), "unresolved": len(unresolved)},
        "omissions": {"objects": max(0, len(objects)-object_limit), "subjects": max(0, len(subjects)-subject_limit),
                      "unresolved": max(0, len(unresolved)-unresolved_limit), "historical_frames": max(0, len(historical)-20)},
        "complete_reduction": True}


def observed_scene(store: RunStore, *, through_event_id: int | None = None, **bounds: Any) -> dict:
    """Build the pure scene using saved rows/time at one explicit event boundary."""
    through = store.last_event_id() if through_event_id is None else through_event_id
    row = store._x("SELECT e.ts, c.clock_basis FROM event e LEFT JOIN event_clock c ON c.event_id=e.id "
                   "WHERE e.id<=? ORDER BY e.id DESC LIMIT 1", (through,)).fetchone()
    # Each domain's last writer selects its current epoch. A new simulation
    # invalidates the old simulation basis; an unrelated UTC writer does not.
    readings = store.event_clock_readings(through)
    bases: dict[int, str] = {}
    events = _evidence_events(store, through, clock_bases=bases)
    return project_observed_scene(events, through_event_id=through,
        as_of=row["ts"] if row is not None else None,
        as_of_clock_basis=row["clock_basis"] if row is not None else None,
        clock_readings=readings, evidence_clock_bases=bases, **bounds)


def record_evidence(store: RunStore, evidence: LabEvidence, *, run_id: str | None = None,
                    supersede_previous: bool = False) -> int:
    """Append one validated assertion; callers retain admission and sensing policy."""
    if evidence.bench != store.bench:
        raise ValueError("evidence bench does not match the RunStore")
    with store._transaction():
        if evidence.source.kind == "operator_report":
            if evidence.source.instruction_id:
                instruction = store.get_instruction(evidence.source.instruction_id)
                if instruction is None or instruction["source"] != "ui":
                    raise ValueError("operator evidence must cite an existing human UI instruction")
            elif not evidence.source.reference.startswith("ui:lab:"):
                raise ValueError("uncited operator evidence requires the admitted named Lab UI report")
        if supersede_previous:
            prior = reduce_evidence(_evidence_events(store, store.last_event_id(), subject=evidence.subject),
                                    limit=None, assertion_limit=None)
            refs = list(dict.fromkeys([*evidence.supersedes, *[
                a["assertion_id"] for row in prior["facts"] if row["property"] == evidence.property
                for a in row["assertions"] if a["source"]["kind"] == evidence.source.kind
                and a["source"]["reference"] == evidence.source.reference]]))
            if len(refs) > 1000:
                raise ValueError("more than 1000 superseded assertions; resolve explicitly in bounded batches")
            evidence = LabEvidence.model_validate({**evidence.model_dump(), "supersedes": refs})
        for ref in [*evidence.supersedes, *evidence.conflicts_with]:
            row = store._x("SELECT payload_json FROM event WHERE kind='lab.evidence' "
                           "AND json_extract(payload_json, '$.assertion_id')=? ORDER BY id LIMIT 1",
                           (ref,)).fetchone()
            if row is None:
                raise ValueError(f"unknown earlier laboratory assertion: {ref}")
            prior = LabEvidence.model_validate_json(row["payload_json"])
            if (prior.bench, prior.subject, prior.property) != (evidence.bench, evidence.subject, evidence.property):
                raise ValueError("evidence references must concern the same bench, subject and property")
        duplicate = store._x("SELECT id FROM event WHERE kind='lab.evidence' "
                             "AND json_extract(payload_json, '$.assertion_id')=? LIMIT 1",
                             (evidence.assertion_id,)).fetchone()
        if duplicate is not None:
            raise ValueError("assertion IDs are immutable and cannot be reused")
        return store.emit("lab.evidence", evidence.model_dump(mode="json"), run_id=run_id)


def record_report(store: RunStore, subject: str, prop: str, value: Any, *,
                  instruction_id: str | None, reporter: str | None = None,
                  via: str = "chat", status: str = "asserted", event_ids: list[int] | None = None) -> int:
    return record_evidence(store, LabEvidence(
        subject=subject, property=prop, bench=store.bench, value=value,
        units="categorical", frame="sample_identity", status=status,
        source=Source(kind="operator_report", reference=f"{via}:{subject}:{prop}",
                      instruction_id=instruction_id, reporter=reporter, event_ids=event_ids or [])))
