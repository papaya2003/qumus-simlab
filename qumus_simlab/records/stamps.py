"""Stamp states and a person's report of a new PC film (2026-10-22, item 15).

A stamp (glass slide, PDMS dome, PC film) is in one of three states in the
sample ledger:

    fresh   a new PC film: the only state a stack may take (stacking.fresh_stamps)
    used    one stack was made with it; availability is not film presence
    torn    its film tore (found at a pick-up or the drop-down)

(`in_use` while it sits on the stamp holder.) Only a person makes a used or
torn stamp fresh again, by putting a new PC film on it and saying so: the Lab
page's "New PC film" control or a chat message to Qumus (tool
`report_new_film`, which must cite the person's message). The code never
treats an unreported stamp as fresh. Every report is kept on the stamp
(`props.film_reports`) and as an event `stamp.film_replaced`.
Known PC/mica identity is retained separately in props.support. Current props.film
records presence after verified release: PC may be absent while mica remains on
a used stamp. Neither retained mica nor an identity report makes it fresh again.
"""

from __future__ import annotations

import re
from typing import Any

from qumus_simlab.flows.bench import stamp_slots
from qumus_simlab.records.lab_geometry import record_report
from qumus_simlab.runtime.store import RunStore

STAMP_STATES = ("fresh", "used", "torn")
#: states a new film may be reported for
REFILMABLE = ("used", "torn")
#: run states in which a job's bookings still hold (set_run_status releases the rest)
_OPEN_RUNS = ("queued", "running", "paused", "parked", "waiting_human")


class FilmReportError(ValueError):
    """The report cannot be recorded (the message says why)."""


def report_new_film(store: RunStore, bench: dict[str, Any], stamp_id: str, *, by: str,
                    via: str, instruction_id: str | None = None,
                    what: str = "new PC film", film_type: str = "PC") -> dict[str, Any]:
    """Record that a person put a new PC film on `stamp_id`: checked, then the
    stamp is `fresh` again. `by`: who (the person, or the agent that passed the
    person's message on); `via`: which way ("ui:lab" or "chat"). Returns the
    record; raises FilmReportError when the stamp may not be reset."""
    if film_type not in ("PC", "mica"):
        raise FilmReportError("The replacement support must be PC or mica.")
    if film_type == "mica" and what == "new PC film":
        what = "new mica support"
    with store._transaction():
        s = store.get_sample(stamp_id)
        if s is None or s["kind"] != "stamp":
            known = ", ".join(x["id"] for x in store.list_samples(kind="stamp")) or "none"
            raise FilmReportError(f"{stamp_id} is not a stamp in the ledger (stamps: {known})")
        slots = stamp_slots(bench)
        if s["location"] not in slots:
            raise FilmReportError(f"{stamp_id} is at {s['location']}, not in its slot on the right "
                                  f"storage board ({', '.join(sorted(slots))}): a stamp in use "
                                  f"cannot be re-filmed")
        booking = store.reservations("sample").get(stamp_id)
        if booking is not None:
            run = store.get_run(booking["run_id"]) or {}
            if run.get("status") in _OPEN_RUNS:
                raise FilmReportError(f"{stamp_id} is booked by job {booking['run_id']} "
                                      f"({run.get('flow')}, {run.get('status')})")
        if s["status"] not in REFILMABLE:
            raise FilmReportError(f"{stamp_id} is {s['status']}; a new film is recorded only on a "
                                  f"used or torn stamp")
        if instruction_id and any(r.get("instruction_id") == instruction_id
                                  for r in s["props"].get("film_reports") or []):
            raise FilmReportError(f"message {instruction_id} was already recorded as a new film "
                                  f"on {stamp_id}")
        record = {"what": what, "by": by, "via": via, "instruction_id": instruction_id,
                  "ts": store.clock.now(), "previous_status": s["status"]}
        store.upsert_sample(stamp_id, "stamp", status="fresh", props={
            "film": film_type, "stack": [],
            "support": {"film_type": film_type, "source": "operator_report", "instruction_id": instruction_id, "ts": record["ts"]},
            "support_history": [*(s["props"].get("support_history") or []),
                {"film_type": film_type, "source": "operator_report", "instruction_id": instruction_id, "ts": record["ts"]}],
            "film_reports": [*(s["props"].get("film_reports") or []), record]})
        event_id = store.emit("stamp.film_replaced", {"stamp_id": stamp_id, "status": "fresh", **record},
                              agent=None if via.startswith("ui") else by)
        record_report(store, stamp_id, "contact_support", film_type, instruction_id=instruction_id,
                      reporter=by if via.startswith("ui") else None, via=via, event_ids=[event_id])
    return {"stamp_id": stamp_id, "status": "fresh", **record}


def names_stamp(text: str, stamp_id: str) -> bool:
    """Does a person's message name this stamp? "stamp_01", "stamp 01",
    "stamp 1", "stamp1", "stamp #1" all name stamp_01."""
    m =re.fullmatch(r"stamp_?0*(\d+)", stamp_id)
    if m is None:
        return stamp_id.lower() in text.lower()
    return re.search(rf"(?<![A-Za-z0-9_])stamp[\s_#-]*0*{m.group(1)}(?![A-Za-z0-9_])", text, re.I) is not None


def record_support(store: RunStore, bench: dict, stamp_id: str, film_type: str, *, instruction_id: str,
                   project_id: str | None) -> dict:
    """Record known contact material separately from disposable film/stack state."""
    with store._transaction():
        ins = store.get_instruction(instruction_id)
        if not ins or ins["source"] != "ui" or ins["project_id"] not in (None, project_id) or not names_stamp(ins["text"], stamp_id):
            raise FilmReportError("Cite the person's report naming this stamp and its contact film.")
        if film_type not in ("PC", "mica") or not re.search(r"(?<![A-Za-z0-9])" + film_type + r"(?![A-Za-z0-9])", ins["text"], re.I):
            raise FilmReportError("The reported contact film must be PC or mica and explicit in the message.")
        sample = store.get_sample(stamp_id)
        if not sample or sample["kind"] != "stamp" or sample["location"] not in stamp_slots(bench) or stamp_id in store.reserved_samples():
            raise FilmReportError("Record only an unreserved stamp in storage.")
        record = {"film_type": film_type, "instruction_id": instruction_id, "source": "operator_report", "ts": store.clock.now()}
        store.upsert_sample(stamp_id, "stamp", props={"support": record,
            "support_history": [*(sample["props"].get("support_history") or []), record]})
        event_id = store.emit("stamp.support_recorded", {"stamp_id": stamp_id, **record})
        record_report(store, stamp_id, "contact_support", film_type,
                      instruction_id=instruction_id, event_ids=[event_id])
    return record


def contact_film(sample: dict) -> str | None:
    """Legacy PC values are explicit existing declarations, never a mica default."""
    return (sample.get("props", {}).get("support") or {}).get("film_type") or sample.get("props", {}).get("film")
