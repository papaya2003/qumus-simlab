from __future__ import annotations

import pytest

from qumus_simlab.core.clock import RealClock, SimClock
from qumus_simlab.runtime.store import RunStore


def test_runs_steps_and_events():
    s = RunStore()
    rid = s.create_run("place_chip_on_transfer_stage", {"chip_id": "c1"}, requested_by="hands")
    s.set_run_status(rid, "running")
    s.begin_step(rid, 0, "a", expected={"x": 1})
    s.finish_step(rid, 0, "done", measurements={"x": 1.1}, corrections=1)
    steps = s.get_steps(rid)
    assert steps[0]["status"] == "done" and steps[0]["corrections"] == 1
    kinds = [e.kind for e in s.events_after(0)]
    assert kinds == ["job.submitted", "job.status", "step.started", "step.finished"]
    assert s.events_after(2)[0].kind == "step.started"
    with pytest.raises(ValueError):
        s.set_run_status(rid, "weird")


def test_device_locks_are_all_or_nothing():
    s = RunStore()
    assert s.acquire_locks(["arm", "cam"], "r1")
    assert not s.acquire_locks(["cam", "stage"], "r2")
    assert "stage" not in s.locks()
    assert s.acquire_locks(["arm"], "r1")  # re-entrant
    s.release_locks("r1")
    assert s.acquire_locks(["cam", "stage"], "r2")


def test_samples_merge_props():
    s = RunStore()
    s.upsert_sample("chip_1", "chip", location="slot_1", status="available", props={"a": 1})
    s.upsert_sample("chip_1", "chip", location="transfer_stage", props={"b": 2})
    c = s.get_sample("chip_1")
    assert c["location"] == "transfer_stage" and c["status"] == "available"
    assert c["props"] == {"a": 1, "b": 2}
    assert [x["id"] for x in s.list_samples(location="transfer_stage")] == ["chip_1"]


def test_plan_revisions_and_instructions():
    s = RunStore()
    pid = s.create_project("p")
    assert s.set_plan(pid, [{"id": "s1"}], author="orchestrator", note="first") == 1
    assert s.set_plan(pid, [{"id": "s1"}, {"id": "s2"}], author="orchestrator") == 2
    assert [h["revision"] for h in s.plan_history(pid)] == [1, 2]
    iid = s.add_instruction("hello", project_id=pid)
    assert s.pending_instructions()[0]["id"] == iid
    s.mark_instruction(iid, "delivered")
    assert s.pending_instructions() == []


def test_file_store_is_shared_between_connections(tmp_path):
    a = RunStore(tmp_path / "q.db")
    b = RunStore(tmp_path / "q.db")
    a.emit("notification", {"text": "hi"})
    assert b.events_after(0)[0].payload["text"] == "hi"


def test_instrument_age_requires_comparable_reader_clock(tmp_path, monkeypatch):
    path = tmp_path / "mixed-frames.db"
    clock = SimClock(start=100)
    lab = RunStore(path, clock=clock)
    parent_clock = RealClock()
    monkeypatch.setattr(parent_clock, "now", lambda: 90)
    parent = RunStore(path, clock=parent_clock)
    try:
        lab.emit("lab.frame", {"device": "camera", "fixture": True})
        clock.advance(7)
        own, = lab.instrument_observations()
        assert own["age_s"] == 7 and own["age_basis"] == "comparable_reader_clock"
        parent.emit("agent.request_usage", {"fixture": True})
        other, = parent.instrument_observations()
        assert other["age_s"] is None and other["age_basis"] == "unknown"
        assert other["fresh_measurement"] is False
        assert lab.instrument_observations()[0]["age_s"] == 7
    finally:
        parent.close()
        lab.close()


def test_instrument_future_and_legacy_frames_have_unknown_age(monkeypatch):
    clock = SimClock(start=100)
    store = RunStore(clock=clock)
    try:
        event_id = store.emit("lab.frame", {"device": "camera"})
        monkeypatch.setattr(clock, "now", lambda: 99)
        future, = store.instrument_observations()
        assert future["age_s"] is None and future["age_basis"] == "unknown"
        store._x("DELETE FROM event_clock WHERE event_id=?", (event_id,))
        monkeypatch.setattr(clock, "now", lambda: 101)
        assert store.instrument_observations()[0]["age_s"] is None
    finally:
        store.close()


def test_instrument_age_rejects_independent_simulation_epochs(tmp_path):
    path = tmp_path / "frame-epochs.db"
    first = RunStore(path, clock=SimClock(start=100))
    second = RunStore(path, clock=SimClock(start=100))
    try:
        prefix = first.emit("lab.frame", {"device": "first-camera"})
        assert first.instrument_observations()[0]["age_s"] == 0
        assert second.instrument_observations()[0]["age_s"] is None
        second.emit("lab.frame", {"device": "second-camera"})
        assert all(row["age_s"] is None for row in first.instrument_observations())
        old, new = second.instrument_observations()
        assert old["age_s"] is None and new["age_s"] == 0
        assert first.event_clock_readings(prefix) == {first.clock.event_clock_basis: 100}
        assert first.event_clock_readings(second.last_event_id()) == {second.clock.event_clock_basis: 100}
    finally:
        second.close()
        first.close()


def test_instrument_age_is_comparable_across_utc_readers(tmp_path, monkeypatch):
    first_clock, second_clock = RealClock(), RealClock()
    monkeypatch.setattr(first_clock, "now", lambda: 100)
    monkeypatch.setattr(second_clock, "now", lambda: 109)
    path = tmp_path / "utc-frames.db"
    first, second = RunStore(path, clock=first_clock), RunStore(path, clock=second_clock)
    try:
        first.emit("lab.frame", {"device": "camera"})
        observed, = second.instrument_observations()
        assert observed["age_s"] == 9 and observed["clock_basis"] == "utc"
        assert observed["fresh_measurement"] is False
    finally:
        second.close()
        first.close()
