"""Declared optical doubles exercise Stop boundaries without a World or jobs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import PreconditionFailed, StopRequested, TransientError
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library import pickup_evidence, search
from qumus_simlab.flows.step import Correction, Verdict
from qumus_simlab.runtime.store import RunStore


@pytest.fixture
def h(tmp_path, monkeypatch):
    store = RunStore(":memory:", clock=SimClock(start=100.), bench="sim")
    rid = store.create_run("search_chip", {"chip_id": "chip", "material": "Gr"})
    store.upsert_sample("chip", "chip", props={"rotation_deg": 0., "corner_stage_xy": [0., 0.]})
    for i, fid in enumerate(("f006", "f008")):
        store.upsert_sample(fid, "flake", location="chip", status="candidate", props={
            "stage_xy": [float(i), 0.], "labels": {"10x": "manylayer"}, "area_um2": 200-i})
    state = SimpleNamespace(cancel=False, stop=True, lost=False, seen=[], motions=[])
    def check_cancel():
        if state.cancel:
            raise StopRequested("declared Stop")
    class Scope:
        objective = 10
        focus = .4
        xy = (0., 0.)
        def get_objective(self):
            return self.objective
        def get_focus(self):
            return self.focus
        def set_objective(self, objective):
            check_cancel()
            self.objective = objective
            state.motions.append(("objective", objective))
        def set_focus(self, focus):
            check_cancel()
            self.focus = focus
            state.motions.append(("focus", focus))
        def capture(self):
            check_cancel()
            fid = "f006" if self.xy[0] == 0 else "f008"
            state.seen.append((fid, self.objective))
            image = tmp_path/f"{len(state.seen)}-{fid}-{self.objective}.png"
            Image.new("RGB", (8, 8), "gray").save(image)
            store.emit("device.command", {"device": "microscope", "command": "capture", "params": {}}, run_id=rid)
            return Frame("microscope", 100., path=str(image), meta={"objective": self.objective})
    scope = Scope()
    def autofocus(ctx, objective):
        if state.stop and scope.xy[0] == 1 and objective == 100:
            scope.set_focus(-.9)
            state.cancel = True
            raise StopRequested("after f00820 evidence, inside f008100 autofocus")
        scope.set_focus(.5 if scope.xy[0] == 0 else .2)
        return 10.
    def move(ctx, x, y):
        check_cancel()
        scope.xy = (x, y)
    def detections(frame, material):
        return [] if state.lost else [SimpleNamespace(dx_mm=0., dy_mm=0., thickness_label="manylayer", meta={})]
    ctx = SimpleNamespace(store=store, run_id=rid, clock=store.clock, params={"chip_id": "chip"},
        gate=SimpleNamespace(microscope=lambda name: scope), analyzer=SimpleNamespace(find_flakes=detections),
        bench={"microscope": {"revisit_objectives": [20, 100]}},
        verification={"revisit": {"refind_radius_um": 100}}, check_cancel=check_cancel)
    monkeypatch.setattr(search, "autofocus", autofocus)
    monkeypatch.setattr(search, "_move_stage", move)
    monkeypatch.setattr(pickup_evidence, "capture_neighbourhood", lambda *a: {"clear": True, "images": [scope.capture().path]})
    yield SimpleNamespace(ctx=ctx, store=store, state=state, scope=scope,
                          step=lambda: search.RevisitCandidates("chip", "Gr", 2))
    store.close()


def paused(h):
    step = h.step()
    with pytest.raises(StopRequested):
        step.act(h.ctx, None)
    h.store.update_run_meta(h.ctx.run_id, paused={"step": step.name, "in_progress": True})
    return step


def resume(h):
    h.state.cancel = h.state.stop = False
    step = h.step()
    step.act(h.ctx, None)
    return step


def test_completed_original_evidence_survives_and_partial_candidate_reacquires(h):
    first = paused(h)
    evidence = deepcopy(first._results["f006"])
    partial = [name for name in h.state.seen if name[0] == "f008"]
    assert partial == [("f008", 20)]
    assert h.store.get_sample("f006")["status"] == h.store.get_sample("f008")["status"] == "candidate"
    completed_captures = h.state.seen.count(("f006", 20)) + h.state.seen.count(("f006", 100))
    anchor = first._read_checkpoint(h.ctx)["current"]
    step = resume(h)
    assert step._results["f006"] == evidence
    assert h.state.seen.count(("f006", 20)) + h.state.seen.count(("f006", 100)) == completed_captures
    assert ("objective", anchor["objective"]) in h.state.motions
    assert ("focus", anchor["focus_z"]) in h.state.motions
    assert h.state.seen.count(("f008", 20)) == 2
    step.on_success(h.ctx, None)
    assert h.store.get_sample("f006")["props"]["images"] == evidence["images"]
    assert all(h.store.get_sample(fid)["status"] == "catalogued" for fid in ("f006", "f008"))


def test_publication_replay_does_not_reobserve_own_refined_outputs(h):
    h.state.stop = False
    step = h.step()
    step.act(h.ctx, None)
    step.on_success(h.ctx, None)
    before = deepcopy(h.state.seen)
    replay = h.step()
    replay.act(h.ctx, None)
    assert replay.verify(h.ctx).verdict == Verdict.OK
    replay.on_success(h.ctx, None)
    assert h.state.seen == before


def test_all_lost_giveup_and_explicit_correction_reacquire(h):
    h.state.stop = False
    h.state.lost = True
    step = h.step()
    step.act(h.ctx, None)
    assert step.verify(h.ctx).verdict == Verdict.UNCLEAR
    step.on_give_up(h.ctx, step.verify(h.ctx))
    before = len(h.state.seen)
    h.state.lost = False
    step = h.step()
    step.act(h.ctx, None)
    assert len(h.state.seen) > before and step.verify(h.ctx).verdict == Verdict.OK
    before = len(h.state.seen)
    h.store.emit("step.corrected", {"name": step.name, "action": "reacquire", "params": {}}, run_id=h.ctx.run_id)
    step.act(h.ctx, Correction("reacquire"))
    assert len(h.state.seen) > before


@pytest.mark.parametrize("change", ["result", "target", "rotation", "params", "foreign", "stale"])
def test_mismatched_or_unreceipted_journal_refuses(h, change):
    first = paused(h)
    cp = first._read_checkpoint(h.ctx)
    if change == "result":
        cp["completed"]["f006"]["found"] = False
        h.store.update_run_meta(h.ctx.run_id, revisit_candidates_checkpoint=cp)
    elif change == "target":
        h.store.upsert_sample("f008", "flake", props={"stage_xy": [5., 5.]})
    elif change == "rotation":
        h.store.upsert_sample("chip", "chip", props={"rotation_deg": 90.})
    elif change == "params":
        h.ctx.params["changed"] = True
    elif change == "foreign":
        other = h.store.create_run("search_chip", {})
        h.store.update_run_meta(other, revisit_candidates_checkpoint=cp)
        h.ctx.run_id = other
    else:
        old = deepcopy(cp)
        first._save_checkpoint(h.ctx, cp)
        h.store.update_run_meta(h.ctx.run_id, revisit_candidates_checkpoint=old)
    h.state.cancel = h.state.stop = False
    before = len(h.state.seen)
    with pytest.raises(PreconditionFailed):
        h.step().act(h.ctx, None)
    assert len(h.state.seen) == before


def test_cancel_at_entry_has_receipt_without_motion_and_can_continue(h):
    h.state.cancel = True
    with pytest.raises(StopRequested):
        h.step().act(h.ctx, None)
    assert not h.state.motions and h.step()._read_checkpoint(h.ctx) is not None
    h.store.update_run_meta(h.ctx.run_id, paused={"step": "revisit_candidates", "in_progress": True})
    assert resume(h).verify(h.ctx).verdict == Verdict.OK


def test_cancel_during_public_anchor_reads_saves_anchor_before_motion(h):
    def stop_read():
        h.state.cancel = True
        return .4
    h.scope.get_focus = stop_read
    with pytest.raises(StopRequested):
        h.step().act(h.ctx, None)
    assert h.step()._read_checkpoint(h.ctx)["current"]["focus_z"] == .4
    assert not h.state.motions


def test_missing_progressed_checkpoint_refuses_but_earlier_step_stop_is_fresh(h):
    h.store.begin_step(h.ctx.run_id, 6, "revisit_candidates")
    h.store.emit("device.command", {"device": "microscope", "command": "capture"}, run_id=h.ctx.run_id)
    h.store.update_run_meta(h.ctx.run_id, paused={"step": "revisit_candidates", "in_progress": True})
    with pytest.raises(PreconditionFailed, match="no original"):
        h.step().act(h.ctx, None)
    h.ctx.run_id = h.store.create_run("search_chip", {})
    h.store.begin_step(h.ctx.run_id, 5, "scan_chip")
    h.store.emit("device.command", {"device": "microscope", "command": "capture"}, run_id=h.ctx.run_id)
    h.store.update_run_meta(h.ctx.run_id, paused={"step": "scan_chip", "in_progress": True})
    h.store.begin_step(h.ctx.run_id, 6, "revisit_candidates")
    h.state.stop = False
    step = h.step()
    step.act(h.ctx, None)
    assert step.verify(h.ctx).verdict == Verdict.OK


@pytest.mark.parametrize("progressed", [False, True])
def test_historical_crash_without_pause_distinguishes_zero_work_entry(h, progressed):
    h.store.begin_step(h.ctx.run_id, 6, "revisit_candidates")
    if progressed:
        h.store.emit("device.command", {"device": "microscope", "command": "capture"}, run_id=h.ctx.run_id)
    h.store.begin_step(h.ctx.run_id, 6, "revisit_candidates")
    h.state.stop = False
    if progressed:
        with pytest.raises(PreconditionFailed, match="no original"):
            h.step().act(h.ctx, None)
    else:
        step = h.step()
        step.act(h.ctx, None)
        assert step.verify(h.ctx).verdict == Verdict.OK


def test_atomic_publication_rolls_back_all_samples_and_journal(h, monkeypatch):
    h.state.stop = False
    step = h.step()
    step.act(h.ctx, None)
    checkpoint = deepcopy(step._read_checkpoint(h.ctx))
    original = h.store.upsert_sample
    def fail_second(fid, *args, **kwargs):
        if fid == "f008":
            raise RuntimeError("declared publication failure")
        return original(fid, *args, **kwargs)
    monkeypatch.setattr(h.store, "upsert_sample", fail_second)
    with pytest.raises(RuntimeError):
        step.on_success(h.ctx, None)
    assert all(h.store.get_sample(fid)["status"] == "candidate" for fid in ("f006", "f008"))
    assert step._read_checkpoint(h.ctx) == checkpoint


def test_tuple_geometry_and_nested_metadata_roundtrip_without_rounding(h, monkeypatch):
    from qumus_simlab.core.polygons import transform

    h.state.stop = False
    h.ctx.params["nested"] = ((1., 2.),)
    h.ctx.bench["microscope"]["declared_tuple"] = (.1, .2)
    step = h.step()
    acquire = step._acquire_candidate
    polygon = transform(((0., 0.), (.0135792468013579, 0.), (0., .02)))
    def with_geometry(ctx, flake):
        result = acquire(ctx, flake)
        result["optical_geometry"] = {"outline_mm": polygon, "job_id": ctx.run_id}
        result["tracking_references"] = {"20x": {"job_id": ctx.run_id,
            "image": result["images"]["20x"], "polygon_px": ((1., 2.), (3., 4.)),
            "meta": {"fov_mm": (.1, .2)}}}
        return result
    monkeypatch.setattr(step, "_acquire_candidate", with_geometry)
    step.act(h.ctx, None)
    assert step._results == step._read_checkpoint(h.ctx)["completed"]
    assert step._results["f006"]["optical_geometry"]["outline_mm"] == [list(point) for point in polygon]
    step.on_success(h.ctx, None)
    replay = h.step()
    replay.act(h.ctx, None)
    assert replay._results == step._results
    replay.on_success(h.ctx, None)


def test_same_correction_transient_retry_retains_prefix_new_correction_reacquires(h, monkeypatch):
    h.state.stop = False
    step = h.step()
    correction = Correction("reacquire")
    def issue():
        h.store.emit("step.corrected", {"name": step.name, "action": correction.action, "params": {}}, run_id=h.ctx.run_id)
    issue()
    acquire = step._acquire_candidate
    failed = []
    def transient(ctx, flake):
        if flake["id"] == "f008" and not failed:
            failed.append(True)
            raise TransientError("declared current candidate interruption")
        return acquire(ctx, flake)
    monkeypatch.setattr(step, "_acquire_candidate", transient)
    with pytest.raises(TransientError):
        step.act(h.ctx, correction)
    original = deepcopy(step._results["f006"])
    f006_count = sum(fid == "f006" for fid, objective in h.state.seen)
    step.act(h.ctx, correction)
    assert step._results["f006"] == original
    assert sum(fid == "f006" for fid, objective in h.state.seen) == f006_count
    issue()
    step.act(h.ctx, correction)
    assert sum(fid == "f006" for fid, objective in h.state.seen) > f006_count


def test_stop_during_objective_restore_prevents_following_focus_motion(h):
    first = paused(h)
    anchor = first._read_checkpoint(h.ctx)["current"]
    original = h.scope.set_objective
    def stop_after_objective(value):
        original(value)
        h.state.cancel = True
    h.scope.set_objective = stop_after_objective
    h.state.cancel = h.state.stop = False
    h.state.motions = []
    with pytest.raises(StopRequested):
        h.step().act(h.ctx, None)
    assert h.state.motions == [("objective", anchor["objective"])]
    assert h.step()._read_checkpoint(h.ctx)["current"] == anchor
