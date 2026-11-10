"""v1 uses 100X only to classify: a 100X frame the recognizer cannot read keeps
the 20X catalogue entry (seed 8: the four largest hBN were lost this way)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from qumus_simlab.core.clock import SimClock
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library import pickup_evidence, search
from qumus_simlab.perception.base import FlakeDetection
from qumus_simlab.runtime.store import RunStore


@pytest.fixture
def revisit(tmp_path, monkeypatch):
    clock = SimClock(start=123.)
    store = RunStore(":memory:", clock=clock, bench="sim")
    rid = store.create_run("search_chip", {"chip_id": "chip", "material": "hBN"})
    store.upsert_sample("chip", "chip", props={"rotation_deg": 0., "corner_stage_xy": [0., 0.]})
    store.upsert_sample("flake", "flake", location="chip", status="candidate",
                        props={"material": "hBN", "stage_xy": [0., 0.], "labels": {"10x": "20-30"}})
    frames = {}
    for objective in (20, 100):
        path = tmp_path/f"frame{objective}.png"
        path.write_bytes(b"png")
        frames[objective] = Frame("microscope", clock.now(), path=str(path), meta={"objective": objective})

    class Scope:
        objective = 20

        def get_objective(self):
            return self.objective

        def get_focus(self):
            return 0.

        def set_objective(self, value):
            self.objective = value

        def capture(self):
            store.emit("device.command", {"device": "microscope", "command": "capture", "params": {}}, run_id=rid)
            return frames[self.objective]

    class Analyzer:
        recognition_backend = "v1_lab_rgb"

        def __init__(self):
            self.fail_100 = True

        def revisit_target(self, frame, material):
            if frame.meta["objective"] == 100 and self.fail_100:
                raise ValueError("Native RGB background lacks unambiguous frame/camera support")
            label = "20-30" if frame.meta["objective"] == 20 else "10-20"
            return FlakeDetection(dx_mm=0., dy_mm=0., area_um2=4000., material=material,
                                  thickness_label=label, confidence=0.)

    an, scope = Analyzer(), Scope()
    ctx = SimpleNamespace(store=store, run_id=rid, clock=clock, analyzer=an, params={}, project_id=None,
                          verification={"revisit": {"refind_radius_um": 100}},
                          bench={"microscope": {"revisit_objectives": [20, 100]}},
                          gate=SimpleNamespace(microscope=lambda name: scope, stage=lambda name: SimpleNamespace(
                              get_pose=lambda: SimpleNamespace(x=0., y=0.), get_fine_position=lambda: (0., 0.))), check_cancel=lambda: None)
    monkeypatch.setattr(search, "analyzer", lambda c: an)
    monkeypatch.setattr(search, "_move_stage", lambda *a, **kw: None)
    monkeypatch.setattr(search, "autofocus", lambda *a, **kw: 10.)
    monkeypatch.setattr(search, "_rgb_revisit_evidence", lambda *a, **kw: {
        "optical_geometry": {"source": "declared"}, "source_anchor20": {"stage_xy": [0., 0.]}}
        if a[2].meta["objective"] == 20 else {"classification_event_100x": 7})
    monkeypatch.setattr(pickup_evidence, "capture_neighbourhood", lambda *a: {"clear": True, "images": []})
    yield ctx, an, frames
    store.close()


def _run(ctx):
    step = search.RevisitCandidates("chip", "hBN", 1)
    step._targets = lambda c: [c.store.get_sample("flake")]
    step.act(ctx, None)
    step.on_success(ctx, None)
    return step._results["flake"], ctx.store.get_sample("flake")


def test_unreadable_100x_keeps_the_20x_entry_and_its_photo(revisit):
    ctx, _, frames = revisit
    result, flake = _run(ctx)
    assert result["found"] and "lost_at" not in result
    assert result["labels"] == {"20x": "20-30"}
    assert result["classification_100x"]["status"] == "unclassified"
    assert "frame/camera support" in result["classification_100x"]["reason"]
    assert flake["status"] == "catalogued" and flake["props"]["thickness_label"] == "20-30"
    assert flake["props"]["images"]["100x"] == frames[100].path
    assert flake["props"]["classification_100x"]["status"] == "unclassified"


def test_readable_100x_still_classifies(revisit):
    ctx, an, _ = revisit
    an.fail_100 = False
    result, flake = _run(ctx)
    assert result["labels"] == {"20x": "20-30", "100x": "10-20"}
    assert "classification_100x" not in result
    assert flake["props"]["thickness_label"] == "10-20" and flake["props"]["classification_100x"] is None


def test_graphite_gate_revisit_recovers_manylayer_like_v1(revisit):
    # v1 names a graphite-gate capture graphite_<mag>x; its colour recovery
    # then accepts only manylayer, so a monolayer speck beside graphite is not taken.
    ctx, an, _ = revisit
    an.fail_100 = False
    ctx.store.upsert_sample("flake", "flake", location="chip", status="candidate",
                            props={"material": "Gr", "stage_xy": [0., 0.], "labels": {"10x": "manylayer"}})
    seen = []
    plain = an.revisit_target

    def revisit_target(frame, material, **kwargs):
        seen.append(kwargs)
        return plain(frame, material)
    an.revisit_target = revisit_target
    step = search.RevisitCandidates("chip", "Gr", 1)
    step._targets = lambda c: [c.store.get_sample("flake")]
    step.act(ctx, None)
    assert seen == [{"accepted_labels": ("manylayer",)}]*2
