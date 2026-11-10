"""v1 rough refind angle as a stacking step: one corner photo, no stage move (#204)."""
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

from qumus_simlab.flows.library import search
from qumus_simlab.flows.library.search import MeasureCornerRotation
from qumus_simlab.flows.step import Verdict
from qumus_simlab.perception.v1_vision import CornerRotation

cv2 = pytest.importorskip("cv2")


class Store:
    def __init__(self, events, samples):
        self.events, self.samples = events, samples

    def events_after(self, after, limit=500, run_id=None):
        return [e for e in self.events if e.id > after and (run_id is None or e.run_id == run_id)][:limit]

    def get_sample(self, sample_id):
        return deepcopy(self.samples.get(sample_id))

    def upsert_sample(self, sample_id, kind, *, props):
        self.samples[sample_id]["props"].update(props)


def _event(i, run, name, verdict="ok", images=(), **measurements):
    return SimpleNamespace(id=i, run_id=run, kind="step.verified", ts=float(i),
        payload={"name": name, "verdict": verdict, "images": list(images), "measurements": measurements})


def _context(tmp_path, *, search_rotation=1.2, current_photo=True):
    ref, cur = tmp_path / "ref.png", tmp_path / "cur.png"
    for path in (ref, cur):
        cv2.imwrite(str(path), (cv2.randu(cv2.UMat(64, 64, cv2.CV_8UC3), 0, 255)).get())
    events = [_event(1, "search", "find_chip_corner", images=[str(ref)]),
              _event(2, "search", "measure_chip_rotation", rotation_deg=search_rotation),
              _event(3, "stack", "find_corner_chip", verdict="unclear"),
              *([_event(4, "stack", "find_corner_chip", images=[str(cur)])] if current_photo else [])]
    samples = {"chip": {"id": "chip", "kind": "chip", "props": {"rotation_deg": search_rotation}},
               "flake": {"id": "flake", "kind": "flake",
                         "props": {"tracking_references": {"20x": {"job_id": "search"}}}}}
    return SimpleNamespace(store=Store(events, samples), run_id="stack"), str(ref), str(cur)


def test_rotation_now_is_the_search_rotation_minus_v1_theta_cw(tmp_path, monkeypatch):
    ctx, ref, cur = _context(tmp_path)
    seen = []
    def match(reference, current):
        seen.append((reference.shape, current.shape))
        return CornerRotation(True, math.radians(-.5), 120, 90)
    monkeypatch.setattr("qumus_simlab.perception.v1_vision.corner_rotation", match)
    step = MeasureCornerRotation("chip", "flake", name="measure_rotation_chip")
    step.preconditions(ctx)
    obs = step.verify(ctx)
    assert obs.verdict == Verdict.OK and len(seen) == 1
    assert obs.measurements["rotation_deg"] == pytest.approx(1.7)
    assert obs.measurements["reference_image"] == ref and obs.images == [cur]
    step.on_success(ctx, obs)
    assert ctx.store.samples["chip"]["props"]["rotation_deg"] == pytest.approx(1.7)
    assert step.devices == () and step.max_corrections == 0


def test_unmatched_corner_photos_are_unclear_and_keep_the_old_rotation(tmp_path, monkeypatch):
    ctx, _, _ = _context(tmp_path)
    monkeypatch.setattr("qumus_simlab.perception.v1_vision.corner_rotation",
                        lambda a, b: CornerRotation(False, None, 3, 0, "match too weak"))
    obs = MeasureCornerRotation("chip", "flake", name="measure_rotation_chip").verify(ctx)
    assert obs.verdict == Verdict.UNCLEAR and "match too weak" in obs.message
    assert "rotation_deg" not in obs.measurements


def test_a_missing_current_corner_photo_is_a_precondition_failure(tmp_path):
    ctx, _, _ = _context(tmp_path, current_photo=False)
    with pytest.raises(search.PreconditionFailed):
        MeasureCornerRotation("chip", "flake", name="measure_rotation_chip").preconditions(ctx)
