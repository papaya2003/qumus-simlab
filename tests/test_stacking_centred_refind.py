"""Stacking views centred on a chosen source flake use v1's centred-capture
area ceiling, so a large flake is not dropped by the scan ceiling."""
from types import SimpleNamespace

from qumus_simlab.flows.library import stacking


class _Analyzer:
    def __init__(self, backend):
        self.recognition_backend = backend
        self.calls = []

    def find_flakes(self, frame, material, **kwargs):
        self.calls.append(kwargs)
        return []


def test_v1_rgb_views_centred_on_a_source_use_the_centred_ceiling():
    an = _Analyzer("v1_lab_rgb")
    assert stacking._centred_flakes(SimpleNamespace(analyzer=an), object(), "hBN") == []
    assert an.calls == [{"centered": True}]


def test_other_analyzers_keep_their_plain_call():
    an = _Analyzer("sim_locator")
    stacking._centred_flakes(SimpleNamespace(analyzer=an), object(), "Gr")
    assert an.calls == [{}]


def test_refind_absence_and_reference_views_go_through_the_centred_helper():
    import inspect
    source = inspect.getsource(stacking)
    for marker in ("_centred_flakes(ctx, frame, self.material)\n                if math.hypot",
                   "detections = _centred_flakes(ctx, frame, self.material)\n            chosen",
                   "for d in _centred_flakes(ctx, frame, material):"):
        assert marker in source
