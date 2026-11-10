"""Local edge clearance and the separate legacy reactor field contract."""
from types import SimpleNamespace

import pytest

from qumus_simlab.flows.library import pickup_evidence


@pytest.mark.parametrize("field,detections,reason,clear", [
    ([2.5988, 1.7228], [(0., 0.)], None, True),
    ([2.5988, 1.7228], [], "target_not_detected", False),
    ([2.5988, 1.70], [(0., .001)], "inspection_radius_outside_image", False),
    ([2.5988, 1.7228], [(0., .02)], "inspection_radius_outside_image", False),
    ([2.5988, 1.7228], [(0., 0.), (.4, 0.)], None, False),
])
def test_actual_field_and_unknown_reason(monkeypatch, field, detections, reason, clear):
    scope = SimpleNamespace(get_objective=lambda: 20, set_objective=lambda _: None,
        capture=lambda **kwargs: SimpleNamespace(meta={"fov_mm": field}, width=3560, height=2360,
                                        saved_path=lambda: "captured.png"))
    detector = SimpleNamespace(find_flakes=lambda *args: [SimpleNamespace(dx_mm=x, dy_mm=y)
                                                        for x, y in detections])
    monkeypatch.setattr(pickup_evidence, "autofocus", lambda *args: None)
    monkeypatch.setattr(pickup_evidence, "analyzer", lambda _: detector)
    context = SimpleNamespace(gate=SimpleNamespace(microscope=lambda _: scope), verification={},
        bench={"microscope": {"fov_mm_10x": [1.3, .86]}}, clock=SimpleNamespace(now=lambda: 1.),
        run_id="observed", store=SimpleNamespace(bench="sim"))
    result = pickup_evidence.capture_neighbourhood(context, "flake", "chip", "Gr", local=False)
    assert result["clear"] is clear and result["clear_radius_mm"] == .85
    assert result["fov_mm"] == field
    assert [view["reason"] for view in result["views"]] == [reason]*3
    if reason:
        assert result["neighbours_per_view"] == [-1]*3


def square(x, half=.010):
    return SimpleNamespace(dx_mm=x, dy_mm=0., meta={"outline_complete": True,
        "outline_mm": [[-half, -half], [half, -half], [half, half], [-half, half]]})


@pytest.mark.parametrize("others,reason,clear", [
    ([], None, True),
    ([square(.100)], None, True),  # Within the old 850 um radius, outside the local gap.
    ([square(.035)], None, False),  # 15 um edge gap, despite centre distance >20 um.
    ([square(.050)], None, True),  # 30 um edge gap.
    ([square(.055, half=.030)], None, False),  # A wide neighbour's edge approaches.
])
def test_local_capture_uses_observed_edges_at_20x(monkeypatch, others, reason, clear):
    objectives, purposes = [], []
    frame = SimpleNamespace(meta={"fov_mm": [.65, .43]}, width=3560, height=2360,
                            saved_path=lambda: "observed.png")
    def capture(**kwargs):
        purposes.append(kwargs["purpose"])
        return frame
    scope = SimpleNamespace(get_objective=lambda: 100, set_objective=objectives.append, capture=capture)
    monkeypatch.setattr(pickup_evidence, "autofocus", lambda *args: None)
    monkeypatch.setattr(pickup_evidence, "analyzer", lambda _: SimpleNamespace(
        find_flakes=lambda *args: [square(0.), *others]))
    ctx = SimpleNamespace(gate=SimpleNamespace(microscope=lambda _: scope), verification={},
        bench={"microscope": {"fov_mm_10x": [1.3, .86]}}, clock=SimpleNamespace(now=lambda: 1.),
        run_id="observed", store=SimpleNamespace(bench="sim"))
    result = pickup_evidence.capture_neighbourhood(ctx, "flake", "chip", "Gr")
    assert result["clear"] is clear
    assert result["assessment_status"] == ("clear" if clear else "occupied")
    assert result["required_edge_clearance_mm"] == .020
    assert result["distance_basis"] == "observed_region_edge_to_edge"
    assert "clear_radius_mm" not in result
    assert objectives == [20, 100] and purposes == ["analysis_capture"]*3
    assert [v["reason"] for v in result["views"]] == [reason]*3


def test_local_field_must_cover_target_and_margin():
    reason, _ = pickup_evidence._local_clearance([square(0., half=.200)], [.65, .43], .020)
    assert reason == "local_neighbourhood_outside_image"


def test_unmeasured_boundary_does_not_become_clear():
    target = square(0.)
    target.meta["outline_complete"] = False
    reason, _ = pickup_evidence._local_clearance([target], [.65, .43], .020)
    assert reason == "observed_boundary_unavailable"


def clipped(x, half=.010):
    """A neighbour cut by the frame edge: only its in-frame part was observed."""
    ring = [[-half, -half], [half, -half], [half, half], [-half, half]]
    return SimpleNamespace(dx_mm=x, dy_mm=0., meta={"outline_complete": False,
        "visible_region_mm": {"version": "polygon-region-v1",
                              "components": [{"exterior": ring, "holes": []}]}})


@pytest.mark.parametrize("neighbour,clear", [
    (clipped(.300), True),   # Cut by the frame edge, far from the target.
    (clipped(.035), False),  # Its visible part is already within the margin.
])
def test_frame_clipped_neighbour_uses_its_visible_pixels(neighbour, clear):
    reason, distances = pickup_evidence._local_clearance([square(0.), neighbour], [.65, .43], .020)
    assert reason is None
    assert (distances[0] > .020) is clear


def test_neighbour_without_any_observed_pixels_stays_unknown():
    blind = SimpleNamespace(dx_mm=.3, dy_mm=0., meta={"outline_complete": False})
    reason, _ = pickup_evidence._local_clearance([square(0.), blind], [.65, .43], .020)
    assert reason == "observed_boundary_unavailable"


def test_rgb_local_view_asks_for_the_centred_capture_ceiling(monkeypatch):
    calls = []
    def find_flakes(frame, material, **kwargs):
        calls.append(kwargs)
        return [square(0.)]
    frame = SimpleNamespace(meta={"fov_mm": [.65, .43]}, width=3840, height=2160, saved_path=lambda: "v.jpg")
    scope = SimpleNamespace(get_objective=lambda: 20, set_objective=lambda _: None, capture=lambda **kw: frame)
    monkeypatch.setattr(pickup_evidence, "autofocus", lambda *args: None)
    monkeypatch.setattr(pickup_evidence, "analyzer", lambda _: SimpleNamespace(
        recognition_backend="v1_lab_rgb", find_flakes=find_flakes))
    ctx = SimpleNamespace(gate=SimpleNamespace(microscope=lambda _: scope), verification={},
        bench={"microscope": {"fov_mm_10x": [1.3, .86]}}, clock=SimpleNamespace(now=lambda: 1.),
        run_id="observed", store=SimpleNamespace(bench="sim"))
    assert pickup_evidence.capture_neighbourhood(ctx, "flake", "chip", "Gr")["clear"]
    assert calls == [{"centered": True}]*3
    calls.clear()
    pickup_evidence.capture_neighbourhood(ctx, "flake", "chip", "Gr", local=False)
    assert calls == [{}]*3  # the reactor's 5X view keeps the scan recognizer
