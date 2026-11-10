"""Native footprint producer controls; no World/job or real image estimator."""
from types import SimpleNamespace

import pytest

from qumus_simlab.core.errors import NeedsHumanError
from qumus_simlab.core.optical_profiles import RELEASE_SUPPORT_ASSUMPTIONS
from qumus_simlab.core.polygons import rectangle
from qumus_simlab.core.regions import region_dict
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library import stacking as m
from qumus_simlab.perception.observations import RGB_CARRIED_PROFILE
from qumus_simlab.perception.tracking import PixelCalibration


@pytest.mark.parametrize("problem", [None, "unknown", "foreign_profile"])
def test_native_footprint_keeps_measured_holes_and_refuses_unknown(tmp_path, monkeypatch, problem):
    path = tmp_path/"original.png"
    path.write_bytes(b"pure image estimator double")
    frame = Frame("microscope", 1., path=str(path), meta={"focus_plane": "stamp", "objective": 20,
        "calibration": {"observation_condition": {"observation_noise_scale": 1.},
            "release_support_profile": RELEASE_SUPPORT_ASSUMPTIONS, "image_size_px": [100, 100]}})
    region = region_dict({"components": [{"exterior": rectangle(.04, .03), "holes": [rectangle(.002, .002)]}]})
    measurement = {"measurement": {"status": "unknown" if problem == "unknown" else "measured", "region_exposure_mm": region}}
    detection = SimpleNamespace(material="Gr", thickness_label="manylayer", meta={
        "recognition_profile": "foreign" if problem == "foreign_profile" else RGB_CARRIED_PROFILE,
        "observation_plane": "stamp"})
    an = SimpleNamespace(recognition_backend="v1_lab_rgb", find_stamp_flakes=lambda f: [detection],
        plane_sharpness=lambda *a: 2., persist_detection_masks=lambda *a: {"original": True})
    scope = SimpleNamespace(get_objective=lambda: 5, get_focus=lambda: 0.,
        set_objective=lambda _: None, set_focus=lambda _: None,
        focus_transition_duration_bound=lambda *a: 0., focus_axis_calibration=lambda _: {})
    emitted = []
    store = SimpleNamespace(get_run=lambda _: {"meta": {"last_focus_event_id": 9,
                                                        "last_focus_observation": {"peak_observable": True}}},
        last_event_id=lambda: 10, emit=lambda kind, body, **kw: emitted.append(body) or 11)
    ctx = SimpleNamespace(store=store, run_id="unit", analyzer=an,
        gate=SimpleNamespace(microscope=lambda _: scope),
        verification={"focus": {"coarse_points": 1, "fine_points": 1}},
        params={"flake_ids": ["gr"], "materials": {"gr": "Gr"}, "stamp_id": "stamp"})
    phase = {"name": "unit", "reference": {"meta": {"focus_plane": "chip"}},
        "process_reference": {"reference_interval_mm": [0., 0.]}, "release_budget": {},
        "release_reference": {"witness": {}}, "reference_event": {"event_id": 1}}
    monkeypatch.setattr(m, "_transfer_state", lambda _: {"target_reference": {"focus_reference": {
        "focus_binding": {"observation": {"focus_axis_calibration": {"scale_mm_per_focus_mm": 1.}}},
        "meta": {"focus_z": 0.}}}})
    monkeypatch.setattr(m, "_pos", lambda _: {"z": 1.})
    monkeypatch.setattr(m, "_held_source", lambda _: {"held": True})
    monkeypatch.setattr(m, "_release_budget_guard", lambda *a: None)
    monkeypatch.setattr(m, "autofocus", lambda *a, **k: None)
    monkeypatch.setattr(m, "microscope_frame", lambda *a, **k: frame)
    monkeypatch.setattr(m, "_contact_capture_receipts", lambda *a: {"capture": {"event_id": 10}})
    monkeypatch.setattr(m, "min_sharpness", lambda *a: 1.)
    monkeypatch.setattr(m, "_image_calibration", lambda _: PixelCalibration(.01, -.01))
    monkeypatch.setattr(m, "_release_receipt", lambda store, i: {"event_id": i})
    monkeypatch.setattr(m, "_transfer_events", lambda *a: [])
    monkeypatch.setattr(m, "_save_phase", lambda *a: None)
    # The loaded footprint uses the original v1 selected mask (no RGB
    # refinement); this double stands for its calibrated exposure region.
    def original_mask(chain):
        assert chain == {"original": True}
        if measurement["measurement"]["status"] != "measured":
            raise ValueError("unsupported original mask")
        return measurement["measurement"]["region_exposure_mm"]
    monkeypatch.setattr(m, "_original_mask_region", original_mask)
    if problem:
        with pytest.raises(NeedsHumanError):
            m._acquire_release_footprint(ctx, phase)
        assert emitted == []
    else:
        m._acquire_release_footprint(ctx, phase)
        loaded = emitted[0]["loaded"][0]
        assert loaded["region_mm"] == region and "outline_mm" not in loaded
        assert loaded["recognition_chain"] == {"original": True}
        assert loaded["owner_error_budget"]["inclusive_error_budget_mm"] == .001
        assert loaded["recognition_profile"] == RGB_CARRIED_PROFILE


@pytest.mark.parametrize("peak", [True, False])
def test_film_focus_seed_comes_from_the_chip_refocused_through_the_stamp(monkeypatch, peak):
    """Seed 8: the parked target focus was observed without the stamp in the
    light path, which shifts the apparent focus 0.3 mm, so the film AF missed.
    The chip is refocused at 20x through the stamp and the film is offset."""
    scope = SimpleNamespace(focus=.29, objective=5, sets=[])
    scope.get_objective = lambda: scope.objective
    scope.get_focus = lambda: scope.focus
    scope.set_objective = lambda value: setattr(scope, "objective", value)
    def set_focus(value):
        scope.sets.append(value)
        scope.focus = value
    scope.set_focus = set_focus
    scope.focus_transition_duration_bound = lambda *a: 0.
    planes = []
    def autofocus(ctx, objective, attempts, half_range, plane, on_capture):
        planes.append((plane, scope.focus))
        if plane == "chip":
            scope.focus = .3
        else:
            raise RuntimeError("stop after the film seed")
    store = SimpleNamespace(get_run=lambda _: {"meta": {"last_focus_observation": {"peak_observable": peak}}})
    ctx = SimpleNamespace(store=store, run_id="unit", gate=SimpleNamespace(microscope=lambda _: scope),
                          verification={"focus": {"coarse_points": 1, "fine_points": 1}})
    phase = {"name": "unit", "reference": {"meta": {"focus_plane": "chip"}},
             "process_reference": {"reference_interval_mm": [5.2, 5.4]}, "release_budget": {}}
    monkeypatch.setattr(m, "_transfer_state", lambda _: {"target_reference": {"focus_reference": {
        "focus_binding": {"observation": {"focus_axis_calibration": {"scale_mm_per_focus_mm": 1.}}},
        "meta": {"focus_z": -.006}}}})
    monkeypatch.setattr(m, "_pos", lambda _: {"z": 6.5})
    monkeypatch.setattr(m, "_held_source", lambda _: {"held": True})
    monkeypatch.setattr(m, "_release_budget_guard", lambda *a: None)
    monkeypatch.setattr(m, "_save_phase", lambda *a: None)
    monkeypatch.setattr(m, "autofocus", autofocus)
    with pytest.raises(RuntimeError if peak else NeedsHumanError):
        m._acquire_release_footprint(ctx, phase)
    assert planes[0] == ("chip", .29)
    if peak:
        assert planes[1][0] == "stamp" and planes[1][1] == pytest.approx(.3+1.2)
    else:
        assert len(planes) == 1
