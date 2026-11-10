"""Ring evidence reads raster changes, never optical-gap metadata."""
import pytest
from PIL import Image, ImageDraw

from qumus_simlab.perception.tracking import PixelCalibration, precontact_ring_evidence
from qumus_simlab.testbench.optics import WHITE_LIGHT_PROFILE
from qumus_simlab.testbench.render import Canvas, MicroscopeRenderer


def ring_image(gap):
    canvas = Canvas(480, 320, (160, 160, 160))
    MicroscopeRenderer._white_light_rings(canvas, {"signed_apex_gap_mm": gap,
        "dome_radius_mm": 1.5, "centre_dx_mm": .03, "centre_dy_mm": -.01,
        "current_contact_radius_mm": 0., "has_contact": False,
        "profile_id": WHITE_LIGHT_PROFILE}, 480/2.6, 320/1.72)
    return Image.frombytes("RGB", (480, 320), bytes(canvas.px))


@pytest.mark.parametrize("gap", [.0001, .0002, .0004, .0005])
def test_actual_low_power_white_light_pixels_supply_alternating_ring_cue(gap):
    before = Image.new("RGB", (480, 320), (160, 160, 160))
    cue = precontact_ring_evidence(before, ring_image(gap), PixelCalibration(2.6/480, -1.72/320))
    assert cue["found"], cue
    assert cue["centre_offset_mm"] == pytest.approx((.03, -.01), abs=.006)


@pytest.mark.parametrize("colour", [(10, 10, 10), (20, 90, 220)])
def test_uniform_dark_or_coloured_patch_is_not_ring_evidence(colour):
    before = Image.new("RGB", (480, 320), (160, 160, 160))
    after = before.copy()
    ImageDraw.Draw(after).ellipse((210, 130, 270, 190), fill=colour)
    assert not precontact_ring_evidence(before, after, PixelCalibration(2.6/480, -1.72/320))["found"]


def test_absent_or_unchanged_rings_are_unknown_not_success():
    before = Image.new("RGB", (480, 320), (160, 160, 160))
    assert not precontact_ring_evidence(before, before, PixelCalibration(.01, -.01))["found"]


def test_missing_or_ambiguous_radial_transitions_cannot_be_pooled():
    import math

    before = Image.new('RGB', (480, 320), (160, 160, 160))
    after = before.copy()
    after.putdata([(160+(10 if (round(math.hypot(x-240, y-160))//2) % 2 == 0 else 25), 160, 160)
                   if round(math.hypot(x-240, y-160)) <= 10 else (160, 160, 160)
                   for y in range(320) for x in range(480)])
    result = precontact_ring_evidence(before, after, PixelCalibration(2.6/480, -1.72/320))
    assert not result['found']
    assert result['candidates'][0]['coherent_derivative_changes'] == [0, 0]


def test_ring_metric_keeps_signed_image_to_motion_mapping():
    before = Image.new('RGB', (480, 320), (160, 160, 160))
    current = ring_image(.0004)
    direct = precontact_ring_evidence(before, current, PixelCalibration(2.6/480, -1.72/320))
    reversed_axes = precontact_ring_evidence(before, current, PixelCalibration(-2.6/480, 1.72/320))
    assert direct['selected'] == reversed_axes['selected']
    assert reversed_axes['centre_offset_mm'] == pytest.approx([-v for v in direct['centre_offset_mm']])
    assert direct['selected']['uncertainty_radius_native_px'] >= 1


@pytest.fixture
def bound_ring_phase(tmp_path):
    import copy
    import hashlib

    from qumus_simlab.config import load_yaml
    from qumus_simlab.core.clock import SimClock
    from qumus_simlab.flows.library.stacking import _digest
    from qumus_simlab.perception.tracking import ring_candidate_binding, v1_first_contact
    from qumus_simlab.runtime.store import RunStore

    calibration = PixelCalibration(2.6/480, -1.72/320)
    policy = load_yaml('verification.yaml')['transfer_sequence']
    clock = SimClock()
    store = RunStore(tmp_path/'run.db', clock=clock)
    rid = store.create_run('stack_device', {})
    name = 'v1_first_contact_stamp_01'
    metadata = {'objective': 5, 'focus_z': .3, 'focus_plane': 'chip',
                'calibration': {'source': 'synthetic', 'mm_per_pixel': [2.6/480, 1.72/320]},
                'stage_kinematics': {'unchanged_sample_pose': True}}
    def record(image, filename):
        path = tmp_path/filename
        image.save(path)
        value = {'image': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'captured_at': clock.now(), 'meta': copy.deepcopy(metadata)}
        payload = {'device': 'microscope', 'command': 'capture', 'microscope_preview': {
            'image': value['image'], 'image_sha256': value['sha256'], 'timestamp': clock.now(),
            'phase': name, **metadata}}
        ident = store.emit('lab.frame', payload, run_id=rid)
        return value, {'event_id': ident, 'event_sha256': _digest(payload)}
    before = Image.new('RGB', (480, 320), (160, 160, 160))
    current = ring_image(.0004)
    reference, reference_event = record(before, 'reference.png')
    clock.advance(.1)
    captured, current_event = record(current, 'current.png')
    candidate = {**v1_first_contact(before, current, calibration, min_area_mm2=policy['minimum_area_mm2']),
                 'reference': reference, 'current': captured}
    observation = precontact_ring_evidence(before, current, calibration)
    cue = {'profile': 'same-exposure-ring-centre-v1', 'reference': reference, 'current': captured,
           'reference_event': reference_event, 'current_event': current_event, 'observation': observation}
    cue['binding'] = ring_candidate_binding(cue, candidate, (2.6/480, 1.72/320))
    assert cue['binding']['compatible']
    candidate['aim_center_px'] = observation['selected']['centre_px']
    candidate['aim_center_source'] = 'same_exposure_ring_family'
    phase = {'name': name, 'reference_event': reference_event, 'ring_cue': cue}
    return store, rid, phase, candidate, calibration, policy


def test_same_exposure_binding_replays_original_captures(bound_ring_phase):
    from qumus_simlab.flows.library.stacking import _first_ring_binding

    store, rid, phase, candidate, calibration, policy = bound_ring_phase
    assert _first_ring_binding(store, rid, phase, candidate, calibration, policy)['compatible']


@pytest.mark.parametrize('change', ['stale', 'wrong_reference', 'wrong_center', 'minimum_area',
                                    'deleted_capture', 'capture_timestamp', 'capture_payload'])
def test_saved_ring_binding_rejects_changed_receipts(bound_ring_phase, change):
    import json

    from qumus_simlab.flows.library.stacking import _first_ring_binding

    store, rid, phase, candidate, calibration, policy = bound_ring_phase
    cue = phase['ring_cue']
    if change == 'stale':
        candidate['current'] = {**candidate['current'], 'captured_at': candidate['current']['captured_at']+1}
    elif change == 'wrong_reference':
        candidate['reference'] = {**candidate['reference'], 'captured_at': candidate['reference']['captured_at']-1}
    elif change == 'wrong_center':
        candidate['aim_center_px'] = candidate['centroid_px']
    elif change == 'minimum_area':
        candidate['minimum_area_mm2'] = policy['minimum_area_mm2']/2
    elif change == 'deleted_capture':
        store._x('DELETE FROM event WHERE id=?', (cue['current_event']['event_id'],))
    elif change == 'capture_timestamp':
        store._x('UPDATE event SET ts=ts+1 WHERE id=?', (cue['current_event']['event_id'],))
    else:
        store._x('UPDATE event SET payload_json=? WHERE id=?',
                 (json.dumps({'changed': True}), cue['current_event']['event_id']))
    with pytest.raises(ValueError):
        _first_ring_binding(store, rid, phase, candidate, calibration, policy)


@pytest.mark.parametrize('design', [True, False])
def test_first_contact_aims_at_the_v1_region_centroid_without_ring_cue(bound_ring_phase, monkeypatch, design):
    # #204: the same-frame ring-centre check (D8) is gone; a new design, too,
    # aims the one reaim at the v1 changed region's centroid.
    from types import SimpleNamespace

    from qumus_simlab.devices.base import Frame
    from qumus_simlab.flows.library import stacking

    store, rid, phase, fixture_candidate, calibration, policy = bound_ring_phase
    phase['reference'] = fixture_candidate['reference']
    state = {'source_references': {'f1': {'sha256': 'a'*64, 'focus_binding': {}}}, 'phases': {phase['name']: phase}}
    position = {'x': 40., 'y': 90., 'z': 6.5}
    ctx = SimpleNamespace(store=store, run_id=rid, params={'flake_ids': ['f1'], 'device_design': design},
        gate=SimpleNamespace(manipulator_z_floor=lambda _: 4.7))
    monkeypatch.setattr(stacking, '_profile', lambda _: policy)
    monkeypatch.setattr(stacking, '_transfer_state', lambda _: state)
    monkeypatch.setattr(stacking, '_validate_focus_reference', lambda *a, **k: None)
    monkeypatch.setattr(stacking, '_begin_phase', lambda *a, **k: phase)
    monkeypatch.setattr(stacking, '_image_calibration', lambda _: calibration)
    monkeypatch.setattr(stacking, '_pos', lambda _: dict(position))
    monkeypatch.setattr(stacking, '_save_phase', lambda *a: None)
    monkeypatch.setattr(stacking, '_save_transfer', lambda *a: None)
    monkeypatch.setattr(stacking, 'autofocus', lambda *a, **k: None)
    monkeypatch.setattr(stacking, '_ta', lambda _: {'safe_z': 6.5})
    monkeypatch.setattr(stacking, '_phase_capture', lambda *a: None)
    frames = [stacking._reference(fixture_candidate[key]) for key in ('reference', 'current')]
    assert all(isinstance(frame, Frame) for frame in frames)
    def move(_, __, target, *, observe=None, **kwargs):
        if observe:
            assert not observe(frames[0])
            assert observe(frames[1])
        else:
            position['z'] = target
        return True
    monkeypatch.setattr(stacking, '_guarded_z', move)
    original = stacking.precontact_ring_evidence
    ring_calls = []
    def rings(*args):
        ring_calls.append(args)
        return original(*args)
    monkeypatch.setattr(stacking, 'precontact_ring_evidence', rings)
    requested = []
    def reaim(self, _, current_phase, dx, dy):
        requested.append((dx, dy))
        current_phase['xy_inputs'] = []
        current_phase['pending_intent'] = {'axes': 'xy'}
        position['x'] += dx
        position['y'] += dy
        return dict(position)
    monkeypatch.setattr(stacking.AdjustFirstContact, '_reaim', reaim)
    step = stacking.AdjustFirstContact('stamp_01', phase['name'])
    step.act(ctx, None)
    result = step.result
    expected_center = fixture_candidate['centroid_px']
    assert result['centroid_px'] == fixture_candidate['centroid_px']
    assert requested[0] == pytest.approx(((240-expected_center[0])*calibration.mm_per_pixel_x,
        (160-policy['first_contact_offset_mm']/abs(calibration.mm_per_pixel_y)-expected_center[1])*calibration.mm_per_pixel_y))
    # A saved-evidence recomputation also reads these exact same candidate pixels.
    assert ring_calls == [] and 'aim_center_px' not in result


@pytest.mark.parametrize('change', [None, 'legacy_center_delta', 'axis_request', 'final_readback'])
def test_saved_completed_reaim_uses_exact_admitted_ring_center(bound_ring_phase, change):
    from qumus_simlab.flows.library.stacking import _first_ring_binding

    store, rid, phase, candidate, calibration, policy = bound_ring_phase
    center = candidate['aim_center_px']
    anchor = [240., 160.]
    target = [240., 160.-policy['first_contact_offset_mm']/abs(calibration.mm_per_pixel_y)]
    delta = [(target[0]-center[0])*calibration.mm_per_pixel_x,
             (target[1]-center[1])*calibration.mm_per_pixel_y]
    before = {'x': 40., 'y': 90., 'z': 6.5}
    x_after = {**before, 'x': before['x']+delta[0]}
    final = {**x_after, 'y': before['y']+delta[1]}
    candidate.update(image_anchor_px=anchor, first_contact_target_px=target,
        reaim_requested_mm=delta, reaim_readback=final,
        reaim_inputs=[{'axis': 'x', 'requested_mm': x_after['x'], 'before': before, 'after': x_after},
                      {'axis': 'y', 'requested_mm': final['y'], 'before': x_after, 'after': final}])
    phase['complete'] = True
    if change == 'legacy_center_delta':
        candidate['reaim_requested_mm'] = [(target[0]-candidate['centroid_px'][0])*calibration.mm_per_pixel_x,
                                         (target[1]-candidate['centroid_px'][1])*calibration.mm_per_pixel_y]
    elif change == 'axis_request':
        candidate['reaim_inputs'][0]['requested_mm'] += .001
    elif change == 'final_readback':
        candidate['reaim_readback'] = {**final, 'x': final['x']+.001}
    if change:
        with pytest.raises(ValueError):
            _first_ring_binding(store, rid, phase, candidate, calibration, policy)
    else:
        assert _first_ring_binding(store, rid, phase, candidate, calibration, policy)['compatible']


def _changed_region(shape):
    from qumus_simlab.perception.tracking import v1_first_contact

    before = Image.new("RGB", (480, 320), (160, 160, 160))
    after = before.copy()
    draw = ImageDraw.Draw(after)
    if shape == "large_disc":   # well over 256 external pixel edges
        draw.ellipse((150, 70, 330, 250), fill=(60, 60, 60))
    else:                       # a loop whose pocket meets the outside only at a pixel corner
        draw.rectangle((200, 120, 239, 159), fill=(60, 60, 60))
        draw.rectangle((240, 160, 279, 199), fill=(60, 60, 60))
        draw.rectangle((200, 160, 210, 199), fill=(60, 60, 60))
        draw.rectangle((200, 190, 279, 199), fill=(60, 60, 60))
    return v1_first_contact(before, after, PixelCalibration(2.6/480, -1.72/320), min_area_mm2=1e-4)


@pytest.mark.parametrize("shape", ["large_disc", "corner_pocket"])
def test_published_candidate_boundary_is_one_closed_loop(shape):
    """a518 seed 2 (combined head): a strided slice of more than 256 unordered
    pixel edges left the published boundary open and crashed the ring binding.
    A pocket that meets the outside only at a pixel corner is filled, so the
    loop stays simple there too."""
    from collections import Counter

    from qumus_simlab.perception.tracking import _ring_candidate_boundary

    candidate = _changed_region(shape)
    assert candidate["found"]
    segments = candidate["external_boundary_segments_px"]
    assert 3 <= len(segments) <= 256
    degrees = Counter(p for s in segments for p in (tuple(s[:2]), tuple(s[2:])))
    assert set(degrees.values()) == {2}
    fit = _ring_candidate_boundary(candidate, (2.6/480, 1.72/320))
    assert fit["mean_boundary_radius_px"] > 0
    if shape == "large_disc":
        assert candidate["boundary_display_sampled"] is True
        assert fit["boundary_center_px"] == pytest.approx((240, 160), abs=1.5)
        assert fit["mean_boundary_radius_px"] == pytest.approx(90, abs=2)


def test_degenerate_candidate_boundary_is_an_incompatible_refusal():
    from qumus_simlab.perception.tracking import _ring_candidate_geometry

    family = {"family_found": True, "centre_px": [240, 160], "uncertainty_radius_mm": .01}
    candidate = {"centroid_px": [240, 160], "external_boundary_segments_px":
                 [[0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 0, 2]]}
    result = _ring_candidate_geometry(family, candidate, (2.6/480, 1.72/320))
    assert result["compatible"] is False
    assert "open or has duplicated vertices" in result["reason"]


def _stamp_overlay(chip_sigma, contact_radius_mm=0.):
    import random

    import numpy as np

    canvas = Canvas(480, 320, (160, 160, 160))
    meta = {"stamp": {"film": True}, "optical_gap": {"signed_apex_gap_mm": .0002,
        "dome_radius_mm": 1.5, "centre_dx_mm": .03, "centre_dy_mm": -.01,
        "current_contact_radius_mm": contact_radius_mm, "has_contact": contact_radius_mm > 0,
        "profile_id": WHITE_LIGHT_PROFILE}}
    if contact_radius_mm:
        meta["contact"] = {"dx_mm": .03, "dy_mm": -.01, "radius_mm": contact_radius_mm}
    MicroscopeRenderer._stamp(canvas, meta, 0., 480/2.6, 320/1.72, random.Random(1), chip_sigma=chip_sigma)
    image = Image.frombytes("RGB", (480, 320), bytes(canvas.px))
    gray = np.asarray(image.convert("L"), dtype=float)
    return image, float((np.diff(gray, axis=0)**2).sum()+(np.diff(gray, axis=1)**2).sum())


@pytest.mark.parametrize("contact", [0., .05])
def test_rings_and_contact_share_the_chip_plane_defocus(contact):
    """a518 seed 2 (16x26 mm chip): rings drawn sharp at every focus made the
    post-candidate chip autofocus curve flat over a bare discovery site. They
    sit in a micrometre gap at the chip surface and now blur with it; at chip
    focus the rendered pixels are unchanged."""
    sharp, sharp_score = _stamp_overlay(0., contact)
    legacy = Canvas(480, 320, (160, 160, 160))
    assert sharp.tobytes() != bytes(legacy.px)
    scores = [_stamp_overlay(sigma, contact)[1] for sigma in (.8, 1.6, 2.8)]
    assert sharp_score > scores[0] > scores[1] > scores[2]
