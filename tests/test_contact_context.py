"""Public image-domain and original event-prefix admission controls."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from qumus_simlab.core.errors import NeedsHumanError
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library import stacking as m
from qumus_simlab.perception.contact_coverage import (
    parked_chip_support,
    source_projection_certificate,
)
from qumus_simlab.perception.tracking import PixelCalibration

PIXELS = PixelCalibration(2.6/480, -1.72/320)
AXIS = {"source": "synthetic", "simulation_only": True,
    "apex_lateral_drift_model": "componentwise-clipped-synthetic-stamp-v1",
    "apex_lateral_drift_speed_norm_bound_mm_s": .0000223606797749979,
    "apex_approach_drift_norm_bound_mm_per_mm": .0036055512754639895,
    "apex_drift_clip_per_axis_mm": .15, "dome_curvature_radius_mm": 1.5,
    "lateral_focus_slope_norm_bound_mm_per_mm": .01414213562373095,
    "source_refraction_projection_norm_bound_mm_per_mm": .005}


def empty_chip_field(size=(480, 320)):
    width, height = size
    yy, xx = np.mgrid[:height, :width]
    gain = (1-.14*((xx-width/2)/(width/2))**2)*(1-.14*((yy-height/2)/(height/2))**2)
    return Image.fromarray(np.floor(gain[..., None]*[24, 22, 28]).astype(np.uint8))


def test_parked_field_proves_only_existence_in_entire_expanded_fov():
    image = empty_chip_field()
    assert parked_chip_support(image, PIXELS)["found"] is False
    image.putpixel((240, 160), (120, 100, 170))
    result = parked_chip_support(image, PIXELS)
    assert result["found"] is True
    box = result["possible_chip_point_box_mm"]
    assert box[0] < -1.3 and box[2] > 1.3
    assert result["finite_raster_support_mm"][0] > 14*abs(PIXELS.mm_per_pixel_x)+.004


def test_parked_field_annotation_and_saturation_do_not_establish_chip():
    image = empty_chip_field()
    image.putpixel((200, 15), (130, 110, 180))
    image.putpixel((240, 160), (255, 255, 255))
    assert parked_chip_support(image, PIXELS)["found"] is False


def projection(**changes):
    values = dict(axis=deepcopy(AXIS), elapsed_s=182.843823433, z_variation_mm=.907353637118,
        z_delta_mm=.907353637118, source_box_mm=[-.3481176471, -.2381111112, .3481176471, .2381111112],
        front={"has_contact": True, "angular_coverage": 1., "film_intact": None,
               "interface_uncertainty_px": 1.0804089},
        front_bounds_px=[124.5, 3.5, 356.5, 235.5], size_px=[480, 320], calibration=PIXELS)
    values.update(changes)
    return source_projection_certificate(**values)


def test_declared_rate_and_actual_total_variation_precede_surface_branch_bound():
    result = projection()
    assert result["admitted"] is True
    assert result["apex_displacement_norm_bound_mm"] == pytest.approx(.007360022248)
    assert result["projection_displacement_bound_mm"] == pytest.approx(.004574088729)
    assert result["source_branch_distance_bound_mm"] < 1.5
    # The global clip alone is not enough to establish the fixed native guard.
    result = projection(elapsed_s=1e6)
    assert result["admitted"] is False


@pytest.mark.parametrize("change", [
    {"z_variation_mm": .1}, {"elapsed_s": True}, {"z_delta_mm": float("nan")},
    {"calibration": PixelCalibration(True, -.005)},
    {"front": {"has_contact": False}}, {"front_bounds_px": [0, 0, 0, 5]},
])
def test_projection_refuses_missing_or_invalid_bound_inputs(change):
    with pytest.raises(ValueError):
        projection(**change)


def test_branch_transition_is_never_given_a_global_lipschitz_certificate():
    result = projection(source_box_mm=[-1., -1., 1., 1.])
    assert result["source_branch_stays_eligible"] is False
    assert result["admitted"] is False


def test_r12_curvature_and_cap5_support_remain_separate_original_context_bounds():
    from qumus_simlab.records.discovery import discovery_policy

    axis = {**AXIS, "dome_curvature_radius_mm": 12., "cap_support_radius_mm": 5.,
            "stamp_discovery": discovery_policy()}
    result = projection(axis=axis, cap_center_mm=[0., 0.])
    assert result["admitted"] and result["dome_radius_mm"] == 12.
    assert result["cap_support_radius_mm"] == 5.
    assert result["cap_center_camera_mm"] == [0., 0.]
    assert "cap_center_camera_mm" not in projection()
    with pytest.raises(ValueError, match="nominal-head"):
        projection(axis=axis)
    changed = projection(axis=axis, cap_center_mm=[5., 0.])
    assert changed["admitted"] is False and changed["cap_support_stays_eligible"] is False


class Journal:
    def __init__(self, events):
        self.events = events

    def events_after(self, after_id=0, limit=500, run_id=None):
        return [e for e in self.events if e.id > after_id and (run_id is None or e.run_id == run_id)][:limit]


def event(identifier, kind, payload, timestamp=10.):
    return SimpleNamespace(id=identifier, ts=timestamp, kind=kind, payload=payload, run_id="r")


def trajectory():
    start = {"x": 40., "y": 90., "z": 6.5}
    end = {**start, "z": 6.4}
    command = {"device": m.MANIP, "command": "move_axis_to",
        "params": {"axis": "z", "value": 6.4, "from_pose": start}}
    encoder = {"device": m.MANIP, "command": "move_axis_to",
        "geometry": {"source": "synthetic", "mechanics_version": 2, "microscope": {
            "source": "synthetic", "stamp_axes_frame": "transfer_arm",
            "optical_axis_world_mm": [200., 300.], "stamp_axes_xyz_mm": end}}}
    events = [event(2, "device.command", command), event(3, "lab.frame", encoder),
        event(4, "device.command", {"device": m.MICROSCOPE, "command": "capture"}),
        event(5, "lab.frame", {})]
    return events, start, end


def replay(events, start, end):
    return m._contact_z_trajectory(Journal(events), "r", source_id=0, first=1, last=5, start=start, end=end)


def test_trajectory_is_id_bounded_even_when_following_motion_shares_timestamp():
    events, start, end = trajectory()
    events.append(event(6, "device.command", {"device": m.STAGE, "command": "move_to"}))
    variation, receipt = replay(events, start, end)
    assert variation == pytest.approx(.1)
    assert receipt[0]["encoder_event_ids"] == [3]
    assert receipt[0]["command_event_id"] == 2


@pytest.mark.parametrize("fault", ["stage", "xy", "missing_encoder", "broken_chain", "foreign_run", "wrong_end"])
def test_original_trajectory_refuses_motion_or_journal_discontinuity(fault):
    events, start, end = trajectory()
    if fault == "stage":
        events.insert(0, event(1, "device.command", {"device": m.STAGE, "command": "move_to"}))
    elif fault == "xy":
        events[0].payload["params"]["axis"] = "x"
    elif fault == "missing_encoder":
        events.pop(1)
    elif fault == "broken_chain":
        events[0].payload["params"]["from_pose"] = {**start, "z": 6.49}
    elif fault == "foreign_run":
        events[0].run_id = "other"
    else:
        end = {**end, "z": 6.3}
    with pytest.raises(ValueError):
        replay(events, start, end)


@pytest.mark.parametrize("paired", [True, False])
def test_contact_trajectory_uses_shared_precision_display_reader(paired):
    events, start, end = trajectory()
    attachment = {"device": m.MANIP, "command": "move_axis_to", "microscope_preview": {
        "purpose": "motion_preview", "verification": False, "phase": None, "immutable": True,
        "source": "synthetic_scene", "state_binding": "containing_lab.frame",
        "mechanics_version": 2, "timestamp": 10.}}
    events.insert(2, event(3.5, "lab.frame", attachment, timestamp=10. if paired else 10.001))
    if paired:
        assert replay(events, start, end)[0] == pytest.approx(.1)
    else:
        with pytest.raises(ValueError):
            replay(events, start, end)


def test_contact_trajectory_rejects_both_missing_precision_fields():
    events, start, end = trajectory()
    events.insert(2, event(3.5, "lab.frame", {"device": m.MANIP, "command": "move_axis_to"}))
    with pytest.raises(ValueError):
        replay(events, start, end)


def mixed_encoder_capture(tmp_path):
    """Portable mixed encoder/display packet, matching the actual Twin shape."""
    path = tmp_path/"analysis.png"
    Image.new("RGB", (80, 64), (122, 96, 168)).save(path)
    record = m._frame_record(Frame(m.MICROSCOPE, 10., path=str(path), meta={
        "objective": 5, "focus_z": .3, "focus_plane": "chip", "calibration": {}, "stage_kinematics": {}}))
    pose = {"x": 40., "y": 90., "z": 6.5}
    encoder = event(1, "lab.frame", {"device": m.MANIP, "command": "move_axis_to",
        "geometry": {"source": "synthetic", "mechanics_version": 2, "microscope": {
            "source": "synthetic", "stamp_axes_frame": m.MANIP,
            "stamp_axes_xyz_mm": pose, "optical_axis_world_mm": [675., 420.],
            "stamp": {"not_public_encoder_evidence": True}}},
        "microscope_preview": {"image": "unrelated-display.png", "purpose": "motion_preview",
                               "verification": False, "source": "synthetic_scene"}})
    capture = event(2, "lab.frame", {"device": m.MICROSCOPE, "command": "capture",
        "microscope_preview": {"phase": "pickup", "image": str(path), "image_sha256": record["sha256"],
            "timestamp": 10., **{key: record["meta"][key] for key in
                                ("objective", "focus_z", "focus_plane", "calibration", "stage_kinematics")}}})
    return Journal([encoder, capture]), record, encoder, capture


def test_contact_encoder_projection_ignores_display_but_preserves_strict_image_digest(tmp_path):
    journal, record, encoder, _ = mixed_encoder_capture(tmp_path)
    with pytest.raises(ValueError, match="matching content SHA"):
        m._digest(encoder.payload)  # Global image evidence stays strict.
    receipts = m._contact_capture_receipts(journal, "r", "pickup", record, 0)
    assert receipts["encoder"]["profile"] == "synthetic-contact-encoder-camera-v1"
    assert "projection_sha256" in receipts["encoder"] and "event_sha256" not in receipts["encoder"]
    actual = m._contact_public_capture(journal, "r", "pickup", record, receipts)
    assert actual == ({"x": 40., "y": 90., "z": 6.5}, [675., 420.])
    encoder.payload["microscope_preview"]["image"] = "another-display.png"
    encoder.payload["geometry"]["microscope"]["stamp"] = {"unused_packet_changed": True}
    assert m._contact_public_capture(journal, "r", "pickup", record, receipts) == actual


@pytest.mark.parametrize("fault", ["x", "camera", "run", "id", "kind", "timestamp", "device", "command",
                                   "source", "version", "frame", "missing_profile", "unknown_profile", "old_hash"])
def test_contact_encoder_receipt_rejects_changed_public_identity_or_fields(tmp_path, fault):
    journal, record, encoder, _ = mixed_encoder_capture(tmp_path)
    receipts = m._contact_capture_receipts(journal, "r", "pickup", record, 0)
    if fault == "x":
        encoder.payload["geometry"]["microscope"]["stamp_axes_xyz_mm"]["x"] += .001
    elif fault == "camera":
        encoder.payload["geometry"]["microscope"]["optical_axis_world_mm"][0] += .001
    elif fault in ("run", "id", "kind", "timestamp"):
        setattr(encoder, {"run": "run_id", "timestamp": "ts"}.get(fault, fault),
                {"run": "other", "id": 3, "kind": "other.event", "timestamp": 10.1}[fault])
    elif fault in ("device", "command"):
        encoder.payload[fault] = "changed"
    elif fault == "source":
        encoder.payload["geometry"]["source"] = "real"
    elif fault == "version":
        encoder.payload["geometry"]["mechanics_version"] = 3
    elif fault == "frame":
        encoder.payload["geometry"]["microscope"]["stamp_axes_frame"] = "world"
    elif fault == "missing_profile":
        del receipts["encoder"]["profile"]
    elif fault == "unknown_profile":
        receipts["encoder"]["profile"] = "future_or_unapproved"
    else:
        receipts["encoder"] = {"event_id": 1, "event_sha256": receipts["encoder"]["projection_sha256"]}
    with pytest.raises(ValueError):
        m._contact_public_capture(journal, "r", "pickup", record, receipts)


@pytest.mark.parametrize("fault", ["image_bytes", "capture_sha", "deleted_encoder"])
def test_projected_encoder_never_waives_original_analysis_image_or_event(tmp_path, fault):
    journal, record, _, capture = mixed_encoder_capture(tmp_path)
    receipts = m._contact_capture_receipts(journal, "r", "pickup", record, 0)
    if fault == "image_bytes":
        Image.new("RGB", (80, 64), (100, 90, 130)).save(record["image"])
    elif fault == "capture_sha":
        capture.payload["microscope_preview"]["image_sha256"] = "0"*64
    else:
        journal.events.pop(0)
    with pytest.raises((ValueError, NeedsHumanError)):
        m._contact_public_capture(journal, "r", "pickup", record, receipts)
