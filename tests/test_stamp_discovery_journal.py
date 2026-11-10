"""Executable mocked original journals; no World, services, motion or models."""

from copy import deepcopy

import pytest

from qumus_simlab.records.discovery import original_low_z_prefix, original_work_usage

PHASE = "v1_first_contact_stamp"
STAGE = {
    "sample_pose": {"x": 10.0, "y": 20.0, "z": 0.0, "rotation": 0.0},
    "coarse_pose": {"x": 10.0, "y": 20.0, "z": 0.0, "rotation": 0.0},
    "fine_xy_mm": [0.0, 0.0],
}
CAL = {"source": "synthetic", "image_size_px": [480, 320], "mm_per_pixel": [2.6 / 480, 1.72 / 320]}


def journal():
    rows = []
    clock = 99.9
    pose = {"x": 40.0, "y": 90.0, "z": 6.5}
    counter = 0

    def emit(kind, payload):
        row = {
            "id": len(rows) + 1,
            "ts": clock,
            "run_id": "r",
            "kind": kind,
            "payload": deepcopy(payload),
        }
        rows.append(row)
        return row

    def raw(device, command):
        return emit(
            "lab.frame",
            {
                "device": device,
                "command": command,
                "manipulator": {k: round(v, 3) for k, v in pose.items()},
                "geometry": {
                    "source": "synthetic",
                    "mechanics_version": 2,
                    "microscope": {
                        "source": "synthetic",
                        "stamp_axes_frame": "transfer_arm",
                        "stamp_axes_xyz_mm": dict(pose),
                        "stamp_holder_xyz": {"x": 200.0, "y": 300.0, "z": pose["z"]},
                        "optical_axis_world_mm": [200.0, 300.0],
                    },
                },
            },
        )

    def capture():
        nonlocal counter
        command = emit(
            "device.command", {"device": "microscope", "command": "capture", "params": {}}
        )
        raw("microscope", "capture")
        counter += 1
        digest = f"{counter:064x}"
        preview = {
            "purpose": "analysis",
            "phase": PHASE,
            "immutable": True,
            "verification": False,
            "source": "synthetic",
            "image": f"/mock/{counter}.png",
            "image_sha256": digest,
            "timestamp": clock,
            "objective": 5,
            "focus_z": 0.0,
            "focus_plane": "chip",
            "stage_kinematics": STAGE,
            "calibration": CAL,
            "fov_mm": [2.6, 1.72],
        }
        event = emit(
            "lab.frame",
            {"device": "microscope", "command": "capture", "microscope_preview": preview},
        )
        record = {
            "image": preview["image"],
            "sha256": digest,
            "captured_at": clock,
            "meta": {
                k: deepcopy(preview[k])
                for k in (
                    "calibration",
                    "objective",
                    "focus_z",
                    "focus_plane",
                    "stage_kinematics",
                    "fov_mm",
                )
            },
        }
        return record, event, command

    old, _, _ = capture()
    clock = 100.0
    previous, _, _ = capture()
    emit(
        "device.command",
        {
            "device": "transfer_arm",
            "command": "move_axis_to",
            "params": {
                "axis": "z",
                "from_pose": dict(pose),
                "value": 6.49975,
                "speed": 0.005,
                "duration_bound_s": 0.05,
            },
        },
    )
    clock += 0.05
    pose["z"] = 6.49975
    raw("transfer_arm", "move_axis_to")
    current, current_event, _ = capture()
    candidate = {"current": current, "reference": old, "actual_holder_pose": dict(pose)}
    for i in range(21):
        if i:
            emit(
                "device.command",
                {"device": "microscope", "command": "set_focus", "params": {"z": 0.0}},
            )
        capture()
        clock += 0.02
    af = emit(
        "microscope.focus",
        {
            "objective": 5,
            "plane": "chip",
            "peak_observable": True,
            "holder_pose": dict(pose),
            "stage_kinematics": STAGE,
        },
    )
    hold = {"started_at": clock, "start_event_id": af["id"], "held": [dict(pose), STAGE, 0.0, 5]}
    for _ in range(5):
        clock += 1.0
        capture()
    hold.update(completed_at=clock, end_event_id=rows[-1]["id"])
    emit(
        "device.command",
        {
            "device": "transfer_arm",
            "command": "move_axis_to",
            "params": {
                "axis": "z",
                "from_pose": dict(pose),
                "value": 6.51975,
                "speed": 0.1,
                "duration_bound_s": 0.2,
            },
        },
    )
    clock += 0.1
    pose["z"] = 6.50975
    raw("transfer_arm", "move_axis_to")
    clock += 0.1
    pose["z"] = 6.51975
    raw("transfer_arm", "move_axis_to")
    capture()
    discovery = {
        "first_contact_step": PHASE,
        "candidate_previous": previous,
        "post_candidate_focus_event_id": af["id"],
        "hold": hold,
        "clear_lift_end_event_id": rows[-1]["id"],
    }
    return {
        "rows": rows,
        "candidate": candidate,
        "discovery": discovery,
        "old_reference": old,
        "current_event_id": current_event["id"],
    }


def proof(case):
    return original_low_z_prefix(
        case["rows"], run_id="r", candidate=case["candidate"], discovery=case["discovery"]
    )


def test_real_mocked_order_derives_encoder_clock_and_complete_work():
    case = journal()
    result = proof(case)
    assert result["actual_downward_mm"] == pytest.approx(0.00025)
    assert result["first_lift_actual_mm"] == pytest.approx(0.02)
    assert result["hold_elapsed_s"] == 5.0
    assert result["low_z_elapsed_s"] == pytest.approx(5.67)
    usage = original_work_usage(case["rows"], run_id="r")
    assert usage["capture_count"] == 30
    assert usage["z_variation_mm"] == pytest.approx(0.02025)


@pytest.mark.parametrize(
    "fault",
    [
        "cached_pose",
        "stale_previous",
        "candidate_gap",
        "af_gap",
        "hold_start",
        "hold_end",
        "af_foreign",
        "af_unexpected_focus",
        "af_stage_command",
        "hold_stage_command",
        "hold_pose",
        "slow_first_lift",
        "downward_lift",
        "missing_raw",
        "both_missing",
    ],
)
def test_original_prefix_rejects_unbound_or_unsafe_intervals(fault):
    case = journal()
    rows, discovery = case["rows"], case["discovery"]
    afid = discovery["post_candidate_focus_event_id"]
    if fault == "cached_pose":
        case["candidate"]["actual_holder_pose"]["z"] -= 0.001
    elif fault == "stale_previous":
        discovery["candidate_previous"] = case["old_reference"]
    elif fault in ("candidate_gap", "af_gap"):
        boundary = case["current_event_id"] if fault == "candidate_gap" else afid - 1
        for row in rows:
            if row["id"] > boundary:
                row["ts"] += 3.0
                if "microscope_preview" in row["payload"]:
                    row["payload"]["microscope_preview"]["timestamp"] = row["ts"]
    elif fault == "hold_start":
        discovery["hold"]["started_at"] += 1.0
    elif fault == "hold_end":
        discovery["hold"]["completed_at"] += 0.01
    elif fault == "af_foreign":
        next(
            r
            for r in rows
            if r["kind"] == "device.command" and case["current_event_id"] < r["id"] < afid
        )["run_id"] = "other"
    elif fault == "af_unexpected_focus":
        next(
            r
            for r in rows
            if r["kind"] == "device.command" and r["payload"].get("command") == "set_focus"
        )["payload"]["params"]["z"] = 1.0
    elif fault == "af_stage_command":
        row = next(
            r
            for r in rows
            if r["kind"] == "device.command" and r["payload"].get("command") == "set_focus"
        )
        row["payload"] = {"device": "transfer_stage", "command": "move_to", "params": {"x": 1.0}}
    elif fault == "hold_stage_command":
        row = next(r for r in rows if r["kind"] == "device.command" and r["id"] > afid)
        row["payload"] = {"device": "transfer_stage", "command": "move_to", "params": {"x": 1.0}}
    elif fault == "hold_pose":
        row = next(
            r
            for r in rows
            if r["kind"] == "lab.frame" and r["id"] > afid and "geometry" in r["payload"]
        )
        row["payload"]["geometry"]["microscope"]["stamp_axes_xyz_mm"]["z"] -= 0.001
    elif fault == "slow_first_lift":
        for row in rows:
            if row["id"] > discovery["hold"]["end_event_id"]:
                row["ts"] += 2.0
    elif fault == "downward_lift":
        row = next(
            r
            for r in rows
            if r["kind"] == "device.command" and r["id"] > discovery["hold"]["end_event_id"]
        )
        row["payload"]["params"]["value"] = 6.49
    else:
        row = next(
            r
            for r in rows
            if r["kind"] == "lab.frame" and r["payload"].get("device") == "transfer_arm"
        )
        row["payload"].pop("geometry")
        if fault == "both_missing":
            row["payload"].pop("manipulator")
    with pytest.raises(ValueError):
        proof(case)


def display_attachment(case, *, paired=True):
    rows = case["rows"]
    raw = next(
        r for r in rows if r["kind"] == "lab.frame" and r["payload"].get("device") == "transfer_arm"
    )
    # Replace an existing non-evidence row following raw with a qualified
    # attachment while retaining a distinct actual capture encoder/publication.
    i = rows.index(raw)
    command = rows[i + 1]
    attachment = deepcopy(raw)
    attachment["id"] = raw["id"] + 0.5
    attachment["payload"] = {
        "device": "transfer_arm",
        "command": "move_axis_to",
        "microscope_preview": {
            "purpose": "motion_preview",
            "verification": False,
            "phase": None,
            "immutable": True,
            "source": "synthetic_scene",
            "state_binding": "containing_lab.frame",
            "mechanics_version": 2,
            "timestamp": raw["ts"],
        },
    }
    if not paired:
        attachment["ts"] += 0.001
    rows.insert(i + 1, attachment)
    mapping = {row["id"]: index + 1 for index, row in enumerate(rows)}
    for row in rows:
        row["id"] = mapping[row["id"]]
    discovery = case["discovery"]
    discovery["post_candidate_focus_event_id"] = mapping[discovery["post_candidate_focus_event_id"]]
    discovery["clear_lift_end_event_id"] = mapping[discovery["clear_lift_end_event_id"]]
    for key in ("start_event_id", "end_event_id"):
        discovery["hold"][key] = mapping[discovery["hold"][key]]
    case["current_event_id"] = mapping[case["current_event_id"]]
    return command


def test_explicit_display_attachment_needs_same_command_clock_raw_companion():
    case = journal()
    display_attachment(case)
    assert proof(case)["first_lift_actual_mm"] > 0.015
    assert original_work_usage(case["rows"], run_id="r")["z_variation_mm"] == pytest.approx(0.02025)
    bad = journal()
    display_attachment(bad, paired=False)
    with pytest.raises(ValueError):
        proof(bad)


def _first_af_publication(case):
    return next(r for r in case["rows"] if r["id"] > case["current_event_id"] and r["kind"] == "lab.frame"
                and r["payload"].get("microscope_preview", {}).get("purpose") == "analysis")


def test_af_baseline_may_repeat_the_candidate_exposure():
    """a518 seed 2: the AF baseline is a separate capture at the candidate's
    pose, focus and Clock time, so the SIM render repeats its digest."""
    case = journal()
    repeat = _first_af_publication(case)["payload"]["microscope_preview"]
    assert repeat["timestamp"] == case["candidate"]["current"]["captured_at"]
    repeat["image_sha256"] = case["candidate"]["current"]["sha256"]
    result = proof(case)
    assert result["candidate_event_id"] == case["current_event_id"]


def test_one_capture_published_twice_is_still_not_unique():
    case = journal()
    rows = case["rows"]
    original = next(r for r in rows if r["id"] == case["current_event_id"])
    raw = next(r for r in rows if r["id"] == original["id"] - 1)
    assert raw["payload"]["command"] == "capture" and raw["kind"] == "lab.frame"
    raw["payload"] = deepcopy(original["payload"])
    with pytest.raises(ValueError, match="not unique"):
        proof(case)
