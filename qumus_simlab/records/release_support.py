"""Pure original-journal contract for conditional simulated release evidence.

Consumes a complete global event interval, not a claimed trajectory subset. No
store, device, flow execution, pixels, World or mutable configuration is read.
Image bytes are separately checked by live sensing; this module binds original
immutable image identities and public encoders, including mixed display packets.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from qumus_simlab.core.optical_profiles import RELEASE_SUPPORT_ASSUMPTIONS

POLICY = "native20-observed-loaded-support-parked-release-v3"
ENCODER = "synthetic-contact-encoder-camera-v1"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _pose(payload):
    geometry = payload.get("geometry", {})
    microscope = geometry.get("microscope", {})
    pose = microscope.get("stamp_axes_xyz_mm")
    if not isinstance(pose, dict) or any(not _finite(pose.get(k)) for k in ("x", "y", "z")):
        raise ValueError("Original release encoder lacks finite public holder axes")
    return {k: pose[k] for k in ("x", "y", "z")}


def _display(preview):
    return (isinstance(preview, dict) and preview.get("purpose") == "motion_preview"
            and preview.get("verification") is False and not preview.get("phase"))



def command_encoder_readings(owned, command):
    """Precision packets plus narrowly paired display-only attachments.

    A display attachment is never a reading. Every command must retain typed
    raw geometry; malformed mixed packets cannot become display exceptions.
    """
    device, op = command["payload"]["device"], command["payload"]["command"]
    frames = [row for row in owned if row.get("kind") == "lab.frame"
              and row.get("payload", {}).get("device") == device
              and row["payload"].get("command") == op]
    readings = []
    for row in frames:
        payload = row["payload"]
        preview = payload.get("microscope_preview")
        display_only = (set(payload) == {"device", "command", "microscope_preview"}
            and _display(preview) and preview.get("immutable") is True
            and preview.get("source") == "synthetic_scene"
            and preview.get("state_binding") == "containing_lab.frame"
            and preview.get("mechanics_version") == 2 and preview.get("timestamp") == row.get("ts"))
        if display_only:
            if not any(raw["id"] < row["id"] and raw["ts"] == row["ts"]
                       and raw.get("run_id") == row.get("run_id") for raw in readings):
                raise ValueError("Motion display attachment has no original same-command/Clock precision companion")
            continue
        _encoder_projection(row)
        if row.get("run_id") != command.get("run_id"):
            raise ValueError("Original precision encoder belongs to another command owner")
        readings.append(row)
    if not readings:
        raise ValueError("Original motion command has no required typed precision readback")
    return readings

def _release_journal_certificate(events: list[dict], *, run_id: str, phase: dict, params: dict,
                                current: dict, current_receipts: dict,
                                digest: Callable) -> dict:
    """Derive actual time/total-Z variation from a complete original event prefix.

    `digest` is the application's fixed versioned content-hash projection, never
    a request/certificate option. Both adapters supply their existing trusted
    implementation. Recorded transport/trace scalars are not inputs here.
    """
    if not isinstance(events, list) or not isinstance(run_id, str) or not run_id:
        raise ValueError("Release journal requires its original global event rows")
    index = {}
    for row in events:
        ident = row.get("id")
        if (type(ident) is not int or ident <= 0 or ident in index or not _finite(row.get("ts"))
                or not isinstance(row.get("payload"), dict)):
            raise ValueError("Release journal has duplicate/invalid original event identity")
        index[ident] = row

    def receipt(ref, kind):
        row = index.get(ref.get("event_id")) if isinstance(ref, dict) else None
        if (not row or row.get("kind") != kind or row.get("run_id") != run_id
                or not _sha(ref.get("event_sha256")) or digest(row["payload"]) != ref["event_sha256"]):
            raise ValueError("Original release receipt is absent, foreign or changed")
        return row

    footprint_event = receipt(phase["release_footprint"], "stack.release_footprint")
    approach_event = receipt(phase["release_approach"], "stack.release_approach")
    footprint = footprint_event["payload"]
    approach = approach_event["payload"]
    if (phase.get("release_policy") != POLICY or params.get("release_policy") != POLICY
            or footprint.get("policy") != POLICY or footprint.get("profile") != RELEASE_SUPPORT_ASSUMPTIONS
            or footprint.get("footprint_captures") != 1 or footprint.get("stamp_id") != params.get("stamp_id")
            or footprint.get("load_ids") != params.get("flake_ids")
            or [b.get("flake_id") for b in footprint.get("loaded", [])] != params.get("flake_ids")
            or [b.get("material") for b in footprint.get("loaded", [])]
                != [params["materials"][f] for f in params["flake_ids"]]
            or footprint.get("design_contract_sha256") != digest(params.get("device_design") or params.get("direct_contact_contract"))
            or footprint.get("classification_sha256") != digest(params.get("contact_interior_calibration", {}))
            or footprint.get("pose_window_admission_sha256") != digest(params.get("design_pose_window_admission"))
            or footprint.get("witness_sha256") != digest(phase["release_reference"]["witness"])
            or approach.get("phase") != phase.get("name") or approach.get("footprint") != phase["release_footprint"]
            or approach.get("reference") != phase["reference_event"]):
        raise ValueError("Original release phase/load/profile binding changed")

    from qumus_simlab.records.device_design import DeviceDesignV1
    from qumus_simlab.records.geometry import DirectContactContract

    materials = [params["materials"][f] for f in params["flake_ids"]]
    if params.get("stamp_film") != "PC":
        raise ValueError("Observed support is limited to its explicit PC load profiles")
    if materials == ["Gr"] and not params.get("device_design"):
        contract = DirectContactContract.model_validate(params.get("direct_contact_contract"))
        if params["flake_ids"] != [contract.source_flake_id] or params.get("target_chip_id") != contract.target_intent["target_contact"]["target_chip_id"]:
            raise ValueError("Observed sole-Gr source/target contract changed")
    elif materials == ["hBN", "Gr"] and not params.get("direct_contact_contract"):
        design = DeviceDesignV1.model_validate(params.get("device_design"))
        if list(design.pickup_order) != params["flake_ids"] or design.target_chip_id != params.get("target_chip_id"):
            raise ValueError("Observed Hall-bar source/order/target contract changed")
    else:
        raise ValueError("Observed support cannot substitute for an unsupported legacy load")
    selection = ("flake_ids", "flake_chips", "materials", "stamp_id", "stamp_film", "procedure_contract", "transfer_profile", "release_policy")
    if footprint.get("selection_sha256") != digest({k: params.get(k) for k in selection}):
        raise ValueError("Observed load lost its original selected-source binding")

    reference = phase["reference"]
    reference_row = receipt(phase["reference_event"], "lab.frame")
    reference_receipts = phase.get("release_reference_receipts")
    if reference_receipts is None:
        # The original reference capture ID is immutable. Its same-clock public
        # encoder is selected by provenance, not by a mutable last-pose value.
        candidates = [row for row in events if row["id"] < reference_row["id"] and row.get("kind") == "lab.frame"
            and row.get("run_id") == run_id and row["ts"] == reference["captured_at"]
            and row["payload"].get("geometry", {}).get("microscope", {}).get("stamp_axes_xyz_mm")]
        if not candidates:
            raise ValueError("Release reference has no original same-clock public encoder")
        encoder = max(candidates, key=lambda row: row["id"])
        reference_receipts = {"capture": phase["reference_event"], "encoder": {
            "event_id": encoder["id"], "profile": ENCODER,
            "projection_sha256": digest(_encoder_projection(encoder))}}

    def capture(record, refs):
        row = receipt(refs["capture"], "lab.frame")
        preview = row["payload"].get("microscope_preview", {})
        if (row["payload"].get("device") != "microscope" or row["payload"].get("command") != "capture"
                or preview.get("purpose") != "analysis" or preview.get("immutable") is not True
                or preview.get("verification") is not False or preview.get("source") != "synthetic"
                or preview.get("phase") != phase["name"] or not _sha(record.get("sha256"))
                or not isinstance(record.get("image"), str) or not record["image"]
                or preview.get("image") != record["image"] or preview.get("image_sha256") != record["sha256"]
                or not _finite(record.get("captured_at")) or preview.get("timestamp") != record["captured_at"]
                or not 0 <= row["ts"]-record["captured_at"] <= 2.
                or any(preview.get(k) != record["meta"].get(k) for k in
                       ("objective", "focus_z", "focus_plane", "calibration", "stage_kinematics", "fov_mm"))):
            raise ValueError("Release capture is not its original immutable analysis publication")
        er = refs["encoder"]
        encoder = index.get(er.get("event_id"))
        if (not encoder or er.get("profile") != ENCODER or not _sha(er.get("projection_sha256"))
                or encoder.get("run_id") != run_id or encoder.get("kind") != "lab.frame"
                or encoder["id"] >= row["id"] or encoder["ts"] != record["captured_at"]
                or digest(_encoder_projection(encoder)) != er["projection_sha256"]):
            raise ValueError("Release capture lost its original same-clock public encoder")
        return row, _pose(encoder["payload"]), _encoder_projection(encoder)["camera_world_xy_mm"]

    native = footprint["frame"]
    nr, np, nc = capture(native, footprint["receipts"])
    rr, rp, rc = capture(reference, reference_receipts)
    cr, cp, cc = capture(current, current_receipts)
    if (not rr["id"] < nr["id"] < footprint_event["id"] < approach_event["id"] < cr["id"]
            or not reference["captured_at"] <= native["captured_at"] <= current["captured_at"]
            or nc != rc or rc != cc or any(np[k] != rp[k] or rp[k] != cp[k] for k in ("x", "y"))
            or any(record["meta"]["stage_kinematics"] != reference["meta"]["stage_kinematics"] for record in (native, current))
            or any(reference["meta"].get(k) != current["meta"].get(k) for k in
                   ("objective", "focus_z", "focus_plane", "calibration"))
            or reference["meta"]["objective"] != 5 or reference["meta"]["focus_plane"] != "chip"
            or native["meta"]["objective"] != 20 or native["meta"]["focus_plane"] != "stamp"):
        raise ValueError("Release capture order/held pose/camera/focus changed")
    focus = receipt(footprint["focus_event"], "microscope.focus")
    fp = focus["payload"]
    if (not rr["id"] < focus["id"] < nr["id"] or fp.get("objective") != 20 or fp.get("plane") != "stamp"
            or fp.get("focus_plane") != "stamp" or fp.get("focus_axis_calibration") != footprint["axis"]
            or fp.get("final_focus_readback_z_mm") != native["meta"]["focus_z"]
            or fp.get("holder_pose") != np or fp.get("stage_kinematics") != native["meta"]["stage_kinematics"]):
        raise ValueError("Loaded footprint focus is not its original exposure context")

    owners = [row for row in events if row.get("run_id") == run_id and row.get("kind") == "step.started" and row["id"] < rr["id"]]
    if not owners:
        raise ValueError("Release has no original Step owner")
    owner = max(owners, key=lambda row: row["id"])
    original_owner = receipt(phase["release_reference"]["witness"]["load"]["release_start"], "step.started")
    if owner != original_owner:
        raise ValueError("Release prefix lost its original source-witness Step boundary")
    lo, hi = owner["id"], cr["id"]
    rows = sorted((row for row in events if lo <= row["id"] <= hi), key=lambda row: row["id"])
    # Exporters retain unfiltered global events, and the store adapter reads the
    # same full interval. Missing global IDs are unknown evidence, not an excuse
    # to silently validate a filtered run-only or supplied command subset.
    if len(rows) != hi-lo+1 or any(row["id"] != lo+i for i, row in enumerate(rows)):
        raise ValueError("Release journal interval is incomplete (missing original global event IDs)")
    if (owner["payload"].get("name") != phase["name"]
            or any(row.get("run_id") == run_id and row.get("kind") in ("step.started", "step.finished")
                   for row in rows[1:])):
        raise ValueError("Release journal crosses an actual Step boundary")
    if len([row for row in rows if row.get("kind") == "stack.release_footprint"]) != 1:
        raise ValueError("A footprint cannot be reacquired or cherry-picked")

    optical = {receipt(ref, "device.command")["id"] for ref in footprint["optical_commands"]}
    heat = {receipt(ref, "device.command")["id"] for ref in approach["entry_heat_commands"]}
    if (not heat or any(not rr["id"] < ident < nr["id"] for ident in optical)
            or any(not footprint_event["id"] < ident < approach_event["id"] for ident in heat)):
        raise ValueError("Release optics/entry heat are not their original owned intervals")
    commands = [row for row in rows if row.get("kind") == "device.command"]
    current_pose = rp
    objective, focus_z = reference["meta"]["objective"], reference["meta"]["focus_z"]
    variation = 0.
    cumulative = {rr["id"]: 0.}
    used_optical, used_heat, restored = set(), set(), []
    trace, encoders_used, previous_time = [], [], reference["captured_at"]
    analysis_times = []
    for pos, command in enumerate(commands):
        ident, body = command["id"], command["payload"]
        if command.get("run_id") != run_id:
            raise ValueError("Foreign device work interrupted the original release journal")
        if ident < rr["id"]:
            if body.get("device") != "microscope" or body.get("command") != "capture":
                raise ValueError("Unexpected work before release reference")
        elif not previous_time <= command["ts"] <= current["captured_at"]:
            raise ValueError("Original release command Clock order changed")
        previous_time = command["ts"]
        following = commands[pos+1]["id"] if pos+1 < len(commands) else hi+1
        owned = [row for row in rows if ident < row["id"] < following]
        device, op, args = body.get("device"), body.get("command"), body.get("params", {})
        if device == "microscope":
            if ident > rr["id"] and ident < nr["id"]:
                if ident not in optical:
                    raise ValueError("Unexpected optics outside the one frozen footprint operation")
                used_optical.add(ident)
            if op == "capture":
                if ident > nr["id"] and (objective != 5 or focus_z != reference["meta"]["focus_z"]):
                    raise ValueError("Extra loaded-plane capture after the single footprint analysis")
                publications = [row for row in owned if row.get("kind") == "lab.frame"
                    and isinstance(row["payload"].get("microscope_preview"), dict)
                    and not _display(row["payload"]["microscope_preview"])]
                if len(publications) != 1:
                    raise ValueError("Capture requires exactly one immutable analysis publication; display alone is not evidence")
                publication = publications[0]
                preview = publication["payload"]["microscope_preview"]
                if (publication.get("run_id") != run_id or publication["payload"].get("device") != "microscope"
                        or publication["payload"].get("command") != "capture" or preview.get("phase") != phase["name"]
                        or preview.get("purpose") != "analysis" or preview.get("immutable") is not True
                        or preview.get("verification") is not False or preview.get("source") != "synthetic"
                        or not _sha(preview.get("image_sha256")) or not isinstance(preview.get("image"), str)
                        or not preview["image"] or not _finite(preview.get("timestamp"))
                        or not command["ts"] <= preview["timestamp"] <= publication["ts"]
                        or not 0 <= publication["ts"]-preview["timestamp"] <= 2.
                        or preview.get("objective") != objective or preview.get("focus_z") != focus_z):
                    raise ValueError("Original analysis publication image/owner/optical binding changed")
                timestamp = preview["timestamp"]
                if analysis_times and not 0 <= timestamp-analysis_times[-1] <= 2.:
                    raise ValueError("Original release capture cadence exceeded its unchanged deadline")
                analysis_times.append(timestamp)
                cumulative[publication["id"]] = variation
            elif op in ("set_focus", "set_objective"):
                if ident > nr["id"]:
                    expected = {"z": reference["meta"]["focus_z"]} if op == "set_focus" else {"magnification": 5}
                    if args != expected or ident >= approach_event["id"]:
                        raise ValueError("Unexpected release focus/objective after footprint analysis")
                    restored.append(op)
                elif ident >= focus["id"]:
                    raise ValueError("Unexpected optics after completed footprint autofocus")
                if op == "set_focus":
                    if not _finite(args.get("z")):
                        raise ValueError("Unknown release focus coordinate")
                    focus_z = args["z"]
                else:
                    if type(args.get("magnification")) is not int or args["magnification"] not in (5, 20):
                        raise ValueError("Unsupported release objective")
                    objective = args["magnification"]
            else:
                raise ValueError("Unexpected release microscope command")
        elif device == "stamp_heater":
            if ident not in heat or op != "set_target" or not _finite(args.get("celsius")) or not 0 < args["celsius"] <= 155.5:
                raise ValueError("Release heating is foreign, unbound or outside entry heat")
            used_heat.add(ident)
        elif device == "transfer_arm" and op == "move_axis_to" and args.get("axis") == "z":
            if ident <= approach_event["id"] or args.get("from_pose") != current_pose:
                raise ValueError("Release Z intent lost its actual settled prefix")
            readings = command_encoder_readings(owned, command)
            duration = args.get("duration_bound_s")
            if (not readings or not _finite(duration) or not 0 <= duration <= 1.95 or not _finite(args.get("value"))
                    or not phase["process_reference"]["minimum_z_mm"] <= args["value"] <= current_pose["z"]):
                raise ValueError("Release lacks its finite command/actual encoder stream")
            previous = current_pose
            for reading in readings:
                pose = _pose(reading["payload"])
                # LabTwin's manipulator field is rounded display state. The
                # typed raw geometry is the sole precision/variation source.
                _encoder_projection(reading)
                if (reading.get("run_id") != run_id or not command["ts"] <= reading["ts"] <= current["captured_at"]
                        or any(pose[k] != rp[k] for k in ("x", "y")) or pose["z"] > previous["z"]
                        or pose["z"] < phase["process_reference"]["minimum_z_mm"]):
                    raise ValueError("Release actual encoder/Clock/held-XY/floor changed")
                variation += abs(pose["z"]-previous["z"])
                previous = pose
                encoders_used.append({"event_id": reading["id"], "timestamp": reading["ts"], "pose": pose})
            roundoff = (len(readings)+1)*max(math.ulp(command["ts"]), math.ulp(readings[-1]["ts"]))
            if (readings[-1]["ts"]-command["ts"] > duration+roundoff
                    or abs(previous["z"]-args["value"]) > .005
                    or abs(previous["z"]-current_pose["z"]) > .025):
                raise ValueError("Release settled readback exceeds original duration/progress bounds")
            current_pose = previous
        else:
            raise ValueError("Unexpected stage/XY/load/device motion in original release prefix")
        trace.append({"event_id": ident, "event_sha256": digest(body), "timestamp": command["ts"]})
    if (not analysis_times or len(analysis_times) > 600 or not _finite(phase.get("started_at"))
            or not owner["ts"] <= phase["started_at"] <= reference["captured_at"]
            or current["captured_at"]-phase["started_at"] > 900. or variation > 8.):
        raise ValueError("Original release prefix exceeded its unchanged phase budget")
    if (used_optical != optical or used_heat != heat or restored != ["set_focus", "set_objective"]
            or current_pose != cp or nr["id"] not in cumulative or cr["id"] not in cumulative):
        raise ValueError("Original complete release prefix does not reach its settled current exposure")
    native_total, current_total = cumulative[nr["id"]], cumulative[cr["id"]]
    return {"method": "original-complete-release-journal-v1", "run_id": run_id,
        "first_event_id": lo, "last_event_id": hi, "event_count": len(rows),
        "event_ids_sha256": digest([row["id"] for row in rows]),
        "reference_receipts": reference_receipts, "footprint": phase["release_footprint"],
        "footprint_to_reference": {"elapsed_s": native["captured_at"]-reference["captured_at"],
                                   "z_total_variation_mm": native_total},
        "footprint_to_current": {"elapsed_s": current["captured_at"]-native["captured_at"],
                                 "z_total_variation_mm": current_total-native_total},
        "reference_to_current": {"elapsed_s": current["captured_at"]-reference["captured_at"],
                                 "z_total_variation_mm": current_total},
        "analysis_observations": len(analysis_times), "commands": trace, "encoders": encoders_used}


def _encoder_projection(row):
    payload = row["payload"]
    geometry = payload.get("geometry", {})
    microscope = geometry.get("microscope", {})
    camera = microscope.get("optical_axis_world_mm")
    if (geometry.get("source") != "synthetic" or geometry.get("mechanics_version") != 2
            or microscope.get("source") != "synthetic" or microscope.get("stamp_axes_frame") != "transfer_arm"
            or not isinstance(camera, (tuple, list)) or len(camera) != 2 or not all(map(_finite, camera))
            or not isinstance(payload.get("device"), str) or not isinstance(payload.get("command"), str)):
        raise ValueError("Original encoder/camera projection is unsupported")
    return {"profile": ENCODER, "event_id": row["id"], "run_id": row["run_id"], "kind": row["kind"],
        "timestamp": row["ts"], "device": payload["device"], "command": payload["command"],
        "geometry_source": geometry["source"], "mechanics_version": geometry["mechanics_version"],
        "microscope_source": microscope["source"], "axes_frame": microscope["stamp_axes_frame"],
        "holder_axes_xyz_mm": _pose(payload), "camera_world_xy_mm": list(camera)}


def release_journal_certificate(events: list[dict], *, run_id: str, phase: dict, params: dict,
                                current: dict, current_receipts: dict, digest: Callable) -> dict:
    """Reject malformed exported/retained records through one deliberate boundary."""
    try:
        return _release_journal_certificate(events, run_id=run_id, phase=phase, params=params,
            current=current, current_receipts=current_receipts, digest=digest)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("Malformed or incomplete original release journal") from exc
