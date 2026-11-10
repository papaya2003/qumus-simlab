"""Source-based v1 normal transfer, bounded through existing Steps/Gate/Clock.

One first-contact candidate and reaim per loaded stamp; later layers reuse its
working XY. Pickup uses fixed-reference image difference and Z-only descent.
Coverage and independent endpoint transfer confirmation are separate facts.
Historical L/focus/pixel constants are adapted by an explicit SIM-only profile.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from qumus_simlab.core.errors import NeedsHumanError, PreconditionFailed, StopRequested
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.aperture import CenterStampAperture, validate_aperture_alignment
from qumus_simlab.flows.bench import is_slot, reachable, slot_station, stamp_slots, to_world
from qumus_simlab.flows.bench import over_stage as head_over_stage
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.flow import Flow, Replan, register_flow
from qumus_simlab.flows.library.place_chip import PlaceChipOnTransferStage
from qumus_simlab.flows.library.primitives import (
    MICROSCOPE,
    analyzer,
    autofocus,
    look,
    microscope_frame,
    min_sharpness,
    ramp_heater,
    require_sample,
    wait_temperature,
)
from qumus_simlab.flows.library.search import (
    STAGE,
    STAGE_VACUUM,
    ChipToMicroscope,
    FindChipCorner,
    MeasureChipRotation,
    MeasureCornerRotation,
    MoveTransferStage,
    _chip_rotation,
    _local_to_stage,
    _move_stage,
)
from qumus_simlab.flows.library.transfer import MoveObject
from qumus_simlab.flows.step import (
    NOT_VERIFIABLE,
    Correction,
    NotVerifiable,
    Observation,
    ProbeResult,
    Step,
    Verdict,
)
from qumus_simlab.flows.workflow import Assembly, Molecule, WorkflowMismatch
from qumus_simlab.perception.tracking import (
    PixelCalibration,
    RegistrationConfig,
    load_pixels,
    precontact_ring_evidence,
    register_specimen,
    ring_candidate_binding,
    v1_first_contact,
    v1_pickup_coverage,
)
from qumus_simlab.records.device_design import (
    DesignPoseWindowAdmission,
    DeviceDesignV1,
    build_design_pose_window_admission,
    design_anchor_mm,
    target_import_binding,
    validate_design_artifact,
    validate_design_current,
    validate_design_pose_window_admission,
)
from qumus_simlab.records.geometry import (
    BridgeRequirement,
    DirectContactContract,
    contact_bridge_geometry,
    direct_contact_intent,
    direct_contact_result,
    geometry_digest,
    resolve_direct_contact_contract,
    target_contact_snapshot,
)

ARM = "arm_right"
CAMERA = "overview_camera"
MANIP = "transfer_arm"
MANIP_VACUUM = "transfer_arm_vacuum"
HEATER = "stamp_heater"  # heats the transfer stage (v1 transfer heater)
FAN = "transfer_stage_fan"
HANDOFF = "stamp_handoff"
REGRIP = "back to the load pose and grip again"


# ----- helpers ---------------------------------------------------------------

def _ta(ctx: FlowContext) -> dict[str, Any]:
    return ctx.bench["transfer_arm"]


def _pos(ctx: FlowContext) -> dict[str, float]:
    return ctx.gate.manipulator(MANIP).get_positions()


def _move(ctx: FlowContext, axis: str, value: float) -> None:
    ctx.check_cancel()
    ctx.gate.manipulator(MANIP).move_axis_to(axis, value)


def _over_stage(ctx: FlowContext) -> bool:
    return head_over_stage(ctx.bench, _pos(ctx))


def _travel(ctx: FlowContext, x: float, y: float, *, phase: dict | None = None) -> None:
    """Holder x/y travel along its two rails (bench.yaml transfer_arm): into
    the microscope it goes up (y) first, then left (x); everywhere else right
    (x) first, then down (y), so it never cuts across the module."""
    if phase is not None and [x, y] != [_ta(ctx)["park"][k] for k in ("x", "y")]:
        raise NeedsHumanError("Terminal release transport may only clear to the declared park")
    axes = (("y", y), ("x", x)) if head_over_stage(ctx.bench, {"x": x, "y": y}) else (("x", x), ("y", y))
    for axis, target in axes:
        if phase is None:
            _move(ctx, axis, target)
            continue
        _release_execution_binding(ctx.store, ctx.run_id, phase)
        tol, start = ctx.verification["manipulator"]["tolerance_mm"], _pos(ctx)
        if (ctx.gate.heater(HEATER).get_target() is not None or start["z"] < _ta(ctx)["safe_z"]-tol
                or "park_pending_intent" in phase):
            raise NeedsHumanError("Terminal park lacks confirmed withdrawal/off or has a pending rail intent")
        _release_budget_guard(ctx, phase)
        bound = ctx.gate.manipulator(MANIP).axis_motion_duration_bound(axis, start[axis], target)
        if not _finite(bound) or bound < 0:
            raise NeedsHumanError("Terminal park rail duration is unknown")
        intent = {"axis": axis, "from_pose": start, "requested_mm": target,
                  "duration_bound_s": bound, "started_at": ctx.clock.now()}
        phase["park_pending_intent"] = intent
        _save_phase(ctx, phase)
        cursor = ctx.store.last_event_id()
        ctx.gate.watchdog_tick()
        _move(ctx, axis, target)
        ctx.gate.watchdog_tick()
        ctx.check_cancel()
        actual, ended = _pos(ctx), ctx.clock.now()
        commands = [e for e in _transfer_events(ctx.store, ctx.run_id, "device.command", cursor)
            if e.payload.get("device") == MANIP and e.payload.get("command") == "move_axis_to"]
        if len(commands) != 1:
            raise NeedsHumanError("Terminal park lacks its actual rail command")
        command = commands[0]
        params = command.payload["params"]
        try:
            frames = _park_command_frames(ctx.store, ctx.run_id, command)
        except (KeyError, TypeError, ValueError) as exc:
            raise NeedsHumanError("Terminal park public encoder evidence is unavailable") from exc
        encoders = [e for e in frames if "manipulator" in e.payload]
        # Each public trajectory sample bounds one floating Clock addition;
        # account for summation roundoff, never for extra physical dwell time.
        roundoff = (len(frames)+1)*max(math.ulp(command.ts), math.ulp(frames[-1].ts))
        if (params.get("from_pose") != start or params.get("duration_bound_s") != bound
                or params.get("axis") != axis or params.get("value") != target or params.get("speed") is not None
                or not encoders
                or ended-intent["started_at"] > bound+roundoff or abs(actual[axis]-target) > tol
                or any(actual[k] != start[k] for k in ("x", "y", "z") if k != axis)):
            raise NeedsHumanError("Terminal park rail timing/readback changed; review", axis=axis,
                elapsed_s=ended-intent["started_at"], duration_bound_s=bound,
                from_pose=start, requested_mm=target, actual_pose=actual)
        encoder_pose = {k: encoders[-1].payload["manipulator"][k] for k in ("x", "y", "z")}
        if any(abs(actual[k]-encoder_pose[k]) > tol for k in encoder_pose):
            raise NeedsHumanError("Terminal park public encoder differs from actual readback")
        phase.setdefault("park_transport", []).append({**phase.pop("park_pending_intent"),
            "started_at": command.ts, "actual_pose": encoder_pose, "ended_at": frames[-1].ts,
            "encoder_end_event_id": frames[-1].id, "encoder_event_count": len(frames),
            "encoder_event_sha256": _park_frames_digest(frames), "clock_roundoff_s": roundoff,
            "command_event": {"event_id": command.id, "event_sha256": _digest(command.payload)}})
        _save_phase(ctx, phase)
        _release_execution_binding(ctx.store, ctx.run_id, phase)


def _near(p: dict[str, float], target: dict[str, float], tol: float) -> bool:
    return all(abs(p[a] - target[a]) <= tol for a in ("x", "y") if a in target)


PROCEDURE_ID = "v1-normal-sequence-v1"
TRANSFER_PROFILE_ID = "v1-normal-sim-v2"
RELEASE_POLICY_ID = "native20-latest-pickup-witness-parked-release-v2"
OBSERVED_RELEASE_POLICY_ID = "native20-observed-loaded-support-parked-release-v3"
# v1 molecule_drop_down for the PC Hall-bar design (#204): heat with autofocus
# every 10 s, lower to the fixed Z 4.9, wait above the PC melt, lift 5 mm;
# after the drop the target is only photographed for the archive.
V1_DROP_POLICY_ID = "v1-drop-down-photo-archive-v1"


def _fresh_release_policy(raw):
    """Only explicit typed PC Hall-bar/direct contracts enter the new policy."""
    ids, materials = raw.get("flake_ids", []), raw.get("materials", {})
    if raw.get("stamp_film") == "PC":
        if raw.get("device_design"):
            design = DeviceDesignV1.model_validate(raw["device_design"])
            if (list(design.pickup_order) == ids and [materials.get(f) for f in ids] == ["hBN", "Gr"]
                    and len(design.target.contacts) == 8):
                return V1_DROP_POLICY_ID
            raise ValueError("Unsupported typed Hall-bar release load")
        if raw.get("direct_contact_contract"):
            direct = DirectContactContract.model_validate(raw["direct_contact_contract"])
            if ids == [direct.source_flake_id] and materials == {ids[0]: "Gr"}:
                return OBSERVED_RELEASE_POLICY_ID
            raise ValueError("Unsupported typed direct-contact release load")
    return RELEASE_POLICY_ID

_WITNESS_SELECTION_FIELDS = ("flake_ids", "flake_chips", "materials", "stamp_id", "stamp_film",
                             "procedure_contract", "transfer_profile", "release_policy")
SOURCE_HEIGHT_MAPPING = "synthetic-native20-source-focus-positive-up-process-reference-v1"
EVIDENCE_DIGEST_ID = "qumus-image-content-evidence-v1"


def _profile(ctx: FlowContext) -> dict[str, Any]:
    c = ctx.verification["transfer_sequence"]
    if (ctx.store.bench != "sim" or c["source"] != "synthetic"
            or ctx.params.get("procedure_contract") != PROCEDURE_ID
            or ctx.params.get("transfer_profile") != TRANSFER_PROFILE_ID
            or c["id"] != TRANSFER_PROFILE_ID
            or c.get("holder_xy_mapping") != "synthetic-parallel-world-axes-v1"
            or c.get("source_height_mapping") != SOURCE_HEIGHT_MAPPING):
        raise PreconditionFailed("Transfer requires its current simulation-only procedure/profile binding")
    return c


def _approach_speed(c, start, target, contact_z, tol):
    """v1 Z profile (#179) for one pickup intent from the actual readback.

    Fast above a standoff over the expected contact, the touching speed
    through the standoff, the pickup press speed past the expected contact.
    Only the speed depends on where the intent starts: the target stays on
    the phase's own intent grid, so the final partial intent still reaches
    the process minimum through its admission instead of ending exactly on
    it. A readback within the actuator tolerance of a boundary counts as
    past it.
    """
    if start > contact_z+c["approach_standoff_mm"]+tol:
        return c["approach_speed_mm_s"], target
    if start > contact_z+tol:
        return c["descent_speed_mm_s"], target
    return c["press_speed_mm_s"], target


def _contact_speeds(c):
    """Z speeds that may carry the command right before a contact witness."""
    return (c["descent_speed_mm_s"], c["press_speed_mm_s"])


def _first_contact_sampling(verification):
    base = verification["transfer_sequence"]
    # v4 (#204): the touch search itself is v1's: 10 um jogs at the touching
    # speed, one frame per jog, the v1 near-round changed region. The coarse
    # touch, the same-frame ring-centre check and the 0.25 um fine cadence
    # of v2 (D6/D8, #128/#129) are gone. Only the landing beside the source
    # remains a v2 addition.
    return {"id": "hallbar-first-contact-v4", "source": "synthetic_sampling_policy",
        # v3 (D12, #136): the first touch lands beside the source, never on it.
        # A touch on the flake lifts it before the pickup step. The nearest
        # ring, 0.35 mm, exceeds a 10 um overshoot contact (~0.17 mm on
        # a 1.5 mm dome) plus the flake and apex error.
        # Landing search (#136, Jialei 2026-11-02): near to far from 0.35 mm,
        # 16 directions per ring (image-up first), the first clean landing
        # wins. Clean = on the measured chip, clear of catalogued flakes (disc
        # of twice the area-equivalent radius, aspect <= 4:1) and no visible
        # feature in the current 5x chip-plane image. The 0.25 mm landing disc
        # is the ~0.17 mm overshoot contact plus 0.08 mm apex allowance; it
        # must stay inside the observed 5x view so the touch is seen.
        "first_touch_min_mm": .35, "first_touch_step_mm": .05, "first_touch_directions": 16,
        "first_touch_landing_radius_mm": .25, "flake_extent_factor": 2.,
        "landing_feature_dn": 12., "landing_feature_min_px": 4, "landing_background_sigma_px": 12.,
        "touch_intent_mm": base["first_touch_intent_mm"], "min_z_mm": base["first_contact_min_z_mm"],
        "descent_speed_mm_s": base["descent_speed_mm_s"],
        "max_total_s": base["max_total_s"], "max_travel_mm": base["max_travel_mm"]}


def _phase_budget(ctx, phase):
    """One budget projection for capture, commands and saved-phase continuation."""
    base = _profile(ctx)
    if phase.get("release_policy") == V1_DROP_POLICY_ID:
        # v1 molecule_drop_down makes each Z move in one command with no
        # frame in between: to the preset, to Z 4.9 at 0.02 mm/s, and the
        # 5 mm lift at 0.1 mm/s (#204). The window covers the longest one.
        moves = ((_ta(ctx)["safe_z"]-base["initialize_z_mm"], base["initialize_speed_mm_s"]),
                 (base["initialize_z_mm"]-base["drop_z_mm"], base["drop_speed_mm_s"]),
                 (base["drop_lift_mm"], base["drop_lift_speed_mm_s"]))
        tol = ctx.verification["manipulator"]["tolerance_mm"]
        return {**base, "max_unobserved_z_mm": max(d for d, _ in moves)+2*tol,
                "max_unobserved_s": max(d/v for d, v in moves)+2*base["capture_reserve_s"]}
    if not ctx.params.get("device_design") or not phase.get("name", "").startswith("v1_first_contact_"):
        return base
    expected = _first_contact_sampling(ctx.verification)
    if ctx.params.get("first_contact_sampling") != expected or phase.get("sampling_policy") != expected:
        raise NeedsHumanError("First-contact sampling policy is missing or changed")
    if phase.get("reference"):
        receipt = phase.get("sampling_policy_event")
        if not receipt:
            raise NeedsHumanError("First-contact sampling lost its immutable policy receipt")
        event = _historical_event(ctx.store, receipt, ctx.run_id, "stack.first_contact_sampling")
        if event.payload != {"phase": phase["name"], "policy": expected, "reference_event": phase["reference_event"]}:
            raise NeedsHumanError("First-contact phase sampling receipt changed")
    # One v1 jog (10 um at 0.005 mm/s = 2 s) is one command and one frame, as
    # in v1, not split by the 2 s observation window into 9.75 + 0.25 um (#204).
    jog_s = expected["touch_intent_mm"]/expected["descent_speed_mm_s"]+base["capture_reserve_s"]
    return {**base, "max_unobserved_s": max(base["max_unobserved_s"], jog_s)}


def _transfer_state(ctx: FlowContext) -> dict[str, Any]:
    return deepcopy(ctx.store.get_run(ctx.run_id)["meta"].get("normal_transfer") or {})


def _save_transfer(ctx: FlowContext, state: dict[str, Any]) -> None:
    ctx.store.update_run_meta(ctx.run_id, normal_transfer=state)


def _save_phase(ctx: FlowContext, phase: dict[str, Any]) -> None:
    state = _transfer_state(ctx)
    state.setdefault("phases", {})[phase["name"]] = deepcopy(phase)
    _save_transfer(ctx, state)


def _transfer_events(store: Any, run_id: str | None, kind: str, after: int = 0):
    """Read the existing journal without silently truncating a long run."""
    while batch := store.events_after(after, limit=500, run_id=run_id):
        yield from (event for event in batch if event.kind == kind)
        after = batch[-1].id


def _pickup_partial_receipt(store: Any, run_id: str, phase: dict, events: list) -> dict | None:
    """Bind an optional one-shot prefix to its actual pre-command decision."""
    from qumus_simlab.config import load_yaml

    matches = [e for e in events if e.payload.get("phase") == phase["name"]]
    receipt = phase.get("final_partial_macro")
    if not matches and receipt is None:
        return None
    if len(matches) != 1 or not isinstance(receipt, dict):
        raise ValueError("Final pickup prefix lacks its unique reservation/event")
    event = _historical_event(store, receipt, run_id, "stack.pickup_macro_admission")
    body = receipt["admission"]
    c = load_yaml("verification.yaml")
    tol = c["manipulator"]["tolerance_mm"]
    minimum = phase["process_reference"]["minimum_z_mm"]
    start, requested, admitted = (body[k] for k in
        ("from_readback_z_mm", "requested_target_mm", "admitted_target_mm"))
    bounds = body["global_z_interval_mm"]
    if (event.id != matches[0].id or event.payload != body
            or body["phase"] != phase["name"] or body["profile"] != TRANSFER_PROFILE_ID
            or body["rule"] != "final-process-budget-prefix-v1"
            or receipt["status"] not in {"consumed", "covered", "no_coverage", "failed", "stopped"}
            or not all(_finite(v) for v in (start, requested, admitted, minimum, *bounds))
            or len(bounds) != 2 or bounds[0] > bounds[1]
            or body["readback_tolerance_mm"] != tol or tol <= 0
            or body["minimum_z_mm"] != minimum
            or body["requested_delta_mm"] != -c["transfer_sequence"]["pickup_intent_mm"]
            or requested != start+body["requested_delta_mm"]
            or admitted != math.nextafter(minimum+tol, math.inf)
            or body["admitted_delta_mm"] != admitted-start
            or not bounds[0] <= requested < minimum < admitted < start <= bounds[1]
            or admitted-tol < minimum or start-admitted <= 1e-8
            or body["process_reference_sha256"] != _digest(phase["process_reference"])
            or body["reference_sha256"] != phase["reference"]["sha256"]
            or type(body["admission_observation_count"]) is not int
            or not 1 <= body["admission_observation_count"] <= phase["observations"]
            or not _finite(body["admitted_at"]) or body["admitted_at"] < phase["started_at"]):
        raise ValueError("Final pickup prefix admission changed")
    capture = _historical_event(store, body["capture_event"], run_id, "lab.frame")
    preview = capture.payload["microscope_preview"]
    if (not phase["reference_event"]["event_id"] <= capture.id < event.id
            or preview["phase"] != phase["name"]
            or not phase["started_at"] <= preview["timestamp"] <= body["admitted_at"]
            or hashlib.sha256(Path(preview["image"]).read_bytes()).hexdigest() != preview["image_sha256"]):
        raise ValueError("Final pickup prefix lacks its historical observation")
    _reference(phase["reference"])
    return receipt


def _frame_record(frame: Frame) -> dict[str, Any]:
    path = frame.saved_path()
    if not path:
        raise NeedsHumanError("Transfer evidence requires a retained actual image")
    return {"image": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "captured_at": frame.timestamp,
            "meta": {k: deepcopy(frame.meta.get(k)) for k in
                     ("objective", "focus_z", "focus_plane", "fov_mm", "calibration",
                      "pose_binding", "stage_kinematics")}}


def _digest(value: Any) -> str:
    def project(item):
        if isinstance(item, dict):
            result = {k: project(v) for k, v in item.items()}
            locator = item.get("image")
            if isinstance(locator, dict):
                # v1 RGB recognition chains carry {"path", "sha256", ...} as
                # their image locator (seed 3 r12); bind it by content too.
                path, h = locator.get("path"), locator.get("sha256")
                if (not isinstance(path, str) or not path or not isinstance(h, str) or len(h) != 64
                        or any(c not in "0123456789abcdef" for c in h)):
                    raise ValueError("Malformed image evidence locator")
                result["image"] = {**result["image"], "path": "sha256:"+h}
                return result
            if "image" in item and locator is not None and (not isinstance(locator, str) or not locator):
                raise ValueError("Malformed image evidence locator")
            if isinstance(item.get("image"), str):
                hashes = [item[k] for k in ("image_sha256", "sha256") if k in item]
                if (not hashes or any(not isinstance(h, str) or len(h) != 64
                    or any(c not in "0123456789abcdef" for c in h) for h in hashes)
                        or len(set(hashes)) != 1):
                    raise ValueError("Image evidence requires its matching content SHA-256")
                result["image"] = "sha256:"+hashes[0]
            return result
        if isinstance(item, (list, tuple)):
            return [project(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(project(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _focus_summary(observation: dict) -> dict:
    from qumus_simlab.records.discovery import FOCUS_SUMMARY_FIELDS

    return {k: deepcopy(observation[k]) for k in FOCUS_SUMMARY_FIELDS}


def _historical_event(store: Any, binding: dict, run_id: str, kind: str) -> Any:
    event_id = binding["event_id"]
    if type(event_id) is not int or event_id <= 0:
        raise ValueError("Missing historical event")
    events = store.events_after(event_id-1, limit=1)
    if (not events or events[0].id != event_id or events[0].kind != kind or events[0].run_id != run_id
            or _digest(events[0].payload) != binding["event_sha256"]):
        raise ValueError("Historical event changed or is unavailable")
    return events[0]


def _verified_body(payload: dict) -> dict:
    # Images are separately checked against the actual content-bound frame records.
    return {k: payload[k] for k in ("seq", "name", "verdict", "measurements", "expected")}


def _verified_endpoint(store: Any, binding: dict, run_id: str, name: str, endpoint: dict, images: list[str]) -> None:
    event_id = binding["event_id"]
    if type(event_id) is not int or event_id <= 0:
        raise ValueError("Historical endpoint observation id is invalid")
    events = store.events_after(event_id-1, limit=1)
    if (not events or events[0].id != event_id or events[0].kind != "step.verified"
            or events[0].run_id != run_id):
        raise ValueError("Historical endpoint observation is unavailable")
    payload = events[0].payload
    if (payload["name"] != name or payload["verdict"] != "ok" or _digest(payload["measurements"]) != _digest(endpoint)
            or payload["images"] != images or _digest(_verified_body(payload)) != binding["observation_sha256"]):
        raise ValueError("Historical endpoint observation changed")


def _bind_verified_endpoint(ctx: FlowContext, name: str, obs: Observation, after_id: int) -> dict:
    events = [e for e in ctx.store.events_after(after_id, limit=32, run_id=ctx.run_id)
        if e.kind == "step.verified" and e.payload.get("name") == name
        and e.payload.get("verdict") == "ok" and _digest(e.payload.get("measurements")) == _digest(obs.measurements)
        and e.payload.get("images") == obs.images]
    if len(events) != 1:
        raise NeedsHumanError("Endpoint lacks its actual historical verified observation")
    return {"event_id": events[0].id, "observation_sha256": _digest(_verified_body(events[0].payload))}


def _historical_focus(store: Any, reference: dict, run_id: str) -> dict:
    binding = reference["focus_binding"]
    event_id = binding["event_id"]
    if type(event_id) is not int or event_id <= 0:
        raise ValueError("Missing historical autofocus event")
    events = store.events_after(event_id-1, limit=1)
    if (not events or events[0].id != event_id or events[0].kind != "microscope.focus"
            or events[0].run_id != run_id or _digest(events[0].payload) != binding["event_sha256"]
            or _focus_summary(events[0].payload) != binding["observation"]):
        raise ValueError("Historical autofocus event changed or is unavailable")
    return events[0].payload


#: Raster-only calibration fields. The v1 camera takes AF sweeps at the
#: preview raster and the analysis exposure on the full sensor, through the
#: same objective and field of view (fov_mm is compared separately).
_RASTER_CALIBRATION_KEYS = frozenset({"version", "image_size_px", "mm_per_pixel"})


def _same_capture_optics(capture: Any, focus: Any) -> bool:
    """The exposure and its AF share every optical assumption but the raster."""
    if not isinstance(capture, dict) or not isinstance(focus, dict):
        return False
    if capture == focus:
        return True
    strip = lambda c: {k: v for k, v in c.items() if k not in _RASTER_CALIBRATION_KEYS}  # noqa: E731
    if strip(capture) != strip(focus):
        return False
    for c in (capture, focus):
        size, scale = c.get("image_size_px"), c.get("mm_per_pixel")
        if (not isinstance(size, list) or not isinstance(scale, list) or len(size) != 2 or len(scale) != 2
                or not all(type(v) is int and v > 0 for v in size)
                or not all(_finite(v) and v > 0 for v in scale)):
            return False
    fov = [a*b for a, b in zip(capture["image_size_px"], capture["mm_per_pixel"], strict=True)]
    return all(math.isclose(a, b*n, rel_tol=1e-9) for a, b, n in
               zip(fov, focus["mm_per_pixel"], focus["image_size_px"], strict=True))


def _validate_focus_reference(reference: dict, run_id: str, *, parked: bool, store: Any) -> None:
    """Validate frozen historical evidence, never a current adapter query."""
    from qumus_simlab.config import load_yaml
    from qumus_simlab.flows.library.primitives import _observed_focus_interval
    from qumus_simlab.testbench.render import supported_photometry_profile

    binding = reference["focus_binding"]
    bench = load_yaml("bench.yaml")["transfer_arm"]
    tolerance = load_yaml("verification.yaml")["manipulator"]["tolerance_mm"]
    if (binding["park_xy_mm"] != [bench["park"][k] for k in ("x", "y")]
            or binding["clear_z_mm"] != bench["safe_z"] or binding["position_tolerance_mm"] != tolerance):
        raise ValueError("Source/target optical-path declaration differs from the v2 profile")
    af, af_frame = _historical_focus(store, reference, run_id), binding["af_frame"]
    meta = reference["meta"]
    axis = af["focus_axis_calibration"]
    interval = af["focus_interval_mm"]
    if (binding["digest_contract"] != EVIDENCE_DIGEST_ID
            or binding["profile"] != TRANSFER_PROFILE_ID or binding["run_id"] != run_id
            or binding["optical_path"] != ("parked_unobstructed" if parked else "held_target")
            or af["objective"] != 20 or af["objective_readback"] != 20 or meta["objective"] != 20
            or af["plane"] != "chip" or af["focus_plane"] != "chip" or meta["focus_plane"] != "chip"
            or af["peak_observable"] is not True
            or axis["source"] != "synthetic" or axis["simulation_only"] is not True
            or axis["calibration_id"] != "synthetic-microscope-focus-axis-v1"
            or axis["objective"] != 20 or axis["comparison_scope"] != "same_objective_only"
            or not all(_finite(axis[k]) and axis[k] >= 0 for k in
                       ("scale_mm_per_focus_mm", "mapping_error_mm", "readback_error_mm"))
            or axis["scale_mm_per_focus_mm"] <= 0
            or not isinstance(interval, list) or len(interval) != 2 or not all(map(_finite, interval))
            or not interval[0] <= af["final_focus_readback_z_mm"] <= interval[1]
            or af["readback_focus_z_mm"] != af["final_focus_readback_z_mm"]
            or meta["focus_z"] != af["final_focus_readback_z_mm"]
            or not _same_capture_optics(meta["calibration"], af["optical_calibration"])
            or meta["fov_mm"] != af["fov_mm"]
            or not supported_photometry_profile(meta["calibration"]["photometry_profile"])
            or meta["stage_kinematics"] != af["stage_kinematics"]
            or binding["holder_pose"] != af["holder_pose"]
            or af_frame["sha256"] != af["image_sha256"]
            or af_frame["captured_at"] != af["captured_at"]
            or not 0 <= reference["captured_at"]-af["captured_at"] <= 2
            or _observed_focus_interval(af["sampled_passes"], af, axis, 20)[0] != interval):
        raise ValueError("Native source/target focus evidence lacks its historical capture binding")
    if (not all(_finite(v) for v in (*binding["park_xy_mm"], binding["clear_z_mm"], binding["position_tolerance_mm"]))
            or binding["position_tolerance_mm"] <= 0):
        raise ValueError("Unknown parked optical-path declaration")
    if parked and (binding["holder_pose"]["z"] < binding["clear_z_mm"]-binding["position_tolerance_mm"]
                   or any(abs(binding["holder_pose"][k]-binding["park_xy_mm"][i]) > binding["position_tolerance_mm"]
                          for i, k in enumerate(("x", "y")))):
        raise ValueError("Source focus was not acquired in the unobstructed parked optical path")
    _image_calibration(_reference(reference))
    _reference(af_frame)
    captured = store.events_after(binding["capture_event_id"]-1, limit=1)
    if (not captured or captured[0].id != binding["capture_event_id"]
            or captured[0].kind != "lab.frame" or captured[0].run_id != run_id
            or _digest(captured[0].payload) != binding["capture_event_sha256"]):
        raise ValueError("Historical source/target exposure event is missing or changed")
    preview = captured[0].payload["microscope_preview"]
    if (preview["image"] != reference["image"] or preview["timestamp"] != reference["captured_at"]
            or any(preview.get(k) != meta[k] for k in ("objective", "focus_z", "focus_plane", "calibration", "stage_kinematics"))):
        raise ValueError("Historical source/target exposure does not match its AF context")


def _capture_chip_focus(ctx: FlowContext, phase: str, *, parked: bool,
                        focus_seed: dict | None = None, on_capture=None, on_af_capture=None) -> tuple[Frame, dict]:
    """One current native20 AF at the final source/target context, with frozen proof."""
    _profile(ctx)
    scope = ctx.gate.microscope(MICROSCOPE)
    scope.set_objective(20)
    options = {}
    if focus_seed is not None:
        seed = focus_seed["meta"]["focus_z"]
        half = _profile(ctx)["post_phase_focus_half_range_mm"]
        margin = half+2*half/ctx.verification["focus"]["coarse_points"]
        lo, hi = ctx.gate.limits.microscopes[MICROSCOPE].focus
        if not _finite(seed) or not lo <= seed-margin <= seed+margin <= hi:
            raise NeedsHumanError("Target wafer focus seed/window is outside its admitted envelope")
        scope.set_focus(seed)
        options.update(attempts=1, half_range=half)
    held = (_held_source(ctx), _pos(ctx))
    frames: list[Frame] = []
    autofocus(ctx, 20, plane="chip", retain_final_image=True, final_frames=frames,
              on_capture=on_capture if on_af_capture is None else on_af_capture, **options)
    af = deepcopy(ctx.store.get_run(ctx.run_id)["meta"].get("last_focus_observation") or {})
    frame = microscope_frame(ctx, phase=phase, on_capture=on_capture)
    _image_calibration(frame)
    quality = analyzer(ctx).plane_sharpness(frame, "chip")
    if (held != (_held_source(ctx, frame), _pos(ctx)) or not _finite(quality)
            or quality < min_sharpness(ctx, 20) or ctx.clock.now()-frame.timestamp > 2
            or frame.timestamp > ctx.clock.now() or not frames):
        raise NeedsHumanError("Current source/target focus capture is unobservable or changed context",
                              optical_observation_unavailable=True)
    record = _frame_record(frame)
    event_id = ctx.store.get_run(ctx.run_id)["meta"]["last_focus_event_id"]
    capture_events = [e for e in ctx.store.events_after(event_id, limit=32)
        if e.kind == "lab.frame" and e.run_id == ctx.run_id
        and e.payload.get("microscope_preview", {}).get("phase") == phase
        and e.payload["microscope_preview"].get("image") == record["image"]
        and e.payload["microscope_preview"].get("timestamp") == frame.timestamp]
    if len(capture_events) != 1:
        raise NeedsHumanError("Current source/target exposure event is not uniquely bound")
    record["focus_binding"] = {"profile": TRANSFER_PROFILE_ID, "run_id": ctx.run_id,
        "digest_contract": EVIDENCE_DIGEST_ID,
        "optical_path": "parked_unobstructed" if parked else "held_target",
        "event_id": event_id, "capture_event_id": capture_events[0].id,
        "capture_event_sha256": _digest(capture_events[0].payload),
        "event_sha256": _digest(af), "observation": _focus_summary(af),
        "af_frame": _frame_record(frames[-1]), "holder_pose": held[1],
        "park_xy_mm": [_ta(ctx)["park"][k] for k in ("x", "y")],
        "clear_z_mm": _ta(ctx)["safe_z"],
        "position_tolerance_mm": ctx.verification["manipulator"]["tolerance_mm"]}
    try:
        _validate_focus_reference(record, ctx.run_id, parked=parked, store=ctx.store)
    except (KeyError, TypeError, ValueError) as exc:
        raise NeedsHumanError("Current source/target lacks a unique calibrated native20 focus interval",
                              optical_observation_unavailable=True) from exc
    return frame, record


def _process_reference(state: dict, flake_id: str, stamp_id: str, run_id: str, max_press: float, store: Any,
                       *, current_reference: dict | None = None) -> dict:
    placement = state["placements"][stamp_id]
    discovery = state.get("discovery")
    phase = state["phases"][placement["first_contact_step"]]
    candidate = phase["first_contact_region"]
    evidence = _historical_event(store, placement["candidate_evidence"], run_id, "lab.evidence").payload
    if (evidence["subject"] != stamp_id or evidence["property"] != "first_contact_region"
            or evidence["bench"] != "sim" or _digest(evidence["value"]) != _digest(candidate)
            or evidence["source"]["kind"] != "synthetic"
            or evidence["source"]["reference"] != f"{run_id}:{placement['first_contact_step']}"
            or evidence["source"]["sha256"] != candidate["current"]["sha256"]
            or evidence["observed_at"] != candidate["current"]["captured_at"]):
        raise ValueError("Placement does not match the actual historical candidate")
    if (phase["profile"] != TRANSFER_PROFILE_ID or phase.get("complete") is not True
            or candidate["found"] is not True or candidate["profile"] != TRANSFER_PROFILE_ID
            or not _finite(candidate["actual_holder_pose"]["z"])
            or placement["candidate_z_mm"] != candidate["actual_holder_pose"]["z"]
            or placement["working_xy_mm"] != [(placement if discovery is not None else candidate)["reaim_readback"][k] for k in ("x", "y")]
            or not all(map(_finite, placement["working_xy_mm"]))):
        raise ValueError("Placement lacks its actual retained candidate/reaim readback")
    _reference(candidate["current"])
    anchor = placement["source_anchor"]
    restored_anchor = state["source_references"][anchor["source_id"]]
    if discovery is not None:
        _validate_discovery_ready(store, run_id)
        original = discovery["focus_reference"]
    else:
        original = restored_anchor
    current = state["source_references"][flake_id] if current_reference is None else current_reference
    if (original["meta"]["stage_kinematics"] != candidate["current"]["meta"]["stage_kinematics"]
            or original["captured_at"] > candidate["current"]["captured_at"]
            or original["focus_binding"]["capture_event_id"] >= placement["candidate_evidence"]["event_id"]):
        raise ValueError("First-source focus did not precede the candidate at its actual source context")
    for ref in (original, current):
        _validate_focus_reference(ref, run_id, parked=True, store=store)
    if (placement["profile"] != TRANSFER_PROFILE_ID or placement["run_id"] != run_id
            or placement["stamp_id"] != stamp_id or not _finite(placement["candidate_z_mm"]) or max_press != .2
            or anchor["focus_binding_sha256"] != _digest(restored_anchor["focus_binding"])
            or anchor["image_sha256"] != restored_anchor["sha256"]):
        raise ValueError("First-source process anchor changed")
    a, b = (ref["focus_binding"]["observation"] for ref in (original, current))
    if a["focus_axis_calibration"] != b["focus_axis_calibration"]:
        raise ValueError("Source focus calibration changed between layers")
    axis = a["focus_axis_calibration"]
    lo0, hi0 = a["focus_interval_mm"]
    lo1, hi1 = b["focus_interval_mm"]
    s, error = axis["scale_mm_per_focus_mm"], 2*axis["mapping_error_mm"]
    delta = ([0., 0.] if discovery is None and current_reference is None and flake_id == anchor["source_id"] else
             [s*(lo1-hi0)-error, s*(hi1-lo0)+error])
    interval = [placement["candidate_z_mm"]+v for v in delta]
    if not all(map(_finite, (*delta, *interval))):
        raise ValueError("Process reference is non-finite")
    return {"profile": TRANSFER_PROFILE_ID, "mapping": SOURCE_HEIGHT_MAPPING,
        "digest_contract": EVIDENCE_DIGEST_ID,
        "source_id": flake_id, "anchor_source_id": anchor["source_id"],
        "anchor_focus_binding_sha256": anchor["focus_binding_sha256"],
        "current_focus_binding_sha256": _digest(current["focus_binding"]),
        "first_candidate_z_mm": placement["candidate_z_mm"], "delta_interval_mm": delta,
        "reference_interval_mm": interval, "max_press_mm": max_press,
        "minimum_z_mm": interval[1]-max_press, "scope": "Optical process reference only; not contact/gap"}


def _target_baseline(state: dict, run_id: str, store: Any, chip_id: str, target: tuple[float, float]) -> dict:
    before = state["target_reference"]
    if (not isinstance(before.get("frames"), list) or len(before["frames"]) != 3
            or not isinstance(before.get("materials_found"), dict)
            or not all(type(v) is int and v >= 0 for v in before["materials_found"].values())):
        raise ValueError("Target baseline material observation is malformed")
    binding = state["target_reference_verified"]
    event_id = binding["event_id"]
    if type(event_id) is not int or event_id <= 0:
        raise ValueError("Target baseline observation id is unknown")
    events = store.events_after(event_id-1, limit=1)
    if not events or events[0].id != event_id or events[0].kind != "step.verified" or events[0].run_id != run_id:
        raise ValueError("Target baseline verified observation is unavailable")
    payload = events[0].payload
    if (before["target_chip_id"] != chip_id or before["target_xy_mm"] != list(target)
            or before["profile"] != TRANSFER_PROFILE_ID or before["run_id"] != run_id
            or payload["name"] != "go_to_target" or payload["verdict"] != "ok"
            or payload["measurements"].get("target_reference_sha256") != _digest(before)
            or payload["measurements"].get("on_chip") is not True
            or payload["expected"]["on_chip"] != chip_id
            or payload["expected"]["stage_xy"] != [round(v, 4) for v in before["requested_stage_xy_mm"]]
            or payload["images"] != [f["image"] for f in before["frames"]]
            or _digest(_verified_body(payload)) != binding["observation_sha256"]
            or not before["frames"][-1]["captured_at"] <= events[0].ts):
        raise ValueError("Historical target baseline classification/ROI/images changed")
    _validate_focus_reference(before["focus_reference"], run_id, parked=True, store=store)
    for frame in before["frames"]:
        _reference(frame)
    return before


def _witness_film_lineage(props: dict) -> dict:
    if (not isinstance(props, dict) or props.get("support") is not None and not isinstance(props["support"], dict)
            or props.get("support_history") is not None and (not isinstance(props["support_history"], list)
                or any(not isinstance(s, dict) for s in props["support_history"]))):
        raise ValueError("Stamp support lineage is malformed")
    reports = props.get("film_reports")
    if reports is not None:
        if not isinstance(reports, list) or any(not isinstance(r, dict) for r in reports):
            raise ValueError("Stamp film-report lineage is malformed")
        # Display attachments and report prose are not loaded-instance identity.
        reports = [{k: v for k, v in r.items() if k not in {"what", "images", "notes"}} for r in reports]
    return {"film": props.get("film"), "support": props.get("support"),
            "support_history": props.get("support_history"), "film_reports": reports}


def _witness_binding(event: Any, *, projection: str | None = None) -> dict:
    """Only new witness references project incidental exported locators/prose."""
    body = event.payload
    if not isinstance(body, dict):
        raise ValueError("Witness event body is malformed")
    if projection == "witness-manipulator-encoder-v1":
        body = {"device": body["device"], "command": body["command"], "timestamp": event.ts,
                "manipulator": {k: body["manipulator"][k] for k in ("x", "y", "z")}}
    elif event.kind == "job.submitted":
        if not isinstance(body["params"], dict):
            raise ValueError("Witness submission parameters are malformed")
        body = {"flow": body["flow"], "params": {k: body["params"].get(k) for k in _WITNESS_SELECTION_FIELDS}}
        projection = "witness-submission-selection-v1"
    elif event.kind == "sample.updated":
        body = {**{k: body.get(k) for k in ("id", "kind", "location", "status")},
                "props": {**_witness_film_lineage(body["props"]), "stack": body["props"].get("stack", [])}}
        projection = "witness-stamp-lineage-v1"
    elif event.kind == "stack.pickup_endpoint":
        body = {**{k: body[k] for k in ("stamp_id", "flake_id", "procedure")},
                "observation": {k: body["observation"][k] for k in ("verdict", "measurements", "expected")}}
        projection = "witness-pickup-endpoint-v1"
    return {"event_id": event.id, "event_sha256": _digest(body),
            **({"projection": projection} if projection else {})}


def _witness_checkpoint(store: Any, run_id: str, events: list, name: str) -> tuple[Any, Any, dict]:
    starts = [e for e in events if e.kind == "step.started" and e.payload.get("name") == name]
    finishes = [e for e in events if e.kind == "step.finished" and e.payload.get("name") == name]
    rows = [s for s in store.get_steps(run_id) if s["name"] == name]
    if len(starts) != 1 or len(finishes) != 1 or len(rows) != 1:
        raise ValueError("Witness lacks its unique original checkpoint history")
    start, finish, row = starts[0], finishes[0], rows[0]
    if (not start.id < finish.id or start.payload["seq"] != finish.payload["seq"]
            or row["seq"] != start.payload["seq"] or row["status"] != "done"
            or finish.payload["status"] != "done" or row["started_at"] != start.ts
            or row["finished_at"] != finish.ts):
        raise ValueError("Witness checkpoint was interrupted, skipped or replaced")
    return start, finish, row


def _witness_stamp_load(state: dict, params: dict, run_id: str, store: Any, events: list) -> tuple[dict, Any]:
    submitted = [e for e in events if e.kind == "job.submitted"]
    keys = _WITNESS_SELECTION_FIELDS
    if (len(submitted) != 1 or not isinstance(submitted[0].payload["params"], dict)
            or not params.get("flake_ids") or params.get("stamp_film") not in {"PC", "mica"}
            or any(submitted[0].payload["params"].get(k) != params.get(k) for k in keys)):
        raise ValueError("Release witness selection differs from the original submission")
    original = submitted[0].payload["params"]
    release_started = [e for e in events if e.kind == "step.started"
        and ("drop_down" in e.payload.get("name", "") or e.payload.get("name") == "release_material_metal_stack")]
    if not release_started:
        raise ValueError("Witness lacks the original release checkpoint boundary")
    end = release_started[0].id
    for event in events:
        if event.kind == "job.params_changed":
            path = event.payload["path"]
            if path and path[0] in keys or not path and any(event.payload["value"].get(k) != original.get(k) for k in keys):
                raise ValueError("Original witness/stamp selection was changed")
    load_start, load_finish, _ = _witness_checkpoint(store, run_id, events, "load_stamp")
    grip_start, grip_finish, _ = _witness_checkpoint(store, run_id, events, "grip_stamp")
    handoff = [e for e in events if load_start.id < e.id < load_finish.id and e.kind == "step.verified"
               and e.payload.get("name") == "load_stamp" and e.payload.get("verdict") == "ok"]
    vacuum = [e for e in events if grip_start.id < e.id < grip_finish.id and e.kind == "device.command"
              and e.payload.get("device") == MANIP_VACUUM and e.payload.get("command") == "set"]
    grip_observed = [e for e in events if grip_start.id < e.id < grip_finish.id and e.kind == "step.verified"
                     and e.payload.get("name") == "grip_stamp"]
    if (len(handoff) != 1 or len(vacuum) != 1 or vacuum[0].payload["params"].get("on") is not True
            or handoff[0].payload.get("seq") != load_start.payload["seq"]
            or len(grip_observed) != 1 or grip_observed[0].payload.get("verdict") != "not_verifiable"
            or grip_observed[0].payload.get("seq") != grip_start.payload["seq"]
            or not isinstance(handoff[0].payload.get("expected"), dict)
            or handoff[0].payload["expected"].get("sample") != params["stamp_id"]
            or handoff[0].payload["expected"].get("at") != HANDOFF
            or not submitted[0].id < load_start.id < load_finish.id < grip_start.id):
        raise ValueError("Release lacks its original successful stamp handoff/grip")
    # sample.updated is an existing global ledger event, not run-scoped.
    updates = [e for e in _transfer_events(store, None, "sample.updated", load_start.id) if e.id < end]
    if any(not isinstance(e.payload, dict) for e in updates):
        raise ValueError("Stamp ledger history is malformed")
    handed = [e for e in updates if handoff[0].id < e.id < load_finish.id and e.payload.get("id") == params["stamp_id"]]
    loaded = [e for e in updates if vacuum[0].id < e.id < grip_finish.id
              and e.payload.get("id") == params["stamp_id"]]
    if len(loaded) != 1 or len(handed) != 1:
        raise ValueError("Loaded stamp instance is unknown")
    stamp = loaded[0].payload
    from qumus_simlab.records.stamps import contact_film
    lineage = _witness_film_lineage(stamp["props"])
    handed_lineage = _witness_film_lineage(handed[0].payload["props"])
    if (handed[0].payload.get("kind") != "stamp" or handed[0].payload.get("location") != HANDOFF
            or handed[0].payload.get("status") != "fresh" or handed[0].payload["props"].get("stack", []) != []
            or stamp.get("kind") != "stamp" or stamp.get("location") != MANIP or stamp.get("status") != "in_use"
            or contact_film(stamp) != params["stamp_film"] or stamp["props"].get("stack", []) != []):
        raise ValueError("Original loaded stamp/film was not fresh and empty")
    if lineage != handed_lineage:
        raise ValueError("Film identity changed between handoff and grip")
    first = state["source_references"][params["flake_ids"][0]]
    if (grip_finish.id >= first["focus_binding"]["event_id"]
            or state["placements"][params["stamp_id"]]["source_anchor"]["source_id"] != original["flake_ids"][0]):
        raise ValueError("The first source anchor predates this stamp load")
    previous = 0
    for event in updates:
        if event.id >= end:
            break
        body = event.payload
        if body.get("kind") == "stamp" and body.get("location") == MANIP and body.get("id") != params["stamp_id"]:
            raise ValueError("Another stamp replaced the loaded witness instance")
        if event.id <= loaded[0].id or body.get("id") != params["stamp_id"]:
            continue
        if not isinstance(body["props"], dict):
            raise ValueError("Stamp ledger properties are malformed")
        stack = body["props"].get("stack", [])
        if (body.get("kind") != "stamp" or body.get("location") != MANIP or body.get("status") != "in_use"
                or _witness_film_lineage(body["props"]) != lineage
                or not isinstance(stack, list) or stack != original["flake_ids"][:len(stack)]
                or not previous <= len(stack) <= len(original["flake_ids"])):
            raise ValueError("Stamp load/film/ordered stack continuity changed")
        previous = len(stack)
    if previous != len(original["flake_ids"]):
        raise ValueError("Original loaded stamp lacks the complete ordered pickup history")
    for kind in ("device.command", "sample.removed", "stamp.film_replaced"):
        for event in _transfer_events(store, None, kind, vacuum[0].id):
            if event.id >= end:
                break
            body = event.payload
            if not isinstance(body, dict):
                raise ValueError("Stamp continuity event body is malformed")
            if (kind == "device.command" and body.get("device") == MANIP_VACUUM
                    or kind != "device.command" and body.get("id", body.get("stamp_id")) == params["stamp_id"]):
                raise ValueError("Loaded stamp was released, removed or re-filmed")
    return {"submission": _witness_binding(submitted[0]), "film": params["stamp_film"],
        "handoff_verified": {"event_id": handoff[0].id, "observation_sha256": _digest(_verified_body(handoff[0].payload))},
        "handoff_event": _witness_binding(handed[0]), "release_start": _witness_binding(release_started[0]),
        "grip_start": _witness_binding(grip_start), "grip_finish": _witness_binding(grip_finish),
        "grip_observation": _witness_binding(grip_observed[0]),
        "vacuum_command": _witness_binding(vacuum[0]), "stamp_event": _witness_binding(loaded[0])}, grip_finish


def _pickup_release_witness(state: dict, params: dict, run_id: str, verification: dict, store: Any) -> dict:
    events = list(_transfer_events(store, run_id, "step.started"))
    events += list(_transfer_events(store, run_id, "step.finished"))
    events += list(_transfer_events(store, run_id, "step.verified"))
    events += list(_transfer_events(store, run_id, "job.submitted"))
    events += list(_transfer_events(store, run_id, "job.params_changed"))
    events += list(_transfer_events(store, run_id, "device.command"))
    events.sort(key=lambda e: e.id)
    if any(not isinstance(e.payload, dict) for e in events):
        raise ValueError("Witness checkpoint history is malformed")
    load, previous = _witness_stamp_load(state, params, run_id, store, events)
    required = []
    for fid in params["flake_ids"]:
        name = pickup_name(params, fid)
        start, finish, row = _witness_checkpoint(store, run_id, events, name)
        phase, source = state["phases"][name], state["source_references"][fid]
        observed = [e for e in events if start.id < e.id < finish.id and e.kind == "step.verified"
            and e.payload.get("name") == name and e.payload.get("verdict") == "ok"]
        process = _process_reference(state, fid, params["stamp_id"], run_id, verification["pickup"]["max_press_mm"], store)
        if (len(observed) != 1 or not previous.id < start.id or phase.get("complete") is not True
                or observed[0].payload.get("seq") != row["seq"]
                or phase["process_reference"] != process or phase["endpoint_verified"]["event_id"] != observed[0].id
                or row["measurements"] != phase["endpoint"]
                or not _visual_pickup_proof(phase["endpoint"], source, store, run_id)):
            raise ValueError("A required original pickup lacks its independent verified endpoint")
        _verified_endpoint(store, phase["endpoint_verified"], run_id, name, phase["endpoint"],
                           [f["image"] for f in phase["endpoint"]["source_frames"]])
        if params.get("device_design"):
            _validate_design_pickup_additions(SimpleNamespace(params=params, store=store, run_id=run_id,
                verification=verification), phase, source, phase["endpoint"])
        endpoints = [e for e in _transfer_events(store, run_id, "stack.pickup_endpoint", observed[0].id)
                     if e.id < finish.id]
        if (len(endpoints) != 1 or not isinstance(endpoints[0].payload, dict)
                or not isinstance(endpoints[0].payload.get("observation"), dict)
                or endpoints[0].payload.get("stamp_id") != params["stamp_id"]
                or endpoints[0].payload.get("flake_id") != fid or endpoints[0].payload.get("procedure") != PROCEDURE_ID
                or endpoints[0].payload["observation"].get("verdict") != "ok"
                or _digest(endpoints[0].payload["observation"]["measurements"]) != _digest(phase["endpoint"])):
            raise ValueError("Original pickup endpoint belongs to another loaded stamp")
        required.append({"flake_id": fid, "verified": deepcopy(phase["endpoint_verified"]),
                         "pickup_endpoint": _witness_binding(endpoints[0])})
        previous = finish
    if previous.id >= load["release_start"]["event_id"]:
        raise ValueError("The required original pickups did not finish before release")
    # The loop deliberately selects the last REQUIRED item, without filtering.
    coverage = observed[0].payload["measurements"]["coverage"]
    if _digest(coverage) != _digest(phase["coverage"]):
        raise ValueError("Original verified coverage was replaced")
    baseline, current = coverage["reference"], coverage["current"]
    reference, image = _reference(baseline), _reference(current)
    calibration = _image_calibration(reference)
    c = verification["transfer_sequence"]
    pixels = load_pixels(reference)
    ox, oy = c["pickup_target_offset_image_mm"]
    target_px = (pixels.width/2+ox/abs(calibration.mm_per_pixel_x), pixels.height/2+oy/abs(calibration.mm_per_pixel_y))
    recomputed = v1_pickup_coverage(reference, image, calibration, target_px, margin_mm=c["pickup_margin_mm"],
        min_area_mm2=c["minimum_area_mm2"], blur_support_mm=c["pickup_blur_support_mm"])
    if (recomputed.get("covered") is not True or baseline["meta"]["objective"] != 5
            or _digest(recomputed) != _digest({k: coverage[k] for k in recomputed})
            or _digest(baseline) != _digest(phase["reference"])
            or any(baseline["meta"][k] != current["meta"][k] for k in ("objective", "focus_z", "calibration", "stage_kinematics"))
            or baseline["meta"]["stage_kinematics"] != source["meta"]["stage_kinematics"]):
        raise ValueError("Original pickup raster coverage/context no longer qualifies")
    captures = [e for e in _transfer_events(store, run_id, "lab.frame", start.id)
        if e.id < finish.id and e.payload.get("microscope_preview", {}).get("phase") == name
        and e.payload["microscope_preview"].get("image_sha256")]
    def capture_for(record):
        matches = [e for e in captures if e.payload["microscope_preview"].get("image") == record["image"]
                   and e.payload["microscope_preview"].get("timestamp") == record["captured_at"]]
        if len(matches) != 1:
            raise ValueError("Witness image lacks its original unique capture")
        capture = matches[0]
        preview = capture.payload["microscope_preview"]
        commands = [e for e in events if e.kind == "device.command" and start.id < e.id < capture.id]
        if (preview["image_sha256"] != record["sha256"] or any(preview.get(k) != record["meta"][k]
                for k in ("objective", "focus_z", "focus_plane", "calibration", "stage_kinematics"))
                or not commands or commands[-1].payload.get("device") != MICROSCOPE
                or commands[-1].payload.get("command") != "capture"
                or not commands[-1].ts <= record["captured_at"] <= capture.ts
                or record["captured_at"]-commands[-1].ts > c["max_unobserved_s"]):
            raise ValueError("Witness capture-time context changed")
        return capture
    first, capture = capture_for(baseline), capture_for(current)
    if not captures or first.id != captures[0].id or _witness_binding(first) != phase["reference_event"]:
        raise ValueError("Pickup witness baseline was rebased")
    commands = [e for e in events if e.kind == "device.command" and e.id < capture.id]
    camera = commands[-1]
    command = next((e for e in reversed(commands) if e.payload.get("device") == MANIP
                    and e.payload.get("command") == "move_axis_to"), None)
    if command is None:
        raise ValueError("Covered witness has no actual manipulator command")
    following = next((e for e in events if e.kind == "device.command" and e.id > command.id), None)
    if following is None:
        raise ValueError("Covered witness has no owned camera command")
    p, tol = command.payload["params"], verification["manipulator"]["tolerance_mm"]
    frames = _park_command_frames(store, run_id, command)
    encoders = [e for e in frames if "manipulator" in e.payload]
    settled = encoders[-1]
    pose = coverage["actual_holder_pose"]
    roundoff = (len(frames)+1)*max(math.ulp(command.ts), math.ulp(frames[-1].ts))
    if (not first.id < command.id < camera.id < capture.id < observed[0].id
            or following.id != camera.id or camera.payload.get("device") != MICROSCOPE
            or camera.payload.get("command") != "capture" or p.get("axis") != "z"
            or p.get("speed") not in _contact_speeds(c)
            or not _finite(p.get("value")) or not _finite(p.get("duration_bound_s"))
            or not 0 <= p["duration_bound_s"] <= c["max_unobserved_s"]-c["capture_reserve_s"]+1e-9
            or not 0 < p["from_pose"]["z"]-p["value"] <= c["max_unobserved_z_mm"]-tol+1e-9
            or p["value"]-tol < process["minimum_z_mm"]
            or not command.ts <= frames[0].ts <= frames[-1].ts <= camera.ts <= current["captured_at"] <= capture.ts
            or current["captured_at"]-command.ts > c["max_unobserved_s"]
            or frames[-1].ts-command.ts > p["duration_bound_s"]+roundoff
            or any(a.ts > b.ts for a, b in zip(frames, frames[1:], strict=False))
            or not all(_finite(v[k]) for v in (p["from_pose"], pose) for k in ("x", "y", "z"))
            or any(abs(pose[k]-settled.payload["manipulator"][k]) > tol for k in ("x", "y", "z"))
            or abs(pose["z"]-p["value"]) > tol or abs(settled.payload["manipulator"]["z"]-p["value"]) > tol
            or [pose["x"], pose["y"]] != phase["holder_xy"]
            or [p["from_pose"]["x"], p["from_pose"]["y"]] != phase["holder_xy"]
            or any(not command.ts <= e.ts <= camera.ts or any(abs(e.payload["manipulator"][k]-pose[k]) > tol
                   for k in ("x", "y")) or e.payload["manipulator"]["z"] < process["minimum_z_mm"] for e in encoders)):
        raise ValueError("Covered pickup segment lacks its actual command/settled encoder/capture ownership")
    partial = _pickup_partial_receipt(store, run_id, phase,
        list(_transfer_events(store, run_id, "stack.pickup_macro_admission")))
    tag = capture.payload["microscope_preview"].get("transfer_motion_intent")
    if partial is not None:
        if (partial["status"] != "covered" or not partial["event_id"] < command.id or not isinstance(tag, dict)
                or tag.get("macro_admission_event_id") != partial["event_id"] or tag.get("to_mm") != p["value"]
                or tag.get("from_mm") != p["from_pose"]["z"] or tag.get("speed_mm_s") != p["speed"]
                or tag.get("duration_bound_s") != p["duration_bound_s"] or tag.get("at") != command.ts
                or tag.get("macro_target_mm") != partial["admission"]["admitted_target_mm"]):
            raise ValueError("Covered final pickup prefix lost its original one-shot provenance")
    elif tag is not None:
        raise ValueError("Unbound pickup prefix tag")
    return {"flake_id": params["flake_ids"][-1], "phase": name, "seq": row["seq"], "load": load,
        "required_pickup_verified": required, "pickup_start": _witness_binding(start), "pickup_finish": _witness_binding(finish),
        "verified": deepcopy(phase["endpoint_verified"]), "source_process_reference_sha256": _digest(process),
        "source_focus_binding_sha256": _digest(source["focus_binding"]), "reference_capture": _witness_binding(first),
        "coverage_capture": _witness_binding(capture), "coverage_command": _witness_binding(command),
        "settled_encoder": _witness_binding(settled, projection="witness-manipulator-encoder-v1"),
        "encoder_end_event_id": frames[-1].id,
        "encoder_event_count": len(frames), "encoder_event_sha256": _park_frames_digest(frames),
        "coverage_sha256": _digest(coverage), "command_z_mm": p["value"], "readback_z_mm": pose["z"],
        "source_reference_upper_mm": process["reference_interval_mm"][1]}


def _release_reference(state: dict, params: dict, run_id: str, verification: dict, store: Any) -> tuple[dict, dict]:
    """Recompute one declared target-relative intent from immutable public inputs."""
    if params.get("release_policy") not in (RELEASE_POLICY_ID, OBSERVED_RELEASE_POLICY_ID):
        raise ValueError("Old or missing relative-release policy is read-only")
    observed = params["release_policy"] == OBSERVED_RELEASE_POLICY_ID
    if observed and _fresh_release_policy(params) != OBSERVED_RELEASE_POLICY_ID:
        raise ValueError("Observed release policy does not match its frozen typed load")
    target = _target_baseline(state, run_id, store, params["target_chip_id"], tuple(params["target_xy_mm"]))
    process = _process_reference(state, params["target_chip_id"], params["stamp_id"], run_id,
        verification["pickup"]["max_press_mm"], store, current_reference=target["focus_reference"])
    c = verification["transfer_sequence"]
    witness = _pickup_release_witness(state, params, run_id, verification, store)
    offset = witness["command_z_mm"]-witness["source_reference_upper_mm"]
    requested = process["reference_interval_mm"][1]+offset
    tol = verification["manipulator"]["tolerance_mm"]
    if (not all(map(_finite, (offset, requested, tol))) or tol <= 0
            or not tol-verification["pickup"]["max_press_mm"] <= offset < 0
            or requested-tol < process["minimum_z_mm"]):
        raise ValueError("Target-relative release intent exceeds its unchanged process budget")
    if observed:
        requested = math.nextafter(process["minimum_z_mm"]+tol, math.inf)
    return process, {"policy": params["release_policy"], "target_chip_id": params["target_chip_id"],
        "target_xy_mm": list(params["target_xy_mm"]), "target_before_sha256": _digest(target),
        "target_verified": deepcopy(state["target_reference_verified"]),
        "nominal_reference_z_mm": c["nominal_contact_z_mm"], "nominal_requested_z_mm": c["release_z_mm"],
        "relative_intent_mm": offset, "requested_z_mm": requested, "readback_tolerance_mm": tol,
        "witness": witness}


def _release_work(ctx: FlowContext, phase: dict) -> dict:
    """Reserve remaining mechanical/optical work, without assuming a thermal rate."""
    c, b = _profile(ctx), phase["release_budget"]
    m, scope = ctx.gate.manipulator(MANIP), ctx.gate.microscope(MICROSCOPE)
    tol = ctx.verification["manipulator"]["tolerance_mm"]
    progress = c["max_unobserved_z_mm"]-2*tol
    if not _finite(progress) or progress <= 0:
        raise NeedsHumanError("No finite accepted-progress reserve is available")
    def segments(start, target):
        return 0 if abs(start-target) <= 1e-8 else math.ceil(max(0., abs(start-target)-tol)/progress)+1
    z, safe, target = _pos(ctx)["z"], _ta(ctx)["safe_z"], phase["release_reference"]["requested_z_mm"]
    n = 0
    if b["stage"] in {"entry_heat", "footprint", "press"}:
        n += segments(z, target)+segments(target-tol, safe)
    elif b["stage"] in {"release_heat", "withdraw"}:
        n += segments(z, safe)
    park_n = 0 if b.get("park_complete") else max(0, 2-b["park_observations"])
    endpoint_n = b["endpoint_limit"]-b["endpoint_observations"]
    seconds = (n+park_n)*c["max_unobserved_s"]+endpoint_n*c["capture_reserve_s"]
    if not b.get("park_complete"):
        for axis in ("x", "y"):
            bound = m.axis_motion_duration_bound(axis, _pos(ctx)[axis], _ta(ctx)["park"][axis])
            if not _finite(bound) or bound < 0:
                raise NeedsHumanError("Safe park lacks a finite adapter duration reserve")
            seconds += bound
    if endpoint_n:
        seed = _transfer_state(ctx)["target_reference"]["focus_reference"]["meta"]["focus_z"]
        half = c["post_phase_focus_half_range_mm"]
        margin = half+2*half/ctx.verification["focus"]["coarse_points"]
        duration = getattr(scope, "focus_transition_duration_bound", lambda *args: None)
        initial, sweep = duration(scope.get_focus(), seed), duration(seed-margin, seed+margin)
        if any(not _finite(v) or v < 0 for v in (initial, sweep)):
            raise NeedsHumanError("Native endpoint lacks a finite focus duration reserve")
        seconds += initial+(ctx.verification["focus"]["coarse_points"]+
                            ctx.verification["focus"]["fine_points"]+3)*sweep
    pending = b.get("footprint_pending", 0)
    seconds += pending*b.get("footprint_focus_bound_s", 0.)
    return {"observations": n+park_n+endpoint_n+pending, "seconds": seconds,
            "z_travel_mm": (n+park_n)*c["max_unobserved_z_mm"]}


def _park_command_frames(store: Any, run_id: str, command: Any) -> list:
    """The actual command owns frames only until the next actual command."""
    following = next(_transfer_events(store, run_id, "device.command", command.id), None)
    end = following.id if following else math.inf
    frames = [e for e in _transfer_events(store, run_id, "lab.frame", command.id)
        if e.id < end and e.payload.get("device") == MANIP and e.payload.get("command") == "move_axis_to"]
    _park_frames_digest(frames)
    if not frames or not any("manipulator" in e.payload for e in frames):
        raise ValueError("Terminal park lacks its bounded public encoder stream")
    return frames


def _park_frames_digest(frames: list) -> str:
    if any("manipulator" in e.payload and (not isinstance(e.payload["manipulator"], dict)
            or any(not _finite(e.payload["manipulator"].get(k)) for k in ("x", "y", "z"))) for e in frames):
        raise ValueError("Malformed public encoder stream")
    return _digest([{"event_id": e.id, "timestamp": e.ts,
        "pose": {k: e.payload["manipulator"][k] for k in ("x", "y", "z")}
            if "manipulator" in e.payload else None} for e in frames])


def _release_execution_binding(store: Any, run_id: str, phase: dict) -> None:
    """A recomputed plan cannot replace the commands of a completed stroke."""
    commands = list(_transfer_events(store, run_id, "device.command", phase["reference_event"]["event_id"]))
    end = phase["release_budget"].get("endpoint_end_event_id", math.inf)
    commands = [e for e in commands if e.id <= end]
    z = [e for e in commands if e.payload.get("device") == MANIP
         and e.payload.get("command") == "move_axis_to" and e.payload.get("params", {}).get("axis") == "z"]
    requested = phase["release_reference"]["requested_z_mm"]
    if phase.get("release_policy") == OBSERVED_RELEASE_POLICY_ID:
        support = _validate_release_support(store, run_id, phase)
        if support["result"]["conditional_modeled_support"] is not True:
            raise ValueError("Completed release lacks its original conditional support")
        requested = support["actual_command_z_mm"]
    presses = [e for e in z if _finite(e.payload["params"].get("value"))
               and abs(e.payload["params"]["value"]-requested) <= 1e-8]
    if not presses:
        raise ValueError("Completed release lacks its target-relative stroke command")
    withdrawals = [e for e in z if e.id > presses[0].id and _finite(e.payload["params"].get("value"))
        and abs(e.payload["params"]["value"]-phase["withdrawal"]["requested_sim_target_mm"]) <= 1e-8]
    if (not withdrawals or not any(e.id > withdrawals[0].id and e.payload.get("device") == HEATER
            and e.payload.get("command") == "off" for e in commands)
            or any(not _finite(e.payload["params"].get("value"))
                   or e.payload["params"]["value"] < phase["process_reference"]["minimum_z_mm"] for e in z)):
        raise ValueError("Release stroke/withdrawal/off command history changed")
    off = next(e for e in commands if e.id > withdrawals[0].id and e.payload.get("device") == HEATER
               and e.payload.get("command") == "off")
    from qumus_simlab.config import load_yaml
    ta = load_yaml("bench.yaml")["transfer_arm"]
    safe = ta["safe_z"]
    if phase["withdrawal"]["requested_sim_target_mm"] != safe:
        raise ValueError("Release withdrawal differs from the declared safe height")
    if any(e.id > off.id and abs(e.payload["params"]["value"]-safe) >
           phase["release_reference"]["readback_tolerance_mm"] for e in z):
        raise ValueError("Unplanned Z processing occurred after release shutdown")
    transport = phase.get("park_transport", [])
    xy = [e for e in commands if e.id > off.id and e.payload.get("device") == MANIP
          and e.payload.get("command") == "move_axis_to" and e.payload.get("params", {}).get("axis") in {"x", "y"}]
    if len(transport) > 2 or len(xy) != len(transport):
        raise ValueError("Unbound terminal park rail command")
    tol = phase["release_reference"]["readback_tolerance_mm"]
    for index, (item, command) in enumerate(zip(transport, xy, strict=True)):
        event = _historical_event(store, item["command_event"], run_id, "device.command")
        axis = item["axis"]
        params = event.payload["params"]
        frames = _park_command_frames(store, run_id, event)
        encoders = [e for e in frames if "manipulator" in e.payload]
        poses = [{k: e.payload["manipulator"][k] for k in ("x", "y", "z")} for e in encoders]
        start, actual = params["from_pose"], poses[-1]
        bound = params["duration_bound_s"]
        roundoff = (len(frames)+1)*max(math.ulp(event.ts), math.ulp(frames[-1].ts))
        if (event.id != command.id or event.id <= off.id
                or axis != ("x", "y")[index] or params["axis"] != axis
                or params["value"] != item["requested_mm"] or params["value"] != ta["park"][axis]
                or params.get("speed") is not None
                or item["from_pose"] != start or item["actual_pose"] != actual
                or item["duration_bound_s"] != bound or not _finite(bound) or bound < 0
                or not all(map(_finite, (item["duration_bound_s"], item["started_at"], item["ended_at"])))
                or item["started_at"] != event.ts or item["ended_at"] != frames[-1].ts
                or item["encoder_end_event_id"] != frames[-1].id
                or item["encoder_event_count"] != len(frames) or item["clock_roundoff_s"] != roundoff
                or item["encoder_event_sha256"] != _park_frames_digest(frames)
                or not all(_finite(p[k]) for p in [start, *poses] for k in ("x", "y", "z"))
                or any(not _finite(e.ts) or not event.ts <= e.ts <= frames[-1].ts for e in frames)
                or any(a.ts > b.ts for a, b in zip(frames, frames[1:], strict=False))
                or frames[-1].ts-event.ts > bound+roundoff
                or start["z"] < safe-tol or abs(actual[axis]-item["requested_mm"]) > tol
                or any(p["z"] < safe-tol or any(abs(p[k]-start[k]) > tol for k in ("x", "y", "z") if k != axis)
                       or not min(start[axis], actual[axis])-tol <= p[axis] <= max(start[axis], actual[axis])+tol
                       for p in poses)
                or index and any(abs(start[k]-transport[index-1]["actual_pose"][k]) > tol for k in ("x", "y", "z"))):
            raise ValueError("Terminal park timing/readback/command binding changed")
    if phase.get("park") and len(transport) != 2:
        raise ValueError("Completed terminal park lacks both guarded rail records")


def _release_budget_guard(ctx: FlowContext, phase: dict, *, consume: int = 0) -> float:
    c, b = _profile(ctx), phase["release_budget"]
    work = _release_work(ctx, phase)
    time_left = c["max_total_s"]-(ctx.clock.now()-phase["started_at"])-work["seconds"]
    observations_left = c["max_observations"]-phase["observations"]-work["observations"]-consume
    if (time_left < 0 or observations_left < 0
            or phase["travel_mm"]+work["z_travel_mm"] > c["max_travel_mm"]):
        raise NeedsHumanError("Release lacks its mandatory exit/native-endpoint budget")
    allowance = min(time_left, observations_left*(c["max_unobserved_s"]-c["capture_reserve_s"]))
    b.update(reserved_observations=work["observations"], reserved_seconds=work["seconds"],
             reserved_z_travel_mm=work["z_travel_mm"], thermal_allowance_s=allowance,
             checked_at=ctx.clock.now())
    _save_phase(ctx, phase)
    return allowance


def _release_exposure(ctx: FlowContext, phase: dict, frame: Frame, *, endpoint: bool = False) -> None:
    """Account every actual exposure, including a subsequently rejected exposure."""
    c, b = _profile(ctx), phase["release_budget"]
    previous = phase["observed_at"]
    phase["observations"] += 1
    if endpoint:
        b["endpoint_observations"] += 1
    elif b["stage"] == "park":
        b["park_observations"] += 1
    elif b["stage"] in {"entry_heat", "release_heat"}:
        b["thermal_observations"] += 1
    phase.update(observed_at=frame.timestamp, last_frame=_frame_record(frame), last_pose=_pos(ctx))
    _save_phase(ctx, phase)
    if b["stage"] == "park":
        try:
            _release_execution_binding(ctx.store, ctx.run_id, phase)
            if (ctx.gate.heater(HEATER).get_target() is not None
                    or _pos(ctx)["z"] < _ta(ctx)["safe_z"]-ctx.verification["manipulator"]["tolerance_mm"]):
                raise ValueError("Park is not withdrawn and off")
        except (KeyError, TypeError, ValueError) as exc:
            phase["held_reason"] = "release_park_boundary_unconfirmed"
            _save_phase(ctx, phase)
            raise NeedsHumanError("Terminal park camera-gap boundary is unconfirmed") from exc
    if (phase["observations"] > c["max_observations"] or b["endpoint_observations"] > b["endpoint_limit"]
            or ctx.clock.now()-phase["started_at"] > c["max_total_s"]
            or frame.timestamp < previous or ctx.clock.now()-frame.timestamp > c["max_unobserved_s"]
            or (b["stage"] in {"entry_heat", "release_heat", "endpoint"}
                and ctx.clock.now()-previous > c["max_unobserved_s"])):
        phase["held_reason"] = "release_exposure_budget_or_deadline"
        _save_phase(ctx, phase)
        raise NeedsHumanError("Release exposure exhausted its actual finite budget/deadline")


def _target_view(ctx: FlowContext, chip_id: str, target: tuple[float, float],
                 materials: list[str], phase: str, *, parked: bool,
                 focus_seed: dict | None = None, on_capture=None) -> dict:
    from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE

    requested = GoToTarget(chip_id, target)._xy(ctx)
    actual = ctx.gate.stage(STAGE).get_sample_pose()
    tolerance = ctx.verification["refind_flake"]["centre_tolerance_um"]/1000
    if math.hypot(actual.x-requested[0], actual.y-requested[1]) > tolerance:
        raise NeedsHumanError("Target endpoint is outside its requested stage ROI")
    frame, reference = _capture_chip_focus(ctx, phase, parked=parked,
                                         focus_seed=focus_seed, on_capture=on_capture)
    scope = ctx.gate.microscope(MICROSCOPE)
    held = (_held_source(ctx, frame), _pos(ctx))
    frames, counts, observed_outlines = [], dict.fromkeys(materials, 0), []
    direct = ctx.params.get("direct_contact_contract")
    designed = ctx.params.get("device_design")
    if direct or designed:
        from qumus_simlab.core.polygons import convex_polygon, transform
        from qumus_simlab.flows.library.search import _stage_to_local

        chip = require_sample(ctx, chip_id, "chip")
        size = chip["props"]["size_mm"]
        corner, rotation = chip["props"]["corner_stage_xy"], _chip_rotation(ctx, chip_id)
        pose_binding = {"corner_stage_xy": list(corner), "chip_rotation_deg": rotation,
            "size_mm": list(size), "stage_sample_pose": actual.model_dump(mode="json")}
    for index in range(3):
        if index:
            frame = microscope_frame(ctx, phase=phase, on_capture=on_capture)
        _require_source_plane(ctx, frame, reference["meta"])
        if (held != (_held_source(ctx, frame), _pos(ctx))
                or frame.meta.get("focus_z") != reference["meta"]["focus_z"]
                or scope.get_focus() != reference["meta"]["focus_z"]):
            raise NeedsHumanError("Target endpoint held capture context changed")
        frames.append(_frame_record(frame))
        for material in materials:
            seen = analyzer(ctx).find_flakes(frame, material)
            counts[material] = max(counts[material], sum(math.hypot(d.dx_mm, d.dy_mm)
                <= ctx.verification["drop"]["stack_radius_um"]/1000 for d in seen))
            if direct or designed:
                outlines = []
                for d in seen:
                    if (designed and getattr(analyzer(ctx), "recognition_backend", None) == "v1_lab_rgb"
                            and frame.meta.get("calibration", {}).get("material_capture_profile") == V1_MATERIAL_CAPTURE_PROFILE
                            and not frame.meta.get("replayed_from")):
                        from qumus_simlab.core.regions import region_dict, transform_region
                        chain = analyzer(ctx).persist_detection_masks(frame, d)
                        try:
                            observed_region = _original_mask_region(chain)
                        except ValueError:
                            continue
                        # Original mask cells already carry their calibrated
                        # exposure coordinates; do not add the centroid twice.
                        origin = _stage_to_local(tuple(corner), rotation, actual.x, actual.y)
                        region = transform_region(observed_region, rotation_deg=-rotation, xy=origin)
                        outlines.append({"material": material, "region_mm": region_dict(region),
                            "outline_complete": True, "frame": "chip_bottom_left_axes", "units": "mm",
                            "geometry_source": "v1_original_mask", "recognition_chain": chain})
                        continue
                    if (not d.meta.get("outline_complete") or not d.meta.get("outline_mm")
                            or d.meta.get("geometry_source") in (None, "legacy_display") or frame.meta.get("replayed_from")):
                        continue
                    try:
                        lx, ly = _stage_to_local(tuple(corner), rotation, actual.x-d.dx_mm, actual.y-d.dy_mm)
                        outline = transform(convex_polygon(d.meta["outline_mm"]),
                            xy=(lx, ly) if designed else (lx-size[0]/2, ly-size[1]/2), rotation_deg=-rotation)
                    except (ValueError, TypeError):
                        continue
                    outlines.append({"material": material, "outline_mm": [list(p) for p in outline],
                        "outline_complete": True, "frame": "chip_bottom_left_axes" if designed else "chip_center_axes", "units": "mm"})
                observed_outlines.append({"material": material, "outlines": outlines,
                    "observation_binding": {"run_id": ctx.run_id, "frame": frames[-1], "pose": pose_binding}})
    result = {"profile": TRANSFER_PROFILE_ID, "run_id": ctx.run_id, "target_chip_id": chip_id,
        "target_xy_mm": list(target), "requested_stage_xy_mm": list(requested),
        "stage_kinematics": held[0], "focus_reference": reference,
        "frames": frames, "materials_found": counts}
    if direct:
        result.update(observed_outlines=observed_outlines,
            receiving_layers_above_contacts=deepcopy(chip["props"].get("layers_top_first")),
            target_contact=target_contact_snapshot(ctx.store, chip_id, ctx.params["target_contact_region_id"]),
            ledger_event_id=ctx.store.last_event_id())
    if designed:
        result.update(observed_outlines=observed_outlines,
            device_design_digest=designed["canonical_digest"], ledger_event_id=ctx.store.last_event_id())
    return result


def _reference(record: dict[str, Any]) -> Frame:
    if hashlib.sha256(Path(record["image"]).read_bytes()).hexdigest() != record["sha256"]:
        raise NeedsHumanError("The original difference reference changed")
    return Frame(MICROSCOPE, record["captured_at"], path=record["image"], meta=record["meta"])


def _image_calibration(frame: Frame) -> PixelCalibration:
    # Metadata admission precedes opening/decoding any old or current image.
    from qumus_simlab.testbench.render import supported_photometry_profile

    metadata = frame.meta if isinstance(frame.meta, dict) else {}
    c = metadata.get("calibration")
    if not isinstance(c, dict):
        raise NeedsHumanError("Transfer requires the complete supported capture calibration/white-light profile")
    scales, size = c.get("mm_per_pixel"), c.get("image_size_px")
    if (c.get("source") != "synthetic" or c.get("frame") != "microscope_image"
            or c.get("u_axis") != "+world_x" or c.get("v_axis") != "-world_y"
            or not isinstance(scales, list) or len(scales) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in scales)
            or not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
            or not supported_photometry_profile(c.get("photometry_profile"))):
        raise NeedsHumanError("Transfer requires the complete supported capture calibration/white-light profile")
    pixels = load_pixels(frame)
    if [pixels.width, pixels.height] != size:
        raise NeedsHumanError("Actual raster size differs from admitted capture calibration")
    return PixelCalibration(scales[0], -scales[1])


def _phase_capture(ctx: FlowContext, phase: dict[str, Any]) -> Frame:
    ctx.check_cancel()
    c = _phase_budget(ctx, phase)
    now = ctx.clock.now()
    if (phase.get("held_reason") or phase.get("observations", 0) >= c["max_observations"]
            or now-phase["started_at"] > c["max_total_s"]):
        raise NeedsHumanError("Transfer phase is held or exhausted its finite observation/time budget")
    if "release_budget" in phase:
        _release_budget_guard(ctx, phase, consume=1)
    scope, stage = ctx.gate.microscope(MICROSCOPE), ctx.gate.stage(STAGE)
    held = (_pos(ctx), stage.kinematics(), scope.get_focus(), scope.get_objective())
    capture_kwargs = {}
    pending = phase.get("pending_intent") or {}
    if "macro_admission_event_id" in pending:
        saved = _transfer_state(ctx)["phases"][phase["name"]].get("pending_intent")
        if saved != pending:
            raise NeedsHumanError("Tail exposure intent differs from its saved pre-command reservation")
        capture_kwargs["transfer_motion_intent"] = deepcopy(saved)
    previous_observed = phase.get("observed_at")
    frame = microscope_frame(ctx, phase=phase["name"], **capture_kwargs)
    if "release_budget" in phase:
        _release_exposure(ctx, phase, frame)
    if (held != (_pos(ctx), stage.kinematics(), scope.get_focus(), scope.get_objective())
            or frame.meta.get("stage_kinematics") != held[1]
            or frame.meta.get("focus_z") != held[2] or frame.meta.get("objective") != held[3]):
        phase["held_reason"] = "capture_pose_changed"
        _save_phase(ctx, phase)
        raise NeedsHumanError("Capture-time held pose/focus/objective binding changed")
    _image_calibration(frame)
    baseline = phase.get("reference")
    if baseline:
        before, after = baseline["meta"], frame.meta
        keys = ["calibration", "objective", "stage_kinematics"]
        if phase.get("difference_active"):
            keys += ["focus_z"]
        if any(before.get(k) != after.get(k) for k in keys):
            phase["held_reason"] = "capture_binding_changed"
            _save_phase(ctx, phase)
            raise NeedsHumanError("Difference phase capture calibration/profile/pose/focus changed")
    end = ctx.clock.now()
    if (end-frame.timestamp > c["max_unobserved_s"]
            or (phase.get("pending_intent") and
                end-(previous_observed if previous_observed is not None else frame.timestamp) > c["max_unobserved_s"])
            or end-phase["started_at"] > c["max_total_s"]):
        phase["held_reason"] = "observation_deadline"
        _save_phase(ctx, phase)
        raise NeedsHumanError("Transfer exposure missed its observation deadline")
    if "release_budget" not in phase:
        phase["observations"] = phase.get("observations", 0)+1
    if ctx.params.get("stamp_discovery_policy") and phase.get("difference_active"):
        phase["previous_frame"] = phase.get("last_frame")
        phase["previous_pose"] = phase.get("last_pose")
    phase["observed_at"] = frame.timestamp
    phase["last_frame"] = _frame_record(frame)
    phase["last_pose"] = _pos(ctx)
    _save_phase(ctx, phase)
    return frame


def _begin_phase(ctx: FlowContext, name: str, *, difference: bool = True) -> dict[str, Any]:
    c = _profile(ctx)
    phases = _transfer_state(ctx).get("phases", {})
    if any(item.get("pending_intent") or "park_pending_intent" in item for item in phases.values()):
        raise NeedsHumanError("Pending transfer intent cannot be bypassed by another phase")
    old = phases.get(name)
    if old:
        _phase_budget(ctx, old)
        if old.get("pending_intent") or old.get("held_reason"):
            raise NeedsHumanError("Interrupted transfer motion/evidence requires review")
        if old.get("profile") != c["id"]:
            raise NeedsHumanError("Interrupted transfer profile changed")
        if not old.get("reference") or not old.get("holder_xy"):
            raise NeedsHumanError("Interrupted transfer lacks its original phase reference")
        _reference(old["reference"])
        return old
    phase = {"name": name, "profile": c["id"], "started_at": ctx.clock.now(),
             "observations": 0, "travel_mm": 0., "difference_active": difference}
    if ctx.params.get("device_design") and name.startswith("v1_first_contact_"):
        phase["sampling_policy"] = _first_contact_sampling(ctx.verification)
        if ctx.params.get("stamp_discovery_policy"):
            discovery = _transfer_state(ctx)["discovery"]
            if discovery.get("state") != "prepared":
                raise NeedsHumanError("First-contact phase has no original prepared discovery domain")
            phase["started_at"] = discovery["started_at"]
            phase["observations"] = sum(1 for event in _transfer_events(ctx.store, ctx.run_id, "device.command", discovery["start_event_id"])
                if event.payload.get("device") == MICROSCOPE and event.payload.get("command") == "capture")
    cursor = ctx.store.last_event_id()
    frame = _phase_capture(ctx, phase)
    phase["reference"] = _frame_record(frame)
    captures = [e for e in ctx.store.events_after(cursor, limit=32, run_id=ctx.run_id)
        if e.kind == "lab.frame" and e.payload.get("microscope_preview", {}).get("phase") == name
        and e.payload["microscope_preview"].get("image") == phase["reference"]["image"]
        and e.payload["microscope_preview"].get("timestamp") == frame.timestamp]
    if len(captures) != 1:
        raise NeedsHumanError("Phase reference lacks its actual capture event")
    phase["reference_event"] = {"event_id": captures[0].id, "event_sha256": _digest(captures[0].payload)}
    if phase.get("sampling_policy"):
        payload = {"phase": name, "policy": phase["sampling_policy"], "reference_event": phase["reference_event"]}
        event_id = ctx.store.emit("stack.first_contact_sampling", payload, run_id=ctx.run_id)
        phase["sampling_policy_event"] = {"event_id": event_id, "event_sha256": _digest(payload)}
    phase["holder_xy"] = [_pos(ctx)["x"], _pos(ctx)["y"]]
    if ctx.params.get("stamp_discovery_policy") and name.startswith("v1_first_contact_"):
        phase["initial_observations"] = phase["observations"]
    _save_phase(ctx, phase)
    return phase


def _guarded_z(ctx: FlowContext, phase: dict[str, Any], target: float, *,
               speed: float = .1, observe=None, macro_admission: dict | None = None,
               max_step_mm: float | None = None) -> bool:
    """Segment one intent with actual readbacks, duration bounds and Clock.

    ``max_step_mm`` caps each observed command further, so a slow phase is
    seen in more frames.
    """
    c = _phase_budget(ctx, phase)
    observation_window = c["max_unobserved_s"]
    floor = ctx.gate.manipulator_z_floor(MANIP)
    lim = ctx.gate.limits.manipulators[MANIP]
    if not math.isfinite(target) or target < max(lim.z[0], floor or lim.z[0]) or target > lim.z[1]:
        raise NeedsHumanError("Transfer Z intent is outside its current floor/envelope; no clamp applied")
    if "release_budget" in phase and phase.get("outputs_off") and abs(target-_ta(ctx)["safe_z"]) > 1e-8:
        raise NeedsHumanError("No further release press is permitted after clear withdrawal/off")
    if macro_admission is not None:
        _pickup_partial_receipt(ctx.store, ctx.run_id, phase,
            list(_transfer_events(ctx.store, ctx.run_id, "stack.pickup_macro_admission")))
        if macro_admission != phase["final_partial_macro"] or target != macro_admission["admission"]["admitted_target_mm"]:
            raise NeedsHumanError("Final pickup prefix differs from its reserved target")
        if macro_admission["status"] != "consumed":
            raise NeedsHumanError("Final pickup prefix cannot execute after its outcome")
        requested = macro_admission["admission"]["requested_target_mm"]
        if not max(lim.z[0], floor or lim.z[0]) <= requested <= lim.z[1]:
            raise NeedsHumanError("Original pickup intent is outside its current floor/envelope; no clamp applied")
    m = ctx.gate.manipulator(MANIP)
    tol = ctx.verification["manipulator"]["tolerance_mm"]
    # An observed descent never commands upward: when actuator noise lands
    # the readback at or below its intent, the intent is reached and the
    # caller chooses the next one from the actual pose (a518 seed 2).
    observed_descent = observe is not None and target < _pos(ctx)["z"]
    while abs(_pos(ctx)["z"]-target) > 1e-8:
        if observed_descent and _pos(ctx)["z"] < target:
            return False
        ctx.check_cancel()
        ctx.gate.watchdog_tick()
        if "release_budget" in phase:
            _release_budget_guard(ctx, phase)
        if any(item.get("pending_intent") or "park_pending_intent" in item
               for item in _transfer_state(ctx).get("phases", {}).values()):
            raise NeedsHumanError("An unresolved transfer intent prevents further Z execution")
        _reference(phase["reference"])
        if (phase.get("held_reason") or phase["observations"] >= c["max_observations"]
                or ctx.clock.now()-phase["started_at"] > c["max_total_s"]):
            raise NeedsHumanError("No finite phase budget remains for another observed command")
        last = phase["last_frame"]["meta"]
        scope, stage = ctx.gate.microscope(MICROSCOPE), ctx.gate.stage(STAGE)
        if (scope.get_focus() != last["focus_z"] or scope.get_objective() != last["objective"]
                or stage.kinematics() != last["stage_kinematics"]):
            raise NeedsHumanError("Observed focus/objective/source pose changed before Z motion")
        if phase.get("pending_intent"):
            raise NeedsHumanError("Pending transfer intent cannot be replayed")
        if ctx.clock.now()-phase["observed_at"] >= observation_window-c["capture_reserve_s"]:
            _phase_capture(ctx, phase)
        start = _pos(ctx)["z"]
        remaining_s = observation_window-(ctx.clock.now()-phase["observed_at"])-c["capture_reserve_s"]
        distance = min(abs(target-start), c["max_unobserved_z_mm"]-tol, speed*remaining_s)
        if "max_command_z_mm" in c:
            distance = min(distance, c["max_command_z_mm"])
        if max_step_mm is not None:
            distance = min(distance, max_step_mm)
        if distance <= 1e-8:
            raise NeedsHumanError("No observed Z execution budget remains")
        end = start+math.copysign(distance, target-start)
        if end < phase.get("process_reference", {}).get("minimum_z_mm", -math.inf):
            raise NeedsHumanError("Coverage unavailable within the declared candidate-relative press budget")
        duration = m.axis_motion_duration_bound("z", start, end, speed)
        if duration is None or not math.isfinite(duration) or duration > remaining_s+1e-9:
            raise NeedsHumanError("Z adapter lacks an admitted finite duration bound")
        if phase["travel_mm"]+distance > c["max_travel_mm"]:
            raise NeedsHumanError("Transfer phase exhausted its cumulative Z travel budget")
        phase["pending_intent"] = {"axis": "z", "from_mm": start, "to_mm": end,
                                   "speed_mm_s": speed, "duration_bound_s": duration,
                                   "macro_target_mm": target, "at": ctx.clock.now()}
        if macro_admission is not None:
            phase["pending_intent"].update(
                requested_macro_target_mm=macro_admission["admission"]["requested_target_mm"],
                macro_admission_event_id=macro_admission["event_id"])
        _save_phase(ctx, phase)
        m.move_axis_to("z", end, speed=speed)
        actual = _pos(ctx)
        if actual["z"] < phase.get("process_reference", {}).get("minimum_z_mm", -math.inf):
            raise NeedsHumanError("Actual Z crossed the current-layer process budget; review before further descent")
        if (abs(actual["z"]-end) > tol or abs(actual["z"]-start) > c["max_unobserved_z_mm"]
                or [actual["x"], actual["y"]] != phase["holder_xy"]):
            raise NeedsHumanError("Transfer segment actual readback or held XY failed")
        frame = _phase_capture(ctx, phase)
        phase["travel_mm"] += abs(actual["z"]-start)
        phase["last_intent"] = phase.pop("pending_intent")
        _save_phase(ctx, phase)
        if observe is not None and observe(frame):
            return True
        if abs(end-target) <= 1e-8 and abs(actual["z"]-target) <= tol:
            return False
    return False


def _clear_lift(ctx: FlowContext, name: str) -> None:
    tol = ctx.verification["manipulator"]["tolerance_mm"]
    if _pos(ctx)["z"] < _ta(ctx)["safe_z"]-tol:
        old = _transfer_state(ctx).get("phases", {}).get("clear_lift_"+name)
        if old and old.get("complete"):
            raise NeedsHumanError("Completed clear-lift checkpoint no longer matches the actual holder; review")
        phase = _begin_phase(ctx, "clear_lift_"+name, difference=False)
        _guarded_z(ctx, phase, _ta(ctx)["safe_z"])
        phase["complete"] = True
        _save_phase(ctx, phase)


def _restore_source_plane(ctx: FlowContext, metadata: dict[str, Any]) -> None:
    scope = ctx.gate.microscope(MICROSCOPE)
    scope.set_objective(metadata["objective"])
    scope.set_focus(metadata["focus_z"])


def _held_source(ctx: FlowContext, source: Frame | None = None) -> dict[str, Any]:
    stage, holder = ctx.gate.stage(STAGE), ctx.gate.manipulator(MANIP)
    if stage.motion_active is not False or holder.motion_active is not False:
        raise NeedsHumanError("Endpoint requires a stationary source and holder")
    state = stage.kinematics()
    if source is not None and source.meta.get("stage_kinematics") != state:
        raise NeedsHumanError("Source pose changed after endpoint exposure")
    return state


def fresh_stamps(store: Any, bench: dict[str, Any], exclude_run: str | None = None
                 ) -> list[dict[str, Any]]:
    """Stamps a stack may take, first choice first: status `fresh` (a new PC
    film, reported by a person once a stamp was used or torn), lying in a stamp
    slot of the right storage board, not booked by another job."""
    booked = store.reserved_samples(exclude_run=exclude_run)
    return sorted((s for s in store.list_samples(kind="stamp")
                   if s["status"] == "fresh" and s["location"] in stamp_slots(bench)
                   and s["id"] not in booked), key=lambda s: s["id"])



# ----- temperature -------------------------------------------------------------


def _centred_flakes(ctx: FlowContext, frame: Any, material: str) -> list[Any]:
    """Detections in a 20X view centred on a chosen source flake.

    v1 LiveSegment lets a capture centred on a chosen flake hold it at up to
    10% of the frame (below 50X). The scan ceiling (100000 px, ~3300 um2 at
    20X) dropped large flakes: a refind then read "not found" and a source
    absence check could pass with the flake still there. Every component is
    still listed, so neighbours stay visible (same rule as pickup_evidence).
    """
    an = analyzer(ctx)
    if getattr(an, "recognition_backend", None) == "v1_lab_rgb":
        return an.find_flakes(frame, material, centered=True)
    return an.find_flakes(frame, material)

class HeatTransferStage(Step):
    """Bring the transfer stage to a temperature (v1 MoleculeSetInitialTemp)."""

    devices = (HEATER,)

    def __init__(self, target_c: float, name: str = "heat_transfer_stage") -> None:
        self.target_c, self.name = target_c, name

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        return {"celsius": self.target_c, "band_c": ctx.verification["heating"]["band_c"]}

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        h = ctx.verification["heating"]
        ramp_heater(ctx, HEATER, self.target_c, h["poll_s"])
        wait_temperature(ctx, HEATER, lambda t: abs(t - self.target_c) <= h["band_c"],
                         what=f"{self.target_c:.0f} C", timeout_s=h["timeout_s"],
                         stuck_window_s=h["stuck_window_s"], stuck_delta_c=h["stuck_delta_c"],
                         poll_s=h["poll_s"])

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        t = ctx.gate.heater(HEATER).read_temperature()
        m = {"celsius": round(t, 1)}
        if abs(t - self.target_c) <= exp["band_c"] * 1.5:
            return Observation(Verdict.OK, m, exp)
        return Observation(Verdict.CORRECTABLE, m, exp, message=f"stage at {t:.1f} C")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return Correction("wait for the temperature again")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE  # heating to a setpoint again is harmless


class CoolTransferStage(Step):
    """Heater off and wait until the arm may handle the chip."""

    name = "cool_transfer_stage"
    devices = (HEATER, FAN)

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        return {"below_c": ctx.verification["drop"]["safe_handling_c"]}

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        h, below = ctx.verification["heating"], self.expected(ctx)["below_c"]
        ctx.gate.heater(HEATER).off()
        ctx.gate.switch(FAN).set(True)
        wait_temperature(ctx, HEATER, lambda t: t <= below, what=f"below {below:.0f} C",
                         timeout_s=h["cool_timeout_s"], stuck_window_s=h["stuck_window_s"],
                         stuck_delta_c=h["stuck_delta_c"], poll_s=h["poll_s"])
        ctx.gate.switch(FAN).set(False)

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        t = ctx.gate.heater(HEATER).read_temperature()
        ok = (t <= exp["below_c"] and ctx.gate.heater(HEATER).get_target() is None
              and ctx.gate.switch(FAN).get() is False)
        return Observation(Verdict.OK if ok else Verdict.CORRECTABLE, {"celsius": round(t, 1)},
                           exp, message="" if ok else f"stage still at {t:.1f} C")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return Correction("wait longer")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


# ----- manipulator and stamp handling ----------------------------------------

class ManipulatorTo(Step):
    """Move the stamp holder between the load pose, over the stage (at its
    approach height, just above contact) and parked (v1 load / initialize /
    remove stamp axis moves). Always lifts before travelling in x/y, and travels
    along the holder's rails: up then left into the microscope, right then down
    out of it.

    The holder waits at the rail corner (bench.yaml park, #181), and a trip over
    the stage is the first moment the camera can see whether the holder's
    vacuum holds the slide: a stamp left lying at the hand-off did not come
    along, and the holder goes back to grip it again."""

    devices = (MANIP,)

    def __init__(self, to: str, name: str | None = None, *, stamp_id: str | None = None) -> None:
        assert to in ("load", "over_stage", "park")
        self.to, self.stamp_id = to, stamp_id
        self.name = name or f"manipulator_to_{to}"

    def _target(self, ctx: FlowContext) -> dict[str, float]:
        ta = _ta(ctx)
        if self.to == "load":
            return dict(ta["load"])
        if self.to == "park":
            return {"x": ta["park"]["x"], "y": ta["park"]["y"], "z": ta["safe_z"]}
        # The independent focus motor no longer lowers a thick/high chip. Enter
        # at the configured clear height; only the observed approach may descend.
        placement = _transfer_state(ctx).get("placements", {}).get(self.stamp_id)
        if placement:
            if (placement["stamp_id"] != self.stamp_id or placement["run_id"] != ctx.run_id
                    or placement["profile"] != TRANSFER_PROFILE_ID):
                raise NeedsHumanError("Working XY binding changed")
            x, y = placement["working_xy_mm"]
        else:
            x, y = ta["over_stage"]["x"], ta["over_stage"]["y"]
        return {"x": x, "y": y, "z": ta["safe_z"]}

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        return {**self._target(ctx), "tolerance_mm": ctx.verification["manipulator"]["tolerance_mm"]}

    def act(self, ctx: FlowContext, correction: Correction | None, *, phase: dict | None = None) -> None:
        if phase is not None and self.to != "park":
            raise NeedsHumanError("Shared release transport is restricted to terminal park")
        tgt = self._target(ctx)
        if phase is None:
            _clear_lift(ctx, self.name)
        elif _pos(ctx)["z"] < _ta(ctx)["safe_z"]-ctx.verification["manipulator"]["tolerance_mm"]:
            raise NeedsHumanError("Release park needs its qualified clear withdrawal first")
        if correction is not None and correction.action == REGRIP:
            ta = _ta(ctx)
            _travel(ctx, ta["load"]["x"], ta["load"]["y"])
            ctx.gate.switch(MANIP_VACUUM).set(True)
        _travel(ctx, tgt["x"], tgt["y"], phase=phase)
        if phase is not None:
            phase["holder_xy"] = [_pos(ctx)["x"], _pos(ctx)["y"]]
            _phase_capture(ctx, phase)
        if "z" in tgt and abs(_pos(ctx)["z"]-tgt["z"]) > 1e-8:
            motion_phase = phase if phase is not None else _begin_phase(ctx, self.name+"_z", difference=False)
            _guarded_z(ctx, motion_phase, tgt["z"])
        if self.to in ("load", "park"):
            state = _transfer_state(ctx)
            init = state.get("initializations", {}).get(state.get("active_initialization"))
            if init:
                scope = ctx.gate.microscope(MICROSCOPE)
                scope.set_focus(init["substrate_focus_mm"])
                scope.set_objective(init["objective"])
                state.pop("active_initialization", None)
                _save_transfer(ctx, state)

    def verify(self, ctx: FlowContext) -> Observation:
        exp, p = self.expected(ctx), _pos(ctx)
        err = max(abs(p[a] - v) for a, v in self._target(ctx).items())
        m = {**{k: round(v, 4) for k, v in p.items()}, "error_mm": round(err, 4)}
        tol = exp["tolerance_mm"] if self.to != "over_stage" else max(exp["tolerance_mm"], 0.01)
        if err > tol:
            return Observation(Verdict.CORRECTABLE, m, exp, message=f"axis off by {err:.4f} mm")
        if self.to == "over_stage":
            imgs: list[str] = []
            captures: list[dict] = []
            left = look(ctx, CAMERA, HANDOFF, ARM, label="stamp", images=imgs, captures=captures)
            handoff = {"stamp_id": self.stamp_id, "run_id": ctx.run_id,
                       "observed_at": ctx.clock.now(), "captures": captures}
            if ctx.params.get("stamp_discovery_policy") and self.name == "initialize_restored_discovery_source_engage":
                self.handoff_observation = deepcopy(handoff)
            if left:
                return Observation(Verdict.CORRECTABLE, {**m, "stamp_at_handoff": len(left),
                    "handoff_observation": handoff},
                                   {**exp, "stamp_at_handoff": 0}, images=imgs,
                                   message="the stamp stayed at the hand-off (the holder's "
                                   "vacuum did not hold)")
            return Observation(Verdict.OK, m, exp, images=imgs)
        return Observation(Verdict.OK, m, exp)

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        if obs.measurements.get("stamp_at_handoff"):
            return Correction(REGRIP)
        return Correction("move again")

    def on_give_up(self, ctx: FlowContext, obs: Observation) -> None:
        selected = ctx.store.get_sample(self.stamp_id) if self.stamp_id else None
        if selected is None or selected["kind"] != "stamp":
            return
        try:
            pose, basis = self._handoff_identity(ctx, obs)
        except (KeyError, TypeError, ValueError, OSError) as error:
            ctx.store.upsert_sample(self.stamp_id, "stamp", props={"location_uncertain": True,
                "last_seen": {"at": None, "candidate_at": HANDOFF if obs.measurements.get("stamp_at_handoff") else None,
                    "reason": str(error), "observation": obs.to_dict()}})
            return
        ctx.store.upsert_sample(self.stamp_id, "stamp", location=HANDOFF, props={"pose": pose,
            "location_uncertain": False, "last_seen": {"at": HANDOFF, "pose": pose,
                "identity_basis": basis, "observation": obs.to_dict()}})

    def _handoff_identity(self, ctx: FlowContext, obs: Observation) -> tuple[dict, str]:
        """Qualify the already taken public look; absence never names a stamp."""
        handoff = obs.measurements["handoff_observation"]
        if (self.to != "over_stage" or obs.measurements.get("stamp_at_handoff") != 1
                or handoff["stamp_id"] != self.stamp_id or handoff["run_id"] != ctx.run_id
                or ctx.params.get("stamp_id") != self.stamp_id
                or ctx.store.get_run(ctx.run_id)["params"].get("stamp_id") != self.stamp_id
                or len(handoff["captures"]) != 1):
            raise ValueError("Selected handoff context is missing or ambiguous")
        capture = handoff["captures"][0]
        times = [capture["capture_started_at"], capture["frame_timestamp"], capture["capture_returned_at"],
                 handoff["observed_at"], ctx.clock.now()]
        if (not all(_finite(value) for value in times) or times != sorted(times)
                or capture["run_id"] != ctx.run_id or capture["camera"] != CAMERA
                or capture["frame_device_id"] != CAMERA or capture["region"] != HANDOFF or capture["frame_of"] != ARM):
            raise ValueError("Handoff exposure/capture context is unknown or stale")
        mark = capture["capture_event_id"]
        if not isinstance(mark, int) or isinstance(mark, bool):
            raise ValueError("Handoff lacks its actual capture command")
        command = ctx.store.events_after(mark-1, limit=1, run_id=ctx.run_id)
        starts = list(_transfer_events(ctx.store, ctx.run_id, "step.started"))
        if (len(command) != 1 or command[0].id != mark or command[0].kind != "device.command"
                or command[0].payload.get("device") != CAMERA or command[0].payload.get("command") != "capture"
                or not times[0] <= command[0].ts <= times[2] or not starts or starts[-1].id >= mark
                or starts[-1].payload["name"] not in {self.name, self.name.removesuffix("_engage")}):
            raise ValueError("Handoff capture is not owned by the current failed engagement")
        if any(event.payload.get("id") == self.stamp_id or event.payload.get("location") in {HANDOFF, MANIP}
               for event in _transfer_events(ctx.store, None, "sample.updated", mark)):
            raise ValueError("Public stamp/occupancy state changed after the handoff look")
        path = capture["image"]
        if (not path or path not in obs.images or not Path(path).is_file()
                or hashlib.sha256(Path(path).read_bytes()).hexdigest() != capture["image_sha256"]):
            raise ValueError("Handoff image is missing or changed")
        detections = capture["detections"]
        if len(detections) != 1 or detections[0]["label"] != "stamp":
            raise ValueError("Handoff occupancy is ambiguous")
        detection = detections[0]
        if (not _finite(detection["confidence"]) or not 0 <= detection["confidence"] <= 1
                or not all(_finite(detection["pose"][axis]) for axis in ("x", "y", "z", "rotation"))
                or any(sample["id"] != self.stamp_id and sample["location"] == HANDOFF
                       for sample in ctx.store.list_samples())):
            raise ValueError("Handoff conflicts with public pose/occupancy evidence")
        identity = detection.get("object_id")
        if identity is not None:
            if identity != self.stamp_id:
                raise ValueError("Readable handoff identity does not match the selected stamp")
            basis = "readable_object_id"
        else:
            self._handoff_load_identity(ctx, mark)
            basis = "same_run_exclusive_load"
        return to_world(ctx.bench, ARM, Pose(**detection["pose"])).model_dump(), basis

    def _handoff_load_identity(self, ctx: FlowContext, capture_id: int) -> None:
        if any(sample["id"] != self.stamp_id and sample["location"] in {HANDOFF, MANIP}
               for sample in ctx.store.list_samples(kind="stamp")):
            raise ValueError("Another stamp conflicts with the selected load/holder history")
        events = sorted([event for kind in ("step.started", "step.finished", "step.verified", "device.command")
            for event in _transfer_events(ctx.store, ctx.run_id, kind) if event.id < capture_id], key=lambda event: event.id)
        start, finish, row = _witness_checkpoint(ctx.store, ctx.run_id, events, "load_stamp")
        grip_start, grip_finish, _ = _witness_checkpoint(ctx.store, ctx.run_id, events, "grip_stamp")
        seen = [event for event in events if start.id < event.id < finish.id and event.kind == "step.verified"
                and event.payload.get("name") == "load_stamp"]
        if (len(seen) != 1 or seen[0].payload.get("verdict") != "ok"
                or seen[0].payload["expected"].get("sample") != self.stamp_id
                or seen[0].payload["expected"].get("at") != HANDOFF
                or seen[0].payload["measurements"].get("found_at") != HANDOFF
                or row["measurements"] != seen[0].payload["measurements"]
                or not seen[0].payload["images"] or not all(Path(path).is_file() for path in seen[0].payload["images"])
                or not finish.id < grip_start.id < grip_finish.id < capture_id):
            raise ValueError("No same-run verified selected handoff placement")
        vacuum = [event for event in events if event.id > finish.id and event.kind == "device.command"
            and event.payload.get("device") == MANIP_VACUUM and event.payload.get("command") == "set"]
        grip = [event for event in events if grip_start.id < event.id < grip_finish.id and event.kind == "step.verified"]
        if (len(vacuum) != 1 or not grip_start.id < vacuum[0].id < grip_finish.id
                or vacuum[0].payload["params"].get("on") is not True
                or len(grip) != 1 or grip[0].payload.get("verdict") != "not_verifiable"):
            raise ValueError("Selected handoff/grip continuity is unverified")
        updates = [event for event in _transfer_events(ctx.store, None, "sample.updated", start.id)
                   if event.id < capture_id]
        handed = [event for event in updates if seen[0].id < event.id < finish.id
                  and event.payload.get("id") == self.stamp_id and event.payload.get("location") == HANDOFF]
        loaded = [event for event in updates if vacuum[0].id < event.id < grip_finish.id
                  and event.payload.get("id") == self.stamp_id and event.payload.get("location") == MANIP]
        if (len(handed) != 1 or len(loaded) != 1
                or any(event.payload.get("id") != self.stamp_id and
                    (event.payload.get("location") == HANDOFF or event.payload.get("kind") == "stamp"
                     and event.payload.get("location") == MANIP) for event in updates)
                or any(event.id > loaded[0].id and event.payload.get("id") == self.stamp_id and
                    (event.payload.get("location") != MANIP or event.payload.get("props", {}).get("location_uncertain"))
                    for event in updates)):
            raise ValueError("Selected handoff identity/occupancy continuity is conflicting")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE if _pos(ctx)["z"] >= _ta(ctx)["safe_z"] else ProbeResult.UNKNOWN

    def safe_state(self, ctx: FlowContext) -> None:
        _clear_lift(ctx, self.name+"_cleanup")


class GripStamp(Step):
    """Manipulator vacuum on at the hand-off (v1 AtomTransferArmVacuumOn,
    unchecked), then up the rail to the corner standby (bench.yaml park, owner
    2026-11-04, #181): no stamp swap follows a grip, and the parked source
    focus needs the holder at park. Whether the vacuum holds shows only at the
    first trip over the stage: ManipulatorTo over_stage looks for a stamp left
    behind at the hand-off and grips again."""

    name = "grip_stamp"
    devices = (MANIP, MANIP_VACUUM)

    def __init__(self, stamp_id: str, name: str | None = None) -> None:
        self.stamp_id = stamp_id
        if name:
            self.name = name

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        ctx.gate.switch(MANIP_VACUUM).set(True)
        ManipulatorTo("park", name=self.name+"_park").act(ctx, None)

    def verify(self, ctx: FlowContext) -> Observation | NotVerifiable:
        return NOT_VERIFIABLE("no sensor reads the holder's vacuum; the overview camera "
                              "sees the stamp ride along only when the holder leaves the "
                              "hand-off (checked by the next stamp_over step)")

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        ctx.store.upsert_sample(self.stamp_id, "stamp", location=MANIP, status="in_use")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        # switching the vacuum on again at the hand-off is harmless
        tol = ctx.verification["manipulator"]["position_tolerance_mm"]
        return ProbeResult.NOT_DONE if _near(_pos(ctx), _ta(ctx)["load"], tol) \
            else ProbeResult.UNKNOWN


def _discovery_rows(store, discovery, *, last=None):
    last = store.last_event_id() if last is None else last
    rows, after = [], discovery["start_event_id"]-1
    while after < last:
        batch = store.events_after(after, limit=1000)
        if not batch:
            break
        rows.extend(event.model_dump() for event in batch if event.id <= last)
        after = batch[-1].id
    return rows


def _discovery_budget(ctx, discovery, *, reserve_observations=0, reserve_seconds=0., reserve_travel=0.):
    from qumus_simlab.records.discovery import original_work_usage

    work = original_work_usage(_discovery_rows(ctx.store, discovery), run_id=ctx.run_id)
    c = _profile(ctx)
    # The legacy discovery budget (first-contact sampling v3, 4096 frames);
    # discovery is opt-in since #204 and the v4 touch search has no such cap.
    if (work["capture_count"]+reserve_observations > 4096
            or ctx.clock.now()-discovery["started_at"]+reserve_seconds > c["max_total_s"]
            or work["z_variation_mm"]+reserve_travel > c["max_travel_mm"]):
        raise NeedsHumanError("Discovery/restore lacks its one original observation/time/travel budget")
    return work


def _discovery_exit_reserve(ctx, phase, discovery):
    """Use existing adapter duration contracts for the mandatory physical exit."""
    arm, scope = ctx.gate.manipulator(MANIP), ctx.gate.microscope(MICROSCOPE)
    tol = ctx.verification["manipulator"]["tolerance_mm"]
    c = _phase_budget(ctx, phase)
    lift_target = max(_pos(ctx)["z"]+.2, _ta(ctx)["safe_z"])
    lift = lift_target-_pos(ctx)["z"]
    max_command = c["max_unobserved_z_mm"]-tol
    segments = math.ceil(lift/(max_command-tol))
    one = arm.axis_motion_duration_bound("z", _pos(ctx)["z"], _pos(ctx)["z"]+max_command, .1)
    if not _finite(one) or not 0 <= one <= .2+1e-9:
        raise NeedsHumanError("Discovery exit lacks its original first clear-progress duration bound")
    rail_seconds = 0.
    current = _pos(ctx)
    for target in (_ta(ctx)["park"], _ta(ctx)["over_stage"]):
        for axis in ("x", "y"):
            bound = arm.axis_motion_duration_bound(axis, current[axis], target[axis])
            if not _finite(bound) or bound < 0:
                raise NeedsHumanError("Discovery parked revisit lacks a finite rail duration")
            rail_seconds += bound
        current = {**current, **{axis: target[axis] for axis in ("x", "y")}}
    original = discovery["original_stage_pose"]
    stage = ctx.gate.stage(STAGE).get_sample_pose()
    # This is an immutable configured-generator declaration, validated before
    # construction, not an inferred speed or measured apparatus calibration.
    stage_seconds = max(abs(stage.x-original["x"]), abs(stage.y-original["y"]))/10.
    focus_a, focus_b = discovery["focus_reference"]["meta"]["focus_z"], discovery["original_reference"]["meta"]["focus_z"]
    focus_bound = scope.focus_transition_duration_bound(min(focus_a,focus_b)-.08, max(focus_a,focus_b)+.38)
    if not _finite(focus_bound) or focus_bound < 0:
        raise NeedsHumanError("Discovery revisit lacks its declared finite focus transition")
    observations = 21+6+segments+23+2
    seconds = 1.538889+5.30+segments*(one+c["capture_reserve_s"])+rail_seconds+stage_seconds+4*focus_bound+1.538889
    travel = lift+2*tol
    _discovery_budget(ctx, discovery, reserve_observations=observations,
                      reserve_seconds=seconds, reserve_travel=travel)
    return {"observations": observations, "seconds": seconds, "withdrawal_mm": lift,
            "reserved_z_travel_mm": travel, "rail_seconds": rail_seconds,
            "stage_seconds": stage_seconds, "focus_transition_bound_s": focus_bound,
            "first_progress_duration_bound_s": one+c["capture_reserve_s"]}


def _validate_discovery_ready(store, run_id):
    """Shared original-journal authority; incomplete staged joins fail closed."""
    from qumus_simlab.records.discovery import validate_ready_receipt

    run = store.get_run(run_id)
    state = run["meta"].get("normal_transfer", {})
    discovery = state.get("discovery")
    if not isinstance(discovery, dict) or not discovery.get("ready_evidence"):
        raise NeedsHumanError("Pickup requires its original complete discovery/source-restore receipt")
    submissions = list(_transfer_events(store, run_id, "job.submitted"))
    if len(submissions) != 1:
        raise NeedsHumanError("Discovery requires its unique immutable job submission")
    rows, after = [], submissions[0].id-1
    last = discovery["ready_evidence"]["event_id"]
    while after < last:
        batch = store.events_after(after, limit=1000)
        if not batch:
            break
        rows.extend(event.model_dump() for event in batch if event.id <= last)
        after = batch[-1].id
    records = [discovery["original_reference"], discovery["focus_reference"], discovery["restored_reference"],
               discovery["first_candidate"]["reference"], discovery["first_candidate"]["current"],
               discovery["candidate_previous"], *[item["frame"] for item in discovery["maps"]]]
    for record in records:
        _reference(record)
        if record["meta"]["calibration"].get("stamp_discovery") != run["params"].get("stamp_discovery_policy"):
            raise NeedsHumanError("Discovery receipt mixed its original physics/observation profile")
    for key in ("original_reference", "focus_reference", "restored_reference"):
        _validate_focus_reference(discovery[key], run_id, parked=True, store=store)
    return validate_ready_receipt(rows, run_id=run_id, params=run["params"], discovery=discovery,
        placement=state["placements"][run["params"]["stamp_id"]], digest=_digest)


class PrepareStampDiscovery(Step):
    """Four fixed unobstructed observations; no adaptive expansion or source shift."""
    name = "prepare_stamp_discovery"
    devices = (STAGE, MICROSCOPE, MANIP)
    max_corrections = 0
    safe_to_resume = False

    def act(self, ctx, correction):
        from qumus_simlab.core.polygons import rectangle, transform
        from qumus_simlab.flows.library.search import _stage_to_local
        from qumus_simlab.records.discovery import choose_clear_domain, map_centre, validate_policy
        from qumus_simlab.records.lab_geometry import LabEvidence, Source, record_evidence

        policy = validate_policy(ctx.params.get("stamp_discovery_policy"))
        from qumus_simlab.records.discovery import validate_verification

        validate_verification(ctx.verification)
        if not callable(getattr(analyzer(ctx), "discovery_obstacles", None)):
            raise NeedsHumanError("The current analyzer has no supported all-material discovery observation")
        if ctx.bench.get("stamp_discovery") != policy or not ctx.params.get("device_design"):
            raise NeedsHumanError("Discovery requires the fresh linked design/profile binding")
        state = _transfer_state(ctx)
        if state.get("discovery"):
            raise NeedsHumanError("Interrupted discovery cannot repeat or select new map exposures")
        source_id = ctx.params["flake_ids"][0]
        original = deepcopy(state["source_references"][source_id])
        _validate_focus_reference(original, ctx.run_id, parked=True, store=ctx.store)
        stage, scope = ctx.gate.stage(STAGE), ctx.gate.microscope(MICROSCOPE)
        original_pose = stage.get_sample_pose()
        if stage.kinematics() != original["meta"]["stage_kinematics"]:
            raise NeedsHumanError("Original selected source moved before discovery preparation")
        chip_id = ctx.params["flake_chips"][source_id]
        chip = require_sample(ctx, chip_id, "chip")
        corner, angle, size = chip["props"]["corner_stage_xy"], _chip_rotation(ctx, chip_id), chip["props"]["size_mm"]
        local = _stage_to_local(tuple(corner), angle, original_pose.x, original_pose.y)
        nominal = map_centre(local, size, angle)
        a = math.radians(angle)
        camera_local = (local[0]*math.cos(a)-local[1]*math.sin(a), local[0]*math.sin(a)+local[1]*math.cos(a))
        chip_polygon = transform(rectangle(*size, (size[0]/2, size[1]/2)), rotation_deg=angle,
                                 xy=(-camera_local[0], -camera_local[1]))
        starts = list(_transfer_events(ctx.store, ctx.run_id, "step.started"))
        if not starts or starts[-1].payload.get("name") != self.name:
            raise NeedsHumanError("Discovery lacks its actual original checkpoint boundary")
        discovery = {"profile": policy["id"], "policy": policy, "state": "mapping",
            "source_id": source_id, "chip_id": chip_id, "original_reference": original,
            "original_stage_pose": original_pose.model_dump(mode="json"), "start_event_id": starts[-1].id,
            "started_at": starts[-1].ts, "nominal_center_mm": list(nominal),
            "chip_polygon_mm": [list(p) for p in chip_polygon], "maps": []}
        state["discovery"] = discovery
        _save_transfer(ctx, state)
        scope.set_objective(5)
        for index, offset in enumerate(policy["map_offsets_mm"]):
            _discovery_budget(ctx, discovery)
            ctx.check_cancel()
            ctx.gate.watchdog_tick()
            center = [nominal[j]+offset[j] for j in (0, 1)]
            _move_stage(ctx, original_pose.x-center[0], original_pose.y-center[1])
            autofocus(ctx, 5, attempts=1, half_range=.08)
            cursor = ctx.store.last_event_id()
            frame = microscope_frame(ctx, phase=f"{self.name}_map_{index}")
            record = _frame_record(frame)
            observed = analyzer(ctx).discovery_obstacles(frame)
            if observed["chip_id"] != chip_id:
                raise NeedsHumanError("Discovery tile does not observe the retained source chip")
            capture = [e for e in _transfer_events(ctx.store, ctx.run_id, "lab.frame", cursor)
                if e.payload.get("microscope_preview", {}).get("image_sha256") == record["sha256"]
                and e.payload["microscope_preview"].get("phase") == f"{self.name}_map_{index}"]
            if len(capture) != 1:
                raise NeedsHumanError("Discovery map lost its unique original analysis exposure")
            eid = record_evidence(ctx.store, LabEvidence(subject=chip_id, property="stamp_discovery_obstacles",
                bench="sim", value=observed, units="mm", frame="microscope_image_axes",
                observed_at=frame.timestamp, timebase="store_clock", source=Source(kind="synthetic",
                    reference=f"{ctx.run_id}:{self.name}", sha256=record["sha256"], event_ids=[capture[0].id])), run_id=ctx.run_id)
            actual = stage.get_sample_pose()
            discovery["maps"].append({"capture": _release_receipt(ctx.store, capture[0].id),
                "observation_event": _release_receipt(ctx.store, eid), "frame": record,
                "camera_center_mm": [original_pose.x-actual.x, original_pose.y-actual.y]})
            state["discovery"] = discovery
            _save_transfer(ctx, state)
            _discovery_budget(ctx, discovery)
        maps = []
        for item in discovery["maps"]:
            event = _historical_event(ctx.store, item["observation_event"], ctx.run_id, "lab.evidence")
            maps.append({"observation": event.payload["value"], "camera_center_mm": item["camera_center_mm"]})
        try:
            domain = choose_clear_domain(policy, maps, nominal_center=nominal,
                                         chip_polygon=chip_polygon, chip_id=chip_id)
        except ValueError as exc:
            raise NeedsHumanError(str(exc)) from exc
        center = domain["center_mm"]
        _move_stage(ctx, original_pose.x-center[0], original_pose.y-center[1])
        _, focus = _capture_chip_focus(ctx, self.name+"_focus", parked=True)
        discovery.update(state="prepared", domain=domain, focus_reference=focus,
                         discovery_stage_pose=stage.get_sample_pose().model_dump(mode="json"))
        _discovery_budget(ctx, discovery)
        state["discovery"] = discovery
        _save_transfer(ctx, state)
        self.result = {"profile": policy["id"], "state": "prepared", "map_frames": len(discovery["maps"])}

    def verify(self, ctx):
        result = getattr(self, "result", None)
        return Observation(Verdict.OK if result else Verdict.UNCLEAR, {"stamp_discovery": result},
                           message="Four original visible-obstacle maps and a distinct discovery focus")

    def probe_state(self, ctx):
        return ProbeResult.NOT_DONE

    def safe_state(self, ctx):
        _clear_lift(ctx, self.name)


class InitializeStamp(Step):
    """Declared SIM mapping of the v1 5x/L/Z/fixed focus initialization."""
    devices = (MANIP, MICROSCOPE, HEATER)
    max_corrections = 0

    def __init__(self, stamp_id: str, name: str) -> None:
        self.stamp_id, self.name = stamp_id, name

    def act(self, ctx, correction):
        c = _profile(ctx)
        state = _transfer_state(ctx)
        inits = state.setdefault("initializations", {})
        scope = ctx.gate.microscope(MICROSCOPE)
        init = inits.get(self.name)
        if init is None:
            init = {"stamp_id": self.stamp_id, "profile": c["id"],
                    "substrate_focus_mm": scope.get_focus(), "objective": scope.get_objective()}
            inits[self.name] = init
            _save_transfer(ctx, state)
        target = init["substrate_focus_mm"]+c["focus_offset_mm"]
        lo, hi = ctx.gate.limits.microscopes[MICROSCOPE].focus
        if init["stamp_id"] != self.stamp_id or init["profile"] != c["id"] or not lo <= target <= hi:
            raise NeedsHumanError("Fixed focus mapping is unsupported; no clamp applied")
        scope.set_objective(5)
        if scope.get_objective() != 5:
            raise NeedsHumanError("Low-power objective readback failed before holder engagement")
        move = ManipulatorTo("over_stage", name=self.name+"_engage", stamp_id=self.stamp_id)
        move.act(ctx, None)
        seen = move.verify(ctx)
        if seen.verdict != Verdict.OK:
            move.on_give_up(ctx, seen)
            raise NeedsHumanError(seen.message, observation=seen.to_dict())
        if ctx.params.get("stamp_discovery_policy") and self.name == "initialize_restored_discovery_source":
            self.discovery_engagement = {"stamp_id": self.stamp_id, "run_id": ctx.run_id,
                "handoff_observation": deepcopy(move.handoff_observation),
                "observation": {key: seen.to_dict()[key] for key in ("verdict", "measurements", "expected")}}
        scope.set_focus(target)
        if abs(scope.get_focus()-target) > 1e-8:
            raise NeedsHumanError("Fixed focus initialization readback failed")
        state = _transfer_state(ctx)
        state["active_initialization"] = self.name
        state["initializations"][self.name]["applied"] = True
        _save_transfer(ctx, state)
        self.result = {"stamp_id": self.stamp_id, "profile": c["id"], "focus_mm": target,
                       "substrate_focus_mm": init["substrate_focus_mm"], "objective": 5,
                       "actual_holder_pose": _pos(ctx), "source": "synthetic"}

    def verify(self, ctx):
        result = getattr(self, "result", None)
        return Observation(Verdict.OK if result else Verdict.UNCLEAR, result or {},
                           message="Fixed synthetic initialization" if result else "Initialization unverified")

    def probe_state(self, ctx):
        return ProbeResult.NOT_DONE

    def safe_state(self, ctx):
        ctx.gate.heater(HEATER).off()
        _clear_lift(ctx, self.name)


def _validate_ring_capture(store, run_id, name, record, receipt):
    event = _historical_event(store, receipt, run_id, "lab.frame")
    preview = event.payload.get("microscope_preview", {})
    meta = record["meta"]
    if (event.ts != record["captured_at"] or preview.get("timestamp") != event.ts
            or preview.get("phase") != name or preview.get("image") != record["image"]
            or preview.get("image_sha256") != record["sha256"]
            or any(preview.get(key) != meta.get(key) for key in
                   ("calibration", "objective", "focus_z", "focus_plane", "stage_kinematics"))):
        raise ValueError("Ring centre lost its exact original phase capture/context")
    _reference(record)
    return event


def _first_contact_binding(candidate, calibration, transfer_profile):
    """A saved v1 first-contact candidate must recompute from its own pixels (#204)."""
    candidate = json.loads(json.dumps(candidate))
    if candidate.get("minimum_area_mm2") != transfer_profile["minimum_area_mm2"]:
        raise ValueError("First-contact candidate changed its approved minimum area")
    observed = json.loads(json.dumps(v1_first_contact(_reference(candidate["reference"]), _reference(candidate["current"]),
                                     calibration, min_area_mm2=transfer_profile["minimum_area_mm2"])))
    if observed.get("found") is not True or any(candidate.get(key) != value for key, value in observed.items()):
        raise ValueError("First-contact optical candidate differs from its original pixels")
    return observed


def _first_ring_binding(store, run_id, phase, candidate, calibration, transfer_profile):
    """Recompute full same-exposure evidence; a cached boolean is insufficient.

    Live and saved evidence differ only by the JSON run store (tuples become
    lists), so both sides are compared in JSON-native form. Before this, every
    restart or replan after first contact was rejected (#109)."""
    candidate = json.loads(json.dumps(candidate))
    cue = json.loads(json.dumps(phase.get("ring_cue") or {}))
    phase = {**phase, "ring_cue": cue, "reference_event": json.loads(json.dumps(phase.get("reference_event")))}
    if cue.get("profile") != "same-exposure-ring-centre-v1":
        raise ValueError("First-contact ring centre has no supported same-exposure binding")
    if cue.get("reference_event") != phase.get("reference_event"):
        raise ValueError("Ring centre changed the original phase reference event")
    reference = _validate_ring_capture(store, run_id, phase["name"], cue["reference"], cue["reference_event"])
    current = _validate_ring_capture(store, run_id, phase["name"], cue["current"], cue["current_event"])
    if current.id <= reference.id or current.ts < reference.ts:
        raise ValueError("Ring centre exposure does not follow its actual phase reference")
    if candidate.get("minimum_area_mm2") != transfer_profile["minimum_area_mm2"]:
        raise ValueError("First-contact candidate changed its approved minimum area")
    observed = json.loads(json.dumps(v1_first_contact(_reference(candidate["reference"]), _reference(candidate["current"]),
                                     calibration, min_area_mm2=transfer_profile["minimum_area_mm2"])))
    if any(candidate.get(key) != value for key, value in observed.items()):
        raise ValueError("First-contact optical candidate differs from its original pixels")
    recomputed = json.loads(json.dumps(precontact_ring_evidence(_reference(cue["reference"]), _reference(cue["current"]), calibration)))
    if recomputed != cue.get("observation"):
        raise ValueError("Ring family differs from its original same-exposure pixels")
    binding = json.loads(json.dumps(ring_candidate_binding(cue, candidate,
        (abs(calibration.mm_per_pixel_x), abs(calibration.mm_per_pixel_y)))))
    if not binding.get("compatible") or binding != cue.get("binding"):
        raise ValueError("Ring centre and optical candidate are unresolved or incompatible")
    center = recomputed["selected"]["centre_px"]
    if (candidate.get("aim_center_px") != center
            or candidate.get("aim_center_source") != "same_exposure_ring_family"):
        raise ValueError("New-design reaim must use its admitted ring centre")
    if phase.get("complete") and not phase.get("discovery_only"):
        image = load_pixels(_reference(candidate["reference"]))
        anchor = [image.width/2, image.height/2]
        target = [anchor[0], anchor[1]-transfer_profile["first_contact_offset_mm"]/
                  abs(calibration.mm_per_pixel_y)]
        delta = [(target[0]-center[0])*calibration.mm_per_pixel_x,
                 (target[1]-center[1])*calibration.mm_per_pixel_y]
        if (candidate.get("image_anchor_px") != anchor or candidate.get("first_contact_target_px") != target
                or candidate.get("reaim_requested_mm") != delta):
            raise ValueError("Saved reaim differs from its exact selected ring centre")
        inputs = candidate.get("reaim_inputs")
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("Saved ring reaim lost its two actual axis readbacks")
        targets = {axis: inputs[0]["before"][axis]+delta[j] for j, axis in enumerate(("x", "y"))}
        for j, axis in enumerate(("x", "y")):
            if (inputs[j].get("axis") != axis or inputs[j].get("requested_mm") != targets[axis]
                    or j and inputs[j]["before"] != inputs[j-1]["after"]):
                raise ValueError("Saved ring reaim axis mapping or actual chronology changed")
        if candidate.get("reaim_readback") != inputs[-1]["after"]:
            raise ValueError("Saved ring reaim differs from its final actual readback")
    return binding


def _discovery_admit_actual_head(ctx, phase):
    """Re-admit the actual nominal head against four ORIGINAL map observations."""
    from qumus_simlab.records.discovery import choose_clear_domain

    state = _transfer_state(ctx)
    discovery = state["discovery"]
    reference = phase["reference"]
    if reference["meta"]["stage_kinematics"] != discovery["focus_reference"]["meta"]["stage_kinematics"]:
        raise NeedsHumanError("Discovery stage changed after its distinct parked focus")
    refs = _contact_capture_receipts(ctx.store, ctx.run_id, phase["name"], reference,
                                     phase["reference_event"]["event_id"]-1)
    event = _contact_encoder_event(ctx.store, refs["encoder"], ctx.run_id)
    head = _contact_nominal_head_projection(event)
    actual_stage = reference["meta"]["stage_kinematics"]["sample_pose"]
    original = discovery["original_stage_pose"]
    center = [original[key]-actual_stage[key]+head["nominal_head_world_xyz_mm"][key]-head["camera_world_xy_mm"][i]
              for i, key in enumerate(("x", "y"))]
    maps = [{"observation": _historical_event(ctx.store, item["observation_event"], ctx.run_id, "lab.evidence").payload["value"],
             "camera_center_mm": item["camera_center_mm"]} for item in discovery["maps"]]
    domain = choose_clear_domain(discovery["policy"], maps, nominal_center=center,
        chip_polygon=discovery["chip_polygon_mm"], chip_id=discovery["chip_id"], exact_center=True)
    discovery.update(actual_head_context=head, actual_head_reference=refs,
                     actual_head_domain=domain, first_contact_reference=deepcopy(reference),
                     first_contact_step=phase["name"])
    state["discovery"] = discovery
    _save_transfer(ctx, state)


def _publish_transfer_analysis(ctx, name, frame):
    """One immutable analysis publication for captured transfer frames."""
    record = _frame_record(frame)
    ctx.store.emit("lab.frame", {"device": MICROSCOPE, "command": "capture", "microscope_preview": {
        "image": record["image"], "image_sha256": record["sha256"], "phase": name,
        "timestamp": frame.timestamp, "source": "synthetic", "objective": frame.meta["objective"],
        "focus_z": frame.meta["focus_z"], "focus_plane": frame.meta.get("focus_plane"),
        "calibration": frame.meta["calibration"], "stage_kinematics": frame.meta["stage_kinematics"],
        "fov_mm": frame.meta.get("fov_mm"), "immutable": True, "verification": False,
        "purpose": "analysis", "state_binding": "capture.pose_binding"}}, run_id=ctx.run_id)
    return record


def _discovery_publish_capture(ctx, phase, frame):
    _publish_transfer_analysis(ctx, phase["name"], frame)
    _discovery_charge_capture(ctx, phase, frame)


def _discovery_charge_capture(ctx, phase, frame):
    record = _frame_record(frame)
    c = _phase_budget(ctx, phase)
    previous = phase["observed_at"]
    phase.update(observations=phase["observations"]+1, observed_at=frame.timestamp,
                 last_frame=record, last_pose=_pos(ctx))
    _save_phase(ctx, phase)
    if (ctx.clock.now()-previous > c["max_unobserved_s"]
            or phase["observations"] > c["max_observations"]
            or ctx.clock.now()-phase["started_at"] > c["max_total_s"]):
        raise NeedsHumanError("Discovery AF/revisit lost its original observation/time budget")


def _precontact_exposure_clear(phase, previous, calibration, minimum_area_mm2):
    """No contact front, and either no changed region or a recorded D8 ring-wait exposure."""
    from qumus_simlab.perception.tracking import contact_front_domain

    if contact_front_domain(_reference(previous))[0].has_contact:
        return False
    if previous.get("sha256") in phase.get("ring_wait_images", []):
        return True
    return not v1_first_contact(_reference(phase["reference"]), _reference(previous), calibration,
                                min_area_mm2=minimum_area_mm2)["found"]


def _discovery_hold_and_lift(ctx, phase, candidate):
    """First-candidate monitored hold/lift; source restoration is a distinct Step."""
    from qumus_simlab.records.discovery import settling_envelope_mm, validate_policy

    policy = validate_policy(ctx.params["stamp_discovery_policy"])
    state = _transfer_state(ctx)
    discovery = state["discovery"]
    if discovery.get("state") != "prepared" or discovery.get("policy") != policy:
        raise NeedsHumanError("First candidate lacks its original four-map discovery admission")
    if candidate["current"]["meta"]["stage_kinematics"] != discovery["focus_reference"]["meta"]["stage_kinematics"]:
        raise NeedsHumanError("Discovery source changed between its focus and optical candidate")
    previous = phase.get("previous_frame", phase["reference"])
    calibration = _image_calibration(_reference(previous))
    # D8 (#129) waits, after the first changed region, until the ring family
    # resolves. The exposure before the candidate is then a recorded ring-wait
    # exposure: changed by the gap's ring tint, judged not yet a candidate,
    # and it must still show no contact front.
    if not _precontact_exposure_clear(phase, previous, calibration, _profile(ctx)["minimum_area_mm2"]):
        raise NeedsHumanError("The last original precontact exposure is ambiguous or already contacting")
    envelope = settling_envelope_mm(policy)
    cue = phase["ring_cue"]["observation"]["selected"]
    image = load_pixels(_reference(candidate["current"]))
    cx, cy = candidate["aim_center_px"]
    offset = math.hypot((cx-image.width/2)*calibration.mm_per_pixel_x,
                        (cy-image.height/2)*calibration.mm_per_pixel_y)
    if (offset > envelope["apex_prior_radius_mm"]+math.sqrt(2)*policy["absolute_drift_per_axis_mm"]
            or offset+envelope["settled_front_radius_mm"]+cue["uncertainty_radius_mm"]+.048969022001210026
                > policy["clear_radius_mm"]):
        raise NeedsHumanError("Observed ring centre/settling envelope exceeds the fixed clear discovery domain")
    c = _phase_budget(ctx, phase)
    discovery["candidate_previous"] = previous
    discovery["settling_envelope"] = envelope
    discovery["reserved_work"] = _discovery_exit_reserve(ctx, phase, discovery)
    state["discovery"] = discovery
    _save_transfer(ctx, state)
    from qumus_simlab.records.discovery import original_low_z_prefix

    original_low_z_prefix(_discovery_rows(ctx.store, discovery), run_id=ctx.run_id,
                          candidate=candidate, discovery=discovery, candidate_segment_only=True)
    af_started = ctx.clock.now()
    autofocus(ctx, 5, attempts=1, half_range=c["post_phase_focus_half_range_mm"], plane="chip",
              on_capture=lambda frame: _discovery_publish_capture(ctx, phase, frame))
    if ctx.clock.now()-af_started > 1.538889+1e-9:
        raise NeedsHumanError("Actual discovery autofocus exceeded its admitted low-Z duration")
    discovery["post_candidate_focus_event_id"] = ctx.store.get_run(ctx.run_id)["meta"]["last_focus_event_id"]
    if ctx.store.get_run(ctx.run_id)["meta"]["last_focus_observation"].get("peak_observable") is not True:
        raise NeedsHumanError("Discovery hold requires the actual observed post-candidate autofocus")
    held = (_pos(ctx), ctx.gate.stage(STAGE).kinematics(),
            ctx.gate.microscope(MICROSCOPE).get_focus(), ctx.gate.microscope(MICROSCOPE).get_objective())
    start = ctx.clock.now()
    capture_start = ctx.store.last_event_id()
    while ctx.clock.now()-start < policy["hold_s"]:
        ctx.check_cancel()
        ctx.gate.watchdog_tick()
        ctx.clock.sleep(min(policy["hold_poll_s"], policy["hold_s"]-(ctx.clock.now()-start)))
        _phase_capture(ctx, phase)
        if held != (_pos(ctx), ctx.gate.stage(STAGE).kinematics(),
                    ctx.gate.microscope(MICROSCOPE).get_focus(), ctx.gate.microscope(MICROSCOPE).get_objective()):
            raise NeedsHumanError("Discovery hold moved its bound holder/source/focus/objective")
    if not 5 <= ctx.clock.now()-start <= 5.30+1e-9:
        raise NeedsHumanError("Actual discovery hold exceeded its admitted low-Z duration")
    discovery["hold"] = {"started_at": start, "completed_at": ctx.clock.now(), "held": list(held),
                         "start_event_id": capture_start, "end_event_id": ctx.store.last_event_id()}
    state = _transfer_state(ctx)
    state["discovery"] = discovery
    _save_transfer(ctx, state)
    _guarded_z(ctx, phase, max(_pos(ctx)["z"]+.2, _ta(ctx)["safe_z"]))
    discovery["clear_lift_end_event_id"] = ctx.store.last_event_id()
    discovery["low_z_prefix"] = original_low_z_prefix(_discovery_rows(ctx.store, discovery),
        run_id=ctx.run_id, candidate=candidate, discovery=discovery)
    _discovery_budget(ctx, discovery)
    discovery.update(state="contact_discovered", first_contact_step=phase["name"],
                     first_candidate=deepcopy(candidate))
    state = _transfer_state(ctx)
    state["discovery"] = discovery
    _save_transfer(ctx, state)


def _discovery_ready_value(discovery):
    """The ready assertion binds the whole discovery journal by digest.

    The complete record (three source references, maps, candidate) is about
    120 KB, past the 64 KiB assertion bound; it stays in the transfer state
    and the validator recomputes this digest from it.
    """
    return {"discovery_sha256": _digest(discovery), "ready_placement": deepcopy(discovery["ready_placement"])}


class RestoreStampDiscoverySource(Step):
    """A distinct parked revisit and ready-placement boundary after discovery."""
    name = "restore_stamp_discovery_source"
    devices = (STAGE, MICROSCOPE, MANIP, HEATER)
    max_corrections = 0
    safe_to_resume = False

    def act(self, ctx, correction):
        from qumus_simlab.records.discovery import validate_policy

        policy = validate_policy(ctx.params.get("stamp_discovery_policy"))
        state = _transfer_state(ctx)
        discovery = state["discovery"]
        if discovery.get("state") != "contact_discovered":
            raise NeedsHumanError("Source restore requires the original completed clear discovery/lift")
        contact = state["phases"][discovery["first_contact_step"]]
        candidate = deepcopy(contact["first_contact_region"])
        evidence = _historical_event(ctx.store, discovery["candidate_evidence"], ctx.run_id, "lab.evidence")
        if evidence.payload["value"] != candidate or candidate != discovery["first_candidate"]:
            raise NeedsHumanError("Source restore lost its immutable first-contact candidate")
        discovery["state"] = "restoring_source"
        state["discovery"] = discovery
        _save_transfer(ctx, state)
        _discovery_budget(ctx, discovery)
        starts = list(_transfer_events(ctx.store, ctx.run_id, "step.started"))
        if not starts or starts[-1].payload.get("name") != self.name:
            raise NeedsHumanError("Source restoration lacks its distinct actual checkpoint start")
        discovery["restore_start_event_id"] = starts[-1].id
        park = ManipulatorTo("park", name="park_for_discovery_source_restore")
        park.act(ctx, None)
        if park.verify(ctx).verdict != Verdict.OK:
            raise NeedsHumanError("Discovery source revisit requires the real verified parked holder")
        stage = ctx.gate.stage(STAGE)
        original = discovery["original_stage_pose"]
        before = stage.get_sample_pose().model_dump(mode="json")
        _move_stage(ctx, original["x"], original["y"])
        actual = stage.get_sample_pose().model_dump(mode="json")
        tol = ctx.verification["transfer_stage"]["tolerance_mm"]
        if (math.hypot(actual["x"]-original["x"], actual["y"]-original["y"]) > tol
                or actual["z"] != original["z"] or actual["rotation"] != original["rotation"]):
            raise NeedsHumanError("Composed stage readback did not restore the original source request")
        ctx.gate.microscope(MICROSCOPE).set_objective(20)
        phase = _begin_phase(ctx, self.name, difference=False)
        frame, restored = _capture_chip_focus(ctx, self.name, parked=True,
            focus_seed=discovery["original_reference"],
            on_capture=lambda frame: _discovery_charge_capture(ctx, phase, frame),
            on_af_capture=lambda frame: _discovery_publish_capture(ctx, phase, frame))
        fid, chip = discovery["source_id"], discovery["chip_id"]
        identity = RefindFlake(fid, chip, ctx.params["materials"][fid])._source_identity(ctx, frame)
        expected = discovery["original_reference"]["target_offset_mm"]
        detections = analyzer(ctx).find_flakes(frame, ctx.params["materials"][fid])
        chosen = min(detections, key=lambda d: math.hypot(d.dx_mm-expected[0], d.dy_mm-expected[1])) if detections else None
        if (chosen is None or not chosen.meta.get("outline_complete")
                or math.hypot(chosen.dx_mm-expected[0], chosen.dy_mm-expected[1]) > .012):
            raise NeedsHumanError("Restored selected source is not independently observed in its original ROI")
        identity.update(design_shift_chip_mm=list(ctx.params["design_shifts_mm"].get(fid, (0.,0.))),
                        post_shift_capture=_frame_record(frame))
        restored.update(source_id=fid, source_chip_id=chip, target_offset_mm=[chosen.dx_mm, chosen.dy_mm],
                        design_source_rotation_deg=discovery["original_reference"]["design_source_rotation_deg"],
                        source_identity=identity)
        discovery.update(restored_reference=restored, restore_before=before, restored_stage_pose=actual)
        state = _transfer_state(ctx)
        state["discovery"] = discovery
        state["source_references"][fid] = restored
        _save_transfer(ctx, state)
        initialization = InitializeStamp(ctx.params["stamp_id"], name="initialize_restored_discovery_source")
        initialization.act(ctx, None)
        from qumus_simlab.records.lab_geometry import LabEvidence, Source, record_evidence

        engagement = initialization.discovery_engagement
        event_id = record_evidence(ctx.store, LabEvidence(subject=ctx.params["stamp_id"],
            property="stamp_discovery_source_engagement", bench="sim", value=engagement,
            units="mm", frame="holder_axes_and_overview", observed_at=ctx.clock.now(), timebase="store_clock",
            source=Source(kind="synthetic", reference=f"{ctx.run_id}:{self.name}", sha256=_digest(engagement),
                event_ids=[capture["capture_event_id"] for capture in engagement["handoff_observation"]["captures"]])), run_id=ctx.run_id)
        discovery["restore_engagement"] = _release_receipt(ctx.store, event_id)
        _discovery_budget(ctx, discovery)
        before = _pos(ctx)
        discovery["reaim_start_event_id"] = ctx.store.last_event_id()
        calibration = _image_calibration(_reference(candidate["current"]))
        origin = candidate["actual_holder_pose"]
        shift = [before[axis]-origin[axis] for axis in ("x", "y")]
        center = [candidate["aim_center_px"][0]+shift[0]/calibration.mm_per_pixel_x,
                  candidate["aim_center_px"][1]+shift[1]/calibration.mm_per_pixel_y]
        image = load_pixels(_reference(candidate["reference"]))
        anchor = [image.width/2, image.height/2]
        aim = [anchor[0], anchor[1]-_profile(ctx)["first_contact_offset_mm"]/abs(calibration.mm_per_pixel_y)]
        delta = [(aim[0]-center[0])*calibration.mm_per_pixel_x,
                 (aim[1]-center[1])*calibration.mm_per_pixel_y]
        reaim = AdjustFirstContact(ctx.params["stamp_id"], name=self.name)
        actual_holder = reaim._reaim(ctx, phase, *delta)
        discovery["reaim_end_event_id"] = ctx.store.last_event_id()
        _discovery_budget(ctx, discovery)
        phase.pop("pending_intent", None)
        phase["complete"] = True
        _save_phase(ctx, phase)
        ready = {"stamp_id": ctx.params["stamp_id"], "run_id": ctx.run_id,
            "profile": _profile(ctx)["id"], "working_xy_mm": [actual_holder["x"],actual_holder["y"]],
            "first_contact_step": discovery["first_contact_step"], "candidate_z_mm": origin["z"],
            "candidate_evidence": discovery["candidate_evidence"],
            "source_anchor": {"source_id": fid, "image_sha256": restored["sha256"],
                              "focus_binding_sha256": _digest(restored["focus_binding"])},
            "discovery_profile": policy["id"], "restored_aim_center_px": center,
            "holder_before_reaim": before, "holder_origin_shift_mm": shift,
            "image_anchor_px": anchor, "first_contact_target_px": aim,
            "reaim_requested_mm": delta, "reaim_inputs": phase["xy_inputs"], "reaim_readback": actual_holder}
        discovery.update(state="ready", ready_placement=ready, restore_step=self.name)
        state = _transfer_state(ctx)
        state["discovery"] = discovery
        state.setdefault("placements", {})[ctx.params["stamp_id"]] = ready
        _save_transfer(ctx, state)
        self.result = {"stamp_discovery_ready": ready}

    def verify(self, ctx):
        result = getattr(self, "result", None)
        return Observation(Verdict.OK if result else Verdict.UNCLEAR, result or {},
                           message="Original source revisited at park; transformed clear-height ready placement")

    def on_success(self, ctx, obs):
        from qumus_simlab.records.lab_geometry import LabEvidence, Source, record_evidence

        state = _transfer_state(ctx)
        discovery = state["discovery"]
        if obs is None or obs.measurements.get("stamp_discovery_ready") != discovery["ready_placement"]:
            raise NeedsHumanError("Ready placement lacks its actual verified restore observation")
        eid = record_evidence(ctx.store, LabEvidence(subject=ctx.params["stamp_id"], property="stamp_discovery_ready",
            bench="sim", value=_discovery_ready_value(discovery), units="mm,s", frame="registered_microscope_axes",
            observed_at=ctx.clock.now(), timebase="store_clock", source=Source(kind="synthetic",
                reference=f"{ctx.run_id}:{self.name}", sha256=_digest(discovery))), run_id=ctx.run_id)
        discovery["ready_evidence"] = _release_receipt(ctx.store, eid)
        state["discovery"] = discovery
        _save_transfer(ctx, state)
        try:
            _validate_discovery_ready(ctx.store, ctx.run_id)
        except (ValueError, KeyError, TypeError) as exc:
            raise NeedsHumanError("Original discovery/restore receipt did not admit its ready placement") from exc

    def safe_state(self, ctx):
        ctx.gate.heater(HEATER).off()
        _clear_lift(ctx, self.name)

    def probe_state(self, ctx):
        return ProbeResult.NOT_DONE



def _landing_features(pixels, policy):
    """Visible-feature mask: pixels off a smooth background, in components of
    at least `landing_feature_min_px` (single-pixel noise is ignored)."""
    import numpy as np
    from scipy import ndimage

    image = np.asarray(pixels, dtype=float)
    background = np.stack([ndimage.gaussian_filter(image[..., c], policy["landing_background_sigma_px"])
                           for c in range(3)], axis=2)
    feature = np.max(np.abs(image-background), axis=2) > policy["landing_feature_dn"]
    labels, count = ndimage.label(feature)
    if not count:
        return feature
    sizes = ndimage.sum(feature, labels, range(1, count+1))
    return np.isin(labels, 1+np.flatnonzero(sizes >= policy["landing_feature_min_px"]))


def _first_touch_landing(ctx, policy, calibration, pixels=None):
    """Find a clean first-touch landing beside the centred source, near to far.

    The apex is predicted on the optical axis. Rings start at
    `first_touch_min_mm`; each ring tries its directions image-up first. A
    landing qualifies when its disc is inside the observed 5x view, on the
    measured chip, clear of catalogued flakes and free of visible features.
    Uncatalogued material that the image does not resolve is not seen.
    """
    import numpy as np

    from qumus_simlab.flows.library.search import _stage_to_local

    source_id = ctx.params["flake_ids"][0]
    chip_id = ctx.params["flake_chips"][source_id]
    chip = require_sample(ctx, chip_id, "chip")
    corner, angle = chip["props"]["corner_stage_xy"], _chip_rotation(ctx, chip_id)
    width, height = chip["props"]["size_mm"]
    pose = ctx.gate.stage(STAGE).get_sample_pose()
    radius = policy["first_touch_landing_radius_mm"]
    flakes = [(f["id"], f["props"]["rel_corner_mm"],
               policy["flake_extent_factor"]*math.sqrt(max(f["props"].get("area_um2") or 0., 0.)/math.pi)/1000)
              for f in ctx.store.list_samples(kind="flake", location=chip_id)
              if f["props"].get("rel_corner_mm")]
    sx, sy = abs(calibration.mm_per_pixel_x), abs(calibration.mm_per_pixel_y)
    signs = (math.copysign(1., calibration.mm_per_pixel_x), math.copysign(1., calibration.mm_per_pixel_y))
    view = (480, 320) if pixels is None else pixels.size
    features = None if pixels is None else _landing_features(pixels, policy)
    yy, xx = np.mgrid[:view[1], :view[0]]
    n = policy["first_touch_directions"]
    order = sorted(range(n), key=lambda k: (min(k, n-k), k))
    candidates, distance = [], policy["first_touch_min_mm"]
    while True:
        if distance+radius > max((view[0]/2-8)*sx, (view[1]/2-32)*sy):
            break  # no direction can keep the landing disc inside the view
        for k in order:
            a = 2*math.pi*k/n
            image = (math.sin(a), -math.cos(a))  # unit step in image pixels, k=0 is image-up
            px, py = view[0]/2+image[0]*distance/sx, view[1]/2+image[1]*distance/sy
            delta = [image[0]*signs[0]*distance, image[1]*signs[1]*distance]
            lx, ly = _stage_to_local(tuple(corner), angle, pose.x-delta[0], pose.y-delta[1])
            reasons = []
            if not (8 <= px-radius/sx and px+radius/sx < view[0]-8 and 30 <= py-radius/sy and py+radius/sy < view[1]-32):
                reasons.append("outside_view")
            if not (radius <= lx <= width-radius and radius <= ly <= height-radius):
                reasons.append("off_chip")
            near = [fid for fid, (fx, fy), extent in flakes if math.hypot(lx-fx, ly-fy) < radius+extent]
            if near:
                reasons.append("near:" + ",".join(near))
            if features is not None and "outside_view" not in reasons:
                disc = ((xx-px)*sx)**2+((yy-py)*sy)**2 <= radius**2
                if bool(np.any(features & disc)):
                    reasons.append("visible_feature")
            candidate = {"distance_mm": round(distance, 6), "direction_deg": 360*k/n, "holder_delta_mm": delta,
                         "landing_px": [px, py], "landing_chip_local_mm": [lx, ly], "rejected": reasons}
            candidates.append(candidate)
            if not reasons:
                return {"chip_id": chip_id, "chip_size_mm": [width, height], "radius_mm": radius,
                        "catalogued_flakes": len(flakes), "image_checked": features is not None,
                        "candidates": candidates, "chosen": candidate}
        distance += policy["first_touch_step_mm"]
    return {"chip_id": chip_id, "chip_size_mm": [width, height], "radius_mm": radius,
            "catalogued_flakes": len(flakes), "image_checked": features is not None,
            "candidates": candidates, "chosen": None}


class AdjustFirstContact(Step):
    """Observe one first candidate, lift clear and reaim XY once."""
    devices = (MANIP, MICROSCOPE, HEATER)
    max_corrections = 0
    safe_to_resume = False
    pause_inside = True

    def __init__(self, stamp_id: str, name: str) -> None:
        self.stamp_id, self.name = stamp_id, name

    def act(self, ctx, correction):
        c = _profile(ctx)
        try:
            first = _transfer_state(ctx)["source_references"][ctx.params["flake_ids"][0]]
            _validate_focus_reference(first, ctx.run_id, parked=True, store=ctx.store)
        except (KeyError, TypeError, ValueError) as exc:
            raise NeedsHumanError("First candidate lacks its admitted first-source focus reference") from exc
        if ctx.params.get("device_design") and self.name not in _transfer_state(ctx).get("phases", {}):
            if not ctx.params.get("stamp_discovery_policy"):
                # With stamp discovery the head already sits on an admitted
                # all-material clear domain (re-admitted below at its exact
                # centre), so moving beside the source would leave it.
                self._beside_first_touch(ctx)
                self._v1_initialize(ctx)
        phase = _begin_phase(ctx, self.name)
        if phase.get("complete"):
            self.result = phase["first_contact_region"]
            return
        if phase.get("observations", 0) > phase.get("initial_observations", 1):
            raise NeedsHumanError("Interrupted first-contact adjustment requires review")
        if ctx.params.get("stamp_discovery_policy"):
            _discovery_admit_actual_head(ctx, phase)
        reference = _reference(phase["reference"])
        calibration = _image_calibration(reference)
        candidate = None
        def detect(frame):
            nonlocal candidate
            observed = v1_first_contact(reference, frame, calibration, min_area_mm2=c["minimum_area_mm2"])
            if not observed["found"]:
                return False
            candidate = {**observed, "source": "synthetic", "kind": "observed_first_contact_candidate",
                         "reference": phase["reference"], "current": _frame_record(frame),
                         "actual_holder_pose": _pos(ctx), "profile": c["id"],
                         "scope": "Image-plane region; not PDMS centre, world point or interferometric contact"}
            return True
        floor = ctx.gate.manipulator_z_floor(MANIP)
        # v1 jogs 10 um per step with no real step limit (MAX_STEP 10000).
        # Anti-crush floor (#204): stop and call a person once the touch
        # search would go below the v1 drop press height without a change.
        while candidate is None:
            target = _pos(ctx)["z"]-c["first_touch_intent_mm"]
            if floor is not None and target < floor:
                raise NeedsHumanError("First-contact candidate unavailable before physical floor")
            if target < c["first_contact_min_z_mm"]-1e-9:
                raise NeedsHumanError("No first contact seen above the anti-crush floor; a person must check the stamp")
            _guarded_z(ctx, phase, target, speed=c["descent_speed_mm_s"], observe=detect)
        phase["difference_active"] = False
        phase["first_contact_region"] = candidate
        _save_phase(ctx, phase)
        if ctx.params.get("stamp_discovery_policy"):
            _discovery_hold_and_lift(ctx, phase, candidate)
            phase.update(complete=True, discovery_only=True, first_contact_region=deepcopy(candidate))
            phase.pop("pending_intent", None)
            _save_phase(ctx, phase)
            self.result = candidate
            return
        else:
            autofocus(ctx, 5, attempts=1, half_range=c["post_phase_focus_half_range_mm"], plane="chip")
            # Preserve the old protocol exactly for every historical profile.
            _phase_capture(ctx, phase)
            _guarded_z(ctx, phase, max(_pos(ctx)["z"]+.2, _ta(ctx)["safe_z"]))
            center = candidate["centroid_px"]
        pixels = load_pixels(reference)
        anchor = [pixels.width/2, pixels.height/2]
        aim = [anchor[0], anchor[1]-c["first_contact_offset_mm"]/abs(calibration.mm_per_pixel_y)]
        dx = (aim[0]-center[0])*calibration.mm_per_pixel_x
        dy = (aim[1]-center[1])*calibration.mm_per_pixel_y
        actual = self._reaim(ctx, phase, dx, dy)
        state = _transfer_state(ctx)
        state.setdefault("placements", {})[self.stamp_id] = {
            "stamp_id": self.stamp_id, "run_id": ctx.run_id, "profile": c["id"],
            "working_xy_mm": [actual["x"], actual["y"]], "first_contact_step": self.name,
            "candidate_z_mm": candidate["actual_holder_pose"]["z"],
            "source_anchor": {"source_id": ctx.params["flake_ids"][0],
                "image_sha256": state["source_references"][ctx.params["flake_ids"][0]]["sha256"],
                "focus_binding_sha256": _digest(state["source_references"][ctx.params["flake_ids"][0]]["focus_binding"])}}
        _save_transfer(ctx, state)
        candidate.update(reaim_requested_mm=[dx, dy], reaim_readback=actual,
                         reaim_inputs=phase["xy_inputs"],
                         image_anchor_px=anchor, first_contact_target_px=aim)
        phase.pop("pending_intent")
        phase["complete"] = True
        phase["first_contact_region"] = candidate
        _save_phase(ctx, phase)
        self.result = candidate

    def _beside_first_touch(self, ctx):
        """One clear-height holder offset so the first touch misses the source.

        Refind centres the source on the optical axis and the holder preset puts
        the apex there too, so an unshifted first touch presses on the flake and
        the lift carries it away (#136). The shift is along image -y, the side
        the later one-time reaim aims at, before the coarse phase captures its
        own reference. It certifies no contact position.
        """
        policy = ctx.params["first_contact_sampling"]
        phase = _begin_phase(ctx, "beside_first_touch_"+self.stamp_id, difference=False)
        if phase.get("complete"):
            return
        if "xy_inputs" in phase:
            raise NeedsHumanError("Interrupted beside-source first-touch offset requires review")
        scope = ctx.gate.microscope(MICROSCOPE)
        previous = scope.get_focus()
        scope.set_objective(5)
        autofocus(ctx, 5, attempts=1, half_range=_profile(ctx)["post_phase_focus_half_range_mm"], plane="chip")
        frame = scope.capture()
        scope.set_focus(previous)
        record = _frame_record(frame)
        calibration = _image_calibration(_reference(record))
        landing = _first_touch_landing(ctx, policy, calibration, load_pixels(_reference(record)))
        landing["image"] = record
        phase["landing"] = landing
        if landing["chosen"] is None:
            _save_phase(ctx, phase)
            raise NeedsHumanError("No first-touch landing beside the source is on the chip and clear of catalogued flakes")
        delta = landing["chosen"]["holder_delta_mm"]
        actual = self._reaim(ctx, phase, *delta)
        phase.pop("pending_intent", None)
        phase.update(complete=True, requested_mm=delta, readback=actual)
        _save_phase(ctx, phase)

    def _v1_initialize(self, ctx):
        """v1 molecule_initialize_stamp: holder to Z 5.4 at 0.1 mm/s (#204).

        The touching search then starts from this preset, as in v1, instead of
        from the clear height. The descent is still observed on the usual
        cadence; the touch search below it stops at the anti-crush floor.
        """
        c = _profile(ctx)
        phase = _begin_phase(ctx, "v1_initialize_stamp_"+self.stamp_id, difference=False)
        if phase.get("complete"):
            return
        if _pos(ctx)["z"] > c["initialize_z_mm"]:
            _guarded_z(ctx, phase, c["initialize_z_mm"], speed=c["initialize_speed_mm_s"])
        phase.pop("pending_intent", None)
        phase["complete"] = True
        _save_phase(ctx, phase)

    def _reaim(self, ctx, phase, dx, dy):
        _profile(ctx)
        old = _pos(ctx)
        tol = ctx.verification["manipulator"]["tolerance_mm"]
        if (old["z"] < _ta(ctx)["safe_z"]-tol
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in [dx, dy, *old.values()])):
            raise NeedsHumanError("One-time XY reaim needs finite actual inputs at declared clearance")
        if any(p.get("pending_intent") for p in _transfer_state(ctx).get("phases", {}).values()):
            raise NeedsHumanError("Unresolved motion prevents another XY reaim")
        targets = {"x": old["x"]+dx, "y": old["y"]+dy}
        phase["xy_inputs"] = []
        for axis in ("x", "y"):
            ctx.check_cancel()
            before = _pos(ctx)
            phase["pending_intent"] = {"axes": "xy", "axis": axis, "before": before,
                "requested_mm": targets[axis], "at": ctx.clock.now(), "mapping": _profile(ctx)["holder_xy_mapping"]}
            _save_phase(ctx, phase)
            _move(ctx, axis, targets[axis])
            after = _pos(ctx)
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in after.values()):
                raise NeedsHumanError("One-time reaim has invalid actual encoder readbacks")
            actual = {key: after[key]-before[key] for key in ("x", "y", "z")}
            phase["xy_inputs"].append({"axis": axis, "requested_mm": targets[axis],
                "before": before, "after": after, "actual_delta_mm": actual, "readback_at": ctx.clock.now()})
            _save_phase(ctx, phase)
            if (abs(after[axis]-targets[axis]) > tol
                    or any(abs(actual[key]) > tol for key in ("x", "y", "z") if key != axis)):
                raise NeedsHumanError("One-time XY reaim actual axis/readback acceptance failed")
        return _pos(ctx)

    def verify(self, ctx):
        value = getattr(self, "result", None)
        return Observation(Verdict.OK if value else Verdict.UNCLEAR,
                           {"first_contact_region": value}, images=[value["current"]["image"]] if value else [],
                           message=(("Discovery candidate, monitored hold and clear lift; source restoration follows"
                                     if ctx.params.get("stamp_discovery_policy") else "First-contact candidate and one clear-height reaim")
                                    if value else "Candidate unavailable"))

    def on_success(self, ctx, obs):
        from qumus_simlab.records.lab_geometry import LabEvidence, Source, record_evidence
        if obs is None or not obs.measurements.get("first_contact_region"):
            raise NeedsHumanError("First-contact evidence cannot be synthesized during recovery")
        value = obs.measurements["first_contact_region"]
        calibration = value["current"]["meta"]["calibration"]
        state = _transfer_state(ctx)
        if _digest(state["phases"][self.name]["first_contact_region"]) != _digest(value):
            raise NeedsHumanError("First-contact verified observation changed")
        event_id = record_evidence(ctx.store, LabEvidence(subject=self.stamp_id, property="first_contact_region",
            bench=ctx.store.bench, value=value, units="px,mm", frame="microscope_image",
            observed_at=value["current"]["captured_at"], timebase="store_clock",
            calibration_id=calibration["id"], calibration_version=calibration["version"],
            source=Source(kind="synthetic", reference=f"{ctx.run_id}:{self.name}",
                          sha256=value["current"]["sha256"])), run_id=ctx.run_id, supersede_previous=True)
        event = ctx.store.events_after(event_id-1, limit=1)[0]
        target = state["discovery"] if ctx.params.get("stamp_discovery_policy") else state["placements"][self.stamp_id]
        target["candidate_evidence"] = {"event_id": event_id, "event_sha256": _digest(event.payload)}
        _save_transfer(ctx, state)

    def safe_state(self, ctx):
        ctx.gate.heater(HEATER).off()
        _clear_lift(ctx, self.name)

    def before_park(self, ctx):
        self.safe_state(ctx)


class RefindElectrodeSite(Step):
    """v1 accurate refind of the electrode site (molecule_flake_refind_accurate, #204).

    v1 goes to the electrode's recorded place at 20x, matches the centre 20 %
    of the user's own 20x photo of the device site (edges, +-15 deg) in the new
    image and centres the stage on the match. Here the match moves the target
    chip's corner by the same offset, so every later move to the site lands on
    the electrodes the person photographed. Rotation stays measured from the
    corners; v1 discards the match angle too. A match farther than the
    configured shift is not used and hands the target back."""

    devices = (STAGE, MICROSCOPE)
    max_corrections = 0

    def __init__(self, chip_id: str, site_mm: tuple[float, float]) -> None:
        self.chip_id, self.site = chip_id, tuple(site_mm)
        self.name = f"refind_electrode_site_{chip_id}"

    def _reference(self, ctx):
        chip = require_sample(ctx, self.chip_id, "chip")
        ref = chip["props"].get("user_reference_20x") or {}
        if not ref.get("image") or not Path(ref["image"]).is_file():
            raise PreconditionFailed(f"{self.chip_id} has no user 20x photo of its electrode site")
        if tuple(ref.get("site_mm") or ()) != self.site:
            raise PreconditionFailed(f"{self.chip_id} user photo shows another site than the design's")
        return ref

    def expected(self, ctx):
        return {"site_mm": list(self.site), "max_shift_um": ctx.verification["electrode_refind"]["max_shift_um"]}

    def preconditions(self, ctx):
        if not require_sample(ctx, self.chip_id, "chip")["props"].get("corner_stage_xy"):
            raise PreconditionFailed(f"{self.chip_id} has no corner position (find it first)")
        self._reference(ctx)

    def act(self, ctx, correction):
        objective = self._reference(ctx)["objective"]
        ctx.gate.microscope(MICROSCOPE).set_objective(objective)
        _move_stage(ctx, *GoToTarget(self.chip_id, self.site)._xy(ctx))
        autofocus(ctx, objective)

    def verify(self, ctx):
        import cv2
        import numpy as np

        from qumus_simlab.perception.v1_vision import refind_template

        exp, ref = self.expected(ctx), self._reference(ctx)
        frame = microscope_frame(ctx, phase=self.name)
        if frame.meta.get("objective") != ref["objective"]:
            raise NeedsHumanError("The electrode photo and the user photo are not at the same objective")
        calibration = _image_calibration(frame)
        pose = ctx.gate.stage(STAGE).get_sample_pose()
        corner = tuple(require_sample(ctx, self.chip_id, "chip")["props"]["corner_stage_xy"])
        rotation = _chip_rotation(ctx, self.chip_id)
        current = np.asarray(load_pixels(frame), dtype=np.uint8)
        reference = cv2.cvtColor(cv2.imread(ref["image"]), cv2.COLOR_BGR2RGB)
        match = refind_template(reference, current)
        height, width = current.shape[:2]
        mx, my = match.centre_px
        seen = (pose.x-(mx-width/2)*calibration.mm_per_pixel_x, pose.y-(my-height/2)*calibration.mm_per_pixel_y)
        planned = _local_to_stage(corner, rotation, *self.site)
        shift_mm = (seen[0]-planned[0], seen[1]-planned[1])
        shift_um = math.hypot(*shift_mm)*1000
        ok = shift_um <= exp["max_shift_um"]
        corrected = [round(corner[0]+shift_mm[0], 6), round(corner[1]+shift_mm[1], 6)]
        path = frame.saved_path()
        m = {"matched": ok, "score": round(float(match.score), 4), "angle_deg": float(match.angle_deg),
             "match_center_px": [round(float(mx), 2), round(float(my), 2)],
             "shift_um": [round(1000*v, 3) for v in shift_mm], "stage_xy": [pose.x, pose.y],
             "chip_rotation_deg": rotation, "corner_stage_xy_before": list(corner),
             "corner_stage_xy_after": corrected if ok else list(corner),
             "reference_image": ref["image"], "image": path,
             "scope": "v1 centre-patch match to the user's 20x site photo; translation only, angle discarded as in v1"}
        ctx.store.emit("stack.electrode_refind", {"chip_id": self.chip_id, **m}, run_id=ctx.run_id)
        if not ok:
            return Observation(Verdict.UNCLEAR, m, exp, images=[path] if path else [],
                message=f"The user photo matched {shift_um:.1f} um from where the corners put the "
                        "site; the target frame is unchanged")
        return Observation(Verdict.OK, m, exp, images=[path] if path else [])

    def correct(self, ctx, obs):
        return None

    def on_success(self, ctx, obs):
        if obs is None or not obs.measurements.get("matched"):
            raise NeedsHumanError("Electrode refind lacks its verified observation")
        m = obs.measurements
        chip = require_sample(ctx, self.chip_id, "chip")
        if list(chip["props"]["corner_stage_xy"]) != m["corner_stage_xy_before"]:
            raise NeedsHumanError("Target corner changed between electrode capture and correction")
        ctx.store.upsert_sample(self.chip_id, "chip", props={"corner_stage_xy": m["corner_stage_xy_after"],
            "electrode_refind": {k: m[k] for k in ("shift_um", "score", "angle_deg", "corner_stage_xy_before",
                                                   "corner_stage_xy_after", "reference_image", "image")}})

    def probe_state(self, ctx):
        return ProbeResult.NOT_DONE


def _rotation_observation(store, run_id, name, captured_at):
    observations = [e for e in _transfer_events(store, run_id, "step.verified")
        if e.payload.get("name") == name and e.payload.get("verdict") == "ok" and e.ts <= captured_at]
    if not observations:
        raise ValueError("Original/current optical chip orientation is unavailable before the specimen capture")
    event = observations[-1]
    angle = event.payload.get("measurements", {}).get("rotation_deg")
    if not _finite(angle):
        raise ValueError("Optical chip orientation must be finite")
    return {"event_id": event.id, "event_sha256": _digest(event.payload),
            "run_id": run_id, "step_name": name, "rotation_deg": angle}


def _validate_design_source_orientation(store, run_id, flake_id, identity, verification):
    flake = store.get_sample(flake_id)
    original = flake["props"]["tracking_references"]["20x"]
    chip = identity["source_chip_id"]
    body = identity["orientation"]
    expected_original = _frame_record(Frame(MICROSCOPE, original["timestamp"], path=original["image"], meta=original["meta"]))
    if identity["original"] != expected_original:
        raise ValueError("Nominal orientation does not identify the selected original native20 exposure")
    bindings = ((body["original"], original["job_id"], "measure_chip_rotation", identity["original"]["captured_at"]),
                (body["current"], run_id, f"measure_rotation_{chip}", identity["current"]["captured_at"]))
    for binding, expected_run, expected_name, captured_at in bindings:
        event = _historical_event(store, binding, expected_run, "step.verified")
        if (binding["run_id"] != expected_run or binding["step_name"] != expected_name
                or event.payload.get("name") != expected_name or event.payload.get("verdict") != "ok"
                or event.payload["measurements"].get("rotation_deg") != binding["rotation_deg"]
                or not _finite(binding["rotation_deg"]) or event.ts > captured_at):
            raise ValueError("Specimen nominal orientation lost its immutable observed rotation")
    nominal = (body["current"]["rotation_deg"]-body["original"]["rotation_deg"]+180)%360-180
    calibration = _image_calibration(_reference(identity["original"]))
    c = verification["vision_control"]
    limits = {"max_translation_px": c["max_translation_px"], "residual_rotation_deg": c["residual_rotation_deg"],
              "min_quality": c["min_match_quality"], "min_ambiguity": c["min_match_ambiguity"]}
    if (body["nominal_rotation_deg"] != nominal or body["limits"] != limits
            or body["pixel_aspect_ratio"] != abs(calibration.mm_per_pixel_y/calibration.mm_per_pixel_x)):
        raise ValueError("Observed orientation projection or residual registration bounds changed")
    return body


class RefindFlake(Step):
    """Put a catalogued flake back under the objective from its chip-local
    position and the freshly found corner, and centre it (v1 rough + accurate
    refind). The flake must look like the ledger says (area within 2x)."""

    devices = (STAGE, MICROSCOPE)

    def __init__(self, flake_id: str, chip_id: str, material: str,
                 name: str | None = None, *, design_shift_mm: tuple[float, float] = (0., 0.)) -> None:
        self.flake_id, self.chip_id, self.material = flake_id, chip_id, material
        self.design_shift_mm = design_shift_mm
        self.name = name or f"refind_{flake_id}"

    def _cfg(self, ctx: FlowContext) -> dict[str, Any]:
        return ctx.verification["refind_flake"]

    def nominal(self, ctx: FlowContext) -> tuple[float, float]:
        chip = require_sample(ctx, self.chip_id, "chip")
        flake = require_sample(ctx, self.flake_id, "flake")
        corner = chip["props"]["corner_stage_xy"]
        rot = _chip_rotation(ctx, self.chip_id)
        lx, ly = flake["props"]["rel_corner_mm"]
        return _local_to_stage((corner[0], corner[1]), rot, lx, ly)

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        c = self._cfg(ctx)
        return {"stage_xy": [round(v, 4) for v in self.nominal(ctx)],
                "search_radius_um": c["search_radius_um"],
                "centre_tolerance_um": c["centre_tolerance_um"]}

    def preconditions(self, ctx: FlowContext) -> None:
        self.max_corrections = self._cfg(ctx)["max_corrections"]
        flake = require_sample(ctx, self.flake_id, "flake")
        if not flake["props"].get("rel_corner_mm"):
            raise PreconditionFailed(f"{self.flake_id} has no position on its chip")
        if not (ctx.store.get_sample(self.chip_id) or {"props": {}})["props"].get(
                "corner_stage_xy"):
            raise PreconditionFailed(f"{self.chip_id} has no corner position (find it first)")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        obj = self._cfg(ctx)["objective"]
        x, y = correction.params["xy"] if correction else self.nominal(ctx)
        ctx.gate.microscope(MICROSCOPE).set_objective(obj)
        _move_stage(ctx, x, y)
        if correction is None:
            autofocus(ctx, obj)

    def _nearest(self, ctx: FlowContext, images: list[str] | None = None) -> Any:
        c = self._cfg(ctx)
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        self._observed_frame = frame
        if images is not None and (path := frame.saved_path()):
            images.append(path)
        dets = [d for d in _centred_flakes(ctx, frame, self.material)
                if math.hypot(d.dx_mm, d.dy_mm) * 1000 <= c["search_radius_um"]]
        return min(dets, key=lambda d: math.hypot(d.dx_mm, d.dy_mm)) if dets else None

    def _source_identity(self, ctx, current):
        flake = require_sample(ctx, self.flake_id, "flake")
        original = flake["props"].get("tracking_references", {}).get("20x") or {}
        geometry = flake["props"].get("optical_geometry") or {}
        if (not original.get("image") or not original.get("polygon_px")
                or original.get("image") != flake["props"].get("images", {}).get("20x")
                or not original.get("job_id") or original["job_id"] != geometry.get("job_id")
                or geometry.get("source_chip_id") != self.chip_id):
            raise NeedsHumanError("Selected source needs its original captured native20 identity/provenance")
        reference = Frame(MICROSCOPE, original["timestamp"], path=original["image"], meta=original["meta"])
        _image_calibration(reference)
        _image_calibration(current)
        if any(reference.meta.get(k) != current.meta.get(k)
               for k in ("objective", "calibration", "focus_plane", "fov_mm")):
            raise NeedsHumanError("Selected source capture optical profile/plane changed")
        c = ctx.verification["vision_control"]
        orientation = None
        nominal, aspect = 0., 1.
        if ctx.params.get("device_design"):
            try:
                first = _rotation_observation(ctx.store, original["job_id"], "measure_chip_rotation", original["timestamp"])
                now = _rotation_observation(ctx.store, ctx.run_id, f"measure_rotation_{self.chip_id}", current.timestamp)
                if now["rotation_deg"] != _chip_rotation(ctx, self.chip_id):
                    raise ValueError("Current optical orientation differs from the observed ledger rotation")
            except (KeyError, TypeError, ValueError) as exc:
                raise NeedsHumanError("Selected source needs original and current observed chip orientations") from exc
            nominal = (now["rotation_deg"]-first["rotation_deg"]+180)%360-180
            calibration = _image_calibration(reference)
            aspect = abs(calibration.mm_per_pixel_y/calibration.mm_per_pixel_x)
            orientation = {"original": first, "current": now, "nominal_rotation_deg": nominal,
                "pixel_aspect_ratio": aspect, "limits": {"max_translation_px": c["max_translation_px"],
                    "residual_rotation_deg": c["residual_rotation_deg"], "min_quality": c["min_match_quality"],
                    "min_ambiguity": c["min_match_ambiguity"]}}
        result = register_specimen(reference, current, polygon_px=original["polygon_px"],
            config=RegistrationConfig(max_translation_px=c["max_translation_px"],
                max_rotation_deg=c["residual_rotation_deg"], nominal_rotation_deg=nominal,
                pixel_aspect_ratio=aspect, min_quality=c["min_match_quality"],
                min_ambiguity=c["min_match_ambiguity"]))
        if not result.ok:
            raise NeedsHumanError(f"Selected source pixels are not identifiable: {result.loss_reason}")
        identity = {"original": _frame_record(reference), "current": _frame_record(current),
                "quality": result.quality, "uncertainty_px": result.uncertainty_px,
                "source_id": self.flake_id, "source_chip_id": self.chip_id}
        if orientation:
            identity["orientation"] = orientation
            _validate_design_source_orientation(ctx.store, ctx.run_id, self.flake_id, identity, ctx.verification)
        return identity

    def verify(self, ctx: FlowContext) -> Observation:
        exp, imgs = self.expected(ctx), []
        d = self._nearest(ctx, imgs) or self._nearest(ctx)  # one more look before "lost"
        s = ctx.gate.stage(STAGE).get_sample_pose()
        if d is None:
            return Observation(Verdict.UNCLEAR, {"found": False, "stage_xy": [s.x, s.y]}, exp,
                               images=imgs, message=f"{self.flake_id} not found within "
                               f"{exp['search_radius_um']} um of where the ledger puts it")
        area = (ctx.store.get_sample(self.flake_id) or {"props": {}})["props"].get("area_um2")
        off_um = math.hypot(d.dx_mm, d.dy_mm) * 1000
        m = {"found": True, "offset_um": round(off_um, 2), "area_um2": d.area_um2,
             "stage_xy": [round(s.x - d.dx_mm, 4), round(s.y - d.dy_mm, 4)]}
        if area and not 0.5 <= d.area_um2 / area <= 2.0:
            return Observation(Verdict.UNCLEAR, m, exp, images=imgs,
                               message=f"the flake there has {d.area_um2:.0f} um2, the ledger "
                                       f"says {area:.0f} um2 (a different flake?)")
        m["source_identity"] = self._source_identity(ctx, self._observed_frame)
        if off_um <= exp["centre_tolerance_um"]:
            return Observation(Verdict.OK, m, exp, images=imgs)
        return Observation(Verdict.CORRECTABLE, m, exp, images=imgs,
                           message=f"flake {off_um:.1f} um off centre")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return Correction("re-centre on the flake", {"xy": obs.measurements["stage_xy"]})

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        if obs is not None:
            # The refound stage position on this placement of the chip. The
            # search-time stage_xy stays: it is bound to the original 20X
            # source anchor that pickup admission re-checks (seed 3 r7).
            ctx.store.upsert_sample(self.flake_id, "flake",
                                    props={"transfer_stage_xy": obs.measurements["stage_xy"]})
            dx, dy = self.design_shift_mm
            if any(not math.isfinite(v) for v in (dx, dy)):
                raise PreconditionFailed("Design shift must be finite chip-local millimetres")
            if dx or dy:
                x, y = obs.measurements["stage_xy"]
                angle = math.radians(_chip_rotation(ctx, self.chip_id))
                _move_stage(ctx, x+dx*math.cos(angle)-dy*math.sin(angle),
                            y+dx*math.sin(angle)+dy*math.cos(angle))
            frame, reference = _capture_chip_focus(ctx, self.name+"_source_reference", parked=True)
            self._source_identity(ctx, frame)
            reference.update(source_id=self.flake_id, source_chip_id=self.chip_id)
            if ctx.params.get("device_design"):
                reference["design_source_rotation_deg"] = _chip_rotation(ctx, self.chip_id)
            state = _transfer_state(ctx)
            _reference(obs.measurements["source_identity"]["original"])
            reference["source_identity"] = {**obs.measurements["source_identity"],
                "design_shift_chip_mm": [dx, dy], "post_shift_capture": _frame_record(frame)}
            angle = math.radians(_chip_rotation(ctx, self.chip_id))
            expected = (dx*math.cos(angle)-dy*math.sin(angle),
                        dx*math.sin(angle)+dy*math.cos(angle))
            detections = _centred_flakes(ctx, frame, self.material)
            chosen = min(detections, key=lambda d: math.hypot(d.dx_mm-expected[0], d.dy_mm-expected[1])) if detections else None
            if chosen is None or math.hypot(chosen.dx_mm-expected[0], chosen.dy_mm-expected[1]) > .012:
                raise NeedsHumanError("The selected shifted source is not observable in its reference capture")
            reference["target_offset_mm"] = [chosen.dx_mm, chosen.dy_mm]
            state.setdefault("source_references", {})[self.flake_id] = reference
            _save_transfer(ctx, state)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


class GoToTarget(Step):
    """Put the drop-down target (chip-local mm from the BL corner) under the
    objective and focus. Aligning to the electrode pattern itself needs image
    registration against the design (stage 3)."""

    name = "go_to_target"
    devices = (STAGE, MICROSCOPE)

    def __init__(self, chip_id: str, target_xy_mm: tuple[float, float]) -> None:
        self.chip_id, self.target = chip_id, target_xy_mm

    def _xy(self, ctx: FlowContext) -> tuple[float, float]:
        chip = require_sample(ctx, self.chip_id, "chip")
        corner = chip["props"]["corner_stage_xy"]
        rot = _chip_rotation(ctx, self.chip_id)
        return _local_to_stage((corner[0], corner[1]), rot, *self.target)

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        obj = ctx.verification["refind_flake"]["objective"]
        return {"stage_xy": [round(v, 4) for v in self._xy(ctx)], "on_chip": self.chip_id,
                "min_sharpness": min_sharpness(ctx, obj)}

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        obj = ctx.verification["refind_flake"]["objective"]
        ctx.gate.microscope(MICROSCOPE).set_objective(obj)
        _move_stage(ctx, *self._xy(ctx))
        autofocus(ctx, obj)

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        sharp = analyzer(ctx).sharpness(frame)
        m = {"on_chip": analyzer(ctx).chip_in_view(frame), "sharpness": round(sharp, 2)}
        path = frame.saved_path()
        if not m["on_chip"]:
            return Observation(Verdict.UNCLEAR, m, exp, images=[path] if path else [], message="the target is not on a chip")
        if sharp < exp["min_sharpness"]:
            return Observation(Verdict.CORRECTABLE, m, exp, images=[path] if path else [], message="target out of focus")
        materials = sorted(set(ctx.params["materials"][fid] for fid in ctx.params["flake_ids"]))
        before = _target_view(ctx, self.chip_id, self.target, materials,
                              self.name+"_before_release", parked=True)
        state = _transfer_state(ctx)
        state["target_reference"] = before
        state.pop("target_reference_verified", None)
        _save_transfer(ctx, state)
        return Observation(Verdict.OK, {**m, "target_reference_sha256": _digest(before)}, exp,
                           images=[f["image"] for f in before["frames"]])

    def on_success(self, ctx, obs):
        state = _transfer_state(ctx)
        before = state["target_reference"]
        if obs is None or obs.measurements.get("target_reference_sha256") != _digest(before):
            raise NeedsHumanError("Target baseline lacks its actual verified observation")
        state["target_reference_verified"] = _bind_verified_endpoint(ctx, self.name, obs,
            before["focus_reference"]["focus_binding"]["event_id"])
        _save_transfer(ctx, state)

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return Correction("refocus")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


# ----- pick-up and drop-down -------------------------------------------------

#: detections closer than this are the same flake (position noise is < 1 um at 20x)
SAME_FLAKE_MM = 0.003


def _require_source_plane(ctx, frame, metadata):
    measure = getattr(analyzer(ctx), "plane_sharpness", None)
    quality = measure(frame, "chip") if callable(measure) else None
    objective = ctx.verification["refind_flake"]["objective"]
    if (isinstance(quality, bool) or not isinstance(quality, (int, float)) or not math.isfinite(quality)
            or quality < min_sharpness(ctx, objective) or frame.meta.get("focus_plane") != "chip"
            or frame.meta.get("objective") != objective or frame.meta.get("calibration") != metadata["calibration"]):
        raise NeedsHumanError("Source absence is unobservable: source-plane focus/quality binding is unknown")
    _held_source(ctx, frame)  # Capture-bound public kinematics; active motion or changed stage cannot prove absence.


def _flat_sweep_absence_plane(evidence: dict[str, Any], source_reference: dict[str, Any],
                              restored_z: float, objective: int, gain: float) -> dict[str, Any] | None:
    """A textureless source spot after pickup, observed at a known plane.

    Bare SiO2 has no optical texture, so no focus peak exists once the flake
    has left. That is still an in-focus look when (a) the sweep brackets the
    restored pre-pickup source plane, (b) the whole sweep is one plateau in
    autofocus's own sense (every sample within `minimum_peak_gain` of the
    best; usually all exactly zero), and (c) the recorded pre-pickup
    reference at that plane scored clearly above that plateau (the flake
    itself). A flake still present would raise a peak inside the sweep.
    Anything else returns None and the caller parks.
    """
    from qumus_simlab.perception.tracking import raster_focus_score

    passes = evidence.get("sampled_passes")
    if (evidence.get("plane") != "chip" or evidence.get("objective") != objective
            or evidence.get("focus_z") != restored_z or not isinstance(passes, list) or not passes
            or not (_finite(gain) and 0 < gain < 1)):
        return None
    zs: list[float] = []
    scores: list[float] = []
    for record in passes:
        for sample in (record.get("samples") or []) if isinstance(record, dict) else []:
            z, score = sample.get("readback_focus_z_mm"), sample.get("raster_score")
            if not (_finite(z) and _finite(score) and score >= 0):
                return None
            zs.append(z)
            scores.append(score)
    if len(zs) < 3 or not min(zs) < restored_z < max(zs):
        return None
    flat_score = max(scores)
    if min(scores) < flat_score * (1 - gain):
        return None
    color = (evidence.get("focus_metric") or {}).get("id") == "raster-focus-rgb-v1"
    try:
        reference_score = raster_focus_score(_reference(source_reference), color=color)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not (_finite(reference_score) and reference_score * (1 - gain) > flat_score):
        return None
    return {"method": "flat_sweep_brackets_textured_reference", "objective": objective,
            "restored_focus_z_mm": restored_z, "sweep_focus_z_mm": [min(zs), max(zs)],
            "samples": len(zs), "flat_raster_score": flat_score,
            "reference_sha256": source_reference.get("sha256"),
            "reference_raster_score": reference_score, "verification": False}


def _source_absence_focus_ok(af: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Re-derived focus proof of a source-absence look: an ordinary bracketed
    peak, or a flat sweep around a textured recorded reference (#109)."""
    from qumus_simlab.config import load_yaml

    if af.get("peak_observable") is True:
        return True
    gain = load_yaml("verification.yaml")["focus"]["minimum_peak_gain"]
    return (af.get("method") == "retained_previous_plane"
            and _flat_sweep_absence_plane(af, reference, af.get("focus_z"), 20, gain) is not None)


def _chip_near_centre(ctx: FlowContext, material: str, radius_mm: float,
                      images: list[str] | None = None, looks: int = 2,
                      frames: list[Frame] | None = None,
                      source_reference: dict | None = None) -> list[Any]:
    """Flakes of `material` on the chip near the view centre; several looks so a
    single missed detection does not read as "gone". Detections of different
    looks closer than SAME_FLAKE_MM are one flake (by position only, as on real
    hardware): two layers of one material that close count as one, which errs
    towards "a layer is missing" (the run parks), never towards a false OK."""
    scope = ctx.gate.microscope(MICROSCOPE)
    objective = ctx.verification["refind_flake"]["objective"]
    if source_reference is None or not isinstance(source_reference.get("target_offset_mm"), list) or len(source_reference["target_offset_mm"]) != 2 or not all(map(_finite, source_reference["target_offset_mm"])):
        raise NeedsHumanError("Source absence lacks its recorded selected-source location")
    held = (_pos(ctx), ctx.gate.stage(STAGE).kinematics())
    metadata = source_reference.get("meta")
    if (not isinstance(metadata, dict) or metadata.get("objective") != objective
            or metadata.get("focus_plane") != "chip" or not metadata.get("calibration")
            or not _finite(metadata.get("focus_z"))):
        raise NeedsHumanError("Source absence lacks its known native source-plane binding")
    _reference(source_reference)
    metadata = deepcopy(metadata)
    if not (ctx.params.get("device_design") and not _over_stage(ctx)):
        metadata["focus_z"] += _profile(ctx)["focus_offset_mm"]
    _restore_source_plane(ctx, metadata)
    autofocus(ctx, objective, attempts=1,
              half_range=_profile(ctx)["post_phase_focus_half_range_mm"], plane="chip")
    evidence = ctx.store.get_run(ctx.run_id)["meta"].get("last_focus_observation") or {}
    if not evidence.get("peak_observable"):
        flat = _flat_sweep_absence_plane(evidence, source_reference, metadata["focus_z"], objective,
                                         ctx.verification["focus"]["minimum_peak_gain"])
        if flat is None or scope.get_focus() != metadata["focus_z"]:
            raise NeedsHumanError("Source endpoint absence has no observable ordinary focus peak")
        ctx.store.emit("stack.source_absence_flat_sweep", flat, run_id=ctx.run_id)
    metadata["focus_z"] = scope.get_focus()
    found: list[Any] = []
    for _ in range(looks):
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        _require_source_plane(ctx, frame, metadata)
        if (frame.meta.get("focus_z") != scope.get_focus()
                or held != (_pos(ctx), ctx.gate.stage(STAGE).kinematics())):
            raise NeedsHumanError("Source absence is unobservable: source-plane focus/quality or held context changed")
        if frames is not None:
            frames.append(frame)
        if images is not None and not images and (path := frame.saved_path()):
            images.append(path)
        for d in _centred_flakes(ctx, frame, material):
            centre = source_reference["target_offset_mm"]
            if math.hypot(d.dx_mm-centre[0], d.dy_mm-centre[1]) > radius_mm:
                continue
            if not any(math.hypot(d.dx_mm - o.dx_mm, d.dy_mm - o.dy_mm) < SAME_FLAKE_MM
                       for o in found):
                found.append(d)
    return found


def pickup_press_names(fid: str, represses: int) -> list[str]:
    """Every planned pickup press of `fid`, first to last (#109). The last one
    is the press whose verified endpoint commits the transfer."""
    return [f"v1_pick_up_{fid}", *(f"v1_pick_up_{fid}_press{k}" for k in range(2, represses+2))]


def pickup_name(params: Any, fid: str) -> str:
    """The committing pickup press of `fid` under the run's (replanned) params."""
    represses = params.get("pickup_represses") if isinstance(params, dict) else params.pickup_represses
    return pickup_press_names(fid, int((represses or {}).get(fid, 0)))[-1]


class PickUpFlake(Step):
    """Z-only difference coverage, with separate optical endpoint verification."""
    devices = (MANIP, MICROSCOPE, HEATER)
    max_corrections = 0
    safe_to_resume = False
    pause_inside = True

    def __init__(self, flake_id: str, stamp_id: str, material: str, name: str | None = None):
        self.flake_id, self.stamp_id, self.material = flake_id, stamp_id, material
        self.name = name or f"v1_pick_up_{flake_id}"
        self._covered = False

    def preconditions(self, ctx):
        if ctx.params.get("stamp_discovery_policy"):
            _validate_discovery_ready(ctx.store, ctx.run_id)
        if ctx.params.get("device_design"):
            _device_design_public_binding(ctx.params, ctx.store, ctx.store.get_run(ctx.run_id))
        if ctx.params.get("direct_contact_contract"):
            try:
                _direct_contact_public_binding(ctx.params, ctx.store, ctx.store.get_run(ctx.run_id), stage="pickup")
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise NeedsHumanError("Frozen direct-contact source/receiver binding changed before pickup") from exc
        _profile(ctx)
        placement = _transfer_state(ctx).get("placements", {}).get(self.stamp_id)
        if not placement or not _over_stage(ctx):
            raise PreconditionFailed("Pickup needs the stamp's first-contact placement and engaged holder")
        if ctx.params.get("device_design") and self.flake_id != ctx.params["flake_ids"][0]:
            _observe_design_pickup_coverage(ctx, self.flake_id, self.stamp_id)

    def expected(self, ctx):
        return {"coverage_margin_mm": ctx.verification["transfer_sequence"]["pickup_margin_mm"],
                "flake_on_chip": False, "method": "source_disappearance", "coverage_is_transfer_proof": False}

    def act(self, ctx, correction):
        c = _profile(ctx)
        admissions = list(_transfer_events(ctx.store, ctx.run_id, "stack.pickup_macro_admission"))
        saved = _transfer_state(ctx).get("phases", {}).get(self.name, {})
        if "final_partial_macro" in saved or any(e.payload.get("phase") == self.name for e in admissions):
            if not saved.get("complete"):
                raise NeedsHumanError("Reserved final pickup prefix cannot be replayed")
            _pickup_partial_receipt(ctx.store, ctx.run_id, saved, admissions)
        try:
            process = _process_reference(_transfer_state(ctx), self.flake_id, self.stamp_id,
                                         ctx.run_id, ctx.verification["pickup"]["max_press_mm"], ctx.store)
        except (KeyError, TypeError, ValueError) as exc:
            raise NeedsHumanError("Pickup lacks its admitted current-layer process reference") from exc
        if not _finite(process.get("minimum_z_mm")):
            raise NeedsHumanError("Pickup process minimum is non-finite")
        phase = _begin_phase(ctx, self.name)
        if phase.get("process_reference", process) != process:
            raise NeedsHumanError("Saved current-layer process reference changed")
        phase["process_reference"] = process
        _save_phase(ctx, phase)
        if phase.get("complete"):
            self._covered = phase.get("coverage", {}).get("covered", False)
            return
        if phase.get("observations", 0) > 1:
            raise NeedsHumanError("Interrupted pickup requires endpoint review; no blind re-press")
        if ctx.params.get("device_design"):
            _contact_reference_context(ctx, phase, _transfer_state(ctx)["source_references"][self.flake_id])
        phase["safe_z_mm"] = _pos(ctx)["z"]+.2
        _save_phase(ctx, phase)
        reference = _reference(phase["reference"])
        calibration = _image_calibration(reference)
        pixels = load_pixels(reference)
        ox, oy = c["pickup_target_offset_image_mm"]
        target_px = (pixels.width/2+ox/abs(calibration.mm_per_pixel_x),
                     pixels.height/2+oy/abs(calibration.mm_per_pixel_y))
        def covered(frame):
            evidence = {"reference": phase["reference"], "current": _frame_record(frame),
                        "actual_holder_pose": _pos(ctx), "transfer_confirmed": False}
            # v1 stop rule for every flow (#204): the fixed target point lies
            # more than 240 historical px (0.1752 mm) inside the largest
            # changed region against the fixed reference.
            result = v1_pickup_coverage(reference, frame, calibration, target_px,
                margin_mm=c["pickup_margin_mm"], min_area_mm2=c["minimum_area_mm2"],
                blur_support_mm=c["pickup_blur_support_mm"])
            phase["coverage"] = {**result, **evidence}
            # v1 has no film-tear check during the pickup press (#204).
            _save_phase(ctx, phase)
            return result["covered"]
        while not self._covered:
            start = _pos(ctx)["z"]
            if not _finite(start):
                raise NeedsHumanError("Pickup actual Z readback is non-finite")
            # A process reference without its expected contact keeps the
            # touching speed throughout.
            speed, target = (_approach_speed(c, start, start-c["pickup_intent_mm"],
                process["reference_interval_mm"][1], ctx.verification["manipulator"]["tolerance_mm"])
                if "reference_interval_mm" in process else (c["descent_speed_mm_s"], start-c["pickup_intent_mm"]))
            admission = None
            if target < process["minimum_z_mm"]:
                lim = ctx.gate.limits.manipulators[MANIP]
                floor = ctx.gate.manipulator_z_floor(MANIP)
                bounds = [max(lim.z[0], floor or lim.z[0]), lim.z[1]]
                tol = ctx.verification["manipulator"]["tolerance_mm"]
                minimum = process["minimum_z_mm"]
                if (not all(_finite(v) for v in (start, target, minimum, tol)) or tol <= 0
                        or not bounds[0] <= target <= bounds[1]):
                    raise NeedsHumanError("Original pickup intent is outside its current floor/envelope; no clamp applied")
                admitted = math.nextafter(minimum+tol, math.inf)
                if (not bounds[0] <= admitted < start <= bounds[1] or admitted-tol < minimum
                        or start-admitted <= 1e-8):
                    raise NeedsHumanError("No safe final pickup prefix remains within the process budget")
                captures = [e for e in _transfer_events(ctx.store, ctx.run_id, "lab.frame",
                    phase["reference_event"]["event_id"]-1)
                    if e.payload.get("microscope_preview", {}).get("phase") == self.name
                    and e.payload["microscope_preview"].get("image") == phase["last_frame"]["image"]
                    and e.payload["microscope_preview"].get("timestamp") == phase["last_frame"]["captured_at"]]
                if len(captures) != 1:
                    raise NeedsHumanError("Final pickup prefix lacks its actual preceding capture")
                body = {"phase": self.name, "profile": c["id"], "rule": "final-process-budget-prefix-v1",
                    "from_readback_z_mm": start, "requested_delta_mm": -c["pickup_intent_mm"],
                    "requested_target_mm": target, "admitted_delta_mm": admitted-start,
                    "admitted_target_mm": admitted, "minimum_z_mm": minimum,
                    "readback_tolerance_mm": tol, "global_z_interval_mm": bounds,
                    "process_reference_sha256": _digest(process), "reference_sha256": phase["reference"]["sha256"],
                    "admission_observation_count": phase["observations"], "admitted_at": ctx.clock.now(),
                    "capture_event": {"event_id": captures[0].id, "event_sha256": _digest(captures[0].payload)}}
                with ctx.store._transaction():
                    if ("final_partial_macro" in _transfer_state(ctx)["phases"][self.name]
                            or any(e.payload.get("phase") == self.name for e in
                                _transfer_events(ctx.store, ctx.run_id, "stack.pickup_macro_admission"))):
                        raise NeedsHumanError("Final pickup prefix is already consumed")
                    event_id = ctx.store.emit("stack.pickup_macro_admission", body, run_id=ctx.run_id)
                    admission = {"admission": body, "event_id": event_id,
                                 "event_sha256": _digest(body), "status": "consumed"}
                    phase["final_partial_macro"] = admission
                    _save_phase(ctx, phase)
                target = admitted
            try:
                reached = _guarded_z(ctx, phase, target, speed=speed, observe=covered,
                                     macro_admission=admission)
            except Exception as exc:
                if admission is not None:
                    admission["status"] = "stopped" if isinstance(exc, StopRequested) else "failed"
                    phase["held_reason"] = "pickup_final_partial_interrupted"
                    _save_phase(ctx, phase)
                raise
            if admission is not None:
                admission["status"] = "covered" if phase.get("coverage", {}).get("covered") else "no_coverage"
                _save_phase(ctx, phase)
                if not reached:
                    phase["held_reason"] = "pickup_final_partial_exhausted"
                    _save_phase(ctx, phase)
                    raise NeedsHumanError("Coverage unavailable after the one final process-budget prefix")
            if reached:
                break
        self._covered = phase.get("coverage", {}).get("covered", False)
        phase["difference_active"] = False
        _save_phase(ctx, phase)
        if ctx.params.get("device_design") and self._covered:
            _design_contact_dwell(ctx, phase)
        if self._covered:
            autofocus(ctx, 5, attempts=1, half_range=c["post_phase_focus_half_range_mm"], plane="chip")
            _phase_capture(ctx, phase)
        _guarded_z(ctx, phase, phase["safe_z_mm"], speed=c["pickup_lift_speed_mm_s"])
        _guarded_z(ctx, phase, _ta(ctx)["safe_z"])
        if ctx.params.get("device_design"):
            park = ManipulatorTo("park", name=self.name+"_source_inspection_park", stamp_id=self.stamp_id)
            park.act(ctx, None)
            parked = park.verify(ctx)
            if parked.verdict != Verdict.OK:
                park.on_give_up(ctx, parked)
                raise NeedsHumanError("Source inspection requires confirmed lateral stamp clearance", observation=parked.to_dict())
            phase["source_inspection_park"] = {"actual_holder_pose": _pos(ctx), "completed_at": ctx.clock.now()}
            body = {"phase": self.name, "park": phase["source_inspection_park"]}
            event_id = ctx.store.emit("stack.design_source_park", body, run_id=ctx.run_id)
            phase["source_inspection_park_event"] = {"event_id": event_id, "event_sha256": _digest(body)}
        phase["complete"] = True
        _save_phase(ctx, phase)

    def verify(self, ctx):
        phase = _transfer_state(ctx).get("phases", {}).get(self.name, {})
        coverage = phase.get("coverage")
        if not phase.get("complete") or not coverage or not coverage.get("covered"):
            return Observation(Verdict.UNCLEAR, {"coverage": coverage, "transfer_confirmed": False},
                               self.expected(ctx), message="Pickup stroke/coverage incomplete")
        images, source_frames = [], []
        reference = _transfer_state(ctx).get("source_references", {}).get(self.flake_id)
        if reference is None:
            raise NeedsHumanError("Pickup endpoint lacks its captured native source reference")
        on_chip = _chip_near_centre(ctx, self.material, .06, images, frames=source_frames,
                                   source_reference=reference)
        source_evidence = [_frame_record(f) for f in source_frames]
        changed = any(frame["sha256"] != reference["sha256"] for frame in source_evidence)
        confirmed = not on_chip and bool(source_evidence) and changed
        measurements = {"profile": TRANSFER_PROFILE_ID, "source_id": self.flake_id,
            "source_reference_sha256": reference["sha256"], "coverage": coverage,
            "method": "source_disappearance", "flake_on_chip": bool(on_chip),
            "source_frames": source_evidence,
            "focus_event_id": ctx.store.get_run(ctx.run_id)["meta"].get("last_focus_event_id"),
            "source_detection_count": len(on_chip), "stamp_evaluated": False, "flake_on_stamp": None,
            "process_confirmation": confirmed,
            "transfer_confirmed": confirmed, "no_transfer_confirmed": None,
            "full_identity_completeness_observed": False, "registry_chemistry_observed": False,
            "scope": "Owner visual process criterion; source disappearance only, partial transfer/identity unverified"}
        if on_chip and not confirmed and ctx.params.get("device_design") and source_frames:
            # A retained source is registered against its original native20
            # identity: the evidence a re-press needs (#109). It never confirms
            # that nothing transferred.
            chip_id = ctx.params.get("flake_chips", {}).get(self.flake_id)
            try:
                measurements["source_presence"] = RefindFlake(self.flake_id, chip_id, self.material)._source_identity(
                    ctx, source_frames[-1])
            except NeedsHumanError as exc:
                measurements["source_presence_unverified"] = exc.message
        phase["endpoint"] = measurements
        _save_phase(ctx, phase)
        return Observation(Verdict.OK if measurements["process_confirmation"] else Verdict.UNCLEAR,
            measurements, self.expected(ctx), images=[f["image"] for f in measurements["source_frames"]],
            message="Source material disappeared at its qualified selected location" if measurements["process_confirmation"] else
            ("Source material remains; transfer unconfirmed and stamp state unknown. Partial transfer cannot be excluded. "
             + ("A re-press is planned only for a whole re-identified source." if ctx.params.get("device_design")
                else "No automatic extra press."))
            if on_chip else "Source images are unchanged; disappearance is unconfirmed. Stamp state unknown; no automatic extra press.")

    def on_success(self, ctx, obs):
        if (obs is None or not obs.measurements.get("transfer_confirmed")
                or not obs.measurements.get("process_confirmation")
                or obs.measurements.get("method") != "source_disappearance"
                or obs.measurements != _transfer_state(ctx).get("phases", {}).get(self.name, {}).get("endpoint")
                or not _visual_pickup_proof(obs.measurements, _transfer_state(ctx)["source_references"][self.flake_id], ctx.store, ctx.run_id)):
            raise NeedsHumanError("Coverage alone cannot commit a transfer")
        phase = _transfer_state(ctx)["phases"][self.name]
        if ctx.params.get("device_design"):
            _validate_design_pickup_additions(ctx, phase,
                _transfer_state(ctx)["source_references"][self.flake_id], obs.measurements)
        try:
            partial = _pickup_partial_receipt(ctx.store, ctx.run_id, phase,
                list(_transfer_events(ctx.store, ctx.run_id, "stack.pickup_macro_admission")))
            if partial is not None and (partial["status"] != "covered" or not phase.get("complete")):
                raise ValueError("Final pickup prefix did not complete with coverage")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise NeedsHumanError("Pickup prefix lacks its historical admission") from exc
        phase["endpoint_verified"] = _bind_verified_endpoint(ctx, self.name, obs, obs.measurements["focus_event_id"])
        _save_phase(ctx, phase)
        stamp = require_sample(ctx, self.stamp_id, "stamp")
        stack = list(stamp["props"].get("stack", []))
        if self.flake_id not in stack:
            stack.append(self.flake_id)
        with ctx.store._transaction():
            ctx.store.upsert_sample(self.stamp_id, "stamp", props={"stack": stack})
            ctx.store.upsert_sample(self.flake_id, "flake", location=self.stamp_id, status="on_stamp")
            ctx.store.emit("stack.pickup_endpoint", {"stamp_id": self.stamp_id, "flake_id": self.flake_id,
                "observation": obs.to_dict(), "procedure": PROCEDURE_ID}, run_id=ctx.run_id)

    def safe_state(self, ctx):
        ctx.gate.heater(HEATER).off()
        _clear_lift(ctx, self.name)

    def before_park(self, ctx):
        self.safe_state(ctx)


class ObserveTransferTarget(Step):
    """Observe a reported coating before release; never derive its shape from a label."""
    name = "observe_transfer_target"
    devices = (MICROSCOPE, STAGE)

    def __init__(self, chip_id: str):
        self.chip_id = chip_id
        self.geometry = None

    def act(self, ctx, correction):
        if not ctx.store.get_sample(self.chip_id)["props"].get("metal_coating"):
            return
        scope = ctx.gate.microscope(MICROSCOPE)
        previous = scope.get_objective()
        try:
            scope.set_objective(5)
            autofocus(ctx, 5)
            frame = scope.capture()
            perceive = getattr(analyzer(ctx), "metal_region", None)
            if perceive is None:
                raise PreconditionFailed("No calibrated metal-region observation adapter is available.")
            self.geometry = {**perceive(frame), "image": frame.saved_path(),
                             "job_id": ctx.run_id, "chip_id": self.chip_id}
        finally:
            scope.set_objective(previous)
            autofocus(ctx, previous)

    def verify(self, ctx):
        if not ctx.store.get_sample(self.chip_id)["props"].get("metal_coating"):
            return Observation(Verdict.OK, {"coating": "not present"})
        from qumus_simlab.core.polygons import EPS, convex_polygon, cross

        if not self.geometry or not self.geometry["image"]:
            return Observation(Verdict.UNCLEAR, message="No saved metal geometry evidence.")
        polygon = convex_polygon(self.geometry["outline_mm"])
        ok = all(cross(a, b, (0, 0)) >= -EPS for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True))
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR, self.geometry,
            images=[self.geometry["image"]], message="Target lies inside observed metal" if ok else "Target is outside observed metal; review placement.")

    def on_success(self, ctx, obs):
        if self.geometry:
            ctx.store.upsert_sample(self.chip_id, "chip", props={"metal_geometry": self.geometry})

    def probe_state(self, ctx):
        return ProbeResult.NOT_DONE


def _contact_classification_snapshot(store, design):
    """Freeze observed nanometre classes, never exact simulator layers."""
    from qumus_simlab.perception.contact_coverage import InteriorCalibration

    sources = [source for source in design.sources if source.material == "hBN"]
    if len(sources) != 1:
        raise ValueError("Contact calibration supports one declared carried hBN source")
    source = sources[0]
    flake = store.get_sample(source.flake_id)
    props = (flake or {}).get("props", {})
    geometry = props.get("optical_geometry") or {}
    if geometry.get("source") == "rgb_optical_outline":
        from qumus_simlab.records.recognition import (
            validate_recognition_chain,
            validate_rgb_capture_binding,
            validate_rgb_outline,
        )

        if not flake or source.observation != geometry or store.bench != "sim":
            raise ValueError("Carried v1 hBN differs from its frozen original20 source")
        # The source observation stays bound to its original chip after pickup.
        # Current material location is checked by the transfer lifecycle gates.
        original_flake = {**flake, "location": geometry["source_chip_id"]}
        validate_rgb_capture_binding(store, original_flake, geometry)
        validate_rgb_outline(geometry, material="hBN")
        image = geometry["image"]
        digest = hashlib.sha256(Path(image).read_bytes()).hexdigest()
        if digest != source.image_sha256:
            raise ValueError("Carried v1 hBN lost its frozen original20 image")
        identifiers = {20: geometry["recognition_event_id"],
                       100: props.get("classification_event_100x")}
        if (identifiers[100] is None and props.get("recognition_100x") is None
                and (props.get("classification_100x") or {}).get("status") == "unclassified"):
            # v1 used 100X only to classify; a frame it could not read leaves
            # the 20X label as the only nanometre classification.
            del identifiers[100]
        observed_labels, label_events = {}, {}
        completion = store._x("SELECT MIN(id) AS id FROM event WHERE run_id=? AND kind='job.status' "
            "AND json_extract(payload_json,'$.status')='succeeded'", (geometry["job_id"],)).fetchone()
        for objective, identifier in identifiers.items():
            row = store._x("SELECT kind,run_id,payload_json FROM event WHERE id=?", (identifier,)).fetchone()
            body = json.loads(row["payload_json"]) if row else {}
            kind = "microscope.rgb_recognition" if objective == 20 else "microscope.rgb_classification"
            chain = body.get("recognition_chain") or {}
            label = body.get("thickness_label")
            if (not row or row["kind"] != kind or row["run_id"] != geometry["job_id"]
                    or not completion or completion["id"] is None or type(identifier) is not int
                    or identifier >= completion["id"] or body.get("source_chip_id") != geometry["source_chip_id"]
                    or body.get("flake_id") != source.flake_id or body.get("material") != "hBN"
                    or not isinstance(label, str) or not label
                    or chain.get("image", {}).get("objective") != objective):
                raise ValueError("Carried v1 hBN lacks its original observed nanometre classification")
            validate_recognition_chain(chain, image_path=chain["image"]["path"], objective=objective)
            if objective == 100:
                capture_id = body.get("capture_event_id")
                capture = store._x("SELECT kind,run_id,payload_json FROM event WHERE id=?", (capture_id,)).fetchone()
                command = json.loads(capture["payload_json"]) if capture else {}
                if (type(capture_id) is not int or not identifiers[20] < capture_id < identifier
                        or not capture or capture["kind"] != "device.command"
                        or capture["run_id"] != geometry["job_id"]
                        or command.get("device") != MICROSCOPE or command.get("command") != "capture"
                        or body.get("geometry_use") != "classification_only"
                        or props.get("recognition_100x") != chain):
                    raise ValueError("Carried v1 hBN100 classification lost its original capture")
            observed_labels[f"{objective}x"] = label
            label_events[f"{objective}x"] = {"event_id": identifier, "payload_sha256": geometry_digest(body)}
        labels = sorted(set(observed_labels.values()))
        carried = InteriorCalibration(tuple(labels)).snapshot()
        return {"profile": carried["profile"], "empty": InteriorCalibration().snapshot(), "carried": carried,
            "source": {"flake_id": source.flake_id, "material": "hBN", "label_unit": "nm",
                "labels": labels, "native_labels": observed_labels, "classification_events": label_events,
                "image": image, "sha256": digest, "job_id": geometry["job_id"],
                "calibration": deepcopy(geometry["recognition_chain"]["calibration"]),
                "source_measurement_mode": "v1_original20_mask",
                "source_measurement_frame": geometry["frame"],
                "rendering_mode": "synthetic_complete_outline_flat_tint_v1"}}
    if geometry.get("source") == "rgb_native100_measured_region":
        from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE
        from qumus_simlab.records.recognition import native_source_observation

        # Contact is revalidated after pickup too. Rebuild immutable source
        # journal evidence; current material location belongs to lifecycle gates.
        expected = native_source_observation(store, geometry["native_source_anchor_event_id"])
        if expected != geometry or source.observation != geometry or store.bench != "sim":
            raise ValueError("Carried native hBN differs from its frozen observed source")
        row = store._x("SELECT kind,run_id,payload_json FROM event WHERE id=?",
                       (geometry["native_source_anchor_event_id"],)).fetchone()
        if not row or row["kind"] != "microscope.native_source_anchor" or row["run_id"] != geometry["job_id"]:
            raise ValueError("Carried native hBN lacks its original source-anchor event")
        anchor = json.loads(row["payload_json"])
        observed_labels, label_events = {}, {}
        for objective in (20, 100):
            identifier = anchor[f"measurement_event_{objective}x"]
            event = store._x("SELECT kind,run_id,payload_json FROM event WHERE id=?", (identifier,)).fetchone()
            body = json.loads(event["payload_json"]) if event else {}
            label = body.get("thickness_label")
            if (not event or event["kind"] != "microscope.rgb_measurement" or event["run_id"] != row["run_id"]
                    or body.get("objective") != objective or body.get("material") != "hBN"
                    or body.get("source_chip_id") != geometry["source_chip_id"]
                    or body.get("flake_id") != source.flake_id or not isinstance(label, str) or not label):
                raise ValueError("Carried native hBN lacks its original observed nanometre classification")
            observed_labels[f"{objective}x"] = label
            label_events[f"{objective}x"] = {"event_id": identifier, "payload_sha256": geometry_digest(body)}
        labels = sorted(set(observed_labels.values()))
        image = geometry["image"]
        digest = hashlib.sha256(Path(image).read_bytes()).hexdigest()
        chain = geometry["recognition_chain"]
        if (digest != source.image_sha256 or digest != chain["image"]["sha256"]
                or chain["image"]["objective"] != 100
                or chain["calibration"].get("material_capture_profile") != V1_MATERIAL_CAPTURE_PROFILE):
            raise ValueError("Carried native hBN classification lost its original100 RGB image/profile")
        carried = InteriorCalibration(tuple(labels)).snapshot()
        return {"profile": carried["profile"], "empty": InteriorCalibration().snapshot(), "carried": carried,
            "source": {"flake_id": source.flake_id, "material": "hBN", "label_unit": "nm",
                "labels": labels, "native_labels": observed_labels, "classification_events": label_events,
                "image": image, "sha256": digest, "job_id": geometry["job_id"],
                "calibration": deepcopy(chain["calibration"]),
                "source_measurement_mode": V1_MATERIAL_CAPTURE_PROFILE,
                "source_measurement_frame": geometry["frame"],
                "native_source_anchor_event_id": geometry["native_source_anchor_event_id"],
                # Source-chip recognition and carried-stamp projection have
                # distinct existing forward laws. No renderer/tint is changed.
                "rendering_mode": "synthetic_complete_outline_flat_tint_v1"}}
    labels = sorted(set([props.get("thickness_label"), *(props.get("labels") or {}).values()])-{None})
    references = props.get("tracking_references") or {}
    reference = references.get("100x") or {}
    if not reference.get("image") or reference.get("image") != geometry.get("image"):
        # Recognition baseline: complete native-20x geometry with the observed
        # 100x thickness class, as in the v1 RGB branch above. An hBN large
        # enough to cover the Gr rarely fits whole in the 100x field of view.
        native20 = references.get("20x") or {}
        if (native20.get("image") and native20.get("image") == geometry.get("image")
                and native20.get("job_id") == geometry.get("job_id")
                and (native20.get("meta") or {}).get("objective") == 20
                and (props.get("labels") or {}).get("100x")):
            reference = native20
    if (store.bench != "sim" or props.get("image_source") != "simulated"
            or geometry.get("source") != "simulated_optical_outline"
            or not reference.get("image") or not labels):
        raise ValueError("Carried hBN lacks supported public complete-outline optical evidence")
    digest = hashlib.sha256(Path(reference["image"]).read_bytes()).hexdigest()
    if digest != source.image_sha256:
        raise ValueError("Carried hBN classification lost its frozen native source image")
    carried = InteriorCalibration(tuple(labels)).snapshot()
    return {"profile": carried["profile"], "empty": InteriorCalibration().snapshot(), "carried": carried,
            "source": {"flake_id": source.flake_id, "material": "hBN", "label_unit": "nm",
                       "labels": labels, "native_labels": deepcopy(props.get("labels") or {}),
                       "image": reference["image"], "sha256": digest, "job_id": reference["job_id"],
                       "calibration": deepcopy(reference["meta"]["calibration"]),
                       "rendering_mode": "synthetic_complete_outline_flat_tint_v1"}}


def _contact_optical(raw, flake_id):
    from qumus_simlab.perception.contact_coverage import InteriorCalibration

    frozen = raw.get("contact_interior_calibration") or {}
    key = "empty" if raw["flake_ids"].index(flake_id) == 0 else "carried"
    body = frozen[key]
    optical = InteriorCalibration(tuple(body["labels_nm"]))
    if geometry_digest(optical.snapshot()) != geometry_digest(body):
        raise ValueError("Frozen contact optical profile/uncertainty policy changed")
    return optical


_CONTACT_ENCODER_PROFILE = "synthetic-contact-encoder-camera-v1"


def _contact_encoder_projection(event):
    """Public encoder/camera fields only; display/renderer packets are not evidence."""
    payload = event.payload
    geometry = payload.get("geometry", {})
    microscope = geometry.get("microscope", {})
    pose, camera = microscope.get("stamp_axes_xyz_mm"), microscope.get("optical_axis_world_mm")
    if (type(event.id) is not int or event.id <= 0 or not isinstance(event.run_id, str) or not event.run_id
            or event.kind != "lab.frame" or not _finite(event.ts)
            or any(not isinstance(payload.get(k), str) or not payload[k] for k in ("device", "command"))
            or geometry.get("source") != "synthetic" or geometry.get("mechanics_version") != 2
            or microscope.get("source") != "synthetic" or microscope.get("stamp_axes_frame") != MANIP
            or not isinstance(pose, dict) or any(not _finite(pose.get(k)) for k in ("x", "y", "z"))
            or not isinstance(camera, (list, tuple)) or len(camera) != 2 or not all(map(_finite, camera))):
        raise ValueError("Contact encoder/camera projection lacks finite declared public fields")
    return {"profile": _CONTACT_ENCODER_PROFILE, "event_id": event.id, "run_id": event.run_id,
            "kind": event.kind, "timestamp": event.ts, "device": payload["device"], "command": payload["command"],
            "geometry_source": geometry["source"], "mechanics_version": geometry["mechanics_version"],
            "microscope_source": microscope["source"], "axes_frame": microscope["stamp_axes_frame"],
            "holder_axes_xyz_mm": {k: pose[k] for k in ("x", "y", "z")}, "camera_world_xy_mm": list(camera)}


def _contact_nominal_head_projection(event):
    from qumus_simlab.records.discovery import nominal_head_projection

    return nominal_head_projection(event.model_dump())


def _contact_encoder_event(store, receipt, run_id):
    """Validate a named projection without reinterpreting old whole-payload hashes."""
    if (not isinstance(receipt, dict) or receipt.get("profile") != _CONTACT_ENCODER_PROFILE
            or set(receipt) != {"event_id", "profile", "projection_sha256"}
            or type(receipt.get("event_id")) is not int or receipt["event_id"] <= 0):
        raise ValueError("Contact encoder receipt requires its explicit supported projection profile")
    events = store.events_after(receipt["event_id"]-1, limit=1)
    if (not events or events[0].id != receipt["event_id"] or events[0].run_id != run_id
            or events[0].kind != "lab.frame"
            or _digest(_contact_encoder_projection(events[0])) != receipt.get("projection_sha256")):
        raise ValueError("Contact public encoder/camera projection changed or is unavailable")
    return events[0]


def _contact_capture_receipts(store, run_id, name, record, after):
    frames = list(_transfer_events(store, run_id, "lab.frame", after))
    captures = [event for event in frames
                if event.payload.get("microscope_preview", {}).get("phase") == name
                and event.payload["microscope_preview"].get("image") == record["image"]
                and event.payload["microscope_preview"].get("timestamp") == record["captured_at"]]
    if len(captures) != 1:
        raise ValueError("Contact evidence lacks its unique original exposure")
    capture = captures[0]
    encoders = [event for event in _transfer_events(store, run_id, "lab.frame")
                if event.id < capture.id and event.ts == capture.ts
                and event.payload.get("geometry", {}).get("microscope", {}).get("stamp_axes_xyz_mm")]
    if not encoders:
        raise ValueError("Contact exposure lacks its same-clock public encoder/camera receipt")
    encoder = encoders[-1]
    return {"capture": {"event_id": capture.id, "event_sha256": _digest(capture.payload)},
            "encoder": {"event_id": encoder.id, "profile": _CONTACT_ENCODER_PROFILE,
                        "projection_sha256": _digest(_contact_encoder_projection(encoder))}}


def _contact_public_capture(store, run_id, name, record, receipts):
    _validate_ring_capture(store, run_id, name, record, receipts["capture"])
    event = _contact_encoder_event(store, receipts["encoder"], run_id)
    if event.ts != record["captured_at"] or event.id >= receipts["capture"]["event_id"]:
        raise ValueError("Contact encoder receipt is not the original same-clock exposure context")
    # Only calibrated axes and camera origin are consumed. Renderer-only stamp
    # contact/apex/gap fields in the containing event are never read.
    body = event.payload["geometry"]["microscope"]
    pose, axis = body["stamp_axes_xyz_mm"], body["optical_axis_world_mm"]
    if not all(_finite(pose.get(k)) for k in ("x", "y", "z")) or len(axis) != 2 or not all(map(_finite, axis)):
        raise ValueError("Contact exposure has invalid public axes/camera calibration")
    return pose, axis


def _contact_support_context(store, run_id, phase, coverage):
    """Observed opening + fixed declared support; unknown overlays never authorize pixels."""
    run = store.get_run(run_id) or {}
    state = run["meta"]["normal_transfer"]
    v1_window = bool(run.get("params", {}).get("skip_aperture_alignment"))
    if v1_window and (store.bench != "sim" or state.get("aperture_alignment")):
        raise ValueError("The v1 no-window profile is SIM-only and cannot mix with an observed alignment")
    alignment = None if v1_window else validate_aperture_alignment(store, run_id, state["aperture_alignment"])
    reference, current = coverage["reference"], coverage["current"]
    _image_calibration(_reference(reference))
    _image_calibration(_reference(current))
    if any(reference["meta"].get(k) != current["meta"].get(k) for k in
           ("calibration", "objective", "focus_z", "focus_plane", "stage_kinematics")):
        raise ValueError("Contact optical context changed after its phase reference")
    if reference["meta"].get("objective") != 5 or reference["meta"].get("focus_plane") != "chip":
        raise ValueError("Contact interior requires the supported native low-power chip plane")
    condition = reference["meta"]["calibration"].get("observation_condition") or {}
    noise = condition.get("observation_noise_scale")
    if type(noise) not in (int, float) or not math.isfinite(noise) or not 0 <= noise <= 1:
        raise ValueError("Contact sensor noise exceeds its declared common-RGB bound")
    before, camera = _contact_public_capture(store, run_id, phase["name"], reference, coverage["reference_receipts"])
    after, camera_after = _contact_public_capture(store, run_id, phase["name"], current, coverage["current_receipts"])
    if (any(before[k] != after[k] for k in ("x", "y")) or camera != camera_after
            or alignment is not None and camera != alignment["frames"][-1]["camera_world_mm"]):
        raise ValueError("Moving holder/glass/cap/opening or optical axis lacks a common reference")
    model = reference["meta"]["calibration"].get("stamp_structure_model") or {}
    if alignment is None:
        # v1 profile: no observed opening, so no paper/cap clearance claim.
        return {"aperture_receipt": None, "support_policy": "v1_no_window_alignment",
                "fixed_holder_xy_mm": [before["x"], before["y"]], "camera_world_mm": camera,
                "paper_clearance_mm": None, "cap_clearance_mm": None,
                "support_model_sha256": geometry_digest(model), "reference_z_mm": before["z"], "current_z_mm": after["z"]}
    definition, slide = model.get("definition", {}), model.get("slide", {})
    radius, offset = definition.get("cap_radius_mm"), definition.get("aperture_offset_mm")
    size, inset = slide.get("size"), slide.get("dome_inset_mm")
    if (model.get("source") != "synthetic" or definition.get("profile_id") != "synthetic-glass-pdms-pc-paper-v2"
            or definition.get("aperture_shape") != "diamond" or type(radius) not in (int, float)
            or not _finite(radius) or radius <= 0 or not isinstance(offset, list) or len(offset) != 2
            or not all(map(_finite, offset)) or not isinstance(size, list) or len(size) != 2
            or not all(_finite(v) and v > 0 for v in size) or not _finite(inset)
            or radius > min(inset, size[0]-inset, size[1]/2)):
        raise ValueError("Glass/cap/opening support is not covered by the declared calibration")
    scales = reference["meta"]["calibration"]["mm_per_pixel"]
    width, height = reference["meta"]["calibration"]["image_size_px"]
    corners = [(x*width*scales[0]/2, y*height*scales[1]/2) for x in (-1, 1) for y in (-1, 1)]
    delta = [before[k]-alignment["centered_holder_xy_mm"][j] for j, k in enumerate(("x", "y"))]
    fit = alignment["confirmation_fit"]
    uncertainty = fit["uncertainty_mm"]+.005  # existing manipulator readback tolerance
    clearances = []
    for line in fit["edge_lines"]:
        lo, hi = sorted(line["intercepts_mm"])
        dots = [sum(n*(v-d) for n, v, d in zip(line["normal"], corner, delta, strict=True)) for corner in corners]
        clearances.extend((min(dots)-lo, hi-max(dots)))
    cap_center = [fit["center_offset_mm"][j]+delta[j]-offset[j] for j in (0, 1)]
    paper_clearance = min(clearances)-uncertainty
    cap_clearance = radius-max(math.dist(corner, cap_center) for corner in corners)-uncertainty
    if min(paper_clearance, cap_clearance) <= 0:
        raise ValueError("Visible field crosses uncertain paper/cap/glass transmission support")
    return {"aperture_receipt": {"event_id": alignment["receipt_event_id"], "event_sha256": alignment["receipt_sha256"]},
            "fixed_holder_xy_mm": [before["x"], before["y"]], "camera_world_mm": camera,
            "paper_clearance_mm": paper_clearance, "cap_clearance_mm": cap_clearance,
            "support_model_sha256": geometry_digest(model), "reference_z_mm": before["z"], "current_z_mm": after["z"]}


def _contact_reference_context(ctx, phase, source_reference):
    """Freeze the ordinary sole-stage-chip ledger at the original reference."""
    from qumus_simlab.records.lab_geometry import LabEvidence, Source, record_evidence

    chips = ctx.store.list_samples(kind="chip", location="transfer_stage")
    expected = ctx.params["flake_chips"][source_reference["source_id"]]
    if (len(chips) != 1 or chips[0]["id"] != expected
            or chips[0]["props"].get("location_uncertain")):
        raise NeedsHumanError("Contact photometry requires one unambiguous source chip on the stage")
    body = {"phase": phase["name"], "source_chip_id": expected,
            "sole_stage_chip_ids": [expected], "source_reference_sha256": source_reference["sha256"],
            "reference_event": phase["reference_event"],
            "stage_kinematics": phase["reference"]["meta"]["stage_kinematics"]}
    event_id = record_evidence(ctx.store, LabEvidence(subject=expected, property="contact_reference_context",
        bench="sim", value=body, units="mm", frame="microscope_image",
        source=Source(kind="synthetic", reference=f"{ctx.run_id}:{phase['name']}",
                      sha256=source_reference["sha256"], event_ids=[phase["reference_event"]["event_id"]]),
        observed_at=phase["reference"]["captured_at"], timebase="store_clock"), run_id=ctx.run_id)
    event = ctx.store.events_after(event_id-1, limit=1)[0]
    phase["contact_reference_context"] = {"event_id": event_id, "event_sha256": _digest(event.payload)}
    _save_phase(ctx, phase)


def _contact_z_trajectory(store, run_id, *, source_id, first, last, start, end):
    """Read only the original ID-bounded command and actual encoder prefix."""
    from qumus_simlab.records.release_support import command_encoder_readings

    def row(event):
        return {"id": event.id, "ts": event.ts, "run_id": event.run_id,
                "kind": event.kind, "payload": event.payload}
    commands = [e for e in _transfer_events(store, None, "device.command", source_id) if e.id < last]
    if any(e.payload.get("device") in (STAGE, "transfer_stage_fine", ARM, "arm_left") for e in commands):
        raise ValueError("Stage/chip motion interrupted the parked source-support binding")
    prefix = [e for e in commands if first < e.id < last]
    prior, variation, timeline = start, 0., []
    for index, command in enumerate(prefix):
        device, operation = command.payload.get("device"), command.payload.get("command")
        if device == MICROSCOPE and operation == "capture":
            continue
        params = command.payload.get("params", {})
        if (command.run_id != run_id or device != MANIP or operation != "move_axis_to"
                or params.get("axis") != "z" or params.get("from_pose") != prior):
            raise ValueError("Contact registration requires the complete unchanged-XY Z-only command prefix")
        following = prefix[index+1].id if index+1 < len(prefix) else last
        owned = [row(e) for e in _transfer_events(store, None, "lab.frame", command.id) if e.id < following]
        encoders = command_encoder_readings(owned, row(command))
        poses = [e["payload"]["geometry"]["microscope"]["stamp_axes_xyz_mm"] for e in encoders]
        if (not poses or any(not p or any(not _finite(p.get(k)) for k in ("x", "y", "z"))
                             or any(p[k] != start[k] for k in ("x", "y")) for p in poses)):
            raise ValueError("Contact command lost its actual encoder trajectory")
        sequence = [prior, *poses]
        differences = [b["z"]-a["z"] for a, b in zip(sequence, sequence[1:], strict=False)]
        if any(delta*(poses[-1]["z"]-prior["z"]) < 0 for delta in differences):
            raise ValueError("Encoder trace is incompatible with the declared monotone synthetic axis motion")
        variation += sum(abs(b["z"]-a["z"]) for a, b in zip(sequence, sequence[1:], strict=False))
        prior = poses[-1]
        timeline.append({"command_event_id": command.id, "command_sha256": _digest(command.payload),
                         "encoder_event_ids": [e["id"] for e in encoders],
                         "encoder_sha256": _digest([{"id": e["id"], "ts": e["ts"], "pose": p}
                                                     for e, p in zip(encoders, poses, strict=True)])})
    if not timeline or prior != end or last <= first:
        raise ValueError("Contact trajectory does not end at the original current exposure")
    return variation, timeline


def _design_contact_dwell(ctx, phase):
    c = _profile(ctx)
    dwell = ctx.params["contact_dwell_s"]
    if not _finite(dwell) or not 0 < dwell <= 60 or ctx.clock.now()+dwell > phase["started_at"]+c["max_total_s"]:
        raise NeedsHumanError("No finite phase allowance remains for the required contact dwell")
    hold = {"started_at": ctx.clock.now(), "duration_s": dwell,
            "source": ctx.params["contact_dwell_source"], "completed": False}
    phase["contact_dwell"] = hold
    _save_phase(ctx, phase)
    deadline = hold["started_at"]+dwell
    while ctx.clock.now() < deadline:
        ctx.check_cancel()
        ctx.gate.watchdog_tick()
        ctx.clock.sleep(min(.2, deadline-ctx.clock.now()))
    hold.update(completed=True, completed_at=ctx.clock.now())
    body = {"phase": phase["name"], "hold": hold}
    event_id = ctx.store.emit("stack.design_contact_dwell", body, run_id=ctx.run_id)
    phase["contact_dwell_event"] = {"event_id": event_id, "event_sha256": _digest(body)}
    _save_phase(ctx, phase)


def _validate_design_pickup_additions(ctx, phase, source_reference, endpoint):
    """Recompute whole-body pixels and bind dwell/clearance to captured readbacks."""
    if phase.get("coverage") != endpoint.get("coverage"):
        raise ValueError("Saved pickup coverage differs from its independently verified endpoint")
    hold = phase.get("contact_dwell") or {}
    started, ended = hold.get("started_at"), hold.get("completed_at")
    if (not all(_finite(v) for v in (started, ended)) or hold.get("completed") is not True
            or hold.get("duration_s") != ctx.params["contact_dwell_s"]
            or hold.get("source") != ctx.params["contact_dwell_source"]
            or ended-started+math.ulp(started)+math.ulp(ended) < ctx.params["contact_dwell_s"]):
        raise ValueError("new-design pickup lost its finite completed Clock dwell")
    receipt = _historical_event(ctx.store, phase["contact_dwell_event"], ctx.run_id, "stack.design_contact_dwell")
    if receipt.payload != {"phase": phase["name"], "hold": hold}:
        raise ValueError("completed Clock dwell differs from its immutable receipt")
    park = phase.get("source_inspection_park") or {}
    actual = park.get("actual_holder_pose") or {}
    expected = ctx.params["design_park_pose"]
    if (not actual or any(not _finite(actual.get(k)) or abs(actual[k]-v) > ctx.verification["manipulator"]["tolerance_mm"]
                          for k, v in expected.items()) or park.get("completed_at", -math.inf) < ended):
        raise ValueError("source inspection lacks its actual declared lateral-clear park")
    receipt = _historical_event(ctx.store, phase["source_inspection_park_event"], ctx.run_id, "stack.design_source_park")
    if receipt.payload != {"phase": phase["name"], "park": park}:
        raise ValueError("source-inspection park differs from its immutable receipt")
    focus = ctx.store.events_after(endpoint["focus_event_id"]-1, limit=1)[0]
    if (focus.payload.get("holder_pose") != actual
            or any(frame["captured_at"] < park["completed_at"] for frame in endpoint["source_frames"])):
        raise ValueError("source images/focus were captured before lateral stamp clearance")
    coverage = phase["coverage"]
    reference, current = _reference(coverage["reference"]), _reference(coverage["current"])
    calibration, pixels = _image_calibration(reference), load_pixels(reference)
    c = ctx.verification["transfer_sequence"]
    ox, oy = c["pickup_target_offset_image_mm"]
    target_px = (pixels.width/2+ox/abs(calibration.mm_per_pixel_x), pixels.height/2+oy/abs(calibration.mm_per_pixel_y))
    recomputed = v1_pickup_coverage(reference, current, calibration, target_px, margin_mm=c["pickup_margin_mm"],
        min_area_mm2=c["minimum_area_mm2"], blur_support_mm=c["pickup_blur_support_mm"])
    if (recomputed.get("covered") is not True or coverage.get("covered") is not True
            or _digest(recomputed) != _digest({k: coverage.get(k) for k in recomputed})):
        raise ValueError("The v1 pickup stop rule is not established by the original frames")
def _complete_detection_outline(detection):
    from qumus_simlab.core.polygons import convex_polygon, transform

    if (not detection.meta.get("outline_complete") or detection.meta.get("geometry_source") in (None, "legacy_display")):
        raise NeedsHumanError("Design coverage requires complete captured outlines")
    return transform(convex_polygon(detection.meta["outline_mm"]),
                     xy=(detection.dx_mm, detection.dy_mm))


def _original_mask_region(chain):
    """Calibrated original selected mask, including holes; no RGB refinement."""
    import numpy as np
    from PIL import Image

    from qumus_simlab.core.regions import transform_region
    from qumus_simlab.core.run_paths import resolve_run_path
    from qumus_simlab.perception.rgb_material import component_geometry
    from qumus_simlab.records.recognition import validate_recognition_chain

    validate_recognition_chain(chain, image_path=chain["image"]["path"],
                               objective=chain["image"]["objective"])
    derivation = chain.get("derivation") or {}
    support = chain.get("recognizer_support") or {}
    if derivation.get("status") != "supported" or support.get("whole_body_status") == "unsupported":
        raise ValueError("Original recognizer did not observe a complete supported component")
    with Image.open(resolve_run_path(chain["geometry_mask"]["path"])) as mask:
        geometry = component_geometry(np.asarray(mask) != 0, chain["calibration"])
    if geometry["status"] != "supported":
        raise ValueError("Original selected mask is unsupported: "+str(geometry["unsupported_reason"]))
    return transform_region(geometry["region_mm"], xy=(geometry["dx_mm"], geometry["dy_mm"]))


def _complete_detection_region(ctx, frame, detection):
    from qumus_simlab.core.regions import as_region

    if "recognition_chain" in detection.meta:
        chain = analyzer(ctx).persist_detection_masks(frame, detection)
        try:
            return _original_mask_region(chain)
        except ValueError as exc:
            raise NeedsHumanError("Original selected material mask is unsupported: "+str(exc)) from exc
    return as_region(_complete_detection_outline(detection))


def _carried_hbn_over_gr(ctx, stamp_frame, carried, gr):
    """The one carried hBN that completely covers the selected Gr.

    Owner decision (#136): a neighbour may be co-picked with the source, so the
    stamp can carry more than one hBN. Exactly one complete carried hBN must
    cover the Gr; any other carried hBN that touches the Gr, or whose complete
    outline is not observable, stops for review.
    """
    from qumus_simlab.core.regions import region_area_um2, region_covers, region_intersection

    if not carried:
        raise NeedsHumanError("Complete carried hBN is not observable before Gr pickup")
    regions = []
    for detection in carried:
        try:
            regions.append(_complete_detection_region(ctx, stamp_frame, detection))
        except (NeedsHumanError, ValueError) as exc:
            raise NeedsHumanError("A carried hBN outline is incomplete before Gr pickup") from exc
    covering = [region for region in regions if region_covers(region, gr)]
    if not covering:
        # Jialei 2026-11-03: the key criterion is that the whole Gr lies
        # inside the MAIN hBN; name how far it reaches beyond it.
        main = max(regions, key=lambda region: region_area_um2(region_intersection(region, gr)))
        raise NeedsHumanError(f"The selected Gr extends {_excess_um(main, gr):.1f} um beyond the main carried hBN")
    if len(covering) != 1:
        raise NeedsHumanError("Exactly one carried hBN must cover the selected Gr before pickup")
    if any(region is not covering[0] and region_intersection(region, gr).components for region in regions):
        raise NeedsHumanError("A co-picked carried hBN besides the main hBN overlaps the selected Gr")
    return covering[0]


def _excess_um(outer, inner) -> float:
    """Farthest reach of ``inner`` outside ``outer``, in micrometres."""
    from shapely.geometry import Point

    from qumus_simlab.core.regions import _geos, region_difference, region_vertices

    body = _geos(outer)
    return max((body.distance(Point(p)) for p in region_vertices(region_difference(inner, outer))),
               default=0.0)*1000


def _observe_design_pickup_coverage(ctx, flake_id, stamp_id):
    """Observe carried hBN and current Gr in the same held image-plane frame."""
    from qumus_simlab.core.regions import region_covers, region_dict

    if getattr(analyzer(ctx), "recognition_backend", None) == "v1_lab_rgb":
        # v1 has no pre-pickup coverage look: it runs the pickup straight away.
        # Here the holder is already engaged, so the Gr would be imaged through
        # the hBN-carrying stamp, and the v1 RGB recognizer only admits bare
        # SiO2 background (seed 3 r11: tinted frames, no background support).
        # Gr-inside-hBN is still checked on the finished stack.
        ctx.store.emit("stack.design_pickup_coverage", {"design_digest": ctx.params["device_design"]["canonical_digest"],
            "stamp_id": stamp_id, "source_id": flake_id, "covered": None, "skipped": "v1_lab_rgb",
            "scope": "v1 flow has no pre-pickup coverage look; the chip would be seen through the engaged stamp"},
            run_id=ctx.run_id)
        return
    scope = ctx.gate.microscope(MICROSCOPE)
    objective, focus = scope.get_objective(), scope.get_focus()
    held = (_pos(ctx), ctx.gate.stage(STAGE).kinematics())
    state = _transfer_state(ctx)
    source = state["source_references"][flake_id]
    try:
        scope.set_objective(20)
        # The scope sits at the fixed stamp focus (substrate + offset) here; seed the
        # chip-plane search from the substrate focus recorded at initialization so
        # the bounded window brackets the chip, not the stamp plane.
        init = state.get("initializations", {}).get(state.get("active_initialization"))
        if init and init.get("applied"):
            scope.set_focus(init["substrate_focus_mm"])
        autofocus(ctx, 20, attempts=1, half_range=.15, plane="chip")
        chip_frame = scope.capture()
        target = source["target_offset_mm"]
        flakes = [d for d in analyzer(ctx).find_flakes(chip_frame, "Gr")
                  if math.hypot(d.dx_mm-target[0], d.dy_mm-target[1]) <= .012]
        if len(flakes) != 1:
            raise NeedsHumanError("Current selected Gr outline is ambiguous before pickup")
        gr = _complete_detection_region(ctx, chip_frame, flakes[0])
        process = _process_reference(state, flake_id, stamp_id, ctx.run_id, .2, ctx.store)
        scale = source["focus_binding"]["observation"]["focus_axis_calibration"]["scale_mm_per_focus_mm"]
        seed = scope.get_focus() + (held[0]["z"]-sum(process["reference_interval_mm"])/2)/scale
        scope.set_focus(seed)
        autofocus(ctx, 20, attempts=1, half_range=.15, plane="stamp")
        stamp_frame = scope.capture()
        carried = [d for d in analyzer(ctx).find_stamp_flakes(stamp_frame) if d.material == "hBN"]
        hbn = _carried_hbn_over_gr(ctx, stamp_frame, carried, gr)
        if held != (_pos(ctx), ctx.gate.stage(STAGE).kinematics()) or not region_covers(hbn, gr):
            raise NeedsHumanError("Observed carried hBN does not fully cover the actual selected Gr")
        ctx.store.emit("stack.design_pickup_coverage", {"design_digest": ctx.params["device_design"]["canonical_digest"],
            "stamp_id": stamp_id, "source_id": flake_id, "covered": True, "frame": "held_microscope_xy", "units": "mm",
            "hbn_region_mm": region_dict(hbn), "gr_region_mm": region_dict(gr), "source_frame": _frame_record(chip_frame),
            "stamp_frame": _frame_record(stamp_frame),
            "scope": "Observed whole-outline containment; polymer adhesion remains a simulation assumption"}, run_id=ctx.run_id)
    finally:
        scope.set_focus(focus)
        scope.set_objective(objective)


def _endpoint_shape(outline, general):
    from qumus_simlab.core.polygons import convex_polygon
    from qumus_simlab.core.regions import as_region

    return (as_region(outline.get("region_mm", outline.get("outline_mm"))) if general
            else convex_polygon(outline["outline_mm"]))


def _endpoint_covers(outer, inner, general):
    from qumus_simlab.core.polygons import contains
    from qumus_simlab.core.regions import region_covers

    return region_covers(outer, inner) if general else contains(outer, inner)


def _endpoint_centroid(polygon, general):
    if general:
        from qumus_simlab.core.regions import _geos

        c = _geos(polygon).centroid
        return (c.x, c.y)
    return tuple(sum(v[i] for v in polygon)/len(polygon) for i in (0, 1))


def _graphite_bridges(target, flakes, general):
    """Electrode groups (connected drawn metal) each extra graphite flake touches."""
    import shapely

    from qumus_simlab.core.regions import _geos, as_region
    from qumus_simlab.records.device_design import target_metal_primitives

    metal = shapely.unary_union([shapely.Polygon(p["points_mm"]) for p in target_metal_primitives(target)])
    electrodes = list(getattr(metal, "geoms", [metal]))
    bridged = []
    for flake in flakes:
        body = _geos(as_region(flake)) if general else shapely.Polygon(flake)
        touched = [i for i, e in enumerate(electrodes) if body.intersects(e)]
        if len(touched) > 1:
            bridged.append(touched)
    return bridged


def _device_design_endpoint(ctx, before, after):
    from qumus_simlab.core.polygons import (
        rigid_polygon_error,
    )
    from qumus_simlab.core.regions import (
        _geos,
        as_region,
        region_compatible_with_bound,
        transform_region,
    )
    from qumus_simlab.records.device_design import (
        source_boundary_uncertainty,
        target_contact_coverage,
    )

    design = _device_design_public_binding(ctx.params, ctx.store, ctx.store.get_run(ctx.run_id))
    result = {"status": "unresolved", "design_digest": design.canonical_digest,
        "run_id": ctx.run_id, "observations": [], "scope": "Complete optical geometry and recorded layer order; no electrical or polymer metrology claim"}
    # The design keeps its frozen pose margin; the final placement is accepted
    # within the owner's endpoint tolerance (Jialei 2026-11-03: ~1.5 um is
    # acceptable). Containment and electrode coverage are still checked
    # directly from the observed outlines below.
    accept_um = max(design.pose_tolerance.translation_um,
                    (getattr(ctx, "verification", None) or {}).get("device_design_endpoint", {}).get(
                        "translation_tolerance_um", design.pose_tolerance.translation_um))
    accept_mm = accept_um/1000
    result["accepted_translation_tolerance_um"] = accept_um
    groups = after.get("observed_outlines") or []
    if (len(groups) != 6 or after.get("device_design_digest") != design.canonical_digest
            or any(before["materials_found"].values())):
        return {**result, "reason": "unqualified design target/source observations"}
    result["extra_flakes"] = []
    for index in range(0, 6, 2):
        pair = {g["material"]: g for g in groups[index:index+2]}
        if set(pair) != {"hBN", "Gr"} or any(not g["outlines"] for g in pair.values()):
            return {**result, "reason": "each target frame needs one unambiguous complete hBN and Gr outline"}
        general = any(hasattr(source, "region_mm") for source in design.sources)
        general = general or any("region_mm" in o for g in pair.values() for o in g["outlines"])
        candidates = {m: [_endpoint_shape(o, general) for o in g["outlines"]] for m, g in pair.items()}
        # Co-picked extras may land with the stack (Jialei 2026-11-03): extra hBN
        # anywhere is harmless; only graphite joining two electrodes is a short.
        # The designed Gr is the observed Gr nearest its designed placement, and
        # the main hBN is the one hBN that covers it completely.
        source_gr = next(src for src in design.sources if src.material == "Gr")
        planned = source_gr.placed_region() if general and hasattr(source_gr, "region_mm") else (
            as_region(source_gr.placed_outline()) if general else source_gr.placed_outline())
        anchor = _endpoint_centroid(planned, general)
        distance = {id(g): math.dist(_endpoint_centroid(g, general), anchor) for g in candidates["Gr"]}
        gr = min(candidates["Gr"], key=lambda g: distance[id(g)])
        main = [h for h in candidates["hBN"] if _endpoint_covers(h, gr, general)]
        if len(main) > 1:
            return {**result, "reason": "more than one observed hBN covers the designed Gr"}
        if not main:
            return {**result, "status": "fail", "reason": "actual hBN does not cover complete actual Gr"}
        polygons = {"hBN": main[0], "Gr": gr}
        extras_gr = [g for g in candidates["Gr"] if g is not gr]
        if extras_gr:
            bridged = _graphite_bridges(design.target, extras_gr, general)
            if bridged:
                return {**result, "status": "fail", "reason": "an extra graphite flake joins two electrodes",
                        "bridged_electrodes": bridged}
        if index == 0:
            result["extra_flakes"] = ([{"material": "hBN"}]*(len(candidates["hBN"])-1)
                                      + [{"material": "Gr", "joins_electrodes": False}]*len(extras_gr))
        coverage = target_contact_coverage(design.target, polygons["Gr"], minimum_overlap_um=design.contact_overlap_um)
        overlaps = {key: value["intersection_area_um2"] for key, value in coverage["contacts"].items()}
        if coverage["status"] != "pass":
            return {**result, "status": "fail", "reason": "actual Gr misses a required electrode contact region",
                    "overlap_area_um2": overlaps, "contact_coverage": coverage}
        for source in design.sources:
            if hasattr(source, "region_mm"):
                actual, expected = polygons[source.material], source.placed_region()
                if (not actual.components or sorted(len(c.holes) for c in actual.components)
                        != sorted(len(c.holes) for c in expected.components)):
                    return {**result, "status": "fail", "reason": "actual complete source topology changed"}
                a, b = _geos(expected), _geos(actual)
                fit = rigid_polygon_error(tuple(a.minimum_rotated_rectangle.exterior.coords)[:-1],
                                          tuple(b.minimum_rotated_rectangle.exterior.coords)[:-1])
                if fit["rotation_deg"] is None:
                    return {**result, "status": "fail", "reason": "actual complete source pose is unresolved"}
                ea, eb = a.centroid, b.centroid
                aligned = transform_region(transform_region(expected, xy=(-ea.x, -ea.y)),
                    rotation_deg=fit["rotation_deg"], xy=(eb.x, eb.y))
                fit["translation_mm"] = [eb.x-ea.x, eb.y-ea.y]
                fit["shape_residual_mm"] = float(_geos(aligned).boundary.hausdorff_distance(b.boundary))
                if not region_compatible_with_bound(aligned, actual, source_boundary_uncertainty(source)):
                    return {**result, "status": "fail", "reason": "actual complete source shape changed", "rigid_fit": fit}
                if (math.hypot(*fit["translation_mm"]) > accept_mm
                        or abs(fit["rotation_deg"]) > design.pose_tolerance.rotation_deg):
                    return {**result, "status": "fail", "reason": "actual source pose exceeds its declared design tolerance", "rigid_fit": fit}
                continue
            actual, expected = polygons[source.material], source.placed_outline()
            if general:
                if len(actual.components) != 1 or actual.components[0].holes:
                    return {**result, "status": "fail", "reason": "actual legacy source topology changed"}
                actual = actual.components[0].exterior
            if len(actual) != len(expected):
                return {**result, "status": "fail", "reason": "actual source outline was not preserved"}
            fit = rigid_polygon_error(expected, actual)
            if fit["shape_residual_mm"] is None or fit["shape_residual_mm"] > 1e-9:
                return {**result, "status": "fail", "reason": "actual complete source shape changed", "rigid_fit": fit}
            if (math.hypot(*fit["translation_mm"]) > accept_mm
                    or abs(fit["rotation_deg"]) > design.pose_tolerance.rotation_deg):
                return {**result, "status": "fail", "reason": "actual source pose exceeds its frozen synthetic design tolerance", "rigid_fit": fit}
        result["observations"].append({"overlap_area_um2": overlaps,
            **({"contact_coverage": coverage} if design.target.version == "hallbar-target-gds-v2" else {}),
            "bindings": [g["observation_binding"] for g in pair.values()]})
    return {**result, "status": "pass", "layers_top_first": list(design.pickup_order)}


def _design_pose_window_binding(raw, design, store, run=None):
    """Validate only an explicitly frozen fresh screen, never backfill history."""
    sidecar = raw.get("design_pose_window_admission")
    if sidecar is not None:
        validated = validate_design_pose_window_admission(design, sidecar)
        if run is not None:
            submitted = list(_transfer_events(store, run["id"], "job.submitted"))
            if len(submitted) != 1 or submitted[0].payload["params"].get("design_pose_window_admission") != sidecar:
                raise ValueError("Design screen differs from its original submission")
            evidence = [e for e in _transfer_events(store, run["id"], "lab.evidence")
                        if e.payload.get("property") == "device_design_pose_window_admission"]
            submission = submitted[0]
            if (len(evidence) != 1 or evidence[0].id <= submission.id
                    or evidence[0].payload.get("subject") != design.design_id
                    or evidence[0].payload.get("value") != validated.model_dump(mode="json")
                    or evidence[0].payload.get("bench") != "sim"
                    or evidence[0].payload.get("source", {}).get("kind") != "synthetic"
                    or evidence[0].payload["source"].get("sha256") != validated.report_sha256
                    or evidence[0].payload["source"].get("event_ids") != [submission.id]
                    or any(submission.id < e.id < evidence[0].id for e in _transfer_events(store, run["id"], "step.started"))):
                raise ValueError("Design pose-window admission lost its original pre-motion submission receipt")
    elif raw.get("release_policy") == OBSERVED_RELEASE_POLICY_ID:
        raise ValueError("Unscreened historical design cannot authorize a new observed-release run")


def _device_design_public_binding(raw: dict, store: Any, run: dict | None = None) -> DeviceDesignV1:
    """Preserve immutable source facts while admitting only own verified lifecycle."""
    from qumus_simlab.core.materials import normalize_material_code
    from qumus_simlab.core.run_paths import resolve_run_path
    from qumus_simlab.records.device_design import parse_hallbar_target, target_import_binding
    from qumus_simlab.records.geometry import _pickup_region
    from qumus_simlab.records.materials import pickup_admission_error
    from qumus_simlab.records.stamps import contact_film

    design = validate_design_artifact(store, raw["device_design"])
    if run is not None:
        if run.get("project_id") != design.project_id:
            raise ValueError("saved design run belongs to another project")
        submitted = list(_transfer_events(store, run["id"], "job.submitted"))
        frozen_keys = ("device_design", "stamp_id", "stamp_film", "flake_ids", "flake_chips", "materials",
                       "target_chip_id", "target_xy_mm", "design_shifts_mm", "design_rotations_deg", "design_path",
                       "contact_dwell_s", "contact_dwell_source", "first_contact_sampling", "design_park_pose", "contact_interior_calibration", "design_pose_window_admission", "stamp_discovery_policy")
        if len(submitted) != 1 or any(geometry_digest({"value": submitted[0].payload["params"].get(k)}) != geometry_digest({"value": raw.get(k)}) for k in frozen_keys):
            raise ValueError("design execution inputs differ from the immutable submitted snapshot")
    _design_pose_window_binding(raw, design, store, run)
    if raw.get("contact_interior_calibration") != _contact_classification_snapshot(store, design):
        raise ValueError("Saved public material classification/contact calibration changed")
    state = (store.get_project(design.project_id) or {}).get("state", {})
    if (state.get("device_design") != design.model_dump(mode="json")
            or geometry_digest(state.get("device_plan") or {}) != design.guidance_sha256
            or raw["flake_ids"] != list(design.pickup_order) or raw["target_chip_id"] != design.target_chip_id):
        raise ValueError("saved design revision, guidance, identities or pickup order changed")
    target = store.get_sample(design.target_chip_id)
    if (not target or target["kind"] != "chip" or target["props"].get("location_uncertain")
            or parse_hallbar_target(target["props"].get("device_design_target")) != design.target
            or tuple(target["props"].get("size_mm") or ()) != design.target.size_mm):
        raise ValueError("saved design target geometry, size or location changed")
    imported_binding = target_import_binding(design.target)
    if imported_binding is not None and target["props"].get("imported_target_binding") != imported_binding:
        raise ValueError("Saved imported target source binding changed")
    expected_shifts, expected_angles = {}, {}
    anchor = design_anchor_mm(design.target)
    if tuple(raw["target_xy_mm"]) != anchor or raw.get("design_path") != design.artifact.path:
        raise ValueError("saved design target pose/artifact reference changed")
    for source in design.sources:
        a = math.radians(-source.placement.rotation_deg)
        x, y = (v-o for v, o in zip(source.placement.translation_mm, anchor, strict=True))
        expected_shifts[source.flake_id] = [x*math.cos(a)-y*math.sin(a), x*math.sin(a)+y*math.cos(a)]
        expected_angles[source.flake_id] = source.placement.rotation_deg
    if ({k: list(v) for k, v in raw.get("design_shifts_mm", {}).items()} != expected_shifts
            or raw.get("design_rotations_deg") != expected_angles):
        raise ValueError("saved design source transforms changed")
    if (raw.get("flake_chips") != {s.flake_id: s.source_chip_id for s in design.sources}
            or raw.get("materials") != {s.flake_id: s.material for s in design.sources}
            or raw.get("stamp_film") != "PC"):
        raise ValueError("saved design source-chip/material or PC-film binding changed")
    history = (run or {}).get("meta", {}).get("normal_transfer", {})
    stamp = store.get_sample(raw["stamp_id"])
    if not stamp or stamp["kind"] != "stamp" or stamp["props"].get("location_uncertain"):
        raise ValueError("selected design stamp is unknown")
    picked, released = [], []
    for source in design.sources:
        flake = store.get_sample(source.flake_id)
        if (not flake or flake["kind"] != "flake"
                or normalize_material_code(flake["props"].get("material")) != source.material
                or flake["props"].get("location_uncertain")
                or flake["props"].get("optical_geometry") != source.observation
                or hashlib.sha256(resolve_run_path(source.observation["image"]).read_bytes()).hexdigest() != source.image_sha256):
            raise ValueError("saved complete source observation or image changed")
        phase = history.get("phases", {}).get(pickup_name(raw, source.flake_id), {})
        if flake["location"] == source.source_chip_id:
            source_chip = store.get_sample(source.source_chip_id)
            if not source_chip or source_chip["kind"] != "chip" or source_chip["props"].get("location_uncertain"):
                raise ValueError("pending design source chip identity/location is unknown")
            if picked and len(picked) != design.pickup_order.index(source.flake_id):
                raise ValueError("source lifecycle is out of pickup order")
            _pickup_region(store, flake)
            if flake["status"] not in STACKABLE:
                raise ValueError("pending design source is no longer stackable")
            if error := pickup_admission_error(store, flake):
                raise ValueError(error)
            continue
        if not run or not phase.get("complete") or not phase.get("endpoint_verified"):
            raise ValueError("moved design source lacks its same-run verified pickup")
        endpoint = phase.get("endpoint")
        reference = history.get("source_references", {}).get(source.flake_id)
        if not reference or not _visual_pickup_proof(endpoint, reference, store, run["id"]):
            raise ValueError("saved design pickup lost its optical endpoint proof")
        from qumus_simlab.config import load_yaml

        _validate_design_pickup_additions(SimpleNamespace(params=raw, store=store, run_id=run["id"],
            verification=load_yaml("verification.yaml")), phase, reference, endpoint)
        _validate_design_source_orientation(store, run["id"], source.flake_id, reference["source_identity"],
                                            load_yaml("verification.yaml"))
        _verified_endpoint(store, phase["endpoint_verified"], run["id"], pickup_name(raw, source.flake_id),
                           endpoint, [f["image"] for f in endpoint["source_frames"]])
        if flake["location"] == raw["stamp_id"] and flake["status"] == "on_stamp":
            picked.append(source.flake_id)
        elif flake["location"] == design.target_chip_id and flake["status"] == "stacked":
            released.append(source.flake_id)
        else:
            raise ValueError("design source has an unexplained location transition")
    if picked and (picked != list(design.pickup_order[:len(picked)]) or stamp["props"].get("stack") != picked):
        raise ValueError("actual recorded pickup order differs from the accepted design")
    expected_layers = released if released else []
    if target["props"].get("layers_top_first") != expected_layers:
        raise ValueError("design receiving layers changed without its verified release")
    if {f["id"] for f in store.list_samples(kind="flake", location=design.target_chip_id)} != set(released):
        raise ValueError("an unqualified layer is recorded on the design receiver")
    if not released and contact_film(stamp) != "PC":
        raise ValueError("selected design stamp no longer has its known PC film")
    if not picked and stamp["props"].get("stack", []) != []:
        raise ValueError("unqualified carried layers are present on the design stamp")
    if released:
        release = history.get("phases", {}).get("v1_drop_down", {})
        if released != list(design.pickup_order) or not release.get("endpoint_verified") or release.get("endpoint", {}).get("device_design_result", {}).get("status") != "pass":
            raise ValueError("design release lacks its complete qualified all-contact endpoint")
        before = history["target_reference"]
        endpoint = release["endpoint"]
        _verified_endpoint(store, release["endpoint_verified"], run["id"], "v1_drop_down", endpoint,
                           [f["image"] for f in before["frames"]+endpoint["target_after"]["frames"]])
    return design


def _direct_contact_public_binding(raw: dict, store: Any, run: dict | None = None,
                                   *, stage: str = "execution", before: dict | None = None) -> dict:
    """Validate frozen public facts at C5, pickup and release; never resolve anew.

    Own verified pickup/release records explain the permitted location/layer
    transitions. External image bytes and geometry remain frozen in every phase.
    This reads records/files only, without a device query or private state.
    """
    from qumus_simlab.records.geometry import _pickup_outline
    from qumus_simlab.records.materials import pickup_admission_error
    from qumus_simlab.records.stamps import contact_film

    c = DirectContactContract.model_validate(raw["direct_contact_contract"])
    target = c.target_intent["target_contact"]
    tid, fid = target["target_chip_id"], c.source_flake_id
    if target_contact_snapshot(store, tid, target["region"]["region_id"]) != target:
        raise ValueError("frozen direct-contact target geometry changed")
    flake, receiver, stamp = store.get_sample(fid), store.get_sample(tid), store.get_sample(raw["stamp_id"])
    if (not flake or flake["kind"] != "flake" or flake["props"].get("material") != "Gr"
            or flake["props"].get("location_uncertain") or receiver["props"].get("location_uncertain")
            or not stamp or stamp["kind"] != "stamp" or stamp["props"].get("location_uncertain")):
        raise ValueError("frozen direct-contact identities/locations are unknown or changed")
    observation = deepcopy(flake["props"].get("optical_geometry"))
    if not isinstance(observation, dict):
        raise ValueError("frozen source optical geometry is missing")
    observation["image_sha256"] = hashlib.sha256(Path(observation["image"]).read_bytes()).hexdigest()
    if observation != c.source_observation:
        raise ValueError("frozen source optical geometry/image bytes changed")
    state = (run or {}).get("meta", {}).get("normal_transfer") or {}
    if before is None and state.get("target_reference") is not None:
        before = _target_baseline(state, run["id"], store, tid, tuple(raw["target_xy_mm"]))
    if before is not None and (before.get("receiving_layers_above_contacts") != []
            or before.get("target_contact") != target):
        raise ValueError("receiving before image lacks the frozen explicit fresh target")
    release = state.get("phases", {}).get("v1_drop_down", {})
    released = flake["location"] == tid and flake["status"] == "stacked"
    verified_release_id = None
    if released:
        endpoint = release.get("endpoint") or {}
        contact = endpoint.get("direct_contact_result") or {}
        if (stage in ("pickup", "release") or not run or not before
                or release.get("complete") is not True or not release.get("endpoint_verified")
                or endpoint.get("release_confirmed") is not True or contact.get("status") != "pass"
                or contact.get("binding", {}).get("contract_sha256") != geometry_digest(c.model_dump(mode="json"))):
            raise ValueError("Gr on the receiver lacks this run's verified direct-contact release")
        _verified_endpoint(store, release["endpoint_verified"], run["id"], "v1_drop_down", endpoint,
            [f["image"] for f in before["frames"]+endpoint["target_after"]["frames"]])
        checkpoint = [s for s in store.get_steps(run["id"]) if s["name"] == "v1_drop_down"]
        device = store.get_sample(flake["props"].get("device", ""))
        if (len(checkpoint) != 1 or checkpoint[0]["status"] not in ("done", "corrected")
                or checkpoint[0]["measurements"] != endpoint or not device or device["kind"] != "device"
                or device["location"] != tid or device["props"].get("run_id") != run["id"]
                or device["props"].get("layers_top_first") != [fid]
                or device["props"].get("direct_contact_contract") != c.model_dump(mode="json")
                or device["props"].get("direct_contact_result") != contact
                or stamp["props"].get("stack") != []):
            raise ValueError("own released Gr ledger/receipt is incomplete or changed")
        verified_release_id = release["endpoint_verified"]["event_id"]
    elif contact_film(stamp) != "PC" and stage != "endpoint":
        raise ValueError("frozen direct contact requires the actual selected PC film")
    if receiver["props"].get("layers_top_first") != ([fid] if released else []):
        raise ValueError("frozen explicit receiving-layer declaration changed")
    if any(f["id"] != fid or not released for f in store.list_samples(kind="flake", location=tid)):
        raise ValueError("another recorded layer is on the fresh receiver")
    if run is not None:
        submitted = list(_transfer_events(store, run["id"], "job.submitted"))
        if len(submitted) != 1:
            raise ValueError("frozen direct contact lacks its original submission")
        events = (e for kind in ("sample.updated", "sample.removed")
            for e in _transfer_events(store, None, kind, submitted[0].id))
        for e in events:
            p = e.payload
            if e.kind == "sample.removed" and p.get("id") in (tid, fid, stamp["id"]):
                raise ValueError("a bound direct-contact specimen was removed")
            if e.kind != "sample.updated":
                continue
            own_release = verified_release_id is not None and e.id > verified_release_id
            if p.get("id") == tid and p.get("props", {}).get("layers_top_first") != ([fid] if own_release else []):
                raise ValueError("receiving-layer continuity changed since frozen submission")
            if p.get("kind") == "flake" and p.get("location") == tid and not (own_release and p.get("id") == fid):
                raise ValueError("an unqualified layer arrived on the receiver")
    incoming, pickup_events = [], []
    if not released and flake["location"] == c.source_observation["source_chip_id"]:
        pickup = state.get("phases", {}).get(f"v1_pick_up_{fid}", {})
        if (stage in ("release", "endpoint") or flake["status"] not in STACKABLE
                or stamp["props"].get("stack", []) != [] or pickup.get("endpoint_verified") or release.get("endpoint_verified")):
            raise ValueError("direct-contact Gr is not in its expected pickup phase")
        _pickup_outline(store, flake)
        if error := pickup_admission_error(store, flake):
            raise ValueError(error)
    elif not released:
        if (not run or flake["location"] != stamp["id"] or flake["status"] != "on_stamp"
                or stamp["location"] != MANIP or stamp["status"] != "in_use"
                or stamp["props"].get("stack") != [fid]
                or stage == "pickup" or release.get("endpoint_verified") and stage != "endpoint"):
            raise ValueError("direct-contact Gr lacks its expected on-stamp/commit phase")
        phase = state.get("phases", {}).get(f"v1_pick_up_{fid}", {})
        endpoints = [e for e in _transfer_events(store, run["id"], "stack.pickup_endpoint")
            if e.payload.get("flake_id") == fid and e.payload.get("stamp_id") == stamp["id"]]
        if len(endpoints) != 1 or phase.get("complete") is not True:
            raise ValueError("incoming layer needs one qualified same-run pickup endpoint")
        measurements = endpoints[0].payload["observation"]["measurements"]
        if (measurements != phase.get("endpoint") or not _visual_pickup_proof(measurements,
                state["source_references"][fid], store, run["id"])):
            raise ValueError("incoming pickup endpoint lost its image/process binding")
        _verified_endpoint(store, phase["endpoint_verified"], run["id"], f"v1_pick_up_{fid}", measurements,
            [f["image"] for f in measurements["source_frames"]])
        incoming, pickup_events = [{"flake_id": fid, "material": "Gr"}], [endpoints[0].id]
    return {"contract": c, "incoming": incoming, "pickup_event_ids": pickup_events, "released": released}


def _direct_contact_endpoint(ctx: FlowContext, before: dict, after: dict) -> dict:
    """Bind existing optical endpoint and qualified public pickup history to D2."""
    unresolved = {"status": "unresolved", "metrics": {}, "binding": {}, "reasons": [],
                  "scope": "geometric/recorded-layer interface only; no electrical or chemical claim"}
    try:
        from qumus_simlab.records.stamps import contact_film

        public = _direct_contact_public_binding(ctx.params, ctx.store, ctx.store.get_run(ctx.run_id),
            stage="endpoint", before=before)
        c = public["contract"]
        stamp = require_sample(ctx, ctx.params["stamp_id"], "stamp")
        incoming, pickup_events = public["incoming"], public["pickup_event_ids"]
        groups = after.get("observed_outlines") or []
        if len(groups) != 3 or any(g["material"] != "Gr" or len(g["outlines"]) != 1 for g in groups):
            raise ValueError("all three current target frames need one unambiguous complete Gr outline")
        actual_target = {"chip_id": after["target_chip_id"], "geometry": after.get("target_contact"),
            "receiving_layers_above_contacts": after.get("receiving_layers_above_contacts"),
            "before_observation_binding": {"run_id": ctx.run_id, "frames": before["frames"],
                "ledger_event_id": before["ledger_event_id"]}}
        results = [direct_contact_result({**g["outlines"][0], "flake_id": c.source_flake_id,
            "stamp_film": contact_film(stamp), "incoming_layers_top_first": incoming,
            "source_geometry_sha256": c.source_geometry_sha256, "observation_binding": g["observation_binding"]}, actual_target, c)
            for g in groups]
        result = deepcopy(next((r for r in results if r["status"] != "pass"), results[0]))
        result["binding"].update(run_id=ctx.run_id, stamp_id=stamp["id"], pickup_event_ids=pickup_events,
            target_before_sha256=_digest(before), terminal_frame_sha256=[f["sha256"] for f in after["frames"]])
        result["observations"] = results
        return result
    except (ValueError, TypeError, KeyError, OSError, NeedsHumanError) as exc:
        return {**unresolved, "reasons": [str(exc)[:400]]}


def _release_receipt(store, event_id):
    event = store.events_after(event_id-1, limit=1)[0]
    return {"event_id": event.id, "event_sha256": _digest(event.payload)}


def _release_publish_capture(ctx, phase, frame):
    _publish_transfer_analysis(ctx, phase["name"], frame)
    phase["release_budget"]["footprint_pending"] -= 1
    _release_exposure(ctx, phase, frame)


def _designed_loaded_detections(ctx, frame, detections, expected):
    """The designed materials on the loaded stamp, and any co-picked flakes.

    Jialei 2026-11-03: a small neighbour flake picked up with the main hBN is
    normal. The designed hBN is the one carried hBN that covers exactly the
    expected count of every other loaded material; those covered layers are
    the designed ones. A co-picked flake of a known material that stays off
    the designed layers is recorded, not loaded. Jialei 2026-11-03: the stamp
    must not carry any unplanned graphite to the target (a518 seed 7 lifted
    a second Gr with the bare PC film), so a co-picked flake other than hBN
    refuses the release before any contact. A co-picked flake on a designed
    layer, a missing material or no unique main hBN stays ambiguous.
    """
    from qumus_simlab.core.regions import region_covers, region_dict, region_intersection

    def count(materials):
        return {m: list(materials).count(m) for m in set(materials)}

    found = [d.material for d in detections]
    if count(found) == count(expected):
        return detections, []
    wanted = count(expected)
    if (not detections or set(found) != set(wanted) or wanted.get("hBN") != 1
            or any(found.count(m) < n for m, n in wanted.items())):
        raise NeedsHumanError("Fresh whole loaded material count is missing or ambiguous")
    try:
        regions = [(d, _complete_detection_region(ctx, frame, d)) for d in detections]
    except (NeedsHumanError, ValueError) as exc:
        raise NeedsHumanError("Fresh whole loaded material count is missing or ambiguous") from exc
    layers = [(d, r) for d, r in regions if d.material != "hBN"]
    choices = []
    for d, region in regions:
        if d.material != "hBN":
            continue
        covered = [(layer, r) for layer, r in layers if region_covers(region, r)]
        if count([layer.material for layer, _ in covered]) == {m: n for m, n in wanted.items() if m != "hBN"}:
            choices.append(((d, region), covered))
    if len(choices) != 1:
        raise NeedsHumanError("Fresh whole loaded material count is missing or ambiguous")
    (main, main_region), designed = choices[0]
    chosen = {id(main), *(id(layer) for layer, _ in designed)}
    extras = [(d, r) for d, r in regions if id(d) not in chosen]
    if any(region_intersection(r, layer).components for _, r in extras for _, layer in designed):
        raise NeedsHumanError("A co-picked carried flake overlaps a loaded layer")
    if any(d.material != "hBN" and region_intersection(r, main_region).components for d, r in extras):
        raise NeedsHumanError("A co-picked carried flake overlaps a loaded layer")
    if any(d.material != "hBN" for d, _ in extras):
        raise NeedsHumanError("The loaded stamp carries an unplanned non-hBN flake; release refused")
    return ([d for d in detections if id(d) in chosen],
            [{"material": d.material, "thickness_label": d.thickness_label, "region_mm": region_dict(r)}
             for d, r in extras])


def _acquire_release_footprint(ctx, phase):
    from qumus_simlab.core.optical_profiles import (
        RELEASE_SUPPORT_ASSUMPTIONS,
        STAMP_RECOGNITION_PROFILE,
    )

    if phase.get("release_footprint"):
        raise NeedsHumanError("A footprint observation cannot be repeated or cherry-picked")
    scope = ctx.gate.microscope(MICROSCOPE)
    held, objective, focus = (_pos(ctx), _held_source(ctx)), scope.get_objective(), scope.get_focus()
    if objective != 5 or phase["reference"]["meta"].get("focus_plane") != "chip":
        raise NeedsHumanError("Release requires its original fixed5x chip-plane reference")
    target = _transfer_state(ctx)["target_reference"]["focus_reference"]
    axis = target["focus_binding"]["observation"]["focus_axis_calibration"]
    delta = (held[0]["z"]-sum(phase["process_reference"]["reference_interval_mm"])/2)/axis["scale_mm_per_focus_mm"]
    # The parked target focus was observed with the stamp out of the light
    # path; the slide and PDMS now over the chip shift its apparent focus
    # (0.3 mm in the SIM, seed 8). Refocus the chip at 20x through the stamp
    # first, as the Gr pickup coverage does, and offset the film from that.
    sweep = ctx.verification["focus"]["coarse_points"]+ctx.verification["focus"]["fine_points"]
    count = 2*sweep+4
    duration = scope.focus_transition_duration_bound(min(focus, focus+delta)-.3, max(focus, focus+delta)+.3)
    if not _finite(duration) or duration < 0:
        raise NeedsHumanError("Fresh loaded footprint lacks a finite focus-work reserve")
    phase["release_budget"].update(stage="footprint", footprint_pending=count, footprint_focus_bound_s=duration)
    _release_budget_guard(ctx, phase)
    try:
        scope.set_objective(20)
        autofocus(ctx, 20, attempts=1, half_range=.15, plane="chip",
                  on_capture=lambda frame: _release_publish_capture(ctx, phase, frame))
        if ctx.store.get_run(ctx.run_id)["meta"]["last_focus_observation"].get("peak_observable") is not True:
            raise NeedsHumanError("Target chip has no observed 20x focus through the loaded stamp")
        seed = scope.get_focus()+delta
        scope.set_focus(seed)
        autofocus(ctx, 20, attempts=1, half_range=.15, plane="stamp",
                  on_capture=lambda frame: _release_publish_capture(ctx, phase, frame))
        focus_event = ctx.store.get_run(ctx.run_id)["meta"]["last_focus_event_id"]
        cursor = ctx.store.last_event_id()
        def charge_final(frame):
            phase["release_budget"]["footprint_pending"] -= 1
            _release_exposure(ctx, phase, frame)
        frame = microscope_frame(ctx, phase=phase["name"], on_capture=charge_final)
        # microscope_frame also publishes the same exposure; use the unique
        # publication made by its normal producer, not a second observation.
        record = _frame_record(frame)
        receipts = _contact_capture_receipts(ctx.store, ctx.run_id, phase["name"], record, cursor)
        noise = frame.meta["calibration"]["observation_condition"]["observation_noise_scale"]
        profile = frame.meta["calibration"].get("release_support_profile")
        if (profile != RELEASE_SUPPORT_ASSUMPTIONS or type(noise) not in (int, float)
                or not 0 <= noise <= 1 or frame.meta.get("focus_plane") != "stamp"
                or (analyzer(ctx).plane_sharpness(frame, "stamp") or 0.) < min_sharpness(ctx, 20)):
            raise NeedsHumanError("Fresh loaded footprint lacks its declared observable SIM profile")
        detections = analyzer(ctx).find_stamp_flakes(frame)
        expected = [ctx.params["materials"][f] for f in ctx.params["flake_ids"]]
        detections, extras = _designed_loaded_detections(ctx, frame, detections, expected)
        loaded = []
        for fid, material in zip(ctx.params["flake_ids"], expected, strict=True):
            candidates = [d for d in detections if d.material == material]
            if len(candidates) != 1:
                raise NeedsHumanError("Loaded material identity cannot be uniquely reconciled")
            detection = candidates[0]
            if getattr(analyzer(ctx), "recognition_backend", None) == "v1_lab_rgb":
                from qumus_simlab.core.regions import region_dict, region_vertices
                from qumus_simlab.perception.contact_coverage import InteriorCalibration
                from qumus_simlab.perception.measurement import owner_error_budget_policy
                from qumus_simlab.perception.observations import RGB_CARRIED_PROFILE

                labels = ctx.params.get("contact_interior_calibration", {}).get("carried", {}).get("labels_nm", [])
                if (detection.meta.get("recognition_profile") != RGB_CARRIED_PROFILE
                        or detection.meta.get("observation_plane") != "stamp"
                        or material == "hBN" and detection.thickness_label not in labels):
                    raise NeedsHumanError("Native loaded material lacks its actual bounded public classification")
                if material == "hBN":
                    InteriorCalibration(tuple(labels)).generators()
                chain = analyzer(ctx).persist_detection_masks(frame, detection)
                try:
                    region = _original_mask_region(chain)
                except ValueError as exc:
                    raise NeedsHumanError("Loaded original material mask is unsupported: "+str(exc)) from exc
                sx, sy = _image_calibration(frame).mm_per_pixel_x, _image_calibration(frame).mm_per_pixel_y
                if any(abs(x) >= frame.meta["calibration"]["image_size_px"][0]*abs(sx)/2
                       or abs(y) >= frame.meta["calibration"]["image_size_px"][1]*abs(sy)/2 for x, y in region_vertices(region)):
                    raise NeedsHumanError("Native loaded region touches the captured field boundary")
                loaded.append({"flake_id": fid, "material": material, "region_mm": region_dict(region),
                    "thickness_label": detection.thickness_label, "recognition_chain": chain,
                    "recognition_profile": RGB_CARRIED_PROFILE, "owner_error_budget": owner_error_budget_policy(),
                    "boundary_uncertainty_mm": .001})
                continue
            # The SIM carried hBN class is the one frozen at the Gr pickup
            # (a518 seed 2 carries a 10-20 nm hBN), as on the native path.
            labels = ctx.params.get("contact_interior_calibration", {}).get("carried", {}).get("labels_nm", [])
            if (detection.meta.get("recognition_profile") != STAMP_RECOGNITION_PROFILE
                    or detection.meta.get("centroid_sigma_mm") != .002*noise
                    or material == "hBN" and detection.thickness_label not in labels):
                raise NeedsHumanError("Unsupported complete-outline or bounded public hBN classification")
            outline = _complete_detection_outline(detection)
            sx, sy = _image_calibration(frame).mm_per_pixel_x, _image_calibration(frame).mm_per_pixel_y
            if any(abs(x) >= frame.meta["calibration"]["image_size_px"][0]*abs(sx)/2
                   or abs(y) >= frame.meta["calibration"]["image_size_px"][1]*abs(sy)/2 for x, y in outline):
                raise NeedsHumanError("Loaded outline touches the native20 field boundary")
            loaded.append({"flake_id": fid, "material": material, "outline_mm": outline,
                           "thickness_label": detection.thickness_label})
        body = {"policy": OBSERVED_RELEASE_POLICY_ID, "stamp_id": ctx.params["stamp_id"],
            "load_ids": list(ctx.params["flake_ids"]), "selection_sha256": _digest({k: ctx.params.get(k) for k in _WITNESS_SELECTION_FIELDS}),
            "witness_sha256": _digest(phase["release_reference"]["witness"]),
            "design_contract_sha256": _digest(ctx.params.get("device_design") or ctx.params.get("direct_contact_contract")),
            "classification_sha256": _digest(ctx.params.get("contact_interior_calibration", {})),
            "pose_window_admission_sha256": _digest(ctx.params.get("design_pose_window_admission")), "frame": record,
            "receipts": receipts, "focus_event": _release_receipt(ctx.store, focus_event), "loaded": loaded,
            "co_picked_flakes": extras,
            "profile": profile, "noise_scale": noise, "footprint_captures": 1,
            "axis": scope.focus_axis_calibration(20),
            "optical_commands": [_release_receipt(ctx.store, e.id) for e in
                _transfer_events(ctx.store, ctx.run_id, "device.command", phase["reference_event"]["event_id"])
                if e.payload.get("device") == MICROSCOPE]}
        eid = ctx.store.emit("stack.release_footprint", body, run_id=ctx.run_id)
        phase["release_footprint"] = _release_receipt(ctx.store, eid)
        _save_phase(ctx, phase)
    finally:
        scope.set_focus(focus)
        scope.set_objective(objective)
        phase["release_budget"].update(stage="entry_heat", footprint_pending=0)
        _save_phase(ctx, phase)
    if held != (_pos(ctx), _held_source(ctx)):
        raise NeedsHumanError("Holder or target moved during the one loaded-footprint observation")


def _freeze_release_approach(ctx, phase):
    footprint = _historical_event(ctx.store, phase["release_footprint"], ctx.run_id, "stack.release_footprint")
    commands = [e for e in _transfer_events(ctx.store, None, "device.command", footprint.id)
                if e.payload.get("device") == HEATER]
    if (not commands or any(e.run_id != ctx.run_id or e.payload.get("command") != "set_target"
            or not _finite(e.payload.get("params", {}).get("celsius"))
            or not 0 < e.payload["params"]["celsius"] <= 155.5 for e in commands)):
        raise NeedsHumanError("Release entry heat lacks its original bounded owned command prefix")
    body = {"phase": phase["name"], "footprint": phase["release_footprint"],
            "entry_heat_commands": [_release_receipt(ctx.store, e.id) for e in commands],
            "reference": phase["reference_event"], "at": ctx.clock.now()}
    eid = ctx.store.emit("stack.release_approach", body, run_id=ctx.run_id)
    phase["release_approach"] = _release_receipt(ctx.store, eid)
    _save_phase(ctx, phase)


def _release_journal_rows(store, phase, last):
    """Read the original GLOBAL interval; never filter away another job's work.

    The interval also keeps the original rows published at the release Step's
    own Clock instant just before it started: the twin publishes an encoder
    frame only when state changes, so the reference capture's same-clock
    encoder can be the previous Step's final readback at that instant.
    """
    first = phase["release_reference"]["witness"]["load"]["release_start"]["event_id"]
    start = store.events_after(first-1, limit=1)
    while start and first > 1 and (prior := store.events_after(first-2, limit=1)) and prior[0].ts == start[0].ts:
        first -= 1
    rows, after = [], first-1
    while after < last:
        batch = store.events_after(after, limit=1000)
        if not batch:
            break
        rows.extend(e.model_dump() for e in batch if e.id <= last)
        after = batch[-1].id
    return rows



def _release_support_result(store, run_id, phase, current, current_receipts):
    from qumus_simlab.core.optical_profiles import RELEASE_SUPPORT_ASSUMPTIONS
    from qumus_simlab.perception.contact_coverage import (
        InteriorCalibration,
        release_load_transport,
        release_support_evidence,
    )

    run = store.get_run(run_id)
    raw = run["params"]
    footprint_event = _historical_event(store, phase["release_footprint"], run_id, "stack.release_footprint")
    body = footprint_event.payload
    if (body["policy"] != OBSERVED_RELEASE_POLICY_ID or _fresh_release_policy(raw) != OBSERVED_RELEASE_POLICY_ID
            or body["stamp_id"] != raw["stamp_id"] or body["load_ids"] != raw["flake_ids"]
            or body["selection_sha256"] != _digest({k: raw.get(k) for k in _WITNESS_SELECTION_FIELDS})
            or body["witness_sha256"] != _digest(phase["release_reference"]["witness"])
            or body["design_contract_sha256"] != _digest(raw.get("device_design") or raw.get("direct_contact_contract"))
            or body["classification_sha256"] != _digest(raw.get("contact_interior_calibration", {}))
            or body["pose_window_admission_sha256"] != _digest(raw.get("design_pose_window_admission"))
            or body["footprint_captures"] != 1 or body["profile"] != RELEASE_SUPPORT_ASSUMPTIONS
            or [b["flake_id"] for b in body["loaded"]] != raw["flake_ids"]):
        raise ValueError("Fresh observed load differs from its original required history")
    duplicates = list(_transfer_events(store, run_id, "stack.release_footprint", phase["reference_event"]["event_id"]))
    if len(duplicates) != 1 or duplicates[0].id != footprint_event.id:
        raise ValueError("Footprint capture cap exceeded or observation substituted")
    frame = body["frame"]
    _contact_public_capture(store, run_id, phase["name"], frame, body["receipts"])
    focus = _historical_event(store, body["focus_event"], run_id, "microscope.focus")
    if (focus.id >= body["receipts"]["capture"]["event_id"] or focus.payload["objective"] != 20
            or focus.payload["focus_plane"] != "stamp" or frame["meta"]["objective"] != 20
            or body["axis"] != focus.payload["focus_axis_calibration"]
            or focus.payload["final_focus_readback_z_mm"] != frame["meta"]["focus_z"]
            or focus.payload["holder_pose"] != _contact_public_capture(store, run_id, phase["name"], frame, body["receipts"])[0]
            or focus.payload["stage_kinematics"] != frame["meta"]["stage_kinematics"]
            or frame["meta"]["calibration"].get("release_support_profile") != body["profile"]):
        raise ValueError("Footprint focus/capture/model binding changed")
    if len(body["loaded"]) == 2:
        frozen = raw["contact_interior_calibration"]["carried"]
        if body["loaded"][0]["thickness_label"] not in frozen["labels_nm"]:
            raise ValueError("Release lost its original bounded public hBN class")
    native = any("region_mm" in item for item in body["loaded"])
    if native:
        from qumus_simlab.perception.measurement import (
            validate_measurement_receipt,
            validate_owner_error_budget,
        )
        from qumus_simlab.perception.observations import RGB_CARRIED_PROFILE

        for item in body["loaded"]:
            validate_owner_error_budget(item)
            if "recognition_chain" in item:
                from qumus_simlab.core.regions import region_dict

                chain = item["recognition_chain"]
                region = region_dict(_original_mask_region(chain))
                if (item.get("recognition_profile") != RGB_CARRIED_PROFILE or chain["material"] != item["material"]
                        or chain["image"]["objective"] != 20 or chain["image"]["sha256"] != frame["sha256"]
                        or region != item["region_mm"]):
                    raise ValueError("Release footprint differs from its original selected mask")
                continue
            # Historical derived-RGB receipts retain their original validator.
            receipt = validate_measurement_receipt(item["measurement_receipt"])
            chain = receipt["original_recognition_chain"]
            if (item.get("recognition_profile") != RGB_CARRIED_PROFILE or chain["material"] != item["material"]
                    or chain["image"]["objective"] != 20 or chain["image"]["sha256"] != frame["sha256"]
                    or receipt["measurement"]["status"] != "measured"
                    or receipt["measurement"]["region_exposure_mm"] != item["region_mm"]):
                raise ValueError("Native release footprint differs from its original measured RGB region")
    ref = phase["reference"]
    reference_receipts = _contact_capture_receipts(store, run_id, phase["name"], ref, phase["reference_event"]["event_id"]-1)
    coverage = {"reference": ref, "current": current, "reference_receipts": reference_receipts,
                "current_receipts": current_receipts}
    support = _contact_support_context(store, run_id, phase, coverage)
    from qumus_simlab.records.release_support import release_journal_certificate

    journal = release_journal_certificate(
        _release_journal_rows(store, phase, current_receipts["capture"]["event_id"]), run_id=run_id,
        phase=phase, params=raw, current=current, current_receipts=current_receipts, digest=_digest)
    transports = []
    for key, record in (("footprint_to_reference", ref), ("footprint_to_current", current)):
        prefix = journal[key]
        transports.append(release_load_transport(axis=body["axis"], elapsed_s=prefix["elapsed_s"],
            z_variation_mm=prefix["z_total_variation_mm"], noise_scale=body["noise_scale"],
            footprint_size_mm=frame["meta"]["fov_mm"], camera_size_mm=record["meta"]["fov_mm"]))
    variation = journal["reference_to_current"]["z_total_variation_mm"]
    axis = body["axis"]
    drift = min(2*math.sqrt(2)*axis["apex_drift_clip_per_axis_mm"],
        axis["apex_lateral_drift_speed_norm_bound_mm_s"]*(current["captured_at"]-ref["captured_at"])
        +axis["apex_approach_drift_norm_bound_mm_per_mm"]*variation)
    background = axis["source_refraction_projection_norm_bound_mm_per_mm"]*(
        abs(support["current_z_mm"]-support["reference_z_mm"])+(1+axis["lateral_focus_slope_norm_bound_mm_per_mm"])*drift)
    if background > 2*min(abs(v) for v in ref["meta"]["calibration"]["mm_per_pixel"]):
        raise ValueError("Release wafer refraction exceeds its fixed5x5 native stencil")
    # The target-before native20 parked exposure is already independently
    # validated by _release_reference; its complete chip support must not cross
    # the declared refraction branch during this stroke.
    target = _transfer_state(SimpleNamespace(store=store, run_id=run_id))["target_reference"]["focus_reference"]
    from qumus_simlab.perception.contact_coverage import parked_chip_support
    chip = parked_chip_support(_reference(target), _image_calibration(_reference(target)))
    if not chip["found"]:
        raise ValueError("Release target has no public stationary-chip support")
    labels = tuple(raw["contact_interior_calibration"]["carried"]["labels_nm"])
    optical = InteriorCalibration(labels) if len(body["loaded"]) == 2 else InteriorCalibration()
    result = release_support_evidence(_reference(ref), _reference(current), _image_calibration(_reference(ref)),
        body["loaded"], optical=optical, reference_transport=transports[0], current_transport=transports[1], profile=body["profile"],
        unknown_flakes=body.get("co_picked_flakes", []))
    front = result["front"]
    if front.get("film_intact") is False:
        raise ValueError("Visible release-film damage is not an admissible support prior")
    branch = {"admitted": False, "reason": "No complete observed current inner front"}
    bounds = front.get("front_bounds_px")
    if front.get("has_contact") and bounds and (front.get("angular_coverage", 0) == 1.
                                                or front.get("front_bounds_from_fitted_circle") is True):
        calibration = _image_calibration(_reference(current))
        sx, sy = calibration.mm_per_pixel_x, calibration.mm_per_pixel_y
        width, height = current["meta"]["calibration"]["image_size_px"]
        fc = [((x-width/2)*sx, (y-height/2)*sy) for x in (bounds[0], bounds[2]) for y in (bounds[1], bounds[3])]
        box = chip["possible_chip_point_box_mm"]
        sc = [(x, y) for x in (box[0], box[2]) for y in (box[1], box[3])]
        uncertainty = front["interface_uncertainty_px"]*max(abs(sx), abs(sy))+math.hypot(sx, sy)
        domain = max(math.dist(a, b) for a in fc for b in sc)+uncertainty+drift
        if front.get("front_bounds_from_fitted_circle") is True:
            # The whole fitted circle (a far-side clipped front, D13): its
            # farthest point from a chip corner is centre distance + radius.
            centre = ((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2)
            radius_mm = (bounds[2]-bounds[0])/2*max(abs(sx), abs(sy))
            centre_mm = ((centre[0]-width/2)*sx, (centre[1]-height/2)*sy)
            domain = max(math.dist(centre_mm, b) for b in sc)+radius_mm+uncertainty+drift
        branch = {"admitted": domain < axis["dome_curvature_radius_mm"],
                  "source_branch_distance_bound_mm": domain, "dome_radius_mm": axis["dome_curvature_radius_mm"]}
        if not branch["admitted"]:
            raise ValueError("Target/refraction support can cross the undeclared surface branch")
    if not branch["admitted"]:
        result["conditional_modeled_support"] = False
    return {"footprint": phase["release_footprint"], "reference": ref, "current": current,
            "reference_receipts": reference_receipts, "current_receipts": current_receipts,
            "journal": journal, "transports": transports, "support_context": support, "target_chip_support": chip,
            "background_bound_mm": background, "surface_branch": branch, "result": result}


def _observe_release_support(ctx, phase, frame):
    current = _frame_record(frame)
    receipts = _contact_capture_receipts(ctx.store, ctx.run_id, phase["name"], current, phase["reference_event"]["event_id"])
    try:
        body = _release_support_result(ctx.store, ctx.run_id, phase, current, receipts)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise NeedsHumanError("Release support lost its original loaded/optical/trajectory binding") from exc
    command = phase["last_intent"]
    body["actual_command_z_mm"] = command["to_mm"]
    eid = ctx.store.emit("stack.release_support", body, run_id=ctx.run_id)
    if body["result"]["conditional_modeled_support"]:
        phase["accepted_release_support"] = _release_receipt(ctx.store, eid)
        _save_phase(ctx, phase)
        return True
    return False


def _validate_release_support(store, run_id, phase):
    event = _historical_event(store, phase["accepted_release_support"], run_id, "stack.release_support")
    body = event.payload
    recomputed = _release_support_result(store, run_id, phase, body["current"], body["current_receipts"])
    command = body["actual_command_z_mm"]
    if {k: v for k, v in body.items() if k != "actual_command_z_mm"} != recomputed:
        raise ValueError("Original release masks/conditional support no longer match the actual evidence")
    if not _finite(command) or not body["result"]["conditional_modeled_support"]:
        raise ValueError("No positive original conditional support")
    commands = [e for e in _transfer_events(store, run_id, "device.command", phase["release_approach"]["event_id"])
                if e.id < body["current_receipts"]["capture"]["event_id"] and e.payload.get("device") == MANIP]
    if (not commands or commands[-1].payload.get("params", {}).get("value") != command
            or commands[-1].payload["params"].get("axis") != "z"):
        raise ValueError("Accepted support is not the original actual Z command exposure")
    return body


class DropDown(Step):
    """Press the stack onto the target, heat until the PC film melts, lift
    (v1 MoleculeDropDown); then every layer must be on the target and the
    stamp empty."""

    name = "v1_drop_down"
    devices = (MANIP, MICROSCOPE, HEATER)
    safe_to_resume = False  # hot film half-released: a human decides (spec 8.2)

    def __init__(self, stamp_id: str, flake_ids: list[str], materials: dict[str, str],
                 target_chip: str, target_xy_mm: tuple[float, float], melt_c: float,
                 design_path: str | None) -> None:
        self.stamp_id, self.flake_ids, self.materials = stamp_id, flake_ids, materials
        self.target_chip, self.target_xy, self.melt_c = target_chip, target_xy_mm, melt_c
        self.design_path = design_path
        self._peak_c = 0.0

    def _cfg(self, ctx: FlowContext) -> dict[str, Any]:
        return ctx.verification["drop"]

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        return {"layers": [self.materials[f] for f in self.flake_ids], "method": "target_material_appearance",
                "melt_c": self.melt_c, "radius_um": self._cfg(ctx)["stack_radius_um"]}

    def preconditions(self, ctx):
        _profile(ctx)
        self.max_corrections = 0
        if not _over_stage(ctx):
            raise PreconditionFailed("The stamp is not over the stage")

    def act(self, ctx, correction):
        c, h = _profile(ctx), ctx.verification["heating"]
        if ctx.params.get("device_design"):
            _device_design_public_binding(ctx.params, ctx.store, ctx.store.get_run(ctx.run_id))
        if ctx.params.get("release_policy") == V1_DROP_POLICY_ID:
            return self._v1_drop(ctx)
        try:
            before = _target_baseline(_transfer_state(ctx), ctx.run_id, ctx.store, self.target_chip, self.target_xy)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise NeedsHumanError("Release lacks its actual requested-target baseline") from exc
        if ctx.params.get("direct_contact_contract"):
            try:
                _direct_contact_public_binding(ctx.params, ctx.store, ctx.store.get_run(ctx.run_id),
                    stage="release", before=before)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise NeedsHumanError("Frozen direct-contact source/receiver binding changed before release") from exc
        if any(before["materials_found"].values()):
            raise NeedsHumanError("Expected material already appears in the target baseline; appearance is ambiguous")
        if before["stage_kinematics"] != _held_source(ctx):
            raise NeedsHumanError("Requested target moved after its baseline")
        try:
            process, reference = _release_reference(_transfer_state(ctx), ctx.params, ctx.run_id,
                                                   ctx.verification, ctx.store)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise NeedsHumanError("Release lacks its admitted target process reference") from exc
        lower = ctx.gate.manipulator_z_floor(MANIP)
        bounds = ctx.gate.limits.manipulators[MANIP].z
        if not max(bounds[0], lower or bounds[0]) <= reference["requested_z_mm"] < _pos(ctx)["z"] <= bounds[1]:
            raise NeedsHumanError("Target-relative release is outside its descending envelope; no clamp applied")
        phase = _begin_phase(ctx, self.name, difference=False)
        if phase.get("release_reference", reference) != reference or phase.get("process_reference", process) != process:
            raise NeedsHumanError("Saved target-relative release plan changed")
        phase.update(release_policy=ctx.params["release_policy"], release_reference=reference, process_reference=process)
        if phase.get("complete"):
            self._peak_c = phase["peak_c"]
            return
        if phase.get("observations", 0) > 1:
            raise NeedsHumanError("Interrupted release requires review; no blind thermal/press retry")
        if "release_budget" in phase and phase["release_budget"].get("stage") != "entry_heat":
            raise NeedsHumanError("Unexecuted release cannot inherit a terminal transport stage")
        if "release_budget" not in phase:
            phase["release_budget"] = {"stage": "entry_heat", "start_event_id": phase["reference_event"]["event_id"],
                "endpoint_limit": ctx.verification["focus"]["coarse_points"]+ctx.verification["focus"]["fine_points"]+6,
                "endpoint_observations": 0, "park_observations": 0, "thermal_observations": 0,
                "park_complete": False}
        _release_budget_guard(ctx, phase)
        if ctx.params["release_policy"] == OBSERVED_RELEASE_POLICY_ID:
            _acquire_release_footprint(ctx, phase)
        heater = ctx.gate.heater(HEATER)
        def observe():
            _phase_capture(ctx, phase)
        def interval():
            allowance = _release_budget_guard(ctx, phase, consume=1)
            age = ctx.clock.now()-phase["observed_at"]
            deadline = c["max_unobserved_s"]-c["capture_reserve_s"]
            if age >= deadline:
                observe()
                allowance = _release_budget_guard(ctx, phase, consume=1)
                age = ctx.clock.now()-phase["observed_at"]
            remaining = min(h["poll_s"], deadline-age, allowance)
            if remaining <= 1e-8:
                raise NeedsHumanError("Release exhausted its remaining thermal allowance")
            return remaining
        def wait_for(test, what):
            wait_temperature(ctx, HEATER, test, what=what, timeout_s=h["timeout_s"],
                stuck_window_s=h["stuck_window_s"], stuck_delta_c=h["stuck_delta_c"],
                poll_s=h["poll_s"], observe=observe, observation_interval=interval)
        mica = ctx.params.get("stamp_film") == "mica"
        entry = ctx.params.get("drop_temp_c", 155.) if mica else 155.
        release = self.melt_c if mica else 170.
        if mica and not ctx.params.get("support_assumptions"):
            raise PreconditionFailed("Mica release needs its explicit supported synthetic recipe")
        try:
            ramp_heater(ctx, HEATER, entry+c["release_setpoint_headroom_c"], observe=observe,
                        observation_interval=interval)
            wait_for(lambda t: t >= entry, f">={entry} C release entry")
            phase["release_budget"]["stage"] = "press"
            if ctx.params["release_policy"] == OBSERVED_RELEASE_POLICY_ID:
                _freeze_release_approach(ctx, phase)
                if not _guarded_z(ctx, phase, reference["requested_z_mm"], speed=c["drop_speed_mm_s"],
                                  observe=lambda frame: _observe_release_support(ctx, phase, frame)):
                    raise NeedsHumanError("No conditional loaded support within the original release floor/budget")
            else:
                _guarded_z(ctx, phase, reference["requested_z_mm"], speed=c["drop_speed_mm_s"])
            phase["release_budget"]["stage"] = "release_heat"
            ramp_heater(ctx, HEATER, release+c["release_setpoint_headroom_c"], observe=observe,
                        observation_interval=interval)
            wait_for((lambda t: t >= release-.25) if mica else (lambda t: t > release),
                     f"configured support release at {release} C" if mica else ">170 C PC release")
            self._peak_c = heater.read_temperature()
            # Historical +5mm is a withdrawal intent, mapped to the current
            # declared clear preset rather than copied across encoder frames.
            withdrawal = {"historical_delta_mm": 5., "requested_sim_target_mm": _ta(ctx)["safe_z"],
                          "mapping": "historical-withdrawal-to-current-clear-preset",
                          "source": "synthetic", "from_readback": _pos(ctx)}
            phase["release_budget"]["stage"] = "withdraw"
            _guarded_z(ctx, phase, _ta(ctx)["safe_z"])
            withdrawal["actual_readback"] = _pos(ctx)
            phase["withdrawal"], phase["peak_c"] = withdrawal, self._peak_c
            _save_phase(ctx, phase)
        finally:
            heater.off()
        if heater.get_target() is not None:
            raise NeedsHumanError("Release output shutdown is unconfirmed")
        phase["outputs_off"], phase["complete"] = True, True
        phase["release_budget"]["stage"] = "off"
        _release_budget_guard(ctx, phase)
        _save_phase(ctx, phase)

    def _v1_drop(self, ctx: FlowContext) -> None:
        """v1 molecule_drop_down, step for step (#204).

        v1 first initializes the stamp over the target (Z 5.4 at 0.1 mm/s),
        heats with an autofocus every 10 s until 155 C, moves Z to 4.9 at
        0.02 mm/s, waits until the stage is above the PC melt (170 C) and
        lifts 5 mm at 0.1 mm/s, then turns the heater off. The holder Z
        commands are still observed on the v2 cadence (one frame per 25 um
        or 2 s); v1 has no release or film checks, and neither has this.
        """
        c, h = _profile(ctx), ctx.verification["heating"]
        phase = _begin_phase(ctx, self.name, difference=False)
        if phase.get("complete"):
            self._peak_c = phase["peak_c"]
            return
        if phase.get("observations", 0) > 1:
            raise NeedsHumanError("Interrupted release requires review; no blind thermal/press retry")
        phase["release_policy"] = V1_DROP_POLICY_ID
        _save_phase(ctx, phase)
        lower = ctx.gate.manipulator_z_floor(MANIP)
        if lower is not None and c["drop_z_mm"] < lower:
            raise NeedsHumanError("The v1 drop height is below the current physical floor; no clamp applied")
        heater, period = ctx.gate.heater(HEATER), c["drop_check_period_s"]
        last_focus = [-math.inf]
        def focus_and_photo():
            # v1: autofocus once per temperature check while heating
            if ctx.clock.now()-last_focus[0] >= period-1e-9:
                autofocus(ctx, 5, attempts=1, half_range=c["post_phase_focus_half_range_mm"], plane="chip")
                last_focus[0] = ctx.clock.now()
                _phase_capture(ctx, phase)
        def wait_for(test, what, observe=None):
            wait_temperature(ctx, HEATER, test, what=what, timeout_s=h["timeout_s"],
                stuck_window_s=h["stuck_window_s"], stuck_delta_c=h["stuck_delta_c"],
                poll_s=period, observe=observe)
        try:
            if _pos(ctx)["z"] > c["initialize_z_mm"]:
                _guarded_z(ctx, phase, c["initialize_z_mm"], speed=c["initialize_speed_mm_s"])
            focus_and_photo()
            ramp_heater(ctx, HEATER, c["drop_entry_c"]+c["release_setpoint_headroom_c"], poll_s=period,
                        observe=focus_and_photo)
            wait_for(lambda t: t >= c["drop_entry_c"], f">={c['drop_entry_c']} C drop-down temperature",
                     observe=focus_and_photo)
            _phase_capture(ctx, phase)
            _guarded_z(ctx, phase, c["drop_z_mm"], speed=c["drop_speed_mm_s"])
            phase["drop_readback"] = _pos(ctx)
            _save_phase(ctx, phase)
            ramp_heater(ctx, HEATER, c["drop_melt_c"]+c["release_setpoint_headroom_c"], poll_s=period)
            wait_for(lambda t: t > c["drop_melt_c"], f">{c['drop_melt_c']} C PC melt")
            self._peak_c = heater.read_temperature()
            _phase_capture(ctx, phase)
            _guarded_z(ctx, phase, _pos(ctx)["z"]+c["drop_lift_mm"], speed=c["drop_lift_speed_mm_s"])
            phase["withdrawal"] = {"historical_delta_mm": c["drop_lift_mm"], "actual_readback": _pos(ctx)}
            phase["peak_c"] = self._peak_c
            _save_phase(ctx, phase)
        finally:
            heater.off()
        if heater.get_target() is not None:
            raise NeedsHumanError("Release output shutdown is unconfirmed")
        phase["outputs_off"], phase["complete"] = True, True
        _save_phase(ctx, phase)

    def _v1_archive(self, ctx: FlowContext, phase: dict) -> Observation:
        """After the v1 drop the target is photographed for the archive only."""
        parked = ManipulatorTo("park", name=self.name+"_photo_park", stamp_id=self.stamp_id)
        parked.act(ctx, None)
        if parked.verify(ctx).verdict != Verdict.OK:
            raise NeedsHumanError("The stamp did not park clear of the target for the archive photos")
        scope = ctx.gate.microscope(MICROSCOPE)
        photos = []
        for objective in (5, 20):
            scope.set_objective(objective)
            autofocus(ctx, objective, attempts=1, plane="chip")
            photos.append(_frame_record(scope.capture()))
        scope.set_objective(5)
        m = {"profile": TRANSFER_PROFILE_ID, "method": "v1_photo_archive", "policy": V1_DROP_POLICY_ID,
             "peak_c": round(self._peak_c, 1), "drop_readback": phase.get("drop_readback"),
             "withdrawal": phase.get("withdrawal"), "photos": photos, "release_confirmed": None,
             "scope": "v1 has no after-drop check: photos are archived, nothing is certified"}
        phase["endpoint"] = m
        _save_phase(ctx, phase)
        return Observation(Verdict.OK, m, self.expected(ctx), images=[p["image"] for p in photos],
                           message="Dropped the v1 way; the target photos are archived, release is not checked")

    def verify(self, ctx: FlowContext) -> Observation:
        phase = _transfer_state(ctx).get("phases", {}).get(self.name, {})
        if not phase.get("complete"):
            return Observation(Verdict.UNCLEAR, {"release_confirmed": False}, self.expected(ctx),
                               message="Thermal/preset/lift sequence incomplete; endpoint cannot certify execution")
        self._peak_c = phase["peak_c"]
        if ctx.params.get("release_policy") == V1_DROP_POLICY_ID:
            if phase.get("endpoint"):
                raise NeedsHumanError("The after-drop photos were already taken; no automatic replay")
            return self._v1_archive(ctx, phase)
        expected = sorted({self.materials[fid] for fid in self.flake_ids})
        try:
            before = _target_baseline(_transfer_state(ctx), ctx.run_id, ctx.store, self.target_chip, self.target_xy)
        except (KeyError, TypeError, ValueError, OSError):
            return Observation(Verdict.UNCLEAR, {"release_confirmed": False}, self.expected(ctx),
                               message="Target appearance lacks its retained verified before image")
        if phase.get("release_policy") != ctx.params["release_policy"] or phase["release_budget"].get("endpoint_status"):
            raise NeedsHumanError("Release endpoint is interrupted or already observed; no automatic replay")
        if ctx.gate.heater(HEATER).get_target() is not None:
            raise NeedsHumanError("Release endpoint needs confirmed heater-off")
        budget = phase["release_budget"]
        budget.update(stage="park", endpoint_status="started")
        _save_phase(ctx, phase)
        _release_budget_guard(ctx, phase)
        parked = ManipulatorTo("park", name=self.name+"_endpoint_park", stamp_id=self.stamp_id)
        parked.act(ctx, None, phase=phase)
        if parked.verify(ctx).verdict != Verdict.OK or before["stage_kinematics"] != _held_source(ctx):
            raise NeedsHumanError("Release endpoint park or held wafer context is unconfirmed")
        phase["park"] = {"requested_pose": parked._target(ctx), "actual_holder_pose": _pos(ctx),
                         "completed_at": ctx.clock.now()}
        budget.update(park_complete=True, stage="endpoint", endpoint_start_event_id=ctx.store.last_event_id())
        _release_budget_guard(ctx, phase)
        try:
            after = _target_view(ctx, self.target_chip, self.target_xy, expected,
                self.name+"_target_after", parked=True, focus_seed=before["focus_reference"],
                on_capture=lambda frame: _release_exposure(ctx, phase, frame, endpoint=True))
        except NeedsHumanError as exc:
            if not exc.details.get("optical_observation_unavailable"):
                raise
            budget["endpoint_status"] = "unavailable"
            _save_phase(ctx, phase)
            return Observation(Verdict.UNCLEAR, {"release_confirmed": False,
                "method": "target_material_appearance", "target_observation_unavailable": exc.message},
                self.expected(ctx), message="Target appearance is optically unobservable; review")
        budget.update(endpoint_status="observed", stage="observed", endpoint_end_event_id=ctx.store.last_event_id())
        _release_budget_guard(ctx, phase)
        _save_phase(ctx, phase)
        qualified = (before["profile"] == TRANSFER_PROFILE_ID and before["run_id"] == ctx.run_id
            and before["target_chip_id"] == after["target_chip_id"]
            and before["target_xy_mm"] == after["target_xy_mm"]
            and before["stage_kinematics"] == after["stage_kinematics"]
            and before["focus_reference"]["meta"]["calibration"] == after["focus_reference"]["meta"]["calibration"]
            and any(f["sha256"] not in {old["sha256"] for old in before["frames"]} for f in after["frames"])
            and before["frames"][-1]["captured_at"] < phase["started_at"] <= after["frames"][0]["captured_at"])
        appeared = {mat: before["materials_found"].get(mat) == 0 and after["materials_found"].get(mat, 0) > 0
                    for mat in expected}
        confirmed = qualified and all(appeared.values())
        contact_result = None
        if ctx.params.get("direct_contact_contract"):
            contact_result = _direct_contact_endpoint(ctx, before, after)
            confirmed = confirmed and contact_result["status"] == "pass"
            ctx.store.emit("stack.direct_contact_result", contact_result, run_id=ctx.run_id)
        design_result = None
        if ctx.params.get("device_design"):
            design_result = _device_design_endpoint(ctx, before, after)
            confirmed = confirmed and design_result["status"] == "pass"
            ctx.store.emit("stack.device_design_result", design_result, run_id=ctx.run_id)
        m = {"profile": TRANSFER_PROFILE_ID, "method": "target_material_appearance",
            "target_before_sha256": _digest(before), "target_after": after,
            "materials_appeared": appeared, "layers_found": after["materials_found"],
            "peak_c": round(self._peak_c, 1), "stamp_evaluated": False, "stamp_flakes_seen": None,
            "release_confirmed": confirmed, "process_confirmation": confirmed,
            "full_identity_completeness_observed": False, "registry_chemistry_observed": False,
            "scope": "Owner visual process criterion; material types appear at requested target, counts diagnostic only"}
        if contact_result is not None:
            m["direct_contact_result"] = contact_result
        if design_result is not None:
            m["device_design_result"] = design_result
        phase["endpoint"] = m
        _save_phase(ctx, phase)
        return Observation(Verdict.OK if confirmed else Verdict.UNCLEAR, m, self.expected(ctx),
            images=[f["image"] for f in before["frames"]+after["frames"]],
            message="Expected material types appeared at the qualified target ROI" if confirmed else
            "Target material appearance is missing or ambiguous; no automatic re-press")

    def _admit_release_commit(self, ctx: FlowContext, obs: Observation | None) -> None:
        try:
            _target_baseline(_transfer_state(ctx), ctx.run_id, ctx.store, self.target_chip, self.target_xy)
            process, reference = _release_reference(_transfer_state(ctx), ctx.params, ctx.run_id,
                                                   ctx.verification, ctx.store)
            saved = _transfer_state(ctx)["phases"][self.name]
            if saved["release_reference"] != reference or saved["process_reference"] != process:
                raise ValueError("Release process plan changed")
            _release_execution_binding(ctx.store, ctx.run_id, saved)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise NeedsHumanError("Release ledger requires current observable endpoint confirmation") from exc
        if (obs is None or not obs.measurements.get("release_confirmed")
                or not obs.measurements.get("process_confirmation")
                or obs.measurements.get("method") != "target_material_appearance"
                or obs.measurements != _transfer_state(ctx).get("phases", {}).get(self.name, {}).get("endpoint")
                or not _visual_release_proof(obs.measurements, _transfer_state(ctx).get("target_reference"),
                    _transfer_state(ctx).get("phases", {}).get(self.name, {}),
                    sorted({self.materials[fid] for fid in self.flake_ids}), ctx.run_id, ctx.store)):
            raise NeedsHumanError("Release ledger requires current observable endpoint confirmation")
        phase = _transfer_state(ctx)["phases"][self.name]
        after_id = obs.measurements["target_after"]["focus_reference"]["focus_binding"]["event_id"]
        phase["endpoint_verified"] = _bind_verified_endpoint(ctx, self.name, obs, after_id)
        _save_phase(ctx, phase)
    def _film_left_on_chip(self, ctx: FlowContext) -> bool:
        """A PC film melts onto the target and stays there (#180); mica does not."""
        return ctx.params.get("stamp_film") != "mica"

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        film_left = self._film_left_on_chip(ctx)
        if ctx.params.get("release_policy") == V1_DROP_POLICY_ID:
            phase = _transfer_state(ctx)["phases"].get(self.name, {})
            if obs is None or obs.measurements != phase.get("endpoint"):
                raise NeedsHumanError("The device record needs the archived after-drop photos")
            self._record_device(ctx, obs, film_left, None, None)
            return
        self._admit_release_commit(ctx, obs)
        if ctx.params.get("device_design"):
            result = _device_design_endpoint(ctx, _transfer_state(ctx)["target_reference"], obs.measurements["target_after"])
            if result["status"] != "pass" or result != obs.measurements.get("device_design_result"):
                raise NeedsHumanError("Design release requires its current complete all-contact observation binding")
        contact = None
        if ctx.params.get("direct_contact_contract"):
            before = _target_baseline(_transfer_state(ctx), ctx.run_id, ctx.store, self.target_chip, self.target_xy)
            contact = _direct_contact_endpoint(ctx, before, obs.measurements["target_after"])
            if contact["status"] != "pass" or contact != obs.measurements.get("direct_contact_result"):
                raise NeedsHumanError("Direct-contact receipt requires the current complete actual endpoint binding")
        design = obs.measurements["device_design_result"] if ctx.params.get("device_design") else None
        self._record_device(ctx, obs, film_left, contact, design)

    def _record_device(self, ctx, obs, film_left, contact, design_result):
        n = len(ctx.store.list_samples(kind="device")) + 1
        device_id = f"device_{n:03d}"
        ctx.store.upsert_sample(device_id, "device", location=self.target_chip, status="stacked",
                                props={"layers_top_first": self.flake_ids,
                                       "materials": [self.materials[f] for f in self.flake_ids],
                                       "target_xy_mm": list(self.target_xy),
                                       "design_path": self.design_path, "stamp": self.stamp_id,
                                       **({"device_design": ctx.params["device_design"], "device_design_result": design_result,
                                           "run_id": ctx.run_id} if ctx.params.get("device_design") else {}),
                                       **({"direct_contact_contract": ctx.params["direct_contact_contract"],
                                           "direct_contact_result": contact, "run_id": ctx.run_id} if contact else {}),
                                       "images": obs.images if obs else [],
                                       "image_source": "simulated" if ctx.store.bench == "sim" else "camera"})
        for f in self.flake_ids:
            ctx.store.upsert_sample(f, "flake", location=self.target_chip, status="stacked",
                                    props={"device": device_id})
        ctx.store.upsert_sample(self.stamp_id, "stamp", props={"film": None, "stack": []})
        target = require_sample(ctx, self.target_chip, "chip")
        prior = target["props"].get("layers_top_first")
        ctx.store.upsert_sample(self.target_chip, "chip", props={"layers_top_first":
            self.flake_ids+prior if isinstance(prior, list) else None})
        if film_left:
            _hand_off_pc_film(ctx, self.target_chip, device_id, self.name)

    def before_park(self, ctx: FlowContext) -> None:
        self.safe_state(ctx)

    def safe_state(self, ctx: FlowContext) -> None:
        ctx.gate.heater(HEATER).off()
        _clear_lift(ctx, self.name)


def _hand_off_pc_film(ctx: FlowContext, chip_id: str, device_id: str, step: str) -> None:
    """Record the PC film left on the target and hand its washing to a person.

    The release process itself implies it: the molten film stays pinned to
    the chip when the PDMS separates. Qumus does not dissolve it (#180).
    """
    item = {"state": "on_chip", "film": "PC", "device": device_id, "source": step,
            "basis": "release process: the molten PC film stays on the chip when the PDMS separates",
            "needs_person": "dissolve the PC film", "recorded_at": ctx.clock.now()}
    ctx.store.upsert_sample(chip_id, "chip", props={"pc_film": item})
    ctx.store.emit("notification", {"level": "warn", "reason": "pc_film_needs_person",
        "text": f"Needs a person: dissolve the PC film left on {chip_id}",
        "chip_id": chip_id, "device_id": device_id}, run_id=ctx.run_id)


class SupportRelease(DropDown):
    """Shared support-dependent release, configured only in the simulation."""

    def _film_left_on_chip(self, ctx):
        from qumus_simlab.records.material_metal_simulation import simulation_profile
        from qumus_simlab.records.stamps import contact_film

        film = contact_film(ctx.store.get_sample(self.stamp_id))
        return film == "PC" and not getattr(simulation_profile(ctx.bench), film).support_retained_on_stamp

    def on_success(self, ctx, obs):
        from qumus_simlab.records.material_metal_simulation import simulation_profile
        from qumus_simlab.records.stamps import contact_film

        film = contact_film(ctx.store.get_sample(self.stamp_id))
        super().on_success(ctx, obs)
        retained = getattr(simulation_profile(ctx.bench), film).support_retained_on_stamp
        ctx.store.upsert_sample(self.stamp_id, "stamp", props={"film": film if retained else None})


# ----- the flow ----------------------------------------------------------------

#: flake statuses stack_device accepts (the picker marks its choice "selected")
STACKABLE = {"catalogued", "selected"}


def _release_initial_history(store: Any, run_id: str, phase: dict, checkpoint: dict, starts: list) -> list:
    """Mutable phase replacement cannot move its own initial journal boundary."""
    own = [e for e in starts if e.payload.get("name") == checkpoint["name"]]
    if not own or any(e.payload.get("seq") != checkpoint["seq"] for e in own):
        raise ValueError("Release lacks its original checkpoint start history")
    exposures = [e for e in _transfer_events(store, run_id, "lab.frame", own[0].id)
        if e.payload.get("microscope_preview", {}).get("phase") == checkpoint["name"]
        and e.payload["microscope_preview"].get("image_sha256")]
    if (not exposures or phase["reference_event"]["event_id"] != exposures[0].id
            or not own[0].ts <= phase["started_at"] <= exposures[0].payload["microscope_preview"]["timestamp"]):
        raise ValueError("Release reference was rebased after its actual initial capture")
    if not phase.get("complete"):
        activity = [e for e in _transfer_events(store, run_id, "device.command", own[0].id)
            if e.payload.get("device") == HEATER or e.payload.get("device") == MANIP
            and e.payload.get("command") == "move_axis_to"]
        if len(exposures) != 1 or activity:
            raise ValueError("Actual release work prevents unexecuted checkpoint admission")
    return exposures


def _admit_saved_transfer(raw: dict, run: dict, store: Any) -> None:
    if store is None:
        raise ValueError("Saved transfer admission requires actual stored checkpoints")
    checkpoints = store.get_steps(run["id"])
    if not isinstance(checkpoints, list) or not isinstance(run.get("meta"), dict):
        raise ValueError("Stored transfer progress is unavailable")
    state = run["meta"].get("normal_transfer", {})
    if not isinstance(state, dict):
        raise ValueError("Stored transfer evidence is malformed")
    refs, phases = state.get("source_references", {}), state.get("phases", {})
    if not isinstance(refs, dict) or not isinstance(phases, dict):
        raise ValueError("Stored transfer evidence is malformed")
    started = {s["name"] for s in checkpoints if s.get("started_at") is not None
               or s.get("status") in {"in_progress", "done", "corrected", "failed", "skipped", "unverified"}}
    completed = {s["name"] for s in checkpoints if s.get("status") in {"done", "corrected", "skipped", "unverified"}}
    skipped = {s["name"] for s in checkpoints if s.get("status") == "skipped"}
    # A pickup press superseded by a planned re-press (#109) is skipped; only
    # the last planned press of a source can commit its transfer.
    presses = {fid: pickup_press_names(fid, (raw.get("pickup_represses") or {}).get(fid, 0))
               for fid in raw["flake_ids"]}
    superseded = {n for names in presses.values() for n in names[:-1]}
    if (any(n in started and n not in skipped for n in superseded)
            or any(names[-1] in skipped for names in presses.values())):
        raise ValueError("A pickup press was skipped without its planned re-press")
    if raw.get("stamp_discovery_policy"):
        from qumus_simlab.records.discovery import validate_policy

        validate_policy(raw["stamp_discovery_policy"])
        if "prepare_stamp_discovery" in started or state.get("discovery"):
            _validate_discovery_ready(store, run["id"])
    if raw.get("device_design"):
        from qumus_simlab.config import load_yaml

        verification = load_yaml("verification.yaml")
        if raw.get("first_contact_sampling") != _first_contact_sampling(verification):
            raise ValueError("saved first-contact sampling differs from its current named policy")
        budget_context = SimpleNamespace(params=raw, verification=verification, store=store, run_id=run["id"])
        for name, phase in phases.items():
            if name.startswith("v1_first_contact_"):
                budget = _phase_budget(budget_context, phase)
                if phase.get("observations", 0) > budget["max_observations"]:
                    raise ValueError("saved first-contact phase exceeded its finite observation ceiling")
                if phase.get("complete"):
                    _first_contact_binding(phase["first_contact_region"], _image_calibration(_reference(phase["reference"])),
                                           verification["transfer_sequence"])
    if raw.get("device_design") or raw.get("release_policy") == OBSERVED_RELEASE_POLICY_ID:
        alignment = state.get("aperture_alignment")
        if alignment:
            validate_aperture_alignment(store, run["id"], alignment)
        if (f"v1_first_contact_{raw['stamp_id']}" in started and not alignment
                and not raw.get("skip_aperture_alignment")):
            raise ValueError("observed-release first contact requires its original aperture alignment")
    admissions = list(_transfer_events(store, run["id"], "stack.pickup_macro_admission"))
    names = {e.payload.get("phase") for e in admissions} | {
        name for name, phase in phases.items() if "final_partial_macro" in phase}
    for name in names:
        if name not in {n for fid in raw["flake_ids"]
                        for n in pickup_press_names(fid, (raw.get("pickup_represses") or {}).get(fid, 0))}:
            raise ValueError("Final pickup prefix names an unknown checkpoint")
        if name not in phases:
            raise ValueError("Final pickup prefix lost its reserved phase")
        phase = phases[name]
        partial = _pickup_partial_receipt(store, run["id"], phase, admissions)
        if (name not in completed or not phase.get("complete") or partial["status"] != "covered"
                or not phase.get("coverage", {}).get("covered")):
            raise ValueError("Incomplete reserved pickup prefix requires review before restoration")
        if name in superseded:
            continue  # its press covered and left the whole source; the next press is the commit
        if phase["endpoint_verified"]["event_id"] <= partial["event_id"]:
            raise ValueError("Pickup endpoint predates its final prefix admission")
    for fid in raw["flake_ids"]:
        required = (f"refind_{fid}" in completed or f"v1_initialize_{fid}" in started
                    or f"v1_pick_up_{fid}" in started
                    or fid == raw["flake_ids"][0] and f"v1_first_contact_{raw['stamp_id']}" in started)
        if required and fid not in refs:
            raise ValueError("Executed source checkpoint lacks its focus proof")
        if fid in refs:
            ref = refs[fid]
            if ref["source_id"] != fid or ref["source_chip_id"] != raw["flake_chips"][fid]:
                raise ValueError("Source identity changed")
            _validate_focus_reference(ref, run["id"], parked=True, store=store)
            offset = ref["target_offset_mm"]
            if not isinstance(offset, list) or len(offset) != 2 or not all(map(_finite, offset)):
                raise ValueError("Selected-source location is missing or unknown")
            identity = ref["source_identity"]
            if identity["source_id"] != fid or identity["source_chip_id"] != raw["flake_chips"][fid]:
                raise ValueError("Original source identity binding changed")
            for key in ("original", "current", "post_shift_capture"):
                _reference(identity[key])
            if raw.get("device_design"):
                from qumus_simlab.config import load_yaml

                _validate_design_source_orientation(store, run["id"], fid, identity, load_yaml("verification.yaml"))
            if identity["post_shift_capture"]["sha256"] != ref["sha256"]:
                raise ValueError("Post-shift source capture changed")
        for name in pickup_press_names(fid, (raw.get("pickup_represses") or {}).get(fid, 0)):
            if name not in started and name not in phases:
                continue
            expected = _process_reference(state, fid, raw["stamp_id"], run["id"], .2, store)
            if phases[name]["process_reference"] != expected:
                raise ValueError("Stored pickup process reference changed")
            if name in superseded:
                if name in started and name not in skipped:
                    raise ValueError("A superseded pickup press cannot have committed")
                continue
            if name in completed and not _visual_pickup_proof(phases[name].get("endpoint"), refs[fid], store, run["id"]):
                raise ValueError("Completed pickup lacks its source disappearance proof")
            if name in completed:
                endpoint = phases[name]["endpoint"]
                _verified_endpoint(store, phases[name]["endpoint_verified"], run["id"], name, endpoint,
                                   [frame["image"] for frame in endpoint["source_frames"]])
    for placement in state.get("placements", {}).values():
        if (placement["source_anchor"]["source_id"] != raw["flake_ids"][0]
                or placement["profile"] != TRANSFER_PROFILE_ID or placement["run_id"] != run["id"]):
            raise ValueError("First-source placement anchor changed")
        _process_reference(state, raw["flake_ids"][0], raw["stamp_id"], run["id"], .2, store)
    placement_checkpoint = "restore_stamp_discovery_source" if raw.get("stamp_discovery_policy") else f"v1_first_contact_{raw['stamp_id']}"
    if (placement_checkpoint in completed
            and raw["stamp_id"] not in state.get("placements", {})):
        raise ValueError("Completed first candidate lacks its working placement")
    starts = list(_transfer_events(store, run["id"], "step.started"))
    release_names = {name for name in started | {e.payload.get("name", "") for e in starts}
                     if "drop_down" in name or name == "release_material_metal_stack"}
    for name in release_names:
        phase = phases[name]
        if raw.get("release_policy") == V1_DROP_POLICY_ID:
            if phase.get("release_policy") != V1_DROP_POLICY_ID:
                raise ValueError("Started v1 drop lacks its policy")
            continue
        own_checkpoints = [row for row in checkpoints if row["name"] == name]
        if len(own_checkpoints) != 1:
            raise ValueError("Release lost its original stored checkpoint")
        checkpoint = own_checkpoints[0]
        exposures = _release_initial_history(store, run["id"], phase, checkpoint, starts)
        from qumus_simlab.config import load_yaml
        verification = load_yaml("verification.yaml")
        process, reference = _release_reference(state, raw, run["id"], verification, store)
        if (phase.get("release_policy") != raw.get("release_policy") or phase.get("release_reference") != reference
                or phase.get("process_reference") != process):
            raise ValueError("Started release lacks its target-relative policy/reference")
        budget = phase["release_budget"]
        if (budget["start_event_id"] != phase["reference_event"]["event_id"]
                or budget["stage"] not in {"entry_heat", "footprint", "press", "release_heat", "withdraw", "off", "park", "endpoint", "observed"}
                or budget["endpoint_limit"] != verification["focus"]["coarse_points"]+verification["focus"]["fine_points"]+6
                or not all(type(budget[k]) is int and budget[k] >= 0 for k in
                           ("endpoint_observations", "park_observations", "thermal_observations"))
                or budget["park_observations"] > 2
                or budget["thermal_observations"] > phase["observations"]
                or budget["endpoint_observations"] > budget["endpoint_limit"]
                or phase["observations"] != len(exposures)+budget["endpoint_observations"]
                or phase["observations"] > verification["transfer_sequence"]["max_observations"]
                or phase["travel_mm"] > verification["transfer_sequence"]["max_travel_mm"]
                or phase["observed_at"]-phase["started_at"] > verification["transfer_sequence"]["max_total_s"]
                or name not in completed and (budget.get("endpoint_status") or phase.get("park_transport")
                                               or budget.get("park_complete"))
                or not phase.get("complete") and (budget["stage"] != "entry_heat" or any(budget[k] for k in
                    ("endpoint_observations", "park_observations", "thermal_observations")))):
            raise ValueError("Saved release budget/endpoint is interrupted or malformed")
        if phase.get("complete"):
            _release_execution_binding(store, run["id"], phase)
        event = _historical_event(store, phase["reference_event"], run["id"], "lab.frame")
        capture = event.payload["microscope_preview"]
        if (phase["name"] != name or type(phase["observations"]) is not int or phase["observations"] < 1
                or not _finite(phase["started_at"]) or phase["started_at"] < checkpoint["started_at"]
                or phase.get("pending_intent") or phase.get("held_reason")
                or phase.get("complete") is not True and phase["observations"] > 1
                or capture["phase"] != name or capture["image"] != phase["reference"]["image"]
                or capture["timestamp"] != phase["reference"]["captured_at"]):
            raise ValueError("Started release phase is missing, interrupted or unbound")
        _reference(phase["reference"])
    for phase in phases.values():
        if phase.get("pending_intent") or "park_pending_intent" in phase:
            raise ValueError("Pending transfer intent requires human review before energy restoration")
        if phase["profile"] != TRANSFER_PROFILE_ID:
            raise ValueError("Old transfer phase cannot execute")
        if "first_contact_region" in phase:
            candidate = phase["first_contact_region"]
            if (candidate["found"] is not True or candidate["profile"] != TRANSFER_PROFILE_ID
                    or not all(_finite(candidate["actual_holder_pose"][k]) for k in ("x", "y", "z"))):
                raise ValueError("Retained candidate readback is unavailable")
            _reference(candidate["current"])
        if "process_reference" in phase:
            process = phase["process_reference"]
            if not all(_finite(v) for v in (process["minimum_z_mm"], process["first_candidate_z_mm"],
                process["max_press_mm"], *process["reference_interval_mm"], *process["delta_interval_mm"])):
                raise ValueError("Stored process reference is non-finite")
    if ("go_to_target" in completed or "v1_initialize_target" in started
            or any("drop_down" in n or n == "release_material_metal_stack" for n in started)
            or "target_reference" in state):
        target = _target_baseline(state, run["id"], store, raw["target_chip_id"], tuple(raw["target_xy_mm"]))
        for name in completed:
            if "drop_down" in name or name == "release_material_metal_stack":
                phase = phases[name]
                if not _visual_release_proof(phase.get("endpoint"), target, phase,
                        sorted({raw["materials"][fid] for fid in raw["flake_ids"]}), run["id"], store):
                    raise ValueError("Completed release lacks its target appearance proof")
                endpoint = phase["endpoint"]
                _verified_endpoint(store, phase["endpoint_verified"], run["id"], name, endpoint,
                    [frame["image"] for frame in target["frames"]+endpoint["target_after"]["frames"]])
                checkpoint = next(row for row in checkpoints if row["name"] == name)
                if checkpoint["measurements"] != endpoint:
                    raise ValueError("Completed release checkpoint observation changed")


def _visual_pickup_proof(endpoint: dict | None, reference: dict, store: Any, run_id: str) -> bool:
    if not isinstance(endpoint, dict):
        return False
    if (endpoint.get("profile") != TRANSFER_PROFILE_ID or endpoint.get("method") != "source_disappearance"
            or endpoint.get("transfer_confirmed") is not True or endpoint.get("process_confirmation") is not True
            or endpoint.get("flake_on_chip") is not False
            or endpoint.get("source_id") != reference["source_id"]
            or endpoint.get("source_reference_sha256") != reference["sha256"]
            or not endpoint.get("source_frames")
            or not any(frame["sha256"] != reference["sha256"] for frame in endpoint["source_frames"])):
        return False
    focus_id = endpoint.get("focus_event_id")
    if type(focus_id) is not int or focus_id <= 0:
        return False
    events = store.events_after(focus_id-1, limit=1)
    if (not events or events[0].id != focus_id or events[0].kind != "microscope.focus"
            or events[0].run_id != run_id):
        return False
    af = events[0].payload
    if af.get("plane") != "chip" or af.get("objective") != 20 or not _source_absence_focus_ok(af, reference):
        return False
    for frame in endpoint["source_frames"]:
        _reference(frame)
        if (frame["meta"]["objective"] != 20 or frame["meta"]["focus_plane"] != "chip"
                or frame["meta"]["calibration"] != reference["meta"]["calibration"]
                or frame["meta"]["stage_kinematics"] != reference["meta"]["stage_kinematics"]
                or frame["meta"]["stage_kinematics"] != af["stage_kinematics"]
                or frame["meta"]["focus_z"] != af["final_focus_readback_z_mm"]
                or not 0 <= frame["captured_at"]-af["captured_at"] <= 2):
            return False
    return True


def _visual_release_proof(endpoint: dict | None, before: dict | None, phase: dict,
                          materials: list[str], run_id: str, store: Any) -> bool:
    if (not isinstance(endpoint, dict) or not isinstance(before, dict)
            or endpoint.get("profile") != TRANSFER_PROFILE_ID
            or endpoint.get("method") != "target_material_appearance"
            or endpoint.get("process_confirmation") is not True or endpoint.get("release_confirmed") is not True
            or phase.get("complete") is not True or phase.get("outputs_off") is not True
            or phase.get("release_policy") not in (RELEASE_POLICY_ID, OBSERVED_RELEASE_POLICY_ID)
            or phase.get("release_budget", {}).get("endpoint_status") != "observed"
            or phase.get("release_budget", {}).get("park_complete") is not True
            or phase.get("release_budget", {}).get("endpoint_observations") !=
                phase.get("release_budget", {}).get("endpoint_limit")
            or not phase.get("park")
            or endpoint.get("target_before_sha256") != _digest(before)):
        return False
    after = endpoint["target_after"]
    budget = phase["release_budget"]
    start, end = budget["endpoint_start_event_id"], budget["endpoint_end_event_id"]
    af_id = after["focus_reference"]["focus_binding"]["event_id"]
    if type(start) is not int or type(end) is not int or not 0 < start < af_id < end:
        return False
    captures = [e for e in _transfer_events(store, run_id, "device.command", start)
        if e.id <= end and e.payload.get("device") == MICROSCOPE and e.payload.get("command") == "capture"]
    if len(captures) != budget["endpoint_observations"]:
        return False
    if (after["focus_reference"]["focus_binding"]["holder_pose"] != phase["park"]["actual_holder_pose"]
            or phase["park"]["completed_at"] > after["frames"][0]["captured_at"]):
        return False
    for view, parked in ((before, True), (after, True)):
        _validate_focus_reference(view["focus_reference"], run_id, parked=parked, store=store)
        if view["profile"] != TRANSFER_PROFILE_ID or view["run_id"] != run_id or len(view["frames"]) != 3:
            return False
        meta = view["focus_reference"]["meta"]
        for frame in view["frames"]:
            _reference(frame)
            if (any(frame["meta"][k] != meta[k] for k in
                    ("objective", "focus_z", "focus_plane", "calibration", "stage_kinematics"))
                    or frame["meta"]["stage_kinematics"] != view["stage_kinematics"]):
                return False
    if (any(before[k] != after[k] for k in ("target_chip_id", "target_xy_mm", "requested_stage_xy_mm", "stage_kinematics"))
            or before["focus_reference"]["meta"]["calibration"] != after["focus_reference"]["meta"]["calibration"]
            or not any(f["sha256"] not in {old["sha256"] for old in before["frames"]} for f in after["frames"])
            or not before["frames"][-1]["captured_at"] < phase["started_at"] <= after["frames"][0]["captured_at"]):
        return False
    return (endpoint["materials_appeared"] == dict.fromkeys(materials, True)
        and all(type(before["materials_found"].get(mat)) is int and before["materials_found"][mat] == 0
                and type(after["materials_found"].get(mat)) is int and after["materials_found"][mat] > 0 for mat in materials))


class StackParams(BaseModel):
    stamp_discovery_policy: dict[str, Any] | None = None
    # v1 has no paper-window alignment (owner decision 2026-11-02, #131): the
    # SIM-only v1 profile skips CenterStampAperture and its clearance checks.
    skip_aperture_alignment: bool = False
    device_design: DeviceDesignV1 | None = None
    design_pose_window_admission: DesignPoseWindowAdmission | None = None
    design_rotations_deg: dict[str, float] = Field(default_factory=dict)
    contact_dwell_s: float | None = Field(None, ge=0, le=60, allow_inf_nan=False)
    contact_dwell_source: str | None = None
    first_contact_sampling: dict[str, Any] | None = None
    contact_interior_calibration: dict[str, Any] = Field(default_factory=dict, description="Internal frozen public optical-class calibration")
    design_park_pose: dict[str, float] = Field(default_factory=dict)
    flake_ids: list[str] = Field(min_length=1, max_length=6,
                                 description="Catalogued flakes in pick-up order: first = top "
                                             "layer of the finished stack")
    target_chip_id: str = Field(description="Chip to drop the stack on (e.g. with electrodes)")
    target_xy_mm: tuple[float, float] | None = Field(
        None, description="Drop point on the target chip, mm from its bottom-left corner; "
                          "None = chip centre")
    procedure_contract: str | None = None  # filled only by new submission
    transfer_profile: str | None = None
    release_policy: str | None = None  # filled only by fresh resolution, never defaulted on replay
    design_shifts_mm: dict[str, tuple[float, float]] = Field(default_factory=dict,
        description="Optional per-source chip-local stage shifts in mm; absent means zero.")
    design_path: str | None = Field(None, description="Device design (KLayout) for the record")
    target_contact_region_id: str | None = None
    direct_contact_material: str | None = None
    bridge_requirement: BridgeRequirement | None = None
    direct_contact_intent: dict | None = None  # target-only accepted guidance
    direct_contact_contract: DirectContactContract | None = None  # resolved from current selected public source
    pickup_temp_c: float = Field(90.0, ge=40, le=130)
    drop_temp_c: float = Field(155.0, ge=100, le=170)
    melt_temp_c: float = Field(170.0, ge=150, le=185)
    # filled in at submit
    stamp_id: str | None = None  # the one stamp selected for this normal run
    support_assumptions: dict[str, Any] = Field(default_factory=dict)
    stamp_film: str = "PC"
    stamp_slot: str | None = None  # where it comes back to
    flake_chips: dict[str, str] = Field(default_factory=dict)
    materials: dict[str, str] = Field(default_factory=dict)
    home_slots: dict[str, str] = Field(default_factory=dict)
    # Historical data remains readable; it cannot authorize replacement Steps.
    stamp_swaps: list[dict[str, str]] = Field(default_factory=list,
        description="Former-process replacement history; not executable by the current normal sequence.")
    # Jialei 2026-11-03: extra presses planned after a pickup left its whole
    # source behind (on_step_failed); each is its own Step and phase.
    pickup_represses: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def validate_saved_execution_binding(cls, raw: dict[str, Any], *, saved_run: dict | None = None, store: Any = None) -> None:
        # The public material-metal wrapper shares these Params. Its explicit
        # non-SIM registered adapters retain their separate legacy contract.
        protocol = cls.model_fields.get("protocol_id")
        if protocol is not None and raw.get("protocol_id", protocol.default) != protocol.default:
            return
        if raw.get("stamp_swaps"):
            raise WorkflowMismatch("historical_stamp_swap_plan",
                "Historical automatic stamp replacement requires read-only review")
        if raw.get("stamp_discovery_policy") is not None:
            from qumus_simlab.config import load_yaml
            from qumus_simlab.records.discovery import validate_policy, validate_verification

            validate_policy(raw["stamp_discovery_policy"])
            validate_verification(load_yaml("verification.yaml"))
            if not raw.get("device_design"):
                raise WorkflowMismatch("stamp_discovery_binding_changed", "Linked discovery requires its immutable typed device design")
        if (raw.get("procedure_contract") != PROCEDURE_ID or raw.get("transfer_profile") != TRANSFER_PROFILE_ID
                or raw.get("release_policy") not in (RELEASE_POLICY_ID, OBSERVED_RELEASE_POLICY_ID, V1_DROP_POLICY_ID)):

            raise WorkflowMismatch("transfer_procedure_changed",
                "Old or missing transfer procedure/profile/release-policy binding requires read-only review")
        if raw.get("release_policy") == OBSERVED_RELEASE_POLICY_ID and _fresh_release_policy(raw) != OBSERVED_RELEASE_POLICY_ID:
            raise WorkflowMismatch("release_policy_changed", "Observed release lost its explicit typed load")
        if raw.get("release_policy") == V1_DROP_POLICY_ID and _fresh_release_policy(raw) != V1_DROP_POLICY_ID:
            raise WorkflowMismatch("release_policy_changed", "The v1 drop lost its explicit typed Hall-bar load")
        if (raw.get("direct_contact_material") or raw.get("target_contact_region_id")
                or raw.get("direct_contact_intent") or raw.get("direct_contact_contract")):
            try:
                c = DirectContactContract.model_validate(raw.get("direct_contact_contract"))
                target = c.target_intent["target_contact"]
                if (raw.get("flake_ids") != [c.source_flake_id] or raw.get("stamp_film") != "PC"
                        or raw.get("direct_contact_material") != "Gr" or raw.get("target_chip_id") != target["target_chip_id"]
                        or raw.get("target_contact_region_id") != target["region"]["region_id"]
                        or raw.get("flake_chips", {}).get(c.source_flake_id) != c.source_observation["source_chip_id"]
                        or raw.get("materials", {}).get(c.source_flake_id) != "Gr"
                        or raw.get("direct_contact_intent") != c.target_intent
                        or raw.get("bridge_requirement") != c.target_intent["bridge_requirement"]):
                    raise ValueError("direct contact execution parameters changed")
                if store is not None:
                    _direct_contact_public_binding(raw, store, saved_run)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise WorkflowMismatch("direct_contact_binding_changed", "Resolved direct-contact binding requires read-only review") from exc
        if saved_run is not None:
            try:
                if raw.get("device_design"):
                    _device_design_public_binding(raw, store, saved_run)
                _admit_saved_transfer(raw, saved_run, store)
            except (KeyError, TypeError, ValueError, OSError, NeedsHumanError) as exc:
                raise WorkflowMismatch("transfer_reference_changed",
                    "Saved transfer focus/process/visual binding is incomplete or changed; review before continuing") from exc

    @field_validator("flake_ids")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("a flake appears twice")
        return v

    @model_validator(mode="after")
    def _temps(self) -> StackParams:
        if self.drop_temp_c >= self.melt_temp_c:
            raise ValueError("drop_temp_c must be below melt_temp_c")
        return self


@register_flow
class StackDeviceFlow(Flow):
    name = "stack_device"
    description = ("Dry-transfer stacking: pick up catalogued flakes one by one with a fresh "
                   "stamp (first = top layer), each pick-up checked, then drop the stack on "
                   "the target chip, check every layer arrived, cool, store, return the stamp.")
    Params = StackParams
    devices = (ARM, CAMERA, STAGE, STAGE_VACUUM, MICROSCOPE, MANIP, MANIP_VACUUM, HEATER, FAN)

    def validate_saved_params(self, raw, *, saved_run=None, store=None):
        super().validate_saved_params(raw, saved_run=saved_run, store=store)
        if raw.get("skip_aperture_alignment") and store is not None and store.bench != "sim":
            raise ValueError("The v1 no-window stamp profile is simulation-only")
        if getattr(self, "_bench", None) is None:
            # Early preflight (saved_flow) runs before the composed bench is
            # bound; the default bench.yaml is not that bench. stack_workflow()
            # repeats this check with the bound bench before building steps.
            return
        policy = self.bench.get("stamp_discovery")
        if raw.get("stamp_discovery_policy") != policy:
            raise WorkflowMismatch("stamp_discovery_binding_changed",
                "Saved stamp discovery policy differs from the actual composed bench")

    def on_step_failed(self, params, step, outcome, ctx):
        """Jialei 2026-11-03: a design pickup that left its whole source on the
        chip is pressed again, at most pickup.max_represses times (#109).

        Only an observed miss qualifies: covered contact, intact film, the
        source still registered to its original identity, no transfer
        confirmed, and no error. Each extra press is a new Step with its own
        phase; for a second layer its preconditions observe again that the
        carried hBN still covers the Gr. Anything else parks as before."""
        if (not isinstance(step, PickUpFlake) or not isinstance(params, StackParams)
                or params.device_design is None or outcome.error or not outcome.observations
                or step.name != pickup_name(params, step.flake_id)):
            return None
        obs = outcome.observations[-1]
        m = obs.measurements
        if (obs.verdict != Verdict.UNCLEAR or m.get("method") != "source_disappearance"
                or m.get("flake_on_chip") is not True or m.get("source_detection_count") != 1
                or m.get("transfer_confirmed") is not False or m.get("film_intact") is False
                or not (m.get("coverage") or {}).get("covered") or not m.get("source_presence")):
            return None
        done = params.pickup_represses.get(step.flake_id, 0)
        if done >= ctx.verification["pickup"].get("max_represses", 0):
            return None
        body = {"flake_id": step.flake_id, "failed_press": step.name, "press": done+2,
                "max_represses": ctx.verification["pickup"]["max_represses"],
                "source_presence": m["source_presence"], "coverage_method": m["coverage"].get("method"),
                "source_frames": m.get("source_frames"),
                "scope": "Observed whole retained source; a partial transfer is excluded only later by the release footprint"}
        ctx.store.emit("stack.pickup_repress", body, run_id=ctx.run_id)
        return Replan(params.model_copy(update={"pickup_represses": {
            **params.pickup_represses, step.flake_id: done+1}}),
            note=f"{step.flake_id} stayed on its chip; press again ({done+2} of {1+body['max_represses']})")

    def target_storage(self, target: dict[str, Any], store: Any) -> str:
        """Physical source for the target; transport subclasses may plan a handover."""
        return target["location"]

    def after_stack(self, params: StackParams) -> list[Step]:
        """Optional checked preparation while the cooled target is under optics."""
        return []

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        assert isinstance(params, StackParams)
        if params.stamp_swaps:
            raise ValueError("Automatic damaged-stamp replacement is not part of the current normal sequence")
        if params.design_pose_window_admission is not None and params.device_design is None:
            raise ValueError("Pose-window admission requires its immutable typed device design")
        p = params.model_copy(deep=True)
        from qumus_simlab.records.materials import pickup_admission_error

        bench = self.bench
        if p.stamp_discovery_policy is not None and p.stamp_discovery_policy != bench.get("stamp_discovery"):
            raise ValueError("Caller stamp discovery policy differs from the actual composed bench")
        if bench.get("stamp_discovery") is not None:
            from qumus_simlab.config import load_yaml
            from qumus_simlab.records.discovery import validate_policy, validate_verification

            policy = validate_policy(bench["stamp_discovery"])
            validate_verification(load_yaml("verification.yaml"))
            if p.device_design is None:
                raise PreconditionFailed("Linked stamp discovery requires its immutable typed device design",
                                         reason="stamp_discovery_requires_design")
            p.stamp_discovery_policy = policy
        places = {n for n in reachable(bench, ARM) if is_slot(bench, n)}
        boards = ", ".join(bench[ARM]["reach"])
        for fid in p.flake_ids:
            f = store.get_sample(fid)
            if f is None or f["kind"] != "flake":
                raise ValueError(f"{fid} is not a known flake")
            if f["status"] not in STACKABLE:
                raise ValueError(f"{fid} is {f['status']}; only catalogued or selected flakes "
                                 f"can be stacked")
            if error := pickup_admission_error(store, f):
                raise PreconditionFailed(error, flake_id=fid, reason="pickup_neighbourhood_not_clear")
            if not f["props"].get("rel_corner_mm"):
                raise ValueError(f"{fid} has no position on its chip")
            chip = store.get_sample(f["location"])
            if chip is None or chip["location"] not in places:
                raise ValueError(f"the chip of {fid} ({f['location']}) is not on a board "
                                 f"{ARM} reaches ({boards})")
            p.flake_chips[fid] = chip["id"]
            p.materials[fid] = f["props"].get("material") or "?"
            p.home_slots[chip["id"]] = chip["location"]
        target = store.get_sample(p.target_chip_id)
        if target is None or target["kind"] != "chip":
            raise ValueError(f"{p.target_chip_id} is not a known chip")
        target_slot = self.target_storage(target, store)
        if target_slot not in places:
            raise ValueError(f"{p.target_chip_id} is at {target['location']}, not on a board "
                             f"{ARM} reaches ({boards})")
        if p.target_chip_id in p.flake_chips.values():
            raise ValueError("the target chip also carries one of the flakes")
        p.home_slots[p.target_chip_id] = target_slot
        if p.device_design:
            design = validate_design_current(store, p.device_design.project_id, p.device_design)
            admission = build_design_pose_window_admission(design)
            if p.design_pose_window_admission is not None and p.design_pose_window_admission != admission:
                raise ValueError("Caller pose-window admission differs from the freshly recomputed design screen")
            p.design_pose_window_admission = admission
            if p.flake_ids != list(design.pickup_order) or p.target_chip_id != design.target_chip_id:
                raise ValueError("Stack identities/order differ from the saved device design")
            if any((p.direct_contact_material, p.target_contact_region_id, p.direct_contact_intent, p.direct_contact_contract)):
                raise ValueError("Eight-contact hBN/Gr cannot use the legacy sole-Gr execution route")
            anchor = design_anchor_mm(design.target)
            if p.target_xy_mm is not None and tuple(p.target_xy_mm) != anchor:
                raise ValueError("Target origin differs from the design-derived electrode-site anchor")
            p.target_xy_mm = anchor
            shifts, rotations = {}, {}
            for source in design.sources:
                angle = math.radians(-source.placement.rotation_deg)
                dx, dy = (a-b for a, b in zip(source.placement.translation_mm, anchor, strict=True))
                shifts[source.flake_id] = (dx*math.cos(angle)-dy*math.sin(angle),
                                            dx*math.sin(angle)+dy*math.cos(angle))
                rotations[source.flake_id] = source.placement.rotation_deg
                if abs(source.placement.rotation_deg) > 1e-9:
                    raise ValueError("v1 never rotates a chip: the design must place each source at 0 deg")
            if p.design_shifts_mm and p.design_shifts_mm != shifts:
                raise ValueError("Caller source shifts differ from the accepted design")
            p.design_shifts_mm, p.design_rotations_deg = shifts, rotations
            if p.design_path is not None and p.design_path != design.artifact.path:
                raise ValueError("Caller artifact path differs from the accepted design")
            p.design_path = design.artifact.path
            optical = _contact_classification_snapshot(store, design)
            if p.contact_interior_calibration and p.contact_interior_calibration != optical:
                raise ValueError("Caller contact calibration differs from frozen public optical evidence")
            p.contact_interior_calibration = optical
            p.contact_dwell_source = "synthetic_recipe_default" if p.contact_dwell_s is None else "explicit_recipe"
            p.contact_dwell_s = 1. if p.contact_dwell_s is None else p.contact_dwell_s
            if p.contact_dwell_s <= 0:
                raise ValueError("new design pickup requires a finite positive contact dwell")
            from qumus_simlab.config import load_yaml

            sampling = _first_contact_sampling(load_yaml("verification.yaml"))
            if p.first_contact_sampling is not None and p.first_contact_sampling != sampling:
                raise ValueError("caller first-contact sampling differs from the named policy")
            p.first_contact_sampling = sampling
            # Floats, as the Params model dumps them: a replanned params
            # snapshot (re-press, #109) must equal the frozen submission.
            park = {"x": float(bench[MANIP]["park"]["x"]), "y": float(bench[MANIP]["park"]["y"]),
                    "z": float(bench[MANIP]["safe_z"])}
            if p.design_park_pose and p.design_park_pose != park:
                raise ValueError("caller source-inspection park differs from the configured clear pose")
            p.design_park_pose = park
        direct = bool(p.direct_contact_material or p.target_contact_region_id or p.direct_contact_intent or p.direct_contact_contract)
        if direct:
            if p.direct_contact_material != "Gr" or not p.target_contact_region_id or p.bridge_requirement is None:
                raise ValueError("explicit direct contact requires Gr, a named contact region and a bridge requirement")
            intent = direct_contact_intent(store, p.target_chip_id, p.target_contact_region_id, p.bridge_requirement)
            if p.direct_contact_intent is not None and p.direct_contact_intent != intent:
                raise ValueError("accepted direct-contact intent changed")
            p.direct_contact_intent = intent
            resolved = resolve_direct_contact_contract(store, p.flake_ids, intent)
            if p.direct_contact_contract is not None and p.direct_contact_contract != resolved:
                raise ValueError("frozen selected-source execution contract changed before dispatch")
            p.direct_contact_contract = resolved
        if p.target_xy_mm is None:
            w, h = target["props"].get("size_mm") or [10.0, 10.0]
            p.target_xy_mm = (w / 2, h / 2)
        if direct:
            from qumus_simlab.core.polygons import transform

            target_contact = p.direct_contact_contract.target_intent["target_contact"]
            w, h = target_contact["size_mm"]
            prospective = transform(p.direct_contact_contract.source_outline_mm,
                xy=(p.target_xy_mm[0]-w/2, p.target_xy_mm[1]-h/2),
                rotation_deg=p.direct_contact_contract.outline_rotation_deg)
            fit = contact_bridge_geometry(prospective, target_contact, p.bridge_requirement)
            if fit["status"] != "pass":
                raise ValueError("selected complete Gr cannot bridge the named contacts at the proposed normal placement")
        stamp = self.select_stamp(p, store)
        p.stamp_id, p.stamp_slot = stamp["id"], stamp["location"]
        from qumus_simlab.records.stamps import contact_film

        p.stamp_film = contact_film(stamp)
        if (direct or p.device_design) and p.stamp_film != "PC":
            raise ValueError("the explicit direct Gr contact intent requires PC")
        if p.stamp_film == "mica":
            from qumus_simlab.records.material_metal_simulation import simulation_profile

            support = simulation_profile(bench).mica
            p.support_assumptions = support.model_dump()
            p.pickup_temp_c, p.drop_temp_c, p.melt_temp_c = support.pickup_c, support.release_c - 15, support.release_c
        p.stamp_swaps = []
        p.procedure_contract, p.transfer_profile = PROCEDURE_ID, TRANSFER_PROFILE_ID
        p.release_policy = _fresh_release_policy(p.model_dump(mode="json"))
        return p

    def select_stamp(self, params: StackParams, store: Any) -> dict:
        fresh = fresh_stamps(store, self.bench)  # not one a queued job already plans on
        if not fresh:
            raise ValueError("no fresh stamp on the right storage board (a used or torn stamp "
                             "counts again once a person reports a new PC film on it)")
        from qumus_simlab.records.stamps import contact_film

        fresh = [stamp for stamp in fresh if (not params.stamp_id or params.stamp_id == stamp["id"])
                 and contact_film(stamp) in (("PC",) if params.direct_contact_material or params.device_design else ("PC", "mica"))]
        if not fresh:
            raise ValueError("Direct Gr contact requires PC: choose a fresh PC stamp in storage."
                if params.direct_contact_material else "Choose a known PC/mica stamp in storage.")
        if contact_film(fresh[0]) == "mica" and store.bench != "sim":
            raise ValueError("Mica transfer requires a validated physical release adapter; current profile is simulation-only.")
        return fresh[0]

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        """The chips' and the stamp's own slots stay theirs while they are away."""
        assert isinstance(params, StackParams) and params.stamp_id and params.stamp_slot
        if params.device_design is not None:
            from qumus_simlab.records.lab_geometry import LabEvidence, Source, record_evidence

            admission = validate_design_pose_window_admission(params.device_design, params.design_pose_window_admission)
            submitted = list(_transfer_events(store, run_id, "job.submitted"))
            if len(submitted) != 1 or submitted[0].payload["params"].get("design_pose_window_admission") != admission.model_dump(mode="json"):
                raise ValueError("Fresh design admission is not bound in the original submission")
            record_evidence(store, LabEvidence(subject=params.device_design.design_id,
                property="device_design_pose_window_admission", bench="sim", value=admission.model_dump(mode="json"),
                units="mm", frame="chip_bottom_left_axes", observed_at=store.clock.now(), timebase="store_clock",
                source=Source(kind="synthetic", reference=f"{run_id}:job.submitted",
                              sha256=admission.report_sha256, event_ids=[submitted[0].id])), run_id=run_id)
        for chip, slot in params.home_slots.items():
            store.reserve("slot", slot, run_id, self.name, note=f"{chip} comes back here")
            store.reserve("sample", chip, run_id, self.name, note="in the stack")
        store.reserve("slot", params.stamp_slot, run_id, self.name,
                      note=f"{params.stamp_id} comes back here")
        store.reserve("sample", params.stamp_id, run_id, self.name, note="the stamp")

    def stack_workflow(self, params: BaseModel) -> Assembly:
        assert isinstance(params, StackParams)
        p = params
        self.validate_saved_params(p.model_dump())
        assert p.stamp_id and p.stamp_slot and p.target_xy_mm
        steps: list[Step] = [
            HeatTransferStage(p.pickup_temp_c),
            ManipulatorTo("load"),
            MoveObject(p.stamp_id, "stamp", arm=ARM, camera=CAMERA, dest=HANDOFF,
                       cfg_key="load_stamp", name="load_stamp"),
            GripStamp(p.stamp_id),
        ]
        molecules = [Molecule("v1_normal_sequence_v1_sim_v2_initialize_transfer", "Warm the transfer stage and load a known stamp", tuple(steps))]
        for fid in p.flake_ids:
            chip = p.flake_chips[fid]
            # v1 rough refind: the angle from one corner photo (#204).
            steps = [*self._chip_in(chip, fid, corner_flake=fid if p.device_design else None)]
            steps += [RefindFlake(fid, chip, p.materials[fid],
                                  design_shift_mm=p.design_shifts_mm.get(fid, (0., 0.)))]
            if fid == p.flake_ids[0] and p.stamp_discovery_policy:
                steps.append(PrepareStampDiscovery())
            steps.append(InitializeStamp(p.stamp_id, name=f"v1_initialize_{fid}"))
            if fid == p.flake_ids[0]:
                if ((p.device_design or p.release_policy == OBSERVED_RELEASE_POLICY_ID)
                        and not p.skip_aperture_alignment):
                    steps.append(CenterStampAperture(p.stamp_id))
                steps.append(AdjustFirstContact(p.stamp_id, name=f"v1_first_contact_{p.stamp_id}"))
                if p.stamp_discovery_policy:
                    steps.append(RestoreStampDiscoverySource())
            steps.append(PickUpFlake(fid, p.stamp_id, p.materials[fid]))
            for name in pickup_press_names(fid, p.pickup_represses.get(fid, 0))[1:]:
                press = name.removeprefix(f"v1_pick_up_{fid}_")
                steps += [HeatTransferStage(p.pickup_temp_c, name=f"reheat_{fid}_{press}"),
                          InitializeStamp(p.stamp_id, name=f"v1_initialize_{fid}_{press}"),
                          PickUpFlake(fid, p.stamp_id, p.materials[fid], name=name)]
            steps += [ManipulatorTo("park", name=f"park_stamp_{fid}"),
                      *self._chip_out(chip, p.home_slots[chip], None)]
            molecules.append(Molecule(f"pickup_layer_{fid}",
                "Refind source, initialize, observe Z-only coverage, verify pickup and store source", tuple(steps)))
        t = p.target_chip_id
        steps = [*self._chip_in(t, "target")]
        if p.device_design:
            if target_import_binding(p.device_design.target) is not None:
                # v1: the user's 20x photo of the site, not the layout (#204)
                steps.append(RefindElectrodeSite(t, p.target_xy_mm))
        steps += [
                  GoToTarget(t, p.target_xy_mm),
                  ObserveTransferTarget(t),
                  InitializeStamp(p.stamp_id, name="v1_initialize_target"),
                  (SupportRelease if p.stamp_film == "mica" else DropDown)(p.stamp_id, p.flake_ids, p.materials, t, p.target_xy_mm,
                           p.melt_temp_c, p.design_path),
                  ManipulatorTo("park", name="park_stamp_after_drop"),
                  CoolTransferStage(),
                  *self.after_stack(p),
                  *self._chip_out(t, p.home_slots[t], "stacked")]
        molecules.append(Molecule("release_stack", "Initialize and release at fixed recipe, verify, cool and store target", tuple(steps)))
        steps = [ManipulatorTo("load", name="manipulator_to_load_for_return"),
                  MoveObject(p.stamp_id, "stamp", arm=ARM, camera=CAMERA, dest=p.stamp_slot,
                             cfg_key="return_stamp", name="return_stamp", source=HANDOFF,
                             release_switch=MANIP_VACUUM, status="used",
                             fallback=(slot_station(self.bench, p.stamp_slot) or "", "stamp"),
                             param_path=["stamp_slot"])]
        molecules.append(Molecule("return_stamp", "Return the used stamp to its reserved storage slot", tuple(steps)))
        return Assembly(self.name, self.description, tuple(molecules))

    def build_workflow(self, params: BaseModel) -> Assembly:
        return self.stack_workflow(params)

    def build_steps(self, params: BaseModel) -> list[Step]:
        return self.stack_workflow(params).build_steps()

    @staticmethod
    def _chip_in(chip: str, tag: str, corner_flake: str | None = None) -> list[Step]:
        to_load = MoveTransferStage("load", vacuum=False, name=f"transfer_stage_to_load_{tag}")
        place = PlaceChipOnTransferStage(chip)
        place.name = f"place_{chip}_on_transfer_stage"
        to_mic = ChipToMicroscope(chip)
        to_mic.name = f"{chip}_to_microscope"
        corner = FindChipCorner(chip)
        corner.name = f"find_corner_{chip}"
        name = f"measure_rotation_{chip}"
        measure = (MeasureCornerRotation(chip, corner_flake, name=name) if corner_flake
                   else MeasureChipRotation(chip, name=name))
        return [to_load, place, to_mic, corner, measure]

    def _chip_out(self, chip: str, slot: str, status: str | None) -> list[Step]:
        return [MoveTransferStage("load", name=f"transfer_stage_to_load_{chip}"),
                MoveObject(chip, "chip", arm=ARM, camera=CAMERA, dest=slot, cfg_key="store_chip",
                           name=f"store_{chip}", release_switch=STAGE_VACUUM, status=status,
                           fallback=(slot_station(self.bench, slot) or "", "chip"),
                           param_path=["home_slots", chip])]

