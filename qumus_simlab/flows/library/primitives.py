"""Shared building blocks for steps: camera looks, arm pick-and-place, heater
ramps and waits, microscope autofocus. v1 had one copy of each per atom file
(exfoliation/ and search/ and stacking/ each had their own); v2 keeps one.

All device access goes through ctx.gate; all waiting through ctx.clock.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any

from qumus_simlab.core.errors import (
    DangerousError,
    NeedsHumanError,
    PreconditionFailed,
    TransientError,
)
from qumus_simlab.core.types import Pose, angle_diff
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.perception.base import Detection, MicroscopeAnalyzer

MICROSCOPE = "microscope"


# ----- overview camera -----------------------------------------------------

OVERVIEW = "overview_camera"
#: detections of different frames this close (mm) are the same object
SAME_OBJECT_MM = 3.0
#: frames tried before a dropped frame / unreadable QR counts as a failure
#: (waits 0.5, 1, 2, 4, 8 s in between)
LOOK_ATTEMPTS = 6


def _look_once(ctx: FlowContext, camera: str, region: str, frame_of: str,
               images: list[str] | None, captures: list[dict] | None = None) -> list[Detection]:
    """One frame, retried a few times when the camera drops a frame or cannot
    read a board's QR markers (transient; v1 retried QR reads with filters)."""
    delay = 0.5
    for attempt in range(LOOK_ATTEMPTS):
        try:
            if captures is not None:
                cursor, started = ctx.store.last_event_id(), ctx.clock.now()
            frame = ctx.gate.camera(camera).capture()
            returned = ctx.clock.now() if captures is not None else None
            dets = ctx.locator.locate(frame, region=region, frame_of=frame_of)
        except TransientError:
            if attempt == LOOK_ATTEMPTS - 1:
                raise
            ctx.clock.sleep(delay)
            delay *= 2
            continue
        if images is not None and (path := frame.saved_path()) and path not in images:
            images.append(path)
        if captures is not None:
            path = frame.saved_path()
            commands = [event for event in ctx.store.events_after(cursor, limit=32, run_id=ctx.run_id)
                if event.kind == "device.command" and event.payload.get("device") == camera
                and event.payload.get("command") == "capture"]
            captures.append({"run_id": ctx.run_id, "camera": camera, "region": region, "frame_of": frame_of,
                "capture_event_id": commands[0].id if len(commands) == 1 else None,
                "capture_started_at": started, "capture_returned_at": returned,
                "frame_device_id": frame.device_id, "frame_timestamp": frame.timestamp,
                "image": path, "image_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    if path and Path(path).is_file() else None,
                "detections": [{"label": d.label, "pose": d.pose.model_dump(), "confidence": d.confidence,
                    **({"object_id": d.object_id} if d.object_id is not None else {})} for d in dets]})
        return dets
    raise AssertionError("unreachable")


def look(ctx: FlowContext, camera: str, region: str, frame_of: str,
         label: str | None = None, images: list[str] | None = None,
         views: int = 1, recheck_empty: int = 0, captures: list[dict] | None = None) -> list[Detection]:
    """Locate objects in `region`; the photo's path goes into `images`.

    With views > 1 the camera takes that many frames and only objects seen in
    a majority of them count, at the median of their sightings: a single
    missed detection, a one-frame phantom or one badly scattered position does
    not decide anything (v1 captured several YOLO frames too,
    live_yolo_capture max_frames=5).

    `recheck_empty` > 0: when nothing (of `label`) is seen, look once more with
    that many frames before reporting "nothing there" (for checks where the
    object is expected: a missing object parks the run, so be sure).
    Optional `captures` retains the existing capture call, adapter timestamp,
    saved image and public detection fields; it never copies free-form metadata.
    """
    out = _vote(ctx, camera, region, frame_of, label, images, views, captures)
    if not out and recheck_empty > 0:
        out = _vote(ctx, camera, region, frame_of, label, images, recheck_empty, captures)
    return out


def _vote(ctx: FlowContext, camera: str, region: str, frame_of: str, label: str | None,
          images: list[str] | None, views: int, captures: list[dict] | None = None) -> list[Detection]:
    frames = [_look_once(ctx, camera, region, frame_of, images, captures) for _ in range(views)]
    if views == 1:
        dets = frames[0]
    else:
        groups: list[list[Detection]] = []
        for fr in frames:
            for d in fr:
                g = next((g for g in groups if g[0].label == d.label
                          and g[0].pose.xy_distance(d.pose) < SAME_OBJECT_MM), None)
                if g is None:
                    groups.append([d])
                else:
                    g.append(d)
        dets = []
        for g in groups:
            if len(g) * 2 > views:
                vote = sum(bool(d.props.get("material_visible")) for d in g) * 2 > len(g)
                first = g[0]
                pose = first.pose.model_copy(update={
                    "x": median(d.pose.x for d in g), "y": median(d.pose.y for d in g),
                    "rotation": first.pose.rotation + median(
                        angle_diff(d.pose.rotation, first.pose.rotation) for d in g)})
                props = first.props
                if "material_visible" in props:
                    props = {**props, "material_visible": vote}
                dets.append(first.model_copy(update={"pose": pose, "props": props}))
    return [d for d in dets if label is None or d.label == label]


# ----- arms ----------------------------------------------------------------

def ensure_homed(ctx: FlowContext, arm_id: str) -> None:
    """After a controller reset the arm does not know where it is: home it
    (straight up, then to its home pose) before any move. A held object stays
    on the nozzle."""
    arm = ctx.gate.arm(arm_id)
    if not arm.is_homed():
        ctx.store.emit("notification", {"level": "info", "text":
                       f"{arm_id} lost its position (controller reset): homing"},
                       run_id=ctx.run_id)
        arm.home()


def move_arm(ctx: FlowContext, arm_id: str, pose: Pose) -> None:
    """One arm move. If the controller reset since the last move (the arm no
    longer knows where it is, so the SafetyGate would refuse the move), home
    first and come down to `pose` from safe height."""
    arm = ctx.gate.arm(arm_id)
    if not arm.is_homed():
        ensure_homed(ctx, arm_id)
        safe_z = ctx.bench[arm_id]["safe_z"]
        if pose.z < safe_z:
            arm.move_to(Pose(x=pose.x, y=pose.y, z=safe_z, rotation=pose.rotation))
    arm.move_to(pose)


def lift(ctx: FlowContext, arm_id: str) -> None:
    """Raise the nozzle to safe height where it is (homing first if needed)."""
    ensure_homed(ctx, arm_id)
    arm = ctx.gate.arm(arm_id)
    p = arm.get_pose()
    safe_z = ctx.bench[arm_id]["safe_z"]
    if p.z < safe_z:
        arm.move_to(Pose(x=p.x, y=p.y, z=safe_z, rotation=p.rotation))


#: extra pick attempts when the camera sees the object still at the source
PICK_RETRIES = 2


def pick_and_place(ctx: FlowContext, arm_id: str, src: Pose, dst: Pose,
                   home: bool = True,
                   still_there: Callable[[], bool] | None = None) -> None:
    """Pick at `src` with the nozzle at 0 deg, place at `dst` with the nozzle
    rotated to cancel the object's measured rotation (v1 rotate_nose).

    `still_there()` (a camera check) runs after each lift at the source: while
    the object is still lying there the pick failed, and the arm tries again
    (up to PICK_RETRIES times) before carrying an empty nozzle across."""
    b = ctx.bench[arm_id]
    arm = ctx.gate.arm(arm_id)
    safe_z, pick_z, place_z = b["safe_z"], b["pick_z"], b["place_z"]
    place_rot = angle_diff(dst.rotation, src.rotation)
    lift(ctx, arm_id)
    for attempt in range(PICK_RETRIES + 1):
        for pose, vac in (
            (Pose(x=src.x, y=src.y, z=safe_z), None),
            (Pose(x=src.x, y=src.y, z=pick_z), True),
            (Pose(x=src.x, y=src.y, z=safe_z), None),
        ):
            ctx.check_cancel()
            move_arm(ctx, arm_id, pose)
            if vac is not None:
                arm.set_vacuum(vac)
        if still_there is None or attempt == PICK_RETRIES or not still_there():
            break
        arm.set_vacuum(False)
        ctx.store.emit("notification", {"level": "info", "text":
                       f"{arm_id}: pick came up empty, picking again"}, run_id=ctx.run_id)
    for pose, vac in (
        (Pose(x=dst.x, y=dst.y, z=safe_z, rotation=place_rot), None),
        (Pose(x=dst.x, y=dst.y, z=place_z, rotation=place_rot), False),
        (Pose(x=dst.x, y=dst.y, z=safe_z, rotation=place_rot), None),
    ):
        ctx.check_cancel()
        move_arm(ctx, arm_id, pose)
        if vac is not None:
            arm.set_vacuum(vac)
    if home:
        move_arm(ctx, arm_id, Pose(**b["home"]))


def place_held(ctx: FlowContext, arm_id: str, dst: Pose, home: bool = True) -> None:
    """Put down what is already on the nozzle (resume after a crash mid-move).
    The object's rotation on the nozzle is unknown, so the nozzle keeps its own."""
    b = ctx.bench[arm_id]
    arm = ctx.gate.arm(arm_id)
    rot = arm.get_pose().rotation
    lift(ctx, arm_id)
    for z in (b["safe_z"], b["place_z"]):
        ctx.check_cancel()
        move_arm(ctx, arm_id, Pose(x=dst.x, y=dst.y, z=z, rotation=rot))
    arm.set_vacuum(False)
    move_arm(ctx, arm_id, Pose(x=dst.x, y=dst.y, z=b["safe_z"], rotation=rot))
    if home:
        move_arm(ctx, arm_id, Pose(**b["home"]))


def massage(ctx: FlowContext, arm_id: str, center: Pose, rounds: int,
            depth_mm: float | None = None) -> None:
    """Press the tape onto the stage column by column, from right to left
    (owner review, 2026-10-22; v1 RobotArmController.massage went row by row from
    the front): each column is one low stroke along y, front to back, at
    x = center.x + span down to center.x - span, with the nozzle lifted to
    safe_z between columns. The pattern covers the same square as v1's.
    `depth_mm`: how far below first contact with the tape the nozzle goes (how
    hard it presses: the rubber tip gives); None = the bench's massage_z."""
    b = ctx.bench[arm_id]
    z = b["massage_z"] if depth_mm is None else b["press_contact_z"] - depth_mm
    span, cols = b["massage_span_mm"], b["massage_columns"]
    span_y = b.get("massage_span_y_mm", span)  # a longer stroke for a tall chip
    lift(ctx, arm_id)
    for _ in range(rounds):
        for i in range(cols):
            x = center.x + span - 2 * span * i / max(1, cols - 1)
            ctx.check_cancel()
            move_arm(ctx, arm_id, Pose(x=x, y=center.y - span_y, z=b["safe_z"]))
            move_arm(ctx, arm_id, Pose(x=x, y=center.y - span_y, z=z))
            move_arm(ctx, arm_id, Pose(x=x, y=center.y + span_y, z=z))
            move_arm(ctx, arm_id, Pose(x=x, y=center.y + span_y, z=b["safe_z"]))
    move_arm(ctx, arm_id, Pose(**b["home"]))


# ----- ledger --------------------------------------------------------------

def free_slots(ctx: FlowContext, slots: list[str]) -> list[str]:
    """Slots no sample in the ledger claims and no other job has booked."""
    used = ctx.store.occupied_slots(exclude_run=ctx.run_id)
    return [s for s in slots if s not in used]


def require_sample(ctx: FlowContext, sample_id: str, kind: str) -> dict[str, Any]:
    s = ctx.store.get_sample(sample_id)
    if s is None or s["kind"] != kind:
        raise PreconditionFailed(f"{sample_id} is not a known {kind}")
    return s


# ----- heaters -------------------------------------------------------------

def sleep_watched(ctx: FlowContext, seconds: float, poll_s: float = 5.0, *,
                  observe: Callable[[], None] | None = None,
                  observation_interval: Callable[[], float] | None = None) -> None:
    """Wait on the clock, letting the safety watchdog and the stop button act."""
    end = ctx.clock.now() + seconds
    while (left := end - ctx.clock.now()) > 0:
        ctx.check_cancel()
        ctx.gate.watchdog_tick()
        interval = observation_interval() if observation_interval is not None else poll_s
        if interval <= 0:
            raise NeedsHumanError("No positive observation interval remains")
        ctx.clock.sleep(min(poll_s, left, interval))
        if observe is not None:
            observe()


def ramp_heater(ctx: FlowContext, heater_id: str, target_c: float, poll_s: float = 5.0, *,
                observe: Callable[[], None] | None = None,
                observation_interval: Callable[[], float] | None = None) -> None:
    """Raise the setpoint to `target_c` no faster than the configured ramp limit."""
    h = ctx.gate.heater(heater_id)
    rate = ctx.gate.limits.heaters[heater_id].max_ramp_c_per_min * 0.95
    while True:
        sp = h.get_target()
        base = sp if sp is not None else h.read_temperature()
        nxt = min(target_c, base + rate) if target_c > base else target_c
        h.set_target(nxt)
        if nxt == target_c:
            return
        sleep_watched(ctx, 60.0, poll_s, observe=observe,
                      observation_interval=observation_interval)


def wait_temperature(ctx: FlowContext, heater_id: str, done: Callable[[float], bool], *,
                     what: str, timeout_s: float, stuck_window_s: float, stuck_delta_c: float,
                     poll_s: float = 5.0, observe: Callable[[], None] | None = None,
                     observation_interval: Callable[[], float] | None = None) -> tuple[float, float]:
    """Poll the heater until done(reading). Returns (reading, seconds waited).

    Raises NeedsHumanError on timeout (v1 waited forever) and DangerousError if
    the reading does not move while it should (spec 4.4: stuck sensor).
    """
    h = ctx.gate.heater(heater_id)
    t0 = ctx.clock.now()
    history: list[tuple[float, float]] = []
    while True:
        ctx.check_cancel()
        ctx.gate.watchdog_tick()
        now, temp = ctx.clock.now(), h.read_temperature()
        if done(temp):
            return temp, now - t0
        if now - t0 > timeout_s:
            raise NeedsHumanError(f"{heater_id}: {what} not reached within {timeout_s:.0f}s "
                                  f"(reading {temp:.1f}C)", heater=heater_id, reading=temp)
        history.append((now, temp))
        old = [tp for t, tp in history if now - t >= stuck_window_s]
        if old and abs(temp - old[-1]) < stuck_delta_c:
            h.off()
            raise DangerousError(f"{heater_id}: reading stuck at {temp:.1f}C for "
                                 f"{stuck_window_s:.0f}s while waiting for {what}; sensor "
                                 f"suspected broken, heater off", heater=heater_id, reading=temp)
        sleep_watched(ctx, poll_s, poll_s, observe=observe,
                      observation_interval=observation_interval)


# ----- microscope ----------------------------------------------------------

def analyzer(ctx: FlowContext) -> MicroscopeAnalyzer:
    if ctx.analyzer is None:
        raise PreconditionFailed("no microscope analyzer configured")
    return ctx.analyzer


def min_sharpness(ctx: FlowContext, objective: int) -> float:
    table = ctx.verification["focus"]["min_sharpness"]
    return float(table.get(objective, table.get(str(objective), 5.0)))


def observed_outline_pixels(frame: Frame, detection: Any) -> list[list[float]]:
    """Project an analyzer's complete observed outline into its captured image."""
    from qumus_simlab.perception.tracking import load_pixels

    if not detection.meta.get("outline_complete") or not detection.meta.get("outline_mm"):
        raise PreconditionFailed("Specimen tracking requires a complete observed optical outline")
    image = load_pixels(frame)
    fw, fh = frame.meta["fov_mm"]
    return [[image.width/2 + (x+detection.dx_mm)*image.width/fw,
             image.height/2 - (y+detection.dy_mm)*image.height/fh]
            for x, y in detection.meta["outline_mm"]]


def microscope_frame(ctx: FlowContext, *, purpose: str = "analysis", phase: str = "",
                     transfer_motion_intent: dict[str, Any] | None = None,
                     on_capture: Callable[[Frame], None] | None = None) -> Frame:
    """Capture immutable pixels and publish their capture-time display metadata.

    A picture shown in the camera card is not a successful verification. The
    existing Step observation remains responsible for that decision. Preview
    polling never calls this function or creates an observation.
    Optional motion intent is copied provenance, never motion or transfer proof.
    """
    frame = ctx.gate.microscope(MICROSCOPE).capture()
    if on_capture is not None:
        on_capture(frame)
    publish_frame(ctx, frame, purpose=purpose, phase=phase,
                  transfer_motion_intent=transfer_motion_intent)
    return frame


def publish_frame(ctx: FlowContext, frame: Frame, *, purpose: str = "analysis", phase: str = "",
                  transfer_motion_intent: dict[str, Any] | None = None) -> None:
    """Save a captured frame and show it in the camera card (lab.frame).

    The 10x scan captures every field in memory and publishes only the fields
    where material was found (#203); every other capture publishes at once.
    """
    path = frame.saved_path()
    ctx.store.emit("lab.frame", {
        "device": MICROSCOPE, "command": "capture",
        "microscope_preview": {
            "image": path, "image_sha256": (hashlib.sha256(
                frame.data if frame.data is not None else Path(path).read_bytes()).hexdigest() if path else None),
            "objective": frame.meta.get("objective"),
            "focus_z": frame.meta.get("focus_z", frame.meta.get("focus")),
            "focus_plane": frame.meta.get("focus_plane"), "timestamp": frame.timestamp,
            "purpose": purpose, "verification": False, "phase": phase,
            "source": "synthetic" if ctx.store.bench == "sim" else "camera",
            "immutable": path is not None, "state_binding": "capture.pose_binding",
            "fov_mm": frame.meta.get("fov_mm"),
            "calibration": frame.meta.get("calibration"),
            "stage_kinematics": frame.meta.get("stage_kinematics"),
            **({"transfer_motion_intent": deepcopy(transfer_motion_intent)}
               if transfer_motion_intent is not None else {}),
        },
    }, run_id=ctx.run_id)


def autofocus(ctx: FlowContext, objective: int, attempts: int = 3,
              half_range: float | None = None, plane: str = "chip", *,
              retain_final_image: bool = False,
              final_frames: list[Frame] | None = None,
              on_capture: Callable[[Frame], None] | None = None) -> float:
    """Sweep the independent microscope Z, leaving sample and stamp unchanged.

    Pixel contrast selects the focus peak. The existing analyzer's configured
    sharpness scale supplies the final quality reading; simulator best-focus
    coordinates never select a motion. A supplied window is never widened, so
    callers switching between two planes can keep their reference binding.
    Optional final-image retention binds downstream identity to that exact
    settled exposure; the sampled sweeps remain numeric records with hashes.
    """
    from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE
    from qumus_simlab.perception.tracking import raster_focus_score

    if plane not in ("chip", "stamp"):
        raise ValueError("focus plane must be chip or stamp")
    cfg = ctx.verification["focus"]
    m = ctx.gate.microscope(MICROSCOPE)
    lim = ctx.gate.limits.microscopes[MICROSCOPE].focus
    center = m.get_focus()
    an = analyzer(ctx)
    half = cfg["coarse_half_range"] * 10.0 / objective if half_range is None else half_range
    if half_range is not None:
        attempts = 1
    def capture():
        frame = m.capture(purpose="focus_capture")
        if on_capture is not None:
            on_capture(frame)
        return frame
    baseline = capture()
    color_focus = (plane == "chip" and
                   baseline.meta.get("calibration", {}).get("material_capture_profile")
                   == V1_MATERIAL_CAPTURE_PROFILE)
    final_binding = _focus_capture_binding(ctx, m, baseline)
    try:
        best_pixels = raster_focus_score(baseline, color=color_focus)
    except (OSError, ValueError) as exc:
        raise PreconditionFailed("Autofocus requires captured raster pixels") from exc
    best_z, best_s = center, an.sharpness(baseline)
    observable = False
    passes: list[dict[str, Any]] = []
    final_frame = baseline
    for attempt in range(attempts):
        for phase, n, h in (("coarse", cfg["coarse_points"], half),
                            ("fine", cfg["fine_points"], 2 * half / cfg["coarse_points"])):
            c = best_z if best_s >= 0 else center
            curve: list[tuple[float, float]] = []
            samples: list[dict[str, Any]] = []
            for i in range(n):
                z = min(max(c - h + 2 * h * i / (n - 1), lim[0]), lim[1])
                ctx.check_cancel()
                m.set_focus(z)
                frame = capture()
                sample_binding = _focus_capture_binding(ctx, m, frame)
                try:
                    pixels = raster_focus_score(frame, color=color_focus)
                except (OSError, ValueError) as exc:
                    raise PreconditionFailed("Autofocus requires captured raster pixels") from exc
                curve.append((z, pixels))
                samples.append({"requested_focus_z_mm": z, "raster_score": pixels,
                                **sample_binding})
            peak = max(p for _, p in curve)
            record: dict[str, Any] = {
                "attempt": attempt, "phase": phase, "samples": samples,
                "threshold_score": peak * (1 - cfg["minimum_peak_gain"]),
                "selected": False, "selected_plateau_mm": None, "bracket_mm": None,
                "unresolved_peaks": False,
            }
            if peak > 0 and peak >= best_pixels * (1 - cfg["minimum_peak_gain"]):
                # Pixel blur can quantize a sharp region into a plateau. Use
                # its centre, not its first edge; later objective changes have
                # a narrower depth of field. Uniform noisy fields have no peak.
                groups: list[list[float]] = [[]]
                for z, pixels in curve:
                    if pixels >= peak * (1 - cfg["minimum_peak_gain"]):
                        groups[-1].append(z)
                    elif groups[-1]:
                        groups.append([])
                candidates = [g for g in groups if g]
                chosen = min(candidates, key=lambda g: abs((g[0] + g[-1]) / 2 - c))
                bracketed = chosen[0] > curve[0][0] and chosen[-1] < curve[-1][0]
                best_z, best_pixels = (chosen[0] + chosen[-1]) / 2, peak
                observable = observable or bracketed
                first = next(i for i, (z, _) in enumerate(curve) if z == chosen[0])
                last = max(i for i, (z, _) in enumerate(curve) if z == chosen[-1])
                record.update(selected=True, selected_plateau_mm=[samples[first]["readback_focus_z_mm"],
                                                                 samples[last]["readback_focus_z_mm"]],
                              unresolved_peaks=len(candidates) != 1)
                # The neighboring below-threshold observations bound the
                # plateau conservatively. A fine pass with no such neighbors
                # cannot invent a narrower interval than an earlier pass.
                if bracketed and len(candidates) == 1:
                    record["bracket_mm"] = [samples[first-1]["readback_focus_z_mm"],
                                            samples[last+1]["readback_focus_z_mm"]]
            passes.append(record)
            m.set_focus(best_z)
            final_frame = capture()
            final_binding = _focus_capture_binding(ctx, m, final_frame)
            best_s = an.sharpness(final_frame)
        # A v1 camera keeps its best frame once the contrast sweep has settled
        # (camera_controller.wait_focus_and_snap); it does not widen the range
        # because a bare field scores under min_blur.
        settled = best_s >= min_sharpness(ctx, objective) or getattr(an, "keeps_best_focus_frame", False)
        if settled and (observable or best_pixels == 0):
            break
        half *= 2
    binding = final_binding
    axis_calibration = binding["focus_axis_calibration"]
    interval, interval_pass = _observed_focus_interval(passes, binding, axis_calibration, objective)
    final_readback = m.get_focus()
    if final_readback != binding["readback_focus_z_mm"]:
        interval, interval_pass = None, None
    focus_observation = {
        "objective": objective, "plane": plane, "focus_z": best_z,
        "pixel_score": best_pixels, "peak_observable": observable,
        "method": "raster_peak" if observable else "retained_previous_plane",
        "verification": False,
        "focus_metric": {"id": "raster-focus-rgb-v1" if color_focus else "raster-focus-gray-v1"},
        "boundary_final_identity": None,
        **binding,
        "final_focus_readback_z_mm": final_readback,
        "image": final_frame.saved_path() if retain_final_image else None,
        "sampled_passes": passes, "focus_interval_mm": interval,
        "interval_pass_index": interval_pass,
    }
    ctx.store.update_run_meta(ctx.run_id, last_focus_observation=focus_observation)
    event_id = ctx.store.emit("microscope.focus", focus_observation, run_id=ctx.run_id)
    ctx.store.update_run_meta(ctx.run_id, last_focus_event_id=event_id)
    if final_frames is not None:
        final_frames.append(final_frame)  # Exact exposure for downstream analyzers; never reconstructed scene metadata.
    return best_s


def _focus_capture_binding(ctx: FlowContext, microscope: Any, frame: Frame) -> dict[str, Any]:
    """Freeze capture optics and public readbacks, never hidden scene geometry."""
    geometry = ctx.gate.limits.geometry
    stage, holder = None, None
    if geometry is not None:
        stage = ctx.gate.stage(geometry.stage).kinematics()
        holder = ctx.gate.manipulator(geometry.manipulator).get_positions()
    pixels = frame.data
    if pixels is None and frame.path is not None:
        pixels = Path(frame.path).read_bytes()
    objective = microscope.get_objective()
    return deepcopy({
        "readback_focus_z_mm": microscope.get_focus(), "captured_at": frame.timestamp,
        "image_sha256": hashlib.sha256(pixels).hexdigest() if pixels is not None else None,
        "objective_readback": objective,
        "focus_plane": frame.meta.get("focus_plane"),
        "focus_axis_calibration": microscope.focus_axis_calibration(objective),
        "optical_calibration": frame.meta.get("calibration"),
        "fov_mm": frame.meta.get("fov_mm"),
        "stage_kinematics": stage, "holder_pose": holder,
    })


def _observed_focus_interval(passes: list[dict[str, Any]], binding: dict[str, Any],
                             calibration: dict[str, Any] | None,
                             objective: int) -> tuple[list[float] | None, int | None]:
    """Admit only a unique sampled bracket containing the final focus readback."""
    if (calibration is None or binding["objective_readback"] != objective
            or not isinstance(binding["optical_calibration"], dict)):
        return None, None
    final_z = binding["readback_focus_z_mm"]
    if isinstance(final_z, bool) or not isinstance(final_z, (int, float)) or not math.isfinite(final_z):
        return None, None
    if any(p["selected"] and p["unresolved_peaks"] for p in passes):
        return None, None
    candidates = []
    for index, record in enumerate(passes):
        bracket = record["bracket_mm"]
        samples = record["samples"]
        if bracket is None or not all(
            isinstance(s["readback_focus_z_mm"], (int, float))
            and not isinstance(s["readback_focus_z_mm"], bool)
            and math.isfinite(s["readback_focus_z_mm"])
            and s["objective_readback"] == objective
            and s["focus_axis_calibration"] == calibration
            and s["optical_calibration"] == binding["optical_calibration"]
            and s["stage_kinematics"] == binding["stage_kinematics"]
            and s["holder_pose"] == binding["holder_pose"] for s in samples
        ):
            continue
        zs = [s["readback_focus_z_mm"] for s in samples]
        if (not all(a < b for a, b in zip(zs, zs[1:], strict=False))
                or not bracket[0] <= final_z <= bracket[1]):
            continue
        error = calibration["readback_error_mm"]
        candidates.append((bracket[1]-bracket[0], index,
                           [bracket[0]-error, bracket[1]+error]))
    if not candidates:
        return None, None
    _, index, interval = min(candidates)
    return interval, index
