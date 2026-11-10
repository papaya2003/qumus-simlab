"""Conditional SIM discovery policy and finite public-observation geometry.

No World, device or flow is imported. Gaussian tails describe only positions;
visible completeness and precontact recognition remain explicit assumptions.
"""
from __future__ import annotations

import math
from copy import deepcopy
from statistics import NormalDist

PHYSICS_PROFILE = "synthetic-local-paraboloid-r12-cap5-v1"
DISCOVERY_PROFILE = "synthetic-clear-domain-first-contact-v1"
OBSTACLE_PROFILE = "synthetic-complete-visible-obstacle-outline-v1"
FOCUS_SUMMARY_FIELDS = ("objective", "objective_readback", "plane", "focus_plane", "peak_observable",
    "focus_interval_mm", "focus_axis_calibration", "final_focus_readback_z_mm", "readback_focus_z_mm",
    "optical_calibration", "stage_kinematics", "holder_pose", "image_sha256", "captured_at")


def discovery_policy():
    return {"id": DISCOVERY_PROFILE, "source": "synthetic", "simulation_only": True,
        "physics_profile": PHYSICS_PROFILE, "curvature_radius_mm": 12.,
        "cap_radius_mm": 5., "front_tau_s": .8, "clear_radius_mm": 1.10,
        "map_offset_mm": 1.50, "chip_proposal_inset_mm": 1.20,
        "proposal_rule": "eight-rays-projected-interior-min-distance-stable-order-v1",
        "candidate_offsets_mm": [[0.,0.],[-.4,0.],[.4,0.],[0.,-.1],[0.,.1],[-.4,-.1],[-.4,.1],[.4,-.1],[.4,.1]],
        "map_offsets_mm": [[-.65, -.43], [-.65, .43], [.65, -.43], [.65, .43]],
        "map_frames": 4, "max_obstacles_per_frame": 64, "alpha_apex_position": .001,
        "alpha_obstacle_positions": .001, "alpha_stage_positions": .001,
        "stage_sigma_mm": .002, "stage_landings": 5, "apex_sigma_mm": .08,
        "absolute_drift_per_axis_mm": .15, "raster_guard_mm": .065,
        "native_diagonal_guard_mm": math.hypot(2.6/480, 1.72/320),
        "hold_s": 5., "hold_poll_s": 1.,
        "scope": "Conditional visible-completeness/precontact recognition; no hard physical or total-confidence guarantee"}


def validate_policy(value):
    if value != discovery_policy():
        raise ValueError("Unsupported or changed fixed stamp-discovery policy")
    return deepcopy(value)


def settling_envelope_mm(policy, *, max_command_mm=.00025, readback_mm=.005,
                         max_seconds=900., max_travel_mm=8.):
    validate_policy(policy)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in (max_command_mm, readback_mm, max_seconds, max_travel_mm)):
        raise ValueError("Discovery needs finite original command/phase bounds")
    # Only the fixed low-Z interval can increase the original candidate's
    # maximum. Runtime/journal admission separately proves AF/hold caps and
    # the first >=15um actual clear-lift progress. Full phase drift remains
    # the independent bound after that strictly larger clear displacement.
    first_lift_seconds = (.025-readback_mm)/.1+.05
    low_seconds = .1+1.538889+5.30+first_lift_seconds
    drift = (math.hypot(.00002, .00001)*low_seconds
             +math.hypot(.003, .002)*(max_command_mm+readback_mm+.025))
    depth = max_command_mm+readback_mm+math.sqrt(2)*.01*drift
    contact = math.sqrt(2*policy["curvature_radius_mm"]*depth)
    apex = math.sqrt(2)*NormalDist().inv_cdf(1-policy["alpha_apex_position"]/4)*policy["apex_sigma_mm"]
    z = NormalDist().inv_cdf(1-policy["alpha_obstacle_positions"]/(4*64*4))
    obstacle = math.sqrt(2)*z*.001
    zstage = NormalDist().inv_cdf(1-policy["alpha_stage_positions"]/(4*policy["stage_landings"]))
    stage = 2*math.sqrt(2)*zstage*policy["stage_sigma_mm"]
    radius = contact+apex+math.sqrt(2)*.15+obstacle+stage+math.sqrt(2)*policy["raster_guard_mm"]+policy["native_diagonal_guard_mm"]
    return {"settled_front_radius_mm": contact, "conditional_depth_bound_mm": depth,
            "apex_prior_radius_mm": apex, "obstacle_position_radius_mm": obstacle,
            "relative_stage_position_radius_mm": stage, "low_z_seconds_bound": low_seconds,
            "raster_radial_guard_mm": math.sqrt(2)*policy["raster_guard_mm"],
            "remaining_phase_height_change_mm": math.sqrt(2)*.01*(math.hypot(.00002,.00001)*max_seconds+math.hypot(.003,.002)*max_travel_mm),
            "minimum_clear_radius_mm": radius, "admitted": radius <= policy["clear_radius_mm"],
            "scope": policy["scope"]}


def point_segment_distance(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    q = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return math.hypot(point[0]-a[0]-q*dx, point[1]-a[1]-q*dy)


def disk_clearance(center, polygon):
    """Signed convex interior clearance, with complete outlines required upstream."""
    from qumus_simlab.core.polygons import convex_polygon, cross
    p = convex_polygon(polygon)
    distance = min(point_segment_distance(center, a, b) for a, b in zip(p, p[1:]+p[:1], strict=True))
    return distance if all(cross(a, b, center) >= 0 for a, b in zip(p, p[1:]+p[:1], strict=True)) else -distance


def map_centre(source_local, size_mm, rotation_deg):
    """One fixed interior proposal; observations alone certify its clear domain."""
    x, y = source_local
    w, h = size_mm
    policy = discovery_policy()
    inset = policy["chip_proposal_inset_mm"]
    if min(w, h) <= 2*inset:
        raise ValueError("Chip has no finite discovery proposal interior")
    directions = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.),
                  (2**-.5, 2**-.5), (-2**-.5, 2**-.5),
                  (2**-.5, -2**-.5), (-2**-.5, -2**-.5)]
    candidates = []
    for i, (dx, dy) in enumerate(directions):
        target = (max(inset, min(w-inset, x+policy["map_offset_mm"]*dx)),
                  max(inset, min(h-inset, y+policy["map_offset_mm"]*dy)))
        distance = math.dist((x, y), target)
        if distance >= policy["map_offset_mm"]-1e-12:
            candidates.append((distance, i, target))
    if not candidates:
        raise ValueError("No fixed source-separated discovery proposal fits the chip")
    _, _, target = min(candidates)
    dx, dy = target[0]-x, target[1]-y
    a = math.radians(rotation_deg)
    return (dx*math.cos(a)-dy*math.sin(a), dx*math.sin(a)+dy*math.cos(a))



def validate_generator_config(config):
    """Reject overrides before publishing a fixed recognition/mechanics claim."""
    stamp, scope = config.get("stamp", {}), config.get("microscope", {})
    policy = validate_policy(stamp.get("discovery"))
    expected = {"dome_radius_mm": 12., "cap_support_radius_mm": 5.,
                "physics_profile": PHYSICS_PROFILE, "apex_sigma_mm": .08}
    dynamics = {"contact_front_tau_s": .8, "maximum_drift_mm": .15}
    # Declared drift rates are upper bounds per axis: a steadier SIM stamp
    # (Jialei 2026-11-03: small errors in SIM) stays inside the envelope.
    drift_bounds = {"lateral_drift_mm_s": [.00002, -.00001], "approach_drift_mm_per_mm": [.003, -.002]}

    def within(value, bound):
        return (isinstance(value, (list, tuple)) and len(value) == 2
                and all(type(v) in (int, float) and math.isfinite(v) and v*b >= 0 and abs(v) <= abs(b)
                        for v, b in zip(value, bound, strict=True)))
    if (any(stamp.get(k) != v for k, v in expected.items())
            or any(stamp.get("dynamics", {}).get(k) != v for k, v in dynamics.items())
            or not all(within(stamp.get("dynamics", {}).get(k), b) for k, b in drift_bounds.items())
            or scope.get("focus_tilt_per_mm") != .01
            or scope.get("observation_noise_scale", 1.) != 1.
            # The declared stage sigma is an upper bound: a quieter stage
            # (Jialei 2026-11-03: the real stage error does not accumulate)
            # stays inside the declared envelope; a noisier one does not.
            or not 0 <= config.get("stage", {}).get("sigma_mm", .002) <= .002
            or config.get("motion", {}).get("focus_speed_mm_s", 1.) != 1.
            or config.get("motion", {}).get("stage_speed_mm_s", 10.) != 10.):
        raise ValueError("R017 discovery declarations differ from the actual configured generators")
    return policy


def _disk_in_rect_union(center, radius, rectangles):
    left, right = center[0]-radius, center[0]+radius
    cuts = sorted({left, right, *[max(left, min(right, r[i])) for r in rectangles for i in (0, 2)]})
    for a, b in zip(cuts, cuts[1:], strict=False):
        if b-a <= 1e-12:
            continue
        middle = (a+b)/2
        intervals = sorted((r[1], r[3]) for r in rectangles if r[0] <= middle <= r[2])
        merged = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        dx = max(a-center[0], center[0]-b, 0.)
        height = math.sqrt(max(0., radius*radius-dx*dx))
        if not any(lo <= center[1]-height and hi >= center[1]+height for lo, hi in merged):
            return False
    return bool(rectangles)


def choose_clear_domain(policy, maps, *, nominal_center, chip_polygon, chip_id, exact_center=False):
    """One finite deterministic decision from FOUR complete public observations.

    map.camera_center_mm is in original-source camera axes, derived from actual
    stage readbacks. Unknown ROI, holes and classes fail closed; a flake cut by a
    map's frame edge counts by its in-frame part (the disk stays inside the maps).
    """
    validate_policy(policy)
    if len(maps) != 4:
        raise ValueError("Discovery requires exactly four original map exposures")
    from qumus_simlab.perception.observations import RGB_DISCOVERY_PROFILE

    rgb_maps = [item.get("observation", {}).get("profile") == RGB_DISCOVERY_PROFILE for item in maps]
    if any(rgb_maps):
        if not all(rgb_maps):
            raise ValueError("Discovery cannot mix pixel support with historical annotated maps")
        return _choose_rgb_clear_domain(policy, maps, nominal_center=nominal_center,
            chip_polygon=chip_polygon, exact_center=exact_center)
    rectangles, obstacles = [], []
    hashes = set()
    for item in maps:
        obs = item["observation"]
        if (obs.get("profile") != OBSTACLE_PROFILE or obs.get("source") != "synthetic"
                or obs.get("conditional_visible_completeness") is not True
                or obs.get("unknown_outside_roi") is not True or obs.get("chip_id") != chip_id
                or obs.get("calibration", {}).get("stamp_discovery") != policy
                or obs["calibration"].get("source") != "synthetic"
                or obs["calibration"].get("image_size_px") != [480, 320]
                or obs["calibration"].get("mm_per_pixel") != [2.6/480, 1.72/320]
                or obs["calibration"].get("u_axis") != "+world_x" or obs["calibration"].get("v_axis") != "-world_y"
                or obs.get("centroid_sigma_mm") != .001
                or len(obs.get("obstacles", [])) > 64
                or obs.get("image_sha256") in hashes):
            raise ValueError("Discovery map lacks its distinct complete supported observation")
        hashes.add(obs.get("image_sha256"))
        cx, cy = item["camera_center_mm"]
        x0, y0, x1, y1 = obs["supported_roi_px"]
        if [x0, y0, x1, y1] != [8, 30, 472, 288]:
            raise ValueError("Discovery unknown camera support changed")
        sx, sy = 2.6/480, 1.72/320
        rectangles.append((cx+(x0-240)*sx, cy-(y1-160)*sy,
                           cx+(x1-240)*sx, cy-(y0-160)*sy))
        for row, left, right in obs.get("unknown_runs_px", []):
            # Every original unknown cell stays excluded; no hull fills holes.
            x0, x1 = cx+(left-240-.5)*sx, cx+(right-240+.5)*sx
            y0, y1 = cy-(row-160+.5)*sy, cy-(row-160-.5)*sy
            obstacles.append([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
        frame = (-2.6/2, -1.72/2, 2.6/2, 1.72/2)
        for obstacle in obs["obstacles"]:
            outline = obstacle.get("outline_mm") or []
            if obstacle.get("complete") is not True:
                # Only a flake cut by this map's own frame edge: its in-frame
                # part is an obstacle; the rest lies outside this map and is
                # either seen by another map or outside every supported
                # rectangle the clear disk must stay within.
                if (obstacle.get("complete") is not False or obstacle.get("clipped_at") != "frame_edge"
                        or len(outline) < 3 or not all(frame[0]-.01 <= x <= frame[2]+.01
                                                       and frame[1]-.01 <= y <= frame[3]+.01 for x, y in outline)):
                    raise ValueError("Incomplete obstacle cannot certify clear background")
            obstacles.append([(cx+x, cy+y) for x, y in outline])
    offsets = [[0., 0.]] if exact_center else policy["candidate_offsets_mm"]
    radius = policy["clear_radius_mm"]
    for dx, dy in offsets:
        center = (nominal_center[0]+dx, nominal_center[1]+dy)
        if (disk_clearance(center, chip_polygon) >= radius
                and _disk_in_rect_union(center, radius, rectangles)
                and all(disk_clearance(center, outline) < -radius for outline in obstacles)):
            return {"profile": DISCOVERY_PROFILE, "center_mm": list(center), "clear_radius_mm": radius,
                    "map_image_sha256": [m["observation"]["image_sha256"] for m in maps],
                    "scope": policy["scope"]}
    raise ValueError("No fixed observed all-material clear discovery domain fits the chip/camera support")


def _choose_rgb_clear_domain(policy, maps, *, nominal_center, chip_polygon, exact_center):
    """Exclude every original unknown cell without a centroid noise model.

    Public stage readbacks bind map centres at the caller. The fixed native5
    support profile is conditional on visible material; no subresolution or
    invisible-material completeness is inferred from a bare-compatible cell.
    """
    import hashlib
    import json
    import re

    from qumus_simlab.core.optical_profiles import CAMERA_RESPONSE_PROFILE
    from qumus_simlab.perception.observations import RGB_DISCOVERY_PROFILE, rgb_discovery_policy

    expected_policy = rgb_discovery_policy()
    policy_sha = hashlib.sha256(json.dumps(expected_policy, sort_keys=True, allow_nan=False).encode()).hexdigest()
    rectangles, unknown, hashes, bounds = [], [], set(), []
    for item in maps:
        obs = item["observation"]
        calibration = obs.get("calibration", {})
        photometry = calibration.get("photometry_profile", {})
        noise = calibration.get("observation_condition", {}).get("observation_noise_scale")
        image_sha = obs.get("image_sha256")
        sx, sy = 2.6/480, 1.72/320
        bound = math.hypot(sx, sy)/2
        if (obs.get("profile") != RGB_DISCOVERY_PROFILE or obs.get("source") != "synthetic"
                or obs.get("status") != "supported" or obs.get("conditional_visible_completeness") is not True
                or obs.get("unknown_outside_roi") is not True
                or obs.get("observation_policy") != expected_policy or obs.get("observation_policy_sha256") != policy_sha
                or calibration.get("stamp_discovery") != policy or calibration.get("source") != "synthetic"
                or calibration.get("image_size_px") != [480, 320] or calibration.get("mm_per_pixel") != [sx, sy]
                or calibration.get("u_axis") != "+world_x" or calibration.get("v_axis") != "-world_y"
                or photometry.get("source") != "synthetic"
                or photometry.get("camera_response") != CAMERA_RESPONSE_PROFILE
                or photometry.get("chip_surface") != "synthetic-chip-footprint-v2"
                or type(noise) not in (int, float) or not math.isfinite(noise) or not 0 <= noise <= 1
                or obs.get("boundary_cell_radius_mm") != bound
                or obs.get("supported_roi_px") != expected_policy["supported_roi_px"]
                or obs.get("obstacles") != [] or "centroid_sigma_mm" in obs
                or not isinstance(image_sha, str) or re.fullmatch(r"[0-9a-f]{64}", image_sha) is None
                or image_sha in hashes
                or not isinstance(obs.get("unknown_mask_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", obs["unknown_mask_sha256"]) is None):
            raise ValueError("Discovery RGB map lacks its original conditional pixel-support profile")
        hashes.add(image_sha)
        bounds.append(bound)
        centre = item.get("camera_center_mm")
        if (not isinstance(centre, (tuple, list)) or len(centre) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in centre)):
            raise ValueError("Discovery RGB map lacks finite public stage registration")
        cx, cy = centre
        x0, y0, x1, y1 = obs["supported_roi_px"]
        rectangles.append((cx+(x0-240-.5)*sx, cy-(y1-160-.5)*sy,
                           cx+(x1-240-.5)*sx, cy-(y0-160-.5)*sy))
        runs = obs.get("unknown_runs_px")
        if not isinstance(runs, list) or len(runs) > (y1-y0)*((x1-x0+1)//2):
            raise ValueError("Discovery RGB unknown-cell runs are absent or exceed the native raster")
        previous = None
        for run in runs:
            if (not isinstance(run, list) or len(run) != 3 or any(type(v) is not int for v in run)
                    or not y0 <= run[0] < y1 or not x0 <= run[1] < run[2] <= x1
                    or previous is not None and (run[0] < previous[0]
                        or run[0] == previous[0] and run[1] <= previous[2])):
                raise ValueError("Discovery RGB unknown-cell runs are malformed, overlapping or unordered")
            row, left, right = run
            previous = run
            xa, xb = cx+(left-240-.5)*sx, cx+(right-240-.5)*sx
            ya, yb = cy-(row-160+.5)*sy, cy-(row-160-.5)*sy
            unknown.append([(xa, ya), (xb, ya), (xb, yb), (xa, yb)])
    # The measured pixel cells are exact. A finite half-cell diagonal allowance
    # keeps visible boundary sampling separate from all stage/apex assumptions.
    radius = policy["clear_radius_mm"]
    exclusion_radius = radius+max(bounds)
    offsets = [[0., 0.]] if exact_center else policy["candidate_offsets_mm"]
    for dx, dy in offsets:
        center = (nominal_center[0]+dx, nominal_center[1]+dy)
        if (disk_clearance(center, chip_polygon) >= exclusion_radius
                and _disk_in_rect_union(center, exclusion_radius, rectangles)
                and all(disk_clearance(center, outline) < -exclusion_radius for outline in unknown)):
            return {"profile": DISCOVERY_PROFILE, "observation_profile": RGB_DISCOVERY_PROFILE,
                "center_mm": list(center), "clear_radius_mm": radius,
                "observed_exclusion_radius_mm": exclusion_radius,
                "boundary_cell_radius_mm": max(bounds), "obstacle_position_model": "original_unknown_pixel_cells_no_centroid_noise",
                "map_image_sha256": [m["observation"]["image_sha256"] for m in maps],
                "scope": policy["scope"], "observation_scope": expected_policy["scope"]}
    raise ValueError("No fixed observed RGB visible-cell clear discovery domain fits the chip/camera support")



def nominal_head_projection(row):
    """Typed nominal-holder/camera geometry only; sampled stamp state is ignored."""
    from qumus_simlab.records.release_support import _encoder_projection

    basic = _encoder_projection(row)
    head = row["payload"]["geometry"]["microscope"].get("stamp_holder_xyz")
    if not isinstance(head, dict) or any(type(head.get(k)) not in (int, float) or not math.isfinite(head[k]) for k in ("x", "y", "z")):
        raise ValueError("Original encoder lacks finite nominal-holder world geometry")
    return {**basic, "profile": "synthetic-nominal-head-camera-v1",
            "nominal_head_world_xyz_mm": {k: head[k] for k in ("x", "y", "z")}}

def _validate_ready_receipt(events, *, run_id, params, discovery, placement, digest):
    """One original-event authority for live, saved and exported new protocols.

    No files, devices, World, store or mutable configuration are consulted.
    Actual image bytes are separately checked by the live capture adapter.
    """
    policy = validate_policy(params.get("stamp_discovery_policy"))
    if not params.get("device_design") or discovery.get("policy") != policy or discovery.get("state") != "ready":
        raise ValueError("No complete original linked discovery/restore admission")
    index = {row["id"]: row for row in events}
    if (len(index) != len(events) or any(type(row["id"]) is not int or row["id"] <= 0
            or type(row.get("ts")) not in (int, float) or not math.isfinite(row["ts"])
            or not isinstance(row.get("payload"), dict) for row in events)):
        raise ValueError("Discovery journal duplicates original event identities")

    def receipt(ref, kind):
        row = index.get(ref.get("event_id")) if isinstance(ref, dict) else None
        if (not row or row.get("kind") != kind or row.get("run_id") != run_id
                or digest(row.get("payload")) != ref.get("event_sha256")):
            raise ValueError("Original discovery receipt is absent, foreign or changed")
        return row

    def frame_record(record, ref):
        row = receipt(ref, "lab.frame")
        p = row["payload"].get("microscope_preview", {})
        if (p.get("purpose") != "analysis" or p.get("immutable") is not True
                or p.get("verification") is not False or p.get("source") != "synthetic"
                or p.get("image_sha256") != record.get("sha256") or p.get("image") != record.get("image")
                or row["ts"] != record.get("captured_at") or p.get("timestamp") != row["ts"]
                or any(p.get(k) != record["meta"].get(k) for k in
                       ("calibration", "objective", "focus_z", "focus_plane", "stage_kinematics", "fov_mm"))):
            raise ValueError("Discovery image lost its immutable original exposure/context")
        return row

    def focus_record(record):
        binding = record["focus_binding"]
        af = receipt({"event_id": binding["event_id"], "event_sha256": binding["event_sha256"]}, "microscope.focus")
        capture = frame_record(record, {"event_id": binding["capture_event_id"],
                                        "event_sha256": binding["capture_event_sha256"]})
        if (af["id"] >= capture["id"] or af["payload"].get("objective") != 20
                or record["meta"].get("objective") != 20
                or record["meta"].get("fov_mm") != [.65, .43]
                or record["meta"]["calibration"].get("image_size_px") != [480, 320]
                or record["meta"]["calibration"].get("mm_per_pixel") != [.65/480, .43/320]
                or af["payload"].get("plane") != "chip" or af["payload"].get("peak_observable") is not True
                or af["payload"].get("optical_calibration") != record["meta"]["calibration"]
                or af["payload"].get("stage_kinematics") != record["meta"]["stage_kinematics"]
                or binding.get("observation") != {key: af["payload"][key] for key in FOCUS_SUMMARY_FIELDS}
                or binding.get("optical_path") != "parked_unobstructed"):
            raise ValueError("Discovery/source focus lost its original parked native20 identity")
        return capture

    ready = receipt(discovery.get("ready_evidence"), "lab.evidence")
    expected = {k: v for k, v in discovery.items() if k != "ready_evidence"}
    if (ready["payload"].get("property") != "stamp_discovery_ready"
            or ready["payload"].get("subject") != params.get("stamp_id")
            or ready["payload"].get("value") != {"discovery_sha256": digest(expected), "ready_placement": placement}
            or discovery.get("ready_placement") != placement):
        raise ValueError("Ready placement differs from the original verified discovery assertion")
    first = discovery["start_event_id"]
    rows = [index[k] for k in sorted(index) if first <= k <= ready["id"]]
    if [row["id"] for row in rows] != list(range(first, ready["id"]+1)):
        raise ValueError("Discovery original GLOBAL journal interval is incomplete")
    if any(a["ts"] > b["ts"] for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("Discovery original Clock chronology changed")
    start = index[first]
    if start["kind"] != "step.started" or start["payload"].get("name") != "prepare_stamp_discovery" or start.get("run_id") != run_id:
        raise ValueError("Discovery lacks its original checkpoint start")
    submitted = [r for r in events if r.get("kind") == "job.submitted" and r.get("run_id") == run_id]
    if (len(submitted) != 1 or submitted[0]["id"] >= first
            or submitted[0]["payload"]["params"].get("stamp_discovery_policy") != policy):
        raise ValueError("Discovery policy was not frozen in the original job submission")
    if discovery["source_id"] != params["flake_ids"][0] or discovery["chip_id"] != params["flake_chips"][discovery["source_id"]]:
        raise ValueError("Discovery changed the originally selected first source")
    original = focus_record(discovery["original_reference"])
    focused = focus_record(discovery["focus_reference"])
    restored = focus_record(discovery["restored_reference"])
    if not original["id"] < first < focused["id"] < restored["id"] < ready["id"]:
        raise ValueError("Original source/discovery/restoration chronology changed")
    # Three separate exposures, not three distinct bit patterns: a precise
    # stage returns to the original pose, so the deterministic SIM restore
    # image may equal the original bit for bit. Each must be its own file
    # from its own later microscope capture command.
    exposures = (original, focused, restored)
    if (len({discovery[k]["image"] for k in ("original_reference", "focus_reference", "restored_reference")}) != 3
            or not all(_separate_capture(events, a, b) for a, b in zip(exposures, exposures[1:], strict=False))
            or discovery["original_reference"]["meta"]["stage_kinematics"]["sample_pose"] != discovery["original_stage_pose"]
            or discovery["focus_reference"]["meta"]["stage_kinematics"]["sample_pose"] != discovery["discovery_stage_pose"]):
        raise ValueError("Original source, discovery and restoration contexts were conflated")
    maps = []
    for item in discovery["maps"]:
        capture = frame_record(item["frame"], item["capture"])
        observed = receipt(item["observation_event"], "lab.evidence")
        body = observed["payload"]
        obs = body.get("value", {})
        if (not first < capture["id"] < observed["id"] < focused["id"]
                or item["frame"]["meta"].get("objective") != 5
                or item["frame"]["meta"].get("fov_mm") != [2.6, 1.72]
                or body.get("property") != "stamp_discovery_obstacles" or body.get("subject") != discovery["chip_id"]
                or body.get("source", {}).get("event_ids") != [capture["id"]]
                or body["source"].get("sha256") != item["frame"]["sha256"]
                or obs.get("image_sha256") != item["frame"]["sha256"]
                or obs.get("captured_at") != capture["ts"]):
            raise ValueError("Map observation is not bound to its original same-exposure pixels")
        stage = item["frame"]["meta"]["stage_kinematics"]["sample_pose"]
        if item["camera_center_mm"] != [discovery["original_stage_pose"][key]-stage[key] for key in ("x", "y")]:
            raise ValueError("Discovery map centre differs from its original stage readback")
        maps.append({"observation": obs, "camera_center_mm": item["camera_center_mm"]})
    domain = choose_clear_domain(policy, maps, nominal_center=discovery["nominal_center_mm"],
                                 chip_polygon=discovery["chip_polygon_mm"], chip_id=discovery["chip_id"])
    if domain != discovery["domain"]:
        raise ValueError("Original four-map clear-domain decision changed")
    from qumus_simlab.records.release_support import _encoder_projection

    head_refs = discovery["actual_head_reference"]
    head_capture = frame_record(discovery["first_contact_reference"], head_refs["capture"])
    head_row = index[head_refs["encoder"]["event_id"]]
    head = nominal_head_projection(head_row)
    if (head_row.get("run_id") != run_id or head_row.get("kind") != "lab.frame"
            or head_row["id"] >= head_capture["id"] or head_row["ts"] != head_capture["ts"]
            or head_refs["encoder"].get("profile") != _encoder_projection(head_row)["profile"]
            or head_refs["encoder"].get("projection_sha256") != digest(_encoder_projection(head_row))
            or head != discovery["actual_head_context"]
            or not focused["id"] < head_capture["id"]
            or discovery["first_contact_reference"]["meta"]["stage_kinematics"] != discovery["focus_reference"]["meta"]["stage_kinematics"]):
        raise ValueError("Actual nominal head lost its original same-exposure public context")
    stage = discovery["first_contact_reference"]["meta"]["stage_kinematics"]["sample_pose"]
    centre = [discovery["original_stage_pose"][key]-stage[key]
              +head["nominal_head_world_xyz_mm"][key]-head["camera_world_xy_mm"][i]
              for i, key in enumerate(("x", "y"))]
    head_domain = choose_clear_domain(policy, maps, nominal_center=centre,
        chip_polygon=discovery["chip_polygon_mm"], chip_id=discovery["chip_id"], exact_center=True)
    if head_domain != discovery["actual_head_domain"]:
        raise ValueError("Actual nominal-head clear domain differs from the original maps")
    candidate_fact = receipt(discovery["candidate_evidence"], "lab.evidence")
    candidate = discovery["first_candidate"]
    if (candidate_fact["payload"].get("property") != "first_contact_region"
            or candidate_fact["payload"].get("value") != candidate
            or candidate_fact["payload"].get("subject") != params["stamp_id"]
            or not focused["id"] < head_capture["id"] < candidate_fact["id"] < restored["id"]
            or candidate.get("reference") != discovery["first_contact_reference"]
            or any(candidate["current"]["meta"].get(key) != candidate["reference"]["meta"].get(key)
                   for key in ("calibration", "stage_kinematics", "objective", "focus_z", "focus_plane"))):
        raise ValueError("First-contact candidate was replaced by an after-restore placement")
    if (placement.get("candidate_evidence") != discovery["candidate_evidence"]
            or placement.get("first_contact_step") != discovery["first_contact_step"]
            or placement.get("candidate_z_mm") != candidate["actual_holder_pose"]["z"]
            or placement.get("source_anchor", {}).get("image_sha256") != discovery["restored_reference"]["sha256"]
            or placement["source_anchor"].get("focus_binding_sha256") != digest(discovery["restored_reference"]["focus_binding"])):
        raise ValueError("Ready placement changed its original candidate or newly observed source")
    approach = original_discovery_approach(rows, run_id=run_id, discovery=discovery,
                                         head_capture_id=head_capture["id"], head_pose=head["holder_axes_xyz_mm"])
    engagement = receipt(discovery["restore_engagement"], "lab.evidence")
    restore = original_restore_reaim(rows, run_id=run_id, params=params, discovery=discovery,
        placement=placement, restored_capture_id=restored["id"], engagement=engagement, digest=digest)
    # These certificates are REQUIRED original-prefix projections, not optional
    # booleans. The producer and callers must supply/recompute them below.
    work = original_work_usage(rows, run_id=run_id)
    if (work["capture_count"] > 4096 or ready["ts"]-start["ts"] > 900 or work["z_variation_mm"] > 8):
        raise ValueError("Discovery/restore exhausted the one original finite phase budget")
    low = discovery["low_z_prefix"]
    if low != original_low_z_prefix(rows, run_id=run_id, candidate=candidate, discovery=discovery):
        raise ValueError("Discovery low-Z/AF/hold/lift proof differs from the complete original journal")
    return {"profile": DISCOVERY_PROFILE, "ready_event_id": ready["id"], "work": work, "low_z_prefix": low,
            "actual_head_domain": head_domain, "restore_reaim": restore,
            "approach": approach,
            "scope": policy["scope"]}


def validate_ready_receipt(events, *, run_id, params, discovery, placement, digest):
    """Fail closed on malformed journals through the same live/saved boundary."""
    try:
        return _validate_ready_receipt(events, run_id=run_id, params=params,
            discovery=discovery, placement=placement, digest=digest)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("Malformed or incomplete original discovery/restore journal") from exc


def original_restore_reaim(rows, *, run_id, params, discovery, placement, restored_capture_id, engagement, digest):
    """Recompute distinct source revisit and two-axis ready motion from originals."""
    from qumus_simlab.records.release_support import command_encoder_readings

    index = {row["id"]: row for row in rows}
    start, lo, hi = (discovery[k] for k in ("restore_start_event_id", "reaim_start_event_id", "reaim_end_event_id"))
    boundary = index[start]
    if (boundary.get("kind") != "step.started" or boundary.get("run_id") != run_id
            or boundary["payload"].get("name") != "restore_stamp_discovery_source"
            or discovery.get("restore_step") != "restore_stamp_discovery_source"
            or not discovery["clear_lift_end_event_id"] < start < restored_capture_id < lo < hi):
        raise ValueError("Restoration/reaim lost its distinct actual checkpoint boundaries")
    restored = discovery["restored_reference"]
    stage = restored["meta"]["stage_kinematics"]["sample_pose"]
    original = discovery["original_stage_pose"]
    if (stage != discovery["restored_stage_pose"] or discovery["restore_before"] != discovery["discovery_stage_pose"]
            or math.hypot(stage["x"]-original["x"], stage["y"]-original["y"]) > .01
            or any(stage[key] != original[key] for key in ("z", "rotation"))
            or placement["source_anchor"].get("source_id") != discovery["source_id"]
            or placement["source_anchor"].get("focus_binding_sha256") is None):
        raise ValueError("Restored source lost its original source request/readback")
    fact = engagement["payload"]
    value = fact["value"]
    handoff, observation = value["handoff_observation"], value["observation"]
    captures = handoff["captures"]
    if len(captures) != 1:
        raise ValueError("Restored engagement requires its one original overview verification capture")
    capture = captures[0]
    camera = index[capture["capture_event_id"]]
    if (fact.get("property") != "stamp_discovery_source_engagement" or fact.get("subject") != params["stamp_id"]
            or fact.get("bench") != "sim" or fact.get("source", {}).get("kind") != "synthetic"
            or fact["source"].get("event_ids") != [camera["id"]] or fact["source"].get("sha256") != digest(value)
            or value.get("stamp_id") != params["stamp_id"] or value.get("run_id") != run_id
            or handoff.get("stamp_id") != params["stamp_id"] or handoff.get("run_id") != run_id
            or camera.get("run_id") != run_id or camera.get("kind") != "device.command"
            or camera["payload"] != {"device": "overview_camera", "command": "capture", "params": {}}
            or not restored_capture_id < camera["id"] < engagement["id"] <= lo
            or capture.get("run_id") != run_id or capture.get("camera") != "overview_camera"
            or capture.get("frame_device_id") != "overview_camera" or capture.get("region") != "stamp_handoff"
            or capture.get("frame_of") != "arm_right" or capture.get("detections") != []
            or not capture["capture_started_at"] <= camera["ts"] <= capture["frame_timestamp"]
                       <= capture["capture_returned_at"] <= handoff["observed_at"] <= engagement["ts"]
            or observation.get("verdict") != "ok"
            or any(observation["measurements"].get(key) != round(placement["holder_before_reaim"][key], 4) for key in ("x", "y", "z"))):
        raise ValueError("Restored overview verification differs from its original bounded producer receipt")
    commands = [r for r in rows if start < r["id"] <= hi and r.get("kind") == "device.command"]
    if any(r.get("run_id") != run_id or r["payload"].get("device") not in
           ("transfer_arm", "transfer_stage", "microscope") and r["id"] != camera["id"] for r in commands):
        raise ValueError("Foreign command interrupted source restoration")
    stage_commands = [r for r in commands if r["payload"].get("device", "").startswith("transfer_stage")]
    expected_coarse = restored["meta"]["stage_kinematics"]["coarse_pose"]
    if (len(stage_commands) != 1 or stage_commands[0]["payload"].get("device") != "transfer_stage"
            or stage_commands[0]["payload"].get("command") != "move_to"
            or stage_commands[0]["payload"].get("params", {}).get("pose") != expected_coarse
            or stage_commands[0]["id"] >= restored["focus_binding"]["event_id"]):
        raise ValueError("Restoration did not execute its one original composed source return")
    tolerance, safe_z = .005, params["design_park_pose"]["z"]
    parked = restored["focus_binding"]["holder_pose"]
    if any(abs(parked[key]-params["design_park_pose"][key]) > tolerance for key in ("x", "y")) or parked["z"] < safe_z-tolerance:
        raise ValueError("Restored native20 source was not observed with the actual holder parked clear")
    for i, command in enumerate(commands):
        if command["payload"].get("device") != "transfer_arm":
            continue
        stop = commands[i+1]["id"] if i+1 < len(commands) else hi+1
        readings = command_encoder_readings([r for r in rows if command["id"] < r["id"] < stop], command)
        args = command["payload"].get("params", {})
        if (command["payload"].get("command") != "move_axis_to" or args.get("axis") not in ("x", "y", "z")
                or args["from_pose"]["z"] < safe_z-tolerance
                or any(_raw_holder(r)["z"] < safe_z-tolerance for r in readings)):
            raise ValueError("Source restore/reaim left its observed clear-height trajectory")
    verified = [r for r in rows if hi < r["id"] and r.get("run_id") == run_id and r.get("kind") == "step.verified"
        and r["payload"].get("name") == "restore_stamp_discovery_source"]
    if (len(verified) != 1 or verified[0]["payload"].get("verdict") != "ok"
            or verified[0]["payload"].get("measurements", {}).get("stamp_discovery_ready") != placement):
        raise ValueError("Ready placement lacks its distinct original verified restore observation")
    reaim = [r for r in commands if lo < r["id"] <= hi]
    if (len(reaim) != 2 or [r["payload"].get("device") for r in reaim] != ["transfer_arm"]*2
            or [r["payload"].get("command") for r in reaim] != ["move_axis_to"]*2
            or [r["payload"].get("params", {}).get("axis") for r in reaim] != ["x", "y"]):
        raise ValueError("Ready reaim is not exactly the two original lateral commands")
    current = placement["holder_before_reaim"]
    origin = discovery["first_candidate"]["actual_holder_pose"]
    calibration = discovery["first_candidate"]["current"]["meta"]["calibration"]
    sx, sy = calibration["mm_per_pixel"]
    if calibration.get("v_axis") == "-world_y" or sy > 0:
        sy = -abs(sy)
    shift = [current[key]-origin[key] for key in ("x", "y")]
    centre = [discovery["first_candidate"]["aim_center_px"][i]+shift[i]/scale for i, scale in enumerate((sx, sy))]
    size = calibration["image_size_px"]
    anchor = [value/2 for value in size]
    aim = [anchor[0], anchor[1]-.219/abs(sy)]
    delta = [(aim[i]-centre[i])*scale for i, scale in enumerate((sx, sy))]
    if (placement["holder_origin_shift_mm"] != shift or placement["restored_aim_center_px"] != centre
            or placement["image_anchor_px"] != anchor or placement["first_contact_target_px"] != aim
            or placement["reaim_requested_mm"] != delta or len(placement["reaim_inputs"]) != 2):
        raise ValueError("Ready reaim changed its original ring-centre transformation")
    for i, command in enumerate(reaim):
        stop = reaim[i+1]["id"] if i == 0 else hi+1
        encoders = command_encoder_readings([r for r in rows if command["id"] < r["id"] < stop], command)
        end = _raw_holder(encoders[-1])
        args, item = command["payload"]["params"], placement["reaim_inputs"][i]
        axis = ("x", "y")[i]
        target = placement["holder_before_reaim"][axis]+delta[i]
        if (args.get("from_pose") != current or args.get("value") != target
                or item != {"axis": axis, "requested_mm": target, "before": current, "after": end,
                    "actual_delta_mm": {key: end[key]-current[key] for key in ("x", "y", "z")},
                    "readback_at": encoders[-1]["ts"]}
                or any(_raw_holder(r)["z"] < safe_z-tolerance for r in encoders)
                or abs(end[axis]-target) > tolerance
                or any(abs(end[key]-current[key]) > tolerance for key in ("x", "y", "z") if key != axis)):
            raise ValueError("Ready reaim differs from its original command/precision readback")
        current = end
    if placement["reaim_readback"] != current or placement["working_xy_mm"] != [current[k] for k in ("x", "y")]:
        raise ValueError("Ready working placement differs from actual final readback")
    return {"restore_start_event_id": start, "reaim_command_ids": [r["id"] for r in reaim],
            "reaim_end_event_id": hi, "working_xy_mm": placement["working_xy_mm"],
            "overview_capture_event_id": camera["id"]}


def original_discovery_approach(rows, *, run_id, discovery, head_capture_id, head_pose):
    """The admitted nominal head stays fixed through the complete fine approach."""
    from qumus_simlab.records.release_support import command_encoder_readings

    candidate = discovery["first_candidate"]["current"]
    captures = [r for r in rows if r["id"] >= head_capture_id and r.get("kind") == "lab.frame"
        and r.get("run_id") == run_id and r["payload"].get("microscope_preview", {}).get("purpose") == "analysis"
        and r["payload"]["microscope_preview"].get("phase") == discovery["first_contact_step"]]
    final = [r for r in captures if r["payload"]["microscope_preview"].get("image_sha256") == candidate["sha256"]
             and r["ts"] == candidate["captured_at"]]
    stop = _original_exposure(rows, final, "Fine approach lacks its unique original candidate exposure")["id"]
    captures = [r for r in captures if r["id"] <= stop]
    reference = discovery["first_contact_reference"]
    if not captures or captures[0]["id"] != head_capture_id:
        raise ValueError("Fine approach changed its admitted nominal-head exposure")
    for i, capture in enumerate(captures):
        preview = capture["payload"]["microscope_preview"]
        if (preview.get("immutable") is not True or preview.get("verification") is not False
                or preview.get("source") != "synthetic" or preview.get("timestamp") != capture["ts"]
                or any(preview.get(key) != reference["meta"].get(key) for key in
                       ("calibration", "stage_kinematics", "objective", "focus_z", "focus_plane", "fov_mm"))
                or i and not 0 <= capture["ts"]-captures[i-1]["ts"] <= .1+1e-9):
            raise ValueError("Fine approach lost its fixed original observation context/cadence")
    commands = [r for r in rows if head_capture_id < r["id"] < stop and r.get("kind") == "device.command"]
    pose, moves = head_pose, []
    for i, command in enumerate(commands):
        body = command["payload"]
        if command.get("run_id") != run_id:
            raise ValueError("Foreign command interrupted the admitted head approach")
        if body.get("device") == "microscope" and body.get("command") == "capture":
            continue
        args = body.get("params", {})
        end_id = commands[i+1]["id"] if i+1 < len(commands) else stop
        readings = command_encoder_readings([r for r in rows if command["id"] < r["id"] < end_id], command)
        actual = _raw_holder(readings[-1])
        if (body.get("device") != "transfer_arm" or body.get("command") != "move_axis_to"
                or args.get("axis") != "z" or args.get("from_pose") != pose
                or args.get("speed") != .005 or not 0 < pose["z"]-args["value"] <= .00025+1e-12
                or any(_raw_holder(r)[key] != head_pose[key] for r in readings for key in ("x", "y"))
                or abs(actual["z"]-args["value"]) > .005+1e-12
                or abs(actual["z"]-pose["z"]) > .00525+1e-12):
            raise ValueError("Original approach differs from the fixed-head finite fine-Z contract")
        pose = actual
        moves.append(command["id"])
    if not moves or pose != discovery["first_candidate"]["actual_holder_pose"]:
        raise ValueError("Fine approach did not end at its original actual candidate pose")
    if sum(r["payload"].get("device") == "microscope" and r["payload"].get("command") == "capture"
           for r in commands) != len(captures)-1:
        raise ValueError("Fine approach capture command/publication count changed")
    return {"reference_capture_id": head_capture_id, "candidate_capture_id": stop,
            "fine_command_ids": moves, "capture_count": len(captures)}


def _raw_holder(row):
    from qumus_simlab.records.release_support import _pose
    return _pose(row["payload"])


def original_work_usage(rows, *, run_id):
    captures = 0
    variation = 0.
    for i, row in enumerate(rows):
        if row.get("kind") != "device.command":
            continue
        body = row["payload"]
        device, command = body.get("device"), body.get("command")
        if device in ("transfer_stage", "transfer_arm", "microscope", "stamp_heater") and row.get("run_id") != run_id:
            raise ValueError("Another run changed discovery's bound apparatus")
        captures += int(device == "microscope" and command == "capture")
        if device != "transfer_arm" or command != "move_axis_to":
            continue
        params = body["params"]
        if params.get("axis") != "z":
            continue
        later = next((j for j in range(i+1, len(rows)) if rows[j].get("kind") == "device.command"), len(rows))
        from qumus_simlab.records.release_support import command_encoder_readings

        frames = command_encoder_readings(rows[i+1:later], row)
        last = params["from_pose"]["z"]
        for frame in frames:
            current = _raw_holder(frame)["z"]
            variation += abs(current-last)
            last = current
    return {"capture_count": captures, "z_variation_mm": variation}


def _separate_capture(rows, first, later):
    """A microscope capture command lies strictly between two publications."""
    return any(first["id"] < r["id"] < later["id"] and r.get("kind") == "device.command"
               and r["payload"].get("device") == "microscope"
               and r["payload"].get("command") == "capture" for r in rows)


def _original_exposure(rows, found, message):
    """The first publication of an exposure that later captures may repeat.

    A zero-duration SIM capture can repeat an exposure bit for bit: the
    post-candidate autofocus baseline is taken at the candidate's own pose,
    focus and Clock time. The original is the first publication; each repeat
    must belong to its own later capture command.
    """
    found = sorted(found, key=lambda row: row["id"])
    if not found or not all(_separate_capture(rows, a, b) for a, b in zip(found, found[1:], strict=False)):
        raise ValueError(message)
    return found[0]


def original_low_z_prefix(rows, *, run_id, candidate, discovery, candidate_segment_only=False):
    """Whitelist the COMPLETE low-Z prefix, using original raw encoders/Clock."""
    from qumus_simlab.records.release_support import _encoder_projection, command_encoder_readings

    phase = discovery["first_contact_step"]
    rows = sorted(rows, key=lambda row: row["id"])
    def analysis(row):
        return row["payload"].get("microscope_preview", {})
    def exposure(record):
        found = [r for r in rows if r.get("kind") == "lab.frame" and r.get("run_id") == run_id
                 and analysis(r).get("purpose") == "analysis" and analysis(r).get("phase") == phase
                 and analysis(r).get("image_sha256") == record["sha256"]
                 and analysis(r).get("timestamp") == record["captured_at"]]
        return _original_exposure(rows, found, "Discovery predecessor/candidate exposure is not unique")
    def encoder(capture):
        found = [r for r in rows if r["id"] < capture["id"] and r.get("kind") == "lab.frame"
                 and r.get("run_id") == run_id and r["ts"] == capture["ts"]
                 and r["payload"].get("geometry", {}).get("microscope", {}).get("stamp_axes_xyz_mm")]
        if not found:
            raise ValueError("Discovery exposure lacks an original same-clock raw encoder")
        result = found[-1]
        _encoder_projection(result)
        return result
    def commands_between(first, last):
        return [r for r in rows if first < r["id"] < last and r.get("kind") == "device.command"]
    def publication(command, end_id, *, focus, objective, stage, calibration):
        found = [r for r in rows if command["id"] < r["id"] < end_id and r.get("kind") == "lab.frame"
                 and analysis(r).get("purpose") == "analysis"]
        if len(found) != 1:
            raise ValueError("Original low-Z capture lacks its unique analysis publication")
        row, p = found[0], analysis(found[0])
        if (row.get("run_id") != run_id or p.get("phase") != phase or p.get("immutable") is not True
                or p.get("verification") is not False or p.get("source") != "synthetic"
                or p.get("calibration") != calibration or p.get("stage_kinematics") != stage
                or p.get("objective") != objective or p.get("focus_z") != focus
                or p.get("timestamp") != row["ts"] or row["ts"] < command["ts"]
                or not isinstance(p.get("image_sha256"), str) or len(p["image_sha256"]) != 64):
            raise ValueError("Original low-Z analysis image/pose/focus/profile binding changed")
        return row

    before, current = exposure(discovery["candidate_previous"]), exposure(candidate["current"])
    preceding = [r for r in rows if r["id"] < current["id"] and r.get("kind") == "lab.frame"
                 and r.get("run_id") == run_id and analysis(r).get("purpose") == "analysis"
                 and analysis(r).get("phase") == phase]
    if not preceding or preceding[-1]["id"] != before["id"]:
        raise ValueError("Discovery borrowed a stale negative instead of the immediate previous exposure")
    before_pose, current_pose = _raw_holder(encoder(before)), _raw_holder(encoder(current))
    if current_pose != candidate.get("actual_holder_pose"):
        raise ValueError("Cached candidate pose differs from its original capture encoder")
    segment = commands_between(before["id"], current["id"])
    moves = [r for r in segment if r["payload"].get("device") == "transfer_arm"]
    if (len(moves) != 1 or any(r.get("run_id") != run_id or not
            ((r["payload"].get("device") == "transfer_arm" and r["payload"].get("command") == "move_axis_to")
             or (r["payload"].get("device") == "microscope" and r["payload"].get("command") == "capture")) for r in segment)):
        raise ValueError("Unexpected command occurred in the single fine-Z observation window")
    command = moves[0]
    params = command["payload"]["params"]
    if params.get("axis") != "z" or params.get("from_pose") != before_pose:
        raise ValueError("Fine-Z command does not start at its original observed raw pose")
    delta = before_pose["z"]-current_pose["z"]
    intended = before_pose["z"]-params["value"]
    if (not 0 < intended <= .00025+1e-12 or not 0 < delta <= .00525+1e-12
            or any(current_pose[k] != before_pose[k] for k in ("x", "y"))):
        raise ValueError("Candidate's original commanded/actual segment exceeds fixed bounds")
    next_id = next((r["id"] for r in rows if r["id"] > command["id"] and r.get("kind") == "device.command"), current["id"])
    command_encoder_readings([r for r in rows if command["id"] < r["id"] < next_id], command)
    if candidate_segment_only:
        return {"predecessor_event_id": before["id"], "candidate_event_id": current["id"],
                "actual_downward_mm": delta, "commanded_downward_mm": intended}

    hold = discovery["hold"]
    af = next((r for r in rows if r["id"] == discovery["post_candidate_focus_event_id"]), None)
    if not af or af.get("kind") != "microscope.focus" or af.get("run_id") != run_id:
        raise ValueError("Discovery has no original owned post-candidate focus event")
    af_commands = commands_between(current["id"], af["id"])
    if (not af_commands or af_commands[0]["payload"].get("device") != "microscope"
            or af_commands[0]["payload"].get("command") != "capture"
            or not 0 <= af_commands[0]["ts"]-before["ts"] <= .1+1e-9):
        raise ValueError("The original predecessor/candidate/AF-start window exceeds .1 seconds")
    focus = candidate["current"]["meta"]["focus_z"]
    stage = candidate["current"]["meta"]["stage_kinematics"]
    calibration = candidate["current"]["meta"]["calibration"]
    captures = 0
    for i, row in enumerate(af_commands):
        body = row["payload"]
        if row.get("run_id") != run_id or body.get("device") != "microscope":
            raise ValueError("Unexpected apparatus motion/energy occurred before or during autofocus")
        op, args = body.get("command"), body.get("params", {})
        if op == "set_focus":
            value = args.get("z")
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value-candidate["current"]["meta"]["focus_z"]) > .08+2*.08/9+1e-12:
                raise ValueError("Unexpected focus command exceeds the original fixed autofocus window")
            focus = value
        elif op == "set_objective":
            if args != {"magnification": 5}:
                raise ValueError("Unexpected objective command in discovery autofocus")
        elif op == "capture":
            stop = af_commands[i+1]["id"] if i+1 < len(af_commands) else af["id"]
            row_capture = publication(row, stop, focus=focus, objective=5, stage=stage, calibration=calibration)
            if _raw_holder(encoder(row_capture)) != current_pose:
                raise ValueError("Holder moved during the original autofocus")
            captures += 1
        else:
            raise ValueError("Unexpected microscope operation in discovery autofocus")
    af_elapsed = af["ts"]-af_commands[0]["ts"]
    if (captures != 21 or not 0 <= af_elapsed <= 1.538889+1e-9
            or af["payload"].get("objective") != 5 or af["payload"].get("plane") != "chip"
            or af["payload"].get("peak_observable") is not True
            or af["payload"].get("stage_kinematics") != stage or af["payload"].get("holder_pose") != current_pose
            or hold["started_at"] != af["ts"] or hold["start_event_id"] != af["id"]):
        raise ValueError("Original autofocus duration/pose/hold-start boundary changed")
    hold_rows = [r for r in rows if hold["start_event_id"] < r["id"] <= hold["end_event_id"]]
    hold_commands = [r for r in hold_rows if r.get("kind") == "device.command"]
    if not 5 <= len(hold_commands) <= 6:
        raise ValueError("Original five-second hold lacks its fixed monitored captures")
    previous = hold["started_at"]
    for i, row in enumerate(hold_commands):
        if (row.get("run_id") != run_id or row["payload"].get("device") != "microscope"
                or row["payload"].get("command") != "capture"):
            raise ValueError("Unexpected apparatus command during discovery hold")
        stop = hold_commands[i+1]["id"] if i+1 < len(hold_commands) else hold["end_event_id"]+1
        capture = publication(row, stop, focus=focus, objective=5, stage=stage, calibration=calibration)
        if (_raw_holder(encoder(capture)) != current_pose or not 0 <= capture["ts"]-previous <= 1.05+1e-9):
            raise ValueError("Original hold lost fixed pose or continuous observation cadence")
        previous = capture["ts"]
    hold_elapsed = hold["completed_at"]-hold["started_at"]
    if previous != hold["completed_at"] or not 5 <= hold_elapsed <= 5.30+1e-9:
        raise ValueError("Original hold end is not its actual last monitored exposure")
    lifting = [r for r in rows if hold["end_event_id"] < r["id"] <= discovery["clear_lift_end_event_id"]]
    lift_commands = [r for r in lifting if r.get("kind") == "device.command"]
    lifts = [r for r in lift_commands if r["payload"].get("device") == "transfer_arm"]
    if not lifts or any(r.get("run_id") != run_id or not
            ((r["payload"].get("device") == "transfer_arm" and r["payload"].get("command") == "move_axis_to"
              and r["payload"].get("params", {}).get("axis") == "z")
             or (r["payload"].get("device") == "microscope" and r["payload"].get("command") == "capture")) for r in lift_commands):
        raise ValueError("Unexpected command occurred before confirmed clear withdrawal")
    progress = None
    last_pose = current_pose
    first_progress_at = None
    for lift in lifts:
        i = next(i for i, r in enumerate(lift_commands) if r["id"] == lift["id"])
        stop = lift_commands[i+1]["id"] if i+1 < len(lift_commands) else discovery["clear_lift_end_event_id"]+1
        readings = command_encoder_readings([r for r in lifting if lift["id"] < r["id"] < stop], lift)
        args = lift["payload"]["params"]
        if args.get("from_pose") != last_pose or args["value"] <= last_pose["z"]:
            raise ValueError("Original clear lift has an unbound or downward command")
        for raw in readings:
            pose = _raw_holder(raw)
            if (pose["z"] < last_pose["z"] or any(pose[k] != current_pose[k] for k in ("x", "y"))):
                raise ValueError("Original clear lift is not a raw monotone Z-only trajectory")
            last_pose = pose
        if progress is None:
            progress = last_pose["z"]-current_pose["z"]
            first_progress_at = readings[-1]["ts"]
            if progress < .015-1e-12 or not 0 <= first_progress_at-hold["completed_at"] <= .25+1e-9:
                raise ValueError("Original first actual15um clear progress missed its .25s bound")
    if first_progress_at-before["ts"] > .1+1.538889+5.30+.25+1e-8:
        raise ValueError("Original low-Z interval exceeds its complete sequence bound")
    return {"predecessor_event_id": before["id"], "candidate_event_id": current["id"],
            "actual_downward_mm": delta, "commanded_downward_mm": intended,
            "focus_event_id": af["id"], "focus_elapsed_s": af_elapsed,
            "hold_elapsed_s": hold_elapsed, "first_lift_actual_mm": progress,
            "low_z_elapsed_s": first_progress_at-before["ts"]}


def validate_verification(verification):
    """Only the unchanged frozen observation/tolerance budget supports R017."""
    sequence, focus = verification["transfer_sequence"], verification["focus"]
    expected = {"max_unobserved_z_mm": .025, "max_unobserved_s": 2., "capture_reserve_s": .05,
                "max_observations": 600, "max_total_s": 900, "max_travel_mm": 8.,
                "descent_speed_mm_s": .005, "post_phase_focus_half_range_mm": .08,
                "first_contact_offset_mm": .219}
    if (any(sequence.get(k) != v for k, v in expected.items())
            or verification["manipulator"].get("tolerance_mm") != .005
            or focus.get("coarse_points") != 9 or focus.get("fine_points") != 9):
        raise ValueError("Discovery requires its original finite sampling/focus/readback contract")
