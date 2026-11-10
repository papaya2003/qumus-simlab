"""Opt-in source-linked SIM layout; no hardware calibration or second map state."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from qumus_simlab.config import load_yaml
from qumus_simlab.core.errors import PreconditionFailed

PROFILE = "v1-linked-first-two-sim-arms-v5"  # v2 real outer boards; v3 14 middle slots; v4 real-size chips and slides; v5 v1-measured chips
WORKSPACE_META = "simulation_bench_geometry"
_SOURCE_SLOTS = ((200, 160), (240, 160), (280, 160), (320, 160),
                 (200, 110), (240, 110), (280, 110), (320, 110),
                 (200, 60), (240, 60), (280, 60), (320, 60), (240, 20), (280, 20))
# v1 (common/chip_slot_policy.py) let the arms use only these middle slots and
# gives no reason. The owner (2026-11-03): in the SIM all 14 are usable; whether
# the real board has that limit is open (#139). Kept on each slot for reference.
_V1_ROBOT_ACCESSIBLE = {1, 2, 5, 6, 9, 10, 13}
_REACTOR_SOURCE_POINTS = {
    "load": {"x": 1200, "y": 270}, "microscope_center": {"x": 1200, "y": 430},
    "arm_base": {"x": 1080, "y": 126}, "storage_origin": {"x": 1320, "y": 140}}
_REACTOR_SOURCE_REGIONS = {
    "chamber_region": {"x": [1140, 1260], "y": [330, 530]},
    "storage_region": {"x": [1295, 1380], "y": [75, 175]},
    "table": {"x": [0, 1440], "y": [0, 630]}}
ARM3_BASE_Y = 0.  # world y of Arm 3's zero: aligned with Arms 1 and 2
_ARM3_SHIFT_Y = ARM3_BASE_Y - (_REACTOR_SOURCE_POINTS["arm_base"]["y"] - 28.5)  # -97.5 mm
_REACTOR_LOCAL_KEYS = {"storage_pitch_mm", "process_y_mm", "chamber_entry_y_mm", "chip_size_mm",
    "metal_patch_outline_mm", "carrier_slot_size_mm", "rod_length_mm", "rod_width_mm", "fov_mm_10x"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _layout_digest(bench: dict) -> str:
    return _digest({k: v for k, v in bench.items() if k != "geometry_profile"})


def _point(point: dict, dx: float, dy: float) -> dict:
    return {**point, "x": point["x"] + dx, "y": point["y"] + dy}


def _region(region: dict, dx: float, dy: float) -> dict:
    return {**region, "x": [x + dx for x in region["x"]], "y": [y + dy for y in region["y"]]}


def _move_workcell(bench: dict, station: str, target: tuple[float, float]) -> None:
    st = bench["stations"][station]
    dx, dy = target[0] - st["x"], target[1] - st["y"]
    st.update(x=target[0], y=target[1])
    for key in ("region", "module"):
        if key in st:
            st[key] = _region(st[key], dx, dy)
    if station == "exfoliation_stage":
        bench["regions"]["tape"] = _region(bench["regions"]["tape"], dx, dy)
        ex = bench["exfoliator"]
        ex["origin"] = _point(ex["origin"], dx, dy)
        ex["rail"] = {"x": [x + dx for x in ex["rail"]["x"]], "y": ex["rail"]["y"] + dy}
    else:
        bench["regions"]["microscope"] = _region(bench["regions"]["microscope"], dx, dy)
        bench["microscope"]["center"] = _point(bench["microscope"]["center"], dx, dy)
        bench["transfer_arm"]["origin"] = _point(bench["transfer_arm"]["origin"], dx, dy)


# The two outer storage boards as they really are (Jialei's sketches,
# 2026-11-02), each in the frozen-v1 nominal frame of the arm that serves it
# (world = frame + world_dx). Anchors that v1 recorded: the QR marker centres
# (stage_geometry.py 1A/1B, 2C/2D) and the camera records in material_base.db
# (Gr / hBN carriers at y 70 / 116 with centre x -242; bare chips at x -317,
# y 71 / 115 / 158; stamps grabbed at x 203-209 in rows y 35 / 72 / 111 / 187;
# the electrode chip at (321, 117)). Everything else is placed from the
# sketch at its QR-anchored scale (about 0.21 mm per pixel): an estimate.
# Slot ids keep their meaning; new frames are appended after the old ids.
# `frame` (drawing only) is the black frame on the board; `size` stays the
# object's own size (the testbench seeds objects with it). Stamp slides keep
# bench.yaml stamp_slide (40 x 25 mm, dome 7 mm from the left end) and lie
# with the dome over a notch in the board's comb-shaped left edge.
_V1_CAMERA = "v1 material_base.db camera record (YOLO + QR homography); not a calibrated slot"
_SKETCH = "Jialei's board sketch 2026-11-02 at QR-anchored scale; estimate"
_CHIP_FRAME = [32, 39]  # fits the largest chip (29.3 x 36.9); grid pitch 40 x 40-50
_QR_SIZE = 36  # mm, the printed QR squares in the board corners (sketch)
# The middle board's v1 slot grid (S1-S14 plus the unused top pair) measured
# from its QR codes: columns 8, 48, 88, 128 mm in from a QR column, rows 5, 45,
# 95, 145, 185 mm up from the lower QR. The owner places the outer boards'
# frames on the same grid relative to each board's own QR codes.
_GRID_COLS = (8, 48, 88, 128)
_GRID_ROWS = (5, 45, 95, 145, 185)
_GRID = "middle-board v1 slot grid placed from this board's QR codes (owner, 2026-11-02)"


def _cell(qr: list, inward: int, col: int, row: int) -> tuple[float, float]:
    """Grid cell (col from the QR column, row from the bottom) in the board's arm frame."""
    return qr[0] + inward * _GRID_COLS[col], qr[1] + _GRID_ROWS[row]


# all three boards are the same physical board (owner, 2026-11-02): the middle
# board's width, extended away from each outer board's QR side
_BOARD_WIDTH = 172
_COMB_X = 344 - _BOARD_WIDTH  # the right board's comb edge, x 172
_LEFT_QR = [[-320, 18], [-320, 214]]
_RIGHT_QR = [[326, 14], [326, 209]]
# tapes lie along x over grid columns 1-3 of rows 1-3; their frame matches the stamp frame
_SLIDE_FRAME = [84, 28]
_LEFT_BOARD = {
    "world_dx": 320., "region": {"x": [-338, -338 + _BOARD_WIDTH], "y": [0, 232]}, "qr": _LEFT_QR,
    "slots": {
        "left_slide_1": (*_cell(_LEFT_QR[0], 1, 2, 1), _GRID), "left_slide_2": (*_cell(_LEFT_QR[0], 1, 2, 2), _GRID),
        "left_chip_1": (*_cell(_LEFT_QR[0], 1, 0, 3), _GRID), "left_chip_2": (*_cell(_LEFT_QR[0], 1, 0, 2), _GRID),
        "left_chip_3": (*_cell(_LEFT_QR[0], 1, 0, 1), _GRID),
        "left_chip_4": (*_cell(_LEFT_QR[0], 1, 1, 4), _GRID), "left_chip_5": (*_cell(_LEFT_QR[0], 1, 2, 4), _GRID),
        "left_chip_6": (*_cell(_LEFT_QR[0], 1, 3, 4), _GRID),
        "left_slide_3": (*_cell(_LEFT_QR[0], 1, 2, 3), _GRID),
        "left_chip_7": (*_cell(_LEFT_QR[0], 1, 1, 0), _GRID), "left_chip_8": (*_cell(_LEFT_QR[0], 1, 2, 0), _GRID),
        "left_chip_9": (*_cell(_LEFT_QR[0], 1, 3, 0), _GRID)}}
# Glass slides (tape carriers and stamps) are 75 x 25 mm lying along x (owner, 2026-11-02).
_SLIDE = [75, 25]
_DOME_INSET = 7  # the stamp's PDMS dome, mm from the slide's left end (unchanged)
_STAMP_X = _COMB_X + 14 - _DOME_INSET + _SLIDE[0] / 2  # slide centre; the dome over the notch at x 186
_RIGHT_BOARD = {
    "world_dx": 840., "region": {"x": [_COMB_X, 344], "y": [-4, 227]}, "qr": _RIGHT_QR,
    "slots": {
        "right_stamp_1": (_STAMP_X, 187, _V1_CAMERA), "right_stamp_2": (_STAMP_X, 111, _V1_CAMERA),
        "right_stamp_3": (_STAMP_X, 72, _V1_CAMERA),
        "right_chip_1": (*_cell(_RIGHT_QR[0], -1, 0, 2), _GRID), "right_chip_2": (*_cell(_RIGHT_QR[0], -1, 0, 3), _GRID),
        "right_chip_3": (*_cell(_RIGHT_QR[0], -1, 0, 1), _GRID), "right_chip_4": (*_cell(_RIGHT_QR[0], -1, 1, 4), _GRID),
        "right_chip_5": (*_cell(_RIGHT_QR[0], -1, 1, 3), _GRID), "right_chip_6": (*_cell(_RIGHT_QR[0], -1, 1, 2), _GRID),
        "right_stamp_4": (_STAMP_X, 35, _V1_CAMERA), "right_stamp_5": (_STAMP_X, 150, _SKETCH),
        "right_chip_7": (*_cell(_RIGHT_QR[0], -1, 1, 1), _GRID), "right_chip_8": (*_cell(_RIGHT_QR[0], -1, 1, 0), _GRID)},
    # the comb's gaps (drawing only): one per stamp row, open to the left edge
    "notches": {"x": [_COMB_X, _COMB_X + 29], "half_height": 9.25},
    # the stamp's black frame on the board (drawing only), same size as the tape frame
    "stamp_frame": {"x": [_COMB_X, _COMB_X + _SLIDE_FRAME[0]], "half_height": _SLIDE_FRAME[1] / 2}}
# The tape slide keeps its source size here: its real 75 x 25 mm (lying along x)
# changes the exfoliation press coverage, so it belongs with the chip/stamp size
# change (#130), not with this drawing change.
# Hand-cleaved bare and exfoliated chips at v1's measured size (owner, 2026-11-04,
# #206): v1 material_base.db chip_detection, bare_silicon_chip (left camera, YOLO
# minAreaRect, n=96 on 8 chips): median 34.9 x 27.8 mm, 5-95 % 34.5-38.9 x
# 25.6-28.8. v1 stores no orientation (width = long side by convention); the SIM
# keeps the chips portrait as before (short side along x). Each chip differs by
# up to +-1.5 x +-2.0 mm (v1 sd 1.2 / 1.7 mm).
_CHIP_SIZE = [27.8, 34.9]
_CHIP_JITTER = [1.5, 2.0]
# Electrode chips (right board): v1 electrode_chip median 26.5 x 17.3 mm (n=30, one
# chip), portrait like the blank chips. The chip itself takes the size its imported
# target declares (testbench/lab.py); this sizes the right board's chip slots.
_ELECTRODE_CHIP_SIZE = [17.3, 26.5]
_SLOT_SIZE = {"carrier": list(_SLIDE), "stamp": list(_SLIDE)}


def _storage_board(bench: dict, name: str, board: dict) -> None:
    dx = board["world_dx"]
    template = {kind: next(s for s in bench["slots"].values() if s["station"] == name and s["holds"] == kind)
                for kind in {s["holds"] for s in bench["slots"].values() if s["station"] == name}}
    for slot_id in [k for k, s in bench["slots"].items() if s["station"] == name]:
        if slot_id not in board["slots"]:
            raise ValueError(f"linked board layout has no place for {slot_id}")
    for slot_id, (x, y, source) in board["slots"].items():
        kind = slot_id.split("_")[1]
        kind = {"slide": "carrier"}.get(kind, kind)
        slot = deepcopy(bench["slots"].get(slot_id) or template[kind])
        slot.update(x=x + dx, y=y, station=name, holds=kind, geometry_source=source)
        if kind in _SLOT_SIZE:
            slot["size"] = list(_SLOT_SIZE[kind])
        if kind == "carrier":
            slot["frame"] = list(_SLIDE_FRAME)
        if kind == "chip":
            slot["frame"] = list(_CHIP_FRAME)
            if name == "left_storage":
                slot.update(size=list(_CHIP_SIZE), size_jitter_mm=list(_CHIP_JITTER))
            elif name == "right_storage":
                slot["size"] = list(_ELECTRODE_CHIP_SIZE)
        if kind == "stamp" and "stamp_frame" in board:
            f = board["stamp_frame"]
            slot["frame"] = [f["x"][1] - f["x"][0], 2 * f["half_height"]]
            slot["frame_offset"] = [(f["x"][0] + f["x"][1]) / 2 - x, 0]
            slot["frame_open_left"] = True  # it starts at the comb edge, over the gap
        bench["slots"][slot_id] = slot
    region = board["region"]
    bench["stations"][name]["region"] = {"x": [x + dx for x in region["x"]], "y": list(region["y"])}
    bench["qr_markers"][name] = [[x + dx, y] for x, y in board["qr"]]
    if "notches" in board:
        n = board["notches"]
        bench["stations"][name]["notches"] = [
            {"x": [n["x"][0] + dx, n["x"][1] + dx], "y": [y - n["half_height"], y + n["half_height"]]}
            for (_, y, _) in sorted(v for k, v in board["slots"].items() if k.startswith("right_stamp"))]


def _move_reactor(bench: dict) -> None:
    """Explicit world keys only; regenerate aliases instead of moving them twice."""
    model = bench["reactor_module"]
    geo = deepcopy(model["geometry"])
    for key in ("load", "microscope_center", "arm_base", "storage_origin"):
        geo[key] = _point(geo[key], 208., -28.5)
    # Arm 3 stands at the table's front edge like Arms 1 and 2 (owner, 2026-11-02);
    # the chamber and its storage stay put, so its synthetic reach grows to 450 mm
    geo["arm_base"]["y"] = ARM3_BASE_Y
    for key in ("chamber_region", "storage_region"):
        geo[key] = _region(geo[key], 208., -28.5)
    table = {"x": [0, 1648], "y": [0, 630]}
    geo["table"] = deepcopy(table)
    model["geometry"] = geo
    bench["table"] = deepcopy(table)
    bench["regions"]["workspace"] = deepcopy(table)
    bench["stations"]["reactor"]["region"] = deepcopy(geo["chamber_region"])
    bench["stations"]["reactor_storage"]["region"] = deepcopy(geo["storage_region"])
    bench["arm_reactor"]["base"] = deepcopy(geo["arm_base"])
    bench["arm_reactor"]["reach_mm"] = 450.
    bench["slots"]["reactor_port"].update(geo["load"])
    origin, pitch = geo["storage_origin"], geo["storage_pitch_mm"]
    for i, (col, row) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1)), 1):
        bench["slots"][f"reactor_chip_slot_{i}"].update(
            x=origin["x"] + col * pitch[0], y=origin["y"] - row * pitch[1])
    x, y = geo["storage_region"]["x"], geo["storage_region"]["y"]
    bench["qr_markers"]["reactor_storage"] = [
        [x[0] + 6, y[0] + 6], [x[1] - 6, y[0] + 6],
        [x[1] - 6, y[1] - 6], [x[0] + 6, y[1] - 6]]


def _require_reactor_source(geometry: dict) -> None:
    expected = _REACTOR_SOURCE_POINTS | _REACTOR_SOURCE_REGIONS
    if (set(geometry) != set(expected) | _REACTOR_LOCAL_KEYS
            or any(geometry.get(key) != value for key, value in expected.items())):
        raise ValueError("Unsupported reactor source geometry keys or world anchors for linked SIM layout")


def compose_bench_geometry(bench: dict, config: dict) -> dict:
    selection = config.get("bench_geometry")
    if selection is None:
        return bench
    if selection != {"profile": PROFILE} or "geometry_profile" in bench:
        raise ValueError("Unknown or already composed SIM bench geometry")
    if "reactor_module" not in bench:
        raise ValueError("Linked Hall-bar geometry requires the complete integrated SIM bench")
    # This source version is tied to the known reactor composition. Do not
    # silently project an arbitrary custom layout into these nominal frames.
    _require_reactor_source(bench["reactor_module"]["geometry"])
    right = bench["stations"]["right_storage"]["region"]
    if [(sum(right[a]) / 2) for a in ("x", "y")] != [890, 140]:
        raise ValueError("Linked geometry source layout version is unsupported")
    out = deepcopy(bench)
    _storage_board(out, "left_storage", _LEFT_BOARD)
    _storage_board(out, "right_storage", _RIGHT_BOARD)
    # the board's QR markers (v1 fiducial centres) are printed squares that fill its corners
    out["qr_size_mm"] = {name: _QR_SIZE for name in ("left_storage", "middle_storage", "right_storage")}
    q = _QR_SIZE / 2
    out["stations"]["middle_storage"]["region"] = {"x": [512 - q, 648 + q], "y": [15 - q, 210 + q]}
    out["qr_markers"]["middle_storage"] = [[512, 15], [648, 15], [648, 210], [512, 210]]
    for index, (x, y) in enumerate(_SOURCE_SLOTS, 1):
        out["slots"][f"mid_chip_{index}"] = {
            "x": x + 320, "y": y, "station": "middle_storage", "holds": "chip", "size": list(_CHIP_SIZE),
            "size_jitter_mm": list(_CHIP_JITTER), "frame": list(_CHIP_FRAME),
            "source_label": f"S{index}", "autonomous": True,
            "v1_robot_accessible": index in _V1_ROBOT_ACCESSIBLE,
            "geometry_source": "frozen-v1 paired slot coordinates; not measured uncertainty"}
    _move_workcell(out, "exfoliation_stage", (320., 256.))
    # a 75 mm tape slide lying along x: the movable roll lays the tape down 50 mm
    # left of the stage centre (was 30), so the tape spans the whole slide
    out["exfoliator"]["apply_path"] = [[40, 0], [40, -60], [-90, -60], [-90, -78]]
    # the massage covers a whole 27.8 x 34.9 mm chip as it covered a 10 x 10 one:
    # nine columns 3 mm apart (the 1.5 mm pad leaves no gap), strokes 32 mm long
    out["arm_left"].update(massage_span_mm=12.0, massage_columns=9, massage_span_y_mm=16.0)
    out["stamp_slide"] = {**out["stamp_slide"], "size": list(_SLIDE), "dome_inset_mm": _DOME_INSET}
    _move_workcell(out, "transfer_stage", (850., 300.))
    for name, x in (("arm_left", 320.), ("arm_right", 840.)):
        out[name]["base"] = {"x": x, "y": 0.}
        out[name]["reach_mm"] = 450.
    _move_reactor(out)
    out["geometry_profile"] = {
        "id": PROFILE, "source": "synthetic", "simulation_only": True, "units": "mm",
        "source_frame": "frozen-v1 left arm nominal XY", "world_translation_mm": [320, 0],
        "right_to_left_translation_mm": [520, 0], "paired_slot_residual_mm": 0.,
        "source_reference": "v1/database/seed_chip_slots.py, tools/cv/stage_geometry.py and database/material_base.db",
        "source_sha256": {"seed_chip_slots.py": "009be004e290ca68c326b7f0ded3c099d01ec6df263b21125ed3c09f5b8ec1c4",
                          "stage_geometry.py": "198f7356115fd7146d807cece74ac607e33aa709d3b52564fc4f70682a47e453",
                          "material_base.db": "2fef83de6a587eb6d71638703134372360aba6dee12326c1c501a465607175ff"},
        "recovered": ["paired shared-slot coordinates", "middle fiducial rectangle", "nominal arm zeros and stage defaults",
                      "outer-board QR centres", "stamp rows from v1 camera records"],
        "estimated": ["outer board outlines and the fifth stamp row from the owner's sketch",
                      "all 14 middle slots usable (v1 allowed 7; real limit open, #139)",
                      "outer-board chip and tape frames on the middle board's v1 grid (owner's instruction)", "450mm SIM arm reach",
                      "Arm 3 at the front edge with 450mm SIM reach (was 280mm)",
                      "synthetic reactor relative placement"],
        "uncertainty": "Physical marker setbacks, tool/nozzle offsets and calibration accuracy remain unknown; SIM only",
        "reactor_translation_mm": [208, -28.5], "arm_reach_mm": 450.,
        "arm3_base_y_mm": ARM3_BASE_Y,
        "label": "Linked SIM layout · v1 nominal coordinates and camera records; board outlines and reach provisional"}
    out["geometry_profile"]["layout_sha256"] = _layout_digest(out)
    out["geometry_profile"]["limits_sha256"] = _digest(limits_for_bench(out, load_yaml("limits.yaml")))
    return out


def geometry_binding(bench: dict) -> dict | None:
    profile = bench.get("geometry_profile")
    if profile is None:
        return None
    if profile.get("id") != PROFILE or profile.get("layout_sha256") != _layout_digest(bench):
        raise ValueError("Linked SIM layout identity changed")
    return deepcopy(profile)


def limits_for_bench(bench: dict, base: dict) -> dict:
    profile = geometry_binding(bench)
    if profile is None:
        return base
    out = deepcopy(base)
    for arm in ("arm_left", "arm_right", "arm_reactor"):
        if arm in out["arms"]:
            out["arms"][arm].update(x=[-450, 450], y=[out["arms"][arm]["y"][0], 450], reach_mm=450.)
    # Arm 3 moved toward the operator; its loading region keeps the same world place
    for lock in out.get("reactor_interlocks", []):
        if lock.get("arm") == "arm_reactor":
            if lock["loading_region"] != {"x": [105, 135], "y": [129, 159]}:
                raise ValueError("Unknown source reactor loading region for linked SIM geometry")
            lock["loading_region"] = _region(lock["loading_region"], 0, -_ARM3_SHIFT_Y)
    def rebase(items, name, old, dx, dy):
        matches = [item for item in items if item["name"] == name]
        if len(matches) != 1 or matches[0]["region"] != old:
            raise ValueError("Unknown source safety region for linked SIM geometry")
        matches[0]["region"] = _region(old, dx, dy)
    left = {"x": [-20, 20], "y": [172, 212]}
    rebase(out["arms"]["arm_left"]["floors"], "press_on_exfoliation_stage", left, 0, 64)
    rebase(out["interlocks"], "exfoliator_locked_while_arm_left_low", left, 0, 64)
    rebase(out["interlocks"], "transfer_stage_locked_while_arm_right_above",
           {"x": [-55, 5], "y": [174, 314]}, 35, 86)
    rebase(out["stamp_interlocks"], "arm_right_out_while_stamp_over_stage",
           {"x": [-65, 25], "y": [254, 334]}, 35, 86)
    geo = out["geometry"]
    for key, old, new in (("stage_origin_xy", [740, 340], [850, 300]),
                          ("manipulator_origin_xy", [700, 330], [810, 290]),
                          ("objective_xy", [740, 420], [850, 380])):
        if geo[key] != old:
            raise ValueError("Unknown source swept safety anchor for linked SIM geometry")
        geo[key] = new
    geo["calibration_id"] = "synthetic-swept-envelope-hallbar-linked-v1"
    if profile.get("limits_sha256", _digest(out)) != _digest(out):
        raise ValueError("Linked SIM safety projection changed")
    return out


def admit_workspace_geometry(store: Any, config: dict | None, *, restoring: bool) -> None:
    """Refuse mixed old/new workspaces before seeding or any physical overlay."""
    selection = (config or {}).get("bench_geometry")
    existing = store.get_meta(WORKSPACE_META)
    if selection is None and existing is None:
        return
    expected = PROFILE if selection == {"profile": PROFILE} else None
    occupied = bool(store.list_samples() or store.list_runs() or store.list_projects())
    if (expected is None or existing not in (None, expected)
            or occupied and (not restoring or existing != expected)):
        raise PreconditionFailed("Linked SIM geometry requires its own fresh workspace or exact matching saved checkpoint; "
                                "existing history/ledger have not been reset or migrated")
