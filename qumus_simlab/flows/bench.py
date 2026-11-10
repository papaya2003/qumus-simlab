"""Named places on the bench (config/bench.yaml), shared by flows and the testbench.

World frame: table front-left corner, x right, y away from the operator. Each
arm commands in its own frame: arm pose = world - arm base (bench[arm]["base"]).
"""

from __future__ import annotations

import math
from typing import Any

from qumus_simlab.core.types import Pose

#: stations whose position is a single place (the others are storage boards)
POINT_STATIONS = ("exfoliation_stage", "transfer_stage")


def slots(bench: dict[str, Any], station: str | None = None,
          holds: str | None = None) -> list[str]:
    """Slot (black frame) names, optionally only on `station` / holding `holds`."""
    return [n for n, s in bench.get("slots", {}).items()
            if (station is None or s.get("station") == station)
            and (holds is None or s.get("holds") == holds)
            and s.get("autonomous", True)]


def is_slot(bench: dict[str, Any], name: str | None) -> bool:
    return bool(name) and name in bench.get("slots", {})


def slot_station(bench: dict[str, Any], name: str | None) -> str | None:
    s = bench.get("slots", {}).get(name or "")
    return s.get("station") if s else None


def positions(bench: dict[str, Any]) -> dict[str, dict[str, float]]:
    """All named places in the world frame: slots, stage drop-off points, hand-off."""
    out: dict[str, dict[str, float]] = {
        n: {"x": float(s["x"]), "y": float(s["y"])} for n, s in bench.get("slots", {}).items()}
    for name in POINT_STATIONS:
        st = bench["stations"][name]
        out[name] = {"x": float(st["x"]), "y": float(st["y"])}
    if (h := stamp_handoff(bench)) is not None:
        out["stamp_handoff"] = h
    return out


def manipulator_head(bench: dict[str, Any], pos: dict[str, float]) -> tuple[float, float]:
    """World point under the stamp holder's head (the stamp's dome) for holder
    positions {x, y}: bench.yaml transfer_arm.origin + (x, y), the holder's
    axes being parallel to the world's. A missing axis counts as the park pose."""
    ta = bench["transfer_arm"]
    o, park = ta["origin"], ta.get("park", {})
    return (float(o["x"]) + float(pos.get("x", park.get("x", 0.0))),
            float(o["y"]) + float(pos.get("y", park.get("y", 0.0))))


def over_stage(bench: dict[str, Any], pos: dict[str, float]) -> bool:
    """Is the holder's head over the finding & stacking stage, under the objective?"""
    ta, c = bench["transfer_arm"], bench["microscope"]["center"]
    hx, hy = manipulator_head(bench, pos)
    return math.hypot(hx - float(c["x"]), hy - float(c["y"])) <= float(
        ta.get("over_stage_tolerance_mm", 12.5))


def stamp_dome_offset(bench: dict[str, Any]) -> tuple[float, float]:
    """World offset of a stamp's PDMS dome from its slide's centre (the point an
    arm holds it by, and a stamp's position in the world): bench.yaml
    stamp_slide, the slide lying along x with the dome near its left end. A
    bench without stamp_slide has the dome at the centre (0, 0)."""
    s = bench.get("stamp_slide") or {}
    length = float((s.get("size") or [0.0])[0])
    return (-(length / 2 - float(s.get("dome_inset_mm", length / 2))), 0.0)


def stamp_seat(bench: dict[str, Any], pos: dict[str, float]) -> tuple[float, float]:
    """World point of the slide's centre of a stamp on the holder at holder
    positions {x, y}: its dome under the head (manipulator_head)."""
    (hx, hy), (dx, dy) = manipulator_head(bench, pos), stamp_dome_offset(bench)
    return hx - dx, hy - dy


def stamp_handoff(bench: dict[str, Any]) -> dict[str, float] | None:
    """Where arm_right puts a stamp (its slide's centre) down for the holder: so
    that its dome lands under the head at the load pose (stamp_seat)."""
    if "transfer_arm" not in bench or "microscope" not in bench:
        return None
    x, y = stamp_seat(bench, bench["transfer_arm"]["load"])
    return {"x": round(x, 3), "y": round(y, 3)}


def exfoliator_roller_x(bench: dict[str, Any], gantry_x: float) -> float:
    """World x of the upper (gantry) cylinder; the fixed one is at gantry x 0."""
    return float(bench["exfoliator"]["origin"]["x"]) + gantry_x


def exfoliator_rolls(bench: dict[str, Any], gantry_x: float,
                     gantry_z: float) -> dict[str, tuple[float, float]]:
    """Front-view (world x, height above the stage top) of the roll axes:
    {"movable": .., "fixed": ..}. The tape is the straight line between them
    (the roll radius is neglected)."""
    ex = bench["exfoliator"]
    ox = float(ex["origin"]["x"])
    return {"movable": (ox + gantry_x, float(ex.get("park_height_mm", 78.0)) + gantry_z),
            "fixed": (ox, float(ex.get("fixed_height_mm", -3.0)))}


def tape_height(rolls: dict[str, tuple[float, float]], x: float) -> float | None:
    """Height of the tape above the stage top at world x (None outside the rolls)."""
    (x0, z0), (x1, z1) = sorted((rolls["movable"], rolls["fixed"]))
    if not x0 <= x <= x1 or x1 - x0 < 1e-9:
        return None
    return z0 + (z1 - z0) * (x - x0) / (x1 - x0)


def regions(bench: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Named world regions: every slot (its frame plus a margin), every station,
    plus the configured ones (tape, microscope, workspace, ...)."""
    m = bench.get("slot_margin_mm", 3.0)
    out: dict[str, dict[str, Any]] = {}
    for name, s in bench.get("slots", {}).items():
        w, h = s.get("size", [10, 10])
        out[name] = {"x": [s["x"] - w / 2 - m, s["x"] + w / 2 + m],
                     "y": [s["y"] - h / 2 - m, s["y"] + h / 2 + m]}
    for name, st in bench.get("stations", {}).items():
        out[name] = {k: list(v) for k, v in st["region"].items()}
    if (h := stamp_handoff(bench)) is not None:
        out["stamp_handoff"] = {"x": [h["x"] - 8, h["x"] + 8], "y": [h["y"] - 8, h["y"] + 8]}
    out.update(bench.get("regions", {}))
    return out


def arm_base(bench: dict[str, Any], arm: str) -> tuple[float, float]:
    b = bench.get(arm, {}).get("base") or {"x": 0.0, "y": 0.0}
    return float(b["x"]), float(b["y"])


def to_arm(bench: dict[str, Any], arm: str, pose: Pose) -> Pose:
    """World pose -> `arm`'s own frame."""
    bx, by = arm_base(bench, arm)
    return pose.offset(dx=-bx, dy=-by)


def to_world(bench: dict[str, Any], arm: str, pose: Pose) -> Pose:
    """`arm`-frame pose -> world."""
    bx, by = arm_base(bench, arm)
    return pose.offset(dx=bx, dy=by)


def reachable(bench: dict[str, Any], arm: str) -> list[str]:
    """Named places `arm` serves: the slots on its stations, its stage points
    and its extra places (bench[arm]["reach"], ["places"])."""
    b = bench.get(arm, {})
    reach = set(b.get("reach", []))
    names = [n for n, s in bench.get("slots", {}).items()
             if s.get("station") in reach and s.get("autonomous", True)]
    names += [n for n in POINT_STATIONS if n in reach]
    names += list(b.get("places", []))
    return names


def arm_positions(bench: dict[str, Any], arm: str) -> dict[str, Pose]:
    """The places `arm` can reach, as poses in that arm's own frame."""
    pos = positions(bench)
    out = {}
    for name in reachable(bench, arm):
        p = pos[name]
        out[name] = to_arm(bench, arm, Pose(x=p["x"], y=p["y"]))
    return out


def within_reach(bench: dict[str, Any], arm: str, pose: Pose) -> bool:
    """Is an arm-frame point inside the arm's reach radius?"""
    r = bench.get(arm, {}).get("reach_mm")
    return r is None or math.hypot(pose.x, pose.y) <= r


# ----- by kind (kept for older callers) -------------------------------------

def chip_slots(bench: dict[str, Any], station: str | None = None) -> list[str]:
    return slots(bench, station, "chip")


def carrier_slots(bench: dict[str, Any]) -> list[str]:
    return slots(bench, holds="carrier")


def stamp_slots(bench: dict[str, Any]) -> list[str]:
    return slots(bench, holds="stamp")
