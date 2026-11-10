"""Simulated physical world (spec 11.1, 11.2): ground truth for the testbench.

Coordinates are in one world frame (bench.yaml: table front-left corner, mm;
the arms convert from their own frames). The world knows where every object really is; sim devices change it, sim cameras and
the microscope report it with noise. Faults (placement error, pick failure,
...) are configured in config/testbench.yaml and can be changed at runtime via
`faults`.

Objects: chips (with the flakes on them), raw-crystal carriers, and the tape
of the exfoliator. An object lying on the motorised transfer stage moves with
it (`on == "transfer_stage"`).
"""

from __future__ import annotations

import json
import math
import random
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from qumus_simlab.core.polygons import Polygon, area_um2, intersection, transform
from qumus_simlab.core.types import Pose, angle_diff
from qumus_simlab.flows.bench import exfoliator_rolls, manipulator_head, over_stage, stamp_seat
from qumus_simlab.testbench.material_adapter import cleavage_candidates, sim_outline

TAPE_HOLDER = "exfoliator"  # held_by value of an object stuck to the tape
STAMP_HOLDER = "transfer_arm"  # held_by value of the stamp on the stacking manipulator
# Jialei 2026-11-05: a material listed here lifts only when the flakes already
# on the film cover at least this fraction of its area (Gr only via hBN).
# Applies when stamp.pickup_stack_requires_overlap is set (hallbar-linked).
# Kept in code, not the stamp config, so the mechanics signature is unchanged.
MIN_CARRIED_COVER = {"Gr": 0.8}


@dataclass
class SimFlake:
    id: str
    material: str
    layers: int | None  # None for a reaction product with unmodelled thickness
    x_mm: float  # in the chip frame, from the chip centre (unrotated)
    y_mm: float
    area_um2: float
    outline_mm: Polygon = ()  # offsets from centre in the current carrier frame
    origin: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimObject:
    id: str
    kind: str  # "chip" | "carrier"
    pose: Pose
    held_by: str | None = None
    rel_rotation: float = 0.0  # object rotation relative to the nozzle while held
    #: object centre relative to the nozzle while held, in the nozzle's frame (mm):
    #: a pick off centre stays off centre until the object is put down
    rel_xy: tuple[float, float] = (0.0, 0.0)
    size_mm: tuple[float, float] = (10.0, 10.0)
    material: str | None = None  # carrier: crystal on it; chip: what was exfoliated onto it
    flakes: list[SimFlake] = field(default_factory=list)
    on: str | None = None  # surface it rests on ("transfer_stage") - moves with it
    focus_offset: float = 0.0  # chip surface height above nominal (mm): focused at stage z = -this
    electrodes: list[dict[str, Any]] = field(default_factory=list)
    material_binding: dict[str, Any] = field(default_factory=dict)
    electrode_geometry_source: dict[str, Any] = field(default_factory=dict)
    #: PC film discs a release left on this chip (chip-local mm), which a
    #: person must dissolve (#180); old checkpoints read this with getattr
    pc_films: list[dict[str, float]] = field(default_factory=list)

    def local_to_world(self, x: float, y: float) -> tuple[float, float]:
        r = math.radians(self.pose.rotation)
        return (self.pose.x + x * math.cos(r) - y * math.sin(r),
                self.pose.y + x * math.sin(r) + y * math.cos(r))

    def world_to_local(self, x: float, y: float) -> tuple[float, float]:
        r = math.radians(self.pose.rotation)
        dx, dy = x - self.pose.x, y - self.pose.y
        return dx * math.cos(r) + dy * math.sin(r), -dx * math.sin(r) + dy * math.cos(r)

    def covers(self, x: float, y: float) -> bool:
        lx, ly = self.world_to_local(x, y)
        return abs(lx) <= self.size_mm[0] / 2 and abs(ly) <= self.size_mm[1] / 2


@dataclass
class StampState:
    """A PDMS dome with a PC film (v1 stacking stamp)."""

    touch_z: float  # manipulator z at which the dome first touches a chip (mm)
    apex_dx: float  # where the dome apex really is, relative to the nominal head point (mm)
    apex_dy: float
    melt_c: float  # stage temperature at which the PC film releases onto the chip
    film: bool = True
    #: picked-up flakes, top (first picked) first, with their offset from the apex (mm)
    stack: list[tuple[SimFlake, float, float]] = field(default_factory=list)
    max_depth: float = 0.0  # deepest press of the current contact (mm below touch_z)
    #: seen through the glass slide and the PDMS dome, everything under the stamp
    #: (the chip, the film) looks this much higher (mm): the focus shifts
    optic_shift: float = 0.0
    #: film reports a person made for this stamp that the bench has acted out
    films_reported: int = 0
    drift_xy_mm: tuple[float, float] = (0.0, 0.0)
    slip_rotation_deg: float = 0.0
    front_radius_mm: float = 0.0
    #: a molten PC film pins to the chip over the contact it had (#180): its
    #: radius and world centre; the PDMS then separates inside it on lift
    pinned_radius_mm: float = 0.0
    pinned_xy: tuple[float, float] | None = None
    # Bounded whole-outline witnesses from one actual footprint, not its union
    # with earlier/later partial contacts. None is a known inactive episode.
    contact_history: dict[str, Any] | None = None


@dataclass
class ManipulatorState:
    """The stamp holder's axes (an x/y/z translation stage)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 10.0


@dataclass
class TapeState:
    material: str | None = None  # crystal picked up from a carrier (mother tape)
    uses: int = 0  # chips exfoliated since the mother tape was prepared
    applied: bool = False
    segment: int = 0
    massages: int = 0  # low massage strokes on the current object since the tape went down
    #: deepest the nozzle pressed the tape on the current object (mm below contact)
    press_depth: float = 0.0
    peak_temp_c: float = 22.0  # hottest stage temperature while applied
    roll_left_mm: float = 3000.0  # white tape still on the supply cylinder
    gantry_x: float = 0.0  # upper cylinder on its gantry (x along the tape, z up)
    gantry_z: float = 0.0
    last_peel_speed: float | None = None
    last_peel_angle: float | None = None  # what the geometry really gave
    last_peel_requested: float | None = None
    #: the applied tape hangs over the object on the stage within press reach
    laid: bool = False
    #: cells (i, j) of the object's surface the press pad went over, and whose they are
    pressed: set[tuple[int, int]] = field(default_factory=set)
    pressed_obj: str | None = None
    band: tuple[float, float] | None = None  # world y range the tape covers


@dataclass
class Faults:
    placement_sigma_mm: float = 0.25
    placement_sigma_deg: float = 1.0
    big_error_prob: float = 0.0
    big_error_mm: float = 2.0
    big_error_deg: float = 6.0
    pick_fail_prob: float = 0.0
    grab_radius_mm: float = 3.0
    locator_sigma_mm: float = 0.05
    locator_sigma_deg: float = 0.2
    stage_sigma_mm: float = 0.002
    #: one-shot overrides consumed by the next placement, e.g. {"dx": 3, "drot": 8}
    next_placement: dict[str, float] = field(default_factory=dict)
    #: the next N picks fail (the nozzle comes up empty)
    pick_fail_next: int = 0
    #: the next carried object falls off the nozzle halfway along the next long move
    drop_next: bool = False
    #: the next N stamp lift-offs pick nothing up / release nothing (stacking)
    pickup_fail_next: int = 0
    release_fail_next: int = 0
    #: the PC film of the next stamp the manipulator grips is torn
    tear_next_stamp: bool = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Faults:
        p, k, loc = cfg.get("placement", {}), cfg.get("pick", {}), cfg.get("locator", {})
        return cls(
            placement_sigma_mm=p.get("sigma_mm", 0.25),
            placement_sigma_deg=p.get("sigma_deg", 1.0),
            big_error_prob=p.get("big_error_prob", 0.0),
            big_error_mm=p.get("big_error_mm", 2.0),
            big_error_deg=p.get("big_error_deg", 6.0),
            pick_fail_prob=k.get("fail_prob", 0.0),
            grab_radius_mm=k.get("grab_radius_mm", 3.0),
            locator_sigma_mm=loc.get("sigma_mm", 0.05),
            locator_sigma_deg=loc.get("sigma_deg", 0.2),
            stage_sigma_mm=cfg.get("stage", {}).get("sigma_mm", 0.002),
        )


@dataclass
class Surface:
    """A motorised surface objects can rest on (the transfer stage)."""

    name: str
    box: dict[str, tuple[float, float]]  # world region at stage offset (0, 0)
    offset: tuple[float, float] = (0.0, 0.0)
    height: float = 0.0  # stage z: how far the surface is raised (mm)
    rotation: float = 0.0  # degrees about the translated station-centre pivot

    def pivot(self) -> tuple[float, float]:
        return (sum(self.box["x"])/2 + self.offset[0],
                sum(self.box["y"])/2 + self.offset[1])

    def contains(self, x: float, y: float) -> bool:
        px, py = self.pivot()
        r = math.radians(self.rotation)
        dx, dy = x-px, y-py
        lx, ly = dx*math.cos(r)+dy*math.sin(r), -dx*math.sin(r)+dy*math.cos(r)
        return (abs(lx) <= (self.box["x"][1]-self.box["x"][0])/2
                and abs(ly) <= (self.box["y"][1]-self.box["y"][0])/2)


class SimWorld:
    def __init__(self, faults: Faults | None = None, seed: int = 0,
                 model: dict[str, Any] | None = None) -> None:
        self.objects: dict[str, SimObject] = {}
        # Renderer-only vertical order of flakes dropped onto a chip, keyed by
        # (chip id, flake id): higher lies above. Not physics or provenance.
        self.stack_levels: dict[tuple[str, str], int] = {}
        self._next_stack_level = 1
        self.faults = faults or Faults()
        self.rng = random.Random(seed)
        self._observation_rng = random.Random(f"material-observation:{seed}")
        self._legacy_observation_rng = False  # only pre-upgrade checkpoint restoration
        # Arm moves draw from their own stream per object and move (see _move_rng),
        # so a chip's placement, and the flakes its exfoliation press then leaves,
        # do not depend on how many other arm cycles came before it.
        self._move_counts: dict[str, int] = {}
        self.seed = seed
        self.lock = threading.RLock()
        self.tape = TapeState()
        self.surfaces: dict[str, Surface] = {}
        self.model = model or {}  # config/testbench.yaml (exfoliation, microscope, ...)
        self._flake_seq = 0
        self.stamps: dict[str, StampState] = {}
        self.manip = ManipulatorState()
        #: what really hit what, whether or not any software noticed (benchmarks
        #: count these; the safety gate's refusals are a different thing)
        self.collisions: list[dict[str, Any]] = []
        self.stamp_geometry: dict[str, Any] = {}  # set by the lab from bench.yaml
        self.reactor_state: dict[str, Any] = {}  # optional synthetic reactor physical state
        self.stage_temp: Any = lambda: 22.0  # transfer-stage temperature (stamp heater)
        #: storage slots (bench.yaml), to name the slot an object sits in
        self.slots: dict[str, dict[str, Any]] = {}
        #: stamp id -> how many new PC films a person reported on it (the ledger's
        #: film_reports, set by the lab side). The testbench has no person in it:
        #: a report stands for the act, and the simulated stamp gets its new film
        #: the next time the holder grips it (works across lab-process restarts)
        self.film_reports: Any = None
        self.sample_properties: Any = lambda sid: {}
        self.active_transfer_profile: Any = lambda sid: None
        self.transfer_dynamics: dict[str, Any] = {"updated_at": None, "holder_z": self.manip.z}
        self.clock_now: Any = None  # injected Clock read; not serialized
        self._transfer_projection_cache: Any = None  # disposable fixed-mesh computation cache

    def _transfer_epoch(self, now: float, *, approach_delta: float = 0.0) -> None:
        """Anchor the physical integrator at a control change, never at a poll."""
        self._transfer_projection_cache = None
        obj = self.held_stamp()
        if obj is None:
            self.transfer_dynamics = {"updated_at": now, "holder_z": self.manip.z, "epoch": None}
            return
        st = self.stamps[obj.id]
        cfg = self.model.get("stamp", {}).get("dynamics", {})
        gains = cfg.get("approach_drift_mm_per_mm", [0.0, 0.0])
        bound = float(cfg.get("maximum_drift_mm", 0.15))
        if self.over_stage():
            st.drift_xy_mm = tuple(max(-bound, min(bound, x+gain*approach_delta))
                                   for x, gain in zip(st.drift_xy_mm, gains, strict=True))
        ap = self.apex()
        st.contact_history = (self._contact_footprint(st.contact_history, ap[2], st.front_radius_mm,
                                                    st.stack, st.stack)
                              if st.film and ap is not None and ap[3] > 0 else None)
        self.transfer_dynamics = {"updated_at": now, "holder_z": self.manip.z, "epoch": {
            "at": now, "stamp_id": obj.id, "drift_xy_mm": st.drift_xy_mm,
            "slip_rotation_deg": st.slip_rotation_deg, "front_radius_mm": st.front_radius_mm,
            "stack": deepcopy(st.stack), "contact_history": deepcopy(st.contact_history)}}

    def _before_transfer_control(self) -> float | None:
        now = self.clock_now() if self.clock_now is not None else self.transfer_dynamics.get("updated_at")
        if now is not None:
            self.advance_transfer(now, contact_boundary=True)
        return now

    def advance_transfer(self, now: float, *, contact_boundary: bool = False) -> None:
        """Project from the last control epoch with a deterministic time mesh.

        A poll never changes the integration anchor. Zero/rapid preview and
        checkpoint reads therefore produce the same physical trajectory for
        the same control/Clock history, including contact ending at a chip edge.
        """
        with self.lock:
            state = self.transfer_dynamics
            obj = self.held_stamp()
            if obj is None:
                state.update(updated_at=now, holder_z=self.manip.z)
                return
            epoch = state.get("epoch")
            if not epoch or epoch["stamp_id"] != obj.id:
                self._transfer_epoch(now)
                epoch = self.transfer_dynamics["epoch"]
            elapsed = max(0.0, now-epoch["at"])
            st = self.stamps[obj.id]
            cfg = self.model.get("stamp", {}).get("dynamics", {})
            rates = cfg.get("lateral_drift_mm_s", [0.0, 0.0])
            if not self.over_stage():
                rates = [0.0, 0.0]
            bound = float(cfg.get("maximum_drift_mm", 0.15))
            tau = max(1e-6, float(cfg.get("contact_front_tau_s", 0.8)))
            front, contact_seconds = epoch["front_radius_mm"], 0.0
            history = deepcopy(epoch["contact_history"])
            slip_rate = float(cfg.get("contact_slip_deg_s", 0.0))

            def drift(at: float) -> tuple[float, float]:
                return tuple(max(-bound, min(bound, old+rate*at))
                             for old, rate in zip(epoch["drift_xy_mm"], rates, strict=True))

            # Fixed mesh is anchored to the control epoch, not observation
            # calls. Only its final fractional cell depends on the target time.
            step = 0.1
            whole = math.floor(elapsed/step)
            key = (id(epoch), tuple(rates), bound, tau,
                   self.manip.x, self.manip.y, self.manip.z)
            cache = self._transfer_projection_cache
            first = 0
            if cache is not None and cache["key"] == key and cache["whole"] <= whole:
                first, front, contact_seconds = cache["whole"], cache["front"], cache["contact_seconds"]
                history = deepcopy(cache["contact_history"])

            def integrate(start: float, end: float, *, witness: bool = True) -> None:
                nonlocal front, contact_seconds, history
                dt = end-start
                st.drift_xy_mm = drift((start+end)/2)
                ap = self.apex()
                if ap is not None and ap[3] > 0:
                    front += (self.contact_radius(ap[3])-front)*(1-math.exp(-dt/tau))
                    contact_seconds += dt
                else:
                    front = 0.0
                if witness:
                    # The existing fixed Clock mesh and real control endpoints
                    # supply witnesses; fractional preview polls do not.
                    st.drift_xy_mm = drift(end)
                    ap = self.apex()
                    history = (self._contact_footprint(history, ap[2], front,
                        self._slipped_stack(epoch["stack"], slip_rate*contact_seconds), epoch["stack"])
                        if st.film and ap is not None and ap[3] > 0 else None)
            for i in range(first, whole):
                integrate(i*step, (i+1)*step)
            self._transfer_projection_cache = {"key": key, "whole": whole, "front": front,
                "contact_seconds": contact_seconds, "contact_history": deepcopy(history)}
            if elapsed > whole*step:
                integrate(whole*step, elapsed, witness=contact_boundary)
            st.drift_xy_mm = drift(elapsed)
            delta = slip_rate*contact_seconds
            st.stack = self._slipped_stack(epoch["stack"], delta)
            if history is not None and history["release"] is not None:
                history["release"]["stack"] = self._slipped_stack(history["release"]["stack"], delta)
            st.contact_history = history
            st.slip_rotation_deg = epoch["slip_rotation_deg"]+delta
            ap = self.apex()
            st.front_radius_mm = front if ap is not None and ap[3] > 0 else 0.0
            self.transfer_dynamics.update(updated_at=now, holder_z=self.manip.z)

    @property
    def material_morphology(self) -> dict[str, Any] | None:
        """Checkpointed population policy; None preserves compatible old worlds."""
        return deepcopy(self.model.get("exfoliation", {}).get("morphology"))

    @material_morphology.setter
    def material_morphology(self, value: dict[str, Any] | None) -> None:
        exf = self.model.setdefault("exfoliation", {})
        if value is None:
            exf.pop("morphology", None)
        else:
            if not isinstance(value, dict):
                raise ValueError("material morphology must be a profile mapping or None")
            exf["morphology"] = deepcopy(value)

    @property
    def observation_rng(self) -> random.Random:
        # New runs isolate measurement noise regardless of morphology choice.
        # The explicit switch is for historical fixture comparisons only.
        legacy = self._legacy_observation_rng or self.model.get("legacy_shared_observation_rng", False)
        return self.rng if legacy else self._observation_rng

    def add(self, obj: SimObject) -> None:
        with self.lock:
            self.objects[obj.id] = obj

    def add_surface(self, surface: Surface) -> None:
        self.surfaces[surface.name] = surface

    def held_by(self, holder: str) -> SimObject | None:
        with self.lock:
            return next((o for o in self.objects.values() if o.held_by == holder), None)

    def nearest_free(self, pose: Pose, radius: float) -> SimObject | None:
        with self.lock:
            best = None
            for o in self.objects.values():
                if o.held_by is None and o.pose.xy_distance(pose) <= radius:
                    if best is None or o.pose.xy_distance(pose) < best.pose.xy_distance(pose):
                        best = o
            return best

    # ----- arms ------------------------------------------------------------

    def _move_rng(self, kind: str, obj_id: str) -> random.Random:
        """Random stream of the k-th `kind` move ("pick", "placement", "drop") of one object.

        Only that object's own move history decides its draw: extra place
        corrections, revisits or retries on other objects leave it unchanged.
        """
        key = f"{kind}:{obj_id}"
        k = self._move_counts.get(key, 0)
        self._move_counts[key] = k + 1
        return random.Random(f"{kind}:{self.seed}:{obj_id}:{k}")

    def pick(self, holder: str, nozzle: Pose) -> SimObject | None:
        with self.lock:
            if nozzle.z > 5.0:  # nozzle not down at the object
                return None
            obj = self.nearest_free(nozzle, self.faults.grab_radius_mm)
            if obj is None or self._move_rng("pick", obj.id).random() < self.faults.pick_fail_prob:
                return None
            if self.faults.pick_fail_next > 0:
                self.faults.pick_fail_next -= 1
                return None
            if obj.on and self.tape.applied and obj.on == "exfoliation_stage":
                return None  # covered by the tape
            obj.held_by = holder
            obj.on = None
            obj.rel_rotation = angle_diff(obj.pose.rotation, nozzle.rotation)
            r = math.radians(nozzle.rotation)
            dx, dy = obj.pose.x - nozzle.x, obj.pose.y - nozzle.y
            obj.rel_xy = (dx * math.cos(r) + dy * math.sin(r), -dx * math.sin(r) + dy * math.cos(r))
            obj.pose = Pose(x=obj.pose.x, y=obj.pose.y, z=nozzle.z, rotation=obj.pose.rotation)
            return obj

    @staticmethod
    def _under_nozzle(obj: SimObject, nozzle: Pose) -> tuple[float, float]:
        """World centre of a held object: the nozzle plus the pick offset."""
        r = math.radians(nozzle.rotation)
        ox, oy = obj.rel_xy
        return (nozzle.x + ox * math.cos(r) - oy * math.sin(r),
                nozzle.y + ox * math.sin(r) + oy * math.cos(r))

    def carry(self, holder: str, nozzle: Pose, start: Pose | None = None) -> None:
        """The nozzle of `holder` arrived at `nozzle` (from `start`, if known:
        a low stroke over the tape presses all along its way, and a path through
        an exfoliator roll is a collision)."""
        with self.lock:
            if start is not None:
                self._check_arm_path(holder, start, nozzle)
            obj = self.held_by(holder)
            if obj and self.faults.drop_next and obj.pose.xy_distance(nozzle) > 10.0:
                self.faults.drop_next = False
                self._drop(obj, Pose(x=(obj.pose.x + nozzle.x) / 2,
                                     y=(obj.pose.y + nozzle.y) / 2))
                obj = None
            if obj:
                x, y = self._under_nozzle(obj, nozzle)
                obj.pose = Pose(x=x, y=y, z=nozzle.z, rotation=nozzle.rotation + obj.rel_rotation)
            self._massage(nozzle, start)

    def _check_arm_path(self, holder: str, a: Pose, b: Pose) -> None:
        """The nozzle travels the straight line a -> b (heights above the
        stage tops). Where it passes through one of the exfoliator's rolls
        (their axes run along y; bench.yaml exfoliator) that is a collision.
        Only the nozzle is modelled, not the arm's links."""
        ex = self.stamp_geometry.get("bench", {}).get("exfoliator")
        if not ex:
            return
        rolls = exfoliator_rolls({"exfoliator": ex}, self.tape.gantry_x, self.tape.gantry_z)
        y0, half = float(ex["origin"]["y"]), float(ex.get("roller", {}).get("length_mm", 34)) / 2
        clear = (float(ex.get("roller", {}).get("diameter_mm", 12)) / 2
                 + self.model.get("arms", {}).get("nozzle_radius_mm", 3.0))
        n = max(1, int(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))))  # every mm
        for k in range(n + 1):
            x, y, z = (a.x + (b.x - a.x) * k / n, a.y + (b.y - a.y) * k / n,
                       a.z + (b.z - a.z) * k / n)
            if abs(y - y0) > half:
                continue
            for name, (rx, rz) in rolls.items():
                if math.hypot(x - rx, z - rz) < clear:
                    self.collisions.append({"what": f"exfoliator_{name}_roll", "by": holder,
                                            "at": [round(x, 1), round(y, 1), round(z, 1)]})
                    return

    def _drop(self, obj: SimObject, where: Pose) -> None:
        """The object falls off the nozzle and lands flat at `where`, a bit turned."""
        obj.held_by = None
        obj.pose = Pose(x=where.x, y=where.y, z=0.0,
                        rotation=obj.pose.rotation + self._move_rng("drop", obj.id).uniform(-20, 20))
        obj.on = next((s.name for s in self.surfaces.values()
                       if s.contains(obj.pose.x, obj.pose.y)), None)

    def release(self, holder: str, nozzle: Pose) -> SimObject | None:
        with self.lock:
            obj = self.held_by(holder)
            if obj is None:
                return None
            f, r = self.faults, self._move_rng("placement", obj.id)
            if f.next_placement:
                dx, dy = f.next_placement.get("dx", 0.0), f.next_placement.get("dy", 0.0)
                drot = f.next_placement.get("drot", 0.0)
                f.next_placement = {}
            elif r.random() < f.big_error_prob:
                dx = r.choice([-1, 1]) * f.big_error_mm * r.uniform(0.6, 1.0)
                dy = r.choice([-1, 1]) * f.big_error_mm * r.uniform(0.6, 1.0)
                drot = r.choice([-1, 1]) * f.big_error_deg * r.uniform(0.6, 1.0)
            else:
                dx, dy = r.gauss(0, f.placement_sigma_mm), r.gauss(0, f.placement_sigma_mm)
                drot = r.gauss(0, f.placement_sigma_deg)
            obj.held_by = None
            x, y = self._under_nozzle(obj, nozzle)
            obj.rel_xy = (0.0, 0.0)
            obj.pose = Pose(x=x + dx, y=y + dy, z=0.0,
                            rotation=nozzle.rotation + obj.rel_rotation + drot)
            obj.on = next((s.name for s in self.surfaces.values()
                           if s.contains(obj.pose.x, obj.pose.y)), None)
            return obj

    # ----- motorised surfaces ----------------------------------------------

    def move_surface(self, name: str, x: float, y: float, z: float | None = None,
                     rotation: float | None = None) -> None:
        """A motorised surface (the transfer stage) moved to stage coordinates
        (x, y) and height z (None = unchanged). Raising it under a stamp presses
        the dome onto the chip, like lowering the stamp would."""
        with self.lock:
            s = self.surfaces.get(name)
            if s is None:
                return
            now = self._before_transfer_control() if name == "transfer_stage" else None

            def move() -> None:
                old_px, old_py = s.pivot()
                theta = s.rotation if rotation is None else rotation
                delta = theta-s.rotation
                r = math.radians(delta)
                s.offset = (x, y)
                s.rotation = theta
                px, py = s.pivot()
                if z is not None:
                    s.height = z
                for o in self.objects.values():
                    if o.on == name and o.held_by is None:
                        dx, dy = o.pose.x-old_px, o.pose.y-old_py
                        o.pose = Pose(x=px+dx*math.cos(r)-dy*math.sin(r),
                                      y=py+dx*math.sin(r)+dy*math.cos(r), z=o.pose.z,
                                      rotation=o.pose.rotation+delta)

            if name == "transfer_stage":
                lateral = (abs(x-s.offset[0]) > 1e-9 or abs(y-s.offset[1]) > 1e-9
                           or (rotation is not None and abs(rotation-s.rotation) > 1e-9))
                self._contact_change(move, lateral, by="transfer_stage")
            else:
                move()
            if now is not None:
                self._transfer_epoch(now)

    # ----- exfoliator (tape) -----------------------------------------------

    def on_exfoliation_stage(self) -> SimObject | None:
        s = self.surfaces.get("exfoliation_stage")
        with self.lock:
            return next((o for o in self.objects.values() if o.held_by is None and s is not None
                         and s.contains(o.pose.x, o.pose.y)), None)

    def press_frac(self, obj: SimObject | None = None) -> float:
        """Fraction of the object's surface the press pad has gone over."""
        obj = obj or self.on_exfoliation_stage()
        if obj is None or self.tape.pressed_obj != obj.id:
            return 0.0
        n = self._cells(obj)
        return len(self.tape.pressed) / (n[0] * n[1])

    def _cells(self, obj: SimObject) -> tuple[int, int]:
        c = self.model.get("exfoliation", {}).get("press_cell_mm", 0.5)
        return max(1, round(obj.size_mm[0] / c)), max(1, round(obj.size_mm[1] / c))

    def tape_advance(self, length_mm: float = 0.0) -> None:
        with self.lock:
            stuck = self.held_by(TAPE_HOLDER)
            if stuck is not None:  # rolled away with the old tape segment: lost
                del self.objects[stuck.id]
            left = max(0.0, self.tape.roll_left_mm - length_mm)
            self.tape = TapeState(segment=self.tape.segment + 1, roll_left_mm=left)

    def tape_apply(self, stage_temp_c: float, laid: bool = True,
                   band: tuple[float, float] | None = None) -> None:
        """The tape mechanism finished its apply move. `laid`: the tape now
        hangs over the object on the stage within press reach (geometry,
        SimTape); if not, pressing cannot bring it onto the object."""
        with self.lock:
            self.tape.applied = True
            self.tape.laid, self.tape.band = laid, band
            self.tape.peak_temp_c = stage_temp_c
            stuck = self.held_by(TAPE_HOLDER)
            if stuck is None:
                obj = self.on_exfoliation_stage()
                if obj is None or self.tape.pressed_obj != (obj.id if obj else None):
                    # a new contact; laying the same segment back on the same chip
                    # (v1's heated recipe) meets the same pressed regions again,
                    # pressed as hard and as often as before
                    self.tape.pressed, self.tape.pressed_obj = set(), None
                    self.tape.massages, self.tape.press_depth = 0, 0.0
                return
            # the same tape segment goes back down with the object still on it
            # (v1's heated recipe lifts the chip with the tape while the stage
            # heats): the object is back on the stage, the tape still pressed
            stuck.held_by = None
            stuck.on = "exfoliation_stage"

    def tape_note_temperature(self, stage_temp_c: float) -> None:
        with self.lock:
            if self.tape.applied:
                self.tape.peak_temp_c = max(self.tape.peak_temp_c, stage_temp_c)

    def _massage(self, nozzle: Pose, start: Pose | None = None) -> None:
        """A low nozzle over the applied tape presses it onto the object below,
        along the whole stroke from `start`: those cells now stick."""
        s = self.surfaces.get("exfoliation_stage")
        exf = self.model.get("exfoliation", {})
        contact = exf.get("press_contact_z", 1.2)
        if not (self.tape.applied and s is not None and nozzle.z <= contact
                and s.contains(nozzle.x, nozzle.y)):
            return
        obj = self.on_exfoliation_stage()
        if obj is None or not self.tape.laid:
            return  # the tape does not reach down to anything
        if self.tape.pressed_obj != obj.id:
            self.tape.pressed, self.tape.pressed_obj = set(), obj.id
            self.tape.massages, self.tape.press_depth = 0, 0.0
        self.tape.massages += 1
        low = min(nozzle.z, start.z) if start is not None else nozzle.z
        self.tape.press_depth = max(self.tape.press_depth, contact - low)
        pad, cell = exf.get("press_pad_radius_mm", 1.5), exf.get("press_cell_mm", 0.5)
        a = start if start is not None and start.z <= contact else nozzle
        n = max(1, int(math.dist((a.x, a.y), (nozzle.x, nozzle.y)) / (cell / 2)))
        nx, ny = self._cells(obj)
        r, band = math.radians(obj.pose.rotation), self.tape.band
        for k in range(n + 1):
            px, py = a.x + (nozzle.x - a.x) * k / n, a.y + (nozzle.y - a.y) * k / n
            dx, dy = px - obj.pose.x, py - obj.pose.y
            lx = dx * math.cos(r) + dy * math.sin(r) + obj.size_mm[0] / 2
            ly = -dx * math.sin(r) + dy * math.cos(r) + obj.size_mm[1] / 2
            for i in range(max(0, int((lx - pad) / cell)), min(nx, int((lx + pad) / cell) + 1)):
                for j in range(max(0, int((ly - pad) / cell)),
                               min(ny, int((ly + pad) / cell) + 1)):
                    if math.hypot((i + 0.5) * cell - lx, (j + 0.5) * cell - ly) > pad:
                        continue
                    if band is not None:  # pressing beside the tape presses nothing
                        _, wy = obj.local_to_world((i + 0.5) * cell - obj.size_mm[0] / 2,
                                                   (j + 0.5) * cell - obj.size_mm[1] / 2)
                        if not band[0] <= wy <= band[1]:
                            continue
                    self.tape.pressed.add((i, j))

    def tape_peel(self, speed: float, stage_vacuum_on: bool, stage_temp_c: float,
                  angle_deg: float = 30.0, requested_deg: float | None = None) -> None:
        """Peel at `angle_deg`, the angle the geometry really gave (`requested_deg`:
        what was asked). Only a pressed tape holds on to the object."""
        with self.lock:
            t = self.tape
            if not t.applied:
                return
            t.applied = False
            t.last_peel_speed, t.last_peel_angle = speed, angle_deg
            t.last_peel_requested = angle_deg if requested_deg is None else requested_deg
            t.peak_temp_c = max(t.peak_temp_c, stage_temp_c)
            obj = self.on_exfoliation_stage()
            if obj is None or self.press_frac(obj) <= 0.0:
                return  # nothing pressed on: the tape comes away clean
            exf = self.model.get("exfoliation", {})
            if not stage_vacuum_on:
                p_lift = exf.get("lift_without_vacuum_prob", 1.0)
            else:  # the stage vacuum holds the chip or carrier down
                p_lift = 0.0
            if self.rng.random() < p_lift:
                obj.held_by = TAPE_HOLDER  # came off with the tape
                obj.on = None
                return
            if obj.kind == "carrier":
                reports = self.sample_properties(obj.id).get("material_reviews") or []
                if reports and reports[-1].get("source") == "operator_report" and reports[-1].get("status") == "confirmed":
                    # Like a new-film report, a human source-replacement report
                    # stands for that external act in the virtual bench only.
                    obj.material = reports[-1]["material"]
            if obj.kind == "carrier" and obj.material:
                t.material, t.uses = obj.material, 0
            elif obj.kind == "chip" and t.material:
                self._exfoliate_onto(obj, speed, angle_deg)
                t.uses += 1

    def _exfoliate_onto(self, chip: SimObject, speed: float, angle_deg: float = 30.0) -> None:
        exf, t = self.model.get("exfoliation", {}), self.tape
        # own random stream per (seed, chip, tape segment, use): faults, retries
        # and recovery draw from self.rng, so they do not change which flakes a
        # seed puts on a chip (seed-matched comparisons, v1compare)
        r = random.Random(f"exfoliation:{self.seed}:{chip.id}:{t.segment}:{t.uses}")
        # No recipe model (Jialei, 2026-11-03): the process parameters are carried
        # out mechanically but do not change the yield or the layer mix.
        mean = exf.get("flakes_per_chip", 20.0)
        mean *= (1 - exf.get("tape_wear", 0.0)) ** t.uses
        cell = exf.get("press_cell_mm", 0.5)
        classes = (exf.get("layers_by_material", {}).get(t.material or "")
                   or exf.get("layers", [{"layers": 1, "p": 1.0}]))
        weights = [c["p"] for c in classes]
        if "morphology" in exf:
            # The cleave is drawn over the whole chip and only the pressed cells
            # keep their flakes (same expected count as cleaving the pressed part).
            # A press footprint a cell larger or smaller, from a chip set down a
            # little differently, then changes only the flakes in that cell, not
            # the whole population of the seed's chip.
            nx, ny = self._cells(chip)
            every = [(i, j) for i in range(nx) for j in range(ny)]
            n = _poisson(r, mean)
            drawn = cleavage_candidates(
                seed=self.seed, chip_id=chip.id, size_mm=chip.size_mm, segment=t.segment, uses=t.uses,
                material=t.material or "?", count=n, cells=every, cell_mm=cell,
                classes=classes, weights=weights, config=exf, rng=r, first_index=self._flake_seq)
            candidates = [c for c in drawn if (int((c["x_mm"] + chip.size_mm[0] / 2) // cell),
                                               int((c["y_mm"] + chip.size_mm[1] / 2) // cell)) in t.pressed]
        else:  # compatible scalar draws of the older fixtures, unchanged
            # only the pressed part of the chip took the tape
            n = _poisson(r, mean * self.press_frac(chip))
            candidates = cleavage_candidates(
                seed=self.seed, chip_id=chip.id, size_mm=chip.size_mm, segment=t.segment, uses=t.uses,
                material=t.material or "?", count=n, cells=sorted(t.pressed), cell_mm=cell,
                classes=classes, weights=weights, config=exf, rng=r, first_index=self._flake_seq)
        chip.flakes.extend(SimFlake(**candidate) for candidate in candidates)
        self._flake_seq += len(candidates)
        if candidates:  # a peel that left nothing behind leaves the chip bare
            chip.material = t.material

    # ----- stacking: manipulator, stamp, contact -----------------------------

    def held_stamp(self) -> SimObject | None:
        return self.held_by(STAMP_HOLDER)

    def head_xy(self) -> tuple[float, float]:
        """World point under the holder's head (nominal dome apex position)."""
        m = self.manip
        return manipulator_head(self.stamp_geometry["bench"], {"x": m.x, "y": m.y})

    def seat_xy(self) -> tuple[float, float]:
        """World point of the slide's centre of a stamp on the holder (its dome
        under the head): where a stamp on the holder is (flows/bench.py stamp_seat)."""
        m = self.manip
        return stamp_seat(self.stamp_geometry["bench"], {"x": m.x, "y": m.y})

    def over_stage(self) -> bool:
        g = self.stamp_geometry
        return bool(g) and over_stage(g["bench"], {"x": self.manip.x, "y": self.manip.y})

    def stage_height(self) -> float:
        s = self.surfaces.get("transfer_stage")
        return s.height if s is not None else 0.0

    def surface_height(self, x: float, y: float) -> float:
        """Height the dome apex at world (x, y) meets first (see dome_surface)."""
        return self.dome_surface(x, y)[1]

    def dome_surface(self, x: float, y: float) -> tuple[SimObject | None, float]:
        """(chip, height) the stamp's dome meets first with its apex at world
        (x, y) on the transfer stage: a chip surface is at the stage height
        plus its own height (focus_offset; the fixed objective's focal plane is
        at 0); just past a chip's edge the dome's curve still meets the chip,
        a little lower (sag d^2 / 2R); the bare stage top is a chip thickness
        below. (None, stage top) when no chip is that close. The chip's tilt
        is the one the microscope sees (focus_tilt_per_mm), so a surface in
        focus and the dome meet at the same height."""
        cfg = self.model.get("stamp", {})
        r = cfg.get("dome_radius_mm", 1.5)
        tilt = self.model.get("microscope", {}).get("focus_tilt_per_mm", 0.0)
        h = self.stage_height()
        best: tuple[SimObject | None, float] | None = None
        for o in self.objects.values():
            if o.kind != "chip" or o.on != "transfer_stage" or o.held_by is not None:
                continue
            lx, ly = o.world_to_local(x, y)
            hw, hh = o.size_mm[0] / 2, o.size_mm[1] / 2
            d = math.hypot(max(0.0, abs(lx) - hw), max(0.0, abs(ly) - hh))
            px, py = max(-hw, min(hw, lx)), max(-hh, min(hh, ly))
            if "cap_support_radius_mm" in cfg:
                from qumus_simlab.core.polygons import cap_surface_point

                point = cap_surface_point((lx, ly), o.world_to_local(*self.head_xy()),
                                          (hw, hh), cfg["cap_support_radius_mm"])
                if point is None:
                    continue
                px, py = point
                d = math.hypot(lx-px, ly-py)
                eligible = True
            else:
                eligible = d < r
            top = h + o.focus_offset + tilt * (px + py) - d * d / (2 * r)
            if eligible and (best is None or top > best[1]):
                best = (o, top)
        # a chip near the apex always wins: its height offset (+-0.3 mm) is
        # mounting and thickness spread, it never sinks below the bare stage
        return best or (None, h - cfg.get("chip_thickness_mm", 0.5))

    def apex(self) -> tuple[SimObject, StampState, tuple[float, float], float] | None:
        """(stamp, state, apex world xy, press depth in mm) when a stamp is over
        the stage, else None. Depth > 0 means the dome touches the chip."""
        obj = self.held_stamp()
        if obj is None or not self.over_stage():
            return None
        st = self.stamps[obj.id]
        hx, hy = self.head_xy()
        # touch_z is the holder z at which the dome meets a surface at height 0
        # (the focal plane); a raised stage or a thicker chip meets it higher
        ax, ay = hx + st.apex_dx + st.drift_xy_mm[0], hy + st.apex_dy + st.drift_xy_mm[1]
        return obj, st, (ax, ay), st.touch_z + self.surface_height(ax, ay) - self.manip.z

    def contact_radius(self, depth: float) -> float:
        """Contact disc of a dome pressed `depth` mm of holder travel past
        touch (Hertz-like r = sqrt(2 R d)). Only the fraction
        indentation_per_travel of that travel indents the dome (the rest is
        holder and PDMS block compliance), and a dome of height dome_height_mm
        never makes a disc wider than its own base."""
        cfg = self.model.get("stamp", {})
        r = cfg.get("dome_radius_mm", 1.5)
        if depth <= 0:
            return 0.0
        radius = math.sqrt(2 * r * depth * cfg.get("indentation_per_travel", 1.0))
        if "dome_height_mm" in cfg:
            radius = min(radius, math.sqrt(2 * r * cfg["dome_height_mm"]))
        return radius

    @staticmethod
    def _carried_cover(stack, apex, flake, world_xy, chip_rotation_deg) -> float:
        """Fraction of the chip flake's area under the flakes already on the film."""
        own = transform(sim_outline(flake), xy=world_xy, rotation_deg=chip_rotation_deg)
        total = area_um2(own)
        if total <= 0:
            return 0.0
        covered = sum(area_um2(intersection(own, transform(sim_outline(carried), xy=(apex[0]+ox, apex[1]+oy))))
                      for carried, ox, oy in stack)
        return min(1.0, covered/total)

    @staticmethod
    def _under_carried(stack, apex, flake, world_xy, chip_rotation_deg) -> bool:
        """Does the chip flake's outline overlap any flake already on the film?"""
        own = transform(sim_outline(flake), xy=world_xy, rotation_deg=chip_rotation_deg)
        return any(area_um2(intersection(own, transform(sim_outline(carried),
                                                        xy=(apex[0]+ox, apex[1]+oy)))) > 0
                   for carried, ox, oy in stack)

    @staticmethod
    def _slipped_stack(stack, delta):
        r = math.radians(delta)
        return [(replace(fl, outline_mm=transform(fl.outline_mm, rotation_deg=delta))
                 if fl.outline_mm and delta else deepcopy(fl),
                 x*math.cos(r)-y*math.sin(r), x*math.sin(r)+y*math.cos(r))
                for fl, x, y in stack]

    def _contact_footprint(self, history, apex, radius, stack, reference_stack):
        """One actual disc must contain every vertex at the same instant.

        Release geometry is stored in the integrator epoch's slip frame, then
        transported by the exact same transform as the carried stack. No radius
        is moved to a new centre and separate partial contacts are never united.
        """
        chip = self.dome_surface(*apex)[0]
        if chip is None:
            return None
        if history is None or history["chip_id"] != chip.id:
            history = {"version": 1, "chip_id": chip.id, "pickup": {}, "release": None}
        current = {fl.id: fl for fl in chip.flakes}
        history["pickup"] = {fid: witness for fid, witness in history["pickup"].items()
                             if current.get(fid) == witness["flake"]}
        if history["release"] is not None and history["release"]["stack"] != reference_stack:
            history["release"] = None
        if radius <= 0:
            return history
        for fl in chip.flakes:
            if fl.id not in history["pickup"] and all(math.hypot(x-apex[0], y-apex[1]) <= radius
                    for x, y in (chip.local_to_world(fl.x_mm+px, fl.y_mm+py) for px, py in sim_outline(fl))):
                history["pickup"][fl.id] = {"flake": deepcopy(fl), "apex_xy": apex, "radius_mm": radius}
        if all(math.hypot(ox+x, oy+y) <= radius for fl, ox, oy in stack for x, y in sim_outline(fl)):
            history["release"] = {"stack": deepcopy(reference_stack), "apex_xy": apex, "radius_mm": radius}
        return history

    def _pin_film(self) -> None:
        """A molten PC film pins to the chip over its current contact (#180).

        Read at control instants only, so the pinned disc does not depend on
        how often the camera polls.
        """
        ap = self.apex()
        if ap is None or ap[3] <= 0:
            return
        st = ap[1]
        if st.film and self.stage_temp() >= st.melt_c and st.front_radius_mm > st.pinned_radius_mm:
            st.pinned_radius_mm, st.pinned_xy = st.front_radius_mm, (ap[2][0], ap[2][1])

    def manip_move(self, axis: str, value: float) -> None:
        with self.lock:
            now = self._before_transfer_control()
            self._pin_film()
            old_z = self.manip.z
            def move() -> None:
                setattr(self.manip, axis, value)
                obj = self.held_stamp()
                if obj is not None:
                    sx, sy = self.seat_xy()
                    obj.pose = Pose(x=sx, y=sy, z=obj.pose.z, rotation=obj.pose.rotation)

            self._contact_change(move, lateral=axis != "z", by="transfer_arm")
            if now is not None:
                self._transfer_epoch(now, approach_delta=old_z-self.manip.z)

    def _contact_change(self, move: Any, lateral: bool, by: str) -> None:
        """Run `move` (the holder or the stage moving) and follow the dome's
        contact with the chip: deeper while it only moves vertically, released
        when it lifts off or is dragged sideways. Pressed past the crush depth,
        the dome is crushed and its film torn: a collision."""
        before = self.apex()
        move()
        after = self.apex()
        if after is not None and after[3] > self.model.get("stamp", {}).get(
                "crush_depth_mm", float("inf")):
            st = after[1]
            if st.film:  # first crush of this stamp (once torn, nothing more to break)
                self.collisions.append({"what": "stamp", "by": by,
                                        "depth_mm": round(after[3], 3),
                                        "stamp": after[0].id})
            st.film = False
            st.contact_history = None
        if before is not None and before[3] > 0:
            st = before[1]
            if after is not None and after[3] > 0 and not lateral:
                st.max_depth = max(st.max_depth, after[3])
            else:  # lifted off, or dragged away sideways: the contact ends here
                self._release_contact(st, before[2])
                st.max_depth = 0.0
                st.front_radius_mm = 0.0
                st.pinned_radius_mm, st.pinned_xy = 0.0, None
                st.contact_history = None
        elif after is not None and after[3] > 0:
            after[1].max_depth = after[3]

    def transfer_grip(self, on: bool) -> None:
        """The manipulator's vacuum: grip a stamp lying on the holder (its dome
        under the head, its slide's centre at seat_xy), or let go."""
        with self.lock:
            now = self._before_transfer_control()
            previous = self.held_stamp()
            if previous is not None and not on:
                self.stamps[previous.id].contact_history = None
            if on:
                if self.held_stamp() is None:
                    sx, sy = self.seat_xy()
                    obj = self.nearest_free(Pose(x=sx, y=sy), 2.0)
                    if obj is not None and obj.kind == "stamp":
                        obj.held_by, obj.on = STAMP_HOLDER, None
                        self._refilm(obj.id)
                        if obj.id in self.stamps:
                            self.stamps[obj.id].contact_history = None
                        if self.faults.tear_next_stamp and obj.id in self.stamps:
                            self.faults.tear_next_stamp = False
                            self.stamps[obj.id].film = False
                        obj.pose = Pose(x=sx, y=sy, z=obj.pose.z, rotation=obj.pose.rotation)
            elif (obj := self.held_stamp()) is not None:
                obj.held_by = None
            if now is not None:
                self._transfer_epoch(now)

    def _refilm(self, stamp_id: str) -> None:
        """A person reported a new PC film on this stamp since the bench last
        looked: it has one (an intact film, nothing on it)."""
        st = self.stamps.get(stamp_id)
        n = int(self.film_reports(stamp_id)) if self.film_reports and st is not None else 0
        if st is not None and n > st.films_reported:
            st.film, st.stack, st.films_reported = True, [], n
            st.max_depth, st.front_radius_mm, st.slip_rotation_deg = 0.0, 0.0, 0.0
            st.contact_history = None

    def _release_contact(self, st: StampState, apex: tuple[float, float]) -> None:
        """The dome leaves the chip: hot enough -> the film and the stack stay
        on the chip (drop-down); else flakes under the contact come along (pick-up)."""
        history = st.contact_history
        chip = self.objects.get(history["chip_id"]) if history is not None else None
        if (chip is None or chip.kind != "chip" or chip.on != "transfer_stage"
                or chip.held_by is not None or not st.film):
            return
        # A new zero-front episode is not a release attempt. Genuine earlier
        # whole-contact witnesses may still qualify after front recession.
        if st.front_radius_mm <= 0 and not history["pickup"] and history["release"] is None:
            return
        cfg, temp, f = self.model.get("stamp", {}), self.stage_temp(), self.faults
        held = self.held_stamp()
        props = self.sample_properties(held.id) if held else {}
        film = (props.get("support") or {}).get("film_type") or props.get("film")
        # Known mica uses an explicit qualitative release law; it never melts.
        # PC on a reported metal carrier uses the same configurable simulation
        # contract. Ordinary PC device stacking retains its historical model.
        from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation

        configured = self.active_transfer_profile(held.id) if held else None
        profile = MaterialMetalSimulation.model_validate(configured or {})
        assumption = getattr(profile, film, None) if film in ("PC", "mica") else None
        metal_target = bool(self.sample_properties(chip.id).get("metal_coating"))
        custom_release = assumption is not None and configured is not None and (metal_target or film == "mica")
        threshold = (assumption.release_c - assumption.release_band_c if custom_release else
                     float("inf") if configured is not None and film == "mica" else st.melt_c)
        if temp >= threshold and f.release_fail_next > 0:
            f.release_fail_next -= 1
            return
        if temp >= threshold:
            if history["release"] is None or history["release"]["stack"] != st.stack:
                return  # an untraversed front cannot release a whole stack
            if custom_release and self.rng.random() >= assumption.release_probability:
                return
            r = math.radians(chip.pose.rotation)
            # The carried stack is listed top first; the last flake touches the chip.
            for depth, (fl, _, _) in enumerate(st.stack):
                self.stack_levels[(chip.id, fl.id)] = self._next_stack_level + len(st.stack) - 1 - depth
            self._next_stack_level += len(st.stack)
            for fl, ox, oy in st.stack:
                dx, dy = apex[0] + ox - chip.pose.x, apex[1] + oy - chip.pose.y
                chip.flakes.append(replace(
                    fl, outline_mm=transform(sim_outline(fl), rotation_deg=-chip.pose.rotation),
                    origin=getattr(fl, "origin", {}),
                    x_mm=dx * math.cos(r) + dy * math.sin(r),
                    y_mm=-dx * math.sin(r) + dy * math.cos(r)))
            if not (custom_release and assumption.support_retained_on_stamp):
                # The PDMS has separated from the pinned film; the film tears
                # at the pinned edge and a disc of it stays on the chip (#180).
                px, py = st.pinned_xy or apex
                lx, ly = chip.world_to_local(px, py)
                chip.pc_films = [*getattr(chip, "pc_films", []), {"x_mm": lx, "y_mm": ly,
                    "radius_mm": st.pinned_radius_mm or history["release"]["radius_mm"]}]
            st.stack, st.film = [], bool(custom_release and assumption.support_retained_on_stamp)
            return
        under = [(fl, *chip.local_to_world(fl.x_mm, fl.y_mm)) for fl in chip.flakes]
        under = [(fl, wx, wy) for fl, wx, wy in under
                 if fl.id in history["pickup"] and history["pickup"][fl.id]["flake"] == fl]
        if under and f.pickup_fail_next > 0:  # only a pick-up that would take a flake fails
            f.pickup_fail_next -= 1
            return
        for fl, wx, wy in under:
            # pickup_stack_requires_overlap: only a flake under an already
            # carried flake is a vdW pick-up; one the bare PC film touches
            # beside the stack uses the bare-film chance (Jialei 2026-11-03).
            onto = bool(st.stack) and (not cfg.get("pickup_stack_requires_overlap")
                                       or self._under_carried(st.stack, apex, fl, (wx, wy), chip.pose.rotation))
            p = (cfg.get("pickup_onto_stack", 0.95) if onto
                 else cfg.get("pickup_first", {}).get(fl.material, cfg.get("pickup_first_default", 0.7)))
            if configured is not None and assumption is not None:
                p = assumption.pickup_probability
            cover_min = MIN_CARRIED_COVER.get(fl.material) if cfg.get("pickup_stack_requires_overlap") else None
            if cover_min is not None:
                # Jialei 2026-11-05: graphene lifts only through the carried
                # hBN. Covered >= 80 % of its area: the whole flake comes up;
                # less (or bare PC film): it stays on the chip.
                p = 1.0 if self._carried_cover(st.stack, apex, fl, (wx, wy), chip.pose.rotation) >= cover_min else 0.0
            if temp < cfg.get("pickup_min_temp_c", 60.0):
                p *= cfg.get("cold_pickup_factor", 0.3)
            if self.rng.random() < p:
                chip.flakes.remove(fl)
                carried = replace(fl, outline_mm=transform(sim_outline(fl),
                                                           rotation_deg=chip.pose.rotation),
                                  origin=getattr(fl, "origin", {}))
                st.stack.append((carried, wx - apex[0], wy - apex[1]))

    def stamp_snapshot(self) -> dict[str, Any]:
        with self.lock:
            obj = self.held_stamp()
            st = self.stamps.get(obj.id) if obj else None
            return {"manipulator": dict(self.manip.__dict__), "held": obj.id if obj else None,
                    "stack": [f.id for f, _, _ in st.stack] if st else [],
                    "film": st.film if st else None}

    # ----- snapshots -------------------------------------------------------

    def evaluation_snapshot(self) -> dict[str, Any]:
        """Pure private truth projection for scoring, never sensing or control.

        It copies the current integrated state without reading/advancing Clock,
        refreshing dynamics, invoking perception or drawing either RNG stream.
        The recorder supplies its own time/event/source binding separately.
        Containers match the existing JSON wire form; default JSON numeric
        behavior is retained, without a new finite-value admission policy.
        """
        with self.lock:
            return json.loads(json.dumps({"schema_version": 1, "mechanics_version": 2,
                "source": "synthetic_ground_truth", "verification": False,
                "objects": [{"id": o.id, "kind": o.kind, "pose": o.pose.model_dump(mode="json"),
                    "size_mm": list(o.size_mm), "held_by": o.held_by, "on": o.on,
                    "material": o.material, "flakes": [asdict(f) for f in o.flakes],
                    **({"electrode_geometry": {**o.electrode_geometry_source,
                        "polygons": deepcopy(o.electrodes)}} if getattr(o, "electrode_geometry_source", None)
                        and not getattr(o, "electrode_reference_only", False) else {})}
                    for o in self.objects.values()],
                "stamps": {sid: {"film": st.film, "max_depth_mm": st.max_depth,
                    "front_radius_mm": st.front_radius_mm, "drift_xy_mm": list(st.drift_xy_mm),
                    "slip_rotation_deg": st.slip_rotation_deg,
                    "stack": [{"flake": asdict(f), "relative_xy_mm": [x, y], "frame": "stamp_film"}
                              for f, x, y in st.stack]} for sid, st in self.stamps.items()},
                "manipulator": asdict(self.manip),
                "surfaces": {name: {"offset_xy_mm": list(s.offset), "height_mm": s.height,
                    "rotation_deg": s.rotation, "pivot_world_mm": list(s.pivot())}
                    for name, s in self.surfaces.items()},
                "collisions": self.collisions,
                "transfer_dynamics": {"updated_at": self.transfer_dynamics.get("updated_at"),
                    "holder_z": self.transfer_dynamics.get("holder_z"),
                    "epoch_started_at": (self.transfer_dynamics.get("epoch") or {}).get("at"),
                    "model": self.model.get("stamp", {}).get("dynamics", {})}}))

    def slot_of(self, o: SimObject) -> str | None:
        """The storage slot (black frame) an object lies in, if any."""
        if o.held_by is not None or o.on is not None:
            return None
        for name, s in self.slots.items():
            w, h = s.get("size", [10, 10])
            if abs(o.pose.x - s["x"]) <= w / 2 and abs(o.pose.y - s["y"]) <= h / 2:
                return name
        return None

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return [{"id": o.id, "kind": o.kind, "held_by": o.held_by, "on": o.on,
                     "slot": self.slot_of(o), "material": o.material,
                     "n_flakes": len(o.flakes), "size_mm": list(o.size_mm),
                     "electrodes": bool(getattr(o, "electrodes", [])),
                     **o.pose.model_dump()}
                    for o in self.objects.values()]

    def tape_snapshot(self) -> dict[str, Any]:
        with self.lock:
            t = self.tape
            return {"material": t.material, "uses": t.uses, "applied": t.applied,
                    "segment": t.segment, "roll_left_mm": round(t.roll_left_mm, 1),
                    "gantry": {"x": t.gantry_x, "z": t.gantry_z}, "laid": t.laid,
                    "pressed_frac": round(self.press_frac(), 3),
                    "last_peel": {"speed": t.last_peel_speed, "angle_deg": t.last_peel_angle,
                                  "requested_deg": t.last_peel_requested}}


def _poisson(r: random.Random, mean: float) -> int:
    if mean <= 0:
        return 0
    limit, k, p = math.exp(-mean), 0, 1.0
    while True:
        p *= r.random()
        if p <= limit:
            return k
        k += 1
