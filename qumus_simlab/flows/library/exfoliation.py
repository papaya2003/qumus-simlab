"""Exfoliation (graphene path, part 1), ported from v1
processing_agent/workflows/exfoliation/ (AssemblyExfoliation,
MoleculeClearStage, MoleculePrepareMotherTape, MoleculeExfoliateChip,
MoleculeResetSystem) as closed-loop steps.

    clear_exfoliation_stage
    [mother tape: advance_tape, place_carrier, load_mother_tape, return_carrier]
    per chip: place_chip_on_exfoliation_stage, press_tape,
              [heat_applied_tape if target > 22 C], cool_and_peel,
              store_exfoliated_chip
    reset_exfoliation

What changed from v1:
- every arm move is checked by the overview camera (MoveObject), each check
  over several frames;
- after peeling, the camera checks the chip is still on the stage and has
  material on it (v1 scan_materials_tool, but v1 never called it here);
- heating has a timeout and a stuck-sensor check, and ramps within limits;
- the material label comes from the ledger (the slide's slot on the left
  board), never from AI;
- the peel angle is a parameter (v1 peeled along one fixed arc);
- exfoliated chips go to the middle board, where the right arm picks them up;
- v1 switched the stage vacuum off before the pre-heat peel. With the vacuum
  off the chip can come off with the tape, so v2 keeps it on (lab to confirm,
  spec 15.3).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.core.materials import normalize_material_code
from qumus_simlab.core.types import Pose
from qumus_simlab.flows.bench import arm_positions, reachable, slot_station, slots
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.flow import Flow, register_flow
from qumus_simlab.flows.library.primitives import (
    free_slots,
    lift,
    look,
    massage,
    move_arm,
    pick_and_place,
    ramp_heater,
    sleep_watched,
    wait_temperature,
)
from qumus_simlab.flows.library.transfer import RECHECK_VIEWS, VIEWS, MoveObject
from qumus_simlab.flows.step import (
    NOT_VERIFIABLE,
    Correction,
    NotVerifiable,
    Observation,
    ProbeResult,
    Step,
    Verdict,
)

ARM = "arm_left"
CAMERA = "overview_camera"
TAPE = "exfoliator"
HEATER = "exfoliation_heater"
FAN = "exfoliation_fan"
VACUUM = "exfoliation_vacuum"
STAGE = "exfoliation_stage"
#: bare chips and material slides come from here; exfoliated chips go to STORE
SOURCE_BOARD = "left_storage"
STORE_BOARD = "middle_storage"
TAPE_SAMPLE = "tape"
AMBIENT_SETPOINT = 22.0  # v1: target_T 22 means "no heating"


def _tape(ctx: FlowContext) -> dict[str, Any]:
    return ctx.store.get_sample(TAPE_SAMPLE) or {"props": {}}


def _heating(ctx: FlowContext) -> dict[str, Any]:
    return ctx.verification["heating"]


def _look(ctx: FlowContext, region: str, label: str | None = None,
          images: list[str] | None = None, expect: bool = False) -> list:
    """`expect`: the object should be there; an empty look is checked again."""
    return look(ctx, CAMERA, region, ARM, label=label, images=images, views=VIEWS,
                recheck_empty=RECHECK_VIEWS if expect else 0)


class ClearExfoliationStage(Step):
    """Nothing may be on the exfoliation stage before a run (v1 MoleculeClearStage,
    which trusted the database; here the camera decides)."""

    name = "clear_exfoliation_stage"
    devices = (ARM, CAMERA, TAPE)

    def preconditions(self, ctx: FlowContext) -> None:
        self.max_corrections = ctx.verification["clear_exfoliation_stage"]["max_corrections"]
        if ctx.gate.tape(TAPE).is_applied():
            raise PreconditionFailed("tape is applied over the exfoliation stage")
        if ctx.gate.arm(ARM).vacuum_on():
            raise PreconditionFailed(f"{ARM} vacuum is on (holding something?)")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        places = arm_positions(ctx.bench, ARM)
        reach = set(reachable(ctx.bench, ARM))
        # Objects carry no id: the ledger says what was left on the stage.
        left_here = [s for s in ctx.store.list_samples() if s["location"] == STAGE]
        for det in _look(ctx, STAGE):
            sample = next((s for s in left_here if s["kind"] == det.label), None)
            if sample is not None:
                left_here.remove(sample)
            props = (sample or {}).get("props", {})
            if det.label == "carrier":
                home = props.get("home_slot")
                dests = [home] if home else free_slots(ctx, slots(ctx.bench, SOURCE_BOARD,
                                                                  "carrier"))
            else:
                exfoliated = (sample or {}).get("status") == "exfoliated" or \
                    bool(det.props.get("material_visible"))
                board = STORE_BOARD if exfoliated else SOURCE_BOARD
                dests = free_slots(ctx, slots(ctx.bench, board, "chip"))
            dests = [d for d in dests if d in reach and not _look(ctx, d)]
            if not dests:
                raise PreconditionFailed(f"no free place to put the {det.label} from the stage")
            pick_and_place(ctx, ARM, det.pose, places[dests[0]])
            if sample is not None:
                ctx.store.upsert_sample(sample["id"], sample["kind"], location=dests[0])

    def verify(self, ctx: FlowContext) -> Observation:
        left = _look(ctx, STAGE)
        m = {"objects_on_stage": [d.label for d in left]}
        if not left:
            return Observation(Verdict.OK, m, {"objects_on_stage": []})
        return Observation(Verdict.CORRECTABLE, m, {"objects_on_stage": []},
                           message=f"{len(left)} object(s) still on the stage")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return Correction("clear again")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        if ctx.gate.arm(ARM).vacuum_on():
            return ProbeResult.UNKNOWN
        return ProbeResult.NOT_DONE  # re-running is harmless

    def safe_state(self, ctx: FlowContext) -> None:
        lift(ctx, ARM)


class AdvanceTape(Step):
    """Roll fresh tape over the stage (v1 AtomTapeRelease), checked by the camera."""

    name = "advance_tape"
    devices = (TAPE, CAMERA, ARM)

    def preconditions(self, ctx: FlowContext) -> None:
        self.max_corrections = ctx.verification["tape"]["max_corrections"]
        if ctx.gate.tape(TAPE).is_applied():
            raise PreconditionFailed("tape is applied; peel it first")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        lift(ctx, ARM)
        # an empty roll raises NeedsHumanError: someone has to load a new one
        ctx.gate.tape(TAPE).advance(ctx.verification["tape"]["advance_mm"])

    def verify(self, ctx: FlowContext) -> Observation:
        on_tape = _look(ctx, "tape")
        m = {"on_tape": [d.label for d in on_tape],
             "roll_left_mm": ctx.gate.tape(TAPE).remaining_mm()}
        if not on_tape:
            return Observation(Verdict.OK, m, {"on_tape": []})
        return Observation(Verdict.CORRECTABLE, m, {"on_tape": []},
                           message="old tape still visible over the stage")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return Correction("advance again")

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        ctx.store.upsert_sample(TAPE_SAMPLE, "tape", location=TAPE, status="fresh",
                                props={"material": None, "uses": 0, "from_carrier": None,
                                       "roll_left_mm": ctx.gate.tape(TAPE).remaining_mm()})

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE  # advancing again only wastes a little tape


def press_tape(ctx: FlowContext, rounds: int, depth_mm: float | None = None) -> None:
    """Vacuum on, tape down, massage (v1 AtomVacuumOn + AtomTapeApply + AtomLeftArmMassage).
    `depth_mm`: how hard the nozzle presses (below first contact with the tape)."""
    ctx.gate.switch(VACUUM).set(True)
    lift(ctx, ARM)
    tape = ctx.gate.tape(TAPE)
    if not tape.is_applied():  # resumed after a kill mid-massage: it is down already
        tape.apply()
    massage(ctx, ARM, arm_positions(ctx.bench, ARM)[STAGE], rounds, depth_mm)


def _lifted_note(ctx: FlowContext, sample_id: str, what: str) -> str:
    """Stop note when the peel took the object with the tape: names the object
    (the records cannot place it any more) and says what the stage vacuum reads,
    the first thing a human checks."""
    reads = "on" if ctx.gate.switch(VACUUM).get() else "off"
    return (f"{sample_id} ({what}) came off with the tape: it hangs on the tape over the "
            f"exfoliation stage, not on the stage. The stage vacuum reads {reads}"
            + (" but did not hold it: a peel faster or steeper than the chip can take, or "
               "the vacuum switch or seal" if reads == "on" else "")
            + ". Take it off the tape by hand")


def _record_lifted(ctx: FlowContext, sample_id: str, kind: str, obs: Observation) -> None:
    """The ledger stops claiming the object is on the stage: location uncertain,
    last seen on the tape (Step.on_give_up; the same bookkeeping as MoveObject)."""
    if obs.message and "came off with the tape" in obs.message:
        ctx.store.upsert_sample(sample_id, kind,
                                props={"location_uncertain": True,
                                       "last_seen": {"at": "tape", "over": STAGE,
                                                     "note": "lifted by the peel"}})


class LoadMotherTape(Step):
    """Press fresh tape onto the raw-crystal carrier and peel: the tape now
    carries crystal (v1 MoleculePrepareMotherTape steps 4-5)."""

    name = "load_mother_tape"
    devices = (ARM, CAMERA, TAPE, VACUUM)

    def __init__(self, carrier_id: str, num_massages: int, peel_speed: float,
                 peel_angle_deg: float | None = None, press_depth_mm: float | None = None
                 ) -> None:
        self.carrier_id, self.rounds, self.speed = carrier_id, num_massages, peel_speed
        self.angle, self.depth = peel_angle_deg, press_depth_mm

    def preconditions(self, ctx: FlowContext) -> None:
        self.max_corrections = 1
        on_stage = _look(ctx, STAGE, label="carrier", expect=True)
        if len(on_stage) != 1:
            raise PreconditionFailed(f"expected the carrier on the stage, camera sees "
                                     f"{len(on_stage)}")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        press_tape(ctx, self.rounds, self.depth)
        ctx.gate.tape(TAPE).peel(self.speed, self.angle)

    def verify(self, ctx: FlowContext) -> Observation:
        imgs: list[str] = []
        on_tape = _look(ctx, "tape", images=imgs)
        on_stage = _look(ctx, STAGE, label="carrier", images=imgs, expect=True)
        m = {"on_tape": [d.label for d in on_tape], "carrier_on_stage": len(on_stage)}
        exp = {"on_tape": ["tape_crystal"], "carrier_on_stage": 1}
        if any(d.label == "carrier" for d in on_tape):
            return Observation(Verdict.UNCLEAR, m, exp, images=imgs,
                               message=_lifted_note(ctx, self.carrier_id, "the material carrier"))
        if not on_stage:
            return Observation(Verdict.UNCLEAR, m, exp, images=imgs, message="carrier not on the stage")
        if any(d.label == "tape_crystal" for d in on_tape):
            return Observation(Verdict.OK, m, exp, images=imgs)
        return Observation(Verdict.CORRECTABLE, m, exp, images=imgs, message="no crystal on the tape")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        return Correction("press and peel again")

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        carrier = ctx.store.get_sample(self.carrier_id) or {"props": {}}
        material = normalize_material_code(carrier["props"].get("material"))
        ctx.store.upsert_sample(TAPE_SAMPLE, "tape", location=TAPE, status="loaded",
                                props={"material": material, "uses": 0,
                                       "from_carrier": self.carrier_id})

    def on_give_up(self, ctx: FlowContext, obs: Observation) -> None:
        _record_lifted(ctx, self.carrier_id, "carrier", obs)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        if ctx.gate.tape(TAPE).is_applied():
            return ProbeResult.NOT_DONE  # killed before the peel: press again and peel
        obs = self.verify(ctx)
        if obs.verdict == Verdict.OK:
            return ProbeResult.DONE
        # killed before pressing: carrier on the stage, no crystal on the tape yet
        return ProbeResult.NOT_DONE if obs.verdict == Verdict.CORRECTABLE else ProbeResult.UNKNOWN

    def safe_state(self, ctx: FlowContext) -> None:
        lift(ctx, ARM)


class PressTape(Step):
    """Tape down on the chip and massage (v1 MoleculeExfoliateChip steps 3-5)."""

    name = "press_tape"
    devices = (ARM, TAPE, VACUUM)
    # a Stop waits for the massage to end: it takes seconds, and pressing again
    # after Continue would press the chip more than a run without the Stop
    pause_inside = False

    def __init__(self, chip_id: str, num_massages: int,
                 press_depth_mm: float | None = None) -> None:
        self.chip_id, self.rounds, self.depth = chip_id, num_massages, press_depth_mm

    def preconditions(self, ctx: FlowContext) -> None:
        tape = _tape(ctx)
        if not tape["props"].get("material"):
            raise PreconditionFailed("the tape carries no crystal (prepare the mother tape)")
        if len(_look(ctx, STAGE, label="chip", expect=True)) != 1:
            raise PreconditionFailed("expected exactly one chip on the exfoliation stage")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        press_tape(ctx, self.rounds, self.depth)

    def verify(self, ctx: FlowContext) -> NotVerifiable:
        return NOT_VERIFIABLE("no sensor confirms tape contact or massage pressure "
                              "(a force sensor on the exfoliator would close this)")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        # A tape that is down is not necessarily pressed on (only the pressed area
        # sticks, and nothing can measure it): press again, which is harmless.
        return ProbeResult.NOT_DONE

    def safe_state(self, ctx: FlowContext) -> None:
        lift(ctx, ARM)


class HeatAndReapplyTape(Step):
    """Heat the already applied tape without a second peel/application.

    Retain the Python class for historical imports; new jobs use the corrected
    step name. Massage count does not control the number of applications.
    """

    name = "heat_applied_tape"
    devices = (TAPE, HEATER, FAN, VACUUM, ARM)
    safe_to_resume = False  # stopped mid-heating: a human decides (spec 8.2)

    def __init__(self, target_c: float, peel_speed: float,
                 peel_angle_deg: float | None = None) -> None:
        self.target, self.speed, self.angle = target_c, peel_speed, peel_angle_deg
        self._reached: tuple[float, float] | None = None

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        return {"target_c": self.target, "band_c": _heating(ctx)["band_c"]}

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        h = _heating(ctx)
        lift(ctx, ARM)
        tape = ctx.gate.tape(TAPE)
        if not tape.is_applied():
            raise PreconditionFailed("Heat only the tape already applied by press_tape")
        ramp_heater(ctx, HEATER, self.target, h["poll_s"])
        self._reached = wait_temperature(
            ctx, HEATER, lambda t: abs(t - self.target) <= h["band_c"],
            what=f"{self.target:.0f}C", timeout_s=h["timeout_s"],
            stuck_window_s=h["stuck_window_s"], stuck_delta_c=h["stuck_delta_c"],
            poll_s=h["poll_s"])
        ctx.gate.heater(HEATER).off()
        ctx.gate.switch(FAN).set(True)

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        if self._reached is None:
            return Observation(Verdict.UNCLEAR, {}, exp, message="heating did not run")
        temp, waited = self._reached
        m = {"reached_c": round(temp, 1), "seconds_to_reach": round(waited),
             "tape_applied": ctx.gate.tape(TAPE).is_applied()}
        if abs(temp - self.target) <= exp["band_c"] and m["tape_applied"]:
            return Observation(Verdict.OK, m, exp)
        return Observation(Verdict.UNCLEAR, m, exp, message="target not held or tape not down")

    def safe_state(self, ctx: FlowContext) -> None:
        ctx.gate.heater(HEATER).off()
        ctx.gate.switch(FAN).set(True)
        lift(ctx, ARM)


class CoolAndPeel(Step):
    """Dwell, cool below the peel temperature, peel; the camera checks the chip
    stayed on the stage and has material on it (v1 steps 6-8, unchecked there)."""

    name = "cool_and_peel"
    devices = (TAPE, HEATER, FAN, VACUUM, ARM, CAMERA)

    def __init__(self, chip_id: str, dwell_s: float, peel_speed: float,
                 peel_angle_deg: float | None = None) -> None:
        self.chip_id, self.dwell, self.speed = chip_id, dwell_s, peel_speed
        self.angle = peel_angle_deg
        self._peel_temp: float | None = None

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        return {"chip_on_stage": True, "material_visible": True,
                "peel_below_c": _heating(ctx)["peel_below_c"]}

    def preconditions(self, ctx: FlowContext) -> None:
        if not ctx.gate.tape(TAPE).is_applied():
            raise PreconditionFailed("tape is not down on the chip")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        h = _heating(ctx)
        lift(ctx, ARM)
        sleep_watched(ctx, self.dwell, h["poll_s"])
        ctx.gate.heater(HEATER).off()
        below = h["peel_below_c"]
        if ctx.gate.heater(HEATER).read_temperature() >= below:
            ctx.gate.switch(FAN).set(True)
        self._peel_temp, _ = wait_temperature(
            ctx, HEATER, lambda t: t < below, what=f"below {below:.0f}C",
            timeout_s=h["cool_timeout_s"], stuck_window_s=h["stuck_window_s"],
            stuck_delta_c=h["stuck_delta_c"], poll_s=h["poll_s"])
        ctx.gate.switch(FAN).set(False)
        ctx.gate.tape(TAPE).peel(self.speed, self.angle)

    def verify(self, ctx: FlowContext) -> Observation:
        exp = self.expected(ctx)
        imgs: list[str] = []
        on_tape = _look(ctx, "tape", label="chip", images=imgs)
        chips = _look(ctx, STAGE, label="chip", images=imgs, expect=True)
        m: dict[str, Any] = {"chip_on_stage": len(chips) == 1, "chip_on_tape": bool(on_tape),
                             "peel_temp_c": None if self._peel_temp is None
                             else round(self._peel_temp, 1)}
        if on_tape:
            return Observation(Verdict.UNCLEAR, m, exp, images=imgs,
                               message=_lifted_note(ctx, self.chip_id, "the chip"))
        if len(chips) != 1:
            return Observation(Verdict.UNCLEAR, m, exp, images=imgs, message="chip not on the stage")
        m["material_visible"] = bool(chips[0].props.get("material_visible"))
        if not m["material_visible"]:
            # "nothing there" parks the run: look again with more frames first
            again = look(ctx, CAMERA, STAGE, ARM, label="chip", images=imgs,
                         views=RECHECK_VIEWS)
            m["material_visible"] = len(again) == 1 and \
                bool(again[0].props.get("material_visible"))
        if not m["material_visible"]:
            return Observation(Verdict.UNCLEAR, m, exp, images=imgs,
                               message="nothing transferred to the chip (mother tape exhausted?)")
        return Observation(Verdict.OK, m, exp, images=imgs)

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        tape = _tape(ctx)
        material = tape["props"].get("material")
        ctx.store.upsert_sample(self.chip_id, "chip", status="exfoliated",
                                props={"material": material,
                                       "from_carrier": tape["props"].get("from_carrier")})
        ctx.store.upsert_sample(TAPE_SAMPLE, "tape", status="used",
                                props={"uses": int(tape["props"].get("uses") or 0) + 1,
                                       "roll_left_mm": ctx.gate.tape(TAPE).remaining_mm()})

    def on_give_up(self, ctx: FlowContext, obs: Observation) -> None:
        _record_lifted(ctx, self.chip_id, "chip", obs)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        if ctx.gate.tape(TAPE).is_applied():
            # Crashed while dwelling or cooling: finishing from here is safe.
            return ProbeResult.NOT_DONE
        return ProbeResult.DONE if self.verify(ctx).verdict == Verdict.OK else ProbeResult.UNKNOWN

    def safe_state(self, ctx: FlowContext) -> None:
        ctx.gate.heater(HEATER).off()
        lift(ctx, ARM)


class ResetExfoliation(Step):
    """Heater, fan and vacuum off, arm home (v1 MoleculeResetSystem)."""

    name = "reset_exfoliation"
    devices = (HEATER, FAN, VACUUM, ARM)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        ctx.gate.heater(HEATER).off()
        ctx.gate.switch(FAN).set(False)
        ctx.gate.switch(VACUUM).set(False)
        lift(ctx, ARM)
        move_arm(ctx, ARM, Pose(**ctx.bench[ARM]["home"]))

    def verify(self, ctx: FlowContext) -> Observation:
        h = ctx.gate.heater(HEATER)
        m = {"heater_setpoint": h.get_target(), "heater_c": round(h.read_temperature(), 1),
             "vacuum_on": ctx.gate.switch(VACUUM).get()}
        exp = {"heater_setpoint": None, "vacuum_on": False}
        ok = m["heater_setpoint"] is None and not m["vacuum_on"]
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR, m, exp)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


class ExfoliationParams(BaseModel):
    material: str = Field(description="Crystal to exfoliate, e.g. 'Gr' (graphite/graphene) "
                          "or 'hBN'. Must match a carrier's QR code.")
    num_chips: int = Field(1, ge=1, le=6)
    chip_ids: list[str] = Field(default_factory=list,
                                description="Bare chips to use; empty = first bare chips")
    prepare_mother_tape: Literal["auto", "always", "never"] = "auto"
    target_temp_c: Literal[22, 30, 50, 70, 90] = Field(
        22, description="Stage temperature during exfoliation (v1 approved setpoints)")
    dwell_s: float = Field(10.0, ge=10, le=90)
    num_massages: int = Field(1, ge=1, le=5)
    press_depth_mm: float = Field(0.2, ge=0.05, le=0.6,
                                  description="How hard the nozzle presses the tape: how far it "
                                  "goes below first contact (the rubber tip gives); limits.yaml "
                                  "arms.arm_left floors")
    peel_speed: float = Field(500.0, ge=50, le=500, description="Peel feedrate (G-code F)")
    peel_angle_deg: float = Field(30.0, ge=10, le=90,
                                  description="Angle between tape and stage while peeling "
                                  "(v1 had one fixed arc, about 30 deg)")
    # Filled in at submit from the ledger (resolve_params); stored with the run.
    carrier_id: str | None = None
    carrier_slot: str | None = None
    home_slots: list[str] = Field(default_factory=list)   # where the bare chips were
    store_slots: list[str] = Field(default_factory=list)  # middle-board slots they go to

    @field_validator("material")
    @classmethod
    def _norm(cls, v: str) -> str:
        code = normalize_material_code(v)
        if not code:
            raise ValueError("material is required")
        return code


#: a mother tape is re-prepared after this many chips (v1 decided from the camera)
MAX_TAPE_USES = 5


@register_flow
class ExfoliationFlow(Flow):
    name = "exfoliate_chips"
    description = ("Exfoliate crystal from a raw-crystal carrier onto bare chips with the left "
                   "arm and the tape mechanism; every move and the result checked by camera.")
    Params = ExfoliationParams
    devices = (ARM, CAMERA, TAPE, HEATER, FAN, VACUUM)

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        assert isinstance(params, ExfoliationParams)
        p = params.model_copy(deep=True)
        bench = self.bench
        from qumus_simlab.records.materials import material_review

        carriers = [s for s in store.list_samples(kind="carrier")
                    if normalize_material_code(s["props"].get("material")) == p.material
                    and material_review(s)["status"] != "uncertain"]
        if p.carrier_id:
            carriers = [c for c in carriers if c["id"] == p.carrier_id]
        if not carriers:
            raise ValueError(f"no raw-crystal carrier with material {p.material!r} is available; check inventory and human material review")
        p.carrier_id, p.carrier_slot = carriers[0]["id"], carriers[0]["location"]
        if slot_station(bench, p.carrier_slot) != SOURCE_BOARD:
            # away on the stage with a running exfoliation: that job books the
            # carrier and brings it back to its slot before this one starts
            holder = store.reservations("sample").get(p.carrier_id)
            back = (store.get_run(holder["run_id"]) or {}).get("params", {}).get(
                "carrier_slot") if holder else None
            if slot_station(bench, back) != SOURCE_BOARD:
                raise ValueError(f"carrier {p.carrier_id} is at {p.carrier_slot!r}, not on the "
                                 f"{SOURCE_BOARD} board")
            p.carrier_slot = back
        booked = store.reserved_samples()  # chips another queued job already plans on
        if not p.chip_ids:
            bare = [s for s in store.list_samples(kind="chip")
                    if s["status"] in ("bare", "available") and s["id"] not in booked
                    and not s["props"].get("reactor_substrate")
                    and slot_station(bench, s["location"]) == SOURCE_BOARD]
            if len(bare) < p.num_chips:
                raise ValueError(f"need {p.num_chips} bare chip(s), the {SOURCE_BOARD} board "
                                 f"has {len(bare)}")
            p.chip_ids = [s["id"] for s in bare[:p.num_chips]]
        p.num_chips = len(p.chip_ids)
        p.home_slots = []
        for cid in p.chip_ids:
            s = store.get_sample(cid)
            if s and s["props"].get("reactor_substrate"):
                raise ValueError(f"{cid} is a dedicated reactor substrate, not an exfoliation chip")
            if s is None or s["kind"] != "chip" or \
                    slot_station(bench, s["location"]) != SOURCE_BOARD:
                raise ValueError(f"{cid} is not a chip on the {SOURCE_BOARD} board")
            if cid in booked:
                raise ValueError(f"{cid} is already planned for by job "
                                 f"{store.reservations('sample')[cid]['run_id']}")
            p.home_slots.append(s["location"])
        # slots a sample sits in or a queued job has booked (a second
        # exfoliation submitted while the first runs must not plan on its slots)
        used = store.occupied_slots()
        free = [n for n in slots(bench, STORE_BOARD, "chip") if n not in used]
        if len(free) < p.num_chips:
            raise ValueError(f"the {STORE_BOARD} board has {len(free)} free slot(s) for "
                             f"{p.num_chips} exfoliated chip(s)")
        p.store_slots = free[:p.num_chips]
        if p.prepare_mother_tape == "auto":
            tape = store.get_sample(TAPE_SAMPLE) or {"props": {}}
            ready = (normalize_material_code(tape["props"].get("material")) == p.material
                     and int(tape["props"].get("uses") or 0) + p.num_chips <= MAX_TAPE_USES)
            p.prepare_mother_tape = "never" if ready else "always"
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        assert isinstance(params, ExfoliationParams)
        for chip, slot in zip(params.chip_ids, params.store_slots, strict=True):
            store.reserve("slot", slot, run_id, self.name, note=f"{chip} after exfoliation")
            store.reserve("sample", chip, run_id, self.name, note="to be exfoliated")
        if params.prepare_mother_tape == "always" and params.carrier_id:
            # the carrier goes to the stage and back; its slot stays its own
            store.reserve("sample", params.carrier_id, run_id, self.name,
                          note=f"comes back to {params.carrier_slot}")
            if params.carrier_slot:
                store.reserve("slot", params.carrier_slot, run_id, self.name,
                              note=f"{params.carrier_id} comes back here")

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, ExfoliationParams)
        p = params
        steps: list[Step] = [ClearExfoliationStage()]
        if p.prepare_mother_tape == "always":
            assert p.carrier_id and p.carrier_slot
            steps += [
                AdvanceTape(),
                MoveObject(p.carrier_id, "carrier", arm=ARM, camera=CAMERA, dest=STAGE,
                           cfg_key="place_carrier", name="place_carrier_on_exfoliation_stage"),
                LoadMotherTape(p.carrier_id, p.num_massages, p.peel_speed, p.peel_angle_deg,
                               p.press_depth_mm),
                MoveObject(p.carrier_id, "carrier", arm=ARM, camera=CAMERA, dest=p.carrier_slot,
                           cfg_key="place_carrier", name="return_carrier",
                           release_switch=VACUUM),
            ]
        for i, (chip, slot) in enumerate(zip(p.chip_ids, p.store_slots, strict=True)):
            steps.append(MoveObject(chip, "chip", arm=ARM, camera=CAMERA, dest=STAGE,
                                    cfg_key="place_chip_on_exfoliation_stage",
                                    name="place_chip_on_exfoliation_stage"))
            steps.append(PressTape(chip, p.num_massages, p.press_depth_mm))
            if p.target_temp_c > AMBIENT_SETPOINT:
                steps.append(HeatAndReapplyTape(float(p.target_temp_c), p.peel_speed,
                                                p.peel_angle_deg))
            steps.append(CoolAndPeel(chip, p.dwell_s, p.peel_speed, p.peel_angle_deg))
            # the slot was booked at submit; taken anyway at run time (a job
            # planned before the booking existed): the next free one on the board
            steps.append(MoveObject(chip, "chip", arm=ARM, camera=CAMERA, dest=slot,
                                    cfg_key="store_chip", name="store_exfoliated_chip",
                                    release_switch=VACUUM, fallback=(STORE_BOARD, "chip"),
                                    param_path=["store_slots", i]))
        steps.append(ResetExfoliation())
        return steps
