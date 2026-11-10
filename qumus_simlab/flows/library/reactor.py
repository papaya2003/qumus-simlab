"""Guarded experimental reactor jobs with observed, rather than verbal, results.

A successful growth-window job verifies execution and observations, not a
material-quality target. Interrupted thermal/exposure history parks for review.
All operating envelopes are explicitly synthetic development assumptions.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from qumus_simlab.core.errors import NeedsHumanError, PreconditionFailed, QumusError
from qumus_simlab.core.types import Pose
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.flow import Flow, register_flow
from qumus_simlab.flows.library.primitives import ramp_heater, require_sample, sleep_watched
from qumus_simlab.flows.library.transfer import MoveObject
from qumus_simlab.flows.step import Correction, Observation, ProbeResult, Step, Verdict
from qumus_simlab.flows.workflow import Assembly, Molecule
from qumus_simlab.records.reactor_pairs import prepared_pair, validate_pair_observation

ARM, STAGE = "arm_reactor", "reactor_slot"
INSERT = "reactor_insert"
HEATER, GAS, SCOPE = "reactor_heater", "reactor_flow", "reactor_microscope"
PORT = "reactor_port"
DEVICES = (ARM, STAGE, INSERT, HEATER, GAS, SCOPE)


class ChipParams(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    chip_id: str


class LoadParams(ChipParams):
    home_slot: str | None = None
    insertion_y_mm: float = 160.0


class GrowthParams(ChipParams):
    temperature_c: float = Field(820, ge=22, le=950)
    flow_sccm: float = Field(200, ge=0, le=400)
    duration_s: float = Field(60, gt=0, le=900)
    observation_interval_s: float = Field(20, ge=5, le=60)


class ObserveParams(ChipParams):
    autofocus: bool = False


class UnloadParams(ChipParams):
    home_slot: str | None = None


class PositionParams(ChipParams):
    x_mm: float = Field(0, ge=-4, le=4)
    y_mm: float = Field(0, ge=-4, le=4)


class ShutdownParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def require_module(bench: dict[str, Any]) -> None:
    if "reactor_module" not in bench:
        raise ValueError("experimental reactor is disabled; enable reactor.enabled in the testbench")


def reactor_chip(store: Any, chip_id: str) -> dict[str, Any]:
    chip = store.get_sample(chip_id)
    if not chip or chip["kind"] != "chip" or not chip["props"].get("reactor_substrate"):
        raise ValueError(f"{chip_id} is not a dedicated reactor substrate")
    if chip_id in store.reserved_samples():
        raise ValueError(f"{chip_id} is reserved by another job")
    return chip


def require_loaded(ctx: FlowContext, chip_id: str) -> None:
    require_module(ctx.bench)
    chip = require_sample(ctx, chip_id, "chip")
    if chip["location"] != PORT or chip["props"].get("location_uncertain"):
        raise PreconditionFailed(f"{chip_id} is not verified in the reactor carrier")


def access_limits(ctx: FlowContext):
    limits = next((x for x in ctx.gate.limits.reactor_interlocks if x.stage == STAGE), None)
    if limits is None:
        raise PreconditionFailed("reactor access limits are not configured")
    return limits


def observe(ctx: FlowContext) -> Observation:
    frame = ctx.gate.microscope(SCOPE).capture()
    m = dict(frame.meta.get("reactor") or {})
    m.pop("chip_id", None)  # identity comes from the verified carrier ledger
    if m.get("experiment_kind") == "material_metal":
        visible = bool((m.get("material_metal_observation") or {}).get("visible"))
    elif m.get("experiment_kind") == "two_flake_reaction":
        visible = bool((m.get("pair_observation") or {}).get("visible"))
    else:
        visible = bool(m.get("visible")) and m.get("coverage") is not None
    image = frame.saved_path()
    return Observation(Verdict.OK if visible else Verdict.UNCLEAR, m,
                       images=[image] if image else [],
                       message="synthetic reactor observation" if visible else
                       f"reactor image is not measurable: {m.get('reason', 'unknown')}")


def record_observation(ctx: FlowContext, chip_id: str, obs: Observation) -> None:
    if obs.measurements.get("experiment_kind") == "material_metal":
        from qumus_simlab.records.material_metal_simulation import device_preparation_errors

        if errors := device_preparation_errors(ctx.store, ctx.project_id, chip_id):
            raise PreconditionFailed("; ".join(errors))
        observed = {**obs.measurements["material_metal_observation"], "job_id": ctx.run_id,
                    "project_id": ctx.project_id, "images": obs.images, "timestamp": ctx.clock.now()}
        chip = ctx.store.get_sample(chip_id)
        with ctx.store._transaction():
            ctx.store.upsert_sample(chip_id, "chip", props={"material_metal_observation": observed,
                "images": {**(chip["props"].get("images") or {}), **({"reactor": obs.images[-1]} if obs.images else {})},
                "image_source": "simulated"})
            ctx.store.emit("material_metal.observed", {"observation": observed}, run_id=ctx.run_id)
        return
    if obs.measurements.get("experiment_kind") == "two_flake_reaction":
        validate_pair_observation(ctx.store, chip_id, obs.measurements.get("pair_observation") or {},
                                  run_id=ctx.run_id)
    result = {**obs.measurements, "job_id": ctx.run_id, "images": obs.images,
              "timestamp": ctx.clock.now(), "source": "synthetic_reactor_observation"}
    chip = require_sample(ctx, chip_id, "chip")
    run = ctx.store.get_run(ctx.run_id)
    previous = chip["props"].get("reactor_result") or {}
    result["growth_job_id"] = (ctx.run_id if run and run["flow"] == "reactor_growth" else
                               previous.get("growth_job_id"))
    result["reaction_job_id"] = (ctx.run_id if run and run["flow"] == "reactor_reaction" else
                                 previous.get("reaction_job_id"))
    # An image cannot reconstruct an interrupted thermal/exposure history.
    props: dict[str, Any] = {"reactor_result": result}
    if obs.images:
        props["images"] = {**(chip["props"].get("images") or {}), "reactor": obs.images[-1]}
        props["image_source"] = "simulated"
    ctx.store.upsert_sample(chip_id, "chip", props=props)
    ctx.store.emit("reactor.observation", {"chip_id": chip_id, **result}, run_id=ctx.run_id)


def disable_targets(ctx: FlowContext) -> None:
    for name, action in ((HEATER, lambda: ctx.gate.heater(HEATER).off()),
                         (GAS, lambda: ctx.gate.gas_flow(GAS).set_flow(0))):
        try:
            action()
        except QumusError as exc:
            ctx.store.emit("notification", {"level": "danger", "text":
                f"reactor shutdown could not confirm {name}: {exc}"}, run_id=ctx.run_id)


def process_evidence(ctx: FlowContext, chip_id: str) -> dict[str, Any]:
    """Fresh guarded feedback for the shared completion contract."""
    heater, gas = ctx.gate.heater(HEATER), ctx.gate.gas_flow(GAS)
    chip = require_sample(ctx, chip_id, "chip")
    lim = access_limits(ctx)
    return {"chip_id": chip_id, "timestamp": ctx.clock.now(), "location": chip["location"],
            "location_uncertain": bool(chip["props"].get("location_uncertain")),
            "temperature_target_c": heater.get_target(), "flow_target_sccm": gas.get_target(),
            "temperature_c": heater.read_temperature(), "flow_sccm": gas.read_flow(),
            "insert_y_mm": ctx.gate.stage(INSERT).get_pose().y,
            "assembly_pose": ctx.gate.stage(STAGE).get_pose().model_dump(),
            "access_max_celsius": lim.access_max_celsius, "access_max_flow_sccm": lim.access_max_flow_sccm,
            "state_uncertain": bool(chip["props"].get("reactor_state_uncertain")),
            "source": "guarded_device_feedback"}


class RecordReactorProcess(Step):
    name = "record_reactor_process"
    devices = (HEATER, GAS, STAGE, INSERT)

    def __init__(self, chip_id: str, *, cold_storage: bool = False) -> None:
        self.chip_id, self.cold_storage = chip_id, cold_storage

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        pass

    def verify(self, ctx: FlowContext) -> Observation:
        m = process_evidence(ctx, self.chip_id)
        ok = True
        if self.cold_storage:
            location = ctx.bench.get("slots", {}).get(m["location"], {})
            ok = (location.get("station") == "reactor_storage" and not m["location_uncertain"]
                  and m["temperature_target_c"] is None
                  and m["flow_target_sccm"] == 0 and m["temperature_c"] <= m["access_max_celsius"]
                  and m["flow_sccm"] <= m["access_max_flow_sccm"] and m["insert_y_mm"] == 0)
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR, m,
                           message="Measured reactor terminal conditions" if ok else "Reactor recovery remains incomplete")

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        assert obs is not None
        ctx.store.emit("reactor.process", obs.measurements, run_id=ctx.run_id)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


class ReactorStep(Step):
    devices = (HEATER, GAS)

    def safe_state(self, ctx: FlowContext) -> None:
        disable_targets(ctx)

    def before_park(self, ctx: FlowContext) -> None:
        self.safe_state(ctx)


class LoadReactorChip(MoveObject):
    def __init__(self, chip_id: str, home_slot: str) -> None:
        super().__init__(chip_id, "chip", arm=ARM, camera="overview_camera", dest=PORT,
                         cfg_key="place_chip_in_reactor", name="load_reactor_chip", source=home_slot)
        self.home_slot = home_slot

    def preconditions(self, ctx: FlowContext) -> None:
        p = ctx.gate.stage(STAGE).get_pose()
        inserted = ctx.gate.stage(INSERT).get_pose().y
        if max(math.hypot(p.x, p.y), abs(inserted)) > access_limits(ctx).loading_tolerance_mm:
            raise PreconditionFailed("reactor carrier is not at its loading position")
        super().preconditions(ctx)
        # The narrow slot needs a bounded sequence of independent observed
        # re-seatings, not a reduced placement-error profile.
        self.max_corrections = ctx.bench["reactor_module"].get("acquisition", {}).get(
            "placement_corrections", self.max_corrections)

    def ledger_props(self, obs: Observation | None) -> dict[str, Any]:
        return {**super().ledger_props(obs), "reactor_home_slot": self.home_slot}

    def verify(self, ctx: FlowContext) -> Observation:
        obs = super().verify(ctx)
        if obs.verdict != Verdict.OK:
            return obs
        # A centre inside the tolerance circle does not prove a narrow chip fits.
        chip = require_sample(ctx, self.sample_id, "chip")
        size = chip["props"].get("size_mm")
        if not size or len(size) != 2 or min(size) <= 0:
            obs.verdict, obs.message = Verdict.UNCLEAR, "chip dimensions are not established"
            return obs
        target = self._target(ctx)
        pose = Pose(**obs.measurements["arm_pose"])
        theta = math.radians(pose.rotation - target.rotation)
        w, h = size
        half_x = abs(pose.x - target.x) + (w * abs(math.cos(theta)) + h * abs(math.sin(theta))) / 2
        half_y = abs(pose.y - target.y) + (w * abs(math.sin(theta)) + h * abs(math.cos(theta))) / 2
        slot_w, slot_h = ctx.bench["reactor_module"]["geometry"]["carrier_slot_size_mm"]
        margin = self._cfg(ctx)["edge_margin_mm"]
        clearance = min(slot_w / 2 - half_x, slot_h / 2 - half_y)
        obs.measurements["slot_edge_clearance_mm"] = clearance
        obs.expected.update({"slot_size_mm": [slot_w, slot_h], "min_edge_clearance_mm": margin})
        if clearance < margin:
            obs.verdict = Verdict.CORRECTABLE
            obs.message = "observed chip footprint is not seated within the narrow carrier slot"
        return obs


class PositionCarrier(ReactorStep):
    devices = (STAGE, INSERT)

    def __init__(self, y: float, x: float = 0, chip_id: str | None = None,
                 *, insertion: bool = True) -> None:
        self.x, self.y, self.chip_id = x, y, chip_id
        self.stage = INSERT if insertion else STAGE
        self.name = ("insert_reactor_carrier" if y else "retract_reactor_carrier") if insertion \
            else "position_reactor_assembly"

    def preconditions(self, ctx: FlowContext) -> None:
        if self.chip_id is not None:
            require_loaded(ctx, self.chip_id)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        ctx.gate.stage(self.stage).move_to(Pose(x=self.x, y=self.y))

    def verify(self, ctx: FlowContext) -> Observation:
        p = ctx.gate.stage(self.stage).get_pose()
        ok = abs(p.x - self.x) < 0.01 and abs(p.y - self.y) < 0.01 and p.z == p.rotation == 0
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR, p.model_dump(),
                           expected={"x": self.x, "y": self.y, "z": 0, "rotation": 0})

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.DONE if self.verify(ctx).verdict == Verdict.OK else ProbeResult.NOT_DONE


class FocusReactor(ReactorStep):
    name = "focus_reactor_microscope"
    devices = (SCOPE,)

    def __init__(self, chip_id: str) -> None:
        self.chip_id = chip_id

    def preconditions(self, ctx: FlowContext) -> None:
        require_loaded(ctx, self.chip_id)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        scope = ctx.gate.microscope(SCOPE)
        limits = ctx.gate.limits.microscopes[SCOPE]
        scope.set_objective(min(limits.objectives))
        low, high = limits.focus
        best, sharpest = 0.0, -1.0
        for z in [low + (high - low) * i / 20 for i in range(21)]:
            ctx.check_cancel()
            scope.set_focus(z)
            m = scope.capture().meta.get("reactor", {})
            score = float(m.get("sharpness", 0)) if m.get("reason") != "carrier_empty" else 0.0
            if score > sharpest:
                best, sharpest = z, score
        centre = best
        for i in range(-5, 6):
            ctx.check_cancel()
            z = max(low, min(high, centre + i / 100))
            scope.set_focus(z)
            m = scope.capture().meta.get("reactor", {})
            score = float(m.get("sharpness", 0)) if m.get("reason") != "carrier_empty" else 0.0
            if score > sharpest:
                best, sharpest = z, score
        scope.set_focus(best)

    def verify(self, ctx: FlowContext) -> Observation:
        return observe(ctx)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.DONE if self.verify(ctx).verdict == Verdict.OK else ProbeResult.NOT_DONE


class AcquireReactor(FocusReactor):
    """Cold, bounded optical acquisition around the measured seating pose.

    No hidden chip coordinates or out-of-view material polygons guide motion.
    A coarse camera measurement seeds the search; only visible optical evidence
    can terminate it. This step stays inside the original load reservation.
    """

    name = "acquire_reactor_pair"
    devices = (SCOPE, STAGE, HEATER, GAS)

    def preconditions(self, ctx: FlowContext) -> None:
        super().preconditions(ctx)
        lim = access_limits(ctx)
        heater, gas = ctx.gate.heater(HEATER), ctx.gate.gas_flow(GAS)
        if (heater.get_target() is not None or gas.get_target() != 0
                or heater.read_temperature() > lim.access_max_celsius
                or gas.read_flow() > lim.access_max_flow_sccm):
            raise PreconditionFailed("optical acquisition requires a cold reactor with targets off")

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        super().act(ctx, correction)
        def complete_view(obs: Observation) -> bool:
            inputs = (obs.measurements.get("material_view") or {}).get("inputs")
            return obs.verdict == Verdict.OK and (inputs is None or bool(inputs)
                and all(f.get("fully_visible") for f in inputs))

        if complete_view(observe(ctx)):
            return
        chip = require_sample(ctx, self.chip_id, "chip")
        geo = ctx.bench["reactor_module"]["geometry"]
        measured = chip["props"].get("pose") or geo["load"]
        seed = (geo["load"]["x"] - measured["x"], geo["load"]["y"] - measured["y"])
        settings = ctx.bench["reactor_module"].get("acquisition", {})
        pitch = float(settings.get("search_pitch_mm", 0.10))
        rings = int(settings.get("search_rings", 5))
        offsets = [(x, y) for x in range(-rings, rings + 1) for y in range(-rings, rings + 1)]
        offsets.sort(key=lambda p: (max(abs(p[0]), abs(p[1])), p[0] ** 2 + p[1] ** 2))
        stage = ctx.gate.stage(STAGE)
        bounds = ctx.gate.limits.stages[STAGE]
        visited = []
        for dx, dy in offsets:
            ctx.check_cancel()
            target = Pose(x=seed[0] + dx * pitch, y=seed[1] + dy * pitch)
            if not bounds.x[0] <= target.x <= bounds.x[1] or not bounds.y[0] <= target.y <= bounds.y[1]:
                continue
            # Respect the existing fine-motion limit even when the first coarse
            # measurement lies far from the current assembly position.
            here = stage.get_pose()
            n = max(1, math.ceil(here.xy_distance(target) / access_limits(ctx).max_fine_step_mm))
            for i in range(1, n + 1):
                ctx.check_cancel()
                stage.move_to(Pose(x=here.x + (target.x - here.x) * i / n,
                                   y=here.y + (target.y - here.y) * i / n))
            obs = observe(ctx)
            visited.append({"x_mm": target.x, "y_mm": target.y, "visible": obs.verdict == Verdict.OK})
            # A clipped pair is not acquired. Centre only the visible portion
            # of its optical contours, then require both full outlines. Never
            # use chip-local model coordinates or an invisible polygon centre.
            for _ in range(3):
                if obs.verdict != Verdict.OK or complete_view(obs):
                    break
                m = obs.measurements
                width, height = m.get("frame_size_px", [640, 420])
                points = [(max(0, min(width, x)), max(0, min(height, y)))
                          for f in (m.get("material_view") or {}).get("inputs", []) if f.get("visible")
                          for x, y in f.get("outline_px", [])]
                if not points:
                    break
                cx = (min(p[0] for p in points) + max(p[0] for p in points)) / 2
                cy = (min(p[1] for p in points) + max(p[1] for p in points)) / 2
                fw, fh = m["fov_mm"]
                here = stage.get_pose()
                refined = Pose(x=here.x - (cx / width - 0.5) * fw,
                               y=here.y - (0.5 - cy / height) * fh)
                if not bounds.x[0] <= refined.x <= bounds.x[1] or not bounds.y[0] <= refined.y <= bounds.y[1]:
                    break
                stage.move_to(refined)
                obs = observe(ctx)
                visited.append({"x_mm": refined.x, "y_mm": refined.y,
                                "visible": obs.verdict == Verdict.OK, "optical_centring": True})
            if complete_view(obs):
                break
        ctx.store.emit("reactor.acquisition", {"chip_id": self.chip_id,
            "source": "measured_seating_and_bounded_optical_search", "positions": visited}, run_id=ctx.run_id)

    def verify(self, ctx: FlowContext) -> Observation:
        obs = super().verify(ctx)
        inputs = (obs.measurements.get("material_view") or {}).get("inputs")
        if inputs is not None and (not inputs or not all(f.get("fully_visible") for f in inputs)):
            obs.verdict, obs.message = Verdict.UNCLEAR, "Both complete input outlines must be acquired before exposure"
        return obs


class SetReactorConditions(ReactorStep):
    name = "set_reactor_conditions"
    safe_to_resume = False
    pause_inside = True

    def __init__(self, params: GrowthParams) -> None:
        self.params = params

    def preconditions(self, ctx: FlowContext) -> None:
        require_loaded(ctx, self.params.chip_id)
        obs = observe(ctx)
        if obs.verdict != Verdict.OK:
            raise PreconditionFailed(obs.message)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        ramp_heater(ctx, HEATER, self.params.temperature_c)
        deadline = ctx.clock.now() + 900
        while abs(ctx.gate.heater(HEATER).read_temperature() - self.params.temperature_c) > 3:
            if ctx.clock.now() >= deadline:
                raise NeedsHumanError("reactor temperature did not settle")
            sleep_watched(ctx, 5)
        ctx.gate.gas_flow(GAS).set_flow(self.params.flow_sccm)
        deadline = ctx.clock.now() + 60
        while abs(ctx.gate.gas_flow(GAS).read_flow() - self.params.flow_sccm) > max(
                1, self.params.flow_sccm * 0.02):
            if ctx.clock.now() >= deadline:
                raise NeedsHumanError("reactor flow did not settle")
            sleep_watched(ctx, 2)

    def verify(self, ctx: FlowContext) -> Observation:
        t, f = ctx.gate.heater(HEATER).read_temperature(), ctx.gate.gas_flow(GAS).read_flow()
        ok = abs(t - self.params.temperature_c) <= 3 and abs(f - self.params.flow_sccm) <= max(
            1, self.params.flow_sccm * 0.02)
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR,
                           {"temperature_c": t, "flow_sccm": f},
                           expected={"temperature_c": self.params.temperature_c,
                                     "flow_sccm": self.params.flow_sccm})

    def before_park(self, ctx: FlowContext) -> None:
        ctx.store.upsert_sample(self.params.chip_id, "chip", props={"reactor_state_uncertain": True})
        super().before_park(ctx)

    def on_give_up(self, ctx: FlowContext, obs: Observation | None) -> None:
        ctx.store.upsert_sample(self.params.chip_id, "chip", props={"reactor_state_uncertain": True})


class GrowthWindow(ReactorStep):
    name = "observe_reactor_growth"
    safe_to_resume = False
    pause_inside = True
    devices = (HEATER, GAS, SCOPE)

    def __init__(self, params: GrowthParams) -> None:
        self.params = params

    def preconditions(self, ctx: FlowContext) -> None:
        require_loaded(ctx, self.params.chip_id)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        left = self.params.duration_s
        while left > 1e-8:
            dt = min(left, self.params.observation_interval_s)
            sleep_watched(ctx, dt)
            obs = observe(ctx)
            if obs.verdict != Verdict.OK:
                raise PreconditionFailed(obs.message)
            record_observation(ctx, self.params.chip_id, obs)
            left -= dt

    def verify(self, ctx: FlowContext) -> Observation:
        return observe(ctx)

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        if obs is not None:
            record_observation(ctx, self.params.chip_id, obs)
        ctx.store.upsert_sample(self.params.chip_id, "chip", status="reactor_processed")

    def before_park(self, ctx: FlowContext) -> None:
        ctx.store.upsert_sample(self.params.chip_id, "chip", props={"reactor_state_uncertain": True})
        super().before_park(ctx)

    def on_give_up(self, ctx: FlowContext, obs: Observation | None) -> None:
        ctx.store.upsert_sample(self.params.chip_id, "chip", props={"reactor_state_uncertain": True})


class ObserveReactor(ReactorStep):
    name = "observe_reactor"
    devices = (SCOPE,)

    def __init__(self, chip_id: str) -> None:
        self.chip_id = chip_id

    def preconditions(self, ctx: FlowContext) -> None:
        require_loaded(ctx, self.chip_id)

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        pass

    def verify(self, ctx: FlowContext) -> Observation:
        return observe(ctx)

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        if obs is not None:
            record_observation(ctx, self.chip_id, obs)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


class ShutdownReactor(ReactorStep):
    name = "disable_reactor_targets"

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        ctx.gate.heater(HEATER).off()
        ctx.gate.gas_flow(GAS).set_flow(0)

    def verify(self, ctx: FlowContext) -> Observation:
        h, f = ctx.gate.heater(HEATER), ctx.gate.gas_flow(GAS)
        ok = h.get_target() is None and f.get_target() == 0
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR,
                           {"temperature_target_c": h.get_target(), "flow_target_sccm": f.get_target(),
                            "temperature_c": h.read_temperature(), "flow_sccm": f.read_flow()},
                           message="targets disabled; residual temperature and flow may remain")

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.DONE if self.verify(ctx).verdict == Verdict.OK else ProbeResult.NOT_DONE


class WaitForAccess(ReactorStep):
    name = "wait_for_reactor_access"

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        end = ctx.clock.now() + 900
        lim = access_limits(ctx)
        while (ctx.gate.heater(HEATER).read_temperature() > lim.access_max_celsius
               or ctx.gate.gas_flow(GAS).read_flow() > lim.access_max_flow_sccm):
            if ctx.clock.now() >= end:
                raise NeedsHumanError("reactor did not reach its synthetic unloading conditions")
            sleep_watched(ctx, 5)

    def verify(self, ctx: FlowContext) -> Observation:
        t, f = ctx.gate.heater(HEATER).read_temperature(), ctx.gate.gas_flow(GAS).read_flow()
        lim = access_limits(ctx)
        return Observation(Verdict.OK if t <= lim.access_max_celsius
                           and f <= lim.access_max_flow_sccm else Verdict.UNCLEAR,
                           {"temperature_c": t, "flow_sccm": f})

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.DONE if self.verify(ctx).verdict == Verdict.OK else ProbeResult.NOT_DONE


def load_and_acquire(chip_id: str, home_slot: str, insertion_y_mm: float) -> Molecule:
    return Molecule("load_and_acquire", "Seat the carrier, insert the rod and acquire complete focused input outlines",
        (LoadReactorChip(chip_id, home_slot), PositionCarrier(insertion_y_mm, chip_id=chip_id), AcquireReactor(chip_id)))


def observe_carrier(chip_id: str, *, autofocus: bool = False, name: str = "observe_carrier") -> Molecule:
    return Molecule(name, "Save a measured image of the loaded carrier without advancing exposure by imaging",
        (*((FocusReactor(chip_id),) if autofocus else ()), ObserveReactor(chip_id)))


def establish_conditions(step: SetReactorConditions) -> Molecule:
    return Molecule("establish_conditions", "Set temperature and gas targets, then verify measured feedback settles", (step,))


def observe_exposure(step: GrowthWindow) -> Molecule:
    return Molecule("observe_exposure", "Run a bounded exposure window with periodic saved observations", (step,))


def shutdown_and_cool() -> Molecule:
    return Molecule("shutdown_and_cool", "Disable targets and wait for measured temperature and flow to permit access",
                    (ShutdownReactor(), WaitForAccess()))


def store_carrier(chip_id: str, home_slot: str, *, step_name: str = "store_reactor_chip") -> Molecule:
    return Molecule("store_carrier", "Reset assembly XY, retract the rod and return the carrier to its reserved slot",
        (PositionCarrier(0, chip_id=chip_id, insertion=False), PositionCarrier(0, chip_id=chip_id),
         MoveObject(chip_id, "chip", arm=ARM, camera="overview_camera", source=PORT, dest=home_slot,
                    cfg_key="store_chip", name=step_name)))


def record_terminal(chip_id: str, *, cold_storage: bool = False) -> Molecule:
    return Molecule("record_terminal", "Read current carrier, temperature, flow, targets and insertion state",
                    (RecordReactorProcess(chip_id, cold_storage=cold_storage),))


class ReactorFlow(Flow):
    devices = DEVICES

    def on_step_failed(self, params: BaseModel, step: Step, outcome: Any,
                       ctx: FlowContext) -> None:
        # Includes precondition/visibility failures, for which the generic step
        # runner intentionally makes no recovery move. Exposure cannot continue
        # while a reactor job is parked for review.
        disable_targets(ctx)
        if self.name in ("reactor_growth", "reactor_reaction"):
            ctx.store.upsert_sample(params.chip_id, "chip",
                                    props={"reactor_state_uncertain": True})

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        require_module(self.bench)
        if isinstance(params, ChipParams):
            reactor_chip(store, params.chip_id)
            receipt = store.get_sample(params.chip_id)["props"].get("material_metal_preparation") or {}
            if receipt.get("scope") == "device_simulation":
                from qumus_simlab.records.material_metal_simulation import device_preparation_errors

                if errors := device_preparation_errors(store, receipt["project_id"], params.chip_id):
                    raise ValueError("; ".join(errors))
            elif self.bench["reactor_module"].get("experiment_kind") == "two_flake_reaction":
                prepared_pair(store, params.chip_id, allow_consumed=True)
        if PORT in store.reservations("slot"):
            raise ValueError("reactor carrier is reserved by another job")
        return params

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        store.reserve("slot", PORT, run_id, self.name)
        if isinstance(params, ChipParams):
            store.reserve("sample", params.chip_id, run_id, self.name)
            receipt = store.get_sample(params.chip_id)["props"].get("material_metal_preparation") or {}
            if receipt.get("scope") == "device_simulation":
                for fid in receipt["top_first_flake_ids"]:
                    store.reserve("sample", fid, run_id, self.name)
            elif self.bench["reactor_module"].get("experiment_kind") == "two_flake_reaction":
                pair = prepared_pair(store, params.chip_id, run_id=run_id, allow_consumed=True)
                for fid in pair.input_flake_ids:
                    store.reserve("sample", fid, run_id, self.name)


@register_flow
class LoadReactorFlow(ReactorFlow):
    name = "load_reactor_chip"
    Params = LoadParams
    devices = (*DEVICES, "overview_camera")

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        p = super().resolve_params(params, store).model_copy()
        assert isinstance(p, LoadParams)
        chip = store.get_sample(p.chip_id)
        slot = self.bench["slots"].get(chip["location"], {})
        if slot.get("station") != "reactor_storage":
            raise ValueError("load a reactor chip from its dedicated storage")
        if any(s["location"] == PORT for s in store.list_samples()):
            raise ValueError("reactor carrier is occupied")
        p.home_slot = chip["location"]
        p.insertion_y_mm = self.bench["reactor_module"]["geometry"]["process_y_mm"]
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        super().reserve(params, store, run_id)
        store.reserve("slot", params.home_slot, run_id, self.name)

    def build_workflow(self, params: BaseModel) -> Assembly:
        assert isinstance(params, LoadParams)
        return Assembly(self.name, "Load and optically acquire a stored reactor carrier", (
            load_and_acquire(params.chip_id, params.home_slot or "", params.insertion_y_mm),
            record_terminal(params.chip_id)))



@register_flow
class ReactorGrowthFlow(ReactorFlow):
    name = "reactor_growth"
    Params = GrowthParams

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        if self.name == "reactor_growth" and self.bench.get("reactor_module", {}).get(
                "experiment_kind") == "two_flake_reaction":
            raise ValueError("this bench uses two-flake reaction; field-growth jobs are disabled")
        p = super().resolve_params(params, store)
        chip = store.get_sample(p.chip_id)
        if chip["location"] != PORT or chip["props"].get("location_uncertain"):
            raise ValueError("load and verify this chip in the reactor before growth")
        if chip["props"].get("reactor_state_uncertain"):
            raise ValueError("reactor exposure history is uncertain; review before starting another window")
        return p

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, GrowthParams)
        return [SetReactorConditions(params), GrowthWindow(params)]


def require_pair_evidence(ctx: FlowContext, chip_id: str) -> None:
    prepared_pair(ctx.store, chip_id, run_id=ctx.run_id)
    obs = observe(ctx)
    if obs.verdict != Verdict.OK or obs.measurements.get("experiment_kind") != "two_flake_reaction":
        raise PreconditionFailed("a visible two-flake material observation is required before exposure")
    try:
        validate_pair_observation(ctx.store, chip_id, obs.measurements.get("pair_observation") or {},
                                  run_id=ctx.run_id)
    except ValueError as exc:
        raise PreconditionFailed(str(exc)) from exc


class PairConditions(SetReactorConditions):
    def preconditions(self, ctx: FlowContext) -> None:
        super().preconditions(ctx)
        require_pair_evidence(ctx, self.params.chip_id)


class PairWindow(GrowthWindow):
    name = "observe_reactor_reaction"

    def preconditions(self, ctx: FlowContext) -> None:
        super().preconditions(ctx)
        require_pair_evidence(ctx, self.params.chip_id)


@register_flow
class ReactorReactionFlow(ReactorGrowthFlow):
    """Reuse guarded mechanics; never substitute the legacy field-growth model."""

    name = "reactor_reaction"

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        p = super().resolve_params(params, store)
        if self.bench["reactor_module"].get("experiment_kind") != "two_flake_reaction":
            raise ValueError("two-flake material backend is not enabled on this bench")
        prepared_pair(store, p.chip_id)
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        pair = prepared_pair(store, params.chip_id, run_id=run_id)
        super().reserve(params, store, run_id)
        for fid in pair.input_flake_ids:
            store.reserve("sample", fid, run_id, self.name)

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, GrowthParams)
        return [PairConditions(params), PairWindow(params), RecordReactorProcess(params.chip_id)]


@register_flow
class VerifyReactorStateFlow(Flow):
    """Read fresh cold-storage proof without repeating preparation or reaction."""

    name = "verify_reactor_state"
    Params = ChipParams
    devices = (ARM, STAGE, INSERT, HEATER, GAS)
    restore_heaters_on_resume = False

    @property
    def shutdown_devices(self) -> tuple[str, ...]:
        return ()  # Readback failure cannot turn a query into physical shutdown.

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        require_module(self.bench)
        chip = reactor_chip(store, params.chip_id)
        location = self.bench.get("slots", {}).get(chip["location"], {})
        if location.get("station") != "reactor_storage" or chip["props"].get("location_uncertain"):
            raise ValueError("Verify a carrier already at its known dedicated reactor storage slot; unload a loaded carrier first")
        if PORT in store.reservations("slot") or chip["location"] in store.reservations("slot"):
            raise ValueError("Reactor or carrier storage is reserved by another job")
        return params

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        store.reserve("sample", params.chip_id, run_id, self.name)
        store.reserve("slot", store.get_sample(params.chip_id)["location"], run_id, self.name)
        store.reserve("slot", PORT, run_id, self.name)

    def build_workflow(self, params: BaseModel) -> Molecule:
        return record_terminal(params.chip_id, cold_storage=True)



@register_flow
class HoldReactorObservationFlow(ReactorFlow):
    name = "hold_reactor_observation"
    Params = ChipParams

    def build_steps(self, params: BaseModel) -> list[Step]:
        return [ObserveReactor(params.chip_id), RecordReactorProcess(params.chip_id)]


@register_flow
class ObserveReactorFlow(ReactorFlow):
    name = "observe_reactor"
    Params = ObserveParams

    def build_workflow(self, params: BaseModel) -> Molecule:
        assert isinstance(params, ObserveParams)
        return observe_carrier(params.chip_id, autofocus=params.autofocus)



@register_flow
class ShutdownReactorFlow(ReactorFlow):
    name = "shutdown_reactor"
    Params = ShutdownParams

    def build_steps(self, params: BaseModel) -> list[Step]:
        return [ShutdownReactor()]


@register_flow
class PositionReactorFlow(ReactorFlow):
    name = "position_reactor_slot"
    Params = PositionParams

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, PositionParams)
        return [PositionCarrier(params.y_mm, params.x_mm, params.chip_id, insertion=False)]


@register_flow
class UnloadReactorFlow(ReactorFlow):
    name = "unload_reactor_chip"
    Params = UnloadParams
    devices = (*DEVICES, "overview_camera")

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        p = super().resolve_params(params, store).model_copy()
        assert isinstance(p, UnloadParams)
        chip = store.get_sample(p.chip_id)
        if chip["location"] != PORT:
            raise ValueError("chip is not in the reactor carrier")
        p.home_slot = chip["props"].get("reactor_home_slot")
        if not p.home_slot or p.home_slot in store.occupied_slots():
            raise ValueError("reactor chip's original storage slot is not available")
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        super().reserve(params, store, run_id)
        store.reserve("slot", params.home_slot, run_id, self.name)

    def build_workflow(self, params: BaseModel) -> Assembly:
        assert isinstance(params, UnloadParams)
        return Assembly(self.name, "Cool and return the loaded carrier to storage", (
            shutdown_and_cool(), store_carrier(params.chip_id, params.home_slot or ""),
            record_terminal(params.chip_id, cold_storage=True)))
