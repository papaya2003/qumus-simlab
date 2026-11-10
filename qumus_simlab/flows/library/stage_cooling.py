"""Final experiment cooling, independent of intermediate handling thresholds.

Fans use the existing Switch interface. Switch readback alone does not prove
airflow; measured temperatures determine completion. Rates are synthetic, not
a real apparatus calibration or a firmware fan protocol.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from qumus_simlab.core.errors import DangerousError, NeedsHumanError
from qumus_simlab.flows.flow import Flow, register_flow
from qumus_simlab.flows.step import Observation, ProbeResult, Step, Verdict

STAGES = {"exfoliation_heater": "exfoliation_fan", "stamp_heater": "transfer_stage_fan"}
STAGE_DEVICES = (*STAGES, *STAGES.values())
FINAL_BELOW_C = 40.0


class CoolingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def stage_readings(gate):
    missing = [device for device in STAGE_DEVICES if device not in gate.devices.devices]
    if missing:
        raise NeedsHumanError("Stage cooling interfaces are unavailable", missing_devices=missing)
    readings = {}
    for heater, fan in STAGES.items():
        h = gate.heater(heater)
        temperature = h.read_temperature()
        if isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not math.isfinite(temperature):
            raise DangerousError(f"{heater}: invalid temperature feedback; experiment is not verified cold")
        readings[heater] = {"temperature_c": temperature, "target_c": h.get_target(),
                            "fan": fan, "fan_commanded_on": gate.switch(fan).get()}
    return readings


def stages_cold(readings):
    return all(r["temperature_c"] < FINAL_BELOW_C and r["target_c"] is None for r in readings.values())


def stages_finished(readings):
    return stages_cold(readings) and all(r["fan_commanded_on"] is False for r in readings.values())


def _disable_stage_heaters(gate):
    """Attempt and confirm both shutdowns, including during failure recovery."""
    failed = []
    for heater in STAGES:
        try:
            h = gate.heater(heater)
            h.off()
            if h.get_target() is not None:
                failed.append(heater)
        except Exception:
            failed.append(heater)
    if failed:
        raise DangerousError("Stage heater shutdown was not confirmed", failed_devices=failed)


class CoolExperimentStages(Step):
    name = "cool_experiment_stages"
    devices = STAGE_DEVICES

    def expected(self, ctx):
        return {"both_stages_strictly_below_c": FINAL_BELOW_C, "heater_targets": None,
                "fans_commanded_off_after_cooling": True}

    def act(self, ctx, correction):
        # Attempt both energy shutdowns even if one output fails. Never restore
        # a pre-cooling target on Continue, including a stop before this act().
        _disable_stage_heaters(ctx.gate)
        missing = [fan for fan in STAGES.values() if fan not in ctx.gate.devices.devices]
        if missing:
            raise NeedsHumanError("Stage fan interface is unavailable; cooling is unverified", missing_devices=missing)
        cfg = ctx.verification["heating"]
        meta = ctx.store.get_run(ctx.run_id)["meta"]
        deadline = meta.get("stage_cooling_deadline")
        if deadline is None:
            deadline = ctx.clock.now() + cfg["cool_timeout_s"]
            ctx.store.update_run_meta(ctx.run_id, stage_cooling_deadline=deadline)
        for fan in STAGES.values():
            ctx.gate.switch(fan).set(True)
            if ctx.gate.switch(fan).get() is not True:
                raise DangerousError(f"{fan}: fan command was not confirmed")
        history = {h: [] for h in STAGES}
        while True:
            ctx.check_cancel()
            ctx.gate.watchdog_tick()
            readings = stage_readings(ctx.gate)
            now = ctx.clock.now()
            ctx.store.emit("experiment.cooling", {"state": "cooling", "readings": readings,
                "below_c": FINAL_BELOW_C, "deadline": deadline}, run_id=ctx.run_id)
            if now > deadline:
                raise NeedsHumanError("Stage cooldown deadline expired; experiment remains incomplete", readings=readings)
            if stages_cold(readings):
                break
            for heater, reading in readings.items():
                temp = reading["temperature_c"]
                if reading["target_c"] is not None:
                    raise DangerousError(f"{heater}: heating target returned during final cooldown")
                if temp < FINAL_BELOW_C:
                    continue
                history[heater].append((now, temp))
                old = [t for ts, t in history[heater] if now - ts >= cfg["stuck_window_s"]]
                if old and abs(temp - old[-1]) < cfg["stuck_delta_c"]:
                    raise DangerousError(f"{heater}: cooling feedback stalled; sensor or airflow needs inspection",
                                         readings=readings)
            ctx.clock.sleep(min(cfg["poll_s"], max(0.001, deadline - now)))
        failed = []
        for fan in STAGES.values():
            try:
                switch = ctx.gate.switch(fan)
                switch.set(False)
                if switch.get() is not False:
                    failed.append(fan)
            except Exception:
                failed.append(fan)
        if failed:
            raise NeedsHumanError("Stage fan shutdown was not confirmed", failed_devices=failed)

    def verify(self, ctx):
        readings = stage_readings(ctx.gate)
        ok = stages_finished(readings)
        ctx.store.emit("experiment.cooling", {"state": "verified_cold" if ok else "unverified",
            "readings": readings, "below_c": FINAL_BELOW_C}, run_id=ctx.run_id)
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR, readings, self.expected(ctx),
                           message="Both stages are strictly below 40 C" if ok else "Stage shutdown/cooling is unverified")

    def probe_state(self, ctx):
        # Re-enter bounded cooling and take new readings after any interruption.
        return ProbeResult.NOT_DONE

    def safe_state(self, ctx):
        # Leave fan commands unchanged here; runtime safety policy may override them.
        _disable_stage_heaters(ctx.gate)


@register_flow
class StageCooling(Flow):
    name = "cool_experiment_stages"
    description = "Final experiment shutdown: measured exfoliation and search/stacking stages strictly below 40 C"
    Params = CoolingParams
    devices = STAGE_DEVICES
    restore_heaters_on_resume = False

    def build_steps(self, params):
        return [CoolExperimentStages()]
