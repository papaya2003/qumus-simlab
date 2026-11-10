"""Independent acceptance contracts for a qualitative device simulation.

Numerical values are software assumptions, never literature-calibrated Gr/Pd
kinetics or evidence for a real experiment. The abstract receipt stays separate.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SIM_PROTOCOL = "qualitative-material-metal-v1"


class TransferAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pickup_c: float = Field(90, ge=40, le=130)
    release_c: float = Field(170, ge=100, le=180)
    release_band_c: float = Field(3, gt=0, le=5)
    pickup_probability: float = Field(0.92, ge=0, le=1)
    release_probability: float = Field(0.9, ge=0, le=1)
    support_retained_on_stamp: bool = False


class MaterialMetalSimulation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol_id: Literal["qualitative-material-metal-v1"] = SIM_PROTOCOL
    provenance: Literal["uncalibrated_software_assumptions"] = "uncalibrated_software_assumptions"
    PC: TransferAssumption = TransferAssumption()
    mica: TransferAssumption = TransferAssumption(pickup_c=80, release_c=150, support_retained_on_stamp=True)
    temperature_c: float = Field(300, ge=22, le=950)
    flow_sccm: float = Field(50, ge=0, le=400)
    duration_s: float = Field(180, gt=0, le=900)
    activation_temperature_c: float = Field(250, ge=22, le=950)
    activation_flow_sccm: float = Field(10, ge=0, le=400)
    characteristic_time_s: float = Field(60, gt=0, le=10000)
    success_probability: float = Field(0.9, ge=0, le=1)
    unsuccessful_max_progress: float = Field(0.25, ge=0, lt=1)
    minimum_progress: float = Field(0.8, gt=0, le=1)
    observation_available: bool = True


def simulation_profile(bench: dict) -> MaterialMetalSimulation:
    return MaterialMetalSimulation.model_validate(bench.get("reactor_module", {}).get("material_metal", {}))


def device_preparation_errors(store, project_id: str, chip_id: str) -> list[str]:
    from qumus_simlab.records.reactor_requirements import (
        ReactorRequirements,
        input_errors,
        requirements_record,
    )

    record = requirements_record(store, project_id)
    chip = store.get_sample(chip_id) or {}
    receipt = chip.get("props", {}).get("material_metal_preparation") or {}
    if not record or receipt.get("scope") != "device_simulation" or store.bench != "sim":
        return ["A device-simulation preparation and current requirements are required."]
    req = ReactorRequirements.model_validate(record["requirements"])
    ids = receipt.get("top_first_flake_ids", [])
    errors = input_errors(store, req, ids, carrier_id=chip_id)
    if any(receipt.get(k) != v for k, v in {"project_id": project_id, "chip_id": chip_id,
            "requirements_event_id": record["event_id"], "protocol_id": SIM_PROTOCOL,
            "layer_order": ["hBN", req.reactive_material, req.metal, "silicon"],
            "metal_coating": chip.get("props", {}).get("metal_coating")}.items()):
        errors.append("Device preparation identity, coating or ordered layers changed.")
    job = store.get_run(receipt.get("preparation_job_id", ""))
    if not job or job["project_id"] != project_id or job["flow"] != "prepare_material_metal" or job["status"] != "succeeded" or job["params"].get("target_chip_id") != chip_id:
        errors.append("A successful matching guarded preparation job is required.")
    event = store._x("SELECT payload_json FROM event WHERE kind='reactor.prepared' AND run_id=? ORDER BY id DESC LIMIT 1", (receipt.get("preparation_job_id", ""),)).fetchone()
    if not event or json.loads(event["payload_json"]).get("material_metal_preparation") != receipt:
        errors.append("Preparation is not backed by its optical job evidence.")
    if not receipt.get("release_images") or receipt.get("simulated_motion_verified") is not True or receipt.get("chemistry_verified") is not False:
        errors.append("Device preparation needs explicit simulation scope and actual release images.")
    try:
        MaterialMetalSimulation.model_validate(receipt.get("model_assumptions", {}))
    except ValueError:
        errors.append("Invalid recorded simulation assumptions.")
    for fid in ids:
        row = store.get_sample(fid) or {}
        if row.get("location") != chip_id or row.get("status") != "stacked" or row.get("props", {}).get("reactor_consumed_by"):
            errors.append(f"{fid} must remain present on the prepared carrier, without consumption.")
        if row.get("props", {}).get("material_placement") != receipt.get("placements", {}).get(fid):
            errors.append(f"{fid} geometry/provenance differs from its observed preparation.")
    if {s["id"] for s in store.list_samples(kind="flake", location=chip_id)} != set(ids):
        errors.append("The carrier contains additional or missing layers.")
    return errors


def simulation_result_errors(store, project_id: str, result: dict, *, bench: dict | None = None) -> list[str]:
    from qumus_simlab.records.plan_completion import unmet_reactor_outcome

    chip_id, job_id = result.get("chip_id", ""), result.get("evidence_job_id", "")
    errors = device_preparation_errors(store, project_id, chip_id)
    chip = store.get_sample(chip_id) or {}
    project = store.get_project(project_id) or {}
    goal = project.get("state", {}).get("reactor_goal") or {}
    obs = chip.get("props", {}).get("material_metal_observation") or {}
    job = store.get_run(job_id)
    if not job or job["status"] != "succeeded" or job["flow"] != "reactor_material_metal" or job["project_id"] != project_id or job["params"].get("chip_id") != chip_id:
        errors.append("Simulation acceptance requires its successful matching Reactor job.")
    event = store._x("SELECT payload_json FROM event WHERE kind='material_metal.observed' AND run_id=? ORDER BY id DESC LIMIT 1", (job_id,)).fetchone()
    if not event or json.loads(event["payload_json"]).get("observation") != obs or obs.get("job_id") != job_id:
        errors.append("Current observation is not backed by the Reactor job.")
    prep = chip.get("props", {}).get("material_metal_preparation") or {}
    if job and job["params"].get("goal") != goal:
        errors.append("The goal differs from the objective frozen at Reactor job admission.")
    if result.get("goal") != goal or result.get("observation") != obs or goal.get("preparation") != prep or goal.get("evidence_scope") != "device_simulation":
        errors.append("Simulation result must retain its current task goal, preparation and observation.")
    if any(obs.get(k) != prep.get(k) for k in ("protocol_id", "preparation_job_id", "requirements_event_id", "layer_order", "top_first_flake_ids", "model_assumptions")):
        errors.append("Observation provenance or assumptions differ from the prepared stack.")
    progress = obs.get("qualitative_progress")
    minimum = goal.get("simulation_criterion", {}).get("minimum_progress", 1)
    if minimum != prep.get("model_assumptions", {}).get("minimum_progress"):
        errors.append("Simulation criterion changed from the recorded model assumptions.")
    if (not obs.get("visible") or not obs.get("fully_visible") or obs.get("status") != "progressing"
            or isinstance(progress, bool) or not isinstance(progress, (float, int)) or not minimum <= progress <= 1):
        errors.append("The observed qualitative interface progress has not met the frozen simulation goal.")
    if not obs.get("images") or obs.get("product_identity") is not None or obs.get("chemistry_verified") is not False:
        errors.append("Simulation needs an observed image without a fabricated chemical product claim.")
    if (result.get("scope") != "device_simulation" or result.get("simulation_accepted") is not True
            or result.get("experiment_accepted") is not False or result.get("chemistry_verified") is not False
            or result.get("physical_motion_verified") is not False or result.get("simulated_motion_verified") is not True):
        errors.append("Simulation success must remain separate from real experiment acceptance.")
    if chip.get("props", {}).get("reactor_state_uncertain"):
        errors.append("Interrupted exposure history remains uncertain.")
    if bench is not None:
        errors.extend(unmet_reactor_outcome(store, bench, goal.get("reactor_outcome"), project_id=project_id))
    # Final project acceptance also rechecks terminal evidence without relying
    # on a mutable assertion in the accepted result.
    row = store._x("SELECT run_id,payload_json,id FROM event WHERE kind='reactor.process' AND json_extract(payload_json,'$.chip_id')=? ORDER BY id DESC LIMIT 1", (chip_id,)).fetchone()
    terminal = json.loads(row["payload_json"]) if row else {}
    if not row or row["run_id"] != job_id or terminal != result.get("terminal_evidence"):
        errors.append("Final simulation result lacks matching current terminal evidence.")
    if row and store._x("SELECT id FROM event WHERE id>? AND kind='device.command' AND json_extract(payload_json,'$.device') IN ('reactor_heater','reactor_flow','reactor_slot','reactor_insert','arm_reactor') LIMIT 1", (row["id"],)).fetchone():
        errors.append("Simulation terminal readings precede later device commands.")
    if terminal.get("location") != chip.get("location") or terminal.get("state_uncertain"):
        errors.append("Current carrier disposition disagrees with terminal evidence.")
    return errors
