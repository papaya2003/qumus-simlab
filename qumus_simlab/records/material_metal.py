"""Typed material/metal adapter boundary and independently checked workflow evidence.

A software fixture may exercise the positive job pipeline. Its receipt is never
chemical evidence and cannot satisfy the original experiment's final acceptance.
Physical adapters must use FlowContext.gate for devices and need a separately
reviewed transfer protocol and an explicitly defined experimental observable.
"""
from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from qumus_simlab.records.reactor_requirements import (
    ReactorRequirements,
    input_errors,
    requirements_record,
)

FIXTURE_PROTOCOL = "synthetic-material-metal-workflow-v1"


class MaterialMetalReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol_id: str
    scope: Literal["synthetic_workflow"]
    project_id: str
    requirements_event_id: int
    chip_id: str
    preparation_job_id: str
    top_first_flake_ids: list[str] = Field(min_length=2, max_length=2)
    layer_order: list[str] = Field(min_length=4, max_length=4)
    stamp_id: str
    stamp_film: Literal["PC", "mica"]
    source_chip_ids: dict[str, str]
    metal_coating: dict
    images: list[str] = Field(min_length=1)
    fixture_instruction_id: str
    physical_motion_verified: Literal[False] = False
    chemistry_verified: Literal[False] = False


class MaterialMetalAdapter(Protocol):
    """A registered adapter performs/observes each action; flows own ledger writes.

    Implementations receive the checked FlowContext, including SafetyGate and
    Clock. A physical adapter must not access a raw device or simulator object.
    Each method returns observed state, never a verbal success assertion.
    """
    def apply(self, ctx, action: str, payload: dict) -> dict: ...
    def observe(self, ctx, chip_id: str) -> dict: ...


def adapter_for(store, protocol_id: str, instruction_id: str, project_id: str) -> MaterialMetalAdapter:
    if protocol_id != FIXTURE_PROTOCOL:
        raise ValueError("No registered material-metal adapter for this protocol. Supply a reviewed transfer protocol and observable definition.")
    if store.bench != "sim":
        raise ValueError("Synthetic workflow adapters are unavailable on physical benches.")
    instruction = store.get_instruction(instruction_id)
    if not instruction or instruction["source"] != "ui" or instruction["project_id"] != project_id:
        raise ValueError("Cite this project's explicit instruction authorizing the synthetic workflow fixture.")
    # The exact marker prevents the orchestrator from silently choosing a
    # demonstration when the person requested an experiment.
    if FIXTURE_PROTOCOL not in instruction["text"]:
        raise ValueError(f"The person's instruction must explicitly authorize {FIXTURE_PROTOCOL}; normal experiment requests do not authorize fixtures.")
    from qumus_simlab.testbench.material_metal_fixture import SyntheticMaterialMetalAdapter

    return SyntheticMaterialMetalAdapter()


def preparation_errors(store, project_id: str, chip_id: str) -> list[str]:
    if (store.get_sample(chip_id) or {}).get("props", {}).get("material_metal_preparation", {}).get("scope") == "device_simulation":
        from qumus_simlab.records.material_metal_simulation import device_preparation_errors

        return device_preparation_errors(store, project_id, chip_id)
    record = requirements_record(store, project_id)
    if not record:
        return ["Established experiment requirements are missing."]
    chip = store.get_sample(chip_id) or {}
    raw = chip.get("props", {}).get("material_metal_preparation") or {}
    try:
        receipt = MaterialMetalReceipt.model_validate(raw)
    except ValueError as exc:
        return [f"Invalid material-metal preparation receipt: {exc}"]
    req = ReactorRequirements.model_validate(record["requirements"])
    errors = input_errors(store, req, receipt.top_first_flake_ids, carrier_id=chip_id)
    if receipt.project_id != project_id or receipt.chip_id != chip_id or receipt.requirements_event_id != record["event_id"]:
        errors.append("Preparation belongs to another project, carrier or requirement revision.")
    if receipt.layer_order != ["hBN", req.reactive_material, req.metal, "silicon"]:
        errors.append("Preparation layer order does not match established requirements.")
    if receipt.metal_coating != chip.get("props", {}).get("metal_coating"):
        errors.append("Metal-coating provenance changed after preparation.")
    run = store.get_run(receipt.preparation_job_id)
    if not run or run["project_id"] != project_id or run["flow"] != "prepare_material_metal" or run["status"] != "succeeded" or run["params"].get("target_chip_id") != chip_id:
        errors.append("Preparation needs its successful material-metal job in this project.")
    event = store._x("SELECT payload_json FROM event WHERE kind='reactor.prepared' AND run_id=? ORDER BY id DESC LIMIT 1",
                     (receipt.preparation_job_id,)).fetchone()
    if not event or json.loads(event["payload_json"]).get("material_metal_preparation") != raw:
        errors.append("Preparation receipt lacks matching job evidence.")
    try:
        adapter_for(store, receipt.protocol_id, receipt.fixture_instruction_id, project_id)
    except ValueError as exc:
        errors.append(str(exc))
    fixture = store._x("SELECT payload_json FROM event WHERE kind='material_metal.fixture' AND run_id=? ORDER BY id DESC LIMIT 1",
                       (receipt.preparation_job_id,)).fetchone()
    state = json.loads(fixture["payload_json"])["state"] if fixture else {}
    if not state.get("released") or state.get("picked_ids") != receipt.top_first_flake_ids or state.get("layer_order") != receipt.layer_order:
        errors.append("The adapter did not observe the ordered fixture pickup/release sequence.")
    for fid in receipt.top_first_flake_ids:
        sample = store.get_sample(fid) or {}
        if sample.get("location") != receipt.source_chip_ids.get(fid) or sample.get("status") not in ("catalogued", "selected"):
            errors.append(f"{fid} no longer matches the source of this abstract fixture; physical sample locations are never rewritten by it.")
    return errors


def workflow_result_errors(store, project_id: str, result: dict) -> list[str]:
    chip_id, job_id = result.get("chip_id", ""), result.get("evidence_job_id", "")
    errors = preparation_errors(store, project_id, chip_id)
    run = store.get_run(job_id)
    if not run or run["project_id"] != project_id or run["flow"] != "reactor_material_metal" or run["status"] != "succeeded" or run["params"].get("chip_id") != chip_id:
        errors.append("Workflow result needs a successful matching Reactor job.")
    project = store.get_project(project_id) or {}
    goal = project.get("state", {}).get("reactor_goal")
    chip = store.get_sample(chip_id) or {}
    observation = chip.get("props", {}).get("material_metal_observation")
    if result.get("goal") != goal or result.get("observation") != observation:
        errors.append("Result must retain the current goal and unmodified observation.")
    event = store._x("SELECT payload_json FROM event WHERE kind='material_metal.observed' AND run_id=? ORDER BY id DESC LIMIT 1", (job_id,)).fetchone()
    if not event or json.loads(event["payload_json"]).get("observation") != observation:
        errors.append("Observation lacks matching Reactor job evidence.")
    if not observation or observation.get("job_id") != job_id or observation.get("observable") != "workflow_sequence_receipt" or observation.get("scope") != "synthetic_workflow" or observation.get("chemistry_verified") is not False:
        errors.append("Only an explicitly nonchemical synthetic workflow receipt is supported by this adapter.")
    if result.get("scope") != "synthetic_workflow" or result.get("experiment_accepted") is not False or result.get("workflow_accepted") is not True:
        errors.append("Software workflow evidence cannot be promoted to experimental acceptance.")
    receipt = chip.get("props", {}).get("material_metal_preparation") or {}
    if observation and any(observation.get(key) != receipt.get(key) for key in (
            "protocol_id", "requirements_event_id", "chip_id", "preparation_job_id", "top_first_flake_ids", "layer_order")):
        errors.append("Observation does not retain the exact preparation identity and ordered layers.")
    if observation and (observation.get("product_identity") is not None or observation.get("reaction_fraction") is not None):
        errors.append("An abstract workflow receipt cannot infer product chemistry or conversion.")
    return errors
