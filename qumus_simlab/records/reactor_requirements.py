"""Established experiment requirements, independent of mutable Task Plan state.

No kinetics or product chemistry is inferred here. The historical two-flake
model cannot certify a material/metal experiment or consume its protective hBN.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from qumus_simlab.core.materials import normalize_material_code


class ReactorRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reactive_material: str | None = Field(None, min_length=1)
    metal: str | None = Field(None, min_length=1)
    protective_material: Literal["hBN"] = "hBN"
    protective_role: Literal["nonreactive_pickup_and_protection"] = "nonreactive_pickup_and_protection"
    substrate: Literal["silicon"] = "silicon"
    assembly_order: Literal["hBN/reactive_material/metal/silicon"] = "hBN/reactive_material/metal/silicon"
    save_images: Literal[True] = True
    retain_source_provenance: Literal[True] = True
    reaction_kind: Literal["material_metal"] = "material_metal"


def requirements_record(store: Any, project_id: str | None) -> dict | None:
    if not project_id:
        return None
    row = store._x("SELECT id,payload_json FROM event WHERE kind='experiment.requirements' AND "
        "json_extract(payload_json,'$.project_id')=? ORDER BY id DESC LIMIT 1", (project_id,)).fetchone()
    return {**json.loads(row["payload_json"]), "event_id": row["id"]} if row else None


def established_requirements(store: Any, project_id: str | None) -> ReactorRequirements | None:
    record = requirements_record(store, project_id)
    return ReactorRequirements.model_validate(record["requirements"]) if record else None


def missing_requirements(req: ReactorRequirements) -> list[str]:
    return [key for key in ("reactive_material", "metal") if not getattr(req, key)]


def norm(value):
    return normalize_material_code(value) if value else None


def plan_coverage_errors(store: Any, project_id: str, plan: list[dict]) -> list[str]:
    req = established_requirements(store, project_id)
    if not req:
        return []
    live = [s for s in plan if s.get("status") != "skipped"]
    errors = []
    materials = [req.protective_material, req.reactive_material]
    for material in filter(None, materials):
        available = any(norm(f["props"].get("material")) == norm(material) and f["props"].get("images")
            and f["status"] in ("catalogued", "selected", "stacked", "reactor_consumed")
            for f in store.list_samples(kind="flake"))
        for op in (() if available else ("exfoliate_chips", "search_chip")):
            if not any(s.get("owner") == "hands" and s.get("operation") == op
                       and norm(material) in [norm(m) for m in s.get("materials", [])] for s in live):
                errors.append(f"Preserve {material} {op} or cite existing photographed input provenance.")
    stages = [("picker", "screen"), ("picker", "confirm"),
              ("hands", "prepare_material_metal"), ("reactor", "reactor_material_metal")]
    if any(step.get("operation") == "assess_reactor_inputs" for step in live):
        stages.insert(1, ("hands", "assess_reactor_inputs"))
    indices = []
    for owner, operation in stages:
        matches = [i for i, s in enumerate(live) if s.get("owner") == owner and
                   (s.get("experiment_stage") if owner == "picker" else s.get("operation")) == operation]
        if not matches:
            errors.append(f"Preserve {owner} {operation} for the established experiment.")
        else:
            indices.append(matches[0])
    if indices != sorted(indices):
        errors.append("Use preparation/search -> Picker screen/confirm -> Hands assembly -> Reactor reaction; local reacquisition is conditional.")
    screen = next((i for i, step in enumerate(live) if step.get("owner") == "picker"), len(live))
    for material in filter(None, materials):
        available = any(norm(f["props"].get("material")) == norm(material) and f["props"].get("images")
            and f["status"] in ("catalogued", "selected", "stacked", "reactor_consumed")
            for f in store.list_samples(kind="flake"))
        if not available:
            for op in ("exfoliate_chips", "search_chip"):
                if not any(step.get("operation") == op and norm(material) in [norm(m) for m in step.get("materials", [])]
                           for step in live[:screen]):
                    errors.append(f"Put missing {material} {op} before the first Picker screening.")
    for s in live:
        if s.get("operation") in ("prepare_reactor_pair", "prepare_material_metal") and req.reactive_material and (
                [norm(m) for m in s.get("materials", [])] != ["hBN", norm(req.reactive_material)]):
            errors.append("Assembly must retain top-first hBN then the specified reactive material.")
    return errors


def input_errors(store: Any, req: ReactorRequirements, flake_ids: list[str], *, carrier_id: str | None = None,
                 require_assessment: bool = False) -> list[str]:
    errors = [f"Missing established {key}; clarify with the user." for key in missing_requirements(req)]
    flakes = [store.get_sample(fid) for fid in flake_ids]
    if len(flakes) != 2 or len(set(flake_ids)) != 2 or any(not f or f["kind"] != "flake" for f in flakes):
        return [*errors, "Choose exactly two distinct catalogued flakes, hBN first then the reactive material."]
    if [norm(f["props"].get("material")) for f in flakes] != ["hBN", norm(req.reactive_material)]:
        errors.append("Actual input materials/order do not match hBN then the established reactive material.")
    for f in flakes:
        if not f["props"].get("images"):
            errors.append(f"{f['id']} lacks saved source images.")
        source = (f["props"].get("material_placement") or {}).get("flake", {}).get("source_chip_id")
        source = source or f["props"].get("chip_id") or f["location"]
        if not store.get_sample(source) or source == carrier_id:
            errors.append(f"{f['id']} lacks independent source-chip provenance.")
        if require_assessment:
            from qumus_simlab.records.materials import pickup_admission_error

            if error := pickup_admission_error(store, f):
                errors.append(error)
    if require_assessment:
        from qumus_simlab.records.geometry import pickup_coverage_errors

        errors += pickup_coverage_errors(store, flake_ids)
    if carrier_id:
        chip = store.get_sample(carrier_id)
        coating = (chip or {}).get("props", {}).get("metal_coating") or {}
        if not chip or not chip["props"].get("reactor_substrate"):
            errors.append("Choose a dedicated reactor carrier.")
        if coating.get("metal") != req.metal or coating.get("substrate") != "silicon":
            errors.append("Carrier must have the specified predeposited metal on silicon.")
        if not coating.get("preparation_id") or not coating.get("prepared_by") or coating.get("source") not in (
                "operator_report", "synthetic_fixture"):
            errors.append("Metal coating needs human preparation provenance (or a labelled synthetic fixture).")
        if store.bench != "sim" and coating.get("source") == "synthetic_fixture":
            errors.append("Synthetic coating provenance cannot establish physical preparation.")
    return errors


class ReactorTaskObjective(BaseModel):
    """Structured part of the Qumus -> Reactor task dialogue; no goal tool."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    chip_id: str
    requirements_event_id: int
    objective: str = Field(min_length=1)
    final_disposition: Literal["cold_storage", "observation_hold"]


def capture_reactor_objective(store: Any, project_id: str, task: str, objective: ReactorTaskObjective,
                              step: dict) -> dict:
    record = requirements_record(store, project_id)
    if not record or objective.requirements_event_id != record["event_id"]:
        raise ValueError("Reaction task must cite the current established requirements event.")
    req = ReactorRequirements.model_validate(record["requirements"])
    chip = store.get_sample(objective.chip_id)
    manifest = (chip or {}).get("props", {}).get("material_metal_preparation") or {}
    if manifest.get("scope") == "device_simulation":
        from qumus_simlab.records.material_metal_simulation import device_preparation_errors

        errors = device_preparation_errors(store, project_id, objective.chip_id)
        disposition = step.get("reactor_outcome") or {}
        if (disposition.get("chip_id") != objective.chip_id or disposition.get("disposition") != objective.final_disposition
                or disposition.get("temperature_target_c") is not None or disposition.get("flow_target_sccm") != 0):
            errors.append("Declare a cooled simulation end state with targets disabled, in storage or an observation hold")
        if step.get("operation") != "reactor_material_metal":
            errors.append("Use the material-metal device simulation operation for this preparation")
        if errors:
            raise ValueError("; ".join(errors))
        goal = {**objective.model_dump(), "scope": "material_metal", "evidence_scope": "device_simulation",
            "task_dialogue": task, "requirements": record["requirements"], "preparation": manifest,
            "metal_coating": chip["props"]["metal_coating"], "source": "reactor_task_dialogue",
            "reactor_outcome": disposition, "simulation_criterion": {"observable": "qualitative_interface_progress",
                "minimum_progress": manifest["model_assumptions"]["minimum_progress"],
                "provenance": "uncalibrated_software_assumptions", "protective_layer_preserved": True},
            "chemistry_verified": False, "physical_motion_verified": False}
        store.update_project_state(project_id, reactor_goal=goal, reactor_result=None,
            reactor_workflow_result=None, reactor_simulation_result=None)
        store.emit("reactor.goal", {"project_id": project_id, "goal": goal}, agent="orchestrator")
        return goal
    if manifest.get("scope") == "synthetic_workflow":
        from qumus_simlab.records.material_metal import preparation_errors

        errors = preparation_errors(store, project_id, objective.chip_id)
        disposition = step.get("reactor_outcome") or {}
        if disposition.get("chip_id") != objective.chip_id or disposition.get("disposition") != objective.final_disposition:
            errors.append("Task objective and final process disposition disagree.")
        if step.get("operation") != "reactor_material_metal" or objective.final_disposition != "cold_storage":
            errors.append("The abstract fixture supports only its registered workflow observation and unchanged cold storage.")
        if errors:
            raise ValueError("; ".join(errors))
        goal = {**objective.model_dump(), "scope": "material_metal", "evidence_scope": "synthetic_workflow",
            "task_dialogue": task, "requirements": record["requirements"], "preparation": manifest,
            "metal_coating": chip["props"]["metal_coating"], "source": "reactor_task_dialogue",
            "chemistry_verified": False, "physical_motion_verified": False}
        store.update_project_state(project_id, reactor_goal=goal, reactor_result=None, reactor_workflow_result=None)
        store.emit("reactor.goal", {"project_id": project_id, "goal": goal}, agent="orchestrator")
        return goal
    errors = input_errors(store, req, manifest.get("top_first_flake_ids", []), carrier_id=objective.chip_id)
    if manifest.get("requirements_event_id") != record["event_id"] or not manifest.get("preparation_job_id"):
        errors.append("Verified assembly must be bound to the established requirements.")
    run = store.get_run(manifest.get("preparation_job_id", ""))
    if not run or run["status"] != "succeeded" or run["project_id"] != project_id or run["flow"] != "prepare_reactor_pair" or run["params"].get("target_chip_id") != objective.chip_id:
        errors.append("A successful assembly job in this project is required.")
    for fid in manifest.get("top_first_flake_ids", []):
        sample = store.get_sample(fid) or {}
        if sample.get("location") != objective.chip_id or sample.get("status") != "stacked":
            errors.append(f"Prepared input {fid} is not actually stacked on the carrier.")
    evidence = store._x("SELECT payload_json FROM event WHERE kind='reactor.prepared' AND run_id=? ORDER BY id DESC LIMIT 1",
                        (manifest.get("preparation_job_id", ""),)).fetchone()
    if not evidence or json.loads(evidence["payload_json"]).get("material_metal_preparation") != manifest:
        errors.append("Preparation manifest lacks matching successful job evidence.")
    if manifest.get("layer_order") != ["hBN", req.reactive_material, req.metal, "silicon"]:
        errors.append("Preparation does not preserve the required final layer order.")
    disposition = step.get("reactor_outcome") or {}
    if disposition.get("chip_id") != objective.chip_id or disposition.get("disposition") != objective.final_disposition:
        errors.append("Task objective and final process disposition disagree.")
    if errors:
        raise ValueError("; ".join(errors))
    goal = {**objective.model_dump(), "scope": "material_metal", "task_dialogue": task,
            "requirements": record["requirements"], "preparation": manifest,
            "metal_coating": chip["props"]["metal_coating"], "source": "reactor_task_dialogue"}
    store.update_project_state(project_id, reactor_goal=goal, reactor_result=None)
    store.emit("reactor.goal", {"project_id": project_id, "goal": goal}, agent="orchestrator")
    return goal


def acceptance_errors(store: Any, project_id: str) -> list[str]:
    record = requirements_record(store, project_id)
    if not record:
        return []
    state = (store.get_project(project_id) or {}).get("state", {})
    if state.get("reactor_simulation_result") and store.bench == "sim":
        from qumus_simlab.records.material_metal_simulation import simulation_result_errors

        return simulation_result_errors(store, project_id, state["reactor_simulation_result"])
    result = state.get("reactor_result") or {}
    if result.get("scope") != "material_metal" or result.get("requirements_event_id") != record["event_id"]:
        return ["No material-metal result independently satisfies the established requirements; old two-flake results are not evidence."]
    # The workflow adapter has no validated material/metal chemical observable.
    # A state write or all-done plan must never bypass this boundary.
    return ["Material-metal reaction acceptance requires a validated chemical-observation adapter; a software workflow receipt cannot verify chemical outcome."]
