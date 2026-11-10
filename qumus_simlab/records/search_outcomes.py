"""Search outcome roles and narrowly evidenced repairs of old model-written IDs."""
from __future__ import annotations

from copy import deepcopy

from qumus_simlab.core.materials import normalize_material_code as material_code

SEARCH_FLOWS = frozenset({"search_chip", "revisit_chip"})


def search_output_chips(store, step: dict) -> set[str]:
    """Resolve declared chip/flake outputs without interpreting a source carrier as a chip."""
    chips = set()
    for condition in step.get("sample_outcomes") or []:
        sample = store.get_sample(condition["sample_id"]) or {}
        if sample.get("kind") == "chip":
            chips.add(sample["id"])
        elif sample.get("kind") == "flake":
            source = sample.get("props", {}).get("chip_id") or sample.get("location")
            if (store.get_sample(source) or {}).get("kind") == "chip":
                chips.add(source)
    return chips


def invalid_search_outcomes(store, step: dict) -> list[dict]:
    if step.get("operation") not in SEARCH_FLOWS:
        return []
    errors = []
    conditions = step.get("sample_outcomes") or []
    targets = search_output_chips(store, step)
    bound = (step.get("operation_args") or {}).get("chip_id")
    for condition in conditions:
        sample = store.get_sample(condition["sample_id"]) or {}
        kind, status = sample.get("kind"), condition.get("status")
        # A source carrier's storage condition may accompany the actual outputs,
        # but cannot stand in for the searched chip or acquire its status.
        invalid = (kind not in ("chip", "flake", "carrier")
                   or kind == "carrier" and (status is not None or not targets)
                   or kind == "chip" and status not in (None, "searched")
                   or kind == "flake" and not targets)
        if invalid:
            errors.append({"step_id": step.get("id"), "operation": step["operation"],
                "sample_id": condition["sample_id"], "role": kind or "unknown",
                "declared_status": status, "allowed_status": "searched" if kind == "chip" else None,
                "guidance": "Search outcomes refer to the actual receiving chip (status searched), or its catalogued flakes. "
                    "A raw-material carrier is not the search target. Leave future outputs unbound until a real chip is known; "
                    "then refine before delegation. Keep source-carrier storage as a separate condition if required."})
    if bound and (targets != {bound} or (store.get_sample(bound) or {}).get("kind") != "chip"):
        errors.append({"step_id": step.get("id"), "operation": step["operation"], "sample_id": bound,
            "guidance": "The search chip_id and declared output chip must match; do not execute against a different sample."})
    return errors


def search_target_error(store, step: dict, chip_id: str | None) -> str | None:
    if step.get("operation") not in SEARCH_FLOWS:
        return None
    if invalid_search_outcomes(store, step) or search_output_chips(store, step) != {chip_id}:
        return "The search job must target the one actual chip bound by this step's completion conditions."
    return None


def correct_search_source_identity(store, project_id: str, previous: dict, current: dict) -> list[dict]:
    """Repair one impossible carrier/searched binding, never a physical retarget.

    The receiving chip must be the unique output of a successful exfoliation in
    this project, with matching material and ledger provenance. The correction
    remains pending; only the original linked job and current outcomes can close it.
    """
    from qumus_simlab.records.plan_completion import (
        completion_contract,
        human_sample_binding_conflict,
        linked_physical_jobs,
        preserves_obligation,
    )

    if (previous.get("operation") not in SEARCH_FLOWS or previous.get("owner") != "hands"
            or previous.get("status") not in ("pending", "failed") or current.get("status") != "pending"
            or previous.get("id") != current.get("id")):
        return []
    old = completion_contract(previous)
    candidates = [c for c in old["sample_outcomes"] if c.get("status") == "searched"
                  and (store.get_sample(c["sample_id"]) or {}).get("kind") == "carrier"]
    if len(candidates) != 1:
        return []
    condition = candidates[0]
    # A fixed position or an explicit physical binding is not an identity typo.
    if condition.get("location") is not None or old["operation_args"].get("chip_id"):
        return []
    source_id = condition["sample_id"]
    source = store.get_sample(source_id)
    material = material_code(source["props"].get("material"))
    if not material or [material_code(m) for m in old["materials"]] != [material]:
        return []
    related = [r for r in store.list_runs() if r["project_id"] == project_id and r["flow"] == "exfoliate_chips"
               and r["params"].get("carrier_id") == source_id]
    if any(r["status"] in ("queued", "running", "parked", "paused", "waiting_human") for r in related):
        return []
    runs = [r for r in related if r["status"] == "succeeded"
            and material_code(r["params"].get("material")) == material]
    outputs = [(r, sid) for r in runs for sid in r["params"].get("chip_ids", [])]
    if len(outputs) != 1:
        return []
    run, target_id = outputs[0]
    target = store.get_sample(target_id) or {}
    props = target.get("props", {})
    if (target.get("kind") != "chip" or props.get("from_carrier") != source_id
            or material_code(props.get("material")) != material or props.get("location_uncertain")
            or human_sample_binding_conflict(store, project_id, source_id, target_id)):
        return []
    if target_id in store.reserved_samples():
        return []
    linked = linked_physical_jobs(store, project_id, previous["id"])
    if any(r["flow"] not in SEARCH_FLOWS or r["status"] != "succeeded"
           or r["params"].get("chip_id") != target_id for r in linked):
        return []
    repaired = deepcopy(previous)
    repaired["sample_outcomes"] = [{**c, "sample_id": target_id} if c["sample_id"] == source_id
                                  and c.get("status") == "searched" else c for c in old["sample_outcomes"]]
    if invalid_search_outcomes(store, current) or not preserves_obligation(repaired, current):
        return []
    return [{"step_id": previous["id"], "correction_kind": "source_carrier_to_receiving_chip",
        "source": "successful_project_exfoliation", "contract_version": 1,
        "reason": "The model assigned a chip-search status to its raw-material carrier; the unique receiving chip is evidenced.",
        "original_condition": condition, "corrected_condition": {**condition, "sample_id": target_id},
        "original_contract": old, "execution_evidence": {"job_id": run["id"], "flow": run["flow"],
            "material": material, "source_carrier_id": source_id, "chip_id": target_id,
            "search_job_ids": [r["id"] for r in linked]},
        "next_action": ("Keep pending. Recheck the original successful search job with Hands.wait_job and the delegation check; do not repeat search."
                        if linked else "Keep pending. Delegate search on the evidenced receiving chip, then verify its declared outcomes.")}]
