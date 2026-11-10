"""Explicit ledger postconditions for a Task Plan, never inferred from prose."""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SampleOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    location: str | None = Field(None, min_length=1, description="Exact final location. For chip outcomes of "
        "exfoliate_chips/search_chip/revisit_chip, leave null and use in_storage for a generic storage goal: "
        "their dynamic allocation/fallback cannot promise a fixed slot. Preserve explicit user fixed-slot requirements.")
    status: str | None = Field(None, min_length=1)
    in_storage: bool = Field(False, description="Require a matching configured storage slot. A flake/device "
        "is stored through its actual carrier chip; both locations must be known and certain. Do not guess a slot ID.")

    @model_validator(mode="after")
    def has_condition(self):
        if self.location is None and self.status is None and not self.in_storage:
            raise ValueError("a sample outcome needs location, status or in_storage")
        return self


class ReactorOutcome(BaseModel):
    """Final process disposition, distinct from accepting a reaction product."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    chip_id: str = Field(min_length=1)
    disposition: Literal["cold_storage", "observation_hold"]
    temperature_target_c: float | None = Field(description="Explicit heater target, or null for off.")
    flow_target_sccm: float = Field(ge=0)
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def cold_targets_off(self):
        if self.disposition == "cold_storage" and (self.temperature_target_c is not None or self.flow_target_sccm != 0):
            raise ValueError("cold_storage requires heater off and gas target zero")
        if not self.rationale.strip():
            raise ValueError("explain the final process disposition")
        return self


def completion_contract(step: dict) -> dict:
    """Normalize additive defaults when comparing current and legacy obligations."""
    samples = step.get("sample_outcomes") or []
    reactor = step.get("reactor_outcome")
    try:
        samples = [SampleOutcome.model_validate(s).model_dump() for s in samples]
        reactor = ReactorOutcome.model_validate(reactor).model_dump() if reactor else None
    except ValueError:
        pass  # the actual outcome checker refuses invalid stored requirements
    return {"sample_outcomes": samples, "reactor_outcome": reactor,
            "operation": step.get("operation"), "owner": step.get("owner"),
            "materials": step.get("materials") or [], "operation_args": step.get("operation_args") or {}}


def preserves_obligation(previous: dict, current: dict) -> bool:
    """Allow refinement, but not removal/reownership of pending physical work."""
    old, new = completion_contract(previous), completion_contract(current)
    if old["owner"] != new["owner"] or old["operation"] and old["operation"] != new["operation"]:
        return False
    if old["materials"] and old["materials"] != new["materials"]:
        return False
    # These optional material-metal identities use null for an unbound source,
    # just like an absent key. Binding them later adds a constraint; it does not
    # replace a chosen flake. Never generalize this to null heater-off targets.
    unbound_flakes = {"flake_ids", "hbn_flake_id", "reactive_flake_id", "gr_flake_id"} if old["operation"] in {
        "prepare_material_metal", "reactor_material_metal"} else set()
    if any(new["operation_args"].get(k) != v for k, v in old["operation_args"].items()
           if not (v is None and k in unbound_flakes)):
        return False
    for wanted in old["sample_outcomes"]:
        if not any(all(actual.get(k) == v for k, v in wanted.items() if v is not None and v is not False)
                   for actual in new["sample_outcomes"]):
            return False
    reactor = old["reactor_outcome"]
    if reactor and any((new["reactor_outcome"] or {}).get(k) != v for k, v in reactor.items() if k != "rationale"):
        return False
    return True


# Fixed outputs only: moving/observing a carrier does not assign a new status.
# Roles matter: preparation writes different statuses to the carrier and flakes.
FLOW_SAMPLE_STATUSES = {
    "exfoliate_chips": {"receiving_chip": "exfoliated"},
    "prepare_reactor_pair": {"carrier": "reactor_pair_prepared", "input_flakes": "stacked"},
    "reactor_reaction": {"carrier": "reactor_processed"},
    "prepare_material_metal": {"carrier": "material_metal_prepared", "input_flakes": "stacked"},
    # GrowthWindow.on_success records processing; both layers remain unconsumed.
    # The observed progress/result and terminal disposition are still required.
    "reactor_material_metal": {"carrier": "reactor_processed", "input_flakes": "stacked"},
}
STATUS_PRESERVING_STORAGE_FLOWS = frozenset({
    "inspect_reactor_product", "unload_reactor_chip", "verify_reactor_state",
})


def flow_storage_contracts() -> dict[str, dict]:
    """Registered storage semantics, not an allocation, reservation or observation."""
    from qumus_simlab.flows.library.exfoliation import STORE_BOARD

    search = {"sample_kind": "chip", "mode": "home_at_submission", "resolved_parameter": "home_slot",
              "destination": "Chip location when this search/revisit is submitted, after earlier work finishes.",
              "fallback": "Another free chip slot in that home slot's storage station; updates home_slot.",
              "exact_location_supported": False}
    return {"exfoliate_chips": {"sample_kind": "chip", "mode": "allocated_at_submission",
                "storage_station": STORE_BOARD, "resolved_parameter": "store_slots",
                "destination": "Free chip slots allocated and reserved at submission, not the chips' original slots.",
                "fallback": "Another free chip slot in the same storage station; updates store_slots.",
                "exact_location_supported": False},
            "search_chip": dict(search), "revisit_chip": dict(search)}


def invalid_flow_sample_locations(store, step: dict) -> list[dict]:
    """Reject unsupported fixed chip destinations without rewriting any goal."""
    operation = step.get("operation")
    if operation is None and step.get("owner") in ("hands", "reactor"):
        return [{"step_id": step["id"], "operation": None, "sample_id": condition["sample_id"],
                 "declared_location": condition["location"], "guidance":
                 "Choose a registered operation before declaring an exact physical destination, so its endpoint "
                 "can be checked before saving. Future drafts may omit outcomes or use status/in_storage. "
                 "Retain any explicit user fixed-slot requirement; do not silently drop it."}
                for condition in step.get("sample_outcomes") or [] if condition.get("location") is not None]
    if operation not in ("exfoliate_chips", "search_chip", "revisit_chip"):
        return []
    contract = flow_storage_contracts()[operation]
    errors = []
    for condition in step.get("sample_outcomes") or []:
        if condition.get("location") is None:
            continue
        sample = store.get_sample(condition["sample_id"])
        if not sample or sample["kind"] != contract["sample_kind"]:
            continue
        guidance = (f"{operation} uses {contract['mode']} with same-station storage fallback; it cannot promise "
            "a fixed chip slot. For a new generic return-to-storage goal, use location=null and in_storage=true, "
            "retaining the required sample/status. Do not copy a chip's pre-exfoliation slot into a future search. "
            "If the person explicitly requires a fixed slot, retain that requirement and report this flow cannot "
            "guarantee it. Do not weaken a saved condition or ask for repeated authorization to fix an unsaved model guess.")
        errors.append({"step_id": step["id"], "operation": operation, "sample_id": condition["sample_id"],
                       "declared_location": condition["location"], "storage_contract": contract, "guidance": guidance})
    return errors


def registered_flow_names() -> list[str]:
    """Load the same registry used by job submission; never infer flow aliases."""
    import qumus_simlab.flows.library  # noqa: F401 - registers built-in flows
    from qumus_simlab.flows.flow import FLOWS

    return sorted(FLOWS)




def flow_sample_role(store, step: dict, sample_id: str) -> str | None:
    sample = store.get_sample(sample_id)
    if not sample:
        return None
    if step.get("operation") == "exfoliate_chips":
        return {"carrier": "source_carrier", "chip": "receiving_chip"}.get(sample["kind"])
    target = (step.get("reactor_outcome") or {}).get("chip_id")
    if sample["kind"] == "chip" and sample["props"].get("reactor_substrate") and target in (None, sample_id):
        return "carrier"
    if sample["kind"] == "flake" and step.get("operation") in (
            "prepare_reactor_pair", "prepare_material_metal", "reactor_material_metal"):
        return "input_flakes"
    return None


def invalid_flow_sample_statuses(store, step: dict) -> list[dict]:
    """Validate declared outputs, never normalize a model's goal to current state."""
    from qumus_simlab.records.search_outcomes import invalid_search_outcomes

    statuses = FLOW_SAMPLE_STATUSES.get(step.get("operation"), {})
    if step.get("operation") in ("prepare_material_metal", "reactor_material_metal"):
        from qumus_simlab.records.material_metal_simulation import SIM_PROTOCOL

        if (step.get("operation_args") or {}).get("protocol_id", SIM_PROTOCOL) != SIM_PROTOCOL:
            # The separately authorized abstract fixture never changes physical
            # sample statuses and cannot satisfy a device preparation outcome.
            statuses = {}
    errors = invalid_search_outcomes(store, step)
    for condition in step.get("sample_outcomes") or []:
        role = flow_sample_role(store, step, condition["sample_id"])
        expected = statuses.get(role)
        if (step.get("operation") == "exfoliate_chips" and role != "receiving_chip"
                and condition.get("status") == "exfoliated"):
            errors.append({"step_id": step["id"], "operation": step["operation"],
                "sample_id": condition["sample_id"], "role": role,
                "declared_status": "exfoliated", "allowed_status": None,
                "guidance": "Exfoliation makes the receiving chip exfoliated, not its raw-material carrier. "
                    "Preserve the source carrier's storage condition without assigning it the chip's status. "
                    "Leave an unknown receiving chip unbound, then bind the chip returned by the successful job."})
        stored_carrier = role == "carrier" and condition.get("status") == "stored"
        if condition.get("status") == "stored" and role is None:
            sample = store.get_sample(condition["sample_id"])
            stored_carrier = bool(sample and sample["kind"] == "chip" and sample["props"].get("reactor_substrate"))
            if stored_carrier:
                role = "carrier"  # A different reactor_outcome target cannot make stored a real status.
        if stored_carrier or expected and condition.get("status") is not None and condition["status"] != expected:
            guidance = ("stored is not a reactor carrier status. Express storage with in_storage: true; "
                        "use the documented output status from get_reactor_setup, or preserve the carrier "
                        "status for transport/inspection. Do not guess a future status."
                        if stored_carrier else f"{step['operation']} leaves {role} status {expected}.")
            errors.append({"step_id": step["id"], "operation": step.get("operation"),
                           "sample_id": condition["sample_id"], "role": role,
                           "declared_status": condition["status"], "allowed_status": expected,
                           "guidance": guidance})
    return errors


def correct_misassigned_flow_statuses(store, project_id: str, previous: dict, current: dict) -> list[dict]:
    """Repair explicit legacy role/location typos, never a real status goal.

    A stored pseudo-status is redundant only when the old condition already
    requires storage and the registered operation preserves status on return.
    After submission require this step's successful carrier-bound job; preparation
    additionally needs its verification event. Completion still needs fresh checks.
    """
    operation = previous.get("operation")
    if operation == "exfoliate_chips":
        return _correct_exfoliation_output(store, project_id, previous, current)
    storage_correction = operation in STATUS_PRESERVING_STORAGE_FLOWS
    if operation != "prepare_reactor_pair" and not storage_correction:
        return []
    repaired = deepcopy(previous)
    corrections = []
    runs = linked_physical_jobs(store, project_id, previous["id"])
    for error in invalid_flow_sample_statuses(store, previous):
        if error["role"] != "carrier":
            return []
        if storage_correction:
            if (previous.get("owner") != "reactor" or error["declared_status"] != "stored"
                    or flow_sample_role(store, previous, error["sample_id"]) != "carrier"):
                return []
            conditions = [s for s in previous["sample_outcomes"]
                          if s["sample_id"] == error["sample_id"] and s.get("status") == "stored"]
            for wanted in conditions:
                if wanted.get("in_storage") is not True or not any(
                    actual.get("status") is None and all(actual.get(k) == v for k, v in wanted.items()
                        if k != "status" and v is not None and v is not False)
                    for actual in current.get("sample_outcomes", [])
                ):
                    return []
        elif error["declared_status"] != FLOW_SAMPLE_STATUSES["prepare_reactor_pair"]["input_flakes"]:
            # A real later reaction status is not a preparation role typo.
            return []
        target_key = "chip_id" if storage_correction else "target_chip_id"
        evidence = None
        for run in runs:
            if (run["flow"] != operation or run["status"] != "succeeded"
                    or run["params"].get(target_key) != error["sample_id"]):
                continue
            if storage_correction:
                evidence = {"job_id": run["id"], "flow": operation, "chip_id": error["sample_id"],
                            "job_status": "succeeded"}
                break
            row = store._x("SELECT id,payload_json FROM event WHERE kind='reactor.prepared' "
                           "AND run_id=? ORDER BY id DESC LIMIT 1", (run["id"],)).fetchone()
            payload = json.loads(row["payload_json"]) if row else {}
            if (payload.get("chip_id") == error["sample_id"] and payload.get("preparation_id") == run["id"]
                    and payload.get("preparation_source") == "stacking_job"
                    and len(payload.get("input_flake_ids", [])) == 2
                    and set(payload["input_flake_ids"]) == set(run["params"].get("flake_ids", []))):
                evidence = {"job_id": run["id"], "event_id": row["id"]}
                break
        submitted = runs or any(run["project_id"] == project_id and run["flow"] == operation
            and run["params"].get(target_key) == error["sample_id"] for run in store.list_runs())
        if submitted and evidence is None:
            return []
        for condition in repaired["sample_outcomes"]:
            if condition["sample_id"] == error["sample_id"] and condition.get("status") == error["declared_status"]:
                condition["status"] = error["allowed_status"]
        corrections.append({**error, "source": "registered_flow_contract", "contract_version": 2 if storage_correction else 1,
                            "correction_kind": "stored_location_redundancy" if storage_correction else "carrier_input_role",
                            "execution_evidence": evidence})
    return corrections if corrections and preserves_obligation(repaired, current) else []


def model_written_contract_events(store, project_id: str, previous: dict) -> list[int]:
    """Use immutable structural events, not mutable latest-revision progress."""
    origins, last = [], None
    for row in store._x("SELECT id,agent,payload_json FROM event WHERE kind='plan.updated' AND "
                        "json_extract(payload_json,'$.project_id')=? ORDER BY id", (project_id,)):
        step = next((s for s in json.loads(row["payload_json"]).get("plan", []) if s.get("id") == previous["id"]), None)
        if step is None:
            if origins:
                return []
            continue
        contract = completion_contract(step)
        if contract != last:
            if row["agent"] != "orchestrator":
                return []
            origins.append(row["id"])
            last = contract
    return origins if last == completion_contract(previous) else []


def human_sample_binding_conflict(store, project_id: str, source_id: str, target_id: str, *, additional_actions: str = "") -> bool:
    """Conservatively refuse to reinterpret a person-named source/target.

    Free text is not enough to prove a named source is merely a model typo.
    Other explicit search targets are also preserved, including Chinese requests.
    """
    samples = {s["id"] for s in store.list_samples(kind="chip")} | {source_id}
    for row in store._x("SELECT text FROM instruction WHERE source='ui' AND kind IN ('message','plan_change') "
                        "AND (project_id=? OR project_id IS NULL)", (project_id,)):
        text = row["text"]
        if re.search(r"(?<![A-Za-z0-9_.-])" + re.escape(source_id) + r"(?![A-Za-z0-9_.-])", text, re.I):
            return True
        for clause in re.split(r"[.!?;\n。！？；]", text):
            actions = r"\b(search|scan|revisit|photograph|target)\b|搜索|扫描|复拍|目标"
            if not re.search(actions + ("|" + additional_actions if additional_actions else ""), clause, re.I):
                continue
            if any(sid != target_id and re.search(r"(?<![A-Za-z0-9_.-])" + re.escape(sid)
                    + r"(?![A-Za-z0-9_.-])", clause, re.I) for sid in samples):
                return True
    return False


def _correct_exfoliation_output(store, project_id: str, previous: dict, current: dict) -> list[dict]:
    """Correct a model's carrier/chip confusion using this step's one real output."""
    from qumus_simlab.core.materials import normalize_material_code as norm
    from qumus_simlab.flows.library.exfoliation import ExfoliationParams

    if (previous.get("owner") != "hands" or previous.get("status") not in ("pending", "failed")
            or current.get("status") != "pending" or previous.get("id") != current.get("id")
            or not (origins := model_written_contract_events(store, project_id, previous))):
        return []
    old = completion_contract(previous)
    candidates = [c for c in old["sample_outcomes"] if c.get("status") == "exfoliated"
                  and (store.get_sample(c["sample_id"]) or {}).get("kind") == "carrier"]
    runs = linked_physical_jobs(store, project_id, previous["id"])
    if len(candidates) != 1:
        return []
    condition = candidates[0]
    source = store.get_sample(condition["sample_id"])
    material = norm(source["props"].get("material"))
    actions = r"\b(exfoliat\w*|prepar\w*|receiv\w*|onto)\b|剥离|制备|接收"
    if not runs:
        # Before submission only remove the impossible output status, never the
        # named source, its storage requirement or any other physical obligation.
        if (condition.get("in_storage") is not True or source["status"] != "available"
                or source["props"].get("location_uncertain") or source["id"] in store.reserved_samples()
                or not material or [norm(m) for m in old["materials"]] != [material]
                or norm(old["operation_args"].get("material")) != material
                or human_sample_binding_conflict(store, project_id, source["id"], source["id"], additional_actions=actions)
                or store._x("SELECT 1 FROM event WHERE kind='agent.job_linked' AND "
                    "json_extract(payload_json,'$.project_id')=? AND json_extract(payload_json,'$.step_id')=? LIMIT 1",
                    (project_id, previous["id"])).fetchone()
                or any(r["project_id"] == project_id and r["flow"] == "exfoliate_chips"
                       and (r["params"].get("carrier_id") == source["id"]
                            or norm(r["params"].get("material")) == material) for r in store.list_runs())):
            return []
        corrected = {**condition, "status": None}
        repaired = {**deepcopy(previous), "sample_outcomes": [corrected if c == condition else c for c in old["sample_outcomes"]]}
        if (completion_contract(current)["sample_outcomes"] != repaired["sample_outcomes"]
                or invalid_flow_sample_statuses(store, current) or not preserves_obligation(repaired, current)):
            return []
        return [{"step_id": previous["id"], "correction_kind": "exfoliation_source_status",
            "source": "registered_flow_contract", "contract_version": 1,
            "original_condition": condition, "corrected_condition": corrected,
            "original_contract": old, "plan_event_ids": origins, "execution_evidence": None,
            "next_action": "Keep pending and execute the original exfoliation step. Its receiving chip acquires exfoliated status; the source carrier retains its storage requirement."}]
    if len(runs) != 1:
        return []
    run = runs[0]
    if condition.get("location") is not None or run["flow"] != "exfoliate_chips" or run["status"] != "succeeded":
        return []
    try:
        params = ExfoliationParams.model_validate(run["params"])
    except ValueError:
        return []
    if params.num_chips != 1 or len(params.chip_ids) != 1 or params.carrier_id != condition["sample_id"]:
        return []
    source_id, target_id = params.carrier_id, params.chip_ids[0]
    source, target = store.get_sample(source_id), store.get_sample(target_id) or {}
    if ([norm(m) for m in old["materials"]] != [params.material]
            or norm(source["props"].get("material")) != params.material
            or target.get("kind") != "chip" or target.get("status") != "exfoliated"
            or norm(target["props"].get("material")) != params.material
            or target["props"].get("from_carrier") != source_id
            or any(s["props"].get("location_uncertain") for s in (source, target))
            or {source_id, target_id} & store.reserved_samples()
            or human_sample_binding_conflict(store, project_id, source_id, target_id, additional_actions=actions)):
        return []
    # A later/repeated preparation cannot silently stand in for this output.
    if any(r["id"] != run["id"] and r["flow"] == "exfoliate_chips"
           and target_id in r["params"].get("chip_ids", []) for r in store.list_runs()):
        return []
    resolved = {k: v for k, v in run["params"].items() if k in ExfoliationParams.model_fields}
    resolved["material"] = params.material
    # Purpose is an admitted planning label, not a physical parameter. Arbitrary
    # metadata must never replace the executed recipe or legitimize unknown args.
    if "purpose" in (run.get("meta") or {}):
        resolved["purpose"] = run["meta"]["purpose"]
    for step in (previous, current):
        for key, value in (step.get("operation_args") or {}).items():
            key = "carrier_id" if key in ("from_carrier", "source_carrier_id") else key
            if key == "material":
                value = norm(value)
            if key == "chip_ids" and value == [] or key == "prepare_mother_tape" and value == "auto":
                continue  # Explicit allocation/auto-resolution, not a fixed target.
            if key not in resolved or resolved[key] != value:
                return []
    # Require typed, successful storage verification for the same single chip;
    # a mutable status label or a merely succeeded unrelated run is insufficient.
    verification = None
    for row in store._x("SELECT id,payload_json FROM event WHERE kind='step.verified' AND run_id=? ORDER BY id", (run["id"],)):
        payload = json.loads(row["payload_json"])
        if (payload.get("name") == "store_exfoliated_chip" and payload.get("verdict") == "ok"
                and payload.get("expected", {}).get("sample") == target_id
                and payload.get("expected", {}).get("at") in params.store_slots
                and payload.get("measurements", {}).get("found_at") == payload["expected"]["at"]):
            verification = row["id"]
    if verification is None:
        return []
    repaired = deepcopy(previous)
    repaired["sample_outcomes"] = [{**c, "sample_id": target_id} if c == condition else c for c in old["sample_outcomes"]]
    if invalid_flow_sample_statuses(store, current) or not preserves_obligation(repaired, current):
        return []
    return [{"step_id": previous["id"], "correction_kind": "exfoliation_source_to_receiving_chip",
        "source": "successful_step_exfoliation", "contract_version": 1,
        "original_condition": condition, "corrected_condition": {**condition, "sample_id": target_id},
        "original_contract": old, "plan_event_ids": origins,
        "execution_evidence": {"job_id": run["id"], "flow": run["flow"], "material": params.material,
            "source_carrier_id": source_id, "chip_id": target_id, "storage_verification_event_id": verification},
        "next_action": "Keep pending. Recheck the original successful exfoliation job with Hands.wait_job and the delegation check; do not repeat exfoliation."}]


def linked_physical_jobs(store, project_id: str | None, step_id: str | None = None) -> list[dict]:
    """Durable step/job associations survive model turns and gateway restarts."""
    if not project_id:
        return []
    sql = ("SELECT DISTINCT run_id FROM event WHERE kind='agent.job_linked' AND "
           "json_extract(payload_json,'$.project_id')=?")
    args = [project_id]
    if step_id is not None:
        sql += " AND json_extract(payload_json,'$.step_id')=?"
        args.append(step_id)
    return [run for row in store._x(sql, args)
            if (run := store.get_run(row["run_id"])) and run["project_id"] == project_id]


def linked_job_evidence(store, project_id: str | None, step_id: str | None) -> list[dict]:
    """Exact, read-only job references for one evidenced step; never ID guesses."""
    if not project_id or not step_id:
        return []
    references = []
    for run in linked_physical_jobs(store, project_id, step_id):
        links = store._x("SELECT id FROM event WHERE kind='agent.job_linked' AND run_id=? AND "
                         "json_extract(payload_json,'$.project_id')=? AND "
                         "json_extract(payload_json,'$.step_id')=? ORDER BY id",
                         (run["id"], project_id, step_id)).fetchall()
        status = store._x("SELECT id FROM event WHERE kind='job.status' AND run_id=? ORDER BY id DESC LIMIT 1",
                          (run["id"],)).fetchone()
        references.append({"job_id": run["id"], "flow": run["flow"], "status": run["status"],
            "targets": {k: v for k, v in run["params"].items() if k in (
                "chip_id", "chip_ids", "flake_id", "flake_ids", "target_chip_id", "carrier_id", "stamp_id", "material")},
            "linked_event_ids": [row["id"] for row in links],
            "status_event_id": status["id"] if status else None})
    return references


def cancelled_job_retired(store, run: dict | None) -> bool:
    """Cancellation is accounted for, never claimed as successful execution."""
    if not run or run["status"] != "cancelled":
        return False
    if any(r["run_id"] == run["id"] for r in store.reservations().values()):
        return False
    if isinstance(run.get("error"), dict) and run["error"].get("category") == "dangerous":
        return False

    def references(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)

    return not any(sample["props"].get("location_uncertain") or sample["props"].get("reactor_state_uncertain")
                   for ref in references(run.get("params", {})) if (sample := store.get_sample(ref)))


def outstanding_physical_jobs(store, project_id: str | None) -> list[dict]:
    return [run for run in linked_physical_jobs(store, project_id)
            if run["status"] != "succeeded" and not cancelled_job_retired(store, run)]


def _storage_failure(store, bench: dict, sample: dict) -> str | None:
    """Typed containment: flake/device -> current chip -> matching storage slot."""
    holder = sample
    if sample["kind"] in ("flake", "device"):
        if sample["location"] == sample["id"]:
            return "cyclic sample containment cannot establish storage"
        holder = store.get_sample(sample["location"])
        if holder is None:
            return f"carrier chip {sample['location']} is missing"
        if holder["kind"] != "chip":
            return f"{holder['id']} is a {holder['kind']}, not a carrier chip"
        if holder["props"].get("location_uncertain"):
            return f"carrier chip {holder['id']} has uncertain location"
        if holder["location"] in (sample["id"], holder["id"]):
            return "cyclic sample containment cannot establish storage"
    slot = bench.get("slots", {}).get(holder["location"], {})
    station = bench.get("stations", {}).get(slot.get("station"), {})
    if station.get("kind") != "storage" or slot.get("holds") != holder["kind"]:
        parent = f"carrier chip {holder['id']}: " if holder is not sample else ""
        return f"{parent}{holder['location']} is not a matching storage slot"
    return None


def unmet_sample_outcomes(store, bench: dict, conditions: list[dict]) -> list[str]:
    failures = []
    for value in conditions:
        try:
            wanted = SampleOutcome.model_validate(value)
        except ValueError:
            failures.append("invalid sample completion condition; revise the Task Plan")
            continue
        sample = store.get_sample(wanted.sample_id)
        if sample is None:
            failures.append(f"{wanted.sample_id}: sample is missing")
            continue
        if sample["props"].get("location_uncertain"):
            failures.append(f"{wanted.sample_id}: current location is uncertain")
            continue
        for key in ("location", "status"):
            expected = getattr(wanted, key)
            if expected is not None and sample[key] != expected:
                failures.append(f"{wanted.sample_id}: {key} is {sample[key]}, expected {expected}")
        if wanted.in_storage and (failure := _storage_failure(store, bench, sample)):
            failures.append(f"{wanted.sample_id}: {failure}")
    return failures


def unmet_reactor_outcome(store, bench: dict, value: dict | None, *, after_event_id: int = 0,
                         project_id: str | None = None) -> list[str]:
    """Check flow-emitted device readbacks, never simulator truth or model claims.

    Reactor flows emit reactor.process with chip_id, temperature_target_c,
    flow_target_sccm, temperature_c and flow_sccm after their final commands.
    A later control/motion command invalidates the reading until verified again.
    """
    if value is None:
        return ["declare reactor_outcome: cold_storage or intentional observation_hold"]
    try:
        wanted = ReactorOutcome.model_validate(value)
    except ValueError:
        return ["invalid reactor_outcome; revise the Task Plan"]
    chip = store.get_sample(wanted.chip_id)
    if not bench.get("reactor_module") or not chip or not chip["props"].get("reactor_substrate"):
        return ["reactor_outcome needs a dedicated chip on an enabled reactor bench"]
    condition = {"sample_id": wanted.chip_id}
    if wanted.disposition == "cold_storage":
        condition["in_storage"] = True
    else:
        condition["location"] = "reactor_port"
    failures = unmet_sample_outcomes(store, bench, [condition])
    if wanted.disposition == "observation_hold" and chip["props"].get("reactor_state_uncertain"):
        failures.append("reactor process state is uncertain; an observation hold is not verified")
    row = store._x("SELECT id,run_id,payload_json FROM event WHERE kind='reactor.process' "
                   "AND json_extract(payload_json,'$.chip_id')=? AND id>? ORDER BY id DESC LIMIT 1",
                   (wanted.chip_id, after_event_id)).fetchone()
    run = store.get_run(row["run_id"]) if row and row["run_id"] else None
    if (run is None or run["status"] != "succeeded"
            or project_id is not None and run["project_id"] != project_id
            or (run["params"].get("chip_id") or run["params"].get("target_chip_id")) != wanted.chip_id):
        return [*failures, "no current flow-verified reactor process readings"]
    later = store._x("SELECT id FROM event WHERE id>? AND kind='device.command' AND "
                     "json_extract(payload_json,'$.device') IN "
                     "('reactor_heater','reactor_flow','reactor_slot','reactor_insert','arm_reactor') LIMIT 1",
                     (row["id"],)).fetchone()
    if later:
        failures.append("reactor process readings precede a later control/motion command; verify again")
    p = json.loads(row["payload_json"])
    for key in ("temperature_target_c", "flow_target_sccm"):
        if key not in p or p[key] != getattr(wanted, key):
            failures.append(f"{key} does not match the declared final target")
    for key in ("temperature_c", "flow_sccm"):
        if not isinstance(p.get(key), (int, float)) or isinstance(p.get(key), bool) or not math.isfinite(p[key]):
            failures.append(f"{key} has no finite measured readback")
    if not failures and wanted.disposition == "cold_storage":
        from qumus_simlab.safety.limits import Limits

        limit = next((x for x in Limits.load().reactor_interlocks if x.stage == "reactor_slot"), None)
        if limit is None:
            failures.append("reactor access limits are not configured")
        elif p["temperature_c"] > limit.access_max_celsius or p["flow_sccm"] > limit.access_max_flow_sccm:
            failures.append("reactor has not reached the configured cold-access conditions")
    elif not failures:
        expected_t = (wanted.temperature_target_c if wanted.temperature_target_c is not None else
                      bench["reactor_module"]["thermal"]["ambient_c"])
        if abs(p["temperature_c"] - expected_t) > 3:
            failures.append("measured temperature has not stabilized at the declared observation hold")
        if abs(p["flow_sccm"] - wanted.flow_target_sccm) > max(1.0, 0.05 * wanted.flow_target_sccm):
            failures.append("measured gas flow has not stabilized at the declared observation hold")
    return failures
