"""Human review of mother-tape identity; images never establish material identity."""

from __future__ import annotations

import json
from typing import Any

from qumus_simlab.core.materials import normalize_material_code
from qumus_simlab.flows.bench import slot_station
from qumus_simlab.records.lab_geometry import record_report
from qumus_simlab.runtime.store import RunStore


def pickup_admission_error(store: RunStore, flake: dict[str, Any]) -> str | None:
    """Admit only current observed clear neighbourhoods; retain unclear evidence for display."""
    from qumus_simlab.records.geometry import current_pickup_evidence

    evidence = current_pickup_evidence(store, flake)
    if evidence is not None and evidence.get("clear") is True:
        return None
    detail = ""
    if evidence is not None:
        reasons = sorted({v["reason"] for v in evidence.get("views", []) if v.get("reason")})
        if reasons:
            detail = " Observation unresolved: " + ", ".join(reasons) + "; this does not establish nearby contamination."
        elif evidence.get("assessment_status") == "occupied":
            detail = " Other detected material lies within the configured local edge clearance."
    return (f"{flake['id']} needs current observed clear pickup-neighbourhood evidence; "
            "request fresh inspection or choose another flake. Unknown, stale or unclear "
            "neighbourhoods cannot be admitted for pickup." + detail)


def material_review(sample: dict[str, Any]) -> dict[str, Any]:
    props = sample.get("props") or {}
    review = (props.get("material_reviews") or [{}])[-1]
    if review.get("material") == props.get("material") and review.get("via") in ("ui:lab", "chat"):
        return review
    return {"status": "unreviewed", "material": props.get("material")}


def simulation_identity(sample: dict[str, Any], bench: str, *,
                        restored: dict[str, dict] | None = None) -> dict[str, Any]:
    """Scope a saved configuration declaration; never create a human review."""
    props = sample.get("props") or {}
    declaration = ((props.get("simulation_identity") or {}) if restored is None
                   else restored.get(sample.get("id"), {}))
    material = normalize_material_code(props.get("material"))
    reviews = props.get("material_reviews") or []
    latest = reviews[-1] if reviews else {}
    conflict = bool(latest and (latest.get("status") == "uncertain"
                    or normalize_material_code(latest.get("material")) != material))
    declared = (sample.get("kind") == "carrier" and material is not None
                and declaration.get("source") in ("testbench_config.initial_samples", "restored_testbench_world")
                and declaration.get("sample_id") == sample.get("id")
                and normalize_material_code(declaration.get("material")) == material)
    usable = bench == "sim" and declared and not conflict
    return {"usable": usable, "bench": bench, "declaration": declaration or None,
            "scope": "simulation only; not physical material confirmation",
            "reason": ("Configured simulation identity may be used while human_review remains unchanged."
                       if usable else "No matching simulation declaration, non-sim bench, or uncertain/conflicting identity; "
                       "resolve identity before using this simulation exception."),
            "readiness": "not evaluated; observation, flow preconditions and SafetyGate still apply"}


def restore_simulation_identities(store: RunStore, world: Any) -> None:
    """Bind current legacy labels to restored objects without altering their history."""
    if store.bench != "sim":
        return
    declarations = {}
    for sample in store.list_samples(kind="carrier"):
        obj = world.objects.get(sample["id"])
        material = normalize_material_code(sample["props"].get("material"))
        if (obj is not None and obj.id == sample["id"] and obj.kind == "carrier" and material
                and normalize_material_code(obj.material) == material):
            declarations[sample["id"]] = {"source": "restored_testbench_world",
                "sample_id": sample["id"], "material": obj.material}
    # Even an empty map overrides old declarations: a missing/conflicting saved
    # object must not acquire identity from the default configuration underneath it.
    store.set_meta("simulation_identity_restore", json.dumps(declarations))


def review_material(store: RunStore, bench: dict[str, Any], carrier_id: str, *,
                    expected_material: str, by: str, status: str, note: str = "") -> dict[str, Any]:
    """Save a person's confirmation or uncertainty, without relabelling physical material.

    The live Lab UI is the only writer; there is deliberately no agent tool.
    A confirmation is tied to the exact inventory label reviewed by the person.
    """
    with store._transaction():
        sample = store.get_sample(carrier_id)
        if not sample or sample["kind"] != "carrier":
            raise ValueError("Choose a mother tape from the material inventory.")
        material = sample["props"].get("material")
        if not material or expected_material != material:
            raise ValueError("The inventory label is missing or changed; reload it before reviewing.")
        if not by.strip():
            raise ValueError("Enter the name of the person who checked the label or preparation record.")
        if status not in {"confirmed", "uncertain"}:
            raise ValueError("Review status must be confirmed or uncertain.")
        if slot_station(bench, sample["location"]) != "left_storage":
            raise ValueError("Return the mother tape to left storage before reviewing its identity.")
        if carrier_id in store.reserved_samples():
            raise ValueError("This mother tape is reserved by a job; finish that work before reviewing.")
        record = {"material": material, "status": status, "by": by.strip(), "note": note.strip(),
                  "via": "ui:lab", "ts": store.clock.now(), "location": sample["location"]}
        store.upsert_sample(carrier_id, "carrier", props={
            "material_reviews": [*(sample["props"].get("material_reviews") or []), record]})
        record_report(store, carrier_id, "material_identity", material, instruction_id=None,
                      reporter=by.strip(), via="ui:lab",
                      status="uncertain" if status == "uncertain" else "asserted",
                      event_ids=[store.last_event_id()])
    return {"carrier_id": carrier_id, **record}


def record_material_source(store, bench, carrier_id, material, *, instruction_id, project_id):
    """Human-reported source replacement/identity, parallel to stamp support reports.

    Updates the ledger, never pretends to move or identify a physical object.
    Previous reports and already produced flakes retain their own provenance.
    """
    from qumus_simlab.core.materials import materials_in_text

    with store._transaction():
        instruction = store.get_instruction(instruction_id)
        sample = store.get_sample(carrier_id)
        material = normalize_material_code(material)
        if (not instruction or instruction["source"] != "ui"
                or instruction["project_id"] not in (None, project_id)
                or carrier_id not in instruction["text"] or material not in materials_in_text(instruction["text"].replace(carrier_id, ""))):
            raise ValueError("Cite the person's report naming this material source and its material.")
        if (not sample or sample["kind"] != "carrier" or carrier_id in store.reserved_samples()
                or slot_station(bench, sample["location"]) != "left_storage"):
            raise ValueError("Report only an unreserved material source in storage.")
        record = {"material": material, "status": "confirmed", "by": "user_report", "via": "chat",
            "instruction_id": instruction_id, "ts": store.clock.now(), "location": sample["location"],
            "previous_material": sample["props"].get("material"), "source": "operator_report"}
        reports = sample["props"].get("material_reviews") or []
        if reports and reports[-1].get("instruction_id") == instruction_id and reports[-1].get("material") == material:
            return reports[-1]
        store.upsert_sample(carrier_id, "carrier", props={"material": material, "material_reviews": [*reports, record]})
        event_id = store.emit("sample.material_reported", {"carrier_id": carrier_id, **record}, agent="orchestrator")
        record_report(store, carrier_id, "material_identity", material,
                      instruction_id=instruction_id, event_ids=[event_id])
    return record
