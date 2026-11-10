"""Material-metal device simulation and a separately authorized abstract fixture.

The default uses guarded simulated mechanics, optics and qualitative interface
progress. The historical abstract protocol never moves physical model objects.
Neither branch operates real hardware or establishes chemical correctness.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.flows.bench import slot_station, stamp_slots
from qumus_simlab.flows.flow import Flow, register_flow
from qumus_simlab.flows.library.reactor import RecordReactorProcess
from qumus_simlab.flows.library.stacking import StackDeviceFlow, StackParams
from qumus_simlab.flows.step import Observation, ProbeResult, Step, Verdict
from qumus_simlab.records.material_metal import (
    MaterialMetalReceipt,
    adapter_for,
    preparation_errors,
)
from qumus_simlab.records.material_metal_simulation import SIM_PROTOCOL
from qumus_simlab.records.reactor_requirements import (
    ReactorRequirements,
    input_errors,
    requirements_record,
)
from qumus_simlab.records.stamps import contact_film


class MaterialMetalPrepareParams(StackParams):
    model_config = ConfigDict(extra="forbid")
    project_id: str
    target_chip_id: str
    flake_ids: list[str] = Field(min_length=2, max_length=2)
    stamp_id: str | None = None
    protocol_id: str = SIM_PROTOCOL
    fixture_instruction_id: str | None = None
    reactor_home_slot: str | None = None
    requirements_event_id: int | None = None
    source_chip_ids: dict[str, str] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)
    receipt_inputs: dict[str, Any] = Field(default_factory=dict)


class MaterialMetalObserveParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str
    chip_id: str
    protocol_id: str = SIM_PROTOCOL
    fixture_instruction_id: str | None = None
    home_slot: str | None = None
    temperature_c: float | None = Field(None, ge=22, le=950)
    flow_sccm: float | None = Field(None, ge=0, le=400)
    duration_s: float | None = Field(None, gt=0, le=900)
    observation_interval_s: float = Field(20, ge=5, le=60)
    requirements_event_id: int | None = None
    goal: dict = Field(default_factory=dict)


def _adapter(ctx):
    try:
        return adapter_for(ctx.store, ctx.params["protocol_id"], ctx.params["fixture_instruction_id"], ctx.project_id)
    except ValueError as exc:
        raise PreconditionFailed(str(exc)) from exc


class AdapterAction(Step):
    """Run an adapter action and check independent persisted adapter observations."""
    def __init__(self, action: str, payload: dict):
        self.action, self.payload = action, payload
        self.name = f"material_metal_{action}" + (f"_{payload['flake_id']}" if "flake_id" in payload else "")

    def preconditions(self, ctx):
        record = requirements_record(ctx.store, ctx.project_id)
        if not record or record["event_id"] != ctx.params["requirements_event_id"]:
            raise PreconditionFailed("Requirements changed since job admission; replan before continuing")
        _adapter(ctx)
        if "receipt_inputs" in ctx.params:
            receipt = ctx.params["receipt_inputs"]
            req = ReactorRequirements.model_validate(record["requirements"])
            errors = input_errors(ctx.store, req, ctx.params["flake_ids"],
                                  carrier_id=receipt["chip_id"], require_assessment=True)
            selection = ctx.store.get_project(ctx.project_id)["state"].get("reactor_selection") or {}
            if selection.get("confirmed_ids") != ctx.params["flake_ids"] or selection.get("requirements_event_id") != record["event_id"]:
                errors.append("Picker confirmation changed during preparation")
            if ctx.store.get_sample(receipt["chip_id"])["props"].get("metal_coating") != receipt["metal_coating"]:
                errors.append("Metal provenance changed during preparation")
            if contact_film(ctx.store.get_sample(receipt["stamp_id"])) != receipt["stamp_film"]:
                errors.append("Stamp support changed during preparation")
            for fid, source in receipt["source_chip_ids"].items():
                sample = ctx.store.get_sample(fid)
                if sample["location"] != source or sample["status"] not in ("catalogued", "selected"):
                    errors.append(f"Source {fid} moved or became unavailable during the software fixture")
            if errors:
                raise PreconditionFailed("; ".join(errors))

    def act(self, ctx, correction):
        ctx.check_cancel()
        try:
            _adapter(ctx).apply(ctx, self.action, self.payload)
        except ValueError as exc:
            raise PreconditionFailed(str(exc)) from exc

    def verify(self, ctx):
        state = _adapter(ctx).observe(ctx, self.payload["chip_id"])
        expected_ids = ctx.params.get("flake_ids", [])
        if self.action == "initialize":
            ok = state.get("preparation_job_id") == ctx.run_id and state.get("top_first_flake_ids") == expected_ids
        elif self.action == "pickup":
            prefix = expected_ids[:expected_ids.index(self.payload["flake_id"]) + 1]
            ok = state.get("picked_ids") == prefix
        elif self.action == "release":
            ok = state.get("released") and state.get("picked_ids") == expected_ids
        else:
            ok = state.get("observation_job_id") == ctx.run_id and state.get("observable") == "workflow_sequence_receipt"
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR, {"adapter_state": state},
            message="Abstract software fixture observed; no physical motion or chemistry is verified")

    def probe_state(self, ctx):
        # Reapply idempotent fixture actions, including their checked ledger
        # writes. Never infer a physical action from these software events.
        return ProbeResult.NOT_DONE


class CommitMaterialMetalPreparation(AdapterAction):
    def on_success(self, ctx, obs):
        receipt = MaterialMetalReceipt.model_validate({**ctx.params["receipt_inputs"], "preparation_job_id": ctx.run_id})
        value = receipt.model_dump()
        with ctx.store._transaction():
            ctx.store.upsert_sample(receipt.chip_id, "chip", props={"material_metal_preparation": value})
            ctx.store.emit("reactor.prepared", {"chip_id": receipt.chip_id,
                "material_metal_preparation": value, "scope": receipt.scope,
                "physical_motion_verified": False, "images": receipt.images,
                "image_role": "original_source_catalogue_not_release_evidence"}, run_id=ctx.run_id)


class CommitMaterialMetalObservation(AdapterAction):
    def preconditions(self, ctx):
        super().preconditions(ctx)
        if errors := preparation_errors(ctx.store, ctx.project_id, self.payload["chip_id"]):
            raise PreconditionFailed("; ".join(errors))
        project = ctx.store.get_project(ctx.project_id)
        if project["state"].get("reactor_goal") != ctx.params["goal"]:
            raise PreconditionFailed("Reactor task objective changed since admission")

    def on_success(self, ctx, obs):
        state = obs.measurements["adapter_state"]
        observation = {"scope": "synthetic_workflow", "protocol_id": ctx.params["protocol_id"],
            "observable": state["observable"], "job_id": ctx.run_id, "project_id": ctx.project_id,
            "requirements_event_id": ctx.params["requirements_event_id"], "chip_id": self.payload["chip_id"],
            "preparation_job_id": state["preparation_job_id"], "top_first_flake_ids": state["top_first_flake_ids"],
            "layer_order": state["layer_order"], "sequence": state["sequence"],
            "chemistry_verified": False, "physical_motion_verified": False,
            "product_identity": None, "reaction_fraction": None, "timestamp": ctx.clock.now()}
        with ctx.store._transaction():
            ctx.store.upsert_sample(self.payload["chip_id"], "chip", props={"material_metal_observation": observation})
            ctx.store.emit("material_metal.observed", {"observation": observation}, run_id=ctx.run_id)


@register_flow
class PrepareMaterialMetalFlow(Flow):
    name = "prepare_material_metal"
    Params = MaterialMetalPrepareParams
    devices = (*StackDeviceFlow.devices, "arm_reactor", "reactor_heater", "reactor_flow", "reactor_slot", "reactor_insert")
    description = "Execute a registered material-metal adapter and preserve top-first input/coating provenance."

    def resolve_params(self, params, store):
        if params.protocol_id == SIM_PROTOCOL:
            from qumus_simlab.flows.library.material_metal_device import device_flow

            return device_flow(self.bench).resolve_params(params, store)
        p = params.model_copy(deep=True)
        adapter_for(store, p.protocol_id, p.fixture_instruction_id, p.project_id)
        record = requirements_record(store, p.project_id)
        if not record:
            raise ValueError("Establish the material-metal experiment before preparation")
        req = ReactorRequirements.model_validate(record["requirements"])
        errors = input_errors(store, req, p.flake_ids, carrier_id=p.target_chip_id, require_assessment=True)
        selection = store.get_project(p.project_id)["state"].get("reactor_selection") or {}
        if selection.get("confirmed_ids") != p.flake_ids or selection.get("chip_id") != p.target_chip_id or selection.get("requirements_event_id") != record["event_id"]:
            errors.append("Picker must confirm this exact pair for the current requirements and carrier")
        chip, stamp = store.get_sample(p.target_chip_id), store.get_sample(p.stamp_id)
        if not chip or slot_station(self.bench, chip["location"]) != "reactor_storage" or chip["props"].get("material_metal_preparation"):
            errors.append("Choose an unprepared dedicated carrier in reactor storage")
        if not stamp or stamp["kind"] != "stamp" or stamp["location"] not in stamp_slots(self.bench) or stamp["status"] != "fresh" or contact_film(stamp) not in ("PC", "mica"):
            errors.append("Choose a fresh stored stamp with explicitly known PC or mica support")
        rows = [store.get_sample(fid) for fid in p.flake_ids]
        if any(not row or row["status"] not in ("catalogued", "selected") for row in rows):
            errors.append("Choose available catalogued or selected inputs")
        reserved = store.reserved_samples()
        if any(sid in reserved for sid in [p.target_chip_id, p.stamp_id, *p.flake_ids, *(row["location"] for row in rows if row)]):
            errors.append("A preparation input or source carrier is reserved")
        if errors:
            raise ValueError("; ".join(errors))
        p.requirements_event_id = record["event_id"]
        p.source_chip_ids = {row["id"]: row["location"] for row in rows}
        p.images = [path for row in rows for path in (row["props"]["images"].values() if isinstance(row["props"]["images"], dict) else row["props"]["images"])]
        p.receipt_inputs = {"project_id": p.project_id, "requirements_event_id": record["event_id"],
            "chip_id": p.target_chip_id, "top_first_flake_ids": p.flake_ids,
            "layer_order": ["hBN", req.reactive_material, req.metal, "silicon"],
            "stamp_id": p.stamp_id, "stamp_film": contact_film(stamp), "source_chip_ids": p.source_chip_ids,
            "metal_coating": chip["props"]["metal_coating"], "images": p.images,
            "protocol_id": p.protocol_id, "scope": "synthetic_workflow", "fixture_instruction_id": p.fixture_instruction_id}
        return p

    def reserve(self, params, store, run_id):
        if params.protocol_id == SIM_PROTOCOL:
            from qumus_simlab.flows.library.material_metal_device import device_flow

            return device_flow(self.bench).reserve(params, store, run_id)
        for sid in dict.fromkeys([params.target_chip_id, params.stamp_id, *params.flake_ids, *params.source_chip_ids.values()]):
            store.reserve("sample", sid, run_id, self.name)

    def build_workflow(self, p):
        if p.protocol_id == SIM_PROTOCOL:
            from qumus_simlab.flows.library.material_metal_device import device_flow

            return device_flow(self.bench).build_workflow(p)
        return None

    def build_steps(self, p):
        if p.protocol_id == SIM_PROTOCOL:
            from qumus_simlab.flows.library.material_metal_device import device_flow

            return device_flow(self.bench).build_steps(p)
        return [AdapterAction("initialize", p.receipt_inputs),
            *(AdapterAction("pickup", {"chip_id": p.target_chip_id, "flake_id": fid}) for fid in p.flake_ids),
            CommitMaterialMetalPreparation("release", {"chip_id": p.target_chip_id})]


@register_flow
class ReactorMaterialMetalFlow(Flow):
    name = "reactor_material_metal"
    Params = MaterialMetalObserveParams
    devices = (*RecordReactorProcess.devices, "arm_reactor", "reactor_microscope", "overview_camera")
    description = "Observe the prepared material-metal workflow through its registered adapter."

    def resolve_params(self, params, store):
        if params.protocol_id == SIM_PROTOCOL:
            from qumus_simlab.flows.library.material_metal_device import resolve_reaction

            return resolve_reaction(self.bench, params, store)
        p = params.model_copy(deep=True)
        adapter_for(store, p.protocol_id, p.fixture_instruction_id, p.project_id)
        errors = preparation_errors(store, p.project_id, p.chip_id)
        project = store.get_project(p.project_id)
        goal = project["state"].get("reactor_goal") or {}
        receipt = (store.get_sample(p.chip_id) or {}).get("props", {}).get("material_metal_preparation") or {}
        if goal.get("source") != "reactor_task_dialogue" or goal.get("chip_id") != p.chip_id or goal.get("preparation") != receipt:
            errors.append("Convey and bind the objective through Reactor task delegation before this job")
        if receipt.get("protocol_id") != p.protocol_id or receipt.get("fixture_instruction_id") != p.fixture_instruction_id:
            errors.append("Keep the preparation adapter and explicit fixture authority")
        if p.chip_id in store.reserved_samples():
            errors.append("The carrier is reserved")
        if errors:
            raise ValueError("; ".join(errors))
        p.requirements_event_id, p.goal = goal["requirements_event_id"], goal
        return p

    def reserve(self, params, store, run_id):
        if params.protocol_id == SIM_PROTOCOL:
            store.reserve("slot", "reactor_port", run_id, self.name)
            store.reserve("slot", params.home_slot, run_id, self.name)
        store.reserve("sample", params.chip_id, run_id, self.name)
        receipt = store.get_sample(params.chip_id)["props"]["material_metal_preparation"]
        for fid in receipt["top_first_flake_ids"]:
            store.reserve("sample", fid, run_id, self.name)

    def build_workflow(self, p):
        if p.protocol_id == SIM_PROTOCOL:
            from qumus_simlab.flows.library.material_metal_device import reaction_workflow

            return reaction_workflow(self.bench, p)
        return None

    def build_steps(self, p):
        if p.protocol_id == SIM_PROTOCOL:
            from qumus_simlab.flows.library.material_metal_device import reaction_steps

            return reaction_steps(self.bench, p)
        return [CommitMaterialMetalObservation("observe", {"chip_id": p.chip_id}),
                RecordReactorProcess(p.chip_id, cold_storage=True)]

    def on_step_failed(self, params, step, outcome, ctx):
        if params.protocol_id == SIM_PROTOCOL:
            from qumus_simlab.flows.library.reactor import disable_targets

            disable_targets(ctx)
            if step.name in ("set_reactor_conditions", "observe_material_metal_progress"):
                ctx.store.upsert_sample(params.chip_id, "chip", props={"reactor_state_uncertain": True})
