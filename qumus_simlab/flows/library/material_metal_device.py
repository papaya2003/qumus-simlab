"""Ordinary simulated device chain; all mechanics use SafetyGate and Clock.

No simulator objects are changed by this flow. Optical adapters independently
verify the device model's transfers before the ledger records their postconditions.
"""
from __future__ import annotations

from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.flows.bench import slot_station
from qumus_simlab.flows.library.place_chip import PlaceChipOnTransferStage
from qumus_simlab.flows.library.primitives import (
    MICROSCOPE,
    analyzer,
    autofocus,
)
from qumus_simlab.flows.library.reactor import (
    GrowthParams,
    GrowthWindow,
    SetReactorConditions,
    establish_conditions,
    load_and_acquire,
    observe_carrier,
    observe_exposure,
    record_terminal,
    shutdown_and_cool,
    store_carrier,
)
from qumus_simlab.flows.library.reactor_preparation import shared_storage
from qumus_simlab.flows.library.stacking import (
    DropDown,
    PickUpFlake,
    StackDeviceFlow,
    SupportRelease,
    fresh_stamps,
)
from qumus_simlab.flows.library.transfer import MoveObject
from qumus_simlab.flows.step import Observation, ProbeResult, Step, Verdict
from qumus_simlab.flows.workflow import Assembly, Molecule
from qumus_simlab.records.material_metal_simulation import (
    SIM_PROTOCOL,
    device_preparation_errors,
    simulation_profile,
)
from qumus_simlab.records.reactor_requirements import (
    ReactorRequirements,
    input_errors,
    requirements_record,
)
from qumus_simlab.records.stamps import contact_film


def require_contract(ctx):
    record = requirements_record(ctx.store, ctx.project_id)
    receipt = ctx.params.get("receipt_inputs") or ctx.params.get("goal", {}).get("preparation", {})
    chip = ctx.store.get_sample(receipt.get("chip_id", "")) or {}
    stamp = ctx.store.get_sample(receipt.get("stamp_id", "")) or {}
    if (ctx.store.bench != "sim" or not record or record["event_id"] != receipt.get("requirements_event_id")
            or chip.get("props", {}).get("metal_coating") != receipt.get("metal_coating")
            or contact_film(stamp) != receipt.get("stamp_film")
            or simulation_profile(ctx.bench).model_dump() != receipt.get("model_assumptions")):
        raise PreconditionFailed("Requirements, coating, known stamp support or simulation assumptions changed")


class CheckDeviceContract(Step):
    name = "check_material_metal_contract"

    def preconditions(self, ctx):
        require_contract(ctx)
        if "goal" in ctx.params:
            errors = device_preparation_errors(ctx.store, ctx.project_id, ctx.params["chip_id"])
            if ctx.store.get_project(ctx.project_id)["state"].get("reactor_goal") != ctx.params["goal"]:
                errors.append("Task goal changed before the device job started")
            if errors:
                raise PreconditionFailed("; ".join(errors))

    def act(self, ctx, correction):
        pass

    def verify(self, ctx):
        self.preconditions(ctx)
        return Observation(Verdict.OK, {"scope": "device_simulation"}, message="Current task and simulation assumptions match")

    def probe_state(self, ctx):
        return ProbeResult.NOT_DONE


class DevicePickup(PickUpFlake):
    def preconditions(self, ctx):
        require_contract(ctx)
        selection = ctx.store.get_project(ctx.project_id)["state"].get("reactor_selection") or {}
        if selection.get("confirmed_ids") != ctx.params["flake_ids"]:
            raise PreconditionFailed("Picker confirmation changed before pickup")
        super().preconditions(ctx)


class DeviceRelease(SupportRelease):
    name = "release_material_metal_stack"

    def preconditions(self, ctx):
        require_contract(ctx)
        super().preconditions(ctx)

    def expected(self, ctx):
        value = super().expected(ctx)
        value.pop("melt_c", None)
        return {**value, "release_c": self.melt_c, "scope": "uncalibrated_device_simulation",
                "support": ctx.params["receipt_inputs"]["stamp_film"]}


    def on_success(self, ctx, obs):
        self._admit_release_commit(ctx, obs)
        film = ctx.params["receipt_inputs"]["stamp_film"]
        for fid in self.flake_ids:
            ctx.store.upsert_sample(fid, "flake", location=self.target_chip, status="stacked")
        retained = getattr(simulation_profile(ctx.bench), film).support_retained_on_stamp
        ctx.store.upsert_sample(self.stamp_id, "stamp", props={"film": film if retained else None, "stack": []})
        ctx.store.emit("material_metal.released", {"chip_id": self.target_chip, "top_first_flake_ids": self.flake_ids,
            "stamp_film": film, "support_retained_on_stamp": retained, "scope": "device_simulation",
            "images": obs.images, "measurement": obs.measurements}, run_id=ctx.run_id)


class VerifyDevicePreparation(Step):
    name = "verify_material_metal_preparation"
    devices = (MICROSCOPE, "transfer_stage")

    def __init__(self, params):
        self.p = params

    def preconditions(self, ctx):
        require_contract(ctx)

    def act(self, ctx, correction):
        autofocus(ctx, ctx.gate.microscope(MICROSCOPE).get_objective())

    def verify(self, ctx):
        samples = [ctx.store.get_sample(fid) for fid in reversed(self.p.flake_ids)]
        if any(s["location"] != self.p.target_chip_id or s["status"] != "stacked" for s in samples):
            return Observation(Verdict.UNCLEAR, message="Selected layers are not on the target carrier")
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        try:
            # Reuse only the independent focused geometry verifier. No legacy
            # reaction model is constructed and neither flake is consumed.
            _, self.updates = analyzer(ctx).reactor_preparation(frame, samples, preparation_id=ctx.run_id)
        except ValueError as exc:
            return Observation(Verdict.UNCLEAR, message=str(exc))
        image = frame.saved_path()
        if not image:
            return Observation(Verdict.UNCLEAR, message="Save a focused release micrograph before preparation is accepted")
        self.receipt = {**self.p.receipt_inputs, "preparation_job_id": ctx.run_id, "release_images": [image],
                        "placements": {fid: value["material_placement"] for fid, value in self.updates.items()}}
        return Observation(Verdict.OK, {"preparation": self.receipt}, images=[image],
                           message="Guarded simulated transfer and actual target micrograph agree")

    def on_success(self, ctx, obs):
        with ctx.store._transaction():
            for fid, value in self.updates.items():
                ctx.store.upsert_sample(fid, "flake", props=value)
            ctx.store.upsert_sample(self.p.target_chip_id, "chip", status="material_metal_prepared",
                props={"material_metal_preparation": self.receipt})
            ctx.store.emit("reactor.prepared", {"chip_id": self.p.target_chip_id,
                "material_metal_preparation": self.receipt, "images": obs.images, "scope": "device_simulation"}, run_id=ctx.run_id)

    def probe_state(self, ctx):
        return ProbeResult.NOT_DONE


class DevicePrepareFlow(StackDeviceFlow):
    name = "prepare_material_metal"

    def select_stamp(self, params, store):
        fresh = fresh_stamps(store, self.bench)
        choices = [s for s in fresh if (not params.stamp_id or s["id"] == params.stamp_id) and contact_film(s) in ("PC", "mica")]
        if not choices:
            raise ValueError("Select a fresh stored stamp whose PC/mica support is already known")
        return choices[0]

    def target_storage(self, target, store):
        if (not target["props"].get("reactor_substrate") or slot_station(self.bench, target["location"]) != "reactor_storage"
                or target["props"].get("material_metal_preparation") or target["props"].get("reactor_pair")
                or store.list_samples(kind="flake", location=target["id"])):
            raise ValueError("Use an unassembled Pd/Si carrier in dedicated reactor storage")
        return shared_storage(self.bench, store)

    def resolve_params(self, params, store):
        if store.bench != "sim" or "reactor_module" not in self.bench:
            raise ValueError("This uncalibrated material-metal device model is simulation-only")
        record = requirements_record(store, params.project_id)
        if not record:
            raise ValueError("Establish current material-metal requirements first")
        req = ReactorRequirements.model_validate(record["requirements"])
        errors = input_errors(store, req, params.flake_ids, carrier_id=params.target_chip_id, require_assessment=True)
        selection = store.get_project(params.project_id)["state"].get("reactor_selection") or {}
        if (selection.get("confirmed_ids") != params.flake_ids or selection.get("chip_id") != params.target_chip_id
                or selection.get("requirements_event_id") != record["event_id"]):
            errors.append("Picker must confirm these exact inputs for the current requirement and carrier")
        if any(fid in store.reserved_samples() for fid in [*params.flake_ids, params.target_chip_id]):
            errors.append("Inputs or carrier are reserved")
        if errors:
            raise ValueError("; ".join(errors))
        p = super().resolve_params(params, store)
        film = contact_film(store.get_sample(p.stamp_id))
        profile = simulation_profile(self.bench)
        transfer = getattr(profile, film)
        p.pickup_temp_c, p.drop_temp_c, p.melt_temp_c = transfer.pickup_c, transfer.release_c - 15, transfer.release_c
        p.requirements_event_id = record["event_id"]
        p.reactor_home_slot = store.get_sample(p.target_chip_id)["location"]
        p.source_chip_ids = dict(p.flake_chips)
        p.receipt_inputs = {"project_id": p.project_id, "requirements_event_id": record["event_id"],
            "chip_id": p.target_chip_id, "top_first_flake_ids": p.flake_ids,
            "layer_order": ["hBN", req.reactive_material, req.metal, "silicon"],
            "stamp_id": p.stamp_id, "stamp_film": film, "source_chip_ids": p.source_chip_ids,
            "metal_coating": store.get_sample(p.target_chip_id)["props"]["metal_coating"],
            "source_images": {fid: store.get_sample(fid)["props"]["images"] for fid in p.flake_ids},
            "protocol_id": SIM_PROTOCOL, "scope": "device_simulation", "model_assumptions": profile.model_dump(),
            "simulated_motion_verified": True, "physical_motion_verified": False, "chemistry_verified": False}
        return p

    def reserve(self, params, store, run_id):
        super().reserve(params, store, run_id)
        store.reserve("slot", params.reactor_home_slot, run_id, self.name)
        for fid in params.flake_ids:
            store.reserve("sample", fid, run_id, self.name)

    def after_stack(self, params):
        return [VerifyDevicePreparation(params)]

    def build_workflow(self, p):
        shared = p.home_slots[p.target_chip_id]
        molecules = [Molecule("admit_and_handover", "Check the frozen task and hand the metal-coated carrier to transfer storage", (
            CheckDeviceContract(), MoveObject(p.target_chip_id, "chip", arm="arm_reactor", camera="overview_camera",
                source=p.reactor_home_slot, dest=shared, cfg_key="store_chip", name="handover_material_metal_carrier")))]
        # Reuse the actual source-based transfer checkpoints. Damaged film
        # holds for human review; source identity cannot exclude partial transfer.
        for part in super().build_workflow(p).molecules:
            steps = []
            for step in part.steps:
                if isinstance(step, PlaceChipOnTransferStage):
                    step.source = p.home_slots[step.chip_id]
                if isinstance(step, PickUpFlake):
                    step = DevicePickup(step.flake_id, step.stamp_id, step.material, name=step.name)
                elif isinstance(step, DropDown):
                    step = DeviceRelease(step.stamp_id, step.flake_ids, step.materials, step.target_chip, step.target_xy, step.melt_c, None)
                steps.append(step)
            molecules.append(Molecule(part.name, part.description, tuple(steps)))
        molecules += [Molecule("return_prepared_carrier", "Return the verified prepared carrier to dedicated reactor storage", (
            MoveObject(p.target_chip_id, "chip", arm="arm_reactor", camera="overview_camera",
                source=shared, dest=p.reactor_home_slot, cfg_key="store_chip", name="return_material_metal_carrier",
                status="material_metal_prepared"),)), record_terminal(p.target_chip_id, cold_storage=True)]
        return Assembly(self.name, "Prepare hBN / reactive material / metal / silicon with guarded transfer", tuple(molecules))

    def build_steps(self, p):
        return self.build_workflow(p).build_steps()


def device_flow(bench):
    flow = DevicePrepareFlow()
    flow._bench = bench
    return flow


def resolve_reaction(bench, params, store):
    p = params.model_copy(deep=True)
    errors = device_preparation_errors(store, p.project_id, p.chip_id)
    chip = store.get_sample(p.chip_id) or {}
    goal = (store.get_project(p.project_id) or {}).get("state", {}).get("reactor_goal") or {}
    if goal.get("evidence_scope") != "device_simulation" or goal.get("preparation") != chip.get("props", {}).get("material_metal_preparation"):
        errors.append("Bind the current simulation objective through Reactor task delegation")
    if (slot_station(bench, chip.get("location")) != "reactor_storage" or chip.get("props", {}).get("reactor_state_uncertain")
            or any(s["location"] == "reactor_port" for s in store.list_samples())
            or "reactor_port" in store.reservations("slot") or p.chip_id in store.reserved_samples()):
        errors.append("Start from a known stored carrier with an available Reactor port")
    profile = simulation_profile(bench)
    if goal.get("preparation", {}).get("model_assumptions") != profile.model_dump():
        errors.append("Simulation assumptions changed since preparation")
    if errors:
        raise ValueError("; ".join(errors))
    p.goal, p.requirements_event_id, p.home_slot = goal, goal["requirements_event_id"], chip["location"]
    for key in ("temperature_c", "flow_sccm", "duration_s"):
        if getattr(p, key) is None:
            setattr(p, key, getattr(profile, key))
    return p


class MaterialMetalConditions(SetReactorConditions):
    def preconditions(self, ctx):
        require_contract(ctx)
        if ctx.store.get_project(ctx.project_id)["state"].get("reactor_goal") != ctx.params["goal"]:
            raise PreconditionFailed("Reactor goal changed before exposure")
        super().preconditions(ctx)


class MaterialMetalWindow(GrowthWindow):
    name = "observe_material_metal_progress"

    def preconditions(self, ctx):
        require_contract(ctx)
        super().preconditions(ctx)


def reaction_workflow(bench, p):
    recipe = GrowthParams(chip_id=p.chip_id, temperature_c=p.temperature_c, flow_sccm=p.flow_sccm,
                          duration_s=p.duration_s, observation_interval_s=p.observation_interval_s)
    molecules = [Molecule("check_contract", "Check the original requirements and preparation before execution", (CheckDeviceContract(),)),
        load_and_acquire(p.chip_id, p.home_slot, bench["reactor_module"]["geometry"]["process_y_mm"]),
        observe_carrier(p.chip_id, name="observe_before_exposure"),
        establish_conditions(MaterialMetalConditions(recipe)), observe_exposure(MaterialMetalWindow(recipe)),
        shutdown_and_cool(), observe_carrier(p.chip_id, name="observe_after_cooling")]
    cold = p.goal["final_disposition"] == "cold_storage"
    if cold:
        molecules.append(store_carrier(p.chip_id, p.home_slot, step_name="store_material_metal_carrier"))
    molecules.append(record_terminal(p.chip_id, cold_storage=cold))
    return Assembly("reactor_material_metal", "Process and observe the prepared carrier, then verify the declared final disposition", tuple(molecules))


def reaction_steps(bench, p):
    return reaction_workflow(bench, p).build_steps()
