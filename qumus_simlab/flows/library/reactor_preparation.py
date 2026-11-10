"""Checked catalog-to-reactor preparation using the existing dry-transfer steps.

The narrow carrier travels through a shared storage slot using both arms. No
ledger relocation stands in for transport or physical pickup/drop-down.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.flows.bench import arm_positions, reachable, slot_station, slots, within_reach
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.flow import Flow, register_flow
from qumus_simlab.flows.library.primitives import MICROSCOPE, analyzer, autofocus
from qumus_simlab.flows.library.reactor import RecordReactorProcess
from qumus_simlab.flows.library.search import MoveTransferStage
from qumus_simlab.flows.library.stacking import (
    CoolTransferStage,
    DropDown,
    GoToTarget,
    RefindFlake,
    StackDeviceFlow,
    StackParams,
)
from qumus_simlab.flows.library.transfer import MoveObject
from qumus_simlab.flows.step import Observation, ProbeResult, Step, Verdict
from qumus_simlab.flows.workflow import Assembly, Molecule
from qumus_simlab.records.reactor_pairs import PairPreparation, prepared_pair


def shared_storage(bench: dict[str, Any], store: Any) -> str:
    used = store.occupied_slots()
    choices = [slot for slot in slots(bench, "right_storage", "chip") if slot not in used
               and all(slot in reachable(bench, arm)
                       and within_reach(bench, arm, arm_positions(bench, arm)[slot])
                       for arm in ("arm_right", "arm_reactor"))]
    if not choices:
        raise ValueError("No free shared chip slot reachable by both Arm2 and Arm3")
    return choices[0]


class PrepareReactorPairParams(StackParams):
    flake_ids: list[str] = Field(min_length=2, max_length=2)
    reactor_home_slot: str | None = None


class AssessReactorPickup(Step):
    """Inspect the full conservative contact neighbourhood before moving flakes."""

    devices = (MICROSCOPE, "transfer_stage")

    def __init__(self, flake_id: str, chip_id: str, material: str, *, require_clear: bool = False) -> None:
        self.flake_id, self.chip_id, self.material, self.require_clear = flake_id, chip_id, material, require_clear
        self.name = f"assess_reactor_pickup_{flake_id}"

    def act(self, ctx: FlowContext, correction: Any) -> None:
        from qumus_simlab.flows.library.pickup_evidence import capture_neighbourhood

        self.measurements = capture_neighbourhood(ctx, self.flake_id, self.chip_id, self.material, local=False)
        self.images = self.measurements["images"]

    def verify(self, ctx: FlowContext) -> Observation:
        ok = self.measurements["clear"] or not self.require_clear
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR, self.measurements, images=self.images,
            message="Pickup neighbourhood inspected" if ok else "Nearby flakes could also transfer; select an isolated candidate")

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        ctx.store.upsert_sample(self.flake_id, "flake", props={
            "pickup_assessment": {**self.measurements, "job_id": ctx.run_id, "images": self.images}})
        ctx.store.emit("reactor.pickup_assessed", self.measurements, run_id=ctx.run_id)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


class AssessReactorInputsParams(BaseModel):
    chip_id: str
    flake_ids: list[str] = Field(min_length=1, max_length=20)
    home_slots: dict[str, str] = Field(default_factory=dict)
    flake_chips: dict[str, str] = Field(default_factory=dict)
    materials: dict[str, str] = Field(default_factory=dict)


@register_flow
class AssessReactorInputsFlow(Flow):
    name = "assess_reactor_inputs"
    Params = AssessReactorInputsParams
    devices = (*StackDeviceFlow.devices, "reactor_heater", "reactor_flow", "reactor_slot", "reactor_insert")

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        p = params.model_copy(deep=True)
        if "reactor_module" not in self.bench:
            raise ValueError("Reactor is disabled")
        carrier = store.get_sample(p.chip_id)
        if not carrier or not carrier["props"].get("reactor_substrate") \
                or slot_station(self.bench, carrier["location"]) != "reactor_storage":
            raise ValueError("Choose the dedicated carrier for this preparation task")
        if len(set(p.flake_ids)) != len(p.flake_ids):
            raise ValueError("Choose distinct catalogued inputs")
        for fid in p.flake_ids:
            flake = store.get_sample(fid)
            if not flake or flake["kind"] != "flake" or flake["status"] not in {"catalogued", "selected"}:
                raise ValueError("Assess only catalogued or selected source flakes")
            chip = store.get_sample(flake["location"])
            if not chip or chip["location"] not in reachable(self.bench, "arm_right") \
                    or not flake["props"].get("images") or fid in store.reserved_samples() \
                    or chip["id"] in store.reserved_samples():
                raise ValueError("Source must be unreserved, photographed and reachable by Arm2")
            p.flake_chips[fid], p.materials[fid] = chip["id"], flake["props"]["material"]
            p.home_slots[chip["id"]] = chip["location"]
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        for fid in [*params.flake_ids, *params.home_slots]:
            store.reserve("sample", fid, run_id, self.name)
        store.reserve("sample", params.chip_id, run_id, self.name)
        for slot in params.home_slots.values():
            store.reserve("slot", slot, run_id, self.name)

    def build_workflow(self, params: BaseModel) -> Assembly:
        p = params
        molecules = [Molecule("cool_transfer_stage", "Cool the ordinary transfer stage before source inspection", (CoolTransferStage(),))]
        for chip, slot in p.home_slots.items():
            steps = StackDeviceFlow._chip_in(chip, f"assess_{chip}")
            for fid in p.flake_ids:
                if p.flake_chips[fid] == chip:
                    steps += [RefindFlake(fid, chip, p.materials[fid]), AssessReactorPickup(fid, chip, p.materials[fid])]
            steps += [MoveTransferStage("load"), MoveObject(chip, "chip", arm="arm_right", camera="overview_camera",
                        source="transfer_stage", dest=slot, cfg_key="store_chip", release_switch="transfer_stage_vacuum",
                        name=f"return_assessed_source_{chip}")]
            molecules.append(Molecule(f"assess_source_{chip}", "Load the source, refind and inspect pickup neighbourhoods, then return it", tuple(steps)))
        molecules.append(Molecule("record_terminal", "Verify the reactor carrier remains in cold storage", (RecordReactorProcess(p.chip_id, cold_storage=True),)))
        return Assembly(self.name, "Inspect screened inputs without transferring any flake", tuple(molecules))



class ReactorDropDown(DropDown):
    """Reuse all guarded transfer and verification, without inventing a device."""

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        self._admit_release_commit(ctx, obs)
        for fid in self.flake_ids:
            ctx.store.upsert_sample(fid, "flake", location=self.target_chip, status="stacked")
        ctx.store.upsert_sample(self.stamp_id, "stamp", props={"film": None, "stack": []})


class VerifyReactorPreparation(Step):
    name = "verify_reactor_preparation"
    devices = (MICROSCOPE, "transfer_stage")

    def __init__(self, chip_id: str, top_first: list[str]) -> None:
        self.chip_id, self.bottom_first = chip_id, list(reversed(top_first))

    def act(self, ctx: FlowContext, correction: Any) -> None:
        autofocus(ctx, ctx.gate.microscope(MICROSCOPE).get_objective())

    def verify(self, ctx: FlowContext) -> Observation:
        samples = [ctx.store.get_sample(fid) for fid in self.bottom_first]
        if any(s is None or s["location"] != self.chip_id or s["status"] != "stacked" for s in samples):
            return Observation(Verdict.UNCLEAR, message="Both selected flakes must be checked on the carrier")
        perceive = getattr(analyzer(ctx), "reactor_preparation", None)
        if perceive is None:
            raise PreconditionFailed("The microscope preparation-verification adapter is unavailable")
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        try:
            manifest, updates = perceive(frame, samples, preparation_id=ctx.run_id)
            pair = PairPreparation.model_validate(manifest)
            if pair.chip_id != self.chip_id or list(pair.input_flake_ids) != self.bottom_first \
                    or pair.preparation_source != "stacking_job":
                raise ValueError("Optical preparation does not match this carrier and selected pair")
        except ValueError as exc:
            return Observation(Verdict.UNCLEAR, message=str(exc))
        self.manifest, self.updates = manifest, updates
        image = frame.saved_path()
        return Observation(Verdict.OK, {"preparation": manifest, "input_flake_ids": self.bottom_first},
                           images=[image] if image else [])

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        with ctx.store._transaction():
            if set(self.updates) != set(self.bottom_first):
                raise ValueError("Preparation updates must cover exactly the selected pair")
            for fid in self.bottom_first:
                ctx.store.upsert_sample(fid, "flake", props=self.updates[fid])
            ctx.store.upsert_sample(self.chip_id, "chip", status="reactor_pair_prepared",
                props={"reactor_pair": self.manifest,
                       "preparation": "Two catalogued flakes transferred by a checked stacking job"})
            prepared_pair(ctx.store, self.chip_id, run_id=ctx.run_id)
            ctx.store.emit("reactor.prepared", {"chip_id": self.chip_id, **self.manifest,
                           "images": obs.images if obs else []}, run_id=ctx.run_id)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


@register_flow
class PrepareReactorPairFlow(StackDeviceFlow):
    name = "prepare_reactor_pair"
    Params = PrepareReactorPairParams
    description = "Route a narrow carrier, stack exactly two catalogued flakes, verify preparation and return it."
    devices = (*StackDeviceFlow.devices, "arm_reactor", "reactor_heater", "reactor_flow", "reactor_slot", "reactor_insert")

    def target_storage(self, target: dict[str, Any], store: Any) -> str:
        if not target["props"].get("reactor_substrate"):
            raise ValueError("Choose a dedicated narrow reactor carrier")
        if slot_station(self.bench, target["location"]) != "reactor_storage":
            raise ValueError("The unprepared carrier must start in dedicated reactor storage")
        if target["props"].get("reactor_pair") or store.list_samples(kind="flake", location=target["id"]):
            raise ValueError("The reactor carrier must be empty and unprepared")
        if target["id"] in store.reserved_samples():
            raise ValueError("The reactor carrier is reserved")
        expected = self.bench["reactor_module"]["geometry"]["chip_size_mm"]
        if target["props"].get("size_mm") != expected:
            raise ValueError("The carrier dimensions do not match the narrow reactor slot")
        return shared_storage(self.bench, store)

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        if "reactor_module" not in self.bench:
            raise ValueError("The reactor is disabled")
        p = super().resolve_params(params, store)
        assert isinstance(p, PrepareReactorPairParams)
        for fid in p.flake_ids:
            sample = store.get_sample(fid)
            if fid in store.reserved_samples() or sample["props"].get("reactor_consumed_by"):
                raise ValueError("A selected input is reserved or already consumed")
            if not sample["props"].get("images"):
                raise ValueError("Selected flakes need saved search/revisit images")
            assessment = sample["props"].get("pickup_assessment") or sample["props"].get("reactor_pickup_assessment") or {}
            if not assessment.get("clear") or assessment.get("source_chip_id") != sample["location"]:
                raise ValueError("Assess the optical pickup neighbourhood and choose two isolated candidates first")
        p.reactor_home_slot = store.get_sample(p.target_chip_id)["location"]
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        super().reserve(params, store, run_id)
        store.reserve("slot", params.reactor_home_slot, run_id, self.name)
        for fid in params.flake_ids:
            store.reserve("sample", fid, run_id, self.name)

    def build_workflow(self, params: BaseModel):
        # This historical subclass has its own additional flat transport steps.
        # Do not mislabel the inherited dry-transfer hierarchy as its full job.
        return None

    def after_stack(self, params: StackParams) -> list[Step]:
        return [VerifyReactorPreparation(params.target_chip_id, params.flake_ids)]

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, PrepareReactorPairParams)
        shared = params.home_slots[params.target_chip_id]
        steps = super().build_steps(params)
        for i, step in enumerate(steps):
            if isinstance(step, DropDown):
                steps[i] = ReactorDropDown(step.stamp_id, step.flake_ids, step.materials,
                    step.target_chip, step.target_xy, step.melt_c, step.design_path)
        checked = []
        for step in steps:
            checked.append(step)
            if isinstance(step, RefindFlake):
                checked.append(AssessReactorPickup(step.flake_id, step.chip_id, step.material, require_clear=True))
        steps = checked
        return [MoveObject(params.target_chip_id, "chip", arm="arm_reactor", camera="overview_camera",
                           source=params.reactor_home_slot, dest=shared, cfg_key="store_chip",
                           name="handover_reactor_carrier_to_arm2"), *steps,
                MoveObject(params.target_chip_id, "chip", arm="arm_reactor", camera="overview_camera",
                           source=shared, dest=params.reactor_home_slot, cfg_key="store_chip",
                           name="return_prepared_reactor_carrier", status="reactor_pair_prepared"),
                RecordReactorProcess(params.target_chip_id, cold_storage=True)]


class InspectReactorProductParams(BaseModel):
    chip_id: str
    shared_slot: str | None = None
    home_slot: str | None = None
    product_id: str | None = None
    chip_size_mm: list[float] = Field(default_factory=list)


class ObserveReactorProduct(Step):
    name = "observe_product_on_ordinary_microscope"
    devices = (MICROSCOPE,)

    def __init__(self, chip_id: str, product_id: str) -> None:
        self.chip_id, self.product_id = chip_id, product_id

    def act(self, ctx: FlowContext, correction: Any) -> None:
        autofocus(ctx, ctx.gate.microscope(MICROSCOPE).get_objective())

    def verify(self, ctx: FlowContext) -> Observation:
        product = ctx.store.get_sample(self.product_id)
        if not product or product["location"] != self.chip_id or product["status"] != "reactor_product":
            return Observation(Verdict.UNCLEAR, message="The accepted product is not on this carrier")
        frame = ctx.gate.microscope(MICROSCOPE).capture()
        seen = analyzer(ctx).find_flakes(frame, product["props"]["material"])
        ok = analyzer(ctx).chip_in_view(frame) and any(
            d.meta.get("sim_id") == self.product_id if ctx.store.bench == "sim" else True for d in seen)
        image = frame.saved_path()
        return Observation(Verdict.OK if ok else Verdict.UNCLEAR,
            {"chip_id": self.chip_id, "product_id": self.product_id, "product_visible": ok,
             "source_flake_ids": product["props"]["source_flake_ids"]}, images=[image] if image else [],
            message="Accepted product inspected on ordinary optics" if ok else "Product is not optically verified")

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        product = ctx.store.get_sample(self.product_id)
        images = dict(product["props"].get("images") or {})
        if obs and obs.images:
            images["ordinary_microscope"] = obs.images[-1]
        ctx.store.upsert_sample(self.product_id, "flake", props={"images": images})
        ctx.store.emit("reactor.product_inspected", {"chip_id": self.chip_id, "product_id": self.product_id,
                       "images": obs.images if obs else []}, run_id=ctx.run_id)


@register_flow
class InspectReactorProductFlow(Flow):
    name = "inspect_reactor_product"
    Params = InspectReactorProductParams
    devices = PrepareReactorPairFlow.devices

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        assert isinstance(params, InspectReactorProductParams)
        p = params.model_copy(deep=True)
        chip = store.get_sample(p.chip_id)
        if not chip or not chip["props"].get("reactor_substrate") \
                or slot_station(self.bench, chip["location"]) != "reactor_storage":
            raise ValueError("Recover the accepted product carrier to dedicated storage first")
        if chip["props"].get("reactor_state_uncertain") or p.chip_id in store.reserved_samples():
            raise ValueError("The carrier is uncertain or reserved")
        products = [s for s in store.list_samples(kind="flake", location=p.chip_id) if s["status"] == "reactor_product"]
        if len(products) != 1:
            raise ValueError("Choose a carrier with exactly one accepted reactor product")
        p.product_id, p.home_slot = products[0]["id"], chip["location"]
        p.chip_size_mm = chip["props"]["size_mm"]
        p.shared_slot = shared_storage(self.bench, store)
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        for sid in (params.chip_id, params.product_id):
            store.reserve("sample", sid, run_id, self.name)
        for slot in (params.home_slot, params.shared_slot):
            store.reserve("slot", slot, run_id, self.name)

    def build_steps(self, params: BaseModel) -> list[Step]:
        p = params
        return [RecordReactorProcess(p.chip_id, cold_storage=True), CoolTransferStage(),
                MoveObject(p.chip_id, "chip", arm="arm_reactor", camera="overview_camera", source=p.home_slot,
                           dest=p.shared_slot, cfg_key="store_chip", name="handover_product_carrier_to_arm2"),
                *StackDeviceFlow._chip_in(p.chip_id, "reactor_product"),
                GoToTarget(p.chip_id, tuple(x / 2 for x in p.chip_size_mm)),
                ObserveReactorProduct(p.chip_id, p.product_id), MoveTransferStage("load"),
                MoveObject(p.chip_id, "chip", arm="arm_right", camera="overview_camera", source="transfer_stage", dest=p.shared_slot,
                           cfg_key="store_chip", release_switch="transfer_stage_vacuum", name="return_product_to_shared_storage"),
                MoveObject(p.chip_id, "chip", arm="arm_reactor", camera="overview_camera", source=p.shared_slot,
                           dest=p.home_slot, cfg_key="store_chip", name="return_product_to_reactor_storage"),
                RecordReactorProcess(p.chip_id, cold_storage=True)]
