"""First worked example of a closed-loop step (spec 4.1, 13):
"arm puts a chip on the transfer stage -> camera verifies -> correct".

v1 did this in right_arm_atoms.AtomRightArmPlaceSiliconChipOnTransferStage and
returned "✅" unconditionally (spec 4.1). Here every placement is measured by
the overview camera and re-placed if it is off (MoveObject, flows/library/transfer.py).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from qumus_simlab.flows.bench import reachable, slots
from qumus_simlab.flows.flow import Flow, register_flow
from qumus_simlab.flows.library.transfer import MoveObject
from qumus_simlab.flows.step import Observation, Step

ARM = "arm_right"
#: the arm that brings a chip from the left board to the middle one
FEED_ARM = "arm_left"
HANDOVER_BOARD = "middle_storage"
CAMERA = "overview_camera"


class PlaceChipOnTransferStage(MoveObject):
    def __init__(self, chip_id: str) -> None:
        super().__init__(chip_id, "chip", arm=ARM, camera=CAMERA, dest="transfer_stage",
                         cfg_key="place_chip_on_transfer_stage",
                         name="place_chip_on_transfer_stage")
        self.chip_id = chip_id

    def ledger_props(self, obs: Observation | None) -> dict[str, Any]:
        # a new placement: the microscope-measured rotation of the last one is void.
        # The pose is in the world frame (MoveObject reports world poses).
        return {"pose_on_stage": obs.measurements.get("pose"), "rotation_deg": None} if obs else {}


class PlaceChipParams(BaseModel):
    chip_id: str = Field(description="Chip id from the sample ledger, e.g. chip_001")
    # filled in at submit: a chip on the left board is first handed over to
    # arm_right through this free slot on the middle board
    handover_slot: str | None = None


@register_flow
class PlaceChipFlow(Flow):
    name = "place_chip_on_transfer_stage"
    description = ("Move a chip from its slot to the transfer stage, verified by camera "
                   "(a chip on the left board goes over the middle board first).")
    Params = PlaceChipParams
    devices = (ARM, FEED_ARM, CAMERA, "transfer_stage")

    def resolve_params(self, params: BaseModel, store: Any) -> BaseModel:
        assert isinstance(params, PlaceChipParams)
        p = params.model_copy()
        bench = self.bench
        chip = store.get_sample(p.chip_id)
        p.handover_slot = None
        # an unknown or unreachable chip is reported by the step's precondition
        if chip is None or chip["kind"] != "chip":
            return p
        here = chip["location"]
        if here in reachable(bench, ARM) or here not in reachable(bench, FEED_ARM):
            return p
        used = store.occupied_slots()
        free = [n for n in slots(bench, HANDOVER_BOARD, "chip") if n not in used]
        if not free:
            raise ValueError(f"no free slot on the {HANDOVER_BOARD} board to hand "
                             f"{p.chip_id} over")
        p.handover_slot = free[0]
        return p

    def reserve(self, params: BaseModel, store: Any, run_id: str) -> None:
        assert isinstance(params, PlaceChipParams)
        if params.handover_slot:
            store.reserve("slot", params.handover_slot, run_id, self.name,
                          note=f"{params.chip_id} handed over here")

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, PlaceChipParams)
        steps: list[Step] = []
        if params.handover_slot:
            steps.append(MoveObject(params.chip_id, "chip", arm=FEED_ARM, camera=CAMERA,
                                    dest=params.handover_slot, cfg_key="store_chip",
                                    name="hand_over_chip", fallback=(HANDOVER_BOARD, "chip"),
                                    param_path=["handover_slot"]))
        return [*steps, PlaceChipOnTransferStage(params.chip_id)]
