"""Read-only look at the bench with the overview camera (eyes' scan_bench and
check_stage_obstruction tools; v1 check_stage_obstruction_tool.py, which read
the database instead of looking). Runs in the job service because the job
service owns the cameras; it moves nothing and locks nothing.

Poses are in the world frame. Objects carry no id (only the storage boards
have QR markers), so each detection says which named place it sits at
("at": a slot or a stage), and the ledger says what that place holds.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from qumus_simlab.core.types import Pose
from qumus_simlab.flows.bench import positions, regions
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.flow import Flow, register_flow
from qumus_simlab.flows.library.primitives import OVERVIEW, look
from qumus_simlab.flows.step import Correction, Observation, ProbeResult, Step, Verdict
from qumus_simlab.records.lab_geometry import LabEvidence, Source, record_evidence

#: which part of the bench each side covers (v1 had a left and a right camera;
#: one overview camera now sees all of it)
SIDE_REGIONS = {
    "left": ["left_storage", "exfoliation_stage", "tape", "middle_storage"],
    "right": ["middle_storage", "transfer_stage", "stamp_handoff", "right_storage"],
}
ALL_REGIONS = ["left_storage", "exfoliation_stage", "tape", "middle_storage", "transfer_stage",
               "stamp_handoff", "right_storage", "reactor"]
#: a detection this close (mm) to a named place is "at" it
AT_MM = 4.0
#: frames taken again before a stage is reported empty (a wrong "empty" here
#: makes the orchestrator plan around a chip that is still on the stage)
RECHECK_EMPTY = 3
#: the stage that carries its chip around: its region in bench.yaml is the
#: load position; the look follows the stage to where it is now
MOVING_STAGE = "transfer_stage"


class InspectBench(Step):
    name = "inspect_bench"

    def __init__(self, regions: list[str], camera: str = OVERVIEW) -> None:
        self.regions, self.camera = regions, camera

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        pass  # looking happens in verify()

    def _at(self, ctx: FlowContext, pose: Pose, places: dict) -> str | None:
        best = min(((Pose(x=p["x"], y=p["y"]).xy_distance(pose), n)
                    for n, p in places.items()), default=None)
        return best[1] if best and best[0] <= AT_MM else None

    @staticmethod
    def _stage_offset(ctx: FlowContext) -> tuple[float, float]:
        """How far the moving stage is from its load position now (a read)."""
        try:
            p = ctx.gate.stage(MOVING_STAGE).get_sample_pose()
        except Exception:  # noqa: BLE001 - no such stage on this bench: it does not move
            return 0.0, 0.0
        load = ctx.bench.get(MOVING_STAGE, {}).get("load", {})
        return p.x - float(load.get("x", 0.0)), p.y - float(load.get("y", 0.0))

    def verify(self, ctx: FlowContext) -> Observation:
        out: dict[str, list[dict]] = {}
        imgs: list[str] = []
        places = positions(ctx.bench)
        dx, dy = self._stage_offset(ctx)
        stage_now = None
        if MOVING_STAGE in places:
            # the chip rides on the stage: "transfer_stage" is where the stage is
            stage_now = {"x": places[MOVING_STAGE]["x"] + dx, "y": places[MOVING_STAGE]["y"] + dy}
            places = {**places, MOVING_STAGE: stage_now}
        carrier_now = None
        if "reactor_module" in ctx.bench:
            p = ctx.gate.stage("reactor_slot").get_pose()
            home = places["reactor_port"]
            carrier_now = {"x": home["x"] + p.x, "y": home["y"] + p.y + ctx.gate.stage("reactor_insert").get_pose().y}
            places = {**places, "reactor_port": carrier_now}
        for region in self.regions:
            if region in out:
                continue
            is_stage = region in ctx.bench.get("stations", {})
            dets = look(ctx, self.camera, region, "world", images=imgs, views=3,
                        recheck_empty=RECHECK_EMPTY if is_stage else 0)
            if region == MOVING_STAGE and stage_now is not None:
                dets = self._stage_elsewhere(ctx, dets, stage_now, imgs)
            if region == "reactor_port" and carrier_now is not None:
                dets = self._stage_elsewhere(ctx, dets, carrier_now, imgs, region)
            out[region] = [{"label": d.label, "at": (self._at(ctx, d.pose, places) or
                              (MOVING_STAGE if region == MOVING_STAGE and stage_now is not None
                               and self._inside_stage(ctx, d.pose, stage_now) else None)),
                            "pose": d.pose.model_dump(), "confidence": d.confidence,
                            **d.props} for d in dets]
            record_evidence(ctx.store, LabEvidence(
                subject=f"region:{region}", property="overview_observation", bench=ctx.store.bench,
                value={"detections": out[region], "images": list(dict.fromkeys(imgs)),
                       "identity": "object kind/position only; sample identity remains reconciled ledger data"},
                units="mm,deg", frame="world", observed_at=ctx.clock.now(), timebase="store_clock",
                source=Source(kind="synthetic" if ctx.store.bench == "sim" else "measured",
                              reference=f"inspect_bench:{self.camera}:{region}")),
                run_id=ctx.run_id, supersede_previous=True)
        m = {"regions": out, "frame": "world", "pose_units": {"position": "mm", "rotation": "deg"},
             "empty": [r for r, d in out.items() if not d],
             "coverage": {"observed_regions": list(out),
                 "unobserved_reactor_regions": [r for r in ("reactor", "reactor_storage", "reactor_port")
                                                if "reactor_module" in ctx.bench and r not in out],
                 "scope": "Empty applies only to the observed region. Reactor chamber, port and storage are separate.",
                 "identity": "Detections identify object kind and occupied slot; sample identity comes from reconciled inventory."}}
        if stage_now is not None and MOVING_STAGE in out:
            m["stage_now"] = {k: round(v, 2) for k, v in stage_now.items()}
        return Observation(Verdict.OK, m, images=list(dict.fromkeys(imgs)))

    def _stage_elsewhere(self, ctx: FlowContext, dets: list, stage_now: dict,
                         imgs: list[str], moving_place: str = MOVING_STAGE) -> list:
        """The stage has moved out of its load region (under the objective, say):
        look where it is now as well, so a chip riding on it counts as on the
        stage instead of the stage counting as empty."""
        regs = regions(ctx.bench)
        box = regs.get(moving_place, {})
        if moving_place == MOVING_STAGE:
            dets = [d for d in dets if self._inside_stage(ctx, d.pose, stage_now)]
        inside = (box and stage_now["x"] == sum(box["x"])/2
                  and stage_now["y"] == sum(box["y"])/2
                  and (moving_place != MOVING_STAGE or
                       ctx.gate.stage(MOVING_STAGE).get_sample_pose().rotation == 0))
        if inside:
            return dets
        there = ("workspace" if moving_place == MOVING_STAGE else next(
                     (n for n, r in regs.items() if n not in ("workspace", moving_place)
                      and r["x"][0] <= stage_now["x"] <= r["x"][1]
                      and r["y"][0] <= stage_now["y"] <= r["y"][1]), "workspace"))
        extra = look(ctx, self.camera, there, "world", images=imgs, views=3,
                     recheck_empty=RECHECK_EMPTY)
        spot = Pose(x=stage_now["x"], y=stage_now["y"])
        seen = {(round(d.pose.x), round(d.pose.y)) for d in dets}
        return dets + [d for d in extra if (self._inside_stage(ctx, d.pose, stage_now)
                      if moving_place == MOVING_STAGE else d.pose.xy_distance(spot) <= AT_MM)
                       and (round(d.pose.x), round(d.pose.y)) not in seen]

    @staticmethod
    def _inside_stage(ctx: FlowContext, pose: Pose, centre: dict) -> bool:
        """Camera occupancy of the full translated/rotated stage footprint.

        This binds a detected kind/position to a region, never a sample ID or
        simulator object. Off-centre mounting is not a misplaced/absent chip.
        """
        box = regions(ctx.bench)[MOVING_STAGE]
        theta = math.radians(ctx.gate.stage(MOVING_STAGE).get_sample_pose().rotation)
        dx, dy = pose.x-centre["x"], pose.y-centre["y"]
        x, y = dx*math.cos(theta)+dy*math.sin(theta), -dx*math.sin(theta)+dy*math.cos(theta)
        return abs(x) <= (box["x"][1]-box["x"][0])/2 and abs(y) <= (box["y"][1]-box["y"][0])/2

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        return ProbeResult.NOT_DONE


class InspectParams(BaseModel):
    regions: list[str] | None = Field(None, description="Bench regions, e.g. "
                                      "exfoliation_stage, transfer_stage, middle_storage, "
                                      "left_chip_3, reactor_storage, reactor_port; reactor is the chamber, not storage. "
                                      "Omitted explicit regions remain unobserved; None = the whole side (see camera)")
    camera: Literal["overview", "left", "right", "both"] = Field(
        "overview", description="One overview camera sees the whole table; 'left'/'right' "
        "only pick which half to scan when no regions are given")


@register_flow
class InspectBenchFlow(Flow):
    name = "inspect_bench"
    description = "Look at bench regions with the overview camera; returns detections per region."
    Params = InspectParams
    devices = ()  # read-only: never waits for another job's devices

    def build_steps(self, params: BaseModel) -> list[Step]:
        assert isinstance(params, InspectParams)
        regions = list(params.regions or SIDE_REGIONS.get(params.camera, ALL_REGIONS))
        if not params.regions and params.camera != "left" and "reactor_module" in self.bench:
            regions += ["reactor_storage", "reactor_port"]
        return [InspectBench(regions)]
