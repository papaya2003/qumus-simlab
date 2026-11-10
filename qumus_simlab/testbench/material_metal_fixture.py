"""Abstract workflow fixture, NOT a device/transfer/chemistry simulator.

No hardware or SimWorld objects are moved. The persisted fixture state tests
ordered operations, interruption/resume and the generic evidence pipeline.
Its source images stay attributed to their original catalogue observations.
"""
from __future__ import annotations

import json
from copy import deepcopy

from qumus_simlab.records.material_metal import FIXTURE_PROTOCOL


class SyntheticMaterialMetalAdapter:
    def observe(self, ctx, chip_id: str) -> dict:
        row = ctx.store._x("SELECT payload_json FROM event WHERE kind='material_metal.fixture' AND "
            "json_extract(payload_json,'$.project_id')=? AND json_extract(payload_json,'$.chip_id')=? "
            "ORDER BY id DESC LIMIT 1", (ctx.project_id, chip_id)).fetchone()
        return deepcopy(json.loads(row["payload_json"])["state"]) if row else {}

    def apply(self, ctx, action: str, payload: dict) -> dict:
        if ctx.store.bench != "sim":
            raise ValueError("Workflow fixture cannot run on physical hardware")
        chip_id = payload["chip_id"]
        state = self.observe(ctx, chip_id)
        if action == "initialize":
            if state:
                if state.get("preparation_job_id") != ctx.run_id:
                    raise ValueError("The fixture carrier already belongs to another preparation")
                return state
            state = {**deepcopy(payload), "protocol_id": FIXTURE_PROTOCOL, "scope": "synthetic_workflow",
                "preparation_job_id": ctx.run_id, "picked_ids": [], "released": False,
                "sequence": [], "physical_motion_verified": False, "chemistry_verified": False}
        elif action == "pickup":
            fid = payload["flake_id"]
            if fid in state.get("picked_ids", []):
                return state
            order = state.get("top_first_flake_ids", [])
            if len(state.get("picked_ids", [])) >= len(order) or fid != order[len(state["picked_ids"])]:
                raise ValueError("Fixture pickup must preserve the declared hBN-first order")
            state["picked_ids"].append(fid)
            state["sequence"].append({"action": action, "flake_id": fid})
        elif action == "release":
            if state.get("picked_ids") != state.get("top_first_flake_ids") or len(state.get("picked_ids", [])) != 2:
                raise ValueError("Both inputs must be picked up before fixture release")
            if not state.get("released"):
                state["released"] = True
                state["sequence"].append({"action": action, "target": chip_id})
        elif action == "observe":
            if not state.get("released"):
                raise ValueError("Cannot observe an unprepared fixture")
            if state.get("observation_job_id") == ctx.run_id:
                return state
            state["observation_job_id"] = ctx.run_id
            state["observable"] = "workflow_sequence_receipt"
            state["sequence"].append({"action": action, "job_id": ctx.run_id})
        else:
            raise ValueError("Unregistered workflow fixture action")
        ctx.store.emit("material_metal.fixture", {"project_id": ctx.project_id, "chip_id": chip_id,
            "action": action, "state": state, "source": "abstract_software_fixture_not_device_execution"}, run_id=ctx.run_id)
        return self.observe(ctx, chip_id)
