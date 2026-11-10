"""Report of the steps that cannot be verified (spec 4.2): the sensor wish-list.

A step whose verify() returns NOT_VERIFIABLE("reason") is never skipped
silently: it is logged as `step.verified` with verdict `not_verifiable`. This
module attempts each registered scenario on a fresh testbench, explicitly
reports unavailable observed prerequisites, collects executed steps and writes
them as a Markdown list (`qumus_simlab report`):

    step (class), the flows that use it, why it cannot be verified, how often

`collect_not_verifiable(store)` also works on a real run store, so the same
report can be made from lab runs later.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import qumus_simlab.flows.library  # noqa: F401 - registers every flow
from qumus_simlab.flows.flow import FLOWS, get_flow


class ReportPrerequisiteUnavailable(ValueError):
    """One declared scenario cannot run from its actual observed input stock."""

    def __init__(self, reason: str, *, code: str):
        super().__init__(reason)
        self.code = code


def _stack_params(store: Any) -> dict[str, Any]:
    from qumus_simlab.records.geometry import ranked_pickup_combinations

    def best(chip: str) -> list[str]:
        fl = [f for f in store.list_samples(kind="flake", location=chip)
              if f["status"] == "catalogued"]
        return [f["id"] for f in sorted(fl, key=lambda f: -f["props"]["area_um2"])]
    h, g = best("chip_002"), best("chip_001")
    ranked = ranked_pickup_combinations(store, [h, g, h])
    if not ranked:
        raise ReportPrerequisiteUnavailable(
            "No compatible observed stack; prepare additional source material before the report transfer",
            code="compatible_observed_stack_unavailable")
    return {"flake_ids": ranked[0]["flake_ids"], "target_chip_id": "chip_e01"}


#: one testbench scenario per flow, in an order that leaves the bench ready
#: for the next (exfoliate -> search -> stack). Every registered flow must be
#: here, so a new flow cannot be left out of the report (tests check it).
SCENARIOS: list[tuple[str, dict[str, Any] | Callable[[Any], dict[str, Any]]]] = [
    ("inspect_bench", {}),
    ("exfoliate_chips", {"material": "Gr", "chip_ids": ["chip_001"]}),
    ("exfoliate_chips", {"material": "hBN", "chip_ids": ["chip_002"], "target_temp_c": 90}),
    ("search_chip", {"chip_id": "chip_001"}),
    ("search_chip", {"chip_id": "chip_002"}),
    ("revisit_chip", {"chip_id": "chip_001"}),
    ("stack_device", _stack_params),
    ("place_chip_on_transfer_stage", {"chip_id": "chip_003"}),
    ("assess_reactor_inputs", lambda store: {"chip_id": "reactor_chip_01", "flake_ids": [
        s["id"] for s in store.list_samples(kind="flake") if s["status"] == "catalogued"][:8]}),
    ("prepare_reactor_pair", lambda store: {"target_chip_id": "reactor_chip_01", "flake_ids": [
        s["id"] for s in store.list_samples(kind="flake")
        if s["status"] == "catalogued" and s["props"].get("reactor_pickup_assessment", {}).get("clear")][:2]}),
    ("load_reactor_chip", {"chip_id": "reactor_chip_01"}),
    ("position_reactor_slot", {"chip_id": "reactor_chip_01", "x_mm": 0.01, "y_mm": 0}),
    ("observe_reactor", {"chip_id": "reactor_chip_01"}),
    ("hold_reactor_observation", {"chip_id": "reactor_chip_01"}),
    ("verify_reactor_state", {"chip_id": "reactor_chip_01"}),
    ("reactor_growth", {"chip_id": "reactor_chip_01", "duration_s": 60}),
    ("reactor_reaction", {"chip_id": "reactor_chip_01", "duration_s": 80}),
    ("shutdown_reactor", {}),
    ("unload_reactor_chip", {"chip_id": "reactor_chip_01"}),
    ("inspect_reactor_product", {"chip_id": "reactor_chip_01"}),
    ("prepare_material_metal", {}),
    ("reactor_material_metal", {}),
    ("cool_experiment_stages", {}),
]


def run_scenarios(services: Any, timeout_s: float = 600) -> list[dict[str, Any]]:
    """Attempt SCENARIOS; every declared scenario gets an honest execution row."""
    from qumus_simlab.testbench.render import ensure_diagnostic_images

    ensure_diagnostic_images(services)
    rows = []
    for flow, params in SCENARIOS:
        if flow in ("prepare_material_metal", "reactor_material_metal"):
            rows.append({"flow": flow, "run_id": None, "status": "disabled",
                "reason": "abstract workflow fixture requires explicit project authorization; positive job tests are in test_material_metal_workflow.py"})
            continue
        if "reactor" in flow and "reactor_module" not in services.lab.bench:
            rows.append({"flow": flow, "run_id": None, "status": "disabled",
                         "reason": "optional synthetic reactor is not enabled"})
            continue
        kind = services.lab.bench.get("reactor_module", {}).get("experiment_kind")
        if ((flow == "reactor_growth" and kind == "two_flake_reaction")
                or (flow == "reactor_reaction" and kind != "two_flake_reaction")):
            rows.append({"flow": flow, "run_id": None, "status": "disabled",
                         "reason": "this flow belongs to a different reactor experiment kind"})
            continue
        try:
            p = params(services.store) if callable(params) else params
        except ReportPrerequisiteUnavailable as error:
            rows.append({"flow": flow, "run_id": None, "status": "not_run",
                         "prerequisite": error.code, "reason": str(error)})
            continue
        st = services.jobs.wait(services.jobs.submit(flow, p), timeout_s)
        rows.append({"flow": flow, "run_id": st.run_id, "status": st.status})
    return rows


def collect_not_verifiable(store: Any) -> list[dict[str, Any]]:
    """Steps logged NOT_VERIFIABLE in `store`, grouped by step class."""
    by_class: dict[str, dict[str, Any]] = {}
    for e in store.events_after(0, limit=10_000_000):
        if e.kind != "step.verified" or e.payload.get("verdict") != "not_verifiable":
            continue
        run = store.get_run(e.run_id)
        if run is None:
            continue
        flow = get_flow(run["flow"])
        step = flow.build_steps(flow.Params.model_validate(run["params"]))[e.payload["seq"]]
        cls = type(step)
        item = by_class.setdefault(cls.__name__, {
            "step": cls.__name__, "module": cls.__module__,
            "doc": (inspect.getdoc(cls) or "").split("\n\n")[0].replace("\n", " "),
            "reasons": [], "flows": [], "step_names": [], "count": 0})
        item["count"] += 1
        for key, v in (("reasons", e.payload.get("reason", "")), ("flows", run["flow"]),
                       ("step_names", e.payload.get("name", ""))):
            if v not in item[key]:
                item[key].append(v)
    return sorted(by_class.values(), key=lambda i: i["step"])


def render_markdown(items: list[dict[str, Any]], runs: list[dict[str, Any]] | None = None) -> str:
    lines = ["# Steps that cannot be verified (sensor wish-list)", "",
             "Each step below acts without any measurement that confirms it worked "
             "(spec 4.2). The reason names the sensor that would close the gap.", ""]
    if not items:
        lines.append("No step was NOT_VERIFIABLE.")
    for i in items:
        lines += [f"## {i['step']}", "",
                  f"- **Why not verifiable:** {'; '.join(i['reasons'])}",
                  f"- **Used by:** {', '.join(f'`{f}`' for f in i['flows'])} "
                  f"(seen {i['count']}x, e.g. `{i['step_names'][0]}`)",
                  f"- **What it does:** {i['doc'] or '-'}",
                  f"- **Code:** `{i['module']}.{i['step']}`", ""]
    if runs is not None:
        lines += ["## Scenario runs", "", "| Flow | Status | Reason |", "|---|---|---|"]
        for row in runs:
            status = "NOT RUN" if row["status"] == "not_run" else row["status"]
            reason = str(row.get("reason", "")).replace("|", r"\|").replace("\n", " ")
            lines.append(f"| `{row['flow']}` | {status} | {reason} |")
        registered = set(FLOWS)
        represented = {row["flow"] for row in runs}
        submitted = {row["flow"] for row in runs if row.get("run_id") is not None}
        succeeded = {row["flow"] for row in runs if row.get("run_id") is not None and row["status"] == "succeeded"}
        missing = sorted(registered - represented)
        coverage = "complete" if registered <= succeeded else "incomplete"
        lines += ["", f"Registered scenario coverage: {len(registered & represented)}/{len(registered)} flows.",
                  f"Submitted flow coverage: {len(registered & submitted)}/{len(registered)} flows.",
                  f"Successful execution coverage: {len(registered & succeeded)}/{len(registered)} ({coverage})."]
        if missing:
            lines += ["", f"Flows without registered scenarios: {', '.join(missing)}"]
        lines.append("")
    return "\n".join(lines)
