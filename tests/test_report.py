"""NOT_VERIFIABLE report (spec 4.2): every flow runs, every unverifiable step
is listed with its reason."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qumus_simlab.flows.flow import FLOWS
from qumus_simlab.testbench.report import (
    SCENARIOS,
    ReportPrerequisiteUnavailable,
    _stack_params,
    collect_not_verifiable,
    render_markdown,
    run_scenarios,
)


def test_every_flow_is_in_the_report_scenarios():
    assert {f for f, _ in SCENARIOS} == set(FLOWS)


def test_report_lists_unverifiable_steps(services):
    runs = run_scenarios(services)
    # The two prepared chips now supply a compatible stack and it is transferred
    # (the missing-source case is test_report_stack_prerequisite_is_typed...).
    stack = next(row for row in runs if row["flow"] == "stack_device")
    assert stack["status"] == "succeeded" and stack["run_id"], stack
    # v1 cannot see the source through the held stamp: without a device design
    # too, every pickup parks the stamp before the source is inspected.
    events = services.store.events_after(0, limit=100000, run_id=stack["run_id"])
    parks = [e.payload["phase"] for e in events if e.kind == "stack.design_source_park"]
    picks = {e.payload.get("name") for e in events if e.kind == "step.started"
             and str(e.payload.get("name", "")).startswith("v1_pick_up_")}
    assert parks and set(parks) >= picks, (parks, picks)
    assert all(r["status"] == "succeeded" or
               (r["status"] == "disabled" and ("reactor" in r["flow"] or
                r["flow"] == "prepare_material_metal" and r["reason"].startswith("abstract workflow fixture requires explicit project authorization")))
               for r in runs if r["flow"] != "stack_device"), runs
    items = collect_not_verifiable(services.store)
    massage = next(i for i in items if i["step"] == "PressTape")
    assert "force sensor" in massage["reasons"][0]
    assert massage["flows"] == ["exfoliate_chips"] and massage["count"] == 2
    text = render_markdown(items, runs)
    assert "## PressTape" in text and "Flows without registered scenarios" not in text
    assert "| `stack_device` | succeeded |" in text
    assert "Submitted flow coverage:" in text and "(incomplete)" in text


def test_report_stack_prerequisite_is_typed_without_admitting_sources():
    store = SimpleNamespace(list_samples=lambda **kwargs: [])
    with pytest.raises(ReportPrerequisiteUnavailable, match="No compatible observed stack") as raised:
        _stack_params(store)
    assert raised.value.code == "compatible_observed_stack_unavailable"


def test_report_keeps_running_declared_scenarios_after_typed_prerequisite(monkeypatch):
    from qumus_simlab.testbench import render, report

    submitted = []
    def unavailable(store):
        raise ReportPrerequisiteUnavailable("Original observed inputs are unavailable", code="fixture_prerequisite")
    def submit(flow, params):
        submitted.append((flow, params))
        return f"synthetic-{len(submitted)}"
    services = SimpleNamespace(store=object(), lab=SimpleNamespace(bench={}),
        jobs=SimpleNamespace(submit=submit,
            wait=lambda rid, timeout: SimpleNamespace(run_id=rid, status="succeeded")))
    monkeypatch.setattr(render, "ensure_diagnostic_images", lambda s: None)
    monkeypatch.setattr(report, "SCENARIOS", [("inspect_bench", {}), ("stack_device", unavailable), ("inspect_bench", {})])
    rows = report.run_scenarios(services)
    assert [row["status"] for row in rows] == ["succeeded", "not_run", "succeeded"]
    assert rows[1] == {"flow": "stack_device", "run_id": None, "status": "not_run",
                       "reason": "Original observed inputs are unavailable", "prerequisite": "fixture_prerequisite"}
    assert submitted == [("inspect_bench", {}), ("inspect_bench", {})]


def test_report_never_converts_an_unrelated_factory_error_to_not_run(monkeypatch):
    from qumus_simlab.testbench import render, report

    def broken(store):
        raise ValueError("Malformed scenario metadata")
    monkeypatch.setattr(render, "ensure_diagnostic_images", lambda s: None)
    monkeypatch.setattr(report, "SCENARIOS", [("stack_device", broken)])
    services = SimpleNamespace(store=object(), lab=SimpleNamespace(bench={}))
    with pytest.raises(ValueError, match="Malformed scenario metadata"):
        report.run_scenarios(services)


def test_report_render_complete_rows_are_explicitly_synthetic_unit_data():
    # Render-only rows test coverage wording; no experiment was executed.
    rows = [{"flow": flow, "run_id": f"synthetic-render-{flow}", "status": "succeeded"} for flow in FLOWS]
    text = render_markdown([], rows)
    assert f"Registered scenario coverage: {len(FLOWS)}/{len(FLOWS)} flows." in text
    assert f"Successful execution coverage: {len(FLOWS)}/{len(FLOWS)} (complete)." in text
    assert "NOT RUN" not in text


def test_report_render_running_submission_is_not_completed_coverage(monkeypatch):
    from qumus_simlab.testbench import report

    monkeypatch.setattr(report, "FLOWS", {"inspect_bench": object()})
    text = report.render_markdown([], [{"flow": "inspect_bench", "run_id": "synthetic", "status": "running"}])
    assert "Submitted flow coverage: 1/1 flows." in text
    assert "Successful execution coverage: 0/1 (incomplete)." in text
