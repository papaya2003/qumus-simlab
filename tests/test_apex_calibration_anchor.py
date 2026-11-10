"""Current prepared-stamp selection; retired apex roots grant no authority."""
from copy import deepcopy

import pytest

from qumus_simlab.core.clock import SimClock
from qumus_simlab.flows.library.stacking import StackDeviceFlow, StackParams, fresh_stamps
from qumus_simlab.records.stamps import FilmReportError, report_new_film
from qumus_simlab.runtime.store import RunStore


@pytest.fixture
def selection():
    store = RunStore(":memory:", clock=SimClock(), bench="sim")
    try:
        for number, status in ((1, "used"), (2, "fresh"), (3, "fresh")):
            store.upsert_sample(f"stamp_0{number}", "stamp", location=f"right_stamp_{number}", status=status,
                props={"film": "PC", "alignment_xy_calibration": {"schema": "retired-root"}})
        yield store, StackDeviceFlow(), StackParams(flake_ids=["source"], target_chip_id="target")

    finally:
        store.close()


def test_current_selection_needs_fresh_prepared_stamp_in_storage(selection):
    store, flow, params = selection
    assert flow.select_stamp(params, store)["id"] == "stamp_02"
    assert flow.select_stamp(params.model_copy(update={"stamp_id": "stamp_03"}), store)["id"] == "stamp_03"
    with pytest.raises(ValueError, match="known PC/mica"):
        flow.select_stamp(params.model_copy(update={"stamp_id": "stamp_01"}), store)
    store.upsert_sample("stamp_02", "stamp", location="transfer_arm")
    assert flow.select_stamp(params, store)["id"] == "stamp_03"


def test_reserved_spare_cannot_be_selected_by_another_run(selection):
    store, flow, params = selection
    run_id = store.create_run("stack_device", {})
    store.reserve("sample", "stamp_02", run_id, flow.name)
    assert flow.select_stamp(params, store)["id"] == "stamp_03"
    assert [stamp["id"] for stamp in fresh_stamps(store, flow.bench, exclude_run=run_id)] == ["stamp_02", "stamp_03"]


def test_unknown_support_cannot_borrow_retired_apex_metadata(selection):
    store, flow, params = selection
    before = deepcopy(store.get_sample("stamp_02")["props"])
    store.upsert_sample("stamp_02", "stamp", props={"film": "unknown"})
    with pytest.raises(ValueError, match="known PC/mica"):
        flow.select_stamp(params.model_copy(update={"stamp_id": "stamp_02"}), store)
    assert store.get_sample("stamp_02")["props"]["alignment_xy_calibration"] == before["alignment_xy_calibration"]


def test_refilm_report_requires_returned_unreserved_stamp(selection):
    store, flow, params = selection
    store.upsert_sample("stamp_01", "stamp", location="transfer_arm")
    with pytest.raises(FilmReportError, match="not in its slot"):
        report_new_film(store, flow.bench, "stamp_01", by="operator", via="ui:lab")
    store.upsert_sample("stamp_01", "stamp", location="right_stamp_1")
    report_new_film(store, flow.bench, "stamp_01", by="operator", via="ui:lab")
    assert flow.select_stamp(params, store)["id"] == "stamp_01"
