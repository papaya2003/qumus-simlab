"""The sim lab as a contract 2 station (qumus_simlab/simlab/contract2.py).

Contract 2 is papaya2003/lab-contract spec/contract-2.md. The three windows are one station; the
v1 routes stay as they are (test_simlab_contract.py). When lab-contract's `conformance` package is
importable (LAB_CONTRACT=<path to a lab-contract checkout>), its full check also runs against a
live SimLabServer.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

from qumus_simlab.lab import build_lab_side
from qumus_simlab.simlab.contract2 import Station
from qumus_simlab.simlab.modules import build_modules
from qumus_simlab.simlab.service import create_app
from tests.conftest import quiet_bench

H = {"X-Lab-Client": "agent"}


@pytest.fixture
def lab():
    side = build_lab_side(":memory:", seed=1, testbench_config=quiet_bench())
    mods = build_modules(side)
    station = Station({m: (mod.service, mod) for m, mod in mods.items()})
    yield side, mods, station
    for m in mods.values():
        m.service.close()
        for t in m._timers.values():
            t.cancel()
    side.jobs.shutdown(10)


def client(mods, station, *, remote: bool = False) -> TestClient:
    def extra(app, loopback):
        station.routes(app, loopback)

    hosts = frozenset() if remote else frozenset({"testclient"})
    return TestClient(create_app(mods["stacking"].service, loopback_hosts=hosts, extra=extra))


def wait_result(c: TestClient, path: str, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rec = c.get(path).json()
        if rec["result"] is not None:
            return rec
        time.sleep(0.02)
    raise AssertionError(f"{path} did not finish: {rec}")


def test_entry_and_describe(lab):
    _, mods, station = lab
    c = client(mods, station)
    entry = c.get("/api/station").json()
    assert entry["contract"] == 2 and entry["station"] == "qumus-simlab"
    assert [s["path"] for s in entry["services"]] == ["exfoliation/", "stacking/", "reactor/"]
    d = c.get("/exfoliation/api/describe").json()
    assert d["boot_id"] == entry["boot_id"] and d["profiles"] == ["jobs"]
    assert d["lease"]["duration_s"] == mods["exfoliation"].service.lease_s
    assert "lease_s" not in d and "lease_seconds" not in d
    heat = d["commands"]["heat"]
    assert heat["motion"] and heat["hazard"] == ["heat"] and heat["deadline_s"]["maximum"] == 3600
    assert heat["params"]["properties"]["target_c"]["unit"] == "C"
    assert heat["params"]["additionalProperties"] is False
    assert "orphan_timeout_s" in d["lease"]
    go = d["commands"]["go_to_point"]
    assert go["lease"] and go["motion"] and not go["local_only"] and "deadline_s" in go
    assert d["commands"]["read_all"]["allowed_when_locked"]
    assert d["commands"]["jog"]["local_only"]
    # the v1 face is unchanged
    assert c.get("/api/describe").json()["contract"] == "1"


def test_state_is_flat_with_readings(lab):
    _, mods, station = lab
    st = client(mods, station).get("/stacking/api/state").json()
    assert set(st) == {"service", "boot_id", "seq", "ts_unix", "demo", "connected", "busy",
                       "locked", "maintenance", "fault", "lease", "running", "last_command",
                       "readings"}
    assert "transfer_arm" in st["readings"]["devices"]


def test_command_ids_records_duplicates_and_refusals(lab):
    side, mods, station = lab
    c = client(mods, station)
    go = {"command_id": "agent-1", "action": "go_to_point",
          "params": {"point": "exfoliation_stage"}}
    r = c.post("/exfoliation/api/command", json=go, headers=H)
    assert r.status_code == 409 and r.json()["error"]["code"] == "lease_required"
    assert c.get("/exfoliation/api/commands/agent-1").json()["result"] == "rejected"
    lease = c.post("/exfoliation/api/lease", json={}, headers=H).json()
    assert lease["holder"] == "agent"
    go["command_id"] = "agent-2"
    r = c.post("/exfoliation/api/command", json=go, headers=H)
    assert r.status_code == 202 and r.json()["command_id"] == "agent-2"
    rec = wait_result(c, "/exfoliation/api/commands/agent-2")
    assert rec["result"] == "ok" and rec["client"] == "agent" and rec["params"] == go["params"]
    assert side.lab.devices.get("arm_left").get_pose().y == 192.0
    again = c.post("/exfoliation/api/command", json=go, headers=H)
    assert again.status_code == 200 and again.json()["duplicate"] is True
    other = {**go, "params": {"point": "home"}}
    r = c.post("/exfoliation/api/command", json=other, headers=H)
    assert r.status_code == 409 and r.json()["error"]["code"] == "id_reused"
    cases = [
        ({"command_id": "a-3", "action": "go_to_point", "point": "home"}, 400, "bad_params"),
        ({"command_id": "a-4", "action": "go_to_point", "params": {"pt": "home"}}, 400,
         "bad_params"),
        ({"command_id": "a-5", "action": "fly", "params": {}}, 400, "unknown_action"),
        ({"action": "go_to_point", "params": {"point": "home"}}, 400, "bad_request"),
        ({"command_id": "a-6", "action": "go_to_point", "params": {"point": "home"},
          "deadline_s": 99999}, 400, "bad_params"),
    ]
    for body, status, code in cases:
        r = c.post("/exfoliation/api/command", json=body, headers=H)
        assert (r.status_code, r.json()["error"]["code"]) == (status, code), body
    r = c.post("/exfoliation/api/command", json=go)
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"  # no X-Lab-Client
    r = c.post("/exfoliation/api/command", content=b"x", headers={**H, "Content-Type": "text/plain"})
    assert r.status_code == 415
    assert c.get("/exfoliation/api/commands/nope").status_code == 404
    r = c.get("/exfoliation/api/cameras/microscope/frame")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    # lease: held for others, renew_only, release
    other_client = {"X-Lab-Client": "someone"}
    r = c.post("/exfoliation/api/lease", json={}, headers=other_client)
    assert r.status_code == 409 and r.json()["error"]["code"] == "lease_held"
    assert c.post("/exfoliation/api/lease", json={"renew_only": True}, headers=H).status_code == 200
    assert c.delete("/exfoliation/api/lease", headers=H).json()["holder"] is None
    r = c.post("/exfoliation/api/lease", json={"renew_only": True}, headers=H)
    assert r.status_code == 409 and r.json()["error"]["code"] == "lease_lost"


def test_deadline_stops_a_slow_motion_as_expired(lab):
    _, mods, station = lab
    mods["exfoliation"].motion_delay_s = 5.0
    c = client(mods, station)
    c.post("/exfoliation/api/lease", json={}, headers=H)
    body = {"command_id": "slow-1", "action": "go_to_point",
            "params": {"point": "exfoliation_stage"}, "deadline_s": 0.3}
    assert c.post("/exfoliation/api/command", json=body, headers=H).status_code == 202
    assert c.get("/exfoliation/api/state").json()["running"] == ["slow-1"]
    assert wait_result(c, "/exfoliation/api/commands/slow-1")["result"] == "expired"


def test_stop_needs_no_lease_and_repeats(lab):
    _, mods, station = lab
    c = client(mods, station)
    for _ in range(2):
        r = c.post("/stacking/api/stop", json={"wait_s": 0}, headers={"X-Lab-Client": "x"})
        assert r.status_code == 200 and r.json()["stop"] == "confirmed"


def test_job_command_and_jobs_route(lab):
    side, mods, station = lab
    c = client(mods, station)
    c.post("/stacking/api/lease", json={}, headers=H)
    body = {"command_id": "job-1", "action": "place_chip_on_transfer_stage",
            "params": {"chip_id": "chip_001"}}
    assert c.post("/stacking/api/command", json=body, headers=H).status_code == 202
    rec = wait_result(c, "/stacking/api/commands/job-1")
    assert rec["result"] == "ok"
    run_id = rec["data"]["run_id"]
    assert side.jobs.wait(run_id, 30).status == "succeeded"
    job = c.get(f"/stacking/api/jobs/{run_id}").json()
    assert job["state"] == "succeeded" and job["step"] == job["steps_total"] > 0
    assert c.get("/stacking/api/jobs/run_nope").status_code == 404


def test_remote_listener_keeps_local_only(lab):
    _, mods, station = lab
    c = client(mods, station, remote=True)
    c.post("/stacking/api/lease", json={}, headers=H)
    body = {"command_id": "jog-1", "action": "jog",
            "params": {"device": "arm_right", "axis": "x", "mm": 0.1}}
    r = c.post("/stacking/api/command", json=body, headers=H)
    assert r.status_code == 403 and r.json()["error"]["code"] == "local_only"


def test_lab_contract_conformance_on_a_live_sim_lab(tmp_path):
    path = os.environ.get("LAB_CONTRACT")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    v2 = pytest.importorskip("conformance.v2")
    from qumus_simlab.simlab.server import SimLabServer

    side = build_lab_side(str(tmp_path / "qumus.db"), seed=1, testbench_config=quiet_bench())
    server = SimLabServer(side, ports={"exfoliation": 0, "stacking": 0, "reactor": 0}).start()
    try:
        report = v2.run(server.url("stacking"), demo=True, timeout_s=10.0)
    finally:
        server.close()
        side.jobs.shutdown(10)
    assert len(report.results) >= 16
    assert report.passed, [r for r in report.results if not r.ok]
