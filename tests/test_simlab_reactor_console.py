"""The simulated reactor's Lab Console API under /lab/ (simlab/reactor_console.py)."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qumus_simlab.simlab.reactor_console import SimReactorConsole, console_routes

H = {"X-MG400-Client": "agent-qumus"}
OTHER = {"X-MG400-Client": "page-owner"}


@pytest.fixture
def lab():
    sim = SimReactorConsole(time_scale=6000.0)
    app = FastAPI()
    console_routes(sim)(app, lambda request: True)
    with TestClient(app) as client:
        yield sim, client


def take(client, *devices, headers=H):
    for d in devices:
        assert client.post(f"/lab/api/console/control/{d}/take", json={}, headers=headers).status_code == 200


def run(client, recipe, params, timeout=20.0):
    assert client.post("/lab/api/heartbeat", json={}, headers=H).status_code == 200
    r = client.post("/lab/api/runs", json={"recipe": recipe, "params": params, "borrow": True}, headers=H)
    assert r.status_code == 202, r.text
    cid = r.json()["command_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/lab/api/state").json()
        active = state["active_run"]
        assert state["last_command"]["id"] == cid
        if active["closed"]:
            steps = client.get(f"/lab/api/runs/{active['run_id']}").json()["steps"]
            client.post("/lab/api/release", json={}, headers=H)
            return active, steps
        time.sleep(0.01)
    raise AssertionError("run did not close")


def furnace(client):
    return client.get("/lab/api/console/state/furnace").json()["state"]


def test_reads_and_describe(lab):
    _, client = lab
    body = client.get("/lab/api/console/state").json()
    assert body["console"]["demo"] is True and body["run"] is None
    state = furnace(client)
    assert state["pv_c"] == 25.0 and state["sequence"]["running"] is False
    assert "start_sequence" in client.get("/lab/api/console/describe/furnace").json()["commands"]
    assert client.get("/lab/api/console/state/nothing").status_code == 404
    assert client.get("/lab/sim/microscope/api/camera/microscope/frame.jpg").content[:2] == b"\xff\xd8"


def test_posts_need_the_page_header_and_control(lab):
    _, client = lab
    assert client.post("/lab/api/console/heartbeat", json={}).status_code == 400
    r = client.post("/lab/api/console/command/furnace", json={"action": "start_sequence", "params": {}},
                    headers=H)
    assert r.status_code == 409 and r.json()["outcome"] == "refused"
    take(client, "robot", headers=OTHER)
    assert client.post("/lab/api/console/control/robot/take", json={}, headers=H).status_code == 409


def test_round_trip(lab):
    sim, client = lab
    assert client.post("/lab/sim/handover", json={"slot": "A1", "chip_id": "chip_r1"}, headers=H).status_code == 200
    take(client, "robot", "vision", "stage", "furnace", "thermal", "microscope")
    active, steps = run(client, "slot-state", {"slot": "A1"})
    assert active["status"] == "completed"
    import json
    assert json.loads(steps[0]["result_json"])["state"] == "present"
    active, _ = run(client, "crucible-load", {"from_slot": "A1"})
    assert active["status"] == "completed" and sim.crucible == "chip_r1"
    seq = {"steps": [{"temperature_c": 400, "ramp_rate_c_per_min": 100, "hold_minutes": 2, "gas": {"0": 30}}],
           "end_gas": {"0": 10}, "end_gas_until_c": 150}
    r = client.post("/lab/api/console/command/furnace", json={"action": "start_sequence", "params": seq}, headers=H)
    assert r.status_code == 202
    cid = r.json()["command_id"]
    assert furnace(client)["last_command"] == furnace(client)["last_command"] | {"id": cid, "result": "ok"}
    peak = 0.0
    deadline = time.monotonic() + 10
    while furnace(client)["sequence"]["running"] and time.monotonic() < deadline:
        peak = max(peak, furnace(client)["pv_c"])
        time.sleep(0.005)
    assert peak > 300
    # loading while hot is refused by the recipe
    active, steps = run(client, "crucible-unload", {"to_slot": "A2"})  # waits until < 60 C
    assert active["status"] == "completed" and sim.slots["A2"] == "chip_r1"
    assert furnace(client)["pv_c"] < 60
    active, steps = run(client, "microscope-record", {"chip_id": "chip_r1", "seconds": 30})
    assert active["status"] == "completed"
    assert client.post("/lab/sim/handover", json={"slot": "A2", "chip_id": None}, headers=H).status_code == 200
    assert sim.slots["A2"] is None


def test_load_refused_when_hot_and_runs_need_devices(lab):
    sim, client = lab
    sim.handover("A1", "c1")
    take(client, "robot", "stage")
    assert client.post("/lab/api/heartbeat", json={}, headers=H).status_code == 200
    r = client.post("/lab/api/runs", json={"recipe": "crucible-load", "params": {"from_slot": "A1"},
                                           "borrow": True}, headers=H)
    assert r.status_code == 409 and "furnace" in r.json()["error"]
    client.post("/lab/api/release", json={}, headers=H)
    take(client, "furnace")
    sim.furnace["pv"] = 300.0
    sim._last = sim.clock()
    active, steps = run(client, "crucible-load", {"from_slot": "A1"})
    assert active["status"] == "aborted" and "furnace" in active["detail"]
    assert sim.slots["A1"] == "c1"


def test_gas_needs_end_plan_and_stop_turns_heat_off(lab):
    sim, client = lab
    take(client, "furnace")
    bad = {"steps": [{"temperature_c": 300, "ramp_rate_c_per_min": 10, "hold_minutes": 1, "gas": {"0": 20}}]}
    client.post("/lab/api/console/command/furnace", json={"action": "start_sequence", "params": bad}, headers=H)
    assert furnace(client)["last_command"]["result"] == "rejected"
    good = {"steps": [{"temperature_c": 300, "ramp_rate_c_per_min": 10, "hold_minutes": 60}]}
    client.post("/lab/api/console/command/furnace", json={"action": "start_sequence", "params": good}, headers=H)
    assert furnace(client)["sequence"]["running"] is True
    assert client.post("/lab/api/console/stop/furnace", json={}, headers=H).json()["outcome"] == "confirmed"
    state = furnace(client)
    assert state["sequence"]["running"] is False and state["flow_pct"] == [0.0, 0.0, 0.0]


def test_motion_stop_holds_a_run_and_abort_closes_it(lab):
    sim, client = lab
    sim.handover("A1", "c1")
    sim.time_scale = 1.0  # the load takes 90 real seconds now: it is still running when stopped
    take(client, "robot", "stage", "furnace")
    client.post("/lab/api/heartbeat", json={}, headers=H)
    client.post("/lab/api/runs", json={"recipe": "crucible-load", "params": {"from_slot": "A1"},
                                       "borrow": True}, headers=H)
    assert client.post("/lab/api/dashboard/motion", json={}, headers=H).status_code == 202
    deadline = time.monotonic() + 5
    while client.get("/lab/api/state").json()["active_run"]["status"] != "held":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    run_id = client.get("/lab/api/state").json()["active_run"]["run_id"]
    assert client.post(f"/lab/api/runs/{run_id}/abort", json={}, headers=H).status_code == 202
    active = client.get("/lab/api/state").json()["active_run"]
    assert active["status"] == "aborted" and active["closed"]
    assert sim.slots["A1"] == "c1"  # the chip did not move
