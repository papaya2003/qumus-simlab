"""The sim lab served through instrument service contract v1 (qumus_simlab/simlab).

Service rules (lease, busy, locked, stop, events, local-only) are checked on
the three simulated modules over one simulated bench, partly through HTTP
(FastAPI TestClient). The records/images extensions are in test_simlab_records.py.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from qumus_simlab.lab import build_lab_side
from qumus_simlab.simlab.modules import FLOW_MODULE, build_modules
from qumus_simlab.simlab.service import CommandRefused, create_app
from tests.conftest import quiet_bench


class Wall:
    """Hand-driven wall clock for lease tests."""

    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def lab():
    side = build_lab_side(":memory:", seed=1, testbench_config=quiet_bench())
    wall = Wall()
    mods = build_modules(side, wall=wall)
    yield side, mods, wall
    for m in mods.values():
        m.service.close()
        for t in m._timers.values():
            t.cancel()
    side.jobs.shutdown(10)


def wait_done(service, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        lc = service.last_command
        if lc and lc["result"] != "running" and not service._motion_running():
            return lc
        time.sleep(0.01)
    raise AssertionError(f"command did not finish: {service.last_command}")


def messages(service, since: int = 0) -> list[str]:
    return [e["message"] for e in service.events(since)["events"]]


def test_describe_lists_devices_commands_and_flags(lab):
    _, mods, _ = lab
    exf, stk = mods["exfoliation"].service, mods["stacking"].service
    d = exf.describe()
    assert d["contract"] == "1" and d["module"] == "exfoliation" and d["demo"] is True
    assert d["client_header"] == "X-Qumus-Client"
    assert {"arm_left", "exfoliation_heater", "exfoliator"} <= set(d["devices"])
    assert d["devices"]["exfoliation_heater"]["fields"] == {"temperature": "C", "target": "C"}
    cmds = d["commands"]
    assert cmds["read_all"]["any_client"] and not cmds["read_all"]["motion"]
    assert cmds["go_to_point"]["motion"] and not cmds["go_to_point"]["local_only"]
    assert cmds["jog"]["local_only"] and cmds["set_switch"]["local_only"]
    assert "exfoliate_chips" in cmds and "stack_device" not in cmds
    assert "stack_device" in stk.describe()["commands"]
    assert set(stk.describe()["devices"]) >= {"arm_right", "microscope", "transfer_arm",
                                             "stamp_heater"}
    # every flow the agent submits has exactly one owner module
    for flow, module in FLOW_MODULE.items():
        assert flow in mods[module].service.describe()["commands"]


def test_state_has_the_contract_fields(lab):
    _, mods, _ = lab
    st = mods["stacking"].service.state()
    assert {"service", "seq", "ts_unix", "demo", "connected", "busy", "locked", "fault",
            "lease", "last_command", "devices"} <= set(st)
    assert st["connected"] and not st["busy"] and not st["locked"] and st["lease"] is None
    assert st["devices"]["arm_right"]["homed"] is True
    assert st["devices"]["stamp_heater"]["temperature"] == pytest.approx(22.0, abs=3)


def test_reads_need_no_lease_but_motion_does(lab):
    _, mods, wall = lab
    s = mods["exfoliation"].service
    out = s.command("ui", {"action": "read_all"})
    assert set(out["result"]["devices"]) == set(mods["exfoliation"].device_ids)
    assert out["state"]["last_command"]["result"] == "ok"
    with pytest.raises(CommandRefused) as e:
        s.command("agent", {"action": "go_to_point", "point": "exfoliation_stage"})
    assert e.value.status == 409 and "lease" in e.value.message
    s.heartbeat("other")
    with pytest.raises(CommandRefused) as e:
        s.command("agent", {"action": "go_to_point", "point": "exfoliation_stage"})
    assert e.value.status == 409 and "other" in e.value.message
    with pytest.raises(CommandRefused):
        s.heartbeat("agent")  # cannot take a lease someone else holds
    wall.t += 3.5  # the other client stopped renewing
    s.heartbeat("agent")
    assert s.state()["lease"]["holder"] == "agent"


def test_go_to_point_moves_the_arm_through_the_safety_gate(lab):
    side, mods, _ = lab
    s = mods["exfoliation"].service
    s.heartbeat("agent")
    seq = s.state()["seq"]
    s.command("agent", {"action": "go_to_point", "point": "exfoliation_stage"})
    lc = wait_done(s)
    assert lc["result"] == "ok", lc
    pose = side.lab.devices.get("arm_left").get_pose()
    assert (pose.y, pose.z) == (192.0, 95.0)  # station (307, 318) - base (307, 126), safe z
    assert messages(s, seq)[:3] == ["command accepted", "command started", "command done"]
    gate_log = [e.payload for e in side.store.events_after(0, limit=10_000)
                if e.kind == "device.command"]
    assert any(p.get("device") == "arm_left" and p.get("command") == "move_to" for p in gate_log)
    for bad in ({"point": "transfer_stage"}, {"point": 3}, {"point": "home", "x": 1}):
        with pytest.raises(CommandRefused) as e:
            s.command("agent", {"action": "go_to_point", **bad})
        assert e.value.status == 422
    with pytest.raises(CommandRefused) as e:
        s.command("agent", {"action": "fly"})
    assert e.value.status == 422


def test_lease_expiry_during_motion_stops_it(lab):
    side, mods, wall = lab
    mod = mods["exfoliation"]
    mod.motion_delay_s = 5.0
    s = mod.service
    s.heartbeat("agent")
    before = side.lab.devices.get("arm_left").get_pose()
    s.command("agent", {"action": "go_to_point", "point": "exfoliation_stage"})
    assert s.state()["busy"]
    with pytest.raises(CommandRefused) as e:  # one motion at a time
        s.command("agent", {"action": "go_to_point", "point": "home"})
    assert e.value.status == 409
    wall.t += 3.1  # the agent died: no renew
    s.tick()
    lc = wait_done(s, 3.0)
    assert lc["result"] == "stopped"
    assert side.lab.devices.get("arm_left").get_pose() == before
    assert "lease expired" in messages(s) and s.state()["lease"] is None


def test_stop_from_any_client_is_idempotent(lab):
    _, mods, _ = lab
    mod = mods["stacking"]
    mod.motion_delay_s = 5.0
    s = mod.service
    s.heartbeat("agent")
    s.command("agent", {"action": "go_to_point", "point": "transfer_stage"})
    first = s.stop(wait_s=2.0, client="ui")  # no lease
    assert first["stop"] == "confirmed" and not first["state"]["busy"]
    assert wait_done(s)["result"] == "stopped"
    assert s.stop(wait_s=0.5, client="ui")["stop"] == "confirmed"


def test_unknown_outcome_locks_until_a_person_unlocks(lab):
    side, mods, _ = lab
    s = mods["stacking"].service
    side.lab.faults.inject("serial_timeout", {"device": "arm_right", "count": 1})
    s.heartbeat("agent")
    s.command("agent", {"action": "go_to_point", "point": "transfer_stage"})
    lc = wait_done(s)
    assert lc["result"] == "failed" and "unknown outcome" in lc["message"]
    st = s.state()
    assert st["locked"] and st["fault"]
    with pytest.raises(CommandRefused) as e:  # never an automatic retry
        s.command("agent", {"action": "go_to_point", "point": "transfer_stage"})
    assert e.value.status == 409 and "locked" in e.value.message
    assert s.command("ui", {"action": "read_all"})["result"]  # reads still work
    with pytest.raises(CommandRefused) as e:
        s.unlock("agent", loopback=False)
    assert e.value.status == 403
    s.unlock("person", loopback=True)
    assert not s.state()["locked"]
    s.command("agent", {"action": "go_to_point", "point": "transfer_stage"})
    assert wait_done(s)["result"] == "ok"


def test_heat_needs_a_timeout_and_switches_itself_off(lab):
    side, mods, _ = lab
    s = mods["exfoliation"].service
    s.heartbeat("agent")
    with pytest.raises(CommandRefused) as e:
        s.command("agent", {"action": "heat", "device": "exfoliation_heater", "target_c": 30})
    assert e.value.status == 422
    with pytest.raises(CommandRefused) as e:  # another module's heater
        s.command("agent", {"action": "heat", "device": "stamp_heater", "target_c": 30,
                            "timeout_s": 5})
    assert e.value.status == 422
    s.command("agent", {"action": "heat", "device": "exfoliation_heater", "target_c": 30,
                        "timeout_s": 0.2})
    heater = side.lab.devices.get("exfoliation_heater")
    assert heater.get_target() == 30.0
    deadline = time.monotonic() + 3
    while heater.get_target() is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert heater.get_target() is None and "auto off" in messages(s)
    s.command("agent", {"action": "heat", "device": "exfoliation_heater", "target_c": 30,
                        "timeout_s": 60})
    s.command("ui", {"action": "heat_off", "device": "exfoliation_heater"})  # any client
    assert heater.get_target() is None


def test_local_only_actions_and_caps_over_http(lab):
    side, mods, _ = lab
    app = create_app(mods["stacking"].service, loopback_hosts=frozenset({"testclient"}))
    local = TestClient(app, headers={"X-Qumus-Client": "debug-page"})
    remote = TestClient(app, client=("10.0.0.7", 5000), headers={"X-Qumus-Client": "agent"})
    assert local.get("/api/describe").json()["module"] == "stacking"
    assert local.post("/api/heartbeat").status_code == 200
    jog = {"action": "jog", "device": "transfer_arm", "axis": "z", "mm": 0.05}
    assert remote.post("/api/command", json=jog).status_code == 403
    assert local.post("/api/command", json={**jog, "mm": 0.2}).status_code == 422  # 0.1 mm cap
    z0 = side.lab.devices.get("transfer_arm").get_positions()["z"]
    r = local.post("/api/command", json=jog)
    assert r.status_code == 202 and r.json()["command_id"]
    assert wait_done(mods["stacking"].service)["result"] == "ok"
    assert side.lab.devices.get("transfer_arm").get_positions()["z"] == pytest.approx(z0 + 0.05, abs=0.01)
    sw = {"action": "set_switch", "device": "transfer_stage_fan", "on": True, "auto_off_s": 0.2}
    assert local.post("/api/command", json={**sw, "auto_off_s": 30}).status_code == 422
    assert local.post("/api/command", json=sw).status_code == 202
    fan = side.lab.devices.get("transfer_stage_fan")
    assert fan.get()
    deadline = time.monotonic() + 3
    while fan.get() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not fan.get()
    assert remote.post("/api/unlock").status_code == 403
    assert remote.post("/api/stop", json={"wait_s": 0}).json()["stop"] == "confirmed"
    page = local.get("/api/events", params={"since": 0}).json()
    assert page["next"] == page["events"][-1]["seq"]
    assert local.get("/api/events", params={"since": page["next"]}).json()["events"] == []
    assert local.post("/api/command", content=b"not json").status_code == 422


def test_flow_action_queues_a_job_and_blocks_single_motion(lab):
    side, mods, _ = lab
    s = mods["stacking"].service
    s.heartbeat("agent")
    out = s.command("agent", {"action": "place_chip_on_transfer_stage", "chip_id": "chip_001",
                              "job": {"requested_by": "hands"}})
    rid = out["run_id"]
    assert side.store.get_run(rid)["flow"] == "place_chip_on_transfer_stage"
    assert out["state"]["last_command"]["data"]["run_id"] == rid
    if s.state()["busy"]:  # the job may already be done on a fast machine
        with pytest.raises(CommandRefused) as e:
            s.command("agent", {"action": "go_to_point", "point": "home"})
        assert e.value.status == 409
    st = side.jobs.wait(rid, 30)
    assert st.done and st.status == "succeeded", st
    assert not s.state()["busy"]
    with pytest.raises(CommandRefused) as e:  # bad flow parameters are a 422
        s.command("agent", {"action": "place_chip_on_transfer_stage", "chip_id": ["chip_001"]})
    assert e.value.status == 422
    with pytest.raises(CommandRefused) as e:  # job controls stay with the owner module
        mods["exfoliation"].service.command("agent", {"action": "job_cancel", "run_id": rid})
    assert e.value.status == 422


def test_heartbeat_is_not_held_up_by_a_slow_job_lookup():
    import threading

    from qumus_simlab.simlab.service import ContractService

    def slow_jobs():
        time.sleep(1.0)
        return []

    svc = ContractService("s", "stacking", [], active_jobs=slow_jobs)
    t = threading.Thread(target=svc.state)
    t.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    svc.heartbeat("agent")
    assert time.monotonic() - t0 < 0.5
    t.join()
    svc.close()


def test_run_statuses_skip_run_meta(lab):
    side, mods, _ = lab
    rid = side.store.create_run("stacking_flow_demo", {"big": "x" * 1000})
    rows = side.store.run_statuses()
    assert {"id": rid, "project_id": None, "flow": "stacking_flow_demo", "status": "queued"} in rows
    assert all(set(r) == {"id", "project_id", "flow", "status"} for r in rows)
