"""[Q] records/images extensions and [Q-sim] bench views of the sim-lab windows
(simlab/records.py, simlab/server.py), and the remote (SSH tunnel) listener."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from qumus_simlab.lab import build_lab_side
from qumus_simlab.simlab.modules import FLOW_MODULE, PORTS, build_modules
from qumus_simlab.simlab.records import IMAGE_PREFIX
from qumus_simlab.simlab.server import sim_routes
from qumus_simlab.simlab.service import create_app
from tests.conftest import quiet_bench

H = {"X-Qumus-Client": "agent"}


@pytest.fixture
def lab(tmp_path):
    side = build_lab_side(tmp_path / "qumus.db", seed=1, testbench_config=quiet_bench())
    mods = build_modules(side)
    records = next(iter(mods.values())).records
    extra = sim_routes(side, records)
    clients = {m: TestClient(create_app(mod.service, extra=extra,
                                       loopback_hosts=frozenset({"testclient"})))
               for m, mod in mods.items()}
    yield side, mods, records, clients
    for m in mods.values():
        m.service.close()
    side.jobs.shutdown(10)


def feed(client, since=0):
    out, nxt = [], since
    while True:
        page = client.get("/api/records", params={"since": nxt, "limit": 500}).json()
        out += page["changes"]
        if page["next"] == nxt:
            return out, page
        nxt = page["next"]


def test_three_windows_share_one_bench_and_every_flow_has_one_owner(lab):
    _, mods, _, clients = lab
    assert set(mods) == set(PORTS) == {"exfoliation", "stacking", "reactor"}
    assert PORTS == {"exfoliation": 8781, "stacking": 8782, "reactor": 8783}
    commands = {m: c.get("/api/describe").json()["commands"] for m, c in clients.items()}
    for flow, module in FLOW_MODULE.items():
        assert commands[module][flow]["job"] is True
        assert all(flow not in cmds for m, cmds in commands.items() if m != module)
    assert {"put_records", "job_status"} <= set(commands["reactor"])


def test_feed_starts_with_the_seeded_ledger_and_is_ordered(lab):
    _, _, _, clients = lab
    changes, page = feed(clients["stacking"])
    assert page["lab_id"] and page["head"] == changes[-1]["seq"]
    seqs = [c["seq"] for c in changes]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    samples = {c["key"]: c["row"] for c in changes if c["table"] == "sample"}
    assert "chip_e01" in samples or any(r["kind"] == "chip" for r in samples.values())
    assert any(c["table"] == "event" for c in changes)
    # every window serves the same lab-wide feed
    assert feed(clients["reactor"])[0] == changes


def test_put_records_upserts_and_echoes_without_lab_events(lab):
    side, _, _, clients = lab
    c = clients["stacking"]
    _, page = feed(c)
    last_event = side.store.last_event_id()
    body = {"action": "put_records",
            "project": [{"id": "prj_x", "name": "x", "state": {"device_plan": {"a": 1}}}],
            "samples": [{"id": "flake_z", "kind": "flake", "location": "chip_001",
                         "status": "found", "props": {"note": "from the agent"}}],
            "instructions": [{"id": "ins_1", "ts": 1.0, "project_id": "prj_x", "source": "ui",
                              "kind": "message", "text": "hello", "status": "done"}],
            "reservations": [{"kind": "slot", "key": "x_slot", "run_id": "job_1", "flow": "f"}]}
    r = c.post("/api/command", json=body, headers=H)
    assert r.status_code == 202, r.text
    assert side.store.get_project("prj_x")["state"] == {"device_plan": {"a": 1}}
    assert side.store.get_sample("flake_z")["props"] == {"note": "from the agent"}
    assert side.store.get_instruction("ins_1")["text"] == "hello"
    assert side.store.reservations("slot")["x_slot"]["run_id"] == "job_1"
    assert side.store.last_event_id() == last_event  # no lab events for agent writes
    echo, _ = feed(c, page["next"])
    assert {(e["table"], e["key"]) for e in echo} >= {("sample", "flake_z"),
                                                       ("reservation", "slot:x_slot")}
    # released reservation and removed sample come back as null rows
    c.post("/api/command", json={"action": "put_records", "removed_samples": ["flake_z"],
                                 "reservations": [{"kind": "slot", "key": "x_slot",
                                                   "released": True}]}, headers=H)
    nulls, _ = feed(c, page["next"])
    assert {(e["table"], e["key"]) for e in nulls if e["row"] is None} == {
        ("sample", "flake_z"), ("reservation", "slot:x_slot")}
    bad = c.post("/api/command", json={"action": "put_records", "samples": [{"id": 1}]},
                 headers=H)
    assert bad.status_code == 422


def test_images_are_ids_in_rows_and_served_with_meta(lab):
    side, _, records, clients = lab
    c = clients["stacking"]
    img_dir = records.image_dir / "microscope"
    img_dir.mkdir(parents=True, exist_ok=True)
    f = img_dir / "pic_1.png"
    Image.new("RGB", (8, 6), "gray").save(f)
    side.store.upsert_sample("flake_img", "flake", location="chip_001",
                             props={"images": {"20x": str(f)}, "other": "/not/a/lab/file.png"})
    changes, _ = feed(c)
    row = [x for x in changes if x["key"] == "flake_img"][-1]["row"]
    assert row["props"]["images"]["20x"] == IMAGE_PREFIX + "microscope/pic_1.png"
    assert row["props"]["other"] == "/not/a/lab/file.png"
    ev = [x for x in changes if x["table"] == "event"
          and x["row"]["kind"] == "sample.updated" and x["row"]["payload"]["id"] == "flake_img"]
    assert ev[-1]["row"]["payload"]["props"]["images"]["20x"].startswith(IMAGE_PREFIX)
    r = c.get("/api/images/microscope/pic_1.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content == f.read_bytes()
    meta = c.get("/api/images/microscope/pic_1.png/meta").json()
    assert (meta["width"], meta["height"]) == (8, 6)
    assert c.get("/api/images/microscope/none.png").status_code == 404
    assert c.get("/api/images/../qumus.db").status_code in (400, 404)
    # the agent sends ids back; the lab stores its own path again
    c.post("/api/command", json={"action": "put_records", "samples": [
        {"id": "flake_img", "kind": "flake", "location": "chip_001", "status": None,
         "props": {"images": {"20x": IMAGE_PREFIX + "microscope/pic_1.png"}}}]}, headers=H)
    assert side.store.get_sample("flake_img")["props"]["images"]["20x"] == str(f.resolve())


def test_job_command_puts_the_run_in_the_feed_and_job_status_reads_it(lab):
    side, mods, _, clients = lab
    c = clients["stacking"]
    _, page = feed(c)
    assert c.post("/api/heartbeat", headers=H).status_code == 200
    r = c.post("/api/command", json={"action": "inspect_bench", "job": {"project_id": None}},
               headers=H)
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    changes, _ = feed(c, page["next"])
    assert any(x["table"] == "run" and x["key"] == run_id for x in changes)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        st = c.post("/api/command", json={"action": "job_status", "run_id": run_id},
                    headers=H).json()["result"]
        if st["done"]:
            break
        time.sleep(0.1)
    assert st["flow"] == "inspect_bench" and st["module"] == "stacking" and st["steps"]
    assert c.post("/api/command", json={"action": "job_status", "run_id": "nope"},
                  headers=H).status_code == 422


def test_sim_views_layout_bench_state_preview_and_faults(lab):
    side, _, _, clients = lab
    c = clients["stacking"]
    lay = c.get("/api/sim/layout").json()
    assert "cameras" in lay["bench"] and {"regions", "positions"} <= set(lay["layout"])
    st = c.get("/api/sim/bench_state").json()
    assert {"devices", "objects", "twin", "stamp", "faults"} <= set(st)
    assert c.get("/api/sim/preview/overview_camera").status_code == 200
    assert c.get("/api/sim/preview/nope").status_code == 404
    menu = c.get("/api/sim/faults").json()
    assert isinstance(menu, list) and menu
    fault = menu[0]["fault"]
    r = c.post("/api/sim/faults", json={"fault": fault, "params": {}})
    assert r.status_code in (200, 400)  # some faults need parameters
    assert c.post("/api/sim/faults", json={"nope": 1}).status_code == 400


def test_remote_listener_refuses_local_only_even_from_loopback(lab):
    side, mods, records, _ = lab
    remote = TestClient(create_app(mods["stacking"].service, loopback_hosts=frozenset(),
                                   extra=sim_routes(side, records)))
    assert remote.get("/api/describe").status_code == 200
    assert remote.get("/api/records").status_code == 200
    assert remote.post("/api/unlock", headers=H).status_code == 403
    assert remote.post("/api/sim/faults", json={"fault": "x"}).status_code == 403
    remote.post("/api/heartbeat", headers=H)
    r = remote.post("/api/command", json={"action": "jog", "device": "arm_right",
                                          "axis": "x", "mm": 0.1}, headers=H)
    assert r.status_code == 403


def test_save_point_and_resume_keep_the_lab_and_its_picture_ids(tmp_path):
    from qumus_simlab.simlab.server import build_side

    side = build_side(tmp_path / "a" / "qumus.db", 3)
    mods = build_modules(side)
    records = next(iter(mods.values())).records
    app = create_app(mods["stacking"].service, loopback_hosts=frozenset({"testclient"}),
                     extra=sim_routes(side, records, side.info))
    c = TestClient(app)
    pic = records.image_dir / "camera" / "old.png"
    pic.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), "gray").save(pic)
    side.store.upsert_sample("flake_old", "flake", location="chip_001",
                             props={"images": {"20x": str(pic)}})
    r = c.post("/api/sim/checkpoint", json={"dir": str(tmp_path / "cp"), "label": "test"})
    assert r.status_code == 200, r.text
    head = r.json()["records_head"]
    remote = TestClient(create_app(mods["stacking"].service, loopback_hosts=frozenset(),
                                   extra=sim_routes(side, records, side.info)))
    assert remote.post("/api/sim/checkpoint", json={"dir": str(tmp_path / "x")}).status_code == 403
    for m in mods.values():
        m.service.close()
    side.jobs.shutdown(10)

    side2 = build_side(tmp_path / "b" / "qumus.db", None, resume=tmp_path / "cp")
    assert side2.info["seed"] == 3
    mods2 = build_modules(side2)
    rec2 = next(iter(mods2.values())).records
    assert rec2.lab_id == records.lab_id  # same lab: a client continues from its cursor
    assert side2.store.get_sample("flake_old")["props"]["images"]["20x"] == str(pic)
    page = rec2.changes(head, 2000)
    assert all(ch["seq"] > head for ch in page["changes"])
    row = rec2._row("sample", "flake_old")
    assert rec2.scrub(row)["props"]["images"]["20x"] == IMAGE_PREFIX + "camera/old.png"
    assert rec2.image_path("camera/old.png") == pic.resolve()
    for m in mods2.values():
        m.service.close()
    side2.jobs.shutdown(10)
