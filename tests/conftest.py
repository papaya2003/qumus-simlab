from __future__ import annotations

import asyncio
import copy
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from qumus_simlab.config import load_yaml


def display_matches(display: bytes, capture: bytes) -> bool:
    """A display-only motion frame (JPEG, #205) shows the same picture as the
    lossless capture: same size, JPEG-level differences only."""
    import io

    import numpy as np
    from PIL import Image

    a, b = (np.asarray(Image.open(io.BytesIO(x)).convert("RGB"), dtype=np.int16) for x in (display, capture))
    if display[:2] != b"\xff\xd8" or a.shape != b.shape:
        return False
    error = abs(a-b)
    return float(error.mean()) < 3 and float(np.percentile(error, 99.9)) <= 24


def quiet_bench(**overrides: Any) -> dict[str, Any]:
    """Testbench config with no random big errors unless asked for."""
    cfg = copy.deepcopy(load_yaml("testbench.yaml"))
    cfg["placement"]["big_error_prob"] = 0.0
    cfg["placement"]["sigma_mm"] = 0.05
    cfg["placement"]["sigma_deg"] = 0.2
    # no random detector errors either (tests that want them turn them on)
    for k in ("tail_prob", "miss_prob", "false_positive_prob", "confuse_material_prob",
              "bias_sigma_mm", "bias_sigma_deg"):
        cfg["locator"][k] = 0.0
    for k, v in overrides.items():
        section, key = k.split("__")
        cfg[section][key] = v
    return cfg


@dataclass
class Services:
    """The lab half of qumus-v2's Services: run store, clock, sim lab, gate, jobs.
    (The agent half lives in qumus-agent; this repository has no agents.)"""

    store: Any
    clock: Any
    lab: Any
    gate: Any
    jobs: Any
    project_id: str


def build_services(db_path: Any = ":memory:", *, seed: int | None = None,
                   testbench_config: dict[str, Any] | None = None,
                   image_dir: Any = None, **_: Any) -> Services:
    from qumus_simlab.lab import build_lab_side

    side = build_lab_side(db_path, seed=seed, testbench_config=testbench_config,
                          image_dir=image_dir)
    side.jobs.close_leftovers("the testbench was restarted from scratch")
    projects = side.store.list_projects()
    pid = projects[0]["id"] if projects else side.store.create_project("demo")
    return Services(side.store, side.clock, side.lab, side.gate, side.jobs, pid)


@pytest.fixture
def services() -> Services:
    services = build_services(":memory:", seed=1, testbench_config=quiet_bench())
    yield services
    _close_owned_services(services)


@pytest.fixture
def make_services():
    owned = []
    def _make(image_dir: Any = None, **overrides: Any) -> Services:
        services = build_services(":memory:", seed=1,
                                  testbench_config=quiet_bench(**overrides), image_dir=image_dir)
        owned.append(services)
        return services
    yield _make
    for services in reversed(owned):
        _close_owned_services(services)


def run_fixture_coroutine(factory, *, timeout_s=35.0, force_thread=False):
    """Run a coroutine factory without nesting Playwright's running event loop.

    The factory is invoked on its owning thread, never pre-created and left
    unawaited. A timed-out worker retains its objects; it is never killed.
    """
    if not force_thread:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(factory())
    outcome = {}
    def run():
        try:
            outcome["value"] = asyncio.run(factory())
        except BaseException as error:
            outcome["error"] = error
    worker = threading.Thread(target=run, name="qumus-fixture-async", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError("Owned fixture coroutine has not settled; its resources remain retained")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _close_owned_services(services):
    report = services.jobs.shutdown(30.0)
    assert not report["alive_workers"] and not report["stop_pending"] and report["stop_error"] is None, report


@pytest.fixture
def pickup_inspection(tmp_path):
    """Synthetic current search evidence for isolated Picker contract fixtures.

    This does not execute or qualify a physical search. Detail-image availability
    remains independent from the explicit pickup-neighbourhood context receipt.
    """
    from PIL import Image

    def add(services, flake_ids, chip_id="chip_001"):
        image = tmp_path / f"{chip_id}-pickup-context.png"
        Image.new("RGB", (4, 4), "gray").save(image)
        job = services.store.create_run("search_chip", {"chip_id": chip_id}, project_id=services.project_id)
        services.store.set_run_status(job, "succeeded")
        for fid in flake_ids:
            services.store.upsert_sample(fid, "flake", location=chip_id, props={"pickup_assessment": {
                "flake_id": fid, "source_chip_id": chip_id, "images": [str(image)], "job_id": job, "clear": True}})
        return job
    return add


def put_sample(s: Services, sample_id: str, place: str) -> None:
    """Set a sample down at a named place by hand (world and ledger), e.g. a
    bare chip on the middle board where arm_right can take it."""
    from qumus_simlab.core.types import Pose
    from qumus_simlab.flows.bench import positions

    p = positions(s.lab.bench)[place]
    w = s.lab.world
    with w.lock:
        o = w.objects[sample_id]
        o.pose = Pose(x=p["x"], y=p["y"], z=o.pose.z, rotation=0.0)
        o.held_by, o.on = None, None
    s.store.upsert_sample(sample_id, s.store.get_sample(sample_id)["kind"], location=place)
