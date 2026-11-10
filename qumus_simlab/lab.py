"""Build the simulated laboratory: run store, clock, sim devices, SafetyGate, jobs.

One `LabSide` is one simulated lab table. The contract windows (simlab/modules.py)
are built on top of it; nothing outside this repository imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qumus_simlab.config import load_yaml
from qumus_simlab.core.clock import Clock, SimClock
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.runtime.jobs import ThreadJobService
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.safety.gate import SafetyGate
from qumus_simlab.safety.limits import Limits
from qumus_simlab.testbench.lab import SimLab, build_sim_lab
from qumus_simlab.testbench.persist import load_lab_state


@dataclass
class LabSide:
    """Everything below the contract windows: run store, clock, devices, safety, jobs."""

    store: RunStore
    clock: Clock
    lab: SimLab
    gate: SafetyGate
    jobs: ThreadJobService
    #: where the testbench writes camera pictures (None: pictures are not kept)
    image_dir: Path | None = None


def build_lab_side(db_path: str | Path = ":memory:", *, seed: int | None = None,
                   testbench_config: dict[str, Any] | None = None,
                   image_dir: str | Path | None = None,
                   state_path: str | Path | None = None) -> LabSide:
    """`image_dir`: where the testbench writes camera pictures; default
    <db dir>/images for a file database, none for ":memory:".
    `state_path`: saved sim world of an earlier lab process (testbench/persist.py);
    when it exists and the database already has a ledger, the world is restored
    from it and the ledger is left as is (a fresh database is always seeded)."""
    material_recognition = (testbench_config or {}).get("material_recognition")
    if material_recognition is not None:
        from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE

        if material_recognition != {"backend": "v1_lab_rgb", "capture_profile": V1_MATERIAL_CAPTURE_PROFILE}:
            raise ValueError("Unknown explicit material recognition/capture profile")
    clock = SimClock()
    store = RunStore(db_path, clock=clock)
    if image_dir is None and str(db_path) != ":memory:":
        image_dir = Path(db_path).parent / "images"
    restore = (state_path is not None and Path(state_path).exists()
               and bool(store.list_samples()))
    from qumus_simlab.testbench.bench_geometry import (
        PROFILE,
        WORKSPACE_META,
        admit_workspace_geometry,
        limits_for_bench,
    )

    try:
        admit_workspace_geometry(store, testbench_config, restoring=restore)
    except Exception:
        store.close()
        raise
    lab = build_sim_lab(clock, store=None if restore else store, seed=seed,
                        config=testbench_config, image_dir=image_dir, reactor_store=store)
    if restore:
        load_lab_state(lab, clock, state_path)
        lab.restore_optical_references(store)
    if material_recognition is not None:
        from qumus_simlab.perception.observations import RGBMicroscopeAnalyzer
        from qumus_simlab.testbench.render import MicroscopeRenderer

        lab.devices.get("microscope").renderer = MicroscopeRenderer(
            Path(image_dir)/"microscope" if image_dir is not None else None,
            material_capture_profile=material_recognition["capture_profile"])
        lab.analyzer = RGBMicroscopeAnalyzer(lab.analyzer)
    # the person's own 20x photo of each electrode site, taken before the run (v1, #204)
    from qumus_simlab.testbench.user_reference import add_user_references

    add_user_references(lab, store)
    # testbench: a person's new-film report stands for the act (testbench/world.py)
    lab.world.film_reports = lambda sid: len(
        ((store.get_sample(sid) or {}).get("props") or {}).get("film_reports") or [])
    if lab.bench.get("geometry_profile") and not restore:
        store.set_meta(WORKSPACE_META, PROFILE)
    gate = SafetyGate(lab.devices, Limits.load(limits_for_bench(lab.bench, load_yaml("limits.yaml"))),
                      clock, sink=lambda k, p: store.emit(k, p),
                      frames=lab.twin.delta)
    verification = load_yaml("verification.yaml")

    def ctx_factory(run_id: str, params: dict[str, Any], project_id: str | None) -> FlowContext:
        run_gate = SafetyGate(lab.devices, gate.limits, clock,
                              sink=lambda k, p: store.emit(k, p, run_id=run_id),
                              frames=lab.twin.delta)
        # share time-based heater state across runs
        run_gate._heater_on_since = gate._heater_on_since
        run_gate._heater_last_set = gate._heater_last_set
        return FlowContext(run_id=run_id, store=store, gate=run_gate, clock=clock,
                           locator=lab.locator, bench=lab.bench, verification=verification,
                           params=params, project_id=project_id, analyzer=lab.analyzer)

    jobs = ThreadJobService(store, gate, ctx_factory, bench=lab.bench)
    # testbench "crash" fault: the job worker dies; restart and recover like a supervisor
    def recover_after_crash() -> None:
        # A scheduled simulator callback may arrive after its owner disposes the
        # local service. It must not restart workers or touch the closed store.
        with jobs._submit_lock:
            if not jobs._closing:
                jobs.recover()
    lab.faults.on_crash = recover_after_crash
    return LabSide(store, clock, lab, gate, jobs,
                   Path(image_dir).resolve() if image_dir is not None else None)


