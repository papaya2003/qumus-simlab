"""Build a complete simulated lab: world + devices + perception + sample ledger.

The device ids match config/limits.yaml and the v1 bench (spec 11.1):

    arm_left, arm_right            G-code arms with vacuum nozzles, each in its own frame
    overview_camera                one camera looking down at the whole table
    exfoliator                     tape on two cylinders (upper one on a gantry) over the
                                   exfoliation stage
    exfoliation_heater / _fan / _vacuum
    transfer_stage                 coarse travel, fine XY and in-plane sample rotation
    transfer_stage_vacuum
    microscope                     independent vertical focus and objective turret
    stamp_heater                   heats the transfer stage (v1 transfer heater)
    transfer_stage_fan            forced-air cooling of the search/stacking stage
    transfer_arm, _vacuum          stamp holder (x/y/z translation stage) holding the stamp
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qumus_simlab.config import load_yaml
from qumus_simlab.core.clock import Clock
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import DeviceBundle
from qumus_simlab.devices.sim.devices import (
    SimArm,
    SimCamera,
    SimHeater,
    SimManipulator,
    SimMicroscope,
    SimStage,
    SimSwitch,
    SimTape,
)
from qumus_simlab.flows.bench import arm_base, positions, regions
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.runtime.twin import LabTwin
from qumus_simlab.testbench.camera_render import BenchCameraRenderer
from qumus_simlab.testbench.faults import FaultInjector
from qumus_simlab.testbench.locator import SimLocator, SimMicroscopeAnalyzer
from qumus_simlab.testbench.optics import electrode_pattern
from qumus_simlab.testbench.reactor_config import with_reactor
from qumus_simlab.testbench.render import MicroscopeRenderer
from qumus_simlab.testbench.world import Faults, SimObject, SimWorld, StampState, Surface


def _configured_target_geometry(props):
    """One validated target supplies both physical metal and its provenance."""
    from qumus_simlab.records.device_design import (
        ImportedHallBarTarget,
        parse_hallbar_target,
        target_import_binding,
    )
    from qumus_simlab.testbench.hallbar import electrode_pattern_from_target

    target = parse_hallbar_target(props["device_design_target"])
    pattern = electrode_pattern_from_target(target)
    source = {"source": target.source, "reference": target.reference,
              "units": "mm", "frame": "chip_center_axes"}
    if isinstance(target, ImportedHallBarTarget):
        binding = target_import_binding(target)
        if props.get("imported_target_binding") != binding:
            raise ValueError("Imported target differs from its configured source binding")
        source.update(source_binding=binding, coverage_contract=target.coverage_policy,
            layout_source_sha256=target.import_receipt.source_sha256,
            end_regions=[{"id": c.id, "role": c.role,
                "source_primitive_index": c.source_primitive_index,
                "points": [[x-target.size_mm[0]/2, y-target.size_mm[1]/2]
                           for x, y in c.points_mm]} for c in target.contacts])
    return target, pattern, source


@dataclass
class SimLab:
    world: SimWorld
    devices: DeviceBundle
    locator: SimLocator
    bench: dict[str, Any]
    analyzer: SimMicroscopeAnalyzer
    faults: FaultInjector = field(init=False)
    #: device readings + world objects after each command (`lab.frame` events)
    twin: LabTwin = field(init=False)

    def __post_init__(self) -> None:
        self.faults = FaultInjector(self)
        self.twin = LabTwin(self.devices, self.bench, self.world.snapshot)
        for d in self.devices.devices.values():
            d.injector = self.faults  # type: ignore[attr-defined]

    def preview(self, camera_id: str) -> dict[str, Any] | None:
        """What a camera shows now, for the UI (Camera.preview / Microscope.preview).
        None when the device has no preview or is offline."""
        dev = self.devices.devices.get(camera_id)
        frame = dev.preview() if dev is not None and hasattr(dev, "preview") else None
        if frame is None:
            return None
        return {"device": camera_id, "path": frame.saved_path(), "timestamp": frame.timestamp,
                "meta": frame.meta}

    def restore_optical_references(self, store: RunStore) -> None:
        """Add optical fixtures missing from an older saved software bench.

        Keep poses, flakes, jobs, existing photos and the physical RNG intact.
        A new picture is explicitly a reference, never a historical capture.
        """
        from qumus_simlab.records.materials import restore_simulation_identities

        restore_simulation_identities(store, self.world)
        for sample in store.list_samples(kind="chip"):
            if not sample["props"].get("electrodes"):
                continue
            chip = self.world.objects.get(sample["id"])
            if chip is None:
                continue
            # Upgrade only a saved, matching synthetic design. Adding a new
            # reference below must never be mistaken for old measured evidence.
            saved_pattern = getattr(chip, "electrodes", None)
            if (store.bench == "sim" and saved_pattern == electrode_pattern()
                    and not getattr(chip, "electrode_reference_only", False)
                    and sample["props"].get("image_source") == "simulated"
                    and not sample["props"].get("electrode_geometry")):
                from qumus_simlab.testbench.optics import electrode_geometry

                store.upsert_sample(chip.id, "chip", props={"electrode_geometry": electrode_geometry()})
            with self.world.lock:
                if not getattr(chip, "electrodes", None):
                    chip.electrodes = electrode_pattern()
                    # Persist on the object with the repaired polygons, so a
                    # later checkpoint cannot promote this display reference
                    # into evidence of the original target design.
                    chip.electrode_reference_only = True
            # The old purple 20x display thumbnail ("images.electrodes") is no
            # longer drawn (Jialei 2026-11-06): it was display only, off the
            # metal and in the pre-v1 colour. The electrode reference the flow
            # uses is the chip's user_reference_20x (testbench/user_reference.py).
        self.restore_material_state()

    def restore_material_state(self) -> None:
        """Project saved/prepared material into ordinary optics without exposure."""
        scope = self.devices.devices.get("reactor_microscope")
        plant = getattr(scope, "plant", None)
        backend = getattr(plant, "pair_backend", None)
        if backend is not None:
            backend.restore_world(self.world)
        material_metal = getattr(plant, "material_metal_backend", None)
        if material_metal is not None:
            material_metal.restore_world(self.world)


def build_sim_lab(clock: Clock, store: RunStore | None = None, seed: int | None = None,
                  config: dict[str, Any] | None = None,
                  image_dir: str | Path | None = None, *,
                  reactor_store: RunStore | None = None) -> SimLab:
    """Build scene-bound synthetic captures; image_dir enables retained files.

    Arbitrary reference photos have no matching pose or optical calibration.
    They remain available to offline renderer fixtures, not live device sensing.
    """
    cfg = config if config is not None else load_yaml("testbench.yaml")
    if (os.environ.get("QUMUS2_MICROSCOPE_REPLAY_DIR")
            or cfg.get("microscope", {}).get("replay_dir")):
        raise ValueError("Live microscope captures require scene-bound pixels; "
                         "reference-photo replay is available only for offline fixtures.")
    bench = load_yaml("bench.yaml")
    bench, cfg = with_reactor(bench, cfg)
    from qumus_simlab.testbench.bench_geometry import compose_bench_geometry

    if cfg.get("stamp", {}).get("discovery") is not None:
        from qumus_simlab.records.discovery import validate_generator_config

        bench["stamp_discovery"] = validate_generator_config(cfg)
    bench = compose_bench_geometry(bench, cfg)
    world = SimWorld(Faults.from_config(cfg), seed=cfg.get("seed", 0) if seed is None else seed,
                     model=cfg)
    world.clock_now = clock.now
    ledger = reactor_store if reactor_store is not None else store
    world.sample_properties = lambda sid: (ledger.get_sample(sid) or {}).get("props", {}) if ledger else {}

    def transfer_profile(stamp_id):
        reservation = ledger.reservations("sample").get(stamp_id) if ledger else None
        run = ledger.get_run(reservation["run_id"]) if reservation else None
        if run and run["flow"] == "stack_device" and run["params"].get("stamp_film") == "mica":
            return {"mica": run["params"]["support_assumptions"]}
        if (run and run["flow"] == "prepare_material_metal" and
                run["params"].get("protocol_id") == "qualitative-material-metal-v1"):
            return run["params"].get("receipt_inputs", {}).get("model_assumptions")
        return None

    world.active_transfer_profile = transfer_profile
    regs = regions(bench)
    world.slots = bench.get("slots", {})
    world.add_surface(Surface("transfer_stage", {k: tuple(v) for k, v in
                                                 regs["transfer_stage"].items()}))
    world.add_surface(Surface("exfoliation_stage", {k: tuple(v) for k, v in
                                                    regs["exfoliation_stage"].items()}))
    world.add_surface(Surface("reactor", {k: tuple(v) for k, v in
                                          regs["reactor_port"].items()}))
    th = cfg.get("heaters", {})
    ambient, tau, fan_tau = th.get("ambient_c", 22.0), th.get("tau_s", 60.0), th.get("fan_tau_s")
    mic = bench["microscope"]
    motion = cfg.get("motion", {})

    bench_r = None
    # Pixel controllers work on image-free benches too. This renderer composes
    # bytes only; it creates no writer, directory or temporary image file.
    mic_r = MicroscopeRenderer(None)
    if image_dir is not None:
        bench_r = BenchCameraRenderer(bench, Path(image_dir) / "camera")
        mic_r = MicroscopeRenderer(Path(image_dir) / "microscope")

    devices = DeviceBundle()
    arms = [SimArm(a, world, Pose(**bench[a]["home"]), base=arm_base(bench, a), clock=clock,
                   speed_mm_s=motion.get("arm_speed_mm_s", 0.0),
                   settle_s=motion.get("arm_settle_s", 0.0))
            for a in ("arm_left", "arm_right", *(
                ["arm_reactor"] if "reactor_module" in bench else []))]
    for a in arms:
        devices.add(a)

    def arm_states() -> dict[str, dict[str, Any]]:
        return {a.device_id: {**a.world_pose().model_dump(), "vacuum": a.vacuum_on()}
                for a in arms}

    devices.add(SimStage("transfer_stage", world, surface="transfer_stage", clock=clock,
                         speed_mm_s=motion.get("stage_speed_mm_s", 0.0),
                         fine_speed_mm_s=motion.get("fine_speed_mm_s", 0.5),
                         rotation_speed_deg_s=motion.get("rotation_speed_deg_s", 10.0)))
    devices.add(SimCamera("overview_camera", world, clock, bench_r, arm_states))
    tilt = cfg.get("microscope", {}).get("focus_tilt_per_mm")
    slope_bound = (math.sqrt(2)*abs(tilt) if isinstance(tilt, (int, float))
                   and not isinstance(tilt, bool) and math.isfinite(tilt) else None)
    projection = cfg.get("stamp", {}).get("dynamics", {}).get("refraction_xy_per_gap_mm")
    projection_bound = (math.hypot(*projection) if isinstance(projection, (list,tuple)) and len(projection) == 2
                        and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in projection) else None)
    def declared_norm(key):
        values = cfg.get("stamp", {}).get("dynamics", {}).get(key, [0., 0.])
        return (math.hypot(*values) if isinstance(values, (list, tuple)) and len(values) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in values) else None)

    devices.add(SimMicroscope("microscope", clock, world,
                              center=(mic["center"]["x"], mic["center"]["y"]),
                              fov_mm_10x=tuple(mic["fov_mm_10x"]),
                              focal_plane=mic.get("focal_plane_z", 0.0), renderer=mic_r,
                              stage=devices.get("transfer_stage"),
                              focus_speed_mm_s=motion.get("focus_speed_mm_s", 1.0),
                              lateral_focus_slope_norm_bound_mm_per_mm=slope_bound,
                              apex_drift_clip_per_axis_mm=cfg.get("stamp", {}).get("dynamics", {}).get("maximum_drift_mm", 0.15),
                              source_refraction_projection_norm_bound_mm_per_mm=projection_bound,
                              apex_lateral_drift_speed_norm_bound_mm_s=declared_norm("lateral_drift_mm_s"),
                              apex_approach_drift_norm_bound_mm_per_mm=declared_norm("approach_drift_mm_per_mm"),
                              dome_curvature_radius_mm=cfg.get("stamp", {}).get("dome_radius_mm", 1.5),
                              contact_slip_rate_bound_deg_s=abs(cfg.get("stamp", {}).get("dynamics", {}).get("contact_slip_deg_s", 0.)),
                              stamp_discovery_profile=cfg.get("stamp", {}).get("discovery")))
    stage_fan = SimSwitch("transfer_stage_fan")
    stamp_heater = SimHeater("stamp_heater", clock, ambient=ambient, tau_s=tau,
                             fan_tau_s=fan_tau, cooling=stage_fan.really_on)
    stage_fan.before_change = stamp_heater._update
    devices.add(stage_fan)
    devices.add(stamp_heater)
    ta = bench["transfer_arm"]
    world.stamp_geometry = {"bench": bench}
    world.stage_temp = stamp_heater.true_temperature
    scfg = cfg.get("stamp", {})
    devices.add(SimManipulator("transfer_arm", world, ta["home"],
                               sigma_mm=scfg.get("manipulator_sigma_mm", 0.0005), clock=clock,
                               speed_mm_s=motion.get("manipulator_speed_mm_s", 0.0),
                               travel_speed_mm_s=motion.get("manipulator_travel_speed_mm_s"),
                               fine_speed_mm_s=motion.get("manipulator_fine_speed_mm_s"),
                               fine_below_mm=motion.get("manipulator_fine_below_mm", 0.0)))
    world.transfer_dynamics.update(updated_at=clock.now(), holder_z=world.manip.z)
    devices.add(SimSwitch("transfer_arm_vacuum", on_change=world.transfer_grip,
                          hold_in_safe_state=True))
    fan = SimSwitch("exfoliation_fan")
    vacuum = SimSwitch("exfoliation_vacuum")
    exf_heater = SimHeater("exfoliation_heater", clock, ambient=ambient, tau_s=tau,
                           fan_tau_s=fan_tau, cooling=fan.really_on)
    fan.before_change = exf_heater._update
    for d in (fan, vacuum, exf_heater, SimSwitch("transfer_stage_vacuum"),
              SimTape("exfoliator", world, vacuum, exf_heater, clock=clock,
                      geometry=bench.get("exfoliator"))):
        devices.add(d)

    if "reactor_module" in bench:
        from qumus_simlab.testbench.reactor_devices import (
            ReactorPlant,
            SimGasFlow,
            SimReactorHeater,
            SimReactorMicroscope,
            SimReactorStage,
        )

        plant = ReactorPlant(world, clock, bench["reactor_module"],
                             Path(image_dir) / "reactor" if image_dir is not None else None,
                             store=reactor_store if reactor_store is not None else store)
        for device in (SimReactorStage(plant), SimReactorStage(plant, insertion=True), SimReactorHeater(plant),
                       SimGasFlow(plant), SimReactorMicroscope(plant)):
            devices.add(device)

    pos = positions(bench)
    focus_sigma = cfg.get("microscope", {}).get("best_focus_sigma", 0.3)
    if store is not None:
        # a new bench in a run store kept from an earlier serve: the ledger must
        # match the bench just built, so what an earlier bench made (flakes,
        # devices) goes, and the seeded samples start without their old props
        store.set_meta("simulation_identity_restore", "null")
        seeded = {s["id"] for s in cfg.get("initial_samples", [])}
        for old in store.list_samples():
            store.remove_sample(old["id"], reason="a new testbench was set up",
                                emit=old["id"] not in seeded)
    for s in cfg.get("initial_samples", []):
        props = dict(s.get("props", {}))
        typed_target = None
        geometry_source = {}
        if props.get("device_design_target"):
            typed_target, pattern, geometry_source = _configured_target_geometry(props)
            props.setdefault("layers_top_first", [])
        else:
            pattern = electrode_pattern() if props.get("electrodes") else []
        if pattern and typed_target is None:
            from qumus_simlab.testbench.optics import electrode_geometry

            props["electrode_geometry"] = electrode_geometry()
            geometry_source = {k: props["electrode_geometry"][k] for k in ("source", "reference")}
            geometry_source.update(units="mm", frame="chip_center_axes")
            # Explicit seeded fresh receiving surface, not inferred from an
            # absent flake ledger or regenerated on checkpoint restoration.
            props.setdefault("layers_top_first", [])
        if s["kind"] in ("chip", "carrier", "stamp"):
            p = pos[s["location"]]
            size = bench["slots"].get(s["location"], {}).get("size") or \
                {"chip": [10, 10], "carrier": [25, 40],
                 "stamp": bench.get("stamp_slide", {}).get("size")}.get(s["kind"]) or [12, 12]
            jitter = bench["slots"].get(s["location"], {}).get("size_jitter_mm")
            if typed_target is not None:
                # an electrode chip is the size its imported target declares (#206:
                # v1's measured 17.3 x 26.5 mm in target-a518-v1chip), wherever it lies
                size = list(typed_target.size_mm)
            elif jitter and s["kind"] == "chip":
                # hand-cleaved chips (owner, 2026-11-02): about the nominal size, each a
                # little different; its own stream so other random draws are unchanged
                jr = random.Random(f"chip-size:{world.seed}:{s['id']}")
                size = [round(float(n) + jr.uniform(-j, j), 2) for n, j in zip(size, jitter, strict=True)]
            props["size_mm"] = [float(size[0]), float(size[1])]
            if typed_target is not None and tuple(props["size_mm"]) != typed_target.size_mm:
                raise ValueError("Target geometry must match its actual configured chip size")
            world.add(SimObject(
                id=s["id"], kind=s["kind"], pose=Pose(x=p["x"], y=p["y"]),
                size_mm=(float(size[0]), float(size[1])), material=props.get("material"),
                electrodes=pattern,
                electrode_geometry_source=geometry_source if pattern else {},
                focus_offset=world.rng.gauss(0, focus_sigma) if s["kind"] == "chip" else 0.0))
        if s["kind"] == "stamp":
            r = world.rng
            world.stamps[s["id"]] = StampState(
                touch_z=scfg.get("touch_z", 5.0) + r.gauss(0, scfg.get("touch_z_sigma", 0.06)),
                apex_dx=r.gauss(0, scfg.get("apex_sigma_mm", 0.08)),
                apex_dy=r.gauss(0, scfg.get("apex_sigma_mm", 0.08)),
                melt_c=scfg.get("melt_c", 165.0) + r.gauss(0, scfg.get("melt_sigma_c", 2.0)),
                optic_shift=max(0.0, scfg.get("optic_shift_mm", 0.0)
                                + r.gauss(0, scfg.get("optic_shift_sigma_mm", 0.0))))
        if store is not None:
            if s["kind"] == "carrier" and store.bench == "sim":
                props["simulation_identity"] = {"source": "testbench_config.initial_samples",
                    "sample_id": s["id"], "material": props.get("material")}
            if pattern and image_dir is not None:
                props.update(image_source="simulated")  # no display thumbnail (see restore_optical_references)
            if s["kind"] == "tape":
                props = {**props, "roll_left_mm": world.tape.roll_left_mm}
            store.upsert_sample(s["id"], s["kind"], location=s["location"],
                                status=s.get("status", "available"), props=props)
    lab = SimLab(world=world, devices=devices, locator=SimLocator(world, regs, bench), bench=bench,
                 analyzer=SimMicroscopeAnalyzer(world))
    if store is not None:
        lab.restore_material_state()
    return lab
