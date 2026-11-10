"""Clock/control history, observed contact front and physical transfer agree."""

from __future__ import annotations

import copy
import math
import pickle

import pytest

from qumus_simlab.config import ideal_stacking_profile, load_yaml, simulation_profile
from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.library.inspect import InspectBench
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.safety.gate import SafetyGate
from qumus_simlab.testbench.lab import build_sim_lab
from qumus_simlab.testbench.locator import SimMicroscopeAnalyzer
from qumus_simlab.testbench.persist import lab_state, load_lab_state, save_lab_state
from qumus_simlab.testbench.world import STAMP_HOLDER, SimFlake, SimObject, StampState
from tests.conftest import quiet_bench


def mounted(*, offset=0, dynamics=True):
    cfg = quiet_bench(stage__sigma_mm=0)
    cfg["locator"].update(sigma_mm=0, sigma_deg=0)
    cfg["microscope"]["focus_tilt_per_mm"] = 0
    cfg["stamp"]["manipulator_sigma_mm"] = 0
    cfg["motion"].update(stage_speed_mm_s=0, fine_speed_mm_s=0,
                         rotation_speed_deg_s=0, manipulator_speed_mm_s=0)
    if not dynamics:
        cfg["stamp"]["dynamics"].update(lateral_drift_mm_s=[0, 0],
            approach_drift_mm_per_mm=[0, 0], contact_slip_deg_s=0)
    clock = SimClock()
    lab = build_sim_lab(clock, config=cfg)
    stage, scope = lab.devices.get("transfer_stage"), lab.devices.get("microscope")
    px, py = lab.world.surfaces["transfer_stage"].pivot()
    chip = SimObject("physics_chip", "chip", Pose(x=px+offset, y=py),
                     on="transfer_stage", size_mm=(4, 3))
    lab.world.add(chip)
    stage.move_to(Pose(x=-offset, y=scope.center[1]-py))
    w = lab.world
    over = lab.bench["transfer_arm"]["over_stage"]
    w.manip.x, w.manip.y, w.manip.z = over["x"], over["y"], 6.5
    sx, sy = w.seat_xy()
    stamp = SimObject("physics_stamp", "stamp", Pose(x=sx, y=sy), held_by=STAMP_HOLDER)
    w.add(stamp)
    st = StampState(5.3, 0, 0, 165)
    w.stamps[stamp.id] = st
    w.advance_transfer(clock.now())
    return lab, clock, chip, st


def test_edge_contact_dwell_is_independent_of_preview_and_checkpoint_frequency():
    outcomes = []
    for cadence in (None, 10, 7):
        lab, clock, chip, st = mounted()
        scope = lab.devices.get("microscope")
        chip.pose = Pose(x=scope.center[0]-2.03, y=scope.center[1])
        st.stack = [(SimFlake("carried", "hBN", 2, 0, 0, 100), .08, -.04)]
        lab.world.manip.z = 5.3-0.0004
        lab.world._transfer_epoch(clock.now())
        rngs = (lab.world.rng.getstate(), lab.world.observation_rng.getstate())
        for t in range(1, 1001):
            clock.advance(1)
            if cadence is not None and t % cadence == 0:
                scope._view()
                lab_state(lab, clock)
        lab.world.advance_transfer(clock.now())
        assert (lab.world.rng.getstate(), lab.world.observation_rng.getstate()) == rngs
        assert st.slip_rotation_deg > 0 and lab.world.apex()[3] < 0
        outcomes.append(copy.deepcopy(st))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("dwell,expected", [(0, 1), (5, 0)])
def test_pickup_requires_the_traversed_contact_front(dwell, expected):
    lab, clock, chip, st = mounted(dynamics=False)
    cfg = lab.world.model["stamp"]
    cfg["pickup_first"]["Gr"] = cfg["cold_pickup_factor"] = 1
    chip.flakes = [SimFlake("source", "Gr", 1, .1, 0, 100)]
    manip = lab.devices.get("transfer_arm")
    manip.move_axis_to("z", 5.29)
    clock.advance(dwell)
    front = lab.devices.get("microscope").capture().meta["contact"]["radius_mm"]
    assert (front > .1) == bool(dwell)
    manip.move_axis_to("z", 6.5)
    assert len(chip.flakes) == expected
    assert len(st.stack) == 1-expected


def test_full_outline_coverage_is_required_for_pickup_and_stack_release():
    lab, clock, chip, st = mounted(dynamics=False)
    cfg = lab.world.model["stamp"]
    cfg["pickup_first"]["Gr"] = cfg["cold_pickup_factor"] = 1
    shape = ((-.04, -.01), (.04, -.01), (.04, .01), (-.04, .01))
    flake = SimFlake("wide", "Gr", 1, .16, 0, 1600, outline_mm=shape)
    chip.flakes = [flake]
    manip = lab.devices.get("transfer_arm")
    manip.move_axis_to("z", 5.29)
    clock.advance(5)
    manip.move_axis_to("z", 6.5)
    assert chip.flakes == [flake] and not st.stack  # centre fits, whole outline does not
    chip.flakes = []
    st.stack = [(flake, .16, 0)]
    lab.world._transfer_epoch(clock.now())
    lab.world.stage_temp = lambda: 200
    manip.move_axis_to("z", 5.29)
    clock.advance(5)
    manip.move_axis_to("z", 6.5)
    assert not chip.flakes and st.stack and st.film


def test_inspection_covers_an_off_centre_mounted_chip():
    lab, clock, chip, _ = mounted(offset=5, dynamics=False)
    store = RunStore(":memory:", clock=clock)
    gate = SafetyGate(lab.devices, clock=clock)
    ctx = FlowContext("inspection", store, gate, clock, lab.locator, lab.bench, load_yaml("verification.yaml"))
    observed = InspectBench(["transfer_stage"]).verify(ctx).measurements
    assert chip.pose.x-lab.devices.get("transfer_stage").kinematics()["pivot_world_mm"][0] == 5
    assert observed["regions"]["transfer_stage"]
    assert any(d["at"] == "transfer_stage" for d in observed["regions"]["transfer_stage"])
    assert "transfer_stage" not in observed["empty"]


def history_case():
    """Resolved model inputs, with the normal model/noise/probability config.

    These component tests exercise World directly, not a quiet normal flow or
    a favourable stochastic seed. No production probability is overridden.
    """
    clock = SimClock()
    lab = build_sim_lab(clock, seed=17)
    w = lab.world
    over = lab.bench["transfer_arm"]["over_stage"]
    w.manip.x, w.manip.y, w.manip.z = over["x"], over["y"], 5.31
    x, y = w.head_xy()
    shape = ((-.003, -.003), (.003, -.003), (.003, .003), (-.003, .003))
    chip = SimObject("history_chip", "chip", Pose(x=x, y=y), on="transfer_stage",
        size_mm=(4, 3), flakes=[SimFlake("far", "hBN", 2, .13, 0, 36, shape),
                              SimFlake("near", "hBN", 2, .05, 0, 36, shape),
                              SimFlake("outside", "hBN", 2, .5, 0, 36, shape)])
    w.add(chip)
    w.add(SimObject("history_stamp", "stamp", Pose(x=x, y=y), held_by=STAMP_HOLDER))
    stamp = StampState(5.3, 0, 0, 165)
    w.stamps["history_stamp"] = stamp
    w.stage_temp = lambda: 90
    w._transfer_epoch(clock.now())
    return lab, clock, chip, stamp


def grow_then_recede(lab, clock):
    w = lab.world
    w.manip_move("z", 5.28)
    clock.advance(1.5)
    w.advance_transfer(clock.now())
    assert set(w.stamps["history_stamp"].contact_history["pickup"]) == {"far", "near"}
    w.manip_move("z", 5.29995)
    clock.advance(5)
    w.advance_transfer(clock.now())
    assert w.stamps["history_stamp"].front_radius_mm < .05


def test_receded_front_keeps_whole_contact_and_chip_ordered_draws(monkeypatch):
    lab, clock, chip, stamp = history_case()
    grow_then_recede(lab, clock)
    assert list(stamp.contact_history["pickup"])[0] == "near"  # reached first, not drawn first
    values, calls = iter([.89, .96]), []
    def draw():
        value = next(values)
        calls.append(value)
        return value
    monkeypatch.setattr(lab.world.rng, "random", draw)
    lab.world.manip_move("z", 5.31)
    # Original chip order: far passes bare hBN .9; near fails onto-stack .95.
    assert [f.id for f, _, _ in stamp.stack] == ["far"]
    assert [f.id for f in chip.flakes] == ["near", "outside"]
    assert calls == [.89, .96] and stamp.contact_history is None


@pytest.mark.parametrize("temperature,draw,picked", [(90, .901, False), (90, .899, True),
                                                    (22, .271, False), (22, .269, True)])
def test_contact_memory_preserves_bare_probability_and_cold_factor(monkeypatch, temperature, draw, picked):
    lab, clock, chip, stamp = history_case()
    chip.flakes = chip.flakes[:1]
    grow = lab.world
    grow.manip_move("z", 5.28)
    clock.advance(1.5)
    grow.advance_transfer(clock.now())
    grow.stage_temp = lambda: temperature
    calls = []
    monkeypatch.setattr(grow.rng, "random", lambda: calls.append(draw) or draw)
    grow.manip_move("z", 5.31)
    assert bool(stamp.stack) == picked and bool(chip.flakes) != picked
    assert calls == [draw]


@pytest.mark.parametrize("film", ["PC", "mica"])
@pytest.mark.parametrize("draw,picked", [(.919, True), (.921, False)])
def test_configured_support_pickup_keeps_its_original_probability(monkeypatch, film, draw, picked):
    from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation

    lab, clock, chip, stamp = history_case()
    chip.flakes = chip.flakes[:1]
    w = lab.world
    w.active_transfer_profile = lambda _: MaterialMetalSimulation().model_dump()
    w.sample_properties = lambda _: {"support": {"film_type": film}}
    w.manip_move("z", 5.28)
    clock.advance(1.5)
    w.advance_transfer(clock.now())
    calls = []
    monkeypatch.setattr(w.rng, "random", lambda: calls.append(draw) or draw)
    w.manip_move("z", 5.31)
    assert bool(stamp.stack) == picked and bool(chip.flakes) != picked and calls == [draw]


def test_partial_patch_union_and_moved_maximum_cannot_invent_whole_contact():
    lab, _, chip, stamp = history_case()
    w = lab.world
    x, y = w.head_xy()
    long = SimFlake("long", "Gr", 1, 0, 0, 3200,
                    ((-.08, -.01), (.08, -.01), (.08, .01), (-.08, .01)))
    chip.flakes = [long]
    history = w._contact_footprint(None, (x-.05, y), .07, [], [])
    history = w._contact_footprint(history, (x+.05, y), .07, [], [])
    assert all(any(math.hypot(px-centre, py) <= .07 for centre in (-.05, .05))
               for px, py in long.outline_mm)
    assert not history["pickup"]  # union covers it, neither complete footprint does
    chip.flakes = [SimFlake("never", "hBN", 1, .18, 0, 36,
                            ((-.003, -.003), (.003, -.003), (.003, .003), (-.003, .003)))]
    history = w._contact_footprint(None, (x, y), .12, [], [])
    history = w._contact_footprint(history, (x+.1, y), .01, [], [])
    assert all(math.hypot(.18+px-.1, py) < .12 for px, py in chip.flakes[0].outline_mm)
    assert not history["pickup"]
    stamp.contact_history = history
    assert stamp.contact_history["chip_id"] == chip.id


def test_changed_outline_chip_and_refilm_invalidate_witnesses():
    lab, clock, chip, stamp = history_case()
    grow_then_recede(lab, clock)
    w = lab.world
    chip.flakes[0].outline_mm = ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))
    w._transfer_epoch(clock.now())
    assert "far" not in stamp.contact_history["pickup"]
    chip.on = None
    replacement = SimObject("other_chip", "chip", chip.pose, on="transfer_stage", size_mm=(4, 3))
    w.add(replacement)
    w._transfer_epoch(clock.now())
    assert stamp.contact_history["chip_id"] == replacement.id and not stamp.contact_history["pickup"]
    w.film_reports = lambda _: 1
    w._refilm("history_stamp")
    assert stamp.contact_history is None and stamp.front_radius_mm == 0


@pytest.mark.parametrize("fault", ["pickup", "film"])
def test_contact_history_does_not_bypass_fault_or_torn_film(monkeypatch, fault):
    lab, clock, chip, stamp = history_case()
    grow_then_recede(lab, clock)
    if fault == "pickup":
        lab.world.faults.pickup_fail_next = 1
    else:
        stamp.film = False
    monkeypatch.setattr(lab.world.rng, "random", lambda: pytest.fail("No transfer draw is permitted"))
    lab.world.manip_move("z", 5.31)
    assert len(chip.flakes) == 3 and not stamp.stack and stamp.contact_history is None
    assert lab.world.faults.pickup_fail_next == 0
    if fault == "pickup":
        lab.world.manip_move("z", 5.28)
        lab.world.manip_move("z", 5.31)  # new zero-dwell episode has no witness
        assert len(chip.flakes) == 3 and not stamp.stack


def test_grip_boundary_clears_contact_history_without_a_transfer_draw(monkeypatch):
    lab, clock, _, stamp = history_case()
    grow_then_recede(lab, clock)
    monkeypatch.setattr(lab.world.rng, "random", lambda: pytest.fail("Grip is not a transfer trial"))
    lab.world.transfer_grip(False)
    assert stamp.contact_history is None and lab.world.transfer_dynamics["epoch"] is None


@pytest.mark.parametrize("film,configured,temp,draw,released,retained,draw_count", [
    ("PC", False, 164, .1, False, True, 0),
    ("PC", False, 165, .1, True, False, 0),
    ("PC", True, 166, .1, False, True, 0),
    ("PC", True, 167, .89, True, False, 1),
    ("PC", True, 167, .91, False, True, 1),
    ("mica", True, 146, .1, False, True, 0),
    ("mica", True, 147, .89, True, True, 1),
    ("mica", True, 147, .91, False, True, 1),
])
def test_receded_whole_stack_release_preserves_support_temperature_and_probability(
        monkeypatch, film, configured, temp, draw, released, retained, draw_count):
    from qumus_simlab.core.polygons import transform
    from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation

    lab, clock, chip, stamp = history_case()
    w = lab.world
    carried = chip.flakes[0]
    carried.origin = {"source_chip_id": "original", "batch_id": "stable"}
    chip.flakes = []
    chip.pose = Pose(x=chip.pose.x, y=chip.pose.y, rotation=17)
    stamp.stack = [(carried, .13, 0)]
    profile = MaterialMetalSimulation().model_dump()
    w.active_transfer_profile = lambda _: profile if configured else None
    w.sample_properties = lambda sid: ({"support": {"film_type": film}} if sid == "history_stamp"
                                       else {"metal_coating": {"metal": "Pd"}})
    w.stage_temp = lambda: temp
    w._transfer_epoch(clock.now())
    w.manip_move("z", 5.28)
    clock.advance(1.5)
    w.advance_transfer(clock.now())
    assert stamp.contact_history["release"] is not None
    w.manip_move("z", 5.29995)
    clock.advance(5)
    w.advance_transfer(clock.now())
    assert stamp.front_radius_mm < .05 and stamp.slip_rotation_deg > 0
    assert stamp.contact_history["release"]["stack"] == stamp.stack
    before_flake, ox, oy = stamp.stack[0]
    apex = w.apex()[2]
    calls = []
    monkeypatch.setattr(w.rng, "random", lambda: calls.append(draw) or draw)
    w.manip_move("z", 5.31)
    assert bool(chip.flakes) == released and bool(stamp.stack) != released
    assert stamp.film == retained and len(calls) == draw_count
    if released:
        landed = chip.flakes[0]
        assert landed.id == before_flake.id and landed.origin == before_flake.origin
        assert chip.local_to_world(landed.x_mm, landed.y_mm) == pytest.approx((apex[0]+ox, apex[1]+oy))
        for actual, expected in zip(transform(landed.outline_mm, rotation_deg=17),
                                    before_flake.outline_mm, strict=True):
            assert actual == pytest.approx(expected)


@pytest.mark.parametrize("blocked", ["incomplete_stack", "release_fault"])
def test_release_requires_one_whole_stack_witness_and_preserves_fail_next(monkeypatch, blocked):
    lab, clock, chip, stamp = history_case()
    w = lab.world
    stamp.stack = [(chip.flakes[0], .13 if blocked == "release_fault" else .6, 0)]
    chip.flakes = []
    w.stage_temp = lambda: 180
    w._transfer_epoch(clock.now())
    w.manip_move("z", 5.28)
    clock.advance(1.5)
    w.advance_transfer(clock.now())
    if blocked == "release_fault":
        assert stamp.contact_history["release"] is not None
        w.faults.release_fail_next = 1
    else:
        assert stamp.contact_history["release"] is None
    monkeypatch.setattr(w.rng, "random", lambda: pytest.fail("No release trial is permitted"))
    w.manip_move("z", 5.31)
    assert stamp.stack and not chip.flakes and stamp.film
    assert w.faults.release_fail_next == 0


@pytest.mark.parametrize("carried", [False, True])
def test_hot_zero_front_episode_preserves_pending_release_fault(monkeypatch, carried):
    lab, clock, chip, stamp = history_case()
    w = lab.world
    stamp.stack = [(chip.flakes[0], .13, 0)] if carried else []
    chip.flakes = []
    w.stage_temp = lambda: 180
    w._transfer_epoch(clock.now())
    w.manip_move("z", 5.28)
    assert stamp.front_radius_mm == 0
    assert not stamp.contact_history["pickup"] and stamp.contact_history["release"] is None
    before = copy.deepcopy(stamp.stack)
    w.faults.release_fail_next = 1
    monkeypatch.setattr(w.rng, "random", lambda: pytest.fail("Zero contact cannot draw a release trial"))
    w.manip_move("z", 5.31)  # same Clock time: no actual footprint was traversed
    assert w.faults.release_fail_next == 1
    assert stamp.stack == before and stamp.film and not chip.flakes
    assert stamp.contact_history is None


def test_hot_positive_partial_front_keeps_original_release_fault_order(monkeypatch):
    lab, clock, chip, stamp = history_case()
    w = lab.world
    stamp.stack = [(chip.flakes[0], .6, 0)]
    chip.flakes = []
    w.stage_temp = lambda: 180
    w._transfer_epoch(clock.now())
    w.manip_move("z", 5.28)
    clock.advance(.5)
    w.advance_transfer(clock.now())
    assert stamp.front_radius_mm > 0 and stamp.contact_history["release"] is None
    before = copy.deepcopy(stamp.stack)
    w.faults.release_fail_next = 1
    monkeypatch.setattr(w.rng, "random", lambda: pytest.fail("A pending fault prevents the release draw"))
    w.manip_move("z", 5.31)
    assert w.faults.release_fail_next == 0  # fault precedes the whole-stack check
    assert stamp.stack == before and stamp.film and not chip.flakes


def test_contact_history_clock_polling_and_checkpoint_roundtrip(tmp_path):
    outcomes = []
    for cadence in ((), (.13, .37, .98, 1.27)):
        lab, clock, _, stamp = history_case()
        lab.world.manip_move("z", 5.28)
        for at in (*cadence, 1.5):
            clock.advance(at-clock.now())
            lab.world.advance_transfer(clock.now())
        outcomes.append(copy.deepcopy(stamp))
    assert outcomes[0] == outcomes[1]
    checkpoint = tmp_path/"history.pkl"
    save_lab_state(lab, clock, checkpoint)
    before = checkpoint.read_bytes()
    restored_clock = SimClock()
    restored = build_sim_lab(restored_clock, seed=17)
    assert load_lab_state(restored, restored_clock, checkpoint)
    for world, time in ((lab.world, clock), (restored.world, restored_clock)):
        world.manip_move("z", 5.29995)
        time.advance(5)
        world.advance_transfer(time.now())
    assert restored.world.stamps == lab.world.stamps
    assert restored.world.transfer_dynamics == lab.world.transfer_dynamics
    assert restored.world.rng.getstate() == lab.world.rng.getstate()
    assert restored.world.observation_rng.getstate() == lab.world.observation_rng.getstate()
    assert checkpoint.read_bytes() == before


def test_ideal_stacking_profile_is_opt_in_and_retains_physical_defaults():
    from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation

    normal = simulation_profile()
    ideal = ideal_stacking_profile()
    assert simulation_profile() == normal
    assert "simulation_condition" not in normal
    assert "observation_noise_scale" not in normal["microscope"]
    assert ideal["simulation_condition"] == "ideal_stacking_v1"
    for section in ("motion", "heaters", "exfoliation", "initial_samples", "process"):
        assert ideal[section] == normal[section]
    for field in ("dome_radius_mm", "chip_thickness_mm", "touch_z", "optic_shift_mm",
                  "through_stamp_contrast", "fringe_max_gap_mm", "fringe_wavelength_mm",
                  "melt_c", "pickup_min_temp_c", "cold_pickup_factor", "crush_depth_mm"):
        assert ideal["stamp"][field] == normal["stamp"][field]
    assert ideal["stamp"]["dynamics"]["contact_front_tau_s"] == normal["stamp"]["dynamics"]["contact_front_tau_s"]
    for field in ("focus_tilt_per_mm", "depth_of_focus_10x", "contact_min_radius_mm"):
        assert ideal["microscope"][field] == normal["microscope"][field]
    expected = MaterialMetalSimulation().model_dump(mode="json")
    for film in ("PC", "mica"):
        expected[film].update(pickup_probability=1, release_probability=1)
    assert ideal["reactor"]["material_metal"] == expected
    assert expected["mica"]["pickup_c"] == 80 and expected["mica"]["release_c"] == 150
    assert expected["mica"]["support_retained_on_stamp"] is True
    assert expected["success_probability"] == .9
    ideal["stamp"]["dynamics"]["lateral_drift_mm_s"][0] = 999
    assert ideal_stacking_profile()["stamp"]["dynamics"]["lateral_drift_mm_s"] == [0, 0]


@pytest.mark.parametrize("seed", [7, 23])
def test_ideal_fresh_world_removes_actual_sampled_static_errors(seed):
    cfg = ideal_stacking_profile()
    lab = configured_lab(SimClock(), cfg, seed=seed)
    assert all(obj.focus_offset == 0 for obj in lab.world.objects.values() if obj.kind == "chip")
    assert lab.world.stamps
    for stamp in lab.world.stamps.values():
        assert (stamp.apex_dx, stamp.apex_dy, stamp.drift_xy_mm, stamp.slip_rotation_deg) == (0, 0, (0, 0), 0)
        assert stamp.touch_z == cfg["stamp"]["touch_z"]
        assert stamp.melt_c == cfg["stamp"]["melt_c"]
        assert stamp.optic_shift == cfg["stamp"]["optic_shift_mm"]
    assert lab.devices.get("transfer_arm").sigma == 0
    assert lab.world.faults.stage_sigma_mm == 0
    assert lab.world.faults.pick_fail_prob == 0
    assert lab.world.model["simulation_condition"] == "ideal_stacking_v1"


def configured_lab(clock, config, *, seed=None):
    """The integrated reactor requires the ordinary shared sample ledger."""
    return build_sim_lab(clock, store=RunStore(":memory:", clock=clock), config=config, seed=seed)


@pytest.mark.parametrize("flow", ["stack_device", "prepare_material_metal"])
def test_ideal_existing_admission_reaches_the_active_transfer_assumptions(flow):
    from qumus_simlab.flows.library.material_metal import MaterialMetalPrepareParams
    from qumus_simlab.flows.library.material_metal_device import device_flow
    from qumus_simlab.flows.library.stacking import StackDeviceFlow, StackParams
    from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation
    from qumus_simlab.records.reactor_requirements import ReactorRequirements

    cfg, clock = ideal_stacking_profile(), SimClock()
    store = RunStore(":memory:", clock=clock)
    lab = build_sim_lab(clock, store=store, config=cfg)
    assumptions = cfg["reactor"]["material_metal"]
    stamp_id = next(iter(lab.world.stamps))
    project_id = store.create_project("Ideal admission unit fixture")
    store.upsert_sample(stamp_id, "stamp", props={"support": {"film_type": "mica"}})
    store.upsert_sample("chip_001", "chip", location="mid_chip_1")
    search_id = store.create_run("search_chip", {}, project_id=project_id)
    store.set_run_status(search_id, "succeeded")
    # Small declared catalogue records exercise real admission, without a
    # source-generation run or bypassing the current evidence validators.
    for fid, material, half in (("admission_top", "hBN", .02), ("admission_lower", "Gr", .01)):
        store.upsert_sample(fid, "flake", location="chip_001", status="catalogued", props={
            "material": material, "rel_corner_mm": [.1, .1], "images": {"context": "unit-optical.png"},
            "optical_geometry": {"outline_mm": [[-half, -half], [half, -half], [half, half], [-half, half]],
                "units": "mm", "frame": "flake_center_chip_axes", "source": "simulated_optical_outline",
                "source_chip_id": "chip_001", "image": "unit-optical.png", "job_id": search_id},
            "pickup_assessment": {"flake_id": fid, "source_chip_id": "chip_001", "images": ["unit-optical.png"],
                "job_id": search_id, "clear": True}})
    if flow == "stack_device":
        declared = StackDeviceFlow()
        declared._bench = lab.bench
        params = StackParams(flake_ids=["admission_top"], target_chip_id="chip_e01", stamp_id=stamp_id)
    else:
        target = "reactor_chip_01"
        requirement = store.emit("experiment.requirements", {"project_id": project_id,
            "requirements": ReactorRequirements(reactive_material="Gr", metal="Pd").model_dump()})
        store.upsert_sample(target, "chip", props={"metal_coating": {"metal": "Pd", "substrate": "silicon",
            "preparation_id": "unit-coating", "prepared_by": "unit fixture", "source": "synthetic_fixture"}})
        store.update_project_state(project_id, reactor_selection={"confirmed_ids": ["admission_top", "admission_lower"],
            "chip_id": target, "requirements_event_id": requirement})
        declared = device_flow(lab.bench)
        params = MaterialMetalPrepareParams(project_id=project_id, flake_ids=["admission_top", "admission_lower"],
                                            target_chip_id=target, stamp_id=stamp_id)
    resolved = declared.resolve_params(params, store)
    run_id = store.create_run(flow, resolved.model_dump(mode="json"), project_id=project_id)
    declared.reserve(resolved, store, run_id)
    saved = store.get_run(run_id)["params"]
    assert saved["stamp_film"] == "mica"
    active = lab.world.active_transfer_profile(stamp_id)
    effective = MaterialMetalSimulation.model_validate(active)
    assert effective.mica.pickup_probability == effective.mica.release_probability == 1
    assert effective.mica.pickup_c == 80 and effective.mica.release_c == 150
    assert effective.mica.support_retained_on_stamp is True
    if flow == "prepare_material_metal":
        assert effective.PC.pickup_probability == effective.PC.release_probability == 1
        assert active == assumptions
        assert saved["receipt_inputs"]["model_assumptions"] == assumptions
    else:
        assert active == {"mica": assumptions["mica"]}
        assert saved["support_assumptions"] == assumptions["mica"]


def test_ideal_configuration_does_not_rewrite_conflicting_frozen_job_assumptions():
    from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation

    clock = SimClock()
    store = RunStore(":memory:", clock=clock)
    lab = build_sim_lab(clock, store=store, config=ideal_stacking_profile())
    original = MaterialMetalSimulation().model_dump(mode="json")
    run_id = store.create_run("prepare_material_metal", {"protocol_id": original["protocol_id"],
        "receipt_inputs": {"model_assumptions": original}})
    stamp_id = next(iter(lab.world.stamps))
    store.reserve("sample", stamp_id, run_id, "prepare_material_metal")
    assert lab.world.active_transfer_profile(stamp_id) == original
    assert store.get_run(run_id)["params"]["receipt_inputs"]["model_assumptions"] == original


def analyzer_frame(*, sharpness=5):
    flake = {"sim_id": "observation_flake", "material": "Gr", "layers": 1,
             "dx_mm": .017, "dy_mm": -.013, "area_um2": 123.4,
             "visible_area_um2": 123.4, "outline_mm": [[.01, -.02], [.02, -.02], [.02, -.01]],
             "geometry_source": "simulation"}
    return Frame("microscope", 0, meta={"objective": 10, "sharpness_chip": sharpness,
        "sharpness_stamp": sharpness, "flakes": [flake], "stamp_flakes": [flake],
        "chip_corners": [{"corner": "bl", "dx_mm": .1, "dy_mm": -.2}],
        "fringes": {"dx_mm": .03, "dy_mm": -.04, "gap_mm": .001, "rings": 3},
        "contact": {"dx_mm": .02, "dy_mm": -.03, "radius_mm": .1, "film": True}})


def test_ideal_observation_has_no_residual_position_area_or_random_omissions():
    lab = configured_lab(SimClock(), ideal_stacking_profile())
    analyzer = lab.analyzer
    frame = analyzer_frame()
    for _ in range(12):
        source, = analyzer.find_flakes(frame, "Gr")
        carried, = analyzer.find_stamp_flakes(frame)
        for detection in (source, carried):
            assert (detection.dx_mm, detection.dy_mm, detection.area_um2) == (.017, -.013, 123.4)
            assert detection.thickness_label == "monolayer" and detection.meta["outline_complete"]
        fringe = analyzer.fringes(frame)
        assert (fringe.dx_mm, fringe.dy_mm, fringe.gap_mm) == (.03, -.04, .001)
        corner = analyzer.find_corner(frame)
        assert (corner.dx_mm, corner.dy_mm) == (.1, -.2)
        contact = analyzer.contact(frame)
        assert (contact.dx_mm, contact.dy_mm, contact.radius_mm) == (.02, -.03, .1)
    assert analyzer.find_flakes(frame, "hBN") == []
    blurred = analyzer_frame(sharpness=2.99)
    assert analyzer.find_flakes(blurred, "Gr") == analyzer.find_stamp_flakes(blurred) == []
    assert analyzer.find_corner(blurred) is analyzer.fringes(blurred) is analyzer.contact(blurred) is None
    frame.meta["contact"]["radius_mm"] = .019
    assert analyzer.contact(frame) is None
    frame.meta["flakes"][0]["visible_area_um2"] = 100
    assert analyzer.find_flakes(frame, "Gr")[0].meta["outline_complete"] is False
    frame.meta["flakes"] = frame.meta["stamp_flakes"] = []
    assert analyzer.find_flakes(frame, "Gr") == analyzer.find_stamp_flakes(frame) == []


def test_observation_noise_scale_default_preserves_normal_draws_and_results():
    normal = simulation_profile()
    explicit = copy.deepcopy(normal)
    explicit["microscope"]["observation_noise_scale"] = 1
    a = configured_lab(SimClock(), normal, seed=11)
    b = configured_lab(SimClock(), explicit, seed=11)
    assert SimMicroscopeAnalyzer(a.world).noise_scale == 1
    frame = analyzer_frame()
    for _ in range(12):
        for method, args in (("find_flakes", ("Gr",)), ("find_stamp_flakes", ()),
                             ("fringes", ()), ("find_corner", ()), ("contact", ())):
            assert getattr(a.analyzer, method)(frame, *args) == getattr(b.analyzer, method)(frame, *args)
    assert a.world.observation_rng.getstate() == b.world.observation_rng.getstate()


def test_ideal_locator_is_exact_and_keeps_real_region_visibility():
    clock = SimClock()
    lab = configured_lab(clock, ideal_stacking_profile())
    gate = SafetyGate(lab.devices, clock=clock)
    frame = gate.camera("overview_camera").capture()
    box = lab.locator.regions["workspace"]
    expected = [(obj["kind"], obj["x"], obj["y"], obj["rotation"])
                for obj in frame.meta["sim_objects"]
                if obj["held_by"] in (None, STAMP_HOLDER)
                and box["x"][0] <= obj["x"] <= box["x"][1]
                and box["y"][0] <= obj["y"] <= box["y"][1]]
    assert expected
    for _ in range(7):
        detections = lab.locator.locate(frame, region="workspace", frame_of="world")
        assert [(d.label, d.pose.x, d.pose.y, d.pose.rotation) for d in detections] == expected
        assert all(d.confidence == round(.865, 2) for d in detections)
    invisible = copy.deepcopy(frame)
    for obj in invisible.meta["sim_objects"]:
        obj["held_by"] = "arm_right"
    assert lab.locator.locate(invisible, region="workspace", frame_of="world") == []
    assert lab.locator.locate(Frame("overview_camera", 0, meta={"sim_objects": []}),
                              region="workspace", frame_of="world") == []


@pytest.mark.parametrize("saved_ideal,current_ideal", [(False, False), (False, True), (True, True), (True, False)])
def test_ideal_checkpoint_uses_existing_mechanics_restore_guard(tmp_path, saved_ideal, current_ideal):
    saved_clock = SimClock()
    saved = configured_lab(saved_clock, ideal_stacking_profile() if saved_ideal else simulation_profile())
    saved_clock.advance(12.5)
    checkpoint = tmp_path / "condition.pkl"
    save_lab_state(saved, saved_clock, checkpoint)
    original = checkpoint.read_bytes()
    fresh_clock = SimClock()
    fresh = configured_lab(fresh_clock, ideal_stacking_profile() if current_ideal else simulation_profile())
    before = pickle.dumps(lab_state(fresh, fresh_clock))
    if saved_ideal != current_ideal:
        with pytest.raises(PreconditionFailed, match="incompatible stage/focus mechanics"):
            load_lab_state(fresh, fresh_clock, checkpoint)
        assert pickle.dumps(lab_state(fresh, fresh_clock)) == before
    else:
        assert load_lab_state(fresh, fresh_clock, checkpoint)
        assert fresh_clock.now() == saved_clock.now()
        assert fresh.world.stamps == saved.world.stamps
        assert fresh.world.rng.getstate() == saved.world.rng.getstate()
    assert checkpoint.read_bytes() == original


def rigid_xy(angle, point):
    """Independent expected 2D rotation; do not use production transforms."""
    theta = math.radians(angle)
    x, y = point
    return (math.cos(theta)*x-math.sin(theta)*y, math.sin(theta)*x+math.cos(theta)*y)


@pytest.mark.parametrize("material", ["Gr", "unlisted_material"])
@pytest.mark.parametrize("roundtrip", [False, True], ids=["uninterrupted", "restored"])
def test_ideal_guarded_pickup_release_preserves_full_coordinate_chain(material, roundtrip, monkeypatch, tmp_path):
    cfg, clock = ideal_stacking_profile(), SimClock()
    lab = configured_lab(clock, cfg)
    w = lab.world
    region = lab.bench["stations"]["transfer_stage"]["region"]
    pivot = (sum(region["x"])/2, sum(region["y"])/2)
    scope_xy = tuple(lab.bench["microscope"]["center"][axis] for axis in ("x", "y"))
    source_mount, target_mount = (1.2, -.7), (8.4, 1.1)
    shape = ((-.004, -.002), (.005, -.002), (.003, .004))
    local = ((.018, -.012), (-.026, .019))
    # A declared initial scene, before any command. Every later state change
    # below is a real guarded grip, stage/holder motion, heating or contact.
    source = SimObject("chain_source", "chip", Pose(x=pivot[0]+source_mount[0],
        y=pivot[1]+source_mount[1], rotation=24), size_mm=(4, 3), on="transfer_stage",
        flakes=[SimFlake(f"chain_{i}", material, 1, x, y, 27, shape) for i, (x, y) in enumerate(local)]
        + [SimFlake("outside_front", material, 1, .45, 0, 27, shape)])
    target = SimObject("chain_target", "chip", Pose(x=pivot[0]+target_mount[0],
        y=pivot[1]+target_mount[1], rotation=-13), size_mm=(4, 3), on="transfer_stage")
    w.add(source)
    w.add(target)
    stamp_obj = next(obj for obj in w.objects.values() if obj.kind == "stamp")
    stamp_obj.pose = Pose(x=w.seat_xy()[0], y=w.seat_xy()[1])
    stamp = w.stamps[stamp_obj.id]
    events = []
    gate = SafetyGate(lab.devices, clock=clock, sink=lambda kind, data: events.append((kind, data)))
    stage, holder, heater = gate.stage("transfer_stage"), gate.manipulator("transfer_arm"), gate.heater("stamp_heater")
    gate.switch("transfer_arm_vacuum").set(True)
    assert stamp_obj.held_by == STAMP_HOLDER and stamp.stack == []
    gate.switch("transfer_stage_vacuum").set(True)
    fine = (.06, -.04)
    source_center = (scope_xy[0]+.015, scope_xy[1]-.012)
    rotated_mount = rigid_xy(17, source_mount)
    stage.move_to(Pose(x=source_center[0]-pivot[0]-rotated_mount[0]-fine[0],
        y=source_center[1]-pivot[1]-rotated_mount[1]-fine[1], rotation=17))
    stage.move_fine_to(*fine)
    assert (source.pose.x, source.pose.y, source.pose.rotation) == pytest.approx((*source_center, 41))
    over = lab.bench["transfer_arm"]["over_stage"]
    holder.move_axis_to("y", over["y"]-.009)
    holder.move_axis_to("x", over["x"]+.012)
    pickup_apex = (scope_xy[0]+.012, scope_xy[1]-.009)
    heater.set_target(80)
    clock.advance(300)
    assert 60 < heater.read_temperature() < stamp.melt_c
    expected_offsets = []
    for point in local:
        dx, dy = rigid_xy(41, point)
        expected_offsets.append((source_center[0]+dx-pickup_apex[0], source_center[1]+dy-pickup_apex[1]))
    before_contact = clock.now()
    holder.move_axis_to("z", 5.29)
    clock.advance(5)
    contact = gate.microscope("microscope").capture().meta["contact"]
    assert contact["radius_mm"] > .08
    monkeypatch.setattr(w.rng, "random", lambda: .999)  # p1, including the unknown-material fallback
    holder.move_axis_to("z", 6.5)
    assert clock.now() > before_contact+5
    assert [flake.id for flake in source.flakes] == ["outside_front"]
    assert [flake.id for flake, _, _ in stamp.stack] == ["chain_0", "chain_1"]
    expected_carried_outline = [rigid_xy(41, point) for point in shape]
    for (flake, ox, oy), expected in zip(stamp.stack, expected_offsets, strict=True):
        assert (ox, oy) == pytest.approx(expected, abs=1e-10)
        for actual, point in zip(flake.outline_mm, expected_carried_outline, strict=True):
            assert actual == pytest.approx(point, abs=1e-10)
    if roundtrip:
        checkpoint = tmp_path / "picked-coordinate-chain.pkl"
        picked_stack = copy.deepcopy(stamp.stack)
        picked_offsets = tuple((ox, oy) for _, ox, oy in stamp.stack)
        saved_objects = copy.deepcopy((source, target, stamp_obj))
        saved_positions, saved_time = holder.get_positions(), clock.now()
        saved_rngs = w.rng.getstate(), w.observation_rng.getstate()
        save_lab_state(lab, clock, checkpoint)
        checkpoint_bytes = checkpoint.read_bytes()
        clock = SimClock()
        lab = configured_lab(clock, cfg)
        assert load_lab_state(lab, clock, checkpoint)
        w = lab.world
        source, target, stamp_obj = (w.objects[obj.id] for obj in saved_objects)
        stamp = w.stamps[stamp_obj.id]
        assert (source, target, stamp_obj) == saved_objects
        assert w.held_stamp() is stamp_obj and stamp_obj.held_by == STAMP_HOLDER
        assert stamp.stack == picked_stack
        assert tuple((ox, oy) for _, ox, oy in stamp.stack) == picked_offsets
        assert (w.rng.getstate(), w.observation_rng.getstate()) == saved_rngs
        assert clock.now() == saved_time and checkpoint.read_bytes() == checkpoint_bytes
        gate = SafetyGate(lab.devices, clock=clock, sink=lambda kind, data: events.append((kind, data)))
        stage, holder, heater = gate.stage("transfer_stage"), gate.manipulator("transfer_arm"), gate.heater("stamp_heater")
        assert holder.get_positions() == saved_positions
        assert gate.switch("transfer_arm_vacuum").get()
        monkeypatch.setattr(w.rng, "random", lambda: .999)
    fine = (-.03, .05)
    target_center = (scope_xy[0]-.02, scope_xy[1]+.025)
    rotated_mount = rigid_xy(-19, target_mount)
    stage.move_to(Pose(x=target_center[0]-pivot[0]-rotated_mount[0]-fine[0],
        y=target_center[1]-pivot[1]-rotated_mount[1]-fine[1], rotation=-19))
    stage.move_fine_to(*fine)
    assert (target.pose.x, target.pose.y, target.pose.rotation) == pytest.approx((*target_center, -32))
    holder.move_axis_to("x", over["x"]-.018)
    holder.move_axis_to("y", over["y"]+.014)
    release_apex = (scope_xy[0]-.018, scope_xy[1]+.014)
    # A fresh gate has no preceding setpoint history. Use the ordinary bounded
    # ramp in both cases rather than copying or bypassing its guard state.
    heater.set_target(130)
    clock.advance(60)
    heater.set_target(180)
    clock.advance(300)
    assert heater.read_temperature() > stamp.melt_c
    holder.move_axis_to("z", 5.29)
    clock.advance(5)
    holder.move_axis_to("z", 6.5)
    assert stamp.stack == [] and stamp.film is False
    assert [flake.id for flake in target.flakes] == ["chain_0", "chain_1"]
    for landed, offset in zip(target.flakes, expected_offsets, strict=True):
        expected_local = rigid_xy(32, (release_apex[0]+offset[0]-target_center[0],
                                      release_apex[1]+offset[1]-target_center[1]))
        assert (landed.x_mm, landed.y_mm) == pytest.approx(expected_local, abs=1e-10)
        for actual, point in zip(landed.outline_mm, shape, strict=True):
            assert actual == pytest.approx(rigid_xy(73, point), abs=1e-10)
    assert math.dist((target.flakes[0].x_mm, target.flakes[0].y_mm),
                     (target.flakes[1].x_mm, target.flakes[1].y_mm)) == pytest.approx(math.dist(*local), abs=1e-10)
    assert math.dist(*local) > .05  # ideal motion preserves distinct positions; it never snaps them together
    assert stamp.drift_xy_mm == (0, 0) and stamp.slip_rotation_deg == 0
    assert not w.collisions and not gate.tripped
    assert not any(kind == "safety.violation" for kind, _ in events)
    assert lab.devices.get("transfer_stage")._err == (0, 0)
    assert holder.get_positions() == {"x": over["x"]-.018, "y": over["y"]+.014, "z": 6.5}
    if roundtrip:
        assert checkpoint.read_bytes() == checkpoint_bytes


def test_real_elapsed_time_can_grow_contact_but_fractional_polls_are_not_witnesses():
    lab, clock, chip, stamp = history_case()
    chip.flakes = [SimFlake("small", "Gr", 1, .01, 0, 36, chip.flakes[0].outline_mm)]
    w = lab.world
    w.manip_move("z", 5.28)
    clock.advance(.09)  # before the first canonical .1s mesh point
    rng = w.rng.getstate()
    for _ in range(5):
        w.advance_transfer(clock.now())
        assert not stamp.contact_history["pickup"]
    assert stamp.front_radius_mm > .02 and w.rng.getstate() == rng
    w.manip_move("z", w.manip.z)  # a real control boundary can retain this footprint
    assert set(stamp.contact_history["pickup"]) == {"small"}
    prior_radius = stamp.front_radius_mm
    clock.advance(1)
    w.advance_transfer(clock.now())
    assert stamp.front_radius_mm > prior_radius and w.rng.getstate() == rng


@pytest.mark.parametrize("active", [False, True])
def test_legacy_contact_history_is_empty_only_when_quiescence_is_proven(tmp_path, active):
    clock = SimClock()
    lab = build_sim_lab(clock, seed=17)
    if active:
        lab, clock, _, _ = history_case()
        lab.world.manip_move("z", 5.28)
        clock.advance(1)
    state = pickle.loads(pickle.dumps(lab_state(lab, clock)))
    for stamp in state["world"]["stamps"].values():
        del vars(stamp)["contact_history"]
        assert stamp.contact_history is None  # class default must not mask legacy absence
    epoch = state["world"]["transfer_dynamics"].get("epoch")
    if epoch is not None:
        epoch.pop("contact_history")
    checkpoint = tmp_path/"legacy.pkl"
    checkpoint.write_bytes(pickle.dumps(state))
    before = checkpoint.read_bytes()
    fresh_clock = SimClock()
    fresh = build_sim_lab(fresh_clock, seed=17)
    initial = pickle.dumps(lab_state(fresh, fresh_clock))
    if active:
        with pytest.raises(PreconditionFailed, match="active contact lacks"):
            load_lab_state(fresh, fresh_clock, checkpoint)
        assert pickle.dumps(lab_state(fresh, fresh_clock)) == initial
    else:
        assert load_lab_state(fresh, fresh_clock, checkpoint)
        assert all("contact_history" in vars(stamp) and stamp.contact_history is None
                   for stamp in fresh.world.stamps.values())
        assert fresh.world.rng.getstate() == state["world"]["rng"]
    assert checkpoint.read_bytes() == before


def test_bare_film_beside_the_stack_is_not_a_vdw_pickup():
    """pickup_stack_requires_overlap (linked Hall-bar SIM): only a flake under
    a carried flake uses pickup_onto_stack; one the bare film touches beside
    it uses the bare-film chance (seed 7, 2026-11-03)."""
    from qumus_simlab.testbench.world import SimFlake, SimWorld

    square = ((-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01))
    carried = SimFlake("hbn", "hBN", 60, 0, 0, 400, square)
    stack = [(carried, 0., -.2)]
    under = SimFlake("gr", "Gr", 1, 0, 0, 100, tuple((x/2, y/2) for x, y in square))
    assert SimWorld._under_carried(stack, (5., 5.), under, (5., 4.8), 30.)
    assert not SimWorld._under_carried(stack, (5., 5.), under, (5., 5.217), 30.)


def test_gr_lifts_only_when_carried_hbn_covers_80_percent():
    """Jialei 2026-11-05: Gr comes up only through the carried hBN, when at
    least 80 % of its area is covered; bare film alone never lifts it."""
    from qumus_simlab.testbench.world import MIN_CARRIED_COVER, SimFlake, SimWorld

    assert MIN_CARRIED_COVER == {"Gr": 0.8}
    square = ((-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01))
    stack = [(SimFlake("hbn", "hBN", 60, 0, 0, 400, square), 0., 0.)]
    gr = SimFlake("gr", "Gr", 1, 0, 0, 100, tuple((x/2, y/2) for x, y in square))
    cover = SimWorld._carried_cover
    assert cover(stack, (5., 5.), gr, (5., 5.), 0.) == pytest.approx(1.0)
    assert cover(stack, (5., 5.), gr, (5.0075, 5.), 0.) == pytest.approx(0.75)
    assert cover(stack, (5., 5.), gr, (5.006, 5.), 0.) == pytest.approx(0.9)
    assert cover([], (5., 5.), gr, (5., 5.), 0.) == 0.0
