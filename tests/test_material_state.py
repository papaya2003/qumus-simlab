"""Material binding and camera regressions; physical transport lives in flow tests."""

import io
import pickle
from copy import deepcopy

import pytest
from PIL import Image

from qumus_simlab.config import load_yaml
from qumus_simlab.core.clock import SimClock
from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.core.polygons import area_um2, rectangle, transform
from qumus_simlab.core.types import Pose
from qumus_simlab.devices.base import Frame
from qumus_simlab.devices.sim.devices import SimMicroscope
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.testbench.lab import build_sim_lab
from qumus_simlab.testbench.locator import SimMicroscopeAnalyzer
from qumus_simlab.testbench.material_adapter import (
    microscope_flake,
    prepare_catalog_pair,
    sync_pair_flakes,
)
from qumus_simlab.testbench.material_reaction import LocalReaction
from qumus_simlab.testbench.material_view import assembly_from_manifest, observe_pair, render_pair
from qumus_simlab.testbench.persist import load_lab_state, save_lab_state
from qumus_simlab.testbench.reactor_pair_backend import PairMaterialBackend
from qumus_simlab.testbench.render import Canvas, MicroscopeRenderer
from qumus_simlab.testbench.world import SimFlake, SimObject, SimWorld


def transferred_pair():
    """An adapter unit fixture; not evidence that a transport job ran."""
    chip = SimObject('target', 'chip', Pose(x=0, y=0), size_mm=(0.15, 0.10))
    samples = []
    for n, (size, xy) in enumerate([((0.04, 0.02), (0, 0)), ((0.02, 0.02), (0.004, 0))]):
        outline = rectangle(*size)
        exact = area_um2(outline)
        flake = SimFlake(f'sim_{n}', 'Gr', n + 1, *xy, exact, outline,
                         {'source_chip_id': f'source_{n}', 'batch_id': 'cleavage_1',
                          'source_outline_mm': transform(outline, xy=(1, 2))})
        chip.flakes.append(flake)
        samples.append({'id': f'catalog_{n}', 'kind': 'flake', 'status': 'stacked',
                        'location': chip.id, 'props': {'material': 'Gr', 'sim_id': flake.id,
                                                     'area_um2': exact * 0.96}})
    manifest, updates = prepare_catalog_pair(chip, samples, preparation_id='prepared_actual')
    for sample in samples:
        sample['props'].update(updates[sample['id']])
    assembly = assembly_from_manifest(manifest, samples, chip_size_mm=chip.size_mm)
    return chip, samples, manifest, LocalReaction(assembly, seed=7)


def test_real_catalog_binding_preserves_measured_area_and_source_geometry():
    chip, samples, manifest, model = transferred_pair()
    assert manifest['preparation_source'] == 'stacking_job'
    assert [s['props']['area_um2'] for s in samples] == [768, 384]
    assert [s['props']['measured_area_um2'] for s in samples] == [768, 384]
    assert [s['props']['geometry_area_um2'] for s in samples] == pytest.approx([800, 400])
    for placed, flake in zip(model.state.assembly.inputs, chip.flakes, strict=True):
        assert placed.flake.outline_mm == flake.origin['source_outline_mm']
        for actual, expected in zip(placed.outline_mm, transform(flake.outline_mm, xy=(flake.x_mm, flake.y_mm)), strict=True):
            assert actual == pytest.approx(expected)
    wrong = deepcopy(samples)
    wrong[0]['props']['geometry_area_um2'] *= 1.01
    with pytest.raises(ValueError, match='geometry area'):
        assembly_from_manifest(manifest, wrong, chip_size_mm=chip.size_mm)


def test_preparation_refuses_extra_missing_consumed_and_unprovenanced_physical_flakes():
    chip, samples, _, _ = transferred_pair()
    chip.flakes.append(deepcopy(chip.flakes[0]))
    with pytest.raises(ValueError, match='exactly'):
        prepare_catalog_pair(chip, samples, preparation_id='extra')
    chip.flakes.pop()
    samples[0]['status'] = 'reactor_consumed'
    with pytest.raises(ValueError, match='consumed'):
        prepare_catalog_pair(chip, samples, preparation_id='consumed')
    samples[0]['status'] = 'stacked'
    chip.flakes[0].origin = {}
    with pytest.raises(ValueError, match='provenance'):
        prepare_catalog_pair(chip, samples, preparation_id='legacy')
    chip.flakes.clear()
    with pytest.raises(ValueError, match='exactly'):
        prepare_catalog_pair(chip, samples, preparation_id='missing')


def test_world_lifecycle_repeated_captures_and_product_transfer_do_not_resurrect_sources():
    chip, samples, _, model = transferred_pair()
    sync_pair_flakes(chip, model, samples)
    assert [f.id for f in chip.flakes] == ['sim_0', 'sim_1']
    initial_list = chip.flakes
    sync_pair_flakes(chip, model, samples)
    assert chip.flakes is initial_list
    model.advance(20, active=True)
    sync_pair_flakes(chip, model, samples)
    assert len(chip.flakes) == 3  # two unconsumed inputs plus one partial product
    partial = chip.flakes[-1]
    assert partial.layers is None and partial.origin['source_flake_ids'] == ['catalog_0', 'catalog_1']
    model.advance(60, active=True)
    sync_pair_flakes(chip, model, samples)
    assert len(chip.flakes) == 1 and chip.flakes[0].id == partial.id
    assert chip.flakes[0].area_um2 == pytest.approx(400)
    assert chip.flakes[0].origin['reaction_status'] == 'complete'
    # Model/world checkpoint unit roundtrip, without advancing exposure.
    restored = LocalReaction.restore(model.snapshot())
    copied = deepcopy(chip)
    before = deepcopy(copied.flakes)
    sync_pair_flakes(copied, restored, samples)
    assert copied.flakes == before
    # Simulate a later transfer in this adapter test; real route tested separately.
    chip.flakes.clear()
    sync_pair_flakes(chip, restored, samples)
    assert chip.flakes == []


def test_binding_does_not_invent_real_sources_or_overwrite_moved_geometry():
    chip, samples, _, model = transferred_pair()
    chip.flakes[0].x_mm += 0.001
    before = deepcopy(chip.flakes)
    with pytest.raises(ValueError, match='geometry'):
        sync_pair_flakes(chip, model, samples)
    assert chip.flakes == before
    chip.flakes.clear()
    with pytest.raises(ValueError, match='missing'):
        sync_pair_flakes(chip, model, samples)
    sync_pair_flakes(chip, model, samples, allow_fixture=True)
    assert len(chip.flakes) == 2


def test_preparation_verifier_requires_current_focused_complete_optical_geometry():
    chip, samples, _, _ = transferred_pair()
    world = SimWorld(seed=7)
    chip.on = 'transfer_stage'
    world.add(chip)
    scope = SimMicroscope('microscope', SimClock(), world, fov_mm_10x=(0.13, 0.086))
    analyzer = SimMicroscopeAnalyzer(world)
    frame = scope.capture()
    manifest, updates = analyzer.reactor_preparation(frame, samples, preparation_id='observed')
    assert manifest['chip_id'] == chip.id and set(updates) == {'catalog_0', 'catalog_1'}
    chip.pose = Pose(x=0.001, y=0)
    with pytest.raises(ValueError, match='moved'):
        analyzer.reactor_preparation(frame, samples, preparation_id='stale')
    chip.pose = Pose(x=0, y=0)
    chip.flakes[0].x_mm += 0.001
    with pytest.raises(ValueError, match='geometry changed'):
        analyzer.reactor_preparation(frame, samples, preparation_id='changed')
    chip.flakes[0].x_mm -= 0.001
    scope.set_objective(100)
    with pytest.raises(ValueError, match='complete input outlines'):
        analyzer.reactor_preparation(scope.capture(), samples, preparation_id='clipped')
    scope.set_objective(10)
    chip.focus_offset = 1.0
    with pytest.raises(ValueError, match='focused'):
        analyzer.reactor_preparation(scope.capture(), samples, preparation_id='blurred')


def test_backend_restores_physical_product_without_restarting_consumed_sources():
    chip, samples, manifest, _ = transferred_pair()
    store = RunStore()
    store.upsert_sample(chip.id, 'chip', props={'reactor_pair': manifest})
    for sample in samples:
        store.upsert_sample(sample['id'], 'flake', location=chip.id, props=sample['props'])
    backend = PairMaterialBackend(store, {}, lambda: 7)
    states = {}
    model = backend.model(chip, states)
    model.advance(80, active=True)
    states[chip.id] = model.snapshot()
    backend.sync_world(chip, model)
    product = deepcopy(chip.flakes)
    for sample in samples:
        store.upsert_sample(sample['id'], 'flake', status='reactor_consumed',
                            props={'reactor_consumed_by': manifest['preparation_id']})
    restored = backend.model(chip, states)
    assert restored.snapshot() == model.snapshot() and chip.flakes == product
    world = SimWorld(seed=7)
    world.add(chip)
    world.reactor_state = {'pair_models': states}
    rng, rows = world.rng.getstate(), store.list_samples()
    assert backend.restore_world(world) == [chip.id]
    assert chip.flakes == product and store.list_samples() == rows and world.rng.getstate() == rng
    with pytest.raises(PreconditionFailed, match='saved reaction state'):
        backend.model(chip, {})
    assert chip.flakes == product
    chip.flakes.clear()
    restored = backend.model(chip, states)
    observation, view = backend.observation(restored, center=(0, 0), fov=(0.13, 0.086), in_focus=True)
    assert not observation['visible'] and observation['product'] is None
    assert view['reason'] == 'product_not_on_carrier' and view['render_product'] is None


def test_camera_rotation_changes_pixels_and_visibility_without_changing_physical_geometry():
    _, _, _, model = transferred_pair()
    initial = model.snapshot()
    zero = observe_pair(model, center_mm=(0.003, 0), fov_mm=(0.05, 0.025))
    one = observe_pair(model, center_mm=(0.003, 0), fov_mm=(0.05, 0.025), camera_rotation_deg=1)
    ninety = observe_pair(model, center_mm=(0.003, 0), fov_mm=(0.05, 0.025), camera_rotation_deg=90)
    assert zero['inputs'][0]['fully_visible']
    assert not ninety['inputs'][0]['fully_visible']
    assert zero['inputs'][0]['outline_mm'] == one['inputs'][0]['outline_mm']
    assert zero['inputs'][0]['outline_px'] != one['inputs'][0]['outline_px']
    assert render_pair(zero) != render_pair(one)
    assert model.snapshot() == initial
    for a, b in zip(ninety['inputs'][0]['camera_outline_mm'],
                    transform(transform(model.observe()['inputs'][0]['outline_mm'], xy=(-0.003, 0)), rotation_deg=90),
                    strict=True):
        assert a == pytest.approx(b)


@pytest.mark.parametrize('seconds', [0, 20, 80])
def test_unfocused_images_hide_specimen_and_unavailable_conversion(seconds):
    _, _, _, model = transferred_pair()
    model.advance(seconds, active=True)
    visible = observe_pair(model)
    hidden = observe_pair(model, in_focus=False)
    assert hidden['product'] is None and hidden['reaction_fraction'] is None
    assert hidden['conversion_of_overlap'] is None and not hidden['visible']
    x, y, width, height = visible['sensor_rect_px']
    sensor = (x, y, x + width, y + height)
    focused = Image.open(io.BytesIO(render_pair(visible))).crop(sensor).tobytes()
    unfocused = Image.open(io.BytesIO(render_pair(hidden))).crop(sensor).tobytes()
    assert focused != unfocused
    assert len(set(zip(unfocused[::3], unfocused[1::3], unfocused[2::3], strict=True))) == 1


def test_unknown_product_thickness_is_renderable_and_never_classified_as_known_layers():
    chip, samples, _, model = transferred_pair()
    model.advance(80, active=True)
    sync_pair_flakes(chip, model, samples)
    f = chip.flakes[0]
    view = microscope_flake(f, (f.x_mm, f.y_mm), (0.13, 0.086))
    assert view['layers'] is None and view['material_role'] == 'product'
    world = SimWorld(seed=2, model={'microscope': {'misclassify_10x': 1}})
    frame = Frame('microscope', 0, meta={'flakes': [view], 'sharpness_chip': 10})
    found = SimMicroscopeAnalyzer(world).find_flakes(frame, f.material)
    assert len(found) == 1 and found[0].thickness_label == 'unknown'
    canvas = Canvas(100, 100, (0, 0, 0))
    MicroscopeRenderer._flake(canvas, view, lambda x, y: (50 + x * 1000, 50 - y * 1000),
                              1000, 1000, world.observation_rng)
    assert bytes(canvas.px).count(bytes((223, 187, 100))) > 0


def test_new_world_profile_matches_named_profile_and_legacy_restore_can_keep_old_policy():
    cfg = load_yaml('testbench.yaml')
    world = SimWorld(model=deepcopy(cfg))
    assert world.material_morphology == load_yaml('material-morphology.yaml')['exfoliation']['morphology']
    saved = world.material_morphology
    saved['cluster_fraction'] = 0.1
    assert world.material_morphology['cluster_fraction'] == 0.65
    world.material_morphology = None
    assert 'morphology' not in world.model['exfoliation']
    world.material_morphology = saved
    assert world.material_morphology['cluster_fraction'] == 0.1


def test_checkpoint_keeps_population_policy_and_old_checkpoints_preserve_compatible_mode(tmp_path):
    clock = SimClock()
    lab = build_sim_lab(clock, seed=7)
    profile = lab.world.material_morphology
    profile['cluster_fraction'] = 0.2
    lab.world.material_morphology = profile
    state = tmp_path / 'current.pkl'
    save_lab_state(lab, clock, state)
    original_rng = lab.world.rng.getstate()
    lab.world.material_morphology = None
    load_lab_state(lab, clock, state)
    assert lab.world.material_morphology['cluster_fraction'] == 0.2
    assert lab.world.rng.getstate() == original_rng
    # Only this generated fixture is modified to represent the older schema.
    old = pickle.loads(state.read_bytes())
    old['world'].pop('material_morphology')
    old_path = tmp_path / 'old.pkl'
    old_path.write_bytes(pickle.dumps(old))
    load_lab_state(lab, clock, old_path)
    assert lab.world.material_morphology is None
    assert lab.world.rng.getstate() == original_rng
