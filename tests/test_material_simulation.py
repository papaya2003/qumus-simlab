"""Offline scientific-boundary, geometry, persistence and observation checks."""

import json
import math

import pytest
from pydantic import ValidationError

from qumus_simlab.core.polygons import (
    area_um2,
    convex_polygon,
    intersection,
    rectangle,
    transform,
)
from qumus_simlab.testbench.material_flakes import (
    CleavageProfile,
    Flake,
    PairAssembly,
    Placement,
    cleave,
    flake_outline,
    project_outline,
)
from qumus_simlab.testbench.material_reaction import LocalReaction, ReactionProfile


def square(fid="A", size=0.04):
    return Flake(id=fid, material="Unspecified input", layers=3, source_chip_id=f"source_{fid}",
                 batch_id="cleave_1", outline_mm=rectangle(size, size, (1, 2)))


def pair(offset=0.01, rotation=0):
    return PairAssembly(id="pair_1", target_chip_id="target", inputs=(
        Placement(flake=square(), target_center_mm=(0, 0)),
        Placement(flake=square("B"), target_center_mm=(offset, 0), rotation_deg=rotation)))


def test_known_overlap_and_touching_edges():
    assert area_um2(pair().overlap_mm) == pytest.approx(1200)
    assert pair(0.04).overlap_mm == ()
    assert pair(0.08).overlap_mm == ()
    assert area_um2(pair(0, 45).overlap_mm) == pytest.approx(1600 * (2 * math.sqrt(2) - 2))


@pytest.mark.parametrize("vertices", [((0, 0), (1, 1), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (0.5, 0.1), (1, 1), (0, 1)), ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (0, float("nan")))])
def test_unsupported_geometry_fails_explicitly(vertices):
    with pytest.raises(ValueError):
        convex_polygon(vertices)


@pytest.mark.parametrize("count", [0, 1, 3, 6])
def test_exactly_two_inputs_required(count):
    inputs = [Placement(flake=square(str(i)), target_center_mm=(0, 0)) for i in range(count)]
    with pytest.raises(ValidationError):
        PairAssembly(id="pair", target_chip_id="target", inputs=inputs)


def test_duplicate_identity_and_off_chip_transfer_rejected():
    p = pair().model_dump()
    p["inputs"] = [p["inputs"][0]] * 2
    with pytest.raises(ValidationError):
        PairAssembly.model_validate(p)
    p = pair().model_dump()
    p["inputs"][1]["target_center_mm"] = (5, 0)
    with pytest.raises(ValidationError):
        PairAssembly.model_validate(p)


def test_source_geometry_is_retained_by_transfer():
    a = pair(rotation=72)
    assert a.inputs[1].flake.outline_mm == square("B").outline_mm
    assert area_um2(a.inputs[1].outline_mm) == pytest.approx(square("B").area_um2)
    assert a.inputs[1].flake.center_mm == (1, 2)
    assert a.inputs[1].target_center_mm == (0.01, 0)


def test_single_nested_product_stays_in_both_inputs_and_has_full_lineage():
    r = LocalReaction(pair(rotation=24), seed=13)
    before = r.observe()
    assert before["product"] is None and before["status"] == "unreacted"
    outlines, ids = [], set()
    for _ in range(4):
        r.advance(15, active=True)
        obs = r.observe()
        product = obs["product"]
        ids.add(product["id"])
        outlines.append(product["outline_mm"])
        assert product["source_flake_ids"] == ["A", "B"]
        assert product["assembly_id"] == "pair_1" and product["chip_id"] == "target"
        assert product["layers"] is None  # no chemical or thickness inference
        assert product["area_um2"] == pytest.approx(obs["overlap_area_um2"] * obs["conversion_of_overlap"])
        for placed in r.state.assembly.inputs:
            assert area_um2(intersection(product["outline_mm"], placed.outline_mm)) == pytest.approx(product["area_um2"])
    assert len(ids) == 1 and obs["status"] == "complete"
    for smaller, larger in zip(outlines, outlines[1:], strict=False):
        assert area_um2(intersection(smaller, larger)) == pytest.approx(area_um2(smaller))
    assert obs["inputs"] == before["inputs"]
    assert obs["product"]["area_um2"] < 1600  # never a chip-sized coating


def test_observation_frequency_and_time_partition_do_not_change_reaction():
    one, many = [LocalReaction(pair(rotation=17), seed=57) for _ in range(2)]
    one.advance(42, active=True)
    for _ in range(42):
        many.advance(1, active=True)
        for _ in range(4):
            assert many.observe() == many.observe()
    assert one.snapshot() == many.snapshot()
    assert one.observe() == many.observe()
    assert LocalReaction.restore(json.loads(json.dumps(one.snapshot()))).observe() == one.observe()


def test_failures_incomplete_and_no_overlap_are_not_hidden():
    r = LocalReaction(pair(), seed=1, profile=ReactionProfile(max_conversion=0.4))
    r.advance(600, active=False)
    assert r.observe()["product"] is None
    r.advance(600, active=True)
    assert r.observe()["status"] == "partial"
    assert r.observe()["conversion_of_overlap"] == 0.4
    partial = r.observe()["product"]
    r.fail("synthetic_contact_lost")
    r.advance(600, active=True)
    assert r.observe()["status"] == "failed" and r.observe()["product"] == partial
    empty = LocalReaction(pair(offset=0.1), seed=1)
    empty.advance(10000, active=True)
    assert empty.observe()["status"] == "no_overlap" and empty.observe()["product"] is None
    failed = LocalReaction(pair(), seed=1, profile=ReactionProfile(failure_probability=1))
    failed.advance(10000, active=True)
    assert failed.observe()["status"] == "failed" and failed.observe()["product"] is None


def test_bad_state_and_exposure_rejected_without_mutation():
    r = LocalReaction(pair(), seed=1)
    original = r.snapshot()
    for value in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            r.advance(value, active=True)
        assert r.snapshot() == original
    for edits in ({"version": "old-grid-model"}, {"active_s": 2}, {"elapsed_s": -2}, {"seed": float("nan")}):
        with pytest.raises(ValidationError):
            LocalReaction.restore({**original, **edits})


def test_cleavage_is_seeded_bounded_angular_and_batch_specific():
    args = dict(seed=31, chip_id="source", batch_id="contact_1", material="Test material", count=100)
    a, b = cleave(**args), cleave(**args)
    assert a == b and len(a) == 100
    assert {f.id for f in a}.isdisjoint(f.id for f in cleave(**{**args, "batch_id": "contact_2"}))
    assert {f.id for f in a}.isdisjoint(f.id for f in cleave(**{**args, "seed": 32}))
    assert len({f.layers for f in a}) >= 5
    assert len({len(f.outline_mm) for f in a}) >= 3
    for f in a:
        assert 12 - 1e-8 <= f.area_um2 <= 12000 + 1e-8
        assert area_um2(intersection(f.outline_mm, rectangle(10, 10))) == pytest.approx(f.area_um2)
        assert Flake.model_validate_json(f.model_dump_json()) == f


def test_cleavage_respects_contact_footprints_and_can_produce_nothing():
    region = rectangle(0.15, 0.3, (0.35, -0.2))
    args = dict(seed=3, chip_id="source", batch_id="contact", material="Test", count=30,
                chip_size_mm=(1, 1), contact_regions=(region,))
    flakes = cleave(**args)
    assert len(flakes) > 15
    for f in flakes:
        assert area_um2(intersection(f.outline_mm, region)) == pytest.approx(f.area_um2)
    assert cleave(**{**args, "contact_regions": ()}) == ()
    assert cleave(**{**args, "contact_regions": (rectangle(1, 1, (20, 0)),)}) == ()
    assert cleave(**{**args, "contact_regions": (rectangle(1e-5, 1e-5),)}) == ()


def test_cluster_control_changes_distribution_without_edge_clamping():
    args = dict(seed=91, chip_id="source", batch_id="contact", material="Test", count=150)
    clustered = cleave(**args, profile=CleavageProfile(cluster_fraction=1, cluster_count=1,
                                                      cluster_radius_mm=0.06))
    uniform = cleave(**args, profile=CleavageProfile(cluster_fraction=0))
    def spread(flakes):
        centers = [f.center_mm for f in flakes]
        mean = tuple(sum(q[i] for q in centers) / len(centers) for i in range(2))
        return sum(math.dist(q, mean) ** 2 for q in centers) / len(centers)
    assert spread(clustered) < spread(uniform) * 0.02


def test_projection_preserves_identity_geometry_and_partial_field_visibility():
    p = flake_outline("fixed", 1400)
    p = transform(p, xy=(0.04, 0.01), rotation_deg=37)
    projections = [project_outline(p, center_mm=(0, 0), fov_mm=(fov, fov)) for fov in (0.5, 0.25, 0.05)]
    for proj in projections:
        assert proj["area_um2"] == pytest.approx(1400)
    for (x1, y1), (x2, y2) in zip(projections[0]["outline_px"], projections[1]["outline_px"], strict=True):
        assert x2 - 320 == pytest.approx(2 * (x1 - 320))
        assert y2 - 210 == pytest.approx(2 * (y1 - 210))
    edge = project_outline(rectangle(0.04, 0.04, (0.03, 0)), center_mm=(0, 0), fov_mm=(0.04, 0.04))
    assert edge["visible"]  # centre outside but part of the flake is visible
    assert edge["visible_area_um2"] == pytest.approx(400)


def test_many_rotated_cases_obey_overlap_bounds():
    for seed in range(100):
        a = flake_outline(f"a:{seed}", 20 + seed * 19)
        b = transform(flake_outline(f"b:{seed}", 40 + seed * 7), xy=(0.005, 0), rotation_deg=seed * 29)
        ab, ba = intersection(a, b), intersection(b, a)
        assert area_um2(ab) == pytest.approx(area_um2(ba), abs=1e-7)
        assert area_um2(ab) <= min(area_um2(a), area_um2(b)) + 1e-7


def test_subresolution_exposure_does_not_emit_zero_area_product():
    reaction = LocalReaction(pair(), seed=5)
    reaction.advance(1e-300, active=True)
    assert reaction.observe()["product"] is None


def test_overflowing_geometry_rejected():
    with pytest.raises(ValueError):
        convex_polygon(((0, 0), (1e300, 0), (0, 1e300)))
