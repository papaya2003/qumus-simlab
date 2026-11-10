"""Causal and reproducibility checks for the synthetic reactor growth model."""

from __future__ import annotations

import json
import math

import pytest

from qumus_simlab.testbench.reactor_model import GrowthModel


def grown(seed=1, seconds=600, temperature=820, flow=200):
    model = GrowthModel(seed, "reactor_chip_01")
    model.advance(seconds, temperature, flow)
    return model


@pytest.mark.parametrize("seed", range(8))
def test_nominal_growth_and_parameter_sensitivity(seed):
    good = grown(seed).measure()
    assert good["coverage"] > 0.8
    assert good["uniformity"] > 0.9
    for temperature, flow in [(600, 200), (820, 30), (820, 700)]:
        poor = grown(seed, temperature=temperature, flow=flow).measure()
        assert poor["quality_score"] < good["quality_score"] * 0.65
    assert grown(seed, seconds=60).measure()["coverage"] < good["coverage"] / 3
    assert grown(seed, seconds=2400).measure()["quality_score"] < good["quality_score"]


@pytest.mark.parametrize("temperature,flow,inside", [(22, 200, True), (820, 0, True),
                                                   (820, 200, False)])
def test_no_reaction_without_required_conditions(temperature, flow, inside):
    model = GrowthModel(1, "chip")
    model.advance(600, temperature, flow, in_chamber=inside)
    assert model.measure()["coverage"] == 0


def test_sampling_and_checkpointing_do_not_change_growth():
    uninterrupted = grown()
    observed = GrowthModel(1, "reactor_chip_01")
    for _ in range(60):
        observed.advance(10, 820, 200)
        observed.measure()
    assert observed.measure()["coverage"] == pytest.approx(
        uninterrupted.measure()["coverage"], abs=1e-12)
    saved = json.loads(json.dumps(observed.snapshot()))
    resumed = GrowthModel.restore(saved)
    observed.advance(100, 780, 250)
    resumed.advance(100, 780, 250)
    assert observed.snapshot() == resumed.snapshot()


def test_exposure_history_matters_and_damage_does_not_disappear():
    hot = grown(seconds=600, temperature=1000)
    damage_before = hot.measure()["defect_fraction"]
    assert damage_before > 0
    hot.advance(600, 820, 200)
    assert hot.measure()["defect_fraction"] >= damage_before
    assert hot.measure()["quality_score"] < grown(seconds=1200).measure()["quality_score"]


def test_seed_and_chip_control_spatial_outcome():
    a = grown()
    assert a.snapshot() == grown().snapshot()
    assert a.snapshot()["cells"] != grown(seed=2).snapshot()["cells"]
    assert a.measure((-1, -0.1, -1, 1))["coverage"] != a.measure((0.1, 1, -1, 1))["coverage"]
    assert a.measure((2, 3, 2, 3))["visible"] is False


@pytest.mark.parametrize("args", [(math.nan, 820, 200), (1, math.inf, 200),
                                  (1, 820, math.nan), (-1, 820, 200), (1, 820, -1)])
def test_invalid_growth_inputs_are_rejected(args):
    with pytest.raises(ValueError):
        GrowthModel(1, "chip").advance(*args)
