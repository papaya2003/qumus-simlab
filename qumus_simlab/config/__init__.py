"""Packaged default configuration. Override directory: env QUMUS2_CONFIG_DIR."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DIR = Path(__file__).parent


def config_dir() -> Path:
    override = os.environ.get("QUMUS2_CONFIG_DIR")
    return Path(override) if override else DEFAULT_DIR


def load_yaml(name: str) -> dict[str, Any]:
    path = config_dir() / name
    if not path.exists():
        path = DEFAULT_DIR / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def with_stamp_discovery(config: dict[str, Any]) -> dict[str, Any]:
    """The v2-only stamp discovery stamp of the linked Hall-bar SIM before #204.

    Off by default since #204 (v1 has no discovery step; the first touch lands
    beside the source instead). Kept opt-in for saved runs and their tests:
    the 12 mm cap dome, the legacy contact heights and the fixed policy.
    """
    from qumus_simlab.records.discovery import PHYSICS_PROFILE, discovery_policy

    legacy = simulation_profile("hallbar")
    stamp = config["stamp"]
    for key in ("dome_height_mm", "indentation_per_travel"):
        stamp.pop(key, None)
    stamp.update({key: legacy["stamp"][key] for key in ("crush_depth_mm", "touch_z", "touch_z_sigma")},
                 dome_radius_mm=12., cap_support_radius_mm=5., physics_profile=PHYSICS_PROFILE,
                 discovery=discovery_policy())
    config["microscope"]["best_focus_sigma"] = legacy["microscope"]["best_focus_sigma"]
    # The discovery claim (records/discovery.py) declares the 10 um/mm focus tilt
    # this stamp was characterised with; the flat-chip Hall-bar default (0, Jialei
    # 2026-11-05) came later and would make the declaration false.
    config["microscope"]["focus_tilt_per_mm"] = legacy["microscope"]["focus_tilt_per_mm"]
    return config


def simulation_profile(name: str = "integrated") -> dict[str, Any]:
    """Everyday bench configuration, also used by release acceptance runs.

    Existing two-arm checkpoints need the previous profile. Their mechanical
    state must never be overlaid with the integrated three-arm layout.
    """
    if name == "hallbar-linked":
        from qumus_simlab.testbench.bench_geometry import PROFILE
        from qumus_simlab.testbench.hallbar import with_hallbar_target

        # The linked bench is set first so the stock follows its nine slots.
        config = simulation_profile("integrated")
        config["bench_geometry"] = {"profile": PROFILE}
        config = with_hallbar_target(config)
        config["seed"] = 8
        _hallbar_supply(config)
        # Stamp dome INFERRED from the v1 presets, not measured (#204; Jialei
        # 2026-11-04 accepted inferred numbers, no lab data): a flat PDMS dome
        # 1.5 mm across and ~0.05 mm high (R = r^2 / 2h ~ 5.7 mm). v1 starts
        # the 10 um touching search at Z 5.4 and presses 0.1 mm steps, so the
        # contact sits near Z 5.2, about a quarter of the holder travel past
        # it indents the dome, and 0.6 mm of travel past contact crushes it.
        # Chip and stamp heights spread +-0.02 mm, so the fixed v1 presets
        # stay valid. v2-only stamp discovery is off: the first touch lands
        # beside the source (#136) as in the v1 flow.
        config["stamp"].update(dome_radius_mm=5.7, dome_height_mm=0.05, indentation_per_travel=0.25,
                               crush_depth_mm=0.6, touch_z=5.2, touch_z_sigma=0.02)
        config["stamp"].pop("cap_support_radius_mm", None)
        config["stamp"].pop("discovery", None)
        config["microscope"]["best_focus_sigma"] = 0.02
        # Jialei 2026-11-05: chips lie flat, no tilt. The former 10 um/mm,
        # set for ~2 mm sim chips, put the corner of a v1-size chip 0.3 mm
        # below its centre and the 20X refind lost focus (seed 3 r8).
        config["microscope"]["focus_tilt_per_mm"] = 0.0
        # Jialei 2026-11-03: SIM errors for the Hall-bar flow may be small.
        # The 20 nm/s stamp drift and 3 um/mm approach drift moved the carried
        # hBN ~13 um between the pre-contact check and Gr contact (a518 seed
        # 2). The approach drift is zero; a residual 10 pm/s lateral drift
        # (10 nm over a 1000 s transfer) is kept because the lab twin records
        # only CHANGED state, and the same-clock encoder readback that every
        # exposure is bound to is published only while the stamp state moves.
        # Both stay inside the declared discovery bounds.
        config["stamp"].setdefault("dynamics", {}).update(
            lateral_drift_mm_s=[1e-8, -1e-8], approach_drift_mm_per_mm=[0.0, 0.0])
        # The synthetic chip-corner detector keeps its 3 um/axis Gaussian error
        # here (Jialei 2026-11-04, after #167): the stack flow now registers
        # the target chip frame on its own electrodes at 20x before release.
        # Jialei 2026-11-04: no synthetic tape residue on chips here. Real
        # residue is too varied to model, and the v1 RGB vision read the
        # drawn specks as ~2000 few-layer graphene candidates per chip.
        config["microscope"]["residue"] = False
        # Jialei 2026-11-04 (#182): flakes are found the v1 way, from the
        # pixels of v1-camera frames (v1 colours, 3840x2160, measured
        # vignetting and surface), not from simulator truth.
        from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE

        config["material_recognition"] = {"backend": "v1_lab_rgb",
                                          "capture_profile": V1_MATERIAL_CAPTURE_PROFILE}
        # Jialei 2026-11-03: a real PDMS stamp does not twist while in
        # contact. The synthetic 0.002 deg/s contact slip rotated the carried
        # stack ~0.45 deg over a transfer (#109), so it is zero here.
        config["stamp"]["dynamics"]["contact_slip_deg_s"] = 0
        # Jialei 2026-11-03: the bare PC film generally does not lift graphene
        # ("一般粘不走"); a Gr under the carried hBN keeps the vdW chance.
        config["stamp"]["pickup_stack_requires_overlap"] = True
        config["stamp"]["pickup_first"] = {**config["stamp"].get("pickup_first", {}), "Gr": 0.02}
        return config
    if name == "hallbar":
        from qumus_simlab.testbench.hallbar import with_hallbar_target

        config = with_hallbar_target(simulation_profile("integrated"))
        config["seed"] = 8
        _hallbar_supply(config)
        return config
    if name not in {"integrated", "previous"}:
        raise ValueError(f"unknown simulation profile: {name}")
    config = load_yaml("testbench.yaml")
    config["reactor"] = {"enabled": name == "integrated", "synthetic_prepared_pairs": False}
    return config


#: Hall-bar lab flake supply (Jialei, 2026-11-03: the simulation models the
#: process, not the material, so yield and size may be raised to make a
#: qualifying hBN/Gr pair within the nine source chips likely). Monte Carlo on
#: the a518 contacts with the flow's filters: about 98 % within 9 chips (#107 B3).
HALLBAR_SUPPLY = {"flakes_per_chip": 30, "median_area_um2": 600}


def _hallbar_supply(config: dict[str, Any]) -> None:
    exf = config["exfoliation"]
    exf["flakes_per_chip"] = HALLBAR_SUPPLY["flakes_per_chip"]
    exf.setdefault("morphology", {})["median_area_um2"] = HALLBAR_SUPPLY["median_area_um2"]


def ideal_stacking_profile() -> dict[str, Any]:
    """Fresh opt-in ideal positioning/observation; retain normal transfer physics.

    This condition cannot retrofit sampled errors in a saved noisy world. The
    existing mechanics signature rejects that checkpoint before restoration.
    Specimen morphology, contact geometry, thermal laws and safety limits stay
    unchanged. Chemistry success remains a separate simulation assumption.
    Existing admission resolves reactor.material_metal through the composed
    bench into frozen job assumptions. Consumers must use this configuration
    for that admission; it cannot rewrite a conflicting saved job's law.
    """
    from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation

    config = simulation_profile()
    config["simulation_condition"] = "ideal_stacking_v1"
    config["placement"].update(sigma_mm=0, sigma_deg=0, big_error_prob=0,
                               big_error_mm=0, big_error_deg=0)
    config["pick"]["fail_prob"] = 0
    config["locator"].update(sigma_mm=0, sigma_deg=0, bias_sigma_mm=0,
                             bias_sigma_deg=0, tail_prob=0, tail_sigma_mm=0,
                             miss_prob=0, false_positive_prob=0, confuse_material_prob=0)
    config["stage"]["sigma_mm"] = 0
    config["microscope"].update(best_focus_sigma=0, misclassify_10x=0,
                                misclassify_100x=0, corner_sigma_mm=0,
                                contact_sigma_mm=0, fringe_gap_sigma_mm=0,
                                observation_noise_scale=0)
    stamp = config["stamp"]
    stamp.update(touch_z_sigma=0, apex_sigma_mm=0, melt_sigma_c=0,
                 optic_shift_sigma_mm=0, manipulator_sigma_mm=0,
                 pickup_first_default=1, pickup_onto_stack=1)
    stamp["pickup_first"] = dict.fromkeys(stamp.get("pickup_first", {}), 1)
    stamp["dynamics"].update(lateral_drift_mm_s=[0, 0],
                             approach_drift_mm_per_mm=[0, 0], maximum_drift_mm=0,
                             contact_slip_deg_s=0, refraction_xy_per_gap_mm=[0, 0])
    assumptions = MaterialMetalSimulation().model_dump(mode="json")
    for film in ("PC", "mica"):
        assumptions[film].update(pickup_probability=1, release_probability=1)
    config["reactor"]["material_metal"] = assumptions
    return config
