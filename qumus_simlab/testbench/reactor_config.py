"""Opt-in synthetic reactor layout, preserving the default software bench."""

from __future__ import annotations

import copy
from typing import Any

from qumus_simlab.config import load_yaml


def _translate_station(bench: dict[str, Any], name: str, dx: float) -> None:
    """Translate a complete station, including its locator targets and fiducials."""
    station = bench["stations"][name]
    if "x" in station:
        station["x"] += dx
    for key in ("region", "module"):
        if key in station:
            station[key]["x"] = [x + dx for x in station[key]["x"]]
    for slot in bench["slots"].values():
        if slot["station"] == name:
            slot["x"] += dx
    for marker in bench["qr_markers"].get(name, []):
        marker[0] += dx


def _space_workcells(bench: dict[str, Any], layout: dict[str, Any]) -> None:
    """Preserve each arm/workcell transform while spacing the storage row evenly."""
    start, pitch = layout["first_storage_center_x_mm"], layout["arm_storage_pitch_mm"]
    for i, name in enumerate(("left_storage", "middle_storage", "right_storage")):
        center = sum(bench["stations"][name]["region"]["x"]) / 2
        _translate_station(bench, name, start + 2 * i * pitch - center)
    for i, (arm, station, region) in enumerate((
            ("arm_left", "exfoliation_stage", "tape"),
            ("arm_right", "transfer_stage", "microscope"))):
        dx = start + (2 * i + 1) * pitch - bench[arm]["base"]["x"]
        bench[arm]["base"]["x"] += dx
        _translate_station(bench, station, dx)
        bench["regions"][region]["x"] = [x + dx for x in bench["regions"][region]["x"]]
        if arm == "arm_left":
            bench["exfoliator"]["origin"]["x"] += dx
            rail = bench["exfoliator"]["rail"]
            rail["x"] = [x + dx for x in rail["x"]]
        else:
            bench["microscope"]["center"]["x"] += dx
            bench["transfer_arm"]["origin"]["x"] += dx


def with_reactor(bench: dict[str, Any], config: dict[str, Any]
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    if not config.get("reactor", {}).get("enabled", False):
        return bench, config
    bench, config = copy.deepcopy(bench), copy.deepcopy(config)
    model = load_yaml("reactor.yaml")
    model["growth"].update(config["reactor"].get("growth", {}))
    kind = config["reactor"].get("experiment_kind", model["experiment_kind"])
    if kind not in ("legacy_field_growth", "two_flake_reaction"):
        raise ValueError("unknown reactor experiment kind")
    from qumus_simlab.records.material_metal_simulation import MaterialMetalSimulation

    model["material_metal"] = MaterialMetalSimulation.model_validate(config["reactor"].get("material_metal", {})).model_dump()
    model["experiment_kind"] = kind
    if kind == "two_flake_reaction":
        model["reaction"].update(copy.deepcopy(config["reactor"].get("reaction", {})))
        model["geometry"]["fov_mm_10x"] = [0.26, 0.172]
    _space_workcells(bench, model["bench_layout"])
    bench["reactor_module"] = model
    geo = model["geometry"]
    config["reactor"]["metal_patch_outline_mm"] = geo["metal_patch_outline_mm"]
    bench["table"] = copy.deepcopy(geo["table"])
    bench["regions"]["workspace"] = copy.deepcopy(geo["table"])
    bench["stations"]["reactor"] = {
        "kind": "reactor", "label": "vdW reactor (simulation)",
        "placeholder": False, "region": geo["chamber_region"]}
    bench["stations"]["reactor_storage"] = {
        "kind": "storage", "label": "Reactor carriers",
        "region": geo["storage_region"]}
    r = geo["storage_region"]
    bench["qr_markers"]["reactor_storage"] = [
        [r["x"][0] + 6, r["y"][0] + 6], [r["x"][1] - 6, r["y"][0] + 6],
        [r["x"][1] - 6, r["y"][1] - 6], [r["x"][0] + 6, r["y"][1] - 6]]
    load = geo["load"]
    bench["slots"]["reactor_port"] = {
        **load, "station": "reactor", "holds": "chip", "size": geo["carrier_slot_size_mm"]}
    bench["arm_reactor"] = {
        "planned": False, "base": geo["arm_base"],
        "reach": ["right_storage", "reactor_storage", "reactor"], "places": ["reactor_port"],
        "reach_mm": 280, "safe_z": 80, "pick_z": 2, "place_z": 3,
        "home": copy.deepcopy(bench["arm_left"]["home"])}
    origin, pitch = geo["storage_origin"], geo["storage_pitch_mm"]
    for i, (col, row) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)], 1):
        x, y = origin["x"] + col * pitch[0], origin["y"] - row * pitch[1]
        slot, chip = f"reactor_chip_slot_{i}", f"reactor_chip_{i:02d}"
        bench["slots"][slot] = {"x": x, "y": y, "station": "reactor_storage",
                                 "holds": "chip", "size": geo["chip_size_mm"]}
        config["initial_samples"].append({
            "id": chip, "kind": "chip", "location": slot, "status": "bare",
            "props": {"reactor_substrate": True, "size_mm": geo["chip_size_mm"],
                      "preparation": "synthetic seed representing a manually cut chip",
                      "image_source": "simulated"}})
        if kind == "two_flake_reaction" and config["reactor"].get(
                "synthetic_prepared_pairs", model["synthetic_prepared_pairs"]):
            carrier = config["initial_samples"][-1]
            manifest, flakes = prepared_pair_fixture(chip)
            carrier["status"] = "reactor_pair_prepared"
            carrier["props"].update(reactor_pair=manifest,
                preparation="Preprepared synthetic pair fixture; no autonomous cleavage or transfer was executed")
            config["initial_samples"].extend(flakes)
    return bench, config


def prepared_pair_fixture(chip_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Explicit small input polygons, for mechanical acceptance recordings only."""
    from qumus_simlab.core.polygons import rectangle
    from qumus_simlab.testbench.material_flakes import Flake, Placement

    ids = [f"{chip_id}_bottom", f"{chip_id}_top"]
    manifest = {"schema_version": 1, "preparation_id": f"{chip_id}_prepared_pair_v1",
                "chip_id": chip_id, "input_flake_ids": ids, "preparation_source": "synthetic_fixture"}
    samples = []
    for i, (fid, size, center) in enumerate(zip(ids, ((0.025, 0.020), (0.018, 0.016)),
                                              ((0, 0), (0.003, 0.002)), strict=True)):
        flake = Flake(id=fid, material=f"Synthetic 2D input {i + 1}", layers=i + 2,
                      source_chip_id=f"fixture_source_{i + 1}", batch_id="preprepared-pair-v1",
                      outline_mm=rectangle(*size))
        placed = Placement(flake=flake, target_center_mm=center)
        samples.append({"id": fid, "kind": "flake", "location": chip_id, "status": "prepared",
                        "props": {"material": flake.material, "area_um2": flake.area_um2,
                                  "material_placement": placed.model_dump(mode="json"),
                                  "image_source": "simulated"}})
    return manifest, samples
