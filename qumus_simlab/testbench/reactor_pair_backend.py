"""Reactor-owned binding to the separately maintained material backend.

No morphology or kinetic algorithm lives here. This adapter maps measured
plant conditions to an explicitly synthetic active-exposure flag, retains the
backend snapshot and normalizes its read-only observation to the ledger contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.records.reactor_pairs import PairObservation


class PairMaterialBackend:
    def __init__(self, store: Any, config: dict[str, Any], seed: Callable[[], int]) -> None:
        try:
            from qumus_simlab.testbench.material_reaction import LocalReaction, ReactionProfile
            from qumus_simlab.testbench.material_view import (
                assembly_from_manifest,
                observe_pair,
                render_pair,
            )
        except ImportError as exc:
            raise PreconditionFailed("Two-flake material modules are not installed; integrate the "
                                     "material branch before enabling pair reaction") from exc
        if store is None:
            raise PreconditionFailed("Two-flake reaction needs a sample ledger")
        self.store, self.config, self.seed = store, config, seed
        self._carriers: dict[str, Any] = {}  # refreshed on each model bind after restore
        self.reaction_type, self.profile_type = LocalReaction, ReactionProfile
        self.assemble, self.observe, self.render = assembly_from_manifest, observe_pair, render_pair

    def model(self, chip: Any, states: dict[str, Any]):
        row = self.store.get_sample(chip.id) or {"props": {}}
        manifest = row["props"].get("reactor_pair")
        if not manifest:
            raise PreconditionFailed("This carrier has no two-flake preparation")
        try:
            samples = [self.store.get_sample(fid) for fid in manifest["input_flake_ids"]]
            assembly = self.assemble(manifest, samples, chip_size_mm=chip.size_mm)
            profile = self.profile_type.model_validate(self.config.get("profile", {}))
            if chip.id in states:
                model = self.reaction_type.restore(states[chip.id])
                if model.state.assembly != assembly or model.state.profile != profile:
                    raise ValueError("material preparation/profile changed after exposure began")
            else:
                if any(s["props"].get("reactor_consumed_by") for s in samples):
                    raise ValueError("consumed sources require their saved reaction state; refusing to restart exposure")
                model = self.reaction_type(assembly, seed=self.seed(), profile=profile)
            self.sync_world(chip, model)
            self._carriers[chip.id] = chip
            states[chip.id] = model.snapshot()
            return model
        except (ValueError, KeyError, TypeError) as exc:
            raise PreconditionFailed(f"Invalid two-flake preparation: {exc}") from exc

    def sync_world(self, chip: Any, model: Any) -> None:
        """Bind model geometry to ordinary optics; caller holds world lock.

        Call after exposure advancement, and once after seeding/restoring the
        ledger and physical world. This never accepts a ledger result.
        """
        from qumus_simlab.testbench.material_adapter import sync_pair_flakes

        row = self.store.get_sample(chip.id) or {"props": {}}
        manifest = row["props"].get("reactor_pair") or {}
        samples = [self.store.get_sample(p.flake.id) for p in model.state.assembly.inputs]
        if any(s is None for s in samples):
            raise PreconditionFailed("Material binding requires both ledger sources")
        sync_pair_flakes(chip, model, samples,
                         allow_fixture=manifest.get("preparation_source") == "synthetic_fixture")

    def restore_world(self, world: Any) -> list[str]:
        """Initialize/restore prepared carriers after ledger and world loading.

        Includes carriers in storage, so ordinary optics never depend on a first
        reactor capture. Does not advance exposure, spend RNG or write the ledger.
        """
        restored = []
        with world.lock:
            states = world.reactor_state.setdefault("pair_models", {})
            for row in self.store.list_samples(kind="chip"):
                if not row["props"].get("reactor_pair"):
                    continue
                chip = world.objects.get(row["id"])
                if chip is None:
                    raise PreconditionFailed("Prepared carrier is absent from the restored physical world")
                self.model(chip, states)
                restored.append(chip.id)
        return restored

    def active(self, temperature: float, flow: float, *, inside: bool) -> bool:
        # Fixture controls only: no claim of a measured chemical activation law.
        return (inside
                and temperature >= self.config.get("activation_temperature_c", 700)
                and flow >= self.config.get("activation_flow_sccm", 10))

    def observation(self, model: Any, *, center: tuple[float, float], fov: tuple[float, float],
                    in_focus: bool, camera_rotation_deg: float = 0) -> tuple[dict[str, Any], dict[str, Any]]:
        view = self.observe(model, center_mm=center, fov_mm=fov, in_focus=in_focus,
                            camera_rotation_deg=camera_rotation_deg)
        chip = self._carriers.get(model.state.assembly.target_chip_id)
        rendered = view["render_product"]
        if (chip is not None and view["input_geometry_role"] == "provenance_only" and rendered
                and rendered["id"] not in {f.id for f in chip.flakes}):
            # Saved lineage is history, not a new optical specimen after a later
            # physical product transfer away from its original carrier.
            view.update(visible=False, reason="product_not_on_carrier", product=None,
                        render_product=None, product_verified=False,
                        reaction_fraction=None, conversion_of_overlap=None)
        product = view["product"]
        normalized = PairObservation.model_validate({
            "preparation_id": view["preparation_id"], "input_flake_ids": view["input_flake_ids"],
            "visible": view["visible"], "status": view["status"], "failure_reason": view["failure_reason"],
            "reaction_fraction": view["reaction_fraction"], "model_id": view["model_version"],
            "product": None if product is None else {
                "id": product["id"], "material": product["material"], "area_um2": product["area_um2"],
                "source_flake_ids": product["source_flake_ids"], "material_state": product},
        }).model_dump(mode="json")
        return normalized, view
