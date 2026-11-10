"""Reviewed molecular/assembly compositions over existing guarded Steps.

These objects compose executable Steps, not free-form model-authored code.
A Step is a recovery checkpoint and may contain many atomic device calls.
Atomic operations remain the fixed device contracts behind SafetyGate.
Flattening preserves every original checkpoint, including unsafe hot steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qumus_simlab.flows.step import Step


@dataclass(frozen=True)
class Molecule:
    name: str
    description: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.description or not self.steps:
            raise ValueError("A molecular operation needs a name, purpose and checked steps")
        if any(not isinstance(step, Step) for step in self.steps):
            raise ValueError("Molecular operations contain reviewed Steps only")
        if len({id(step) for step in self.steps}) != len(self.steps):
            raise ValueError("Each checkpoint needs its own Step instance")

    def build_steps(self) -> list[Step]:
        return list(self.steps)

    def manifest(self, start_seq: int = 0) -> dict[str, Any]:
        return {"name": self.name, "level": "molecular", "description": self.description,
                "start_seq": start_seq, "end_seq": start_seq + len(self.steps) - 1,
                "checkpoints": [{"seq": start_seq + i, "name": step.name,
                                 "safe_to_resume": step.safe_to_resume}
                                for i, step in enumerate(self.steps)]}


@dataclass(frozen=True)
class Assembly:
    name: str
    description: str
    molecules: tuple[Molecule, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.description or not self.molecules:
            raise ValueError("An assembly needs a name, purpose and molecular operations")
        if any(not isinstance(part, Molecule) for part in self.molecules):
            raise ValueError("Assemblies contain molecular operations only")
        if len({part.name for part in self.molecules}) != len(self.molecules):
            raise ValueError("Molecular operation identities must be unique within an assembly")
        steps = self.build_steps()
        if len({id(step) for step in steps}) != len(steps):
            raise ValueError("A Step instance cannot be shared by two molecular operations")

    def build_steps(self) -> list[Step]:
        return [step for part in self.molecules for step in part.steps]

    def manifest(self) -> dict[str, Any]:
        children, seq = [], 0
        for part in self.molecules:
            children.append(part.manifest(seq))
            seq += len(part.steps)
        return {"name": self.name, "level": "assembly", "description": self.description,
                "start_seq": 0, "end_seq": seq - 1, "children": children}


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep molecular identities/ranges in routine tool replies; full=true retains leaves."""
    return {key: [compact_manifest(child) for child in value] if key == "children" else value
            for key, value in manifest.items() if key not in ("checkpoints", "description")}


class WorkflowMismatch(ValueError):
    """The saved hierarchy cannot authorize the currently constructed Steps."""

    def __init__(self, reason: str, message: str, *, exception_type: str | None = None):
        super().__init__(message)
        self.reason, self.exception_type = reason, exception_type

    def to_dict(self) -> dict[str, Any]:
        return {"code": "workflow_mismatch", "reason": self.reason, "message": str(self),
                **({"exception_type": self.exception_type} if self.exception_type else {})}


def _contract(manifest: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read structural bindings only; descriptive copy may change independently."""
    if not isinstance(manifest, dict):
        raise ValueError("Expected a workflow object")
    name, level = manifest.get("name"), manifest.get("level")
    start, end = manifest.get("start_seq"), manifest.get("end_seq")
    if (not isinstance(name, str) or not name or level not in ("assembly", "molecular")
            or type(start) is not int or type(end) is not int or start < 0 or end < start):
        raise ValueError("Invalid workflow identity or checkpoint range")
    contract = {"name": name, "level": level, "start_seq": start, "end_seq": end}
    leaves = []
    if level == "assembly":
        children = manifest.get("children")
        if not isinstance(children, list) or not children or "checkpoints" in manifest:
            raise ValueError("An assembly needs molecular children")
        bindings, next_seq, names = [], start, set()
        for child in children:
            binding, checkpoints = _contract(child)
            if (binding["level"] != "molecular" or binding["start_seq"] != next_seq
                    or binding["name"] in names):
                raise ValueError("Ambiguous molecular boundaries")
            names.add(binding["name"])
            bindings.append(binding)
            leaves.extend(checkpoints)
            next_seq = binding["end_seq"] + 1
        if next_seq != end + 1:
            raise ValueError("Assembly range does not match its children")
        contract["children"] = bindings
    else:
        checkpoints = manifest.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != end - start + 1 or "children" in manifest:
            raise ValueError("Invalid molecular checkpoint coverage")
        for seq, checkpoint in enumerate(checkpoints, start):
            if (not isinstance(checkpoint, dict) or type(checkpoint.get("seq")) is not int
                    or checkpoint["seq"] != seq or not isinstance(checkpoint.get("name"), str)
                    or not checkpoint["name"] or type(checkpoint.get("safe_to_resume")) is not bool):
                raise ValueError("Invalid checkpoint identity or recovery flag")
            leaves.append({key: checkpoint[key] for key in ("seq", "name", "safe_to_resume")})
        contract["checkpoints"] = leaves
    return contract, leaves


def validate_workflow(saved: Any, current: Any, steps: list[Step]) -> None:
    """Reject structural drift against both the current hierarchy and actual Steps.

    Jobs with no nonempty saved hierarchy retain the old flat-flow contract.
    This check never fabricates or replaces their original plan. It is not a
    source-code fingerprint: unchanged structure does not prove identical code.
    """
    if not saved:
        return
    try:
        expected, _ = _contract(saved)
        if expected["start_seq"] != 0:
            raise ValueError("A job hierarchy must start at checkpoint zero")
    except (ValueError, TypeError, KeyError) as exc:
        raise WorkflowMismatch("invalid_saved_workflow", "Saved workflow is malformed; review it before continuing") from exc
    if not current:
        raise WorkflowMismatch("missing_current_workflow", "The current flow no longer supplies the saved hierarchy")
    try:
        actual, leaves = _contract(current)
    except (ValueError, TypeError, KeyError) as exc:
        raise WorkflowMismatch("invalid_current_workflow", "The current workflow cannot be matched to saved checkpoints") from exc
    if expected != actual:
        raise WorkflowMismatch("hierarchy_changed", "Current workflow identities, boundaries or recovery flags differ from the saved plan")
    if any(not isinstance(step, Step) or type(step.safe_to_resume) is not bool for step in steps):
        raise WorkflowMismatch("invalid_actual_steps", "Actual execution does not contain valid checked Steps")
    built = [{"seq": seq, "name": step.name, "safe_to_resume": step.safe_to_resume}
             for seq, step in enumerate(steps)]
    if leaves != built:
        raise WorkflowMismatch("steps_changed", "Actual execution Steps differ from the saved workflow checkpoints")
