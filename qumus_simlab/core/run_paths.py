"""Resolve relocated private run assets without rewriting historical records.

The local runs/layout.json records directory moves. Keep it with the runs:
the reverse mapping also preserves the original per-run budget scope.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"


def _moves() -> list[tuple[Path, Path]]:
    manifest = RUNS_ROOT / "layout.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("Unsupported private run layout; restore runs/layout.json")
    origin = Path(data["original_root"])
    if not origin.is_absolute():
        raise ValueError("The original run root must be absolute")
    moves = []
    for old, new in data["moves"].items():
        before, after = origin / old, RUNS_ROOT / new
        if (Path(old).is_absolute() or Path(new).is_absolute()
                or ".." in Path(old).parts or ".." in Path(new).parts):
            raise ValueError("Run relocations must stay inside their run roots")
        moves.append((before, after))
    return moves


def resolve_run_path(value: str | Path) -> Path:
    """Map an old absolute path (or a cwd-relative path) to its new location."""
    path = Path(value).absolute()
    for before, after in sorted(_moves(), key=lambda pair: len(pair[0].parts), reverse=True):
        if path.is_relative_to(before):
            return (after / path.relative_to(before)).resolve()
    return path.resolve()


def original_run_path(value: str | Path) -> Path:
    """Keep existing ledger scope keys stable when a run directory is renamed."""
    path = Path(value).resolve()
    for before, after in sorted(_moves(), key=lambda pair: len(pair[1].parts), reverse=True):
        if path.is_relative_to(after):
            return before / path.relative_to(after)
    return path
