"""A bare scan tile falls back to the focus plane of the in-focus tiles."""
import pytest

from qumus_simlab.flows.library.search import ScanChip


def _scan(tiles):
    step = ScanChip("chip_001", "hBN", 50.0)
    xs, ys = [0.5, 1.5, 2.5], [0.5, 1.5, 2.5]
    step._scan = {"binding": {"grid_x_mm": xs, "grid_y_mm": ys}, "captures": {}}
    for (i, j), z in tiles.items():
        step._scan["captures"][f"{i},{j}"] = {"focus_z": z}
        step._tiles.add((i, j))
    return step


def test_planned_focus_follows_a_tilted_chip_plane():
    # z = -0.3 + 0.01 x + 0.01 y (the SIM chip tilt), three sharp tiles
    plane = {(i, j): -0.3 + 0.01 * (0.5 + i) + 0.01 * (0.5 + j) for i, j in ((0, 0), (2, 0), (0, 2))}
    step = _scan(plane)
    assert step._planned_focus(2.5, 2.5) == pytest.approx(-0.3 + 0.05)


def test_planned_focus_needs_three_non_collinear_sharp_tiles():
    assert _scan({(0, 0): -0.2, (1, 0): -0.2})._planned_focus(1.0, 1.0) is None
    assert _scan({(0, 0): -0.2, (1, 0): -0.2, (2, 0): -0.2})._planned_focus(1.0, 1.0) is None


def test_planned_focus_ignores_blurred_tiles():
    step = _scan({(0, 0): -0.2, (2, 0): -0.2, (0, 2): -0.2})
    step._scan["captures"]["1,1"] = {"focus_z": 5.0}  # captured but never acquired
    assert step._planned_focus(1.5, 1.5) == pytest.approx(-0.2)


def test_planned_focus_ignores_tiles_kept_below_min_blur():
    # v1 keeps the best frame of a bare tile; its focus readback is no plane evidence.
    step = _scan({(0, 0): -0.2, (2, 0): -0.2, (0, 2): -0.2, (1, 1): 5.0})
    step._scan["captures"]["1,1"]["focus_verdict"] = "below_min_blur_kept_best_frame"
    assert step._planned_focus(1.5, 1.5) == pytest.approx(-0.2)
