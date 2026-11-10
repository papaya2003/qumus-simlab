"""Camera noise is temporal: a new view draws a new realisation (#205 follow-up).

The scene-only noise seed gave every exposure of a chip the same pixel pattern,
fixed in image space, and v1's ORB corner refind locked onto it at 0 degrees.
"""
import io

import numpy as np
from PIL import Image

from qumus_simlab.testbench.render import MicroscopeRenderer


def _pixels(packet):
    return np.asarray(Image.open(io.BytesIO(MicroscopeRenderer._compose(packet, size=(128, 84)))), dtype=float)


def _noise(packet):
    return _pixels(packet)-_pixels({**packet, "observation_noise_scale": 0})


def test_each_view_draws_its_own_noise_with_the_same_statistics():
    base = {"on_chip": "compat", "fov_mm": [.65, .43], "objective": 20, "sharpness_chip": 10.5,
            "flakes": [], "chip_corners": [], "chip_pose": {"x": 10., "y": 20., "rotation": 1.8}}
    moved = {**base, "chip_pose": {"x": 10.05, "y": 20., "rotation": 1.8}}
    turned = {**base, "chip_pose": {"x": 10., "y": 20., "rotation": .4}}
    a, b, c = _noise(base), _noise(moved), _noise(turned)
    # the same state renders the same bytes
    assert np.array_equal(_pixels(base), _pixels(dict(base)))
    for other in (b, c):
        assert abs(np.corrcoef(a.ravel(), other.ravel())[0, 1]) < .1
        assert abs(other.std()-a.std()) < .1*a.std()
