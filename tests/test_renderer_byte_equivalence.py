"""Portable exact-byte checks against the renderer's original scalar rules."""

import math

import numpy as np
import pytest

from qumus_simlab.testbench.render import (
    Canvas,
    _blend_tables,
    _camera,
    _camera_fields,
    _camera_native,
    _camera_rgb8,
    _PremultipliedStampPlane,
)


class ScalarCanvas(Canvas):
    """Keep virtual set/blend dispatch on the unchanged scanline path."""

    def __init__(self, *args):
        super().__init__(*args)
        self.blends = 0

    def blend(self, x, y, color, alpha):
        self.blends += 1
        super().blend(x, y, color, alpha)


@pytest.mark.parametrize("color", [(180, 205, 216), (174, 184, 206), (150, 160, 196),
                                   (30, 22, 40), (0, 255, 127)])
@pytest.mark.parametrize("alpha", [0, .025, .1, .28, .55, .85, .9, .999])
def test_polygon_blend_all_channel_bytes_equal_original_scalar(color, alpha):
    canvas = Canvas(256, 1, (0, 0, 0))
    canvas.px[:] = bytes(value for value in range(256) for _ in range(3))
    canvas.rect(0, 0, 256, 1, color, alpha)
    expected = bytes(int(value * (1-alpha) + channel * alpha)
                     for value in range(256) for channel in color)
    assert bytes(canvas.px) == expected


@pytest.mark.parametrize("points", [
    [(-7.4, -2.3), (20.2, -2.3), (20.2, 13.8), (-7.4, 13.8)],
    [(-2.5, 3.2), (9.5, -4.1), (17.6, 9.9), (2.2, 15.4)],
    [(1.2, 1.1), (13.8, 1.1), (7.4, 6.5), (13.8, 10.9), (1.2, 10.9)],
    [(-4.8, 2.3), (-1.1, 2.3), (-1.1, 7.4), (-4.8, 7.4)],
    [(4.5, 2.5), (4.5, 2.5), (4.5, 2.5)],
    [],
])
def test_clipped_overlapping_scanlines_and_subclass_dispatch(points):
    fast, scalar = Canvas(16, 12, (73, 19, 251)), ScalarCanvas(16, 12, (73, 19, 251))
    for canvas in (fast, scalar):
        canvas.polygon(points, (180, 205, 216), .025)
        canvas.polygon(points, (174, 184, 206), .025)
        canvas.rect(3.7, -1.8, 11.2, 9.6, (30, 22, 40), .85)
        canvas.rect(7, 5, 19, 19, (44, 137, 236), 1)
    assert fast.px == scalar.px
    assert scalar.blends > 0  # The subclass's override was actually used.


def test_float_stamp_plane_preserves_unquantized_virtual_blending():
    plane = _PremultipliedStampPlane(4, 3)
    plane.rect(-2, -2, 8, 8, (180, 205, 216), .025)
    plane.rect(-2, -2, 8, 8, (174, 184, 206), .025)
    for channel, first, second in zip(plane.channels[:3], (180, 205, 216), (174, 184, 206), strict=True):
        assert list(channel) == [first * .025 * (1-.025) + second * .025] * 12
    assert list(plane.channels[3]) == [.025 * (1-.025) + .025] * 12
    assert plane.channels[0][0] != int(plane.channels[0][0])


@pytest.mark.parametrize("color,alpha", [((50, 50, 50), -.2), ((50, 50, 50, 200), .1)])
def test_non_rgb8_table_cases_retain_scalar_polygon_semantics(color, alpha):
    fast, scalar = Canvas(3, 2, (100, 100, 100)), ScalarCanvas(3, 2, (100, 100, 100))
    for canvas in (fast, scalar):
        canvas.rect(-1, -1, 5, 4, color, alpha)
    assert fast.px == scalar.px


@pytest.mark.parametrize("numpy_first", [False, True])
def test_equal_hash_float32_alpha_cannot_contaminate_python_float_table(numpy_first):
    alpha32 = np.float32(.1)
    alpha64 = float(alpha32)
    assert alpha32 == alpha64 and hash(alpha32) == hash(alpha64)
    _blend_tables.cache_clear()
    for alpha in ([alpha32, alpha64] if numpy_first else [alpha64, alpha32]):
        actual, expected = Canvas(1, 1, (250, 250, 250)), ScalarCanvas(1, 1, (250, 250, 250))
        cached = _blend_tables.cache_info().currsize
        actual.rect(0, 0, 1, 1, (0, 0, 0), alpha)
        expected.rect(0, 0, 1, 1, (0, 0, 0), alpha)
        assert actual.px == expected.px  # Preserve the installed NumPy's scalar arithmetic.
        if type(alpha) is not float:
            assert _blend_tables.cache_info().currsize == cached


def scalar_camera(canvas, seed, noise_scale):
    """Original camera body: separate scalar multiply/int/noise/clamp."""
    width, height = canvas.w, canvas.h
    fx = [1.0 - .14 * ((x-width/2)/(width/2)) ** 2 for x in range(width)]
    fy = [1.0 - .14 * ((y-height/2)/(height/2)) ** 2 for y in range(height)]
    state = seed & 0xFFFFFFFF or 1
    for y in range(height):
        for x in range(width):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            noise = round(((state >> 16) % 7 - 3) * noise_scale)
            gain = fx[x] * fy[y]
            for channel in range(3):
                index = (y*width+x)*3+channel
                canvas.px[index] = max(0, min(255, int(canvas.px[index]*gain)+noise))


@pytest.mark.parametrize("seed", [0, 1, 0x12345678, -7])
@pytest.mark.parametrize("noise_scale", [0., 1., .375, 1.25, -1.5])
def test_camera_native_field_sample_positions_all_256_values(seed, noise_scale):
    # Every channel byte at native corners/edges/centre; the real production
    # transform consumes these exact cached fields, not a separate formula.
    width, height = 480, 320
    gain, noise = _camera_fields(width, height, seed, noise_scale)
    positions = [(0, 0), (479, 0), (0, 319), (479, 319), (240, 160), (11, 217)]
    indices = [y*width+x for x, y in positions]
    scalar_gain = [(1-.14*((x-width/2)/(width/2))**2)
                   * (1-.14*((y-height/2)/(height/2))**2) for x, y in positions]
    scalar_noise = {}
    state = seed & 0xFFFFFFFF or 1
    for index in range(max(indices)+1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        if index in indices:
            scalar_noise[index] = round(((state >> 16) % 7 - 3) * noise_scale)
    assert gain[indices].tolist() == scalar_gain
    assert noise[indices].tolist() == [scalar_noise[index] for index in indices]
    selected_gain = np.tile(gain[indices], 256)
    selected_noise = np.tile(noise[indices], 256)
    rgb = np.repeat(np.arange(256, dtype=np.uint8), len(positions)*3).reshape(-1, 3)
    actual = _camera_rgb8(rgb, selected_gain, selected_noise)
    expected = [[max(0, min(255, int(value*gain[index])+int(noise[index])))]*3
                for value in range(256) for index in indices]
    assert actual.tolist() == expected
    assert not gain.flags.writeable and not noise.flags.writeable


@pytest.mark.parametrize("size", [(1, 1), (17, 11), (256, 2)])
@pytest.mark.parametrize("seed,noise_scale", [(0, 0.), (19, 1.), (-1, .375), (7, 2**41)])
def test_camera_full_dispatch_matches_scalar_and_safe_fallback(size, seed, noise_scale):
    fast, expected = Canvas(*size, (0, 0, 0)), Canvas(*size, (0, 0, 0))
    values = bytes((index*73+index//7) % 256 for index in range(size[0]*size[1]*3))
    fast.px[:], expected.px[:] = values, values
    _camera(fast, seed, noise_scale)
    scalar_camera(expected, seed, noise_scale)
    assert fast.px == expected.px


def test_camera_and_blend_caches_are_finite():
    assert _camera_fields.cache_info().maxsize == 8
    assert _blend_tables.cache_info().maxsize == 256
    assert _camera_fields.cache_info().currsize <= 8
    assert _blend_tables.cache_info().currsize <= 256


@pytest.mark.parametrize("seed,noise_scale", [(0, 1.), (-7, .375), (0xFFFFFFFF, -1.5)])
def test_native_chunk_camera_keeps_scalar_lcg_across_block_and_row_boundaries(seed, noise_scale):
    size = (133, 37)  # More than one 4096-pixel chunk, not a row multiple.
    actual, expected = Canvas(*size, (0, 0, 0)), Canvas(*size, (0, 0, 0))
    values = bytes((i*73+i//7) % 256 for i in range(size[0]*size[1]*3))
    actual.px[:], expected.px[:] = values, values
    _camera_native(actual, seed, noise_scale)
    scalar_camera(expected, seed, noise_scale)
    assert actual.px == expected.px


@pytest.mark.parametrize("noise_scale", [math.nan, math.inf])
def test_nonfinite_camera_noise_keeps_original_scalar_failure(noise_scale):
    with pytest.raises((ValueError, OverflowError)) as original:
        scalar_camera(Canvas(2, 2, (73, 91, 146)), 1, noise_scale)
    with pytest.raises(type(original.value)) as actual:
        _camera(Canvas(2, 2, (73, 91, 146)), 1, noise_scale)
    assert str(actual.value) == str(original.value)
