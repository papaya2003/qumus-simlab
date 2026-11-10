"""Annotations must never mask sensor evidence or change acquisition coordinates."""

import io
import math
from copy import deepcopy

import pytest
from PIL import Image
from test_material_state import transferred_pair

from qumus_simlab.testbench.material_flakes import project_outline
from qumus_simlab.testbench.material_view import camera_outline, observe_pair, render_pair


@pytest.mark.parametrize('size', [(640, 420), (800, 600)])
@pytest.mark.parametrize('edge', ['top', 'bottom', 'left', 'right'])
def test_edge_product_is_unobscured_and_sensor_coordinates_still_drive_acquisition(size, edge):
    _, _, _, model = transferred_pair()
    model.advance(80, active=True)
    before = model.snapshot()
    product = model.observe()['product']
    polygon = product['outline_mm']
    xs, ys = zip(*polygon, strict=True)
    fw, fh, gap = 0.13, 0.086, 0.0002
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    if edge == 'top':
        cy = max(ys) - fh / 2 + gap
    elif edge == 'bottom':
        cy = min(ys) + fh / 2 - gap
    elif edge == 'left':
        cx = min(xs) + fw / 2 - gap
    else:
        cx = max(xs) - fw / 2 + gap
    view = observe_pair(model, center_mm=(cx, cy), fov_mm=(fw, fh), image_size=size)
    saved_view = deepcopy(view)
    assert view['product_verified'] and view['product']['fully_visible']
    assert view['product']['area_um2'] == pytest.approx(product['area_um2'], rel=1e-12)
    sensor_projection = project_outline(camera_outline(polygon, (cx, cy), 0),
                                         center_mm=(0, 0), fov_mm=(fw, fh), image_size=size)
    # Reactor acquisition divides these coordinates by the original frame size.
    assert view['product']['outline_px'] == sensor_projection['outline_px']
    assert view['product']['area_um2'] == sensor_projection['area_um2']
    for source in view['inputs']:
        assert source['outline_px'] == project_outline(
            camera_outline(source['outline_mm'], (cx, cy), 0), center_mm=(0, 0),
            fov_mm=(fw, fh), image_size=size)['outline_px']
    image = Image.open(io.BytesIO(render_pair(view)))
    assert image.size == size and view == saved_view and model.snapshot() == before
    x, y, width, height = view['sensor_rect_px']
    field = image.crop((x, y, x + width, y + height))
    gold, edge_color = (223, 187, 100), (255, 240, 188)
    pixels = field.tobytes()
    assert set(zip(pixels[::3], pixels[1::3], pixels[2::3], strict=True)) <= {(51, 61, 88), gold, edge_color}
    specimen = [(px, py) for py in range(size[1]) for px in range(size[0])
                if image.getpixel((px, py)) in (gold, edge_color)]
    assert specimen
    assert all(x <= px < x + width and y <= py < y + height for px, py in specimen)
    # All four extents survive, including the edge formerly covered by the title.
    seen_x, seen_y = zip(*specimen, strict=True)
    display_x, display_y = zip(*view['product']['display_outline_px'], strict=True)
    for actual, expected in zip((min(seen_x), max(seen_x), min(seen_y), max(seen_y)),
                                (min(display_x), max(display_x), min(display_y), max(display_y)), strict=True):
        assert abs(actual - math.floor(expected)) <= 1
    scale_y = size[1] - 16
    white = [px for px in range(size[0]) if image.getpixel((px, scale_y)) == (255, 255, 255)]
    assert scale_y >= y + height
    assert white == list(range(x, x + math.floor(width / 5) + 1))


def test_clipped_product_cannot_paint_over_annotation_margins():
    _, _, _, model = transferred_pair()
    model.advance(80, active=True)
    view = observe_pair(model, fov_mm=(0.012, 0.008))
    assert view['product'] is None  # geometric acceptance is still withheld
    image = Image.open(io.BytesIO(render_pair(view)))
    x, y, width, height = view['sensor_rect_px']
    for py in range(image.height):
        for px in range(image.width):
            if image.getpixel((px, py)) in ((223, 187, 100), (255, 240, 188)):
                assert x <= px < x + width and y <= py < y + height
