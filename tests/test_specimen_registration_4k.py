"""On a full-sensor v1 capture the specimen translation bound keeps its
distance: the bound is declared on the 480 px working raster."""
from PIL import Image, ImageDraw

from qumus_simlab.perception.tracking import RegistrationConfig, register_specimen


def _frame(size, box, scale):
    image = Image.new("RGB", size, (164, 98, 103))
    draw = ImageDraw.Draw(image)
    draw.polygon([(x*scale, y*scale) for x, y in box], fill=(120, 70, 120))
    draw.polygon([(x*scale, y*scale) for x, y in box[:3]], fill=(100, 60, 130))
    return image


BOX = [(200, 110), (290, 120), (280, 190), (210, 180)]


def test_4k_shift_inside_the_bound_on_the_working_raster_is_found():
    scale = 8  # 480 -> 3840
    shifted = [(x+15, y+10) for x, y in BOX]  # 15 working px = 120 native px
    ref, cur = _frame((3840, 2160), BOX, scale), _frame((3840, 2160), shifted, scale)
    result = register_specimen(ref, cur, polygon_px=[(x*scale, y*scale) for x, y in BOX],
                               config=RegistrationConfig(max_translation_px=40, max_rotation_deg=0))
    assert result.ok, result.loss_reason
    assert abs(result.dx_px-120) <= 16 and abs(result.dy_px-80) <= 16


def test_shift_beyond_the_bound_is_still_refused():
    shifted = [(x+15, y+10) for x, y in BOX]
    ref, cur = _frame((480, 320), BOX, 1), _frame((480, 320), shifted, 1)
    result = register_specimen(ref, cur, polygon_px=BOX,
                               config=RegistrationConfig(max_translation_px=10, max_rotation_deg=0))
    assert not result.ok
