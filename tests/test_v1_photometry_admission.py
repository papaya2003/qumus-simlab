"""Frames from a renderer with the v1 material profile carry the extended
photometry declaration; stacking and SIM capture admission accept it."""
import pytest
from PIL import Image

from qumus_simlab.core.errors import NeedsHumanError
from qumus_simlab.core.optical_profiles import V1_MATERIAL_CAPTURE_PROFILE
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library import stacking
from qumus_simlab.flows.library.primitives import MICROSCOPE
from qumus_simlab.testbench.render import (
    MicroscopeRenderer,
    microscope_photometry_profile,
    supported_photometry_profile,
)


def _frame(tmp_path, profile):
    path = tmp_path / "f.png"
    Image.new("RGB", (8, 6)).save(path)
    calibration = {"source": "synthetic", "frame": "microscope_image", "u_axis": "+world_x",
                   "v_axis": "-world_y", "mm_per_pixel": [0.001, 0.001], "image_size_px": [8, 6],
                   "photometry_profile": profile}
    return Frame(MICROSCOPE, 0.0, path=str(path), meta={"calibration": calibration})


def test_renderer_profile_is_the_declared_one():
    assert MicroscopeRenderer(None).photometry_profile == microscope_photometry_profile()
    v1 = MicroscopeRenderer(None, material_capture_profile=V1_MATERIAL_CAPTURE_PROFILE)
    assert v1.photometry_profile == microscope_photometry_profile(V1_MATERIAL_CAPTURE_PROFILE)
    assert v1.photometry_profile["stacking_camera"]


@pytest.mark.parametrize("material", [None, V1_MATERIAL_CAPTURE_PROFILE])
def test_stacking_admits_both_renderer_declarations(tmp_path, material):
    profile = microscope_photometry_profile(material)
    assert supported_photometry_profile(profile)
    assert stacking._image_calibration(_frame(tmp_path, profile)).mm_per_pixel_x == 0.001


def test_stacking_still_rejects_an_unknown_declaration(tmp_path):
    profile = {**microscope_photometry_profile(V1_MATERIAL_CAPTURE_PROFILE), "stacking_camera": "other"}
    assert not supported_photometry_profile(profile)
    with pytest.raises(NeedsHumanError):
        stacking._image_calibration(_frame(tmp_path, profile))
