"""Independent carried-layer PSF and original-profile contracts."""
import io
from copy import deepcopy

import pytest
from PIL import Image

from qumus_simlab.core.errors import NeedsHumanError
from qumus_simlab.devices.base import Frame
from qumus_simlab.flows.library.stacking import _image_calibration
from qumus_simlab.testbench.render import (
    STAMP_PSF_PROFILE,
    _convolve_stamp_plane,
    _PremultipliedStampPlane,
    _stamp_psf_kernel,
    microscope_photometry_profile,
)


@pytest.mark.parametrize("sharpness", [0.5, 4.5, 8.5])
def test_positive_unit_kernel_conserves_premultiplied_mass_centroid_and_support(sharpness):
    plane = _PremultipliedStampPlane(96, 96)
    plane.polygon([(35, 37), (59, 37), (59, 57), (35, 57)], (80, 120, 190), 0.3)
    sigma, weights = _stamp_psf_kernel(sharpness)
    assert min(weights) > 0 and sum(weights) == pytest.approx(1.0)
    assert weights == list(reversed(weights))
    for image in plane.images():
        before = image.get_flattened_data()
        after = _convolve_stamp_plane(image, weights).get_flattened_data()
        assert sum(after) == pytest.approx(sum(before), rel=2e-6)
        for axis in (0, 1):

            def coordinate(i, axis=axis):
                return i % 96 if axis == 0 else i // 96

            a = sum(v * coordinate(i) for i, v in enumerate(before)) / sum(before)
            b = sum(v * coordinate(i) for i, v in enumerate(after)) / sum(after)
            assert a == pytest.approx(b, abs=2e-5)
        assert min(after) >= 0
    radius = len(weights) // 2
    assert _convolve_stamp_plane(plane.images()[3], weights).getbbox() == (
        35 - radius,
        37 - radius,
        59 + radius,
        57 + radius,
    )
    if sigma == 0:
        assert (
            _convolve_stamp_plane(plane.images()[3], weights).tobytes()
            == plane.images()[3].tobytes()
        )


@pytest.mark.parametrize("profile", ["missing", "old"])
def test_old_or_missing_raster_profile_refuses_current_capture_reuse(profile):
    image = Image.new("RGB", (40, 30))
    data = io.BytesIO()
    image.save(data, format="PNG")
    calibration = {"source": "synthetic", "frame": "microscope_image", "u_axis": "+world_x", "v_axis": "-world_y",
                   "image_size_px": [40, 30], "mm_per_pixel": [.002, .002],
                   "photometry_profile": microscope_photometry_profile()}
    original = deepcopy(calibration)
    if profile == "missing":
        calibration["photometry_profile"].pop("raster")
    else:
        calibration["photometry_profile"]["raster"] = "old-raster"
    frame = Frame("microscope", 100, data=data.getvalue(), meta={"calibration": calibration})
    with pytest.raises(NeedsHumanError, match="capture calibration"):
        _image_calibration(frame)
    assert original["photometry_profile"]["raster"] == STAMP_PSF_PROFILE
