"""The v1 camera focuses on the preview raster and captures on the full
sensor through the same objective: the focus binding admits that pairing
and nothing else."""
from qumus_simlab.flows.library.stacking import _same_capture_optics


def _cal(size, fov=(0.69888, 0.39312), **extra):
    return {"id": "synthetic-microscope-optics", "version": f"v2:20:{size[0]}x{size[1]}:{fov[0]}:{fov[1]}",
            "source": "synthetic", "image_size_px": list(size),
            "mm_per_pixel": [fov[0]/size[0], fov[1]/size[1]],
            "material_capture_profile": "v1-reference-material-capture-v2", **extra}


def test_full_sensor_capture_matches_its_preview_raster_af():
    assert _same_capture_optics(_cal((3840, 2160)), _cal((480, 320)))
    assert _same_capture_optics(_cal((480, 320)), _cal((480, 320)))


def test_other_optics_or_field_still_differ():
    assert not _same_capture_optics(_cal((3840, 2160)), _cal((480, 320), fov=(1.4, 0.79)))
    assert not _same_capture_optics(_cal((3840, 2160)), _cal((480, 320), camera_profile="other"))
    other = _cal((480, 320))
    other["material_capture_profile"] = None
    assert not _same_capture_optics(_cal((3840, 2160)), other)
    bad = _cal((480, 320))
    bad["mm_per_pixel"] = [0.002, 0.002]
    assert not _same_capture_optics(_cal((3840, 2160)), bad)
    assert not _same_capture_optics(None, _cal((480, 320)))
