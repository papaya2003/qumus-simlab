"""Independent renderer and pixel diagnostics retained after servo retirement."""
from pathlib import Path

from qumus_simlab.devices.base import Frame
from qumus_simlab.perception.tracking import contact_front_evidence
from qumus_simlab.testbench.render import MicroscopeRenderer


def renderer_scene(**extra):
    return dict(fov_mm=[.65, .43], objective=20, sharpness_chip=8., sharpness_stamp=8.,
                on_chip="fixture", stamp={"film": True}, **extra)


def test_fileless_renderer_matches_saved_pixels_without_writer(tmp_path, monkeypatch):
    scene = renderer_scene(contact=dict(dx_mm=.02, dy_mm=.1, radius_mm=.06))
    saved, _ = MicroscopeRenderer(tmp_path).render("fixture", scene)
    expected = Path(saved).read_bytes()
    before = tuple(tmp_path.iterdir())

    def forbidden(*args, **kwargs):
        raise AssertionError("A fileless raster render attempted filesystem access")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "write_bytes", forbidden)
        fileless = MicroscopeRenderer(None)
        assert fileless.writer is None
        assert fileless.cache is None
        assert fileless.render_bytes("fixture", scene) == expected
    assert tuple(tmp_path.iterdir()) == before


def test_fringe_pixels_cannot_become_contact_from_metadata(tmp_path):
    scene = renderer_scene(fringes=dict(dx_mm=0., dy_mm=.1, gap_mm=.001, rings=4))
    data = MicroscopeRenderer(None).render_bytes("fixture", scene)
    frame = Frame("scope", 0., data=data,
                  meta=dict(fov_mm=[.65, .43], contact=dict(radius_mm=1., film=True)))
    assert contact_front_evidence(frame).has_contact is False


def test_contact_diagnostic_uses_pixels_ignoring_contradictory_metadata(tmp_path):
    scene = renderer_scene(contact=dict(dx_mm=.02, dy_mm=.1, radius_mm=.06))
    path, _ = MicroscopeRenderer(tmp_path).render("fixture", scene)
    frame = Frame("scope", 123., path=path, meta={"contact": None, "stamp": {"film": False}})
    evidence = contact_front_evidence(frame)
    assert evidence.has_contact and evidence.centroid_px is not None and evidence.radius_px > 0
    assert evidence.film_intact is None
    assert evidence == contact_front_evidence(path)
    frame.meta.update(contact={"radius_mm": 1.}, fov_mm=[999, 999])
    assert contact_front_evidence(frame) == evidence
