"""A resumed lab never reuses a picture name of the runs it continues."""

from __future__ import annotations

from qumus_simlab.simlab.server import highest_picture_number, number_pictures_after
from qumus_simlab.testbench import render


def test_resumed_writer_numbers_after_earlier_pictures(tmp_path):
    old = tmp_path / "old" / "images"
    (old / "microscope").mkdir(parents=True)
    (old / "microscope" / "microscope_000516.jpg").write_bytes(b"x")
    (old / "camera").mkdir()
    (old / "camera" / "overview_camera_000012.png").write_bytes(b"x")
    (old / "microscope" / "microscope_live.jpg").write_bytes(b"x")
    assert highest_picture_number([old, tmp_path / "missing"]) == 516
    w = render._Writer(tmp_path / "new" / "microscope")
    assert number_pictures_after([old]) == 516
    path = w.write("microscope", b"\xff\xd8\xff" + b"new picture")
    assert path.endswith("microscope_000517.jpg")


def test_fresh_writer_starts_at_one(tmp_path):
    w = render._Writer(tmp_path / "m")
    assert w.write("microscope", b"\x89PNG\r\n\x1a\n" + b"a").endswith("microscope_000001.png")
