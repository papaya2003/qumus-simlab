"""v1-like approach near contact, rings at the front, and the PC film a
release leaves on the chip (#179, #180)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qumus_simlab.config import load_yaml
from qumus_simlab.flows.library.stacking import _approach_speed
from qumus_simlab.testbench.optics import WHITE_LIGHT_PROFILE
from qumus_simlab.testbench.render import Canvas, MicroscopeRenderer, _film_disc
from qumus_simlab.testbench.world import SimFlake
from tests.test_transfer_physics import mounted

SX, SY = 1/0.005416666666666667, 1/0.005375  # 5x px per mm


def _rings(depth, front, radius=12.):
    c = Canvas(480, 320, (120, 120, 120))
    MicroscopeRenderer._white_light_rings(c, {"signed_apex_gap_mm": -depth, "dome_radius_mm": radius,
        "centre_dx_mm": 0., "centre_dy_mm": 0., "current_contact_radius_mm": front,
        "has_contact": True, "profile_id": WHITE_LIGHT_PROFILE}, SX, SY)
    return np.frombuffer(bytes(c.px), np.uint8).reshape(320, 480, 3)


@pytest.mark.parametrize("depth,front", [(.010, .372), (.025, .710), (.004, .235)])
def test_rings_start_at_a_lagging_front(depth, front):
    """The fringe family starts at the drawn front, not at the rigid overlap."""
    row = _rings(depth, front)[160, 240:].astype(int)
    start = math.ceil(front*SX)
    changed = np.nonzero(np.abs(row[start:]-120).max(axis=1) > 1)[0]
    assert changed[0] <= 1
    assert math.sqrt(2*12.*depth)*SX-front*SX > 5  # the old profile left a gap here


def test_rings_before_contact_are_unchanged():
    c = Canvas(480, 320, (120, 120, 120))
    MicroscopeRenderer._white_light_rings(c, {"signed_apex_gap_mm": .001, "dome_radius_mm": 12.,
        "centre_dx_mm": 0., "centre_dy_mm": 0., "current_contact_radius_mm": 0.,
        "has_contact": False, "profile_id": WHITE_LIGHT_PROFILE}, SX, SY)
    row = np.frombuffer(bytes(c.px), np.uint8).reshape(320, 480, 3)[160, 240:]
    first = np.nonzero(np.abs(row.astype(int)-120).max(axis=1) > 1)[0][0]
    assert first == 0  # the paraboloid apex at 1 um gap is tinted


def test_v1_approach_profile():
    c = load_yaml("verification.yaml")["transfer_sequence"]
    assert (c["approach_speed_mm_s"], c["approach_standoff_mm"], c["descent_speed_mm_s"],
            c["press_speed_mm_s"], c["drop_speed_mm_s"]) == (.05, .2, .005, .002, .02)
    tol, contact = .005, 5.4
    # far: fast; the target stays on the phase's intent grid
    assert _approach_speed(c, 5.65, 5.55, contact, tol) == (.05, 5.55)
    assert _approach_speed(c, 6.5, 6.4, contact, tol) == (.05, 6.4)
    # within the standoff: touching speed
    assert _approach_speed(c, 5.603, 5.503, contact, tol) == (.005, 5.503)
    assert _approach_speed(c, 5.45, 5.35, contact, tol) == (.005, 5.35)
    # past the expected contact: the pickup press speed
    assert _approach_speed(c, 5.402, 5.302, contact, tol) == (.002, 5.302)


def test_hot_release_pins_the_film_and_leaves_a_disc_on_the_chip():
    lab, clock, chip, st = mounted(dynamics=False)
    w = lab.world
    flake = SimFlake("carried", "hBN", 30, 0, 0, 100, outline_mm=((-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01)))
    chip.flakes = []
    st.stack = [(flake, 0, 0)]
    w._transfer_epoch(clock.now())
    w.stage_temp = lambda: 200
    manip = lab.devices.get("transfer_arm")
    manip.move_axis_to("z", 5.29)
    clock.advance(5)
    assert st.pinned_radius_mm == 0  # pinned at a control instant, not at a poll
    manip.move_axis_to("z", 5.295)
    pinned, centre = st.pinned_radius_mm, st.pinned_xy
    assert pinned == pytest.approx(math.sqrt(2*w.model["stamp"]["dome_radius_mm"]*.01), rel=.02)
    clock.advance(5)
    w.advance_transfer(clock.now())
    assert 0 < st.front_radius_mm < pinned and st.pinned_radius_mm == pinned  # separating inside it
    manip.move_axis_to("z", 6.5)
    assert [f.id for f in chip.flakes] == ["carried"] and not st.stack and not st.film
    assert st.pinned_radius_mm == 0 and st.pinned_xy is None
    [disc] = chip.pc_films
    assert disc["radius_mm"] == pinned
    assert chip.local_to_world(disc["x_mm"], disc["y_mm"]) == pytest.approx(centre)


def test_cold_lift_pins_nothing():
    lab, clock, chip, st = mounted(dynamics=False)
    manip = lab.devices.get("transfer_arm")
    manip.move_axis_to("z", 5.29)
    clock.advance(5)
    manip.move_axis_to("z", 5.295)
    assert st.pinned_radius_mm == 0
    manip.move_axis_to("z", 6.5)
    assert chip.pc_films == [] and st.film


def test_film_disc_keeps_the_stack_visible_and_draws_a_torn_edge():
    c = Canvas(100, 100, (100, 60, 140))
    _film_disc(c, 50, 50, 30, fill=(236, 226, 196), fill_alpha=.06, edge=(92, 80, 70), edge_alpha=.45, torn=True)
    img = np.frombuffer(bytes(c.px), np.uint8).reshape(100, 100, 3).astype(int)
    inside, outside = img[50, 50], img[50, 2]
    assert list(outside) == [100, 60, 140]
    assert np.abs(inside-[100, 60, 140]).max() <= 10  # faint film over whatever is under it
    ring = img[50, 50+27:50+34]
    assert (np.abs(ring-[100, 60, 140]).max(axis=1) > 15).any()  # visible edge near r = 30
