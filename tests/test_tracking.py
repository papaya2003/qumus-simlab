"""Raster-only tracking fixtures: labels here are not algorithm inputs."""

import io
import math
import random

import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from qumus_simlab.devices.base import Frame
from qumus_simlab.perception.tracking import (
    PixelROI,
    RegistrationConfig,
    contact_front_evidence,
    load_pixels,
    polygon_roi,
    raster_focus_score,
    register_specimen,
)


def target_image():
    image = Image.new("RGB", (160, 144), (125, 102, 171))
    draw = ImageDraw.Draw(image)
    draw.polygon([(61, 45), (97, 53), (91, 91), (56, 82)], fill=(45, 40, 63))
    draw.polygon([(66, 51), (86, 61), (72, 74)], fill=(200, 177, 211))
    draw.rectangle((83, 76, 91, 86), fill=(137, 98, 155))
    draw.line((56, 82, 96, 52), fill=(236, 230, 246), width=2)
    return image


ROI = PixelROI(47, 34, 108, 102)


def test_raster_focus_rejects_blur():
    image = target_image()
    blurred = image.filter(ImageFilter.GaussianBlur(5))
    assert raster_focus_score(blurred, ROI) < raster_focus_score(image, ROI)*0.2


def test_color_focus_retains_equal_luminance_specimen_edges():
    image = Image.new("RGB", (160, 144), (200, 50, 50))
    # Both colours convert to luminance 95; geometry is visible in RGB alone.
    ImageDraw.Draw(image).rectangle((50, 35, 110, 105), fill=(50, 126, 50))
    assert image.convert("L").getextrema() == (95, 95)
    assert raster_focus_score(image) == 0
    sharp = raster_focus_score(image, color=True)
    assert sharp > 0
    assert raster_focus_score(image.filter(ImageFilter.GaussianBlur(5)), color=True) < sharp
    assert raster_focus_score(Image.new("RGB", image.size, (200, 50, 50)), color=True) == 0


def test_frame_pixels_ignore_contradictory_metadata_and_save_only_once(tmp_path):
    image = target_image()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    frame = Frame("scope", 0, data=buf.getvalue(), meta={"dx_mm": 100, "contact": True})
    assert load_pixels(frame).tobytes() == image.tobytes()
    path = tmp_path/"capture.png"
    image.save(path)
    calls = []
    lazy = Frame("scope", 1, save=lambda: calls.append("save") or str(path))
    assert load_pixels(lazy).tobytes() == image.tobytes()
    load_pixels(lazy)
    assert calls == ["save"]


def contact_image(*, fringes=False, broken=False):
    # Exact composited palette around the renderer's ordinary substrate.
    image = Image.new("RGB", (160, 144), (125, 102, 171))
    draw = ImageDraw.Draw(image)
    if fringes:
        for radius in (48, 40, 31, 23):
            draw.ellipse((80-radius, 72-radius, 80+radius, 72+radius),
                         outline=(111, 90, 155), width=3)
    else:
        draw.ellipse((46, 38, 114, 106), fill=(87, 70, 126))
        draw.ellipse((49, 41, 111, 103), fill=(128, 115, 160))
        if broken:
            draw.line((53, 61, 99, 82, 65, 91, 105, 59), fill=(41, 30, 53), width=3)
    return image


def test_contact_filled_front_and_precontact_rings_differ():
    contact = contact_front_evidence(contact_image())
    assert contact.has_contact
    assert contact.centroid_px == pytest.approx((80, 72), abs=1)
    assert contact.radius_px == pytest.approx(31, abs=1)
    assert contact.film_intact is None  # No visible crack cannot prove integrity.
    rings = contact_front_evidence(contact_image(fringes=True))
    assert not rings.has_contact
    assert rings.rings > 0
    assert rings.failure_reason == "fringes_only"


def test_broken_film_and_dark_flake_are_not_intact_contact_proof():
    result = contact_front_evidence(contact_image(broken=True))
    assert result.has_contact
    assert result.film_intact is False
    assert result.failure_reason == "film_discontinuity"
    assert not contact_front_evidence(target_image()).has_contact


def render_stamp(tmp_path, *, radius=None, gap=None, film=True, size=(800, 530), flakes=()):
    from qumus_simlab.testbench.render import MicroscopeRenderer

    scene = dict(fov_mm=[.65, .43], objective=20, sharpness_chip=8., sharpness_stamp=8.,
                 on_chip="fixture", stamp={"film": film}, flakes=list(flakes))
    if radius is not None:
        scene["contact"] = dict(dx_mm=0., dy_mm=.1, radius_mm=radius)
    if gap is not None:
        scene["fringes"] = dict(dx_mm=0., dy_mm=.1, gap_mm=gap,
                                rings=int(2*gap/.00055)+1)
    path, _ = MicroscopeRenderer(tmp_path, size=size).render("fixture", scene)
    return path


@pytest.mark.parametrize("radius", [.002, .03, .1, .2])
def test_actual_renderer_first_front_growth_and_partial_fov(tmp_path, radius):
    evidence = contact_front_evidence(render_stamp(tmp_path, radius=radius))
    assert evidence.has_contact, evidence
    assert evidence.centroid_px == pytest.approx((400, 265-.1*530/.43), abs=3)
    # Physical contact is the light interior, excluding dark annular thickness.
    reserve_px = 3 if evidence.failure_reason == "boundary_clipped" else 2
    assert evidence.radius_px == pytest.approx(radius*800/.65, abs=reserve_px)
    assert evidence.film_intact is None
    if radius == .2:
        assert evidence.failure_reason == "boundary_clipped"


def test_actual_renderer_fringes_and_torn_film(tmp_path):
    for gap in (.0005, .003):
        evidence = contact_front_evidence(render_stamp(tmp_path, gap=gap))
        assert not evidence.has_contact, evidence
    evidence = contact_front_evidence(render_stamp(tmp_path, radius=.06, film=False))
    assert evidence.has_contact, evidence
    assert evidence.film_intact is False


def test_actual_renderer_default_size_and_flake_outlines(tmp_path):
    evidence = contact_front_evidence(render_stamp(tmp_path, radius=.03, size=(480, 320)))
    assert evidence.has_contact
    assert evidence.centroid_px == pytest.approx((240, 160-.1*320/.43), abs=2)
    assert evidence.radius_px == pytest.approx(.03*480/.65, abs=2)
    flakes = [dict(sim_id=f"flake_{i}", material=material, layers=layers, area_um2=1000,
                   dx_mm=(i-1)*.08, dy_mm=0.)
              for i, (material, layers) in enumerate((("Gr", 1), ("hBN", 1), ("Gr", 100)))]
    evidence = contact_front_evidence(render_stamp(tmp_path, flakes=flakes))
    assert not evidence.has_contact, evidence


def test_focus_ignores_constant_border_legend():
    image = Image.new("RGB", (160, 144), "gray")
    draw = ImageDraw.Draw(image)
    draw.text((4, 2), "20X", fill="white")
    draw.line((120, 137, 154, 137), fill="white", width=3)
    assert raster_focus_score(image) == 0


def optical_scene(*, sharpness=10.5, dx=0., dy=0., fov=.65, width=.03, height=.02,
                  absent=False):
    flake = dict(sim_id="same-specimen", material="hBN", layers=60, area_um2=width*height*1e6,
                 dx_mm=dx, dy_mm=dy,
                 outline_mm=[[dx+x, dy+y] for x, y in
                             [(-width/2, -height/2), (width/2, -height/2),
                              (width/2, height/2), (-width/2, height/2)]])
    return dict(fov_mm=[fov, fov*2/3], objective=20, on_chip="same-chip",
                sharpness_chip=sharpness, flakes=[] if absent else [flake])


def render_optical(scene, *, size=(480, 320)):
    from qumus_simlab.devices.base import Frame
    from qumus_simlab.testbench.render import MicroscopeRenderer

    return Frame("scope", 0., data=MicroscopeRenderer(None, size=size).render_bytes("scope", scene))


def test_renderer_focus_curve_rejects_noise_and_resolves_optical_plateau():
    levels = (.5, 2., 4., 6., 8., 9., 10.5)
    frames = [render_optical(optical_scene(sharpness=level, width=.16, height=.12))
              for level in levels]
    scores = [raster_focus_score(frame) for frame in frames]
    assert scores == sorted(scores)
    assert scores[-1] > 2*scores[0]
    # The renderer quantizes its near-focus blur: the observed peak is a
    # plateau, and cannot justify inventing a more precise focus coordinate.
    assert scores[-1] == scores[-2]
    interior_frames = [render_optical(optical_scene(sharpness=level, fov=.13,
                                                   width=.16, height=.12)) for level in levels]
    assert [raster_focus_score(frame) for frame in interior_frames] == [0.]*len(levels)


@pytest.mark.parametrize("gap", [.0005, .001, .003])
@pytest.mark.parametrize("fov,dy,objective", [(2.6, .35, 5), (.65, .07, 20)])
def test_renderer_white_precontact_fringes_report_presence_without_contact_geometry(gap, fov, dy, objective):
    scene = dict(fov_mm=[fov, fov*1.72/2.6], objective=objective, on_chip="fringe-fixture",
                 sharpness_chip=8., sharpness_stamp=8., stamp={"film": True},
                 fringes=dict(dx_mm=.03, dy_mm=dy, gap_mm=gap, rings=int(2*gap/.00055)+1))
    frame = render_optical(scene)
    # Contradictory metadata cannot turn the captured fringes into contact.
    frame.meta = {"fringes": {"dx_mm": -100., "dy_mm": -100.}, "contact": True}
    evidence = contact_front_evidence(frame)
    assert evidence == contact_front_evidence(load_pixels(frame))
    assert not evidence.has_contact
    assert evidence.rings == 1  # Presence, not inferred ring order or air gap.
    assert evidence.centroid_px is None
    assert evidence.radius_px is None and evidence.outer_radius_px is None
    assert evidence.angular_coverage == 0.
    assert evidence.quality == 0.
    assert evidence.film_intact is None
    assert evidence.boundary_diagnostics is None
    assert evidence.failure_reason == "fringes_only"


@pytest.mark.parametrize("gap", [.0005, .001, .003])
def test_off_fov_fringe_centre_cannot_authorize_servo(gap):
    scene = dict(fov_mm=[.65, .43], objective=20, on_chip="fringe-fixture",
                 sharpness_chip=8., sharpness_stamp=8., stamp={"film": True},
                 fringes=dict(dx_mm=.03, dy_mm=.35, gap_mm=gap, rings=int(2*gap/.00055)+1))
    evidence = contact_front_evidence(render_optical(scene))
    assert not evidence.has_contact
    assert evidence.centroid_px is None
    assert evidence.quality == 0


def test_solid_purple_flake_and_sensor_texture_cannot_supply_fringe_centre():
    image = Image.new("RGB", (160, 144), (125, 102, 171))
    draw = ImageDraw.Draw(image)
    # Same purple-channel discriminator as fringe light/dark bands, but one
    # uniformly filled circle has no concentric radial alternations.
    draw.ellipse((45, 37, 115, 107), fill=(132, 118, 194))
    evidence = contact_front_evidence(image)
    assert not evidence.has_contact
    assert evidence.centroid_px is None
    assert evidence.quality == 0
    frame = render_optical(optical_scene(fov=.13, width=.16, height=.12))
    assert contact_front_evidence(frame).centroid_px is None


def test_tiny_specimen_deleted_from_stationary_texture_cannot_pass_presence():
    rng = random.Random(331)
    background = Image.new("RGB", (60, 40))
    background.putdata([tuple(rng.randrange(30, 230) for _ in range(3))
                        for _ in range(60*40)])
    background = background.resize((240, 160), Image.Resampling.NEAREST)
    reference = background.copy()
    ImageDraw.Draw(reference).rectangle((117, 77, 123, 83), fill=(45, 25, 90))
    polygon = [(117, 77), (123, 77), (123, 83), (117, 83)]
    cfg = RegistrationConfig(max_translation_px=12, max_rotation_deg=0, min_quality=.72)
    absent = register_specimen(reference, background, polygon_px=polygon, config=cfg)
    assert not absent.ok
    assert absent.loss_reason == "specimen_not_present"
    present = register_specimen(reference, reference, polygon_px=polygon, config=cfg)
    assert present.ok
    current = background.copy()
    ImageDraw.Draw(current).rectangle((123, 73, 129, 79), fill=(45, 25, 90))
    moved = register_specimen(reference, current, polygon_px=polygon, config=cfg)
    assert moved.ok, moved
    assert (moved.dx_px, moved.dy_px) == pytest.approx((6, -4), abs=1)
    assert polygon_roi(polygon).right-polygon_roi(polygon).left >= 12


@pytest.mark.parametrize("material", ["Gr", "hBN"])
@pytest.mark.parametrize("brightness", [1., .8])
def test_low_contrast_renderer_specimen_tracks_despite_fixed_noise_and_light(material, brightness):
    source = optical_scene()
    source["flakes"][0].update(material=material, layers=1)
    moved = optical_scene(dx=.01, dy=.006)
    moved["flakes"][0].update(material=material, layers=1)
    reference, current = render_optical(source), render_optical(moved)
    current = ImageEnhance.Brightness(load_pixels(current)).enhance(brightness)
    polygon = [(240+x*480/.65, 160-y*320/(.65*2/3))
               for x, y in source["flakes"][0]["outline_mm"]]
    # Orientation remains bound by the complete observed outline; this weak
    # contrast image supplies translational presence, not a rotation reading.
    config = RegistrationConfig(max_translation_px=24, max_rotation_deg=0, min_quality=.72)
    result = register_specimen(reference, current, polygon_px=polygon, config=config)
    assert result.ok, result
    assert result.dx_px == pytest.approx(.01*480/.65, abs=1.5)
    assert result.dy_px == pytest.approx(-.006*320/(.65*2/3), abs=1.5)
    assert abs(result.rotation_deg) <= 2
    missing = render_optical(optical_scene(absent=True))
    assert not register_specimen(reference, missing, polygon_px=polygon, config=config).ok


def test_specimen_outline_validation_and_unobservable_contrast():
    image = Image.new("RGB", (160, 144), (125, 102, 171))
    polygon = [(60, 50), (90, 50), (90, 80), (60, 80)]
    result = register_specimen(image, image, polygon_px=polygon)
    assert not result.ok
    assert result.loss_reason == "no_specimen_contrast"
    assert not register_specimen(image, image, polygon_px=[(1, 2)]).ok
    clipped = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert register_specimen(image, image, polygon_px=clipped).loss_reason == "specimen_outline_clipped"


def test_repeated_specimen_signatures_remain_ambiguous():
    reference = Image.new("RGB", (160, 144), (125, 102, 171))
    polygon = [(70, 60), (80, 60), (80, 70), (70, 70)]
    ImageDraw.Draw(reference).polygon(polygon, fill=(55, 35, 90))
    current = Image.new("RGB", reference.size, (125, 102, 171))
    for delta in (-14, 14):
        ImageDraw.Draw(current).polygon([(x+delta, y) for x, y in polygon], fill=(55, 35, 90))
    result = register_specimen(reference, current, polygon_px=polygon,
                               config=RegistrationConfig(max_translation_px=24, max_rotation_deg=0))
    assert not result.ok
    assert result.loss_reason == "ambiguous_specimen"


def test_partial_front_strict_identity_and_opaque_or_deleted_source_refusal():
    polygon = [(276.7, 153.), (254.3, 140.), (218.6, 150.), (215.7, 174.7), (241.1, 183.3)]
    outline = [[(x-240)*.65/480, -(y-160)*.43/320] for x, y in polygon]
    flake = dict(sim_id="partial-source", material="hBN", layers=95, area_um2=1000,
                 dx_mm=0., dy_mm=0., outline_mm=outline)
    scene = dict(fov_mm=[.65, .43], objective=20, on_chip="fixed-textured-substrate",
                 sharpness_chip=8., sharpness_stamp=8., flakes=[flake])
    reference = render_optical(scene)
    front = dict(stamp={"film": True}, contact=dict(dx_mm=0., dy_mm=.275, radius_mm=.26))
    current = render_optical(scene | front)
    config = RegistrationConfig(max_translation_px=12, max_rotation_deg=0,
                                min_quality=.72, min_ambiguity=.02)
    baseline = register_specimen(reference, reference, polygon_px=polygon, config=config)
    strict = register_specimen(reference, current, polygon_px=polygon, config=config)
    assert baseline.ok
    assert baseline.photometric_calibration is None
    assert baseline.photometric_noise_gain == 1.
    assert not strict.ok
    # Keep the renderer's exact camera/residue seed and front, moving only the
    # specimen off the source. Removing its label would change the noise seed.
    absent_flake = flake | dict(dx_mm=100., outline_mm=[[x+100, y] for x, y in outline])
    absent = render_optical(scene | front | {"flakes": [absent_flake]})
    assert load_pixels(current).crop((20, 190, 180, 280)).tobytes() == load_pixels(absent).crop((20, 190, 180, 280)).tobytes()
    deleted = register_specimen(reference, absent, polygon_px=polygon, config=config)
    assert not deleted.ok
    assert deleted.loss_reason == "specimen_not_present"
    covered_front = dict(stamp={"film": True}, contact=dict(dx_mm=0., dy_mm=.275, radius_mm=.28))
    covered = render_optical(scene | covered_front)
    absent_inside = render_optical(scene | {"flakes": [absent_flake], "stamp": {"film": True},
                                           "contact": dict(dx_mm=0., dy_mm=.275, radius_mm=.28)})
    assert not register_specimen(reference, absent_inside, polygon_px=polygon, config=config).ok
    opaque = load_pixels(covered)
    ImageDraw.Draw(opaque).rectangle((210, 135, 283, 190), fill=(56, 44, 90))
    assert not register_specimen(reference, opaque, polygon_px=polygon, config=config).ok


@pytest.mark.parametrize("radius", [.1, .2, .3, .4, .5, .6])
@pytest.mark.parametrize("dx", [0., .03])
def test_clipped_thick_annulus_has_shared_boundary_centre(radius, dx):
    scene = dict(fov_mm=[1.3, .86], objective=10, on_chip="clipped-annulus-fixture",
                 sharpness_chip=8., sharpness_stamp=8., stamp={"film": True},
                 contact=dict(dx_mm=dx, dy_mm=.35, radius_mm=radius))
    evidence = contact_front_evidence(render_optical(scene))
    assert evidence.has_contact, evidence
    assert evidence.failure_reason == "boundary_clipped"
    # The centre is an independent renderer oracle only. The measured raster
    # contains a thick annulus cut by the top image border, not a full circle.
    assert evidence.centroid_px == pytest.approx((240+dx*480/1.3, 160-.35*320/.86), abs=2.)
    # The declared clipped-front reserve is three pixels. Require an even
    # tighter radius error here; display-band thickness must not add coverage.
    assert evidence.radius_px == pytest.approx(radius*480/1.3, abs=2.)
    assert evidence.angular_coverage >= .45
    assert evidence.film_intact is None


def test_contact_without_observable_inner_and_outer_arcs_cannot_supply_centre():
    full_field = dict(fov_mm=[1.3, .86], objective=10, on_chip="full-field-fixture",
                      sharpness_chip=8., sharpness_stamp=8., stamp={"film": True},
                      contact=dict(dx_mm=0., dy_mm=.35, radius_mm=2.))
    evidence = contact_front_evidence(render_optical(full_field))
    assert not evidence.has_contact
    assert evidence.centroid_px is None
    # Image-border cut edges alone cannot qualify as physical annular arcs.
    dark = Image.new("RGB", (480, 320), (87, 70, 126))
    evidence = contact_front_evidence(dark)
    assert not evidence.has_contact
    assert evidence.centroid_px is None


def large_source_contact_image(*, dx=0., radius=.376):
    scene = optical_scene(fov=1.3, width=.16, height=.12)
    scene.update(fov_mm=[1.3, .86], objective=10, sharpness_chip=8., sharpness_stamp=8.,
                 stamp={"film": True}, contact=dict(dx_mm=dx, dy_mm=.35, radius_mm=radius))
    return load_pixels(render_optical(scene))


def test_large_source_interrupts_front_but_consistent_opposite_arcs_are_observable():
    image = large_source_contact_image()
    evidence = contact_front_evidence(image)
    assert evidence.has_contact, evidence
    assert evidence.centroid_px == pytest.approx((240., 160-.35*320/.86), abs=2.)
    assert evidence.radius_px == pytest.approx(.376*480/1.3, abs=2.)
    assert evidence.angular_coverage >= .45
    assert evidence.failure_reason == "boundary_clipped"
    # Neither visible side alone establishes the unchanged coverage threshold.
    for left in (False, True):
        isolated = image.copy()
        ImageDraw.Draw(isolated).rectangle((240 if left else 0, 0,
                                            479 if left else 239, 319), fill=(125, 102, 171))
        assert not contact_front_evidence(isolated).has_contact


@pytest.mark.parametrize("separate", ["centres", "radii"])
def test_unrelated_or_concentric_different_radius_arcs_cannot_be_associated(separate):
    if separate == "centres":
        left, right = large_source_contact_image(dx=-.08), large_source_contact_image(dx=.08)
    else:
        left, right = large_source_contact_image(radius=.35), large_source_contact_image(radius=.4)
    unrelated = left.copy()
    unrelated.paste(right.crop((240, 0, 480, 320)), (240, 0))
    evidence = contact_front_evidence(unrelated)
    assert not evidence.has_contact, evidence
    assert evidence.centroid_px is None


@pytest.mark.parametrize("offset_px", [(.25, -.75), (1.5, .25), (-2.75, 3.5),
                                       (14.7692307692, 0.)])
@pytest.mark.parametrize("size", [(480, 320), (960, 640)])
def test_fractional_native_grid_translation_error_is_inside_specimen_reserve(offset_px, size):
    pixel_mm = .65/size[0]
    factor = size[0]/480
    offset_px = tuple(value*factor for value in offset_px)
    outline = ((-.065, -.035), (.04, -.035), (.065, .005), (.017, .05), (-.05, .03))
    polygon = [(size[0]/2+x/pixel_mm, size[1]/2-y/pixel_mm) for x, y in outline]
    flake = dict(sim_id="fractional-hbn90", material="hBN", layers=90,
                 area_um2=8000., dx_mm=0., dy_mm=0., outline_mm=list(outline))
    moved = flake | dict(dx_mm=offset_px[0]*pixel_mm, dy_mm=-offset_px[1]*pixel_mm,
                         outline_mm=[(x+offset_px[0]*pixel_mm, y-offset_px[1]*pixel_mm)
                                     for x, y in outline])
    scene = dict(fov_mm=[.65, .65*2/3], objective=20, on_chip="fractional-grid-fixture",
                 sharpness_chip=10.5, sharpness_stamp=10.5)
    reference = render_optical(scene | {"stamp": {"film": True}, "stamp_flakes": [flake]}, size=size)
    current = render_optical(scene | {"stamp": {"film": True}, "stamp_flakes": [moved]}, size=size)
    result = register_specimen(reference, current, polygon_px=polygon,
                               config=RegistrationConfig(max_translation_px=24*factor, max_rotation_deg=0,
                                                         min_quality=.72, min_ambiguity=.02))
    assert result.ok, result
    error = math.dist((result.dx_px, result.dy_px), offset_px)
    assert error <= result.uncertainty_px
    # Native and working-image geometric reserves stay fixed; only the
    # photometric residual/ambiguity penalty receives inverse-noise gain.
    assert result.uncertainty_px >= 1.5


        # A single observed-background fit retains its original scope. The
        # crossing-front case above cannot borrow retired regional fitting.

def test_annular_palette_membership_without_a_real_outer_edge_is_not_contact():
    image = Image.new("RGB", (160, 144), (102, 86, 144))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 32, 120, 112), fill=(98, 80, 140))
    draw.ellipse((44, 36, 116, 108), fill=(106, 96, 145))
    # The mask describes a circular annulus and the interior is brighter, but
    # the outer six-level step does not rise above raster noise/quantization.
    evidence = contact_front_evidence(image)
    assert not evidence.has_contact
    assert evidence.centroid_px is None


def test_growing_clipped_front_does_not_follow_vignette_mask_edges():
    centers = []
    dx, dy = .0024, .34825
    for radius in (.467, .469, .471, .472, .473, .474, .475):
        scene = dict(fov_mm=[1.3, .86], objective=10, on_chip="chip_002",
                     sharpness_chip=10.5, sharpness_stamp=10.5, stamp={"film": True},
                     contact=dict(dx_mm=dx, dy_mm=dy, radius_mm=radius))
        evidence = contact_front_evidence(render_optical(scene))
        assert evidence.has_contact, evidence
        assert math.dist(evidence.centroid_px, (240+dx*480/1.3, 160-dy*320/.86)) <= 1.
        assert evidence.radius_px == pytest.approx(radius*480/1.3, abs=2.)
        centers.append(evidence.centroid_px)
    # The physical centre stays fixed as the radius crosses the vignette's
    # mask threshold. Camera-fixed background classification cannot move it.
    assert max(math.dist(left, right) for left, right in zip(centers, centers[1:], strict=False)) < .25


@pytest.mark.parametrize('condition',['contact','no_contact','fringes_close','fringes_far','tear'])
def test_local_dark_front_over_metal_keeps_contact_and_integrity_gates(condition):
    scene=dict(fov_mm=[1.3,.86],objective=10,on_chip='local-front-metal',
               sharpness_chip=8.5,sharpness_stamp=8.5,stamp={'film':condition!='tear'},
               metal_region={'outline_mm':[(-.25,-.18),(.29,-.16),(.25,.18),(-.19,.20)]})
    if condition in {'contact','tear'}:
        scene['contact']=dict(dx_mm=0.,dy_mm=.35,radius_mm=.30)
    if condition.startswith('fringes'):
        gap=.0005 if condition=='fringes_close' else .003
        scene['fringes']=dict(dx_mm=0.,dy_mm=.1,gap_mm=gap,rings=int(2*gap/.00055)+1)
    evidence=contact_front_evidence(render_optical(scene))
    if condition=='contact':
        assert evidence.has_contact,evidence
        assert evidence.failure_reason=='boundary_clipped'
        assert evidence.angular_coverage>=.45
        assert evidence.film_intact is None
    elif condition=='tear':
        assert not evidence.has_contact or evidence.film_intact is False,evidence
    else:
        assert not evidence.has_contact,evidence
