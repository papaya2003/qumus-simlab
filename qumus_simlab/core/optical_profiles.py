"""Immutable public synthetic optical laws shared by renderer and calibration.

These describe simulated image formation only, never measured laboratory optics.
No world, device, controller, renderer state or hidden geometry is imported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RGB = tuple[int, int, int]

def _lerp(a: RGB, b: RGB, t: float) -> RGB:
    t = max(0.0, min(1.0, t))
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))

@dataclass(frozen=True)
class ContactPhotometry:
    """Declared synthetic contact colours used by the renderer.

    These public RGB/opacity laws model transmitted contrast, not contact or
    mechanics. They cannot authorize a physical microscope/film calibration.
    """

    calibration_id: str
    source: str
    front_color: tuple[int, int, int]
    front_opacity: float
    interior_color: tuple[int, int, int]
    interior_opacity: float

SYNTHETIC_CONTACT_PHOTOMETRY = ContactPhotometry(
    "synthetic-contact-photometry-v1", "synthetic", (56, 44, 90), .55,
    (234, 230, 246), .28)

@dataclass(frozen=True)
class SyntheticHBNSlab:
    """Declared illustrative hBN contribution to a transmitted stamp image.

    The visible-index fit is from DOI10.1002/pssb.201800417. A lossless
    free-standing slab at normal incidence, monochromatic illumination and
    0.333nm layer spacing are assumptions, not this microscope's calibration.
    The result weights the existing qualitative tint; it is not an absorption
    measurement or a full PDMS/PC/substrate transfer-matrix calculation.
    """

    calibration_id: str = "synthetic-hbn-freestanding-slab-v1"
    source: str = "synthetic"
    wavelength_um: float = .55
    layer_spacing_um: float = .000333
    sellmeier_a: float = 3.263
    sellmeier_pole_um: float = .1644

    def projection_weight(self, layers: int) -> float:
        n = math.sqrt(1 + self.sellmeier_a*self.wavelength_um**2 /
                      (self.wavelength_um**2-self.sellmeier_pole_um**2))
        surface = ((n-1)/(n+1))**2
        phase = 2*math.pi*n*max(0, layers)*self.layer_spacing_um/self.wavelength_um
        numerator = 4*surface*math.sin(phase)**2
        return numerator / ((1-surface)**2+numerator)

SYNTHETIC_HBN_SLAB = SyntheticHBNSlab()

_TINTS: dict[str, list[tuple[float, RGB]]] = {
    "Gr": [(1, (138, 114, 186)), (2, (132, 118, 194)), (3, (122, 122, 200)),
           (6, (100, 120, 206)), (12, (112, 154, 214)), (25, (170, 194, 216)),
           (60, (222, 210, 172)), (200, (240, 232, 200))],
    "hBN": [(1, (126, 102, 174)), (30, (120, 118, 198)), (60, (104, 144, 208)),
            (90, (122, 178, 190)), (120, (178, 188, 128)), (240, (218, 200, 112))],
    "*": [(1, (112, 108, 164)), (10, (142, 142, 172)), (50, (212, 210, 198))],
}

def flake_tint(material: str, layers: int) -> RGB:
    """The colour a flake of `material` with `layers` layers shows on the
    substrate (see _TINTS)."""
    table = _TINTS.get(material, _TINTS["*"])
    if layers <= table[0][0]:
        return table[0][1]
    for (l0, c0), (l1, c1) in zip(table, table[1:], strict=False):
        if layers <= l1:
            return _lerp(c0, c1, (layers - l0) / (l1 - l0))
    return table[-1][1]

STAMP_PSF_PROFILE = "synthetic-stamp-plane-psf-v1"
CONTACT_INTERIOR_PROFILE = "synthetic-contact-interior-affine-v1"
CAMERA_RESPONSE_PROFILE = "synthetic-rgb8-achromatic-vignette-v1"
HBN_LABEL_NM_PER_LAYER = .33


# The v1 search microscope camera, measured from v1's own photographs (#182).
# scripts/measure_v1_optics.py reproduces every number from the read-only v1
# copy (corner/CORNER_*.jpg: this camera at 5x, 3840x2160; and the 20x hBN
# test photos agents/processing_agent/segmenter/test_images/hBN, Canon 6240x4160).
#  - Vignetting is radial. Canon 20x frames: 23 % darker at the left/right edge
#    (kx 0.22-0.24, ky 0.10-0.13 = kx*(H/W)^2). This camera sees only ~0.39 of
#    that field width (0.70 mm vs ~1.8 mm at 20x), so 0.23*0.39^2 = 0.035; its
#    own out-of-focus off-chip background fits 0.04-0.05. 0.04 is used, i.e.
#    4 % at the left/right edges. The synthetic camera used 14 % per axis.
#  - Substrate texture: 20x background brightness after removing the smooth
#    field has sd 0.9-2.6 DN (median 1.7) at ~1 um, drawn as achromatic
#    band-limited noise fixed to the chip (it moves with the stage and blurs
#    with defocus, as a real surface does); its amplitude is set below.
#  - Exfoliation debris: tiny fragments of the exfoliated crystal. Median
#    counts of 0.5-1, 1-2 and 2-3 um particles are 345, 107 and 71 per mm2
#    (Canon pitch taken as 5.67 um / 20). Debris larger than 2.5 um is not
#    drawn: at 20x it would exceed v1's 200 px minimum candidate area and be
#    read as a (false) flake candidate.
V1_SEARCH_CAMERA_PROFILE = "v1-search-camera-measured-v1"
#  - Pixel noise and texture together (#207): on v1's own JPEGs (9 hBN
#    segmenter test photos, CORNER_BL/BR/TL) bare SiO2 has a median 16x16 px
#    brightness sd of 0.87-1.07 DN and a fine-grain (minus a 2 px Gaussian)
#    sd of 0.45-0.77 DN. With 2 DN texture and +-3 DN pixel noise the sim was
#    2.4 / 2.1 DN after JPEG q95, and the JPEG blocks of that noise read as
#    monolayer Gr on blank fields. +-1 DN pixel noise and 1 DN texture give
#    0.94 / 0.66 DN.
V1_SEARCH_CAMERA = {
    "vignette_kx": .04, "vignette_ky": .04*(2160/3840)**2,
    "sensor_noise_dn": 1, "texture_sd_dn": 1.0, "texture_octaves_um": (1., 2., 4., 8., 16., 32.),
    # (smallest um, largest um, count per mm2)
    "debris_bins": ((.5, 1., 345.), (1., 2., 107.), (2., 2.5, 35.)),
}


def camera_vignette_axes(width: int, height: int, profile: str | None = None):
    """The camera's separable gain, shared by forward and inverse use.

    `profile` V1_SEARCH_CAMERA_PROFILE selects the measured v1 search camera;
    anything else keeps the synthetic camera (stamp/contact optics).
    """
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Camera raster dimensions must be positive integers")
    kx, ky = ((V1_SEARCH_CAMERA["vignette_kx"], V1_SEARCH_CAMERA["vignette_ky"])
              if profile == V1_SEARCH_CAMERA_PROFILE else (.14, .14))
    return ([1.-kx*((x-width/2)/(width/2))**2 for x in range(width)],
            [1.-ky*((y-height/2)/(height/2))**2 for y in range(height)])


def native_substrate_support(rgb, noise_scale):
    """Necessary pixel compatibility with the native v1 search camera law.

    This does not prove substrate identity or exclude optically indistinguishable
    material. Unknown conditions are refused. No flake/material annotation,
    renderer seed, geometry, desired result or random draw is consumed.
    """
    import numpy as np

    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Native substrate support needs an original RGB8 image")
    if type(noise_scale) not in (int, float) or not math.isfinite(noise_scale) or not 0 <= noise_scale <= 1:
        raise ValueError("Native material camera noise condition is unsupported")
    height, width = pixels.shape[:2]
    fx, fy = camera_vignette_axes(width, height, V1_SEARCH_CAMERA_PROFILE)
    fx, fy = np.asarray(fx), np.asarray(fy)
    # Substrate = vignetted substrate colour + achromatic surface texture
    # (bounded at 4 sd) + achromatic sensor noise. Truncation can split an
    # achromatic offset by one count between channels.
    bound = 4*V1_SEARCH_CAMERA["texture_sd_dn"] + 1 + abs(round(V1_SEARCH_CAMERA["sensor_noise_dn"]*noise_scale))
    support = np.zeros((height, width), dtype=bool)
    for top in range(0, height, 64):
        bottom = min(top+64, height)
        gain = fy[top:bottom, None]*fx[None, :]
        expected = np.trunc(gain[:, :, None]*np.asarray(V1_NATIVE_SUBSTRATE_RGB)).astype(np.int16)
        residual = pixels[top:bottom].astype(np.int16)-expected
        support[top:bottom] = ((residual.max(axis=2)-residual.min(axis=2) <= 1)
            & (np.abs(residual).max(axis=2) <= bound)
            & (pixels[top:bottom].min(axis=2) > 0) & (pixels[top:bottom].max(axis=2) < 255))
    return support

# Existing finite chip raster support, unchanged from MicroscopeRenderer.
SYNTHETIC_SUBSTRATE_RGB = (122, 96, 168)
SYNTHETIC_OFF_CHIP_RGB = (24, 22, 28)
CHIP_MAX_DOWNSAMPLE = 7
CHIP_EDGE_ILLUMINATION_MM = .004

# A fresh, opt-in material-camera profile. The donor camera's empirical LAB
# references define forward colour differences; the classifier tables are not
# fitted to these synthetic images. No contact, stamp or white-light law changes.
V1_MATERIAL_CAPTURE_PROFILE = "v1-reference-material-capture-v2"
# Native material-only camera: frozen hBN donor substrate LAB (49,27.5,8.5)
# converted once to RGB8. Checked against v1's real 20x photos (#182): the
# frame-centre background median is (159, 96, 106), within 5 counts per
# channel. Legacy/stamp optics retain SYNTHETIC_SUBSTRATE_RGB.
V1_NATIVE_SUBSTRATE_RGB = (164, 98, 103)
# v1's search camera raster: corner/CORNER_*.jpg, taken by v1's corner
# detection through this camera (cam_wait_focus_photo), are 3840x2160, as is
# the stacking target TARGET_PT=(1920,1080). AutoSearch's CENTER_X=1780,
# CENTER_Y=1180 is its calibrated optical centre, not the frame centre.
# Pitches are AutoSearch's PIXEL_SIZE_UM_MAP; accurate refind's separate
# .183um20X remains a historical calibration discrepancy.
V1_MATERIAL_SENSOR_PX = (3840, 2160)
V1_MATERIAL_PITCH_UM = {10: .365, 20: .182, 100: .036}
# v1 stacking views go through the same UVCHAM camera at 5X (turret position
# 1; TARGET_PT (1920, 1080) in molecule_drop_down/pick_up/adjust_touching_point):
# twice the 10x pitch, a 2.80 x 1.58 mm field (#197, #205). Analysis frames are
# that field at a quarter of the sensor; display frames use the whole sensor.
V1_STACKING_OBJECTIVE = 5
V1_STACKING_PITCH_UM = 2*V1_MATERIAL_PITCH_UM[10]
V1_STACKING_ANALYSIS_PX = (960, 540)
V1_MATERIAL_REFERENCE_SOURCE = {
    "source": "frozen_v1_empirical_lab_references",
    "material_py_sha256": "08fb82921e4cd50e59b2d9b0d46da8783aed251e194c9f123e29e627869c989c",
    "scope": "Synthetic forward contrast from historical references; not measured camera calibration",
}


def v1_material_reference(material: str, layers: int, objective: int):
    """One unchanged donor LAB anchor and its substrate reference.

    Layer classes are the existing material vocabulary. hBN uses .33 nm/layer;
    no exact thickness is inferred by perception from a rendered class colour.
    Multiple empirical references per class remain in the recognition tables.
    """
    high = objective >= 50
    if material == "Gr":
        background = (49.62793933, 12.17574073, 5.43972594)
        # LiveSegment's native20 centered-revisit recovery explicitly uses the
        # high-mag thin-Gr colour calibration (donor livesegment.py:271–277).
        thin_revisit = high or objective == 20
        if layers <= 1:
            reference = (46., 16.2, .4) if thin_revisit else (46.6, 11., 4.2)
        elif layers == 2:
            reference = (43.9, 20., -1.2) if thin_revisit else (43.9, 18.9, 1.6)
        elif layers == 3:
            reference = (40.4, 23.6, -.6)
        elif layers < 10:
            reference = (30., 20., -20.)
        else:
            # The paired run-373 many-layer plate is explicitly documented in
            # the donor, rather than choosing a new synthetic optical colour.
            reference = (24.4, 42.2, -41.5) if high else (34.4, 25.2, -19.2)
        return reference, background
    if material == "hBN":
        nm = layers * HBN_LABEL_NM_PER_LAYER
        background = (49., 27.5, 8.5)
        if nm < 10:
            reference = (50., 20., -2.)
        elif nm < 20:
            reference = (48., 15., -9.)
        elif nm < 30:
            reference = (56., -16., -15.) if high else (70.7, -31., 24.2)
        elif nm < 40:
            reference = (81., -41., 1.) if high else (90.7, -45., 20.)
        else:
            reference = (80., -57., 65.) if high else (77., -17., 42.)
        return reference, background
    return None


# Conditional SIM release model. This is not a physical calibration or an
# overall confidence/transfer guarantee. No realized geometry is declared.
RELEASE_SUPPORT_PROFILE = "synthetic-observed-convex-release-support-v1"
STAMP_RECOGNITION_PROFILE = "synthetic-complete-outline-centroid-gaussian-v1"
RELEASE_SUPPORT_ASSUMPTIONS = {
    "id": RELEASE_SUPPORT_PROFILE, "source": "synthetic", "simulation_only": True,
    "support_topology": "single-simultaneous-filled-convex-contact",
    "recognition": STAMP_RECOGNITION_PROFILE, "centroid_sigma_mm_per_noise_scale": .002,
    "alpha_centroid_only": .001, "Kmax": 2, "N": 1,
    "collar_mm": .1752, "front_cell_erosion_mm": .05963091755805144,
    "psf_support_px": 12, "opaque_raster_px": 1, "reference_stencil_radius_px": 2,
    "optical_whole_interior_observed": False, "hidden_integrity": "unknown",
    "scope": "Conditional synthetic recognition/support only; not metrology or transfer proof",
}
