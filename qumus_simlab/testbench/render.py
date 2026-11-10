"""One exposure path for simulated microscope sensing and live/replay pictures.

- BenchRenderer draws the overview camera's top-down view of the sim world
  (regions, chips, carriers, tape, nozzles) as a PNG.
- MicroscopeRenderer either replays a real microscope photo (v1
  segmenter/test_images, picked by material; spec 11.2) or draws a synthetic
  micrograph of the flakes in view: SiO2/Si substrate, interference tints by
  layer count, crystal-shaped flakes with terraces, tape residue, the chip
  edge, the stamp's contact front, defocus from the focus error, a scale bar.
  Synthetic pictures are projected from SimWorld. They share the same encoded
  exposure used by raster autofocus, registration and
  contact control. The synthetic MicroscopeAnalyzer separately reports noisy
  metadata observations; it does not supply those pixel-controller results.

Images are written under the run-data directory (<run-directory>/images/), which
is git-ignored and served by the UI's /api/image. The renderer uses the shared
synthetic photometry calibration; pixel analysis uses the core Pillow dependency.
"""

from __future__ import annotations

import itertools
import math
import random
import shutil
import struct
import threading
import zlib
from array import array
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageMath

from qumus_simlab.core.optical_profiles import (
    CAMERA_RESPONSE_PROFILE,
    HBN_LABEL_NM_PER_LAYER,
    STAMP_PSF_PROFILE,
    SYNTHETIC_CONTACT_PHOTOMETRY,
    SYNTHETIC_HBN_SLAB,
    SYNTHETIC_SUBSTRATE_RGB,
    V1_MATERIAL_CAPTURE_PROFILE,
    V1_MATERIAL_PITCH_UM,
    V1_MATERIAL_REFERENCE_SOURCE,
    V1_MATERIAL_SENSOR_PX,
    V1_NATIVE_SUBSTRATE_RGB,
    V1_SEARCH_CAMERA,
    V1_SEARCH_CAMERA_PROFILE,
    V1_STACKING_ANALYSIS_PX,
    V1_STACKING_OBJECTIVE,
    V1_STACKING_PITCH_UM,
    camera_vignette_axes,
    flake_tint,
    v1_material_reference,
)
from qumus_simlab.core.optical_profiles import (
    ContactPhotometry as ContactPhotometry,
)
from qumus_simlab.core.optical_profiles import (
    SyntheticHBNSlab as SyntheticHBNSlab,
)
from qumus_simlab.flows.bench import regions
from qumus_simlab.testbench.optics import (
    STAMP_STRUCTURE_PROFILE,
    WHITE_LIGHT_MAX_GAP_MM,
    WHITE_LIGHT_PROFILE,
    StampStructure,
    white_light_table,
)

RGB = tuple[int, int, int]









#: v1 saves its search-camera photos with cv2.imwrite(".jpg"), whose default
#: quality is 95 (4:2:0 chroma); Pillow's libjpeg gives the same bytes.
JPEG_QUALITY = 95
#: zlib level for the lossless frames. Level 6 cost ~2 s per 4K frame in pure
#: Python for ~7 % smaller files; level 1 is several times faster (#203).
PNG_ZLIB_LEVEL = 1


def encode_jpeg(width: int, height: int, rgb: bytearray, quality: int = JPEG_QUALITY) -> bytes:
    """RGB8 pixels -> JPEG bytes at v1's quality (search photos, #203)."""
    import io

    out = io.BytesIO()
    pixels = np.frombuffer(bytes(rgb), dtype=np.uint8).reshape(height, width, 3)
    Image.fromarray(pixels).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def image_suffix(data: bytes) -> str:
    """File suffix of encoded frame bytes (JPEG or PNG)."""
    return ".jpg" if data[:2] == b"\xff\xd8" else ".png"


def encode_png(width: int, height: int, rgb: bytearray, level: int = PNG_ZLIB_LEVEL) -> bytes:
    """RGB8 pixels (row-major, 3 bytes per pixel) -> PNG bytes."""
    stride = width * 3
    raw = b"".join(b"\x00" + bytes(rgb[y * stride:(y + 1) * stride]) for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, level))
            + chunk(b"IEND", b""))


@lru_cache(maxsize=256)
def _blend_tables(color: RGB, alpha: float) -> tuple[bytes, bytes, bytes]:
    """The scalar RGB8 blend, including its original per-pass truncation."""
    return tuple(bytes(int(value * (1 - alpha) + channel * alpha)
                       for value in range(256)) for channel in color)


def _film_disc(c: Canvas, cx: float, cy: float, r: float, *, fill, fill_alpha: float, edge,
               edge_alpha: float, edge_px: float = 1.5, torn: bool = False) -> None:
    """A thin film disc: a faint fill and an edge line, blended in place.

    A torn edge wanders by a few percent of the radius, deterministically.
    """
    if r <= 0:
        return
    reach = r*1.06+edge_px+1
    x0, x1 = max(0, math.floor(cx-reach)), min(c.w, math.ceil(cx+reach))
    y0, y1 = max(0, math.floor(cy-reach)), min(c.h, math.ceil(cy+reach))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx, dy = xs+.5-cx, ys+.5-cy
    d = np.hypot(dx, dy)
    edge_r = np.full(d.shape, r)
    if torn:
        theta = np.arctan2(dy, dx)
        edge_r = r*(1+.025*np.sin(3*theta+.7)+.015*np.sin(7*theta+2.1)+.01*np.sin(13*theta+.3))
    image = np.frombuffer(c.px, dtype=np.uint8).reshape(c.h, c.w, 3)
    region = image[y0:y1, x0:x1].astype(np.float64)
    inside = d < edge_r
    region[inside] = region[inside]*(1-fill_alpha)+np.asarray(fill, np.float64)*fill_alpha
    line = np.clip(1-np.abs(d-edge_r)/edge_px, 0, 1)*edge_alpha
    region = region*(1-line[..., None])+np.asarray(edge, np.float64)*line[..., None]
    image[y0:y1, x0:x1] = np.clip(np.rint(region), 0, 255).astype(np.uint8)


class Canvas:
    def __init__(self, width: int, height: int, bg: RGB) -> None:
        self.w, self.h = width, height
        self.px = bytearray(bytes(bg) * (width * height))

    def set(self, x: int, y: int, c: RGB) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.px[i:i + 3] = bytes(c)

    def blend(self, x: int, y: int, c: RGB, a: float) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            for k in range(3):
                self.px[i + k] = int(self.px[i + k] * (1 - a) + c[k] * a)

    def polygon(self, pts: list[tuple[float, float]], c: RGB, alpha: float = 1.0) -> None:
        """Scanline fill of a convex or simple polygon (pixel coordinates)."""
        if not pts:
            return
        y0 = max(0, int(math.floor(min(p[1] for p in pts))))
        y1 = min(self.h - 1, int(math.ceil(max(p[1] for p in pts))))
        n = len(pts)
        for y in range(y0, y1 + 1):
            yc = y + 0.5
            xs = []
            for i in range(n):
                (ax, ay), (bx, by) = pts[i], pts[(i + 1) % n]
                if (ay <= yc < by) or (by <= yc < ay):
                    xs.append(ax + (yc - ay) * (bx - ax) / (by - ay))
            xs.sort()
            for xa, xb in zip(xs[::2], xs[1::2], strict=False):
                left, right = max(0, int(xa + 0.5)), min(self.w, int(xb + 0.5))
                if (type(self) is Canvas and right > left and alpha >= 1.0
                        and len(c) == 3 and all(type(value) is int and 0 <= value <= 255 for value in c)):
                    start, stop = (y * self.w + left) * 3, (y * self.w + right) * 3
                    self.px[start:stop] = bytes(c) * (right-left)
                    continue
                # Float premultiplied planes and other subclasses retain their
                # own set/blend methods. Reuse the exact scanline coverage.
                if (type(self) is Canvas and right > left and len(c) == 3
                        and type(alpha) in (int, float) and 0 <= alpha < 1
                        and all(type(value) is int and 0 <= value <= 255 for value in c)):
                    start, stop = (y * self.w + left) * 3, (y * self.w + right) * 3
                    for channel, table in enumerate(_blend_tables(tuple(c), alpha)):
                        self.px[start+channel:stop:3] = self.px[start+channel:stop:3].translate(table)
                    continue
                for x in range(left, right):
                    if alpha >= 1.0:
                        self.set(x, y, c)
                    else:
                        self.blend(x, y, c, alpha)

    def rect(self, x0: float, y0: float, x1: float, y1: float, c: RGB, alpha: float = 1.0) -> None:
        self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], c, alpha)

    def outline(self, x0: float, y0: float, x1: float, y1: float, c: RGB) -> None:
        for x in range(int(x0), int(x1) + 1):
            self.set(x, int(y0), c)
            self.set(x, int(y1), c)
        for y in range(int(y0), int(y1) + 1):
            self.set(int(x0), y, c)
            self.set(int(x1), y, c)

    def disc(self, cx: float, cy: float, r: float, c: RGB, alpha: float = 1.0,
             *, segments: int = 16) -> None:
        pts = [(cx + r * math.cos(t * 2 * math.pi / segments),
                cy + r * math.sin(t * 2 * math.pi / segments)) for t in range(segments)]
        self.polygon(pts, c, alpha)

    def text(self, x: float, y: float, s: str, c: RGB, size: int = 1) -> None:
        """Upper-case label with its top-left corner at (x, y); unknown
        characters are left blank."""
        for k, ch in enumerate(s.upper()):
            glyph = _FONT.get(ch, _FONT[" "])
            for row, bits in enumerate(glyph):
                for col, b in enumerate(bits):
                    if b == "#":
                        for dx in range(size):
                            for dy in range(size):
                                self.set(int(x) + (k * 6 + col) * size + dx,
                                         int(y) + row * size + dy, c)

    def png(self) -> bytes:
        return encode_png(self.w, self.h, self.px)

    def jpeg(self, quality: int = JPEG_QUALITY) -> bytes:
        return encode_jpeg(self.w, self.h, self.px, quality)


#: JPEG quality of display-only microscope frames (#205)
DISPLAY_JPEG_QUALITY = 90


# Fixed native-pixel synthetic carried-layer PSF, not microscope calibration.
STAMP_PSF_PADDING_PX = 14


def _stamp_psf_kernel(sharpness, px_scale=1.0):
    """`px_scale`: pixels per synthetic 480 px/2.6 mm pixel (same physical PSF)."""
    q = max(0.0, min(1.0, (sharpness - 0.5) / 8.0))
    sigma = math.sqrt(9 * (1 - q))*px_scale
    if sigma == 0:
        return sigma, [1.0]
    radius = math.ceil(4 * sigma)
    raw = [math.exp(-x * x / (2 * sigma * sigma)) for x in range(-radius, radius + 1)]
    total = math.fsum(raw)
    return sigma, [x / total for x in raw]


class _PremultipliedStampPlane(Canvas):
    """Use the SAME native production polygon scanlines; composite in float."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.channels = [array("d", [0.0]) * (width * height) for _ in range(4)]

    def set(self, x, y, color):
        self.blend(x, y, color, 1.0)

    def blend(self, x, y, color, alpha):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = y * self.w + x
            for k in range(3):
                self.channels[k][i] = self.channels[k][i] * (1 - alpha) + color[k] * alpha
            self.channels[3][i] = self.channels[3][i] * (1 - alpha) + alpha

    def images(self):
        return [Image.fromarray(np.frombuffer(values, dtype=np.float64).astype(np.float32)
                                .reshape(self.h, self.w), mode="F") for values in self.channels]


def _convolve_stamp_plane(plane, weights):
    """Transparent extension; explicit identical finite kernel in each axis."""
    if len(weights) == 1:
        return plane.copy()
    if plane.size[0]*plane.size[1] > _CAMERA_CACHE_MAX_PIXELS:
        # v1-camera frames: the same kernel and zero extension in OpenCV
        # float32, like _gaussian_defocus (#205; may differ in the last bit).
        import cv2

        kernel = np.asarray(weights, dtype=np.float32)
        result = cv2.sepFilter2D(np.asarray(plane, dtype=np.float32), -1, kernel, kernel,
                                 borderType=cv2.BORDER_CONSTANT)
        return Image.fromarray(result, mode="F")
    radius = len(weights) // 2
    result = plane
    for dimension in (0, 1):
        accumulation = Image.new("F", plane.size, 0.0)
        for offset, weight in enumerate(weights):
            delta = offset - radius
            shifted = Image.new("F", plane.size, 0.0)
            shifted.paste(result, (delta if dimension == 0 else 0, delta if dimension == 1 else 0))
            accumulation = ImageMath.lambda_eval(
                lambda a, weight=weight: a["sum"] + a["term"] * weight,
                sum=accumulation,
                term=shifted,
            )
        result = accumulation
    return result


def _stamp_plane(meta, width, height, sx, sy, pad=STAMP_PSF_PADDING_PX):
    plane = _PremultipliedStampPlane(width + 2 * pad, height + 2 * pad)

    def at(dx, dy):
        return width / 2 + pad + dx * sx, height / 2 + pad - dy * sy

    # Top first stored order; lower first projection, matching production.
    for flake in reversed(meta.get("stamp_flakes", [])):
        alpha = 1.0
        if flake.get("material") == "hBN" and flake.get("layers") is not None:
            alpha = SYNTHETIC_HBN_SLAB.projection_weight(int(flake["layers"]))
        MicroscopeRenderer._flake(plane, flake, at, sx, sy, None, alpha=alpha, grow=1.0)
    return plane


#: recent rasters kept by MicroscopeRenderer.render_bytes (a few MB at most
#: for analysis tiles; the descent repeats only the latest frame)
_MEMO_FRAMES = 4
#: frame fields that never reach the raster: when and from where it was taken
_NON_PIXEL_FIELDS = frozenset({"captured_at", "pose_binding", "replayed_from", "timestamp",
                               "stage_kinematics", "sample_pose", "best_holder_z",
                               "best_stamp_focus_z", "geometry_source", "mechanics_version",
                               "scene"})


#: scene fields drawn above the chip plane (the stamp and what it carries)
_ABOVE_CHIP_FIELDS = frozenset({"stamp", "stamp_flakes", "contact", "optical_gap", "fringes",
                                "stamp_structure", "stamp_texture_seed", "sharpness_stamp",
                                "sharpness", "focus_plane", "purpose", "verification", "immutable",
                                "source", "best_holder_z", "best_stamp_focus_z"})


def _chip_plane_key(meta: dict[str, Any], size: tuple[int, int], sigma: float) -> str:
    import hashlib
    import json

    data = {k: v for k, v in meta.items() if k not in _NON_PIXEL_FIELDS and k not in _ABOVE_CHIP_FIELDS}
    blob = json.dumps([data, list(size), sigma], sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class _SmallCache:
    """A few recent large chip planes (each about 25 MB at 3840x2160)."""

    def __init__(self, size: int) -> None:
        from collections import OrderedDict

        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            value = self.items.get(key)
            if value is not None:
                self.items.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        with self.lock:
            self.items[key] = value
            while len(self.items) > self.size:
                self.items.popitem(last=False)


_CHIP_PLANES = _SmallCache(2)


def _pixel_key(meta: dict[str, Any]) -> str:
    """Digest of everything that can change the microscope raster.

    Fields not listed in _NON_PIXEL_FIELDS are all kept, so a new scene field
    only lowers the hit rate, never reuses a wrong picture. White-light rings
    beyond WHITE_LIGHT_MAX_GAP_MM draw nothing, so their gap is dropped there.
    """
    import hashlib
    import json

    data = {k: v for k, v in meta.items() if k not in _NON_PIXEL_FIELDS}
    gap = data.get("optical_gap")
    if (isinstance(gap, dict) and not gap.get("has_contact")
            and isinstance(gap.get("signed_apex_gap_mm"), (int, float))
            and gap["signed_apex_gap_mm"] >= WHITE_LIGHT_MAX_GAP_MM):
        data["optical_gap"] = {"beyond_rings": True, "profile_id": gap.get("profile_id")}
    blob = json.dumps(data, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


#: a resumed lab starts numbering after every picture of the runs it continues
#: (simlab/server.py sets it from the save point's image roots): a reused name
#: would make the brain keep an older picture under the same name
NUMBER_FLOOR = 0


def highest_picture_number(roots: Any) -> int:
    """Largest <prefix>_<n> number of any picture under these folders."""
    best = 0
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for p in root.rglob("*_*.*"):
            tail = p.stem.rsplit("_", 1)[-1]
            if tail.isdigit() and p.is_file():
                best = max(best, int(tail))
    return best


class _Writer:
    """Numbered image files in one directory, thread-safe."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # Restarting a lab must not overwrite photos referenced by earlier
        # sessions in the same store.
        last = max((int(p.stem.rsplit('_', 1)[-1]) for p in self.dir.iterdir()
                    if p.is_file() and p.stem.rsplit('_', 1)[-1].isdigit()), default=0)
        self._n = itertools.count(max(last, NUMBER_FLOOR) + 1)
        self._lock = threading.Lock()
        self._last: dict[str, tuple[bytes, str]] = {}  # prefix -> (digest, numbered path)

    def write(self, prefix: str, data: bytes, suffix: str | None = None,
              fixed: bool = False) -> str:
        """`fixed`: one file per prefix, replaced each time (a live preview).

        Without a suffix, the encoded bytes choose .jpg or .png.
        """
        if suffix is None:
            suffix = image_suffix(data)
        if fixed:
            path = self.dir / f"{prefix}_live{suffix}"
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)  # readers never see half a file
            return str(path.resolve())
        import hashlib

        digest = hashlib.sha256(data).digest()
        with self._lock:
            # Numbered files are never rewritten, so a frame identical to the
            # previous one of this camera (an unchanged descent step) shares
            # its file instead of writing a copy.
            last = self._last.get(prefix + suffix)
            if last is not None and last[0] == digest and Path(last[1]).is_file():
                return last[1]
            n = next(self._n)
        path = self.dir / f"{prefix}_{n:06d}{suffix}"
        path.write_bytes(data)
        resolved = str(path.resolve())
        with self._lock:
            self._last[prefix + suffix] = (digest, resolved)
        return resolved


# ----- overview camera -------------------------------------------------------

#: fill colour per station kind / named zone
REGION_COLOR: dict[str, RGB] = {"storage": (40, 92, 160),          # blue storage boards
                                "exfoliation": (214, 226, 214),
                                "microscope": (205, 214, 226),
                                "reactor": (226, 214, 206),
                                "stamp_handoff": (220, 228, 232)}
#: printed above each station so a reader of the photo knows which is which
REGION_LABEL = {"left_storage": "LEFT STORAGE", "middle_storage": "MIDDLE STORAGE",
                "right_storage": "RIGHT STORAGE", "exfoliation_stage": "EXFOLIATION",
                "transfer_stage": "SEARCH AND STACKING", "reactor": "REACTOR"}
TABLE: RGB = (236, 236, 232)
TEXT: RGB = (90, 90, 96)
SLOT: RGB = (20, 20, 22)          # black frames on the boards
QR_DARK: RGB = (10, 10, 10)
QR_LIGHT: RGB = (250, 250, 250)

# 5x7 bitmap font for labels (rows top to bottom, "#" = ink)
_FONT: dict[str, tuple[str, ...]] = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    " ": (".....",) * 7,
}




def _lerp(a: RGB, b: RGB, t: float) -> RGB:
    t = max(0.0, min(1.0, t))
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def _shade(c: RGB, k: float) -> RGB:
    return (max(0, min(255, int(c[0] * k))), max(0, min(255, int(c[1] * k))),
            max(0, min(255, int(c[2] * k))))


class BenchRenderer:
    """Top-down drawing of the whole table in the world frame, as the overview
    camera sees it: the optical breadboard with its hole grid, the storage
    boards with their QR corners and black slot frames, the stations, the
    exfoliator gantry with its two rolls, the arms with their links (world +y
    is up in the picture). `scale` px per mm. Flat colours, no camera noise:
    tests read pixels off it."""

    def __init__(self, bench: dict[str, Any], out_dir: Path, scale: float = 1.0) -> None:
        self.bench, self.scale = bench, scale
        t = bench.get("table") or {"x": [0, 1260], "y": [0, 630]}
        self.x0, self.y0, self.x1, self.y1 = t["x"][0], t["y"][0], t["x"][1], t["y"][1]
        self.writer = _Writer(out_dir)

    def _p(self, x: float, y: float) -> tuple[float, float]:
        # image y grows downwards; world y upwards
        return (x - self.x0) * self.scale, (self.y1 - y) * self.scale

    def _box(self, r: dict[str, Any]) -> tuple[float, float, float, float]:
        (ax, ay), (bx, by) = self._p(r["x"][0], r["y"][1]), self._p(r["x"][1], r["y"][0])
        return ax, ay, bx, by

    def _plate(self, c: Canvas, box: tuple[float, float, float, float], col: RGB,
               edge: RGB = (150, 150, 146)) -> None:
        """A plate standing a little above the table: a soft shadow down-right,
        a lighter top edge, a darker outline."""
        ax, ay, bx, by = box
        s = 2 * self.scale
        c.rect(ax + s, ay + s, bx + s, by + s, (0, 0, 0), 0.18)
        c.rect(ax, ay, bx, by, col)
        c.outline(ax, ay, bx - 1, by - 1, edge)
        for x in range(int(ax) + 1, int(bx) - 1):
            c.blend(x, int(ay) + 1, (255, 255, 255), 0.35)

    def _background(self, c: Canvas) -> None:
        b = self.bench
        # M6 hole grid of the breadboard, 25 mm pitch, 12.5 mm from the edges
        pitch, r = 25.0, 1.4 * self.scale
        x = self.x0 + pitch / 2
        while x < self.x1:
            y = self.y0 + pitch / 2
            while y < self.y1:
                px, py = self._p(x, y)
                c.disc(px, py, r, (176, 178, 174))
                c.disc(px, py, r * 0.6, (120, 122, 118))
                y += pitch
            x += pitch
        for st in b.get("stations", {}).values():
            if (mod := st.get("module")) is not None:  # the machine's frame around its parts
                ax, ay, bx, by = self._box(mod)
                c.outline(ax, ay, bx - 1, by - 1, (120, 120, 116))
        for name, st in b.get("stations", {}).items():
            self._plate(c, self._box(st["region"]),
                        REGION_COLOR.get(st.get("kind", ""), (225, 225, 225)))
            if name in REGION_LABEL:
                ax, ay, bx, by = self._box(st.get("module") or st["region"])
                y = ay - 10 if ay >= 10 else by + 3
                c.text(ax, y, REGION_LABEL[name], TEXT)
        if (mc := b.get("microscope", {}).get("center")) is not None and \
                "module" in b.get("stations", {}).get("transfer_stage", {}):
            # the fixed microscope seen from above: its body over the field of view
            px, py = self._p(mc["x"], mc["y"])
            c.disc(px + 2, py + 2, 26 * self.scale, (0, 0, 0), 0.12)
            c.disc(px, py, 26 * self.scale, (90, 92, 98), 0.35)
        regs = regions(b)
        for name in ("microscope", "stamp_handoff"):
            if (r_ := regs.get(name)) is not None:
                ax, ay, bx, by = self._box(r_)
                c.outline(ax, ay, bx - 1, by - 1, (150, 120, 150))
        self._exfoliator(c)
        stations = b.get("stations", {})
        for sl in b.get("slots", {}).values():
            if stations.get(sl.get("station"), {}).get("placeholder"):
                continue  # nothing built there yet (the reactor)
            w, h = sl.get("size", [10, 10])
            ax, ay, bx, by = self._box({"x": [sl["x"] - w / 2 - 1, sl["x"] + w / 2 + 1],
                                        "y": [sl["y"] - h / 2 - 1, sl["y"] + h / 2 + 1]})
            c.outline(ax, ay, bx, by, SLOT)
        q = 3.0  # half size of a QR marker (mm)
        for corners in b.get("qr_markers", {}).values():
            for x, y in corners:
                ax, ay, bx, by = self._box({"x": [x - q, x + q], "y": [y - q, y + q]})
                c.rect(ax, ay, bx, by, QR_LIGHT)
                c.rect(ax + 1, ay + 1, (ax + bx) / 2, (ay + by) / 2, QR_DARK)
                c.rect((ax + bx) / 2, (ay + by) / 2, bx - 1, by - 1, QR_DARK)
        for arm in ("arm_left", "arm_right", "arm_reactor"):
            base = b.get(arm, {}).get("base")
            if base is None:
                continue
            if b[arm].get("planned", False):
                continue  # not built yet: the camera sees nothing there
            px, py = self._p(base["x"], base["y"])
            rr = 13 * self.scale
            c.disc(px + 2, py + 2, rr, (0, 0, 0), 0.2)
            c.disc(px, py, rr, (120, 70, 64))
            c.disc(px, py, rr * 0.55, (170, 100, 90))

    def _exfoliator(self, c: Canvas) -> None:
        """The slide rail along the back of the module, the fixed roll (at the
        origin) hanging from its slider; the movable roll is drawn by _tape."""
        ex = self.bench.get("exfoliator", {})
        o = ex.get("origin")
        if o is None:
            return
        if (reg := ex.get("region")) is not None:  # older layouts: the gantry footprint
            ax, ay, bx, by = self._box(reg)
            c.outline(ax, ay, bx - 1, by - 1, (120, 120, 116))
        roll = ex.get("roller", {})
        ln, dia = roll.get("length_mm", 34), roll.get("diameter_mm", 12)
        trav = ex.get("travel", {}).get("x", [-150, 60])
        rail = ex.get("rail") or {"x": [o["x"] + trav[0], o["x"] + trav[1]], "y": o["y"]}
        (rx0, ry0), (rx1, _) = self._p(rail["x"][0], rail["y"]), self._p(rail["x"][1], rail["y"])
        c.rect(rx0, ry0 - 2 * self.scale, rx1, ry0 + 2 * self.scale, (96, 96, 100))
        if "rail" in ex:  # the fixed roll's slider on the rail and its hanger
            sx, sy = self._p(o["x"], rail["y"])
            c.rect(sx - 1 * self.scale, sy, sx + 1 * self.scale,
                   self._p(o["x"], o["y"] + ln / 2)[1], (110, 110, 112))
            c.rect(sx - 8 * self.scale, sy - 5 * self.scale, sx + 8 * self.scale,
                   sy + 5 * self.scale, (226, 226, 222))
            c.outline(sx - 8 * self.scale, sy - 5 * self.scale, sx + 8 * self.scale,
                      sy + 5 * self.scale, (110, 110, 112))
        # fixed roll: a cylinder across the tape path at the origin
        fx, fy = self._p(o["x"], o["y"])
        c.rect(fx - dia / 2 * self.scale, fy - ln / 2 * self.scale,
               fx + dia / 2 * self.scale, fy + ln / 2 * self.scale, (210, 212, 214))
        c.outline(fx - dia / 2 * self.scale, fy - ln / 2 * self.scale,
                  fx + dia / 2 * self.scale, fy + ln / 2 * self.scale, (110, 110, 112))

    def _slide(self, c: Canvas, pts: list[tuple[float, float]], glass: RGB) -> None:
        """A glass slide: a soft shadow, the glass, a bright rim."""
        s = 1.5 * self.scale
        c.polygon([(x + s, y + s) for x, y in pts], (0, 0, 0), 0.22)
        c.polygon(pts, glass)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=False):
            n = max(2, int(math.hypot(x1 - x0, y1 - y0)))
            for i in range(n):
                t = i / n
                c.blend(int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t), (255, 255, 255), 0.5)

    def render(self, device_id: str, objects: list[dict[str, Any]], tape: dict[str, Any],
               arms: dict[str, dict[str, Any]], fixed: bool = False) -> str:
        """`fixed`: overwrite one live file instead of numbering (UI preview)."""
        w = int((self.x1 - self.x0) * self.scale)
        h = int((self.y1 - self.y0) * self.scale)
        c = Canvas(w, h, TABLE)
        self._background(c)
        for o in objects:
            if o.get("held_by") and o["held_by"] != "exfoliator":
                continue
            ow, oh = o.get("size_mm", [10, 10])
            r = math.radians(o["rotation"])
            corners = [(-ow / 2, -oh / 2), (ow / 2, -oh / 2), (ow / 2, oh / 2), (-ow / 2, oh / 2)]
            pts = [self._p(o["x"] + a * math.cos(r) - b * math.sin(r),
                           o["y"] + a * math.sin(r) + b * math.cos(r)) for a, b in corners]
            seed = sum(map(ord, o["id"]))
            if o["kind"] == "carrier":
                self._slide(c, pts, (214, 222, 228))
                # the crystal under its tape: a dark, uneven patch in the middle
                cx, cy = self._p(o["x"], o["y"])
                rng = random.Random(seed)
                for _ in range(5):
                    c.disc(cx + rng.uniform(-3, 3) * self.scale, cy + rng.uniform(-6, 6) * self.scale,
                           rng.uniform(2, 4) * self.scale, (70, 56, 46), 0.8)
                c.polygon(pts, (196, 170, 120), 0.35)  # the tape over the slide
            elif o["kind"] == "stamp":
                self._slide(c, pts, (204, 216, 224))
                # the PDMS dome sits 7 mm from the left end of a slide lying along x
                # (bench.yaml stamp_slide; the near end of an upright one), as on the page
                ins = float(self.bench.get("stamp_slide", {}).get("dome_inset_mm", 7))
                a, b = (-ow / 2 + ins, 0.0) if ow > oh else (0.0, -oh / 2 + ins)
                cx, cy = self._p(o["x"] + a * math.cos(r) - b * math.sin(r),
                                 o["y"] + a * math.sin(r) + b * math.cos(r))
                c.disc(cx, cy, 2.2 * self.scale, (150, 172, 186), 0.9)  # PDMS dome
                c.disc(cx - 0.6 * self.scale, cy - 0.6 * self.scale, 0.7 * self.scale,
                       (240, 246, 250), 0.9)
            else:
                s = 1.2 * self.scale
                c.polygon([(x + s, y + s) for x, y in pts], (0, 0, 0), 0.25)
                col: RGB = (124, 96, 170) if o.get("n_flakes") else (166, 170, 184)
                c.polygon(pts, col)
                c.polygon(pts, (255, 255, 255), 0.12)
                if o.get("n_flakes"):  # glitter of thick graphite left by the tape
                    rng = random.Random(seed)
                    cx, cy = self._p(o["x"], o["y"])
                    for _ in range(min(12, 3 + int(o["n_flakes"]) // 2)):
                        c.set(int(cx + rng.uniform(-ow / 2.5, ow / 2.5) * self.scale),
                              int(cy + rng.uniform(-oh / 2.5, oh / 2.5) * self.scale), (228, 222, 200))
        self._tape(c, tape)
        for name, arm in arms.items():
            self._arm(c, name, arm)
        return self.writer.write(device_id, c.png(), fixed=fixed)

    def _tape(self, c: Canvas, tape: dict[str, Any]) -> None:
        ex = self.bench.get("exfoliator", {})
        st = self.bench["stations"]["exfoliation_stage"]["region"]
        ax, ay, bx, by = self._box(st)
        o = ex.get("origin")
        g = tape.get("gantry") or {}
        if o is not None and g.get("x") is not None:
            # the movable roll on its cross positioner, and the tape band
            # between the two rolls (over the stage when applied)
            roll = ex.get("roller", {})
            ln, dia = roll.get("length_mm", 34), roll.get("diameter_mm", 12)
            mx, my = self._p(o["x"] + g["x"], o["y"])
            fx, fy = self._p(o["x"], o["y"])
            if tape.get("applied") or tape.get("laid"):
                tw = ex.get("tape_width_mm", 25) / 2 * self.scale
                x0, x1 = min(mx, fx), max(mx, fx)
                c.rect(x0, fy - tw, x1, fy + tw, (250, 240, 190), 0.6)
                if tape.get("material"):
                    c.disc((ax + bx) / 2, (ay + by) / 2, 3 * self.scale, (70, 70, 70), 0.6)
            if (rail := ex.get("rail")) is not None:  # its slider on the rail and the hanger
                _, sy = self._p(0, rail["y"])
                c.rect(mx - 1 * self.scale, sy, mx + 1 * self.scale, my - ln / 2 * self.scale,
                       (110, 110, 112))
                c.rect(mx - 8 * self.scale, sy - 5 * self.scale, mx + 8 * self.scale,
                       sy + 5 * self.scale, (236, 236, 232))
                c.outline(mx - 8 * self.scale, sy - 5 * self.scale, mx + 8 * self.scale,
                          sy + 5 * self.scale, (110, 110, 112))
            c.rect(mx - dia / 2 * self.scale + 2, my - ln / 2 * self.scale + 2,
                   mx + dia / 2 * self.scale + 2, my + ln / 2 * self.scale + 2, (0, 0, 0), 0.2)
            c.rect(mx - dia / 2 * self.scale, my - ln / 2 * self.scale,
                   mx + dia / 2 * self.scale, my + ln / 2 * self.scale, (236, 236, 232))
            c.outline(mx - dia / 2 * self.scale, my - ln / 2 * self.scale,
                      mx + dia / 2 * self.scale, my + ln / 2 * self.scale, (110, 110, 112))
        elif tape.get("applied") or tape.get("material"):
            c.rect(ax, ay, bx, by, (250, 236, 170), 0.55 if tape.get("applied") else 0.25)
            if tape.get("material"):
                c.disc((ax + bx) / 2, (ay + by) / 2, 3 * self.scale, (70, 70, 70), 0.6)

    def _arm(self, c: Canvas, name: str, arm: dict[str, Any]) -> None:
        """Base to nozzle as one link (the sim arm has no elbow), the nozzle a
        disc that grows and darkens as it comes down, red when its vacuum is on."""
        px, py = self._p(arm["x"], arm["y"])
        base = self.bench.get(name, {}).get("base")
        low = arm["z"] < 20
        if base is not None:
            bx, by = self._p(base["x"], base["y"])
            n = max(2, int(math.hypot(px - bx, py - by)))
            for i in range(n):
                t = i / n
                x, y = bx + (px - bx) * t, by + (py - by) * t
                c.disc(x + 2, y + 2, 2.5 * self.scale, (0, 0, 0), 0.08)
                c.disc(x, y, 2.5 * self.scale, (110, 112, 116), 0.9)
        rr = (5 if low else 7) * self.scale
        c.disc(px + 3, py + 3, rr, (0, 0, 0), 0.25 if low else 0.12)
        c.disc(px, py, rr, (210, 80, 60) if arm.get("vacuum") else (80, 82, 86),
               0.95 if low else 0.5)
        c.disc(px, py, rr * 0.4, (240, 240, 240), 0.6 if low else 0.3)


# ----- microscope ------------------------------------------------------------

#: 285 nm SiO2 on Si under white light: the violet every flake contrasts with
SUBSTRATE: RGB = SYNTHETIC_SUBSTRATE_RGB
OFF_CHIP: RGB = (24, 22, 28)     # beyond the chip edge: the dark stage



#: How much of a 2D layer's colour change on SiO2/Si it shows on a metal
#: electrode (#185). Metal reflects strongly and has no oxide interference,
#: so a few-nm to tens-of-nm layer changes it only slightly. Illustrative.
METAL_LAYER_CONTRAST = .25


def layer_opacity(material: str, layers: int) -> float:
    """Illustrative opacity of one 2D-material layer (not a measured value).

    Monolayer graphene about 0.25, rising slowly with layers; hBN about 0.25
    plus 0.5 per 100 nm (20-30 nm about 0.37). Capped at 0.9.
    """
    if material == "Gr":
        return min(.9, .25 + .04 * max(0, layers - 1))
    if material == "hBN":
        return min(.9, .25 + .005 * layers * HBN_LABEL_NM_PER_LAYER)
    return .5


def flake_layer(shown, substrate, opacity: float):
    """(opacity, layer colour) whose normal blend over `substrate` gives `shown`.

    The opacity is raised just enough that the layer colour stays in 0..255.
    """
    import numpy as np

    shown, substrate = np.asarray(shown, dtype=np.float64), np.asarray(substrate, dtype=np.float64)
    diff = shown - substrate
    room = np.where(diff > 0, 255 - substrate, substrate)
    need = float(np.max(np.where(room > 0, np.abs(diff) / np.maximum(room, 1e-9), 0)))
    alpha = min(1.0, max(opacity, need))
    return alpha, substrate + diff / alpha
MIC_SIZE: tuple[int, int] = (480, 320)

#: interference tints on 285 nm SiO2 by layer count (rough, from the usual
#: contrast charts and v1's photos): monolayer graphene is a shade lighter than
#: the substrate, few layers go blue, graphite turns pale and then yellowish;
#: hBN goes blue, cyan, green and yellow as it thickens (0.33 nm per layer)




def _crystal(rng: random.Random, cx: float, cy: float, rx: float, ry: float,
             n: int = 6) -> list[tuple[float, float]]:
    """A flake outline: a polygon whose edges mostly follow the crystal's 60
    degree directions, as cleaved graphite and hBN do. Its area is about pi
    rx ry."""
    rot = rng.uniform(0, math.pi / 3)
    pts = []
    for k in range(n):
        t = rot + 2 * math.pi * k / n + rng.uniform(-0.22, 0.22)
        r = rng.uniform(0.75, 1.3)
        pts.append((cx + rx * r * math.cos(t), cy + ry * r * math.sin(t)))
    return pts


def _scaled(pts: list[tuple[float, float]], k: float, ox: float = 0.0,
            oy: float = 0.0) -> list[tuple[float, float]]:
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [(cx + (x - cx) * k + ox, cy + (y - cy) * k + oy) for x, y in pts]


#: Gaussian defocus sigma at blur 1 (px): the former 7x downsample (~2.8 px).
DEFOCUS_SIGMA_PX = 2.8
#: Below about half a pixel, RGB8 rounding makes the raster focus metric rise
#: and fall by several percent, so sub-half-pixel defocus is not rendered.
#: Onset (sharpness ~8.7) matches the former first 2x downsample (~8.6).
DEFOCUS_MIN_SIGMA_PX = .5


def _gaussian_defocus(px: bytearray, w: int, h: int, sigma: float) -> bytes:
    """Exact separable Gaussian (edge-replicated), continuous in sigma.

    Sampled Gaussians compose, so a larger sigma never restores gradient
    energy; box-filter approximations do not have that property.
    """
    if w*h > _CAMERA_CACHE_MAX_PIXELS:
        # Native multi-megapixel frames: the same sampled kernel and edge mode
        # in OpenCV float32 (#182: ~10x faster; may differ by one count where
        # a value rounds at exactly .5).
        import cv2

        radius = max(1, math.ceil(4*sigma))
        kernel = np.exp(-.5*(np.arange(-radius, radius+1, dtype=np.float64)/sigma)**2)
        kernel = (kernel/kernel.sum()).astype(np.float32)
        image = np.frombuffer(bytes(px), dtype=np.uint8).reshape(h, w, 3).astype(np.float32)
        blurred = cv2.sepFilter2D(image, -1, kernel, kernel, borderType=cv2.BORDER_REPLICATE)
        return np.clip(np.rint(blurred), 0, 255).astype(np.uint8).tobytes()
    image = np.frombuffer(bytes(px), dtype=np.uint8).reshape(h, w, 3).astype(np.float64)
    return np.clip(np.rint(_separable_gaussian(image, sigma)), 0, 255).astype(np.uint8).tobytes()


def _separable_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Sampled Gaussian over rows then columns, edge-replicated, radius ceil(4 sigma).

    SciPy's correlate1d (mode "nearest") computes the same kernel sum in C; the
    NumPy loop is the dependency-free fallback with identical arithmetic.
    """
    radius = max(1, math.ceil(4*sigma))
    offsets = np.arange(-radius, radius+1, dtype=np.float64)
    kernel = np.exp(-.5*(offsets/sigma)**2)
    kernel /= kernel.sum()
    try:
        from scipy.ndimage import correlate1d
    except ImportError:
        correlate1d = None
    for axis in (0, 1):
        if correlate1d is not None:
            image = correlate1d(image, kernel, axis=axis, mode="nearest")
            continue
        padded = np.pad(image, [(radius, radius) if a == axis else (0, 0) for a in range(3)], mode="edge")
        size = image.shape[axis]
        image = sum(weight*np.take(padded, np.arange(i, i+size), axis=axis)
                    for i, weight in enumerate(kernel))
    return image


def _defocus_change(before: bytes, after: bytearray, w: int, h: int, sigma: float) -> bytes:
    """Spread only what a chip-plane overlay changed, with the chip sigma."""
    if w*h > _CAMERA_CACHE_MAX_PIXELS:
        # v1-camera frames (#205): blur only the changed box (plus the
        # kernel reach) in OpenCV float32; zero change outside stays zero.
        import cv2

        old = np.frombuffer(before, dtype=np.uint8).reshape(h, w, 3)
        new = np.frombuffer(bytes(after), dtype=np.uint8).reshape(h, w, 3)
        rows, cols = np.nonzero((old != new).any(axis=2))
        if not len(rows):
            return bytes(after)
        radius = max(1, math.ceil(4*sigma))
        y0, y1 = max(0, rows.min()-radius), min(h, rows.max()+radius+1)
        x0, x1 = max(0, cols.min()-radius), min(w, cols.max()+radius+1)
        kernel = np.exp(-.5*(np.arange(-radius, radius+1, dtype=np.float64)/sigma)**2)
        kernel = (kernel/kernel.sum()).astype(np.float32)
        change = new[y0:y1, x0:x1].astype(np.float32)-old[y0:y1, x0:x1]
        spread = cv2.sepFilter2D(change, -1, kernel, kernel, borderType=cv2.BORDER_REPLICATE)
        out = old.copy()
        out[y0:y1, x0:x1] = np.clip(np.rint(old[y0:y1, x0:x1]+spread), 0, 255).astype(np.uint8)
        return out.tobytes()
    base = np.frombuffer(before, dtype=np.uint8).reshape(h, w, 3).astype(np.float64)
    change = np.frombuffer(bytes(after), dtype=np.uint8).reshape(h, w, 3).astype(np.float64)-base
    return np.clip(np.rint(base+_separable_gaussian(change, sigma)), 0, 255).astype(np.uint8).tobytes()


_CAMERA_CACHE_MAX_PIXELS = 640 * 424


@lru_cache(maxsize=1)
def _camera_affine_block():
    """Exact successive LCG coefficients; no new noise distribution or draws."""
    multiplier, offset, multipliers, offsets = 1, 0, [], []
    for _ in range(4096):
        multiplier = (1103515245*multiplier) & 0x7FFFFFFF
        offset = (1103515245*offset+12345) & 0x7FFFFFFF
        multipliers.append(multiplier)
        offsets.append(offset)
    return np.asarray(multipliers, dtype=np.uint64), np.asarray(offsets, dtype=np.uint64)


@lru_cache(maxsize=2)
def _camera_gain(width: int, height: int, profile: str | None = None):
    """Per-pixel vignette gain fx[x]*fy[y] in raster order (read-only, cached)."""
    fx, fy = (np.asarray(axis) for axis in camera_vignette_axes(width, height, profile))
    gain = (fx[None, :]*fy[:, None]).ravel()
    gain.flags.writeable = False
    return gain


def _camera_native(c: Canvas, seed: int, noise_scale: float, profile: str | None = None) -> None:
    """Bounded chunks of the unchanged scalar camera for large frames."""
    width, height = c.w, c.h
    gain = _camera_gain(width, height, profile)
    # |noise| <= 3*|noise_scale|, and gain <= 1 keeps the scaled pixel in 0..255:
    # 16-bit integers hold every sum exactly when the noise is that small.
    small = np.int16 if abs(noise_scale) <= 1000 else np.int64
    multipliers, offsets = _camera_affine_block()
    state = seed & 0xFFFFFFFF or 1
    pixels = np.frombuffer(c.px, dtype=np.uint8).reshape(-1, 3)
    for start in range(0, width*height, len(multipliers)):
        stop = min(start+len(multipliers), width*height)
        count = stop-start
        states = (multipliers[:count]*state+offsets[:count]) & 0x7FFFFFFF
        state = int(states[-1])
        noise = np.rint((((states >> 16) % 7).astype(np.int16)-3)*noise_scale).astype(small)
        pixels[start:stop] = _camera_rgb8(pixels[start:stop], gain[start:stop], noise)


@lru_cache(maxsize=8)
def _camera_fields(width: int, height: int, seed: int, noise_scale: float, profile: str | None = None):
    """Cache bounded native fields using the original scalar order and RNG."""
    fx, fy = camera_vignette_axes(width, height, profile)
    gain = np.fromiter((fx[x] * fy[y] for y in range(height) for x in range(width)),
                       dtype=np.float64, count=width * height)
    noise = np.empty(width * height, dtype=np.int64)
    state = seed & 0xFFFFFFFF or 1
    for index in range(width * height):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        noise[index] = round(((state >> 16) % 7 - 3) * noise_scale)
    gain.flags.writeable = noise.flags.writeable = False
    return gain, noise


def _camera_rgb8(rgb, gain, noise):
    # Separate operations preserve scalar multiply -> int -> add -> clamp.
    # There is no fused multiply-add, new random draw, or interpolation.
    values = np.trunc(rgb.astype(np.float64) * gain[:, None]).astype(
        np.int16 if noise.dtype == np.int16 else np.int64)
    values += noise[:, None]
    np.clip(values, 0, 255, out=values)
    return values.astype(np.uint8)


def _camera(c: Canvas, seed: int, noise_scale: float = 1.0, profile: str | None = None) -> None:
    """What the camera adds: the illumination falls off towards the corners,
    and every pixel carries a little noise. `profile` V1_SEARCH_CAMERA_PROFILE
    is the measured v1 search camera; the default is the synthetic camera."""
    w, h, px = c.w, c.h, c.px
    if profile == V1_SEARCH_CAMERA_PROFILE and isinstance(noise_scale, int | float):
        # The shared law draws -3..3 DN; the v1 search camera's measured pixel
        # noise is +-sensor_noise_dn.
        noise_scale = noise_scale*V1_SEARCH_CAMERA["sensor_noise_dn"]/3
    # The vectorised path draws the same LCG sequence as the scalar loop (byte
    # equivalent). Every view now has its own noise seed, so a per-seed field
    # cache would only miss; frames of any size take this path.
    if (type(c) is Canvas and w*h > 0
            and isinstance(noise_scale, int | float) and math.isfinite(noise_scale)
            and abs(noise_scale) <= 2**40):
        _camera_native(c, seed, noise_scale, profile)
        return
    fx, fy = camera_vignette_axes(w, h, profile)
    s = seed & 0xFFFFFFFF or 1
    i = 0
    for y in range(h):
        gy = fy[y]
        for x in range(w):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            n = round(((s >> 16) % 7 - 3) * noise_scale)
            g = fx[x] * gy
            px[i] = max(0, min(255, int(px[i] * g) + n))
            px[i + 1] = max(0, min(255, int(px[i + 1] * g) + n))
            px[i + 2] = max(0, min(255, int(px[i + 2] * g) + n))
            i += 3



def _exposure_noise_seed(seed: int, meta: dict[str, Any]) -> int:
    """Camera-noise seed of one view: the scene seed mixed with where the
    camera looks on the sample (view centre, sample and chip pose).

    Noise is temporal on a real sensor. Seeding it from the scene alone gave
    every exposure of a chip the same pixel pattern, fixed in image space, so
    image matching (v1's ORB corner refind) locked onto the noise at 0 degrees.
    The same state still renders the same bytes; any stage move or chip
    rotation draws a new realisation with unchanged statistics.
    """
    import zlib

    texture = meta.get("specimen_texture") or {}
    parts = [seed]
    for source in (texture.get("view_center_world_mm"), texture.get("pose"), meta.get("chip_pose"),
                   meta.get("sample_pose")):
        if isinstance(source, dict):
            source = [source.get(k) for k in ("x", "y", "rotation")]
        if isinstance(source, list | tuple):
            parts += [round(float(v), 4) if isinstance(v, int | float) and math.isfinite(v) else None
                      for v in source]
    # A packet that says nothing about where the camera looks keeps the scene seed.
    return seed if len(parts) == 1 else zlib.crc32(repr(parts).encode()) & 0x7FFFFFFF

def _lattice_noise(ix, iy, key: int):
    """Uniform values in [-1, 1) fixed to integer lattice points (splitmix64)."""
    h = (ix.astype(np.uint64)*np.uint64(0x9E3779B97F4A7C15)
         ^ iy.astype(np.uint64)*np.uint64(0xC2B2AE3D27D4EB4F) ^ np.uint64(key))
    for shift, mult in ((33, 0xFF51AFD7ED558CCD), (33, 0xC4CEB9FE1A85EC53)):
        h ^= h >> np.uint64(shift)
        h *= np.uint64(mult)
    h ^= h >> np.uint64(33)
    return (h >> np.uint64(11)).astype(np.float64)/2.**52-1.


#: Values of bilinearly interpolated unit-variance lattice noise, summed over
#: octaves, have about this sd; dividing by it gives the declared texture sd.
_TEXTURE_NORMALISER = .62
_DEBRIS_CELL_MM = .1
_DEBRIS_LAYERS = {"hBN": (40, 75, 105, 150, 220, 300), "Gr": (12, 20, 30, 45, 60)}
#: Length of one chipped piece of a hand-cut chip edge (#204).
_CUT_EDGE_SEGMENT_MM = .004


def _cut_edge_hash(key: int, side: int, index: int) -> int:
    """A fixed 16-bit value per edge piece (splitmix64 finaliser)."""
    h = (key*0x9E3779B97F4A7C15 + side*0xC2B2AE3D27D4EB4F + index*0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF
    for shift, mult in ((33, 0xFF51AFD7ED558CCD), (33, 0xC4CEB9FE1A85EC53)):
        h ^= h >> shift
        h = (h*mult) & 0xFFFFFFFFFFFFFFFF
    return (h ^ (h >> 33)) & 0xFFFF


def _surface_texture(lay: Canvas, texture: dict[str, Any], fw: float, fh: float,
                     objective: int, substrate: RGB) -> None:
    """The v1 search camera's view of bare SiO2 (#182): faint surface texture
    and tiny exfoliation debris, both fixed to the chip (V1_SEARCH_CAMERA).

    Everything is generated in chip-local coordinates from the chip's seed, so
    overlapping frames, objectives and the focus preview show the same
    surface. Drawn on the chip layer, so defocus blurs it like the flakes.
    """
    import zlib

    import cv2

    w, h = lay.w, lay.h
    sx, sy = w/fw, h/fh
    pose = texture["pose"]
    shift = texture.get("apparent_shift_xy_mm", (0, 0))
    cx, cy = (value-offset for value, offset in zip(texture["view_center_world_mm"], shift, strict=True))
    rot = math.radians(pose["rotation"])
    c, s = math.cos(rot), math.sin(rot)
    hw, hh = (value/2 for value in texture["size_mm"])
    ox, oy = cx-pose["x"], cy-pose["y"]
    # pixel (px, py) -> chip-local mm, an affine map
    cols = np.arange(w, dtype=np.float64)-w/2
    rows = np.arange(h, dtype=np.float64)-h/2
    lx = (c*ox+s*oy) + (c/sx)*cols[None, :] - (s/sy)*rows[:, None]
    ly = (-s*ox+c*oy) - (s/sx)*cols[None, :] - (c/sy)*rows[:, None]
    on_chip = (np.abs(lx) <= hw) & (np.abs(ly) <= hh)
    if not on_chip.any():
        return
    image = np.frombuffer(lay.px, dtype=np.uint8).reshape(h, w, 3)
    key = zlib.crc32(str(texture["seed"]).encode())
    camera = V1_SEARCH_CAMERA
    pixel_um = 1000/min(sx, sy)
    octaves = [p/1000 for p in camera["texture_octaves_um"] if p >= pixel_um]
    if octaves:
        p0 = octaves[0]
        i0, i1 = math.floor(lx.min()/p0)-1, math.ceil(lx.max()/p0)+2
        j0, j1 = math.floor(ly.min()/p0)-1, math.ceil(ly.max()/p0)+2
        gx = np.arange(i0, i1, dtype=np.float64)*p0
        gy = np.arange(j0, j1, dtype=np.float64)*p0
        field = np.zeros((len(gy), len(gx)))
        for k, pitch in enumerate(octaves):
            u, v = gx/pitch, gy/pitch
            iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
            tu, tv = (u-iu)[None, :], (v-iv)[:, None]
            okey = (key+0x632BE5AB*(k+1)) & 0xFFFFFFFFFFFF
            # Each lattice point's value once, then gathered for the four
            # corners of every grid cell (a pure function of the integer point,
            # so identical to evaluating it per corner, at a fraction of the cost)
            lattice = _lattice_noise(np.arange(iu[0], iu[-1]+2, dtype=np.int64)[None, :],
                                     np.arange(iv[0], iv[-1]+2, dtype=np.int64)[:, None], okey)
            xi, yi = iu-iu[0], iv-iv[0]
            rows = [lattice.take(yi+b, axis=0) for b in (0, 1)]
            corners = [rows[b].take(xi+a, axis=1) for b in (0, 1) for a in (0, 1)]
            field += ((corners[0]*(1-tu)+corners[1]*tu)*(1-tv)
                      + (corners[2]*(1-tu)+corners[3]*tu)*tv)
        field *= math.sqrt(3)*camera["texture_sd_dn"]/(_TEXTURE_NORMALISER*math.sqrt(len(octaves)))
        np.clip(field, -4*camera["texture_sd_dn"], 4*camera["texture_sd_dn"], out=field)
        values = cv2.remap(field.astype(np.float32), (lx/p0-i0).astype(np.float32),
                           (ly/p0-j0).astype(np.float32), cv2.INTER_LINEAR)
        offset = np.where(on_chip, np.rint(values), 0).astype(np.int16)
        image[:] = np.clip(image.astype(np.int16)+offset[:, :, None], 0, 255).astype(np.uint8)
    material = texture.get("material")
    if material not in _DEBRIS_LAYERS:
        return
    cell = _DEBRIS_CELL_MM
    for ix in range(math.floor(max(-hw, lx.min())/cell), math.floor(min(hw, lx.max())/cell)+1):
        for iy in range(math.floor(max(-hh, ly.min())/cell), math.floor(min(hh, ly.max())/cell)+1):
            rng = np.random.default_rng(zlib.crc32(f"{texture['seed']}:debris:{ix}:{iy}".encode()))
            for d0, d1, density in camera["debris_bins"]:
                for _ in range(int(rng.poisson(density*cell*cell))):
                    x, y = (ix+rng.random())*cell, (iy+rng.random())*cell
                    radius = rng.uniform(d0, d1)/2000
                    layers = int(rng.choice(_DEBRIS_LAYERS[material]))
                    corners = rng.uniform(.6, 1., 5)*radius
                    turn = rng.uniform(0, 2*math.pi)
                    if abs(x) > hw or abs(y) > hh:
                        continue
                    color = _v1_material_tint(material, layers, objective, substrate)
                    # chip-local -> pixel
                    wx, wy = pose["x"]+c*x-s*y-cx, pose["y"]+s*x+c*y-cy
                    px, py = w/2+wx*sx, h/2-wy*sy
                    r_px = radius*sx
                    if not (-r_px <= px < w+r_px and -r_px <= py < h+r_px):
                        continue
                    if r_px < .75:
                        # smaller than a pixel: the pixel shows its area share
                        ix_px, iy_px = int(px), int(py)
                        if 0 <= ix_px < w and 0 <= iy_px < h and on_chip[iy_px, ix_px]:
                            alpha = min(1., math.pi*r_px*r_px)
                            image[iy_px, ix_px] = np.rint(alpha*np.asarray(color)
                                                          + (1-alpha)*image[iy_px, ix_px])
                        continue
                    angles = turn+np.arange(5)*2*math.pi/5
                    points = np.stack([px+corners*sx*np.cos(angles), py+corners*sy*np.sin(angles)], axis=1)
                    cv2.fillPoly(image, [np.rint(points*16).astype(np.int32)], tuple(int(v) for v in color),
                                 lineType=cv2.LINE_8, shift=4)


def microscope_photometry_profile(material_capture_profile: str | None = None) -> dict[str, str]:
    """One pure declaration for renderer production and SIM capture admission.

    A renderer with the v1 material capture profile stamps the extended
    declaration on every frame it takes (search, revisit and 5X stacking).
    """
    profile = {"source": "synthetic",
               "fringes": WHITE_LIGHT_PROFILE,
               "contact": SYNTHETIC_CONTACT_PHOTOMETRY.calibration_id,
               "carried_hbn": SYNTHETIC_HBN_SLAB.calibration_id,
               "raster": STAMP_PSF_PROFILE,
               "chip_surface": "synthetic-chip-footprint-v2",
               "stamp_structure": STAMP_STRUCTURE_PROFILE}
    if material_capture_profile is None:
        return profile
    if material_capture_profile != V1_MATERIAL_CAPTURE_PROFILE:
        raise ValueError("Unknown material capture profile")
    profile["material"] = material_capture_profile
    profile["material_reference_sha256"] = V1_MATERIAL_REFERENCE_SOURCE["material_py_sha256"]
    profile["material_reference_selection"] = "v1-objective-references-revisit20-gr-v3"
    profile["native_material_substrate"] = "v1-hbn-donor-background-rgb8-v1"
    profile["camera_response"] = CAMERA_RESPONSE_PROFILE
    # Native material frames are taken by the measured v1 search camera.
    profile["material_camera_response"] = V1_SEARCH_CAMERA_PROFILE
    # 5X stamp/contact views: v1 camera, substrate and surface; the
    # white-light rings, contact front and PC film keep the synthetic
    # formulas with inferred parameters, not measured from v1 images.
    profile["stacking_camera"] = V1_SEARCH_CAMERA_PROFILE
    profile["stacking_overlays"] = "inferred-formula-not-measured-v1"
    return profile


def supported_photometry_profile(profile: object) -> bool:
    """Capture admission: either declaration a SIM renderer can produce."""
    return profile in (microscope_photometry_profile(),
                       microscope_photometry_profile(V1_MATERIAL_CAPTURE_PROFILE))


@lru_cache(maxsize=128)
def _v1_material_tint(material: str, layers: int, objective: int,
                      substrate_rgb: RGB = V1_NATIVE_SUBSTRATE_RGB) -> RGB:
    """Render donor contrast onto the same substrate used by this exposure."""
    # Thin Gr at 20X takes v1's revisit-camera colour, which v1 LiveSegment
    # shares between its 20X and 100X captures (livesegment.py:271-273): the
    # 20X centred refind looks for that contrast, and with the 10X colour it
    # found only specks inside true monolayers (seed 8, #207).
    reference = v1_material_reference(material, layers, objective)
    if reference is None:
        return flake_tint(material, layers)
    import cv2

    anchor, substrate = reference
    actual_bg = cv2.cvtColor(np.asarray([[substrate_rgb]], dtype=np.float32)/255.,
                             cv2.COLOR_RGB2LAB)[0, 0]
    target = np.asarray(anchor)-np.asarray(substrate)+actual_bg
    rgb = cv2.cvtColor(target.astype(np.float32)[None, None, :], cv2.COLOR_LAB2RGB)[0, 0]
    return tuple(int(value) for value in np.rint(np.clip(rgb, 0., 1.)*255))


@lru_cache(maxsize=4)
def _paper_coverage(w: int, h: int, sx: float, sy: float, glass, centre, radius: float,
                    shape: str) -> np.ndarray:
    """Antialiased paper coverage (0-255 per pixel); unchanged while the stamp
    only moves in z, so a descent draws it once (#205)."""
    scale = 2
    mask = Image.new("L", (w*scale, h*scale))
    draw = ImageDraw.Draw(mask)

    def project(point):
        return ((w/2+point[0]*sx)*scale, (h/2-point[1]*sy)*scale)

    draw.polygon([project(p) for p in glass], fill=255)
    cx, cy = centre
    if shape == "diamond":
        vertex = math.sqrt(2)*radius  # radius is the declared inradius.
        aperture = [(cx+vertex, cy), (cx, cy+vertex), (cx-vertex, cy), (cx, cy-vertex)]
    else:
        # Preserve immutable v1 packets' original circle raster exactly.
        aperture = [(cx+radius*math.cos(i*math.tau/256), cy+radius*math.sin(i*math.tau/256))
                    for i in range(256)]
    draw.polygon([project(p) for p in aperture], fill=0)
    mask = mask.resize((w, h), Image.Resampling.LANCZOS)
    coverage = np.asarray(mask, dtype=np.float64).reshape(-1)
    coverage.flags.writeable = False
    return coverage


class MicroscopeRenderer:
    """Replays real photos when a replay directory is configured, else draws one."""

    @property
    def photometry_profile(self) -> dict[str, str]:
        """Public assumptions copied into every capture-specific calibration."""
        return microscope_photometry_profile(self.material_capture_profile)

    def __init__(self, out_dir: Path | None, replay_dir: str | Path | None = None,
                 size: tuple[int, int] = MIC_SIZE, *, material_capture_profile: str | None = None) -> None:
        if material_capture_profile not in (None, V1_MATERIAL_CAPTURE_PROFILE):
            raise ValueError("Unknown material capture profile")
        self.writer = _Writer(out_dir) if out_dir is not None else None
        self.cache = Path(out_dir) / "replay" if out_dir is not None else None
        self.replay = self._index(Path(replay_dir)) if replay_dir else {}
        self.size = size
        self.material_capture_profile = material_capture_profile
        from collections import OrderedDict
        self._memo: OrderedDict[str, bytes] = OrderedDict()
        self._memo_lock = threading.Lock()
        self.memo_hits = 0

    def exposure_geometry(self, objective: int, fov_mm, purpose: str = "analysis_capture") -> dict:
        """Actual rerasterization geometry, shared with capture-time calibration.

        Focus and preview use the same optical FOV as the native exposure, at
        their configured smaller raster. Historical/default renderers stay exact.
        """
        native = bool(self.material_capture_profile and objective in V1_MATERIAL_PITCH_UM)
        neighbourhood = bool(self.material_capture_profile and objective == 5
                             and purpose == "neighbourhood_capture")
        # the v1 camera at 5X for stamp/contact views (#205)
        stacking = bool(self.material_capture_profile and objective == V1_STACKING_OBJECTIVE
                        and not neighbourhood)
        if native:
            pitch = V1_MATERIAL_PITCH_UM[objective]/1000
            fov = [axis*pitch for axis in V1_MATERIAL_SENSOR_PX]
        elif stacking:
            fov = [axis*V1_STACKING_PITCH_UM/1000 for axis in V1_MATERIAL_SENSOR_PX]
        else:
            fov = list(fov_mm)
        if neighbourhood:
            # The ordinary 5x field on the full sensor height, square pixels.
            height = V1_MATERIAL_SENSOR_PX[1]
            size = (min(V1_MATERIAL_SENSOR_PX[0], round(height*fov[0]/fov[1])), height)
        elif stacking:
            size = V1_MATERIAL_SENSOR_PX if purpose == "motion_preview" else V1_STACKING_ANALYSIS_PX
        else:
            size = V1_MATERIAL_SENSOR_PX if native and purpose == "analysis_capture" else self.size
        return {"image_size_px": list(size), "fov_mm": fov,
            "mm_per_pixel": [field/pixels for field, pixels in zip(fov, size, strict=True)],
            "material_capture_profile": self.material_capture_profile if native else None,
            "neighbourhood_capture_profile": "full-sensor-5x-neighbourhood-v1" if neighbourhood else None,
            "camera_profile": V1_SEARCH_CAMERA_PROFILE if stacking else None}

    @staticmethod
    def _index(root: Path) -> dict[str, list[tuple[Path, str | None]]]:
        """material code -> [(photo, thickness label or None)]. Folder names as in
        v1 segmenter/test_images; a thickness class in a sub-folder or file name
        (e.g. graphene/monolayer/a.jpg, hbn/b_20-30.png; ">40" is written "gt40")
        lets replay pick a photo of the same thickness as the simulated flake."""
        from qumus_simlab.testbench.locator import GR_LABELS, HBN_LABELS

        names = {"graphene": "Gr", "graphite": "Gr", "gr": "Gr", "hbn": "hBN", "wte2": "WTe2"}
        labels = [(lab.replace(">", "gt"), lab) for lab in GR_LABELS + HBN_LABELS]
        out: dict[str, list[tuple[Path, str | None]]] = {}
        if not root.is_dir():
            return out
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            code = names.get(d.name.lower(), d.name)
            photos = sorted(p for p in d.rglob("*")
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            out[code] = [(p, next((lab for key, lab in labels
                                   if key in str(p.relative_to(d).with_suffix("")).lower()),
                                  None)) for p in photos]
        return out

    def render(self, device_id: str, meta: dict[str, Any]) -> tuple[str, str | None]:
        """Returns (path, replayed_from)."""
        if self.writer is None or self.cache is None:
            raise ValueError("Fileless microscope renderer has no image output directory")
        flakes = meta.get("flakes", [])
        material = flakes[0]["material"] if flakes else None
        photos = self.replay.get(material or "", [])
        if photos and flakes[0].get("layers") is not None:
            from qumus_simlab.testbench.locator import thickness_label

            # same thickness class as the flake if such photos exist; the photo's
            # flake size and shape still differ from the simulated one
            label = thickness_label(material, flakes[0]["layers"])
            photos = [p for p, tag in photos if tag == label] or [p for p, _ in photos]
            key = flakes[0]["sim_id"]
            src = photos[sum(map(ord, key)) % len(photos)]  # same flake -> same photo
            self.cache.mkdir(parents=True, exist_ok=True)
            dst = self.cache / f"{material}_{src.name}"
            if not dst.exists():
                shutil.copyfile(src, dst)
            return str(dst.resolve()), str(src)
        return self._draw(device_id, meta), None

    def save_bytes(self, device_id: str, data: bytes) -> str:
        """Persist one already captured PNG without rendering the scene again."""
        if self.writer is None:
            raise ValueError("Fileless microscope renderer has no image output directory")
        return self.writer.write(device_id, data, suffix=".png")

    def render_preview(self, device_id: str, meta: dict[str, Any], *,
                       immutable: bool = False) -> str:
        """Project a scene without substituting reference photographs.

        Live polling replaces one ephemeral image. Event producers request an
        immutable snapshot so later fields cannot overwrite replay's pixels.
        Rendering uses only the supplied scene and private deterministic noise;
        it never captures, samples physics or consumes observation randomness.
        """
        return self._draw(device_id, meta, fixed=not immutable)

    # -- drawing --------------------------------------------------------------

    def _draw(self, device_id: str, meta: dict[str, Any], *, fixed: bool = False) -> str:
        if self.writer is None:
            raise ValueError("Fileless microscope renderer has no image output directory")
        data = self.render_bytes(device_id, meta)
        # Fixed writes share a temporary name across concurrent UI requests.
        if fixed:
            with self.writer._lock:
                return self.writer.write(device_id, data, fixed=True)
        return self.writer.write(device_id, data)

    def render_bytes(self, device_id: str, meta: dict[str, Any]) -> bytes:
        """Project exposure-frozen scene metadata to PNG with no filesystem access.

        Memory-only captures, saved evidence and live previews use exactly this
        raster implementation. This method has no device/world/RNG side effects.

        The raster is a pure function of its pixel inputs, so a frame whose
        inputs equal a recent one is returned from a small memo, byte for byte.
        During a stamp descent most frames are such repeats; the picture only
        changes (and is drawn again) once rings, contact or focus change it.
        """
        key = _pixel_key(meta)
        with self._memo_lock:
            data = self._memo.get(key)
            if data is not None:
                self._memo.move_to_end(key)
                self.memo_hits += 1
                return data
        data = self._render_bytes(meta)
        with self._memo_lock:
            self._memo[key] = data
            while len(self._memo) > _MEMO_FRAMES:
                self._memo.popitem(last=False)
        return data

    def _render_bytes(self, meta: dict[str, Any]) -> bytes:
        calibration = meta.get("calibration") or {}
        selected = calibration.get("material_capture_profile")
        neighbourhood = calibration.get("neighbourhood_capture_profile")
        if selected is not None or neighbourhood is not None or calibration.get("camera_profile") is not None:
            if selected is not None and selected != self.material_capture_profile:
                raise ValueError("Material capture/renderer profile differs")
            geometry = self.exposure_geometry(meta["objective"], meta["fov_mm"], meta.get("purpose", "analysis_capture"))
            if (calibration.get("image_size_px") != geometry["image_size_px"]
                    or calibration.get("mm_per_pixel") != geometry["mm_per_pixel"]
                    or list(meta["fov_mm"]) != geometry["fov_mm"]
                    or neighbourhood != geometry.get("neighbourhood_capture_profile")
                    or calibration.get("camera_profile") != geometry.get("camera_profile")):
                raise ValueError("Material capture calibration differs from actual raster/FOV")
            return self._compose(meta, size=tuple(geometry["image_size_px"]))
        return self._compose(meta, size=self.size)

    @classmethod
    def _compose(cls, meta: dict[str, Any], *, size: tuple[int, int]) -> bytes:
        w, h = size
        fw, fh = meta.get("fov_mm", [1.3, 0.86])
        flakes = meta.get("flakes", [])
        corners = meta.get("chip_corners", [])
        on_chip = meta.get("on_chip")
        sharp = float(meta.get("sharpness", 0.5))
        sharp_chip = float(meta.get("sharpness_chip", sharp))
        sharp_stamp = float(meta.get("sharpness_stamp", 0.5))
        seen = on_chip is not None or bool(corners) or bool(flakes)
        seed = (sum(map(ord, str(on_chip))) * 131
                + sum(sum(map(ord, f["sim_id"])) for f in flakes)) & 0x7FFFFFFF
        texture = meta.get("specimen_texture")
        footprint = cls._specimen_outline(texture) if texture is not None else None
        if texture is not None:
            seed = sum((i+1)*ord(char) for i, char in enumerate(texture["seed"])) & 0x7FFFFFFF
            seen = cls._footprint_in_view(footprint, fw, fh)
        # The chip layer is drawn sharp at full size, then defocused by a
        # Gaussian whose sigma grows with |z - focus| (sharpness 10.5 at best
        # focus, 0.5 far from it). The former integer 1/k downsample let a
        # nearer plane score below a farther one, splitting one focus plane
        # into two raster peaks (#133).
        calibration = meta.get("calibration") or {}
        native_material = calibration.get("material_capture_profile") == V1_MATERIAL_CAPTURE_PROFILE
        # v1 camera at 5X for stamp/contact views (#205): the measured camera,
        # substrate and surface; the synthetic stamp optics keep their
        # physical sizes (their pixel constants were set on 480 px over 2.6 mm).
        stacking = calibration.get("camera_profile") == V1_SEARCH_CAMERA_PROFILE
        v1_camera = native_material or stacking
        px_scale = (w/fw)/(MIC_SIZE[0]/2.6) if stacking else 1.0
        blur = max(0.0, min(1.0, 1.0 - (sharp_chip - 0.5) / 10.0))
        sigma = DEFOCUS_SIGMA_PX * px_scale * blur if seen else 0.0
        if sigma < DEFOCUS_MIN_SIGMA_PX:
            sigma = 0.0
        substrate = V1_NATIVE_SUBSTRATE_RGB if v1_camera else SUBSTRATE
        lay = Canvas(w, h, substrate if seen else OFF_CHIP)
        sx, sy = lay.w / fw, lay.h / fh  # px per mm in the layer
        rng = random.Random(seed)

        def at(dx: float, dy: float) -> tuple[float, float]:
            return lay.w / 2 + dx * sx, lay.h / 2 - dy * sy

        c = Canvas(w, h, OFF_CHIP)
        # A stamp moving in z leaves the (defocused) chip plane unchanged; a
        # v1-camera frame reuses it instead of drawing it again (#205).
        chip_key = _chip_plane_key(meta, size, sigma) if w*h > _CAMERA_CACHE_MAX_PIXELS else None
        cached = _CHIP_PLANES.get(chip_key) if chip_key else None
        if cached is not None:
            c.px[:] = cached[0]
            rng.setstate(cached[1])
        else:
            if footprint is not None:
                edge_key = (zlib.crc32(f"{texture['seed']}:cut-edge".encode())
                            if v1_camera and texture is not None else None)
                inside = (cls._specimen_chip(lay, footprint, at, substrate, edge_key=edge_key, view=(fw, fh))
                          if seen else lambda x, y: False)
            else:
                inside = cls._chip(lay, corners, at, sx, substrate)
            if seen:
                metal = meta.get("metal_region") or {}
                if metal.get("outline_mm"):
                    lay.polygon([at(x, y) for x, y in metal["outline_mm"]], (177, 175, 168), .9)
                for shape in meta.get("electrodes", []):
                    pts = [at(x, y) for x, y in shape["points"]]
                    color = (196, 168, 92) if shape["kind"] == "contact" else (154, 127, 91)
                    lay.polygon([(x+1, y+1) for x, y in pts], (75, 64, 77), .7)
                    lay.polygon(pts, color)
                    if shape["kind"] == "contact":
                        lay.polygon([(x-.4, y-.4) for x, y in pts], (223, 200, 126), .28)
                if texture is None:
                    cls._residue(lay, rng, inside, sx, sy, fw, fh)
                elif texture.get("residue", True):
                    cls._specimen_residue(lay, texture, inside, sx, sy, fw, fh)
                material_profile = calibration.get("material_capture_profile")
                if calibration.get("neighbourhood_capture_profile") or stacking:
                    material_profile = V1_MATERIAL_CAPTURE_PROFILE
                def tint(f):
                    return (_v1_material_tint(f.get("material", "Gr"), int(f["layers"]), meta["objective"], substrate)
                            if material_profile == V1_MATERIAL_CAPTURE_PROFILE and f.get("layers") is not None else None)
                metal_shapes = [[at(x, y) for x, y in shape["points"]] for shape in meta.get("electrodes", [])]
                if (meta.get("metal_region") or {}).get("outline_mm"):
                    metal_shapes.append([at(x, y) for x, y in meta["metal_region"]["outline_mm"]])
                layered = (flakes and all(f.get("outline_mm") and f.get("layers") is not None for f in flakes)
                           and (metal_shapes or any(f.get("stack_level") for f in flakes)))
                if v1_camera and texture is not None:
                    # Under the flakes: debris lies on the bare SiO2 between them.
                    _surface_texture(lay, texture, fw, fh, meta["objective"], substrate)
                if layered:
                    cls._layered_flakes(lay, sorted(flakes, key=lambda f: f.get("stack_level", 0)), at,
                                        metal_shapes, substrate, tint)
                else:
                    for f in flakes:
                        cls._flake(lay, f, at, sx, sy, rng, color=tint(f))
                for disc in meta.get("pc_films", []):
                    # A released PC film left on the chip (#180): a faint film over
                    # the stack with a torn edge; the stack stays visible under it.
                    _film_disc(lay, *at(disc["dx_mm"], disc["dy_mm"]), disc["radius_mm"]*sx,
                               fill=(236, 226, 196), fill_alpha=.06, edge=(92, 80, 70), edge_alpha=.45,
                               torn=True)
            c.px[:] = lay.px
            if sigma > 0:
                c.px[:] = _gaussian_defocus(c.px, w, h, sigma)
            if chip_key:
                _CHIP_PLANES.put(chip_key, (bytes(c.px), rng.getstate()))
        if meta.get("stamp") is not None:
            cls._stamp(c, meta, sharp_stamp, w / fw, h / fh, rng, chip_sigma=sigma, px_scale=px_scale)
        _camera(c, _exposure_noise_seed(seed, meta), float(meta.get("observation_noise_scale", 1.0)),
                V1_SEARCH_CAMERA_PROFILE if v1_camera else None)
        # A real camera frame has no burned-in caption. Analysis frames of the
        # v1 camera stay clean so the caption cannot enter focus/vision pixels.
        if not v1_camera or meta.get("purpose") == "motion_preview":
            cls._legend(c, int(meta.get("objective", 10) or 10), fw)
        # Motion previews are display only (never verification): JPEG (#205).
        if meta.get("purpose") == "motion_preview":
            return c.jpeg(DISPLAY_JPEG_QUALITY)
        # v1 keeps its search photos as JPEG (cv2.imwrite test_N.jpg) and its
        # recognizer reads them back; the v1-camera material exposure is the
        # same JPEG, in memory and on disk (#203). Stamp/contact frames stay
        # lossless (v1 stacking saves step_N.png).
        if native_material and meta.get("purpose", "analysis_capture") == "analysis_capture":
            return c.jpeg()
        return c.png()

    @staticmethod
    def _specimen_outline(texture: dict[str, Any]) -> list[tuple[float, float]]:
        """Finite chip footprint in the same apparent frame as its texture."""
        pose = texture["pose"]
        shift = texture.get("apparent_shift_xy_mm", (0, 0))
        cx, cy = (value-offset for value, offset in
                  zip(texture["view_center_world_mm"], shift, strict=True))
        c, s = math.cos(math.radians(pose["rotation"])), math.sin(math.radians(pose["rotation"]))
        hw, hh = (value/2 for value in texture["size_mm"])
        return [(pose["x"]+c*x-s*y-cx, pose["y"]+s*x+c*y-cy)
                for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]

    @staticmethod
    def _footprint_in_view(points: list[tuple[float, float]], fw: float, fh: float) -> bool:
        """Rectangle intersection, including edge crossings with no visible vertex."""
        axes = [(1.0, 0.0), (0.0, 1.0)]
        axes.extend((-(by-ay), bx-ax) for (ax, ay), (bx, by) in
                    zip(points[:2], points[1:3], strict=True))
        for nx, ny in axes:
            extent = abs(nx)*fw/2 + abs(ny)*fh/2
            projections = [nx*x+ny*y for x, y in points]
            if min(projections) >= extent or max(projections) <= -extent:
                return False
        return True

    @staticmethod
    def _specimen_chip(lay: Canvas, outline: list[tuple[float, float]], at: Any,
                       substrate: RGB = SUBSTRATE, *, edge_key: int | None = None,
                       view: tuple[float, float] | None = None) -> Any:
        """Clip the supplied chip surface and finite illuminated edges to the view."""
        points = [at(x, y) for x, y in outline]
        lay.rect(0, 0, lay.w, lay.h, OFF_CHIP)
        lay.polygon(points, substrate)
        # The lit cleaved edge is part of the chip: an 8 um band just inside
        # the outline (the outline runs anticlockwise, so inward is left).
        # Drawn across the outline it put light 4 um off the chip and moved
        # pixel-found corners by ~6 um (#183).
        for side, ((ax, ay), (bx, by)) in enumerate(zip(outline, outline[1:]+outline[:1], strict=True)):
            length = math.hypot(bx-ax, by-ay)
            nx, ny = -(by-ay)*.008/length, (bx-ax)*.008/length
            if edge_key is None or view is None:
                lay.polygon([at(ax, ay), at(bx, by), at(bx+nx, by+ny), at(ax+nx, ay+ny)], (214, 206, 228), .75)
                continue
            # Hand-cut chips (#204): the scribed and cleaved edge is chipped, so
            # its lit band varies in brightness and depth every few um. Fixed
            # to the chip (seeded by chip and side, indexed by distance along
            # the side), so two photos of one corner share the same marks, as
            # v1's single-corner ORB match relies on. The outline itself is
            # unchanged, so the edge position the corner finder fits is too.
            ux, uy = (bx-ax)/length, (by-ay)/length
            reach = math.hypot(*view)/2 + .02
            along = -(ax*ux + ay*uy)  # distance along the side closest to the view centre
            if abs(ax*uy - ay*ux) > reach:
                continue
            k0 = max(0, math.floor((along-reach)/_CUT_EDGE_SEGMENT_MM))
            k1 = min(math.ceil(length/_CUT_EDGE_SEGMENT_MM), math.ceil((along+reach)/_CUT_EDGE_SEGMENT_MM))
            for k in range(k0, k1):
                u = _cut_edge_hash(edge_key, side, k)
                t0, t1 = k*_CUT_EDGE_SEGMENT_MM, min(length, (k+1)*_CUT_EDGE_SEGMENT_MM)
                depth = .004 + .008*((u >> 8) & 0xFF)/255  # 4-12 um inward
                gain = .55 + .55*(u & 0xFF)/255
                colour = tuple(min(255, int(round(v*gain))) for v in (214, 206, 228))
                p0, p1 = (ax+ux*t0, ay+uy*t0), (ax+ux*t1, ay+uy*t1)
                dx, dy = nx/.008*depth, ny/.008*depth
                lay.polygon([at(*p0), at(*p1), at(p1[0]+dx, p1[1]+dy), at(p0[0]+dx, p0[1]+dy)], colour, .75)

        def inside(x: float, y: float) -> bool:
            return all((bx-ax)*(y-ay)-(by-ay)*(x-ax) <= 0 for
                       (ax, ay), (bx, by) in zip(points, points[1:]+points[:1], strict=True))

        return inside

    @staticmethod
    def _chip(lay: Canvas, corners: list[dict[str, Any]], at: Any, sx: float,
              substrate: RGB = SUBSTRATE) -> Any:
        """Dark stage beyond the chip's edge when a corner is in view; returns
        a predicate (layer px) -> on the chip."""
        if not corners:
            return lambda x, y: True
        cr = corners[0]
        cx, cy = at(cr["dx_mm"], cr["dy_mm"])
        th = math.radians(cr.get("chip_rotation", 0.0))
        # chip-local axes in layer px (image y points down)
        ux, uy = math.cos(th), -math.sin(th)
        vx, vy = -math.sin(th), -math.cos(th)
        su = 1.0 if cr["corner"] in ("bl", "tl") else -1.0
        sv = 1.0 if cr["corner"] in ("bl", "br") else -1.0
        L = 60 * sx  # far beyond the view
        pts = [(cx, cy), (cx + su * ux * L, cy + su * uy * L),
               (cx + su * ux * L + sv * vx * L, cy + su * uy * L + sv * vy * L),
               (cx + sv * vx * L, cy + sv * vy * L)]
        lay.rect(0, 0, lay.w, lay.h, OFF_CHIP)
        lay.polygon(pts, substrate)
        # the cleaved edge catches the light: a bright line along both edges
        e = max(1.0, 0.004 * sx)
        for ax, ay in ((su * ux, su * uy), (sv * vx, sv * vy)):
            nx, ny = -ay, ax
            lay.polygon([(cx - nx * e, cy - ny * e), (cx + ax * L - nx * e, cy + ay * L - ny * e),
                         (cx + ax * L + nx * e, cy + ay * L + ny * e), (cx + nx * e, cy + ny * e)],
                        (214, 206, 228), 0.75)

        def inside(x: float, y: float) -> bool:
            dx, dy = x - cx, y - cy
            return (dx * ux + dy * uy) * su >= 0 and (dx * vx + dy * vy) * sv >= 0

        return inside

    @staticmethod
    def _specimen_residue(lay: Canvas, texture: dict[str, Any], inside: Any,
                          sx: float, sy: float, fw: float, fh: float) -> None:
        """Existing residue morphology in fixed one-mm chip-local tiles.

        Tile generation is independent of camera position, objective, visible
        flakes and exposure count. Only the final projection uses the view.
        Historical packets without a specimen binding retain _residue.
        """
        pose = texture["pose"]
        shift = texture.get("apparent_shift_xy_mm", (0, 0))
        center = [value-offset for value, offset in zip(texture["view_center_world_mm"], shift, strict=True)]
        c, s = math.cos(math.radians(pose["rotation"])), math.sin(math.radians(pose["rotation"]))
        hw, hh = (value/2 for value in texture["size_mm"])
        viewport = []
        for x, y in ((-fw/2, -fh/2), (-fw/2, fh/2), (fw/2, -fh/2), (fw/2, fh/2)):
            dx, dy = center[0]+x-pose["x"], center[1]+y-pose["y"]
            viewport.append((c*dx+s*dy, -s*dx+c*dy))

        def at(x, y):
            return (lay.w/2+(pose["x"]+c*x-s*y-center[0])*sx,
                    lay.h/2-(pose["y"]+s*x+c*y-center[1])*sy)

        for ix in range(math.floor(max(-hw, min(x for x, _ in viewport)-.02)),
                        math.floor(min(hw, max(x for x, _ in viewport)+.02))+1):
            for iy in range(math.floor(max(-hh, min(y for _, y in viewport)-.02)),
                            math.floor(min(hh, max(y for _, y in viewport)+.02))+1):
                rng = random.Random(f"{texture['seed']}:{ix}:{iy}")
                for density, radius, vertices, color in ((6, (3, 14), 8, (92, 70, 72)),
                                                         (12, (.8, 3), 5, (228, 222, 196))):
                    for _ in range(int(density+rng.random()*3)):
                        x, y = ix+rng.random(), iy+rng.random()
                        r = rng.uniform(*radius)/1000
                        aspect = rng.uniform(.5, 1) if vertices == 8 else .8
                        polygon = _crystal(rng, x, y, r, r*aspect, n=vertices)
                        alpha = rng.uniform(.2, .4) if vertices == 8 else .9
                        px, py = at(x, y)
                        if abs(x) <= hw and abs(y) <= hh and r*sx >= .5 and inside(px, py):
                            if any(abs(x) > hw or abs(y) > hh for x, y in polygon):
                                polygon = MicroscopeRenderer._clip_texture(polygon, hw, hh)
                            lay.polygon([at(x, y) for x, y in polygon], color, alpha)

    @staticmethod
    def _clip_texture(points: list[tuple[float, float]], hw: float, hh: float
                      ) -> list[tuple[float, float]]:
        """Keep residue morphology within the finite chip-local surface."""
        for axis, sign, bound in ((0, 1, hw), (0, -1, hw), (1, 1, hh), (1, -1, hh)):
            clipped = []
            for a, b in zip(points, points[1:]+points[:1], strict=True):
                da, db = sign*a[axis]-bound, sign*b[axis]-bound
                if da <= 0:
                    clipped.append(a)
                if (da < 0 < db) or (db < 0 < da):
                    t = da/(da-db)
                    clipped.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
            points = clipped
        return points

    @staticmethod
    def _residue(lay: Canvas, rng: random.Random, inside: Any, sx: float, sy: float,
                 fw: float, fh: float) -> None:
        """Tape glue left by the peel (brownish, translucent patches) and
        crumbs of thick graphite too small to be flakes (bright specks), at a
        fixed density per mm2 so a 100x view holds a hundredth of a 10x view."""
        area = fw * fh
        for _ in range(int(6 * area + rng.random() * 3)):
            x, y = rng.uniform(0, lay.w), rng.uniform(0, lay.h)
            if not inside(x, y):
                continue
            r = rng.uniform(3, 14) / 1000 * sx  # 3-14 um
            if r < 0.6:
                continue
            pts = _crystal(rng, x, y, r, r * rng.uniform(0.5, 1.0), n=8)
            lay.polygon(pts, (92, 70, 72), rng.uniform(0.2, 0.4))
        for _ in range(int(12 * area + rng.random() * 3)):
            x, y = rng.uniform(0, lay.w), rng.uniform(0, lay.h)
            if not inside(x, y):
                continue
            r = rng.uniform(0.8, 3) / 1000 * sx  # 0.8-3 um
            if r < 0.5:
                continue
            lay.polygon(_crystal(rng, x, y, r, r * 0.8, n=5), (228, 222, 196), 0.9)

    @staticmethod
    def _layered_flakes(lay: Canvas, flakes: list[dict[str, Any]], at: Any,
                        metal_shapes: list[list[tuple[float, float]]], substrate: RGB, tint: Any) -> None:
        """A see-through stack: every layer adds its own colour change (#185).

        On SiO2/Si a flake shifts the reflected colour from the substrate's to
        its tint (the v1 colour when that profile is on). In a stack these
        shifts add up, so every overlap has its own colour and every layer's
        outline stays traceable through the layers above it (Gr under hBN
        shows as a slightly darker hBN; hBN over Gr as a coloured Gr). On metal
        a thin layer barely changes the strong metal reflection, so electrodes
        show through at about their own colour. The order of drawing does not
        matter. The surface texture under the stack is kept. Illustrative
        thin-film behaviour, not a transfer-matrix calculation.
        """
        w, h = lay.w, lay.h
        image = np.frombuffer(bytes(lay.px), dtype=np.uint8).reshape(h, w, 3).astype(np.float64)
        base = np.asarray(substrate, dtype=np.float64)
        metal = np.zeros((h, w), dtype=bool)
        for shape in metal_shapes:
            m = Canvas(w, h, (0, 0, 0))
            m.polygon(shape, (255, 255, 255))
            metal |= np.frombuffer(bytes(m.px), dtype=np.uint8).reshape(h, w, 3)[..., 0] > 0
        for f in flakes:
            shown = np.asarray(tint(f) or flake_tint(f.get("material", "Gr"), int(f["layers"])), dtype=np.float64)
            m = Canvas(w, h, (0, 0, 0))
            m.polygon([at(x, y) for x, y in f["outline_mm"]], (255, 255, 255))
            own = np.frombuffer(bytes(m.px), dtype=np.uint8).reshape(h, w, 3)[..., 0] > 0
            change = shown - base
            image[own & ~metal] += change
            image[own & metal] += METAL_LAYER_CONTRAST*change
        lay.px[:] = np.rint(np.clip(image, 0, 255)).astype(np.uint8).tobytes()

    @staticmethod
    def _flake(lay: Canvas, f: dict[str, Any], at: Any, sx: float, sy: float,
               rng: random.Random, alpha: float = 1.0, grow: float = 1.0, color: RGB | None = None) -> None:
        """One flake: its outline in the tint of its layer count, a dark edge
        line, and a terrace (thin flakes carry a thicker patch, thick ones a
        thin tail) as cleaved crystals do."""
        layers, mat = f.get("layers"), f.get("material", "Gr")
        if layers is None:
            # Qualitative reaction product: no invented layer count or optical
            # thickness. Use the same explicit synthetic product tint as reactor.
            if f.get("outline_mm"):
                lay.polygon([at(x, y) for x, y in f["outline_mm"]], (223, 187, 100), alpha)
            return
        layers = int(layers)
        if f.get("outline_mm"):
            # Stored geometry is the physical boundary. Pixel size, focus and
            # objective must not grow it, add tails or redraw a new polygon.
            pts = [at(x, y) for x, y in f["outline_mm"]]
            lay.polygon(pts, color if color is not None else flake_tint(mat, layers), alpha)
            return
        frng = random.Random(sum(map(ord, f["sim_id"])))
        r_mm = math.sqrt(f["area_um2"] / math.pi) / 1000.0
        asp = frng.uniform(1.0, 1.9)
        rx = max(1.0, r_mm * sx * 1.05 * asp * grow)
        ry = max(1.0, r_mm * sy * 1.05 / asp * grow)
        cx, cy = at(f["dx_mm"], f["dy_mm"])
        pts = _crystal(frng, cx, cy, rx, ry)
        col = color if color is not None else flake_tint(mat, layers)
        lay.polygon(_scaled(pts, 1.0 + 1.6 / max(rx, ry)), _shade(col, 0.72), 0.8 * alpha)
        lay.polygon(pts, col, alpha)
        if layers <= 6:
            inner = _scaled(pts, frng.uniform(0.3, 0.5), rx * frng.uniform(-0.3, 0.3),
                            ry * frng.uniform(-0.3, 0.3))
            lay.polygon(inner, flake_tint(mat, layers * 2 + 1), alpha)
        else:
            tail = _scaled(pts, 0.35, rx * frng.choice((-0.9, 0.9)), ry * frng.uniform(-0.4, 0.4))
            lay.polygon(tail, flake_tint(mat, max(1, layers // 10)), 0.9 * alpha)
            lay.polygon(_scaled(pts, 0.5, rx * 0.1, -ry * 0.1), flake_tint(mat, layers * 2), alpha)

    @staticmethod
    def _white_light_rings(c: Canvas, geometry: dict, sx: float, sy: float) -> None:
        """Project frozen SIM geometry; never infer contact or advance physics.

        Before contact the gap is the local paraboloid. After contact the dome
        is flattened onto the chip inside the current front radius ``a`` and
        keeps its curvature outside it, so the gap is ``(r**2-a**2)/(2R)``:
        zero at the front, which the first fringe therefore hugs while the
        front still lags the rigid overlap radius (#179). Four subpixels
        integrate narrow fringes at the native raster scale.
        """
        if geometry.get("profile_id") != WHITE_LIGHT_PROFILE:
            raise ValueError("unknown white-light optical profile")
        gap, radius, dx, dy, contact = (geometry[k] for k in (
            "signed_apex_gap_mm", "dome_radius_mm", "centre_dx_mm", "centre_dy_mm",
            "current_contact_radius_mm"))
        if (any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                for v in (gap, radius, dx, dy, contact)) or radius <= 0 or contact < 0
                or not isinstance(geometry.get("has_contact"), bool)):
            raise ValueError("invalid synthetic optical-gap geometry")
        deformed = geometry["has_contact"]
        # Inside the front the gap is closed; outside it the apex offset is
        # replaced by the front radius (zero gap at r = a).
        extent_squared = (2*radius*WHITE_LIGHT_MAX_GAP_MM+contact**2 if deformed
                          else 2*radius*(WHITE_LIGHT_MAX_GAP_MM-gap))
        if extent_squared <= 0:
            return
        extent = math.sqrt(extent_squared)
        cx, cy = c.w/2+dx*sx, c.h/2-dy*sy
        x0, x1 = max(0, math.floor(cx-extent*sx-1)), min(c.w, math.ceil(cx+extent*sx+1))
        y0, y1 = max(0, math.floor(cy-extent*sy-1)), min(c.h, math.ceil(cy+extent*sy+1))
        if x1 <= x0 or y1 <= y0:
            return
        # Elementwise float64 in the original per-pixel order (four subpixels,
        # x-major), so the RGB8 output is unchanged; only the loop is NumPy.
        table = np.asarray(white_light_table(), dtype=np.float64)
        scale = (len(table)-1)/WHITE_LIGHT_MAX_GAP_MM
        contact_squared = contact**2 if geometry["has_contact"] else 0.
        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        sums = np.zeros((len(ys), len(xs), 4))
        for ax in ((xs+.25-cx)/sx)**2, ((xs+.75-cx)/sx)**2:
            for by in ((ys+.25-cy)/sy)**2, ((ys+.75-cy)/sy)**2:
                r2 = by[:, None]+ax[None, :]
                g = (r2-contact_squared)/(2*radius) if deformed else gap+r2/(2*radius)
                use = (r2 > contact_squared) & (g > 0) & (g < WHITE_LIGHT_MAX_GAP_MM)
                if not use.any():
                    continue
                position = g[use]*scale
                index = position.astype(np.int64)
                fraction = position-index
                low, high = table[index], table[index+1]
                sums[use] += low+fraction[:, None]*(high-low)
        alpha = sums[..., 3]
        lit = alpha != 0
        if not lit.any():
            return
        image = np.frombuffer(c.px, dtype=np.uint8).reshape(c.h, c.w, 3)
        region = image[y0:y1, x0:x1]
        old = region[lit].astype(np.float64)
        mixed = old*(1-alpha[lit, None]/4)+sums[lit][:, :3]/4
        region[lit] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)

    @classmethod
    def _stamp(cls, c: Canvas, meta: dict[str, Any], sharp_stamp: float, sx: float,
               sy: float, rng: random.Random, chip_sigma: float = 0.0, px_scale: float = 1.0) -> None:
        """The PC film seen through the dome: a faint veil over everything, the
        contact area (lighter, with a dark front where the film meets the
        chip), a torn film's ragged dark edge, and the flakes on the film."""
        w, h = c.w, c.h
        structure = StampStructure.from_packet(meta["stamp_structure"]) if "stamp_structure" in meta else None
        if structure is not None:
            # Illustrative transmission only; plan-view outlines cannot provide
            # glass/PDMS thickness, index or deformation.
            def project(point):
                return w/2+point[0]*sx, h/2-point[1]*sy

            c.polygon([project(p) for p in structure.glass_outline_mm], (180, 205, 216), .025)
            cx, cy = structure.cap_center_mm
            cap = [(cx+structure.cap_radius_mm*math.cos(i*math.tau/128),
                    cy+structure.cap_radius_mm*math.sin(i*math.tau/128)) for i in range(128)]
            c.polygon([project(p) for p in cap], (174, 184, 206), .025)
        if "stamp_texture_seed" in meta:
            rng = random.Random(meta["stamp_texture_seed"])
        c.rect(0, 0, w, h, (150, 160, 196), 0.10)

        def at(dx: float, dy: float) -> tuple[float, float]:
            return w / 2 + dx * sx, h / 2 - dy * sy

        # White-light rings and the contact front sit in the micrometre gap at
        # the chip surface, so they share the chip-plane defocus (a518 seed 2:
        # drawn sharp at every focus, they flattened the chip autofocus curve
        # over a bare discovery site). Their change to the composed image is
        # spread with the chip sigma; at chip focus nothing changes.
        before = bytes(c.px) if chip_sigma > 0 else None
        optical_gap = meta.get("optical_gap")
        if optical_gap is not None:
            cls._white_light_rings(c, optical_gap, sx, sy)
        ct = meta.get("contact")
        film_ok = bool(meta["stamp"].get("film", True))
        if ct is not None and ct.get("pinned"):
            # Molten PC pinned to the chip (#180): its pinned edge stays put
            # while, on lift, the PDMS separates inside it at the front.
            pin = ct["pinned"]
            _film_disc(c, *at(pin["dx_mm"], pin["dy_mm"]), pin["radius_mm"]*sx, fill=(236, 226, 196),
                       fill_alpha=.06, edge=(92, 80, 70), edge_alpha=.45, edge_px=1.5*px_scale)
        if ct is not None:
            cx, cy = at(ct["dx_mm"], ct["dy_mm"])
            r = max(2.0*px_scale, ct["radius_mm"] * sx)
            optics = SYNTHETIC_CONTACT_PHOTOMETRY
            # The physical front is circular. A coarse 16-gon biases centroid
            # estimates from clipped arcs; resolve this boundary before sensing.
            separating = ct.get("pinned") and ct["radius_mm"] < ct["pinned"]["radius_mm"]-1/sx
            # The bonded interior keeps its look while the PDMS peels; only
            # the dark wetting band outside the front goes once it separates.
            c.disc(cx, cy, r if separating else r + max(2.0*px_scale, r * 0.06), optics.front_color,
                   optics.front_opacity, segments=64)
            c.disc(cx, cy, r, optics.interior_color, optics.interior_opacity, segments=64)
            if separating:
                # The PDMS/PC separation boundary is a thin bright air wedge,
                # not the dark wetting front of an advancing contact.
                _film_disc(c, cx, cy, r, fill=(250, 248, 240), fill_alpha=0., edge=(250, 248, 240),
                           edge_alpha=.45, edge_px=1.5*px_scale)
            if not film_ok:
                crack = [(cx + r * rng.uniform(-1.0, 1.0), cy + r * rng.uniform(-1.0, 1.0))
                         for _ in range(6)]
                for (x0, y0), (x1, y1) in zip(crack, crack[1:], strict=False):
                    nx, ny = -(y1 - y0), x1 - x0
                    d = max(1e-6, math.hypot(nx, ny))
                    nx, ny = nx / d * 1.5*px_scale, ny / d * 1.5*px_scale
                    c.polygon([(x0 - nx, y0 - ny), (x1 - nx, y1 - ny), (x1 + nx, y1 + ny),
                               (x0 + nx, y0 + ny)], (30, 22, 40), 0.85)
        if before is not None and bytes(c.px) != before:
            c.px[:] = _defocus_change(before, c.px, w, h, chip_sigma)
        # Constant physical transmission; defocus spreads the premultiplied
        # carried layer, never the already composed chip/contact image. This
        # fixed 3px/4-sigma model is synthetic and profile-bound, not calibrated.
        if meta.get("stamp_flakes"):
            pad = math.ceil(STAMP_PSF_PADDING_PX*px_scale)
            plane = _stamp_plane(meta, c.w, c.h, sx, sy, pad)
            _, weights = _stamp_psf_kernel(sharp_stamp, px_scale)
            channels = [_convolve_stamp_plane(p, weights) for p in plane.images()]
            channels = [p.crop((pad, pad, pad+c.w, pad+c.h)) for p in channels]
            # Only where the carried layer reaches: elsewhere every channel
            # is zero and the composite is the image itself.
            box = None
            for p in channels:
                b = p.getbbox()
                if b is not None:
                    box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                                 max(box[2], b[2]), max(box[3], b[3]))
            if box is not None:
                x0, y0, x1, y1 = box
                values = np.stack([np.asarray(p.crop(box), dtype=np.float64) for p in channels], axis=2)
                image = np.frombuffer(c.px, dtype=np.uint8).reshape(c.h, c.w, 3)[y0:y1, x0:x1]
                mixed = values[..., :3]+(1-values[..., 3:4])*image
                image[:] = np.clip(np.trunc(mixed), 0, 255).astype(np.uint8)
        if structure is not None:
            cls._paper_mask(c, structure, sx, sy)

    @staticmethod
    def _paper_mask(c: Canvas, structure: StampStructure, sx: float, sy: float) -> None:
        """Opaque apertured paper above the slide, from frozen support geometry.

        Antialiasing is a raster operation, not a mask-plane focus model. It
        masks source, contact and carried material equally; no desired pickup
        outcome or process phase affects transmission.
        """
        cx, cy = structure.aperture_center_mm
        coverage = _paper_coverage(c.w, c.h, sx, sy,
                                   tuple(tuple(p) for p in structure.glass_outline_mm),
                                   (cx, cy), structure.aperture_radius_mm, structure.aperture_shape)
        covered = coverage != 0
        alpha = coverage[covered, None]/255
        paper = np.asarray((184, 179, 163), dtype=np.float64)
        image = np.frombuffer(c.px, dtype=np.uint8).reshape(-1, 3)
        image[covered] = np.rint(image[covered]*(1-alpha)+paper*alpha).astype(np.uint8)

    @staticmethod
    def _legend(c: Canvas, objective: int, fw: float) -> None:
        """The objective at the top left, a scale bar at the bottom right.

        Drawn at the same share of the picture on a 4K display frame (#205)
        as on the 480 px synthetic one.
        """
        w, h = c.w, c.h
        k = max(1, round(w/480)) if w > 640 else 1  # 1 on every historical raster
        size = (2 if w >= 400 else 1)*k
        c.text(8*k, 8*k, f"{objective}X", (240, 240, 240), size)
        c.text(8*k, h - 15*k, "SIMULATED", (205, 205, 215), k)
        fov_um = fw * 1000.0
        bar_um = max((b for b in (5, 10, 20, 50, 100, 200, 500, 1000) if b <= fov_um / 4),
                     default=5)
        bar = bar_um / fov_um * w
        x1, y1 = w - 12*k, h - 12*k
        c.rect(x1 - bar + k, y1 - 3*k + k, x1 + k, y1 + k, (20, 20, 20), 0.6)
        c.rect(x1 - bar, y1 - 3*k, x1, y1, (245, 245, 245))
        label = f"{bar_um} UM"
        c.text(x1 - len(label) * 6 * size, y1 - 6*k - 7 * size, label, (245, 245, 245), size)


def ensure_diagnostic_images(services: Any) -> None:
    """Keep real rendered observations for an in-memory diagnostic's lifetime.

    Existing output directories/renderers are retained. The directory owner is
    attached to the services so source-image references survive scenario return.
    """
    from tempfile import TemporaryDirectory

    microscope = services.lab.devices.get("microscope")
    if microscope.renderer is None or microscope.renderer.writer is None:
        directory = TemporaryDirectory(prefix="qumus-diagnostic-images-")
        services._diagnostic_images = directory
        microscope.renderer = MicroscopeRenderer(Path(directory.name))
