"""Simple 2D frames for exercising the simulated overview-camera interface.

Regions and markers follow the simulated state. This is an interface demo,
not a model of the real laboratory's appearance or a vision benchmark.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from qumus_simlab.testbench.render import _Writer


class BenchCameraRenderer:
    """A flat bench diagram with samples and arm-position markers."""

    def __init__(self, bench: dict[str, Any], out_dir: Path) -> None:
        self.bench = bench
        self.writer = _Writer(out_dir)
        self._lock = threading.Lock()
        self._cache: tuple[str, bytes, str] | None = None
        self.size = (1100, 650)

    def _p(self, x: float, y: float) -> tuple[float, float]:
        table = self.bench['table']
        x0, x1 = table['x']
        y0, y1 = table['y']
        return 36 + (x-x0) / (x1-x0) * 1028, 568 - (y-y0) / (y1-y0) * 480

    def render(self, device_id: str, objects: list[dict[str, Any]], tape: dict[str, Any],
               arms: dict[str, dict[str, Any]], fixed: bool = False) -> str:
        key = hashlib.sha256(json.dumps([objects, tape, arms], sort_keys=True).encode()).hexdigest()
        with self._lock:
            if self._cache and self._cache[0] == key:
                data, path = self._cache[1:]
                if not fixed:
                    return path
            else:
                data = self._draw(objects, tape, arms)
                path = self.writer.write(device_id, data)
                self._cache = (key, data, path)
            return self.writer.write(device_id, data, fixed=True) if fixed else path

    def _draw(self, objects: list[dict[str, Any]], tape: dict[str, Any],
              arms: dict[str, dict[str, Any]]) -> bytes:
        im = Image.new('RGB', self.size, '#f8fafc')
        d = ImageDraw.Draw(im)
        font = ImageFont.load_default(size=17)
        small = ImageFont.load_default(size=14)
        ink, muted, border = '#263348', '#66758a', '#cad3df'
        colors = {'chip': '#8974cc', 'carrier': '#5592bf', 'stamp': '#4ba596'}
        d.text((28, 22), 'Overview camera', font=ImageFont.load_default(size=24), fill=ink)
        d.text((1072, 29), 'SIMULATED / 2D DIAGRAM', font=font, anchor='ra', fill=muted)
        d.line((28, 62, 1072, 62), fill=border)

        def rect(region):
            return (*self._p(region['x'][0], region['y'][1]),
                    *self._p(region['x'][1], region['y'][0]))

        labels = {'left_storage': 'Left storage', 'middle_storage': 'Middle storage',
                  'right_storage': 'Right storage', 'exfoliation_stage': 'Exfoliation',
                  'transfer_stage': 'Search / stacking', 'reactor_storage': 'Reactor storage',
                  'reactor': 'vdW reactor'}
        for name, st in self.bench['stations'].items():
            if st.get('placeholder'):
                continue
            box = rect(st.get('module', st['region']))
            d.rounded_rectangle(box, radius=8, fill='#eef2f7', outline=border, width=2)
            d.text(((box[0]+box[2])/2, box[1]-12), labels.get(name, st['label']),
                   font=font, anchor='ms', fill=ink)
            if st['kind'] != 'storage':
                d.rectangle(rect(st['region']), outline=muted, width=2)
        for slot in self.bench['slots'].values():
            if self.bench['stations'][slot['station']].get('placeholder'):
                continue
            x, y = self._p(slot['x'], slot['y'])
            w, h = slot.get('size', [10, 10])
            w, h = max(6, w*.41), max(6, h*.38)
            d.rectangle((x-w, y-h, x+w, y+h), outline=border)

        # A tape strip and sample markers convey state without machine geometry.
        if tape.get('applied') or tape.get('laid'):
            origin = self.bench['exfoliator']['origin']
            end = origin['x'] + tape.get('gantry', {}).get('x', 0)
            d.line([self._p(origin['x'], origin['y']), self._p(end, origin['y'])],
                   fill='#dfb960', width=9)
        for obj in objects:
            x, y = self._p(obj['x'], obj['y'])
            w, h = obj.get('size_mm', [10, 10])
            w, h = max(5, w*.41), max(5, h*.38)
            angle = math.radians(obj.get('rotation', 0))
            points = [(x+a*math.cos(angle)+b*math.sin(angle),
                       y-a*math.sin(angle)+b*math.cos(angle))
                      for a, b in [(-w,-h),(w,-h),(w,h),(-w,h)]]
            d.polygon(points, fill=colors.get(obj['kind'], muted), outline=ink)
            if obj.get('electrodes'):
                d.line((x-3, y, x+3, y), fill='#f8df8a', width=2)

        states = []
        for name, arm in arms.items():
            label = 'Arm 1' if name == 'arm_left' else 'Arm 2'
            x, y = self._p(arm['x'], arm['y'])
            color = '#be6b36' if name == 'arm_left' else '#347ca4'
            d.ellipse((x-11, y-11, x+11, y+11), outline=color, width=2)
            d.line((x-15, y, x+15, y), fill=color, width=2)
            d.line((x, y-15, x, y+15), fill=color, width=2)
            d.text((x+19, y-8), label, font=small, fill=color)
            states.append(f"{label}: vacuum {'on' if arm.get('vacuum') else 'off'}")

        d.line((28, 584, 1072, 584), fill=border)
        for i, (kind, color) in enumerate(colors.items()):
            x = 28 + i*125
            d.rectangle((x, 604, x+12, 616), fill=color)
            d.text((x+20, 601), kind.title(), font=small, fill=ink)
        states.append('Tape: applied' if tape.get('applied') or tape.get('laid') else 'Tape: clear')
        d.text((1072, 601), '  /  '.join(states), font=small, anchor='ra', fill=muted)
        out = io.BytesIO()
        im.save(out, format='PNG')
        return out.getvalue()
