"""Maintained pure RGB core ported from frozen v1; no SDK, devices or inventory.

Donor segmenter.py SHA256: 44bd5966134169b4feb9d76f1995687696b2f07cd48a359bafec3e4e9429f36c
Algorithm equations/tables are historical implementation, not camera calibration.
The historical/default branch retains donor equations and tables, omitting UI
helpers and using a private bounded RNG. The explicit native SIM branch adds
operator-bounded frame support, necessary raw-substrate camera compatibility,
joint-foreground exclusion and direct conditional-background sampling. These are
versioned compatibility adaptations, not unchanged v1 or real-camera accuracy.
"""
import concurrent.futures
import math

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import label


class _BoundedRandom:
    def __init__(self, seed):
        self.generator = np.random.RandomState(seed)
        self.draws = 0

    def randint(self, **kwargs):
        self.draws += 1
        if self.draws > 1_000_000:
            raise ValueError("V1 RGB background sampling exceeded its bounded work budget")
        return self.generator.randint(**kwargs)


class Segmenter:

    def __init__(self, img, material, colors=None, numbers=None, min_area=None, max_area=None, magnification=100, k=3, focus_disks=None, thickness=40, surround_bg=None, seed=0, background_support="historical_v1", observation_noise_scale=None):
        if background_support not in ("historical_v1", "native_border_and_joint_foreground_v1"):
            raise ValueError("Unsupported RGB background-support policy")
        self.background_support = background_support
        self.observation_noise_scale = observation_noise_scale
        self.img = img
        self.size = img.shape[:2]
        self.target_bg_lab = material.target_bg_lab_for_mag(magnification)
        self.layer_labels = material.layer_labels_for_mag(magnification).copy()
        if magnification >= 50:
            self.edge_methods = [material.Edge_Method(args, k=k) for args in material.args[:2]]
        else:
            self.edge_methods = [material.Edge_Method(args, k=k) for args in material.args[1:]]
        self.colors = colors
        self.numbers = numbers
        if min_area is None:
            min_area = 2000 if magnification >= 50 else 200
        if max_area is None:
            max_area = 10000000 if magnification >= 50 else 100000
        self.min_area = min_area
        self.max_area = max_area
        self.focus_disks = [] if focus_disks is None else focus_disks
        self.rng = _BoundedRandom(seed)
        self.mag = magnification
        self.lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.int16)
        self.lab[:, :, 0] = self.lab[:, :, 0] * 100.0 / 255.0
        self.lab[:, :, 1] -= 128
        self.lab[:, :, 2] -= 128
        self.thickness = thickness
        if surround_bg is None:
            surround_bg = magnification == 20
        if surround_bg:
            self.lab = self._border(self.lab, thickness)
        self.surround_bg = surround_bg
        if magnification >= 50:
            self.thin_label_distance_thresholds = {'monolayer': 6.0, 'bilayer': 7.0}
            self.thin_label_bg_margin = 0.9
        else:
            self.thin_label_distance_thresholds = {}
            self.thin_label_bg_margin = None

    def _seal_original_frame(self, edges):
        if not self.surround_bg or self.mag < 50:
            return edges
        t = self.thickness
        h, w = self.size
        top = max(t - 1, 0)
        left = max(t - 1, 0)
        bottom = min(t + h, edges.shape[0] - 1)
        right = min(t + w, edges.shape[1] - 1)
        edges[top, left:right + 1] = 1
        edges[bottom, left:right + 1] = 1
        edges[top:bottom + 1, left] = 1
        edges[top:bottom + 1, right] = 1
        return edges

    def make_masks(self):
        native = self.background_support == "native_border_and_joint_foreground_v1"
        if native:
            from qumus_simlab.core.optical_profiles import native_substrate_support

            camera_support = native_substrate_support(self.img, self.observation_noise_scale)
            pad = self.thickness if self.surround_bg else 0
            camera_support = np.pad(camera_support, pad, constant_values=True) if pad else camera_support
        else:
            camera_support = None

        def edgeNtemplate(edge_method):
            support_observation = self._native_operator_support(edge_method) if native else None
            edges = edge_method(self)
            edges = self._seal_original_frame(edges)
            if self.surround_bg:
                t = self.thickness // 2
                edges[:t, :] = 0
                edges[-t:, :] = 0
                edges[:, :t] = 0
                edges[:, -t:] = 0
            template = np.logical_not(edges)
            masks, num_masks = label(template)
            mask_ids, mask_areas = np.unique(masks, return_counts=True)
            if mask_ids.size > 0 and mask_ids[0] == 0:
                mask_areas[0] = 0
            boundary_ids = np.unique(np.concatenate((masks[0, :], masks[-1, :], masks[:, 0], masks[:, -1])))
            boundary_ids = boundary_ids[boundary_ids != 0]
            if self.surround_bg and self.mag >= 50 and (boundary_ids.size > 0):
                flat_masks = masks.ravel()
                component_count = int(masks.max()) + 1
                component_areas = np.bincount(flat_masks, minlength=component_count)
                boundary_id = boundary_ids[np.argmax(component_areas[boundary_ids])]
                component_means = np.column_stack([np.divide(np.bincount(flat_masks, weights=self.lab[:, :, channel].ravel(), minlength=component_count), component_areas, out=np.zeros(component_count, dtype=np.float64), where=component_areas != 0) for channel in range(3)])
                background_reference = component_means[boundary_id]
                if native:
                    # v1's reference is the largest border component, i.e. the
                    # substrate seen at the frame edge. A flake covering one
                    # side of a 100X field can out-size it, so take the
                    # camera-compatible (substrate-like) pixels at the frame
                    # edge as that reference instead.
                    reference = self._native_edge_substrate_lab(camera_support)
                    if reference is not None:
                        background_reference = reference
                background_ids = np.flatnonzero(np.linalg.norm(component_means - background_reference, axis=1) <= 20.0)
                background_ids = background_ids[background_ids != 0]
                if native:
                    # The historical colour radius includes thin hBN anchors
                    # only ~13 LAB units from substrate. An isolated interior
                    # flake must not become background just by matching colour.
                    t, (height, width) = self.thickness, self.size
                    frame_ids, support_observation = self._native_frame_background_ids(masks, background_ids, edge_method)
                    background_ids = np.union1d(boundary_ids, frame_ids)
                    real_support = (np.isin(masks[t:t+height, t:t+width], frame_ids)
                        & camera_support[t:t+height, t:t+width])
                    if not real_support.any():
                        raise ValueError("Native RGB background lacks unambiguous frame/camera support")
                if background_ids.size > 0 and np.any(np.isin(masks[t:-t, t:-t], background_ids)):
                    bg_mask = np.isin(masks, background_ids)
                else:
                    bg_mask_id = mask_ids[np.argmax(mask_areas)]
                    bg_mask = masks == bg_mask_id
            else:
                bg_mask_id = mask_ids[np.argmax(mask_areas)]
                bg_mask = masks == bg_mask_id
            if native:
                # Intersect existing component support with the public camera
                # law. Never dilate/grow it into colour-similar foreground.
                bg_mask &= camera_support
            template[bg_mask] = 0
            return [template, bg_mask, edges, support_observation]
        if len(self.focus_disks) > 0:
            black_zone_mask = self.focus_disks[0][0]
        else:
            black_zone_mask = None
        self.templates = []
        self.bg_masks = []
        self.edge_masks = []
        self.background_support_observations = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(edgeNtemplate, em) for em in self.edge_methods]
            for f in futures:
                r = f.result()
                self.templates.append(r[0])
                self.bg_masks.append(r[1])
                self.edge_masks.append(r[2])
                if r[3] is not None:
                    self.background_support_observations.append(r[3])
        self.bg_mask = self.bg_masks[0]
        self.edges = self.edge_masks[-1]
        if black_zone_mask is not None:
            self.edges[black_zone_mask] = 1
        full_template = self.templates[0]
        for template in self.templates[1:]:
            full_template = np.logical_or(template, full_template)
        self.full_template = full_template
        if self.background_support == "native_border_and_joint_foreground_v1":
            # Foreground is the UNION of operators. Normalizing from the first
            # operator's background can otherwise erase that same foreground.
            self.bg_mask = np.logical_and.reduce(self.bg_masks) & ~full_template
        self.masks, self.num_masks = label(full_template)
        self.mask_ids, self.mask_areas = np.unique(self.masks, return_counts=True)

    def _native_edge_substrate_lab(self, camera_support):
        """Mean LAB of camera-compatible pixels along the real frame edge."""
        t, (height, width) = self.thickness, self.size
        support = camera_support[t:t+height, t:t+width]
        edge = np.zeros_like(support)
        edge[:t, :] = edge[-t:, :] = edge[:, :t] = edge[:, -t:] = True
        pixels = self.lab[t:t+height, t:t+width][edge & support]
        return pixels.mean(axis=0) if len(pixels) else None

    def _native_frame_background_ids(self, masks, colour_ids, edge_method):
        """Colour-matched components that reach an operator-bounded frame edge.

        v1 takes the largest border component as the substrate reference and
        every component within its colour radius as background. This keeps
        that rule but only for components reaching a frame edge, so an
        isolated interior flake never becomes background by colour alone. A
        large flake may cover whole sides of a 100X field (seed 8), so one
        side is enough. The operator's finite halo can hide literal border
        labels; connectivity within a derived edge band handles that halo
        without filling edge pixels or merging labels. Camera compatibility
        remains a separate requirement.
        """
        observation = self._native_operator_support(edge_method)
        band = observation["edge_support_band_px"]
        height, width = self.size
        if 2*band >= min(height, width):
            raise ValueError("Native RGB frame is too small for the edge operator's support")
        t = self.thickness
        frame = masks[t:t+height, t:t+width]
        edge_ids = np.unique(np.concatenate([np.unique(side) for side in
                                             (frame[:band, :], frame[-band:, :], frame[:, :band], frame[:, -band:])]))
        selected = np.intersect1d(np.asarray(colour_ids), edge_ids)
        selected = selected[selected != 0]
        return selected, {**observation, "frame_background_component_count": int(len(selected))}

    def _native_operator_support(self, edge_method):
        args = getattr(edge_method, "args", None)
        if (not isinstance(args, (list, tuple)) or len(args) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in args)
                or int(args[0]) != args[0] or int(args[3]) != args[3] or min(args[0], args[3]) < 1
                or args[1] < 0):
            raise ValueError("Native RGB cannot bound the edge operator's support")
        radius = int(round(3*args[1])) + (int(args[0])//2)*int(args[3])
        band = radius+1
        return {"rule": "native-edge-reaching-component-camera-support-v2",
            "operator_radius_px": radius, "edge_support_band_px": band,
            "observation_noise_scale": self.observation_noise_scale,
            "foreground_and_edge_growth": "none"}

    def get_all_avg_lab(self):
        flat_masks = self.masks.ravel()
        flat_lab = self.lab.reshape(-1, 3)
        num_masks = self.num_masks + 1
        avg_labs = np.zeros((num_masks, 3), dtype=np.float32)
        var_labs = np.zeros((num_masks, 3), dtype=np.float32)
        mask_areas_full = np.zeros(num_masks, dtype=np.float32)
        mask_areas_full[self.mask_ids] = self.mask_areas
        for c in range(3):
            sums = np.bincount(flat_masks, weights=flat_lab[:, c], minlength=num_masks)
            means = np.divide(sums, mask_areas_full, out=np.zeros_like(sums, dtype=np.float32), where=mask_areas_full != 0)
            avg_labs[:, c] = means
        self.avg_labs = avg_labs
        return (avg_labs, var_labs)

    def lab_equalize(self, num_points=2000):
        points = []
        values = []
        if self.background_support == "native_border_and_joint_foreground_v1":
            t = self.thickness if self.surround_bg else 0
            support = self.bg_mask[t:t+self.size[0], t:t+self.size[1]]
            if (int(support.sum()) < num_points or np.count_nonzero(support.any(axis=0)) < 2
                    or np.count_nonzero(support.any(axis=1)) < 2):
                raise ValueError("Native RGB background has insufficient two-dimensional observed support")
            indices = np.flatnonzero(support)
            # Uniform sampling of observed background, directly. This has the
            # donor's conditional sampling distribution without rejection loops
            # when native foreground occupies most of a high-mag frame.
            for _ in range(num_points-4):
                index = int(indices[int(self.rng.randint(low=0, high=len(indices)))])
                y, x = divmod(index, self.size[1])
                y, x = y+t, x+t
                points.append(np.array([y, x]))
                values.append(self.lab[y, x, :])
        return self._equalize_from_samples(points, values, num_points)

    def _equalize_from_samples(self, points, values, num_points):
        while len(points) < 3 * num_points // 4:
            x = self.rng.randint(low=0, high=self.lab.shape[1])
            y = self.rng.randint(low=0, high=self.lab.shape[0])
            if self.bg_mask[y, x] != 0:
                points.append(np.array([y, x]))
                values.append(self.lab[y, x, :])
        if len(self.focus_disks) > 0:
            i = 0
            while len(points) < num_points:
                r = self.focus_disks[0][1]
                theta = i * np.pi / (num_points // 4)
                x = int(r * np.cos(theta)) + self.lab.shape[1] // 2
                y = int(r * np.sin(theta)) + self.lab.shape[0] // 2
                i += 1
                if not ((x < 0 or x >= self.lab.shape[1]) or (y < 0 or y >= self.lab.shape[0])):
                    if self.bg_mask[y, x] != 0:
                        points.append(np.array([y, x]))
                        values.append(self.lab[y, x, :])
                    else:
                        i -= 0.5
            x0 = self.lab.shape[1] // 2 - np.sqrt(np.pow(r, 2) - np.pow(self.lab.shape[0] // 2, 2))
            y0 = 0
            x0 = int(x0)
            x1 = self.lab.shape[1] - x0
            y1 = self.lab.shape[0] - 1
            points.append(np.array([y0, x0]))
            points.append(np.array([y0, x1]))
            points.append(np.array([y1, x0]))
            points.append(np.array([y1, x1]))
            values.append(self.lab[y0, x0, :])
            values.append(self.lab[y0, x1, :])
            values.append(self.lab[y1, x0, :])
            values.append(self.lab[y1, x1, :])
        else:
            while len(points) < num_points - 4:
                x = self.rng.randint(low=0, high=self.lab.shape[1])
                y = self.rng.randint(low=0, high=self.lab.shape[0])
                if self.bg_mask[y, x] != 0:
                    points.append(np.array([y, x]))
                    values.append(self.lab[y, x, :])
            for i in [0, self.lab.shape[0] - 1]:
                for j in [0, self.lab.shape[1] - 1]:
                    points.append(np.array([i, j]))
                    values.append(self.lab[i, j, :])
        points = np.stack(points, axis=0)
        self.interpolation_points = points
        values = np.stack(values, axis=0)
        grid_y, grid_x = np.mgrid[0:self.lab.shape[0], 0:self.lab.shape[1]]
        grid_points = np.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)
        interpolator = LinearNDInterpolator(points, values, fill_value=0, rescale=True)
        lab_bg = interpolator(grid_points).reshape(self.lab.shape[0], self.lab.shape[1], 3)
        l_bg = lab_bg[:, :, 0]
        a_bg = lab_bg[:, :, 1]
        b_bg = lab_bg[:, :, 2]
        l_bg[l_bg < 0] = 0
        self.lab = self.lab.astype('float64')
        self.lab[:, :, 0] -= l_bg - int(self.target_bg_lab[0])
        self.lab[:, :, 1] -= a_bg - int(self.target_bg_lab[1])
        self.lab[:, :, 2] -= b_bg - int(self.target_bg_lab[2])
        if len(self.focus_disks) > 0:
            f2 = self.focus_disks[-1][0]
            self.lab[f2] = 0
        self.l_bg = l_bg
        if self.surround_bg:
            t = self.thickness
            self.lab = self.lab[t:-t, t:-t, :]
            self.edges = self.edges[t:-t, t:-t]
            self.masks = self.masks[t:-t, t:-t]
            self.bg_mask = self.bg_mask[t:-t, t:-t]
            for i in range(len(self.templates)):
                self.templates[i] = self.templates[i][t:-t, t:-t]
                self.edge_masks[i] = self.edge_masks[i][t:-t, t:-t]
        return l_bg

    def adjust_layer_labels(self):
        lightness = np.mean(self.lab[:, :, 0][self.bg_mask])
        a = np.mean(self.lab[:, :, 1][self.bg_mask])
        b = np.mean(self.lab[:, :, 2][self.bg_mask])
        self.avg_bg_lab = np.array([lightness, a, b])
        adjustment_factor = self.avg_bg_lab - np.array(self.target_bg_lab)
        new_layer_labels = {}
        for key, value in self.layer_labels.items():
            new_key = tuple(key + adjustment_factor)
            new_layer_labels[new_key] = value
        self.layer_labels = new_layer_labels
        self.layer_labels[tuple(self.avg_bg_lab)] = 'bg'

    def label_masks(self):
        self.mask_labels = []
        base_labs = np.array(list(self.layer_labels.keys()))
        layer_types = list(self.layer_labels.values())
        layer_types_array = np.array(layer_types, dtype=object)
        dists = np.linalg.norm(self.avg_labs[:, None, :] - base_labs[None, :, :], axis=2)
        min_indices = np.argmin(dists, axis=1)
        bg_indices = np.where(layer_types_array == 'bg')[0]
        for idx, i in enumerate(self.mask_ids):
            area = self.mask_areas[idx]
            if i == 0:
                self.mask_labels.append('bg')
            elif area < self.min_area:
                self.mask_labels.append('dirt')
            else:
                label = layer_types[min_indices[i]]
                if label in self.thin_label_distance_thresholds:
                    nearest_dist = dists[i, min_indices[i]]
                    bg_dist = np.min(dists[i, bg_indices]) if bg_indices.size > 0 else np.inf
                    max_dist = self.thin_label_distance_thresholds[label]
                    if nearest_dist > max_dist or nearest_dist >= self.thin_label_bg_margin * bg_dist:
                        label = 'bg'
                self.mask_labels.append(label)

    def process_frame(self):
        self.make_masks()
        self.lab_equalize()
        self.get_all_avg_lab()
        self.adjust_layer_labels()
        self.label_masks()

    def mask_coords(self, mask_id):
        """
        Returns the (row, col) coordinates of the centroid of the given mask id.
        """
        coords = np.argwhere(self.masks == mask_id)
        if coords.size == 0:
            return None
        centroid = coords.mean(axis=0)
        return np.array(centroid)

    def _border(self, lab, thickness=30, n=30, threshold=10):
        dims = [lab.shape[0] + 2 * thickness, lab.shape[1] + 2 * thickness, lab.shape[2]]
        result = np.ones(dims, dtype=lab.dtype)
        h, w = (lab.shape[0], lab.shape[1])
        hs = [int(i * h / n) for i in range(n)] + [h]
        ws = [int(i * w / n) for i in range(n)] + [w]
        hr = [int(i * result.shape[0] / n) for i in range(n)] + [result.shape[0]]
        wr = [int(i * result.shape[1] / n) for i in range(n)] + [result.shape[1]]
        ms = np.zeros([n, n, lab.shape[2]])
        for i in range(n):
            for j in range(n):
                region = lab[hs[i]:hs[i + 1], ws[j]:ws[j + 1], :]
                ms[i, j, :] = np.median(region, axis=(0, 1))
        ms_m = np.median(ms, axis=(0, 1))
        mask = np.linalg.norm(ms - ms_m, axis=2) > threshold
        ms[mask] = ms_m
        for i in range(n):
            for j in range(n):
                if i == 0 or i == n - 1 or j == 0 or (j == n - 1):
                    result[hr[i]:hr[i + 1], wr[j]:wr[j + 1], :] = ms[i, j, :]
        result[thickness:-thickness, thickness:-thickness, :] = lab
        return result
