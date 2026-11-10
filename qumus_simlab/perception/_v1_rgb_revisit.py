"""Maintained pure LiveSegment centered-revisit recovery; no donor runtime import.

Donor livesegment.py SHA256 cdacf697cd2b7b5ad246d57549d467e1e0c7f322a51ce0cc8622d0b94b3a467f.
Thresholds, ROI, morphology and selection follow the donor. The high-mag cluster
initializer uses a private RandomState and deterministic Lloyd updates instead
of changing OpenCV's process-global RNG; this distinction is versioned explicitly.
"""
import math

import cv2
import numpy as np

RECOVERY_VERSION = "v1-centered-recovery-private-kmeans-roi-v2"


def _private_kmeans(samples, count):
    """Fixed seed/5 attempts/50 iterations/.05 tolerance, private kmeans++."""
    rng = np.random.RandomState(3701)
    best = None
    for _ in range(5):
        centers = [samples[rng.randint(len(samples))].copy()]
        closest = np.sum((samples-centers[0])**2, axis=1)
        for _ in range(1, count):
            total = float(closest.sum(dtype=np.float64))
            if total <= 0:
                centers.append(centers[-1].copy())
                continue
            cumulative = np.cumsum(closest, dtype=np.float64)
            trials = []
            for _ in range(3):
                index = min(int(np.searchsorted(cumulative, rng.random_sample()*total)), len(samples)-1)
                distance = np.minimum(closest, np.sum((samples-samples[index])**2, axis=1))
                trials.append((float(distance.sum(dtype=np.float64)), index, distance))
            _, index, closest = min(trials, key=lambda item: (item[0], item[1]))
            centers.append(samples[index].copy())
        centers = np.asarray(centers, dtype=np.float32)
        for _ in range(50):
            distances = np.stack([np.sum((samples-center)**2, axis=1) for center in centers], axis=1)
            labels = np.argmin(distances, axis=1).astype(np.int32)
            updated = []
            for i in range(count):
                selected = samples[labels == i]
                updated.append(selected.mean(axis=0) if len(selected) else samples[np.argmax(distances.min(axis=1))])
            updated = np.asarray(updated, dtype=np.float32)
            displacement = float(np.max(np.sum((updated-centers)**2, axis=1)))
            centers = updated
            if displacement <= .05**2:
                break
        distances = np.stack([np.sum((samples-center)**2, axis=1) for center in centers], axis=1)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        score = float(distances[np.arange(len(samples)), labels].sum(dtype=np.float64))
        if best is None or score < best[0]:
            best = score, labels, centers.copy()
    return best[1], best[2]


class CenteredRevisitRecovery:
    def __init__(self, material, material_type, magnification, minimum_area, maximum_area):
        self._material = material
        self.material_type = material_type
        self.magnification = magnification
        self.min_area, self.max_area = minimum_area, maximum_area

    def _is_graphene(self):
        normalized = str(self.material_type or "").strip().lower()
        return normalized in {"gr", "graphene"}

    def _center_color_fallback(self, img_rgb, accepted_labels=None):
        """Recover a centered graphene revisit flake by normalized LAB color.

        Native 20X captures use a deterministic residual-color mask. At 100X,
        where the target can fill most of the frame and touch its borders, LAB
        clustering is more reliable. Both paths use side substrate as the
        background reference and retain only the component nearest the center.
        Ordinary graphene revisits recover monolayer/bilayer targets; callers
        can restrict recovery to another calibrated graphene-family label for
        purpose-specific captures such as a many-layer graphite gate.
        """
        if not self._is_graphene():
            return None

        if accepted_labels is None:
            accepted_labels = {"monolayer", "bilayer"}
        else:
            accepted_labels = {
                str(label).strip().lower() for label in accepted_labels
            }

        height, width = img_rgb.shape[:2]
        shrink = 4 if min(height, width) >= 400 else 1
        small_width = max(1, width // shrink)
        small_height = max(1, height // shrink)
        small_rgb = cv2.resize(
            img_rgb,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )
        lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] *= 100.0 / 255.0
        lab[:, :, 1:] -= 128.0

        target_bg = np.asarray(
            self._material.target_bg_lab_for_mag(self.magnification),
            dtype=float,
        )
        # The revisit-camera color calibration is shared by the 20X and 100X
        # captures even though their edge profiles must remain different.
        color_profile_mag = 100 if self.magnification == 20 else self.magnification
        reference_rows = []
        for reference_lab, label in self._material.layer_labels_for_mag(
            color_profile_mag
        ).items():
            if label in accepted_labels:
                reference_rows.append((np.asarray(reference_lab, dtype=float), label))
        if not reference_rows:
            return None

        target_label = None
        if self.magnification < 50:
            # Build a row-wise substrate estimate from the two side bands. This
            # removes the strong illumination/color gradient in native 20X
            # captures without sampling the centered flake itself.
            x0, x1 = 0, small_width
            band_width = max(8, int(round(0.12 * small_width)))
            left_bg = np.median(lab[:, :band_width], axis=1)
            right_bg = np.median(lab[:, -band_width:], axis=1)
            smooth_height = min(101, small_height if small_height % 2 else small_height - 1)
            smooth_height = max(1, smooth_height)
            left_bg = cv2.GaussianBlur(
                left_bg[:, None, :], (1, smooth_height), 0
            )[:, 0, :]
            right_bg = cv2.GaussianBlur(
                right_bg[:, None, :], (1, smooth_height), 0
            )[:, 0, :]
            alpha = np.linspace(0.0, 1.0, small_width, dtype=np.float32)[None, :, None]
            local_bg = left_bg[:, None, :] * (1.0 - alpha) + right_bg[:, None, :] * alpha
            residual = lab - local_bg
            reference_deltas = np.stack(
                [reference_lab - target_bg for reference_lab, _ in reference_rows],
                axis=0,
            )
            distances = np.linalg.norm(
                residual[None, :, :, :] - reference_deltas[:, None, None, :],
                axis=3,
            )
            projections = np.stack(
                [
                    np.sum(residual * delta, axis=2) / max(float(np.sum(delta ** 2)), 1e-6)
                    for delta in reference_deltas
                ],
                axis=0,
            )
            binary = (
                (np.min(distances, axis=0) < 4.5)
                & (np.max(projections, axis=0) > 0.30)
            ).astype(np.uint8)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                np.ones((7, 7), dtype=np.uint8),
                iterations=2,
            )
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
            )
        else:
            # Side margins normally contain bare substrate even when a large
            # 100X flake reaches the top and bottom of the image.
            x0 = int(round(0.20 * small_width))
            x1 = int(round(0.80 * small_width))
            if x1 - x0 < 8:
                return None
            roi = lab[:, x0:x1]
            samples = np.ascontiguousarray(roi.reshape(-1, 3), dtype=np.float32)
            cluster_count = min(4, len(samples))
            if cluster_count < 2:
                return None
            labels, centers = _private_kmeans(samples, cluster_count)
            labels = labels.reshape(roi.shape[:2])
            strip_width = max(2, int(round(0.04 * labels.shape[1])))
            side_labels = np.concatenate(
                (labels[:, :strip_width].ravel(), labels[:, -strip_width:].ravel())
            )
            bg_cluster = int(
                np.bincount(side_labels, minlength=cluster_count).argmax()
            )
            bg_lab = centers[bg_cluster]
            cluster_choices = []
            for cluster_idx, center_lab in enumerate(centers):
                if cluster_idx == bg_cluster:
                    continue
                corrected_lab = center_lab - bg_lab + target_bg
                reference_lab, label = min(
                    reference_rows,
                    key=lambda row: float(np.linalg.norm(corrected_lab - row[0])),
                )
                color_distance = float(np.linalg.norm(corrected_lab - reference_lab))
                contrast = float(np.linalg.norm(center_lab - bg_lab))
                max_distance_by_label = {
                    "monolayer": 6.0,
                    "bilayer": 7.0,
                    # Thick graphite has substantially wider optical contrast
                    # than atomically thin graphene. This still requires a
                    # strong separation from the learned substrate cluster.
                    "manylayer": 18.0,
                }
                max_distance = max_distance_by_label.get(label, 7.0)
                if color_distance <= max_distance and contrast >= 2.5:
                    cluster_choices.append(
                        (color_distance, -contrast, cluster_idx, label)
                    )
            if not cluster_choices:
                return None
            _, _, target_cluster, target_label = min(cluster_choices)
            binary = (labels == target_cluster).astype(np.uint8)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                np.ones((3, 3), dtype=np.uint8),
            )

        component_count, component_labels, stats, centroids = (
            cv2.connectedComponentsWithStats(binary, connectivity=8)
        )
        roi_center = np.asarray(
            [small_height / 2.0, small_width / 2.0 - x0],
            dtype=float,
        )
        components = []
        for component_idx in range(1, component_count):
            area_small = int(stats[component_idx, cv2.CC_STAT_AREA])
            area_estimate = float(area_small * shrink * shrink)
            if not (self.min_area < area_estimate < self.max_area):
                continue
            left = int(stats[component_idx, cv2.CC_STAT_LEFT])
            comp_width = int(stats[component_idx, cv2.CC_STAT_WIDTH])
            if left == 0 and left + comp_width >= binary.shape[1]:
                continue
            centroid = np.asarray(
                [centroids[component_idx, 1], centroids[component_idx, 0]],
                dtype=float,
            )
            center_distance = float(np.linalg.norm(centroid - roi_center))
            components.append((center_distance, -area_small, component_idx))
        if not components:
            return None

        center_distance, _, component_idx = min(components)
        if center_distance > 0.35 * min(small_height, small_width):
            return None

        small_mask = np.zeros((small_height, small_width), dtype=np.uint8)
        small_mask[:, x0:x1][component_labels == component_idx] = 1
        mask = cv2.resize(
            small_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        area = float(np.count_nonzero(mask))
        if not (self.min_area < area < self.max_area):
            return None
        centroid = np.argwhere(mask).mean(axis=0)
        selected_small = component_labels == component_idx
        selected_y, selected_x = np.nonzero(selected_small)
        # Closing is dilation+erosion; opening adds another erosion+dilation.
        # Include one working pixel for the area/nearest raster derivation.
        # This is an observation-support guard, not a boundary-error estimate.
        working_guard = 3 if self.magnification >= 50 else 15
        scale_x, scale_y = width/small_width, height/small_height
        roi_clipped = (selected_x.min() < working_guard or selected_y.min() < working_guard
            or selected_x.max() >= selected_small.shape[1]-working_guard
            or selected_y.max() >= selected_small.shape[0]-working_guard)
        support = {"producer": "live_segment_center_color_recovery", "version": RECOVERY_VERSION,
            "source_size_px": [width, height], "working_size_px": [small_width, small_height],
            "working_roi_px": [x0, 0, x1, small_height],
            "support_roi_px": [math.ceil(x0*scale_x), 0, math.floor(x1*scale_x), height],
            "working_to_source_scale_px": [scale_x, scale_y],
            "operator_guard_working_px": working_guard,
            "operator_guard_native_px": max(scale_x, scale_y)*working_guard,
            "morphology": "close3x3" if self.magnification >= 50 else "close7x7_iterations2_then_open3x3",
            "whole_body_status": "unsupported" if roi_clipped else "eligible_within_declared_roi",
            "whole_body_reason": "centered_recovery_roi_clipped_or_unobserved" if roi_clipped else None}
        if target_label is None:
            mean_residual = np.mean(residual[selected_small], axis=0)
            corrected_lab = mean_residual + target_bg
            _, target_label = min(
                reference_rows,
                key=lambda row: float(np.linalg.norm(corrected_lab - row[0])),
            )
        return mask, {
            "mask_id": -1,
            "label": target_label,
            "area": area,
            "centroid": np.asarray(centroid, dtype=float),
            "source": "center_color_fallback",
            "support": support,
        }

    @staticmethod
    def _is_centered_monolayer_recovery(candidate, frame_size, max_center_fraction=0.12):
        """Return whether color recovery found a credible centered monolayer.

        A mixed-thickness graphene revisit can contain a valid thicker model
        candidate as well as the intended monolayer.  Previously, the thicker
        candidate prevented color recovery from running at all.  Restrict the
        override to an actual monolayer component inside a tight central zone
        so graphite-only revisit targets keep their normal model result.
        """
        if candidate is None or candidate.get("label") != "monolayer":
            return False
        centroid = np.asarray(candidate.get("centroid"), dtype=float)
        if centroid.shape != (2,) or not np.all(np.isfinite(centroid)):
            return False
        frame_center = np.asarray(frame_size, dtype=float) / 2.0
        center_distance = float(np.linalg.norm(centroid - frame_center))
        return center_distance <= float(max_center_fraction) * min(frame_size)
