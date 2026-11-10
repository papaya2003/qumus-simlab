"""Unretuned Gr/hBN material tables and edge operators from frozen v1.

Donor material.py SHA256: 08fb82921e4cd50e59b2d9b0d46da8783aed251e194c9f123e29e627869c989c
These tables retain their historical magnification split and are not a claim
of measured accuracy on the current camera or synthetic palette.
"""
import concurrent.futures

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny

from qumus_simlab.perception._v1_rgb_core import Segmenter


def blur(img, sigma):
    ksize = int(2 * round(3 * sigma) + 1) if sigma > 0 else 1
    if ksize % 2 == 0:
        ksize += 1
    if sigma == 0:
        return img.copy()
    channels = []
    if len(img.shape) == 3:
        num_channels = img.shape[-1]
    else:
        num_channels = 1
    for i in range(num_channels):
        channels.append(cv2.GaussianBlur(img[:, :, i], (ksize, ksize), sigmaX=sigma, sigmaY=sigma))
    return np.stack(channels, axis=-1)

class EdgeMethod:

    def __init__(self, canny_sigma=0.7):
        self.canny_sigma = canny_sigma

    def __call__(self, img):
        return canny(rgb2gray(img), sigma=self.canny_sigma).astype('float32')

def rank_gradient(image, selem, dilate_iters=1, erode_iters=1):
    """
    Mimics skimage.filters.rank.gradient using OpenCV and NumPy.

    Args:
        image (np.ndarray): Grayscale input image (uint8).
        selem (np.ndarray): Structuring element (binary mask, e.g., from skimage.morphology.disk).

    Returns:
        np.ndarray: Gradient image (uint8).
    """
    kernel = selem.astype(np.uint8)
    local_max = cv2.dilate(image, kernel, iterations=dilate_iters)
    local_min = cv2.erode(image, kernel, iterations=erode_iters)
    gradient = local_max - local_min
    return gradient

class RankGradient(EdgeMethod):

    def __init__(self, args, k=3, sigma=0, threshold=None):
        self.k = k
        self.args = [] if args is None else args
        self.sigma = sigma
        self.threshold = threshold

    def __call__(self, seg_or_img, apply_threshold=True):
        """
        Input 2K or 4K resolution images for best results.
        """
        r, b, threshold, dilate_iterations = self.args
        if type(seg_or_img) is Segmenter:
            blurred = blur(seg_or_img.lab.copy(), b)
        else:
            blurred = blur(seg_or_img.copy(), b)
        footprint = np.ones((r, r))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(rank_gradient, blurred[:, :, i], footprint, dilate_iterations, dilate_iterations) for i in range(3)]
            channel_gradients = [f.result() for f in futures]
        edges = channel_gradients[0] + channel_gradients[1] + channel_gradients[2]
        if apply_threshold:
            return edges > threshold
        else:
            return edges

class Material:

    def __init__(self, name, target_bg_lab, layer_labels, sigma=0.7, fat_threshold=0.1, Edge_Method=EdgeMethod, args=None, high_mag_target_bg_lab=None, high_mag_layer_labels=None):
        self.name = name
        self.target_bg_lab = target_bg_lab
        self.layer_labels = layer_labels
        self.high_mag_target_bg_lab = high_mag_target_bg_lab
        self.high_mag_layer_labels = high_mag_layer_labels
        self.recommended_sigma = sigma
        self.recommended_fat_threshold = fat_threshold
        self.Edge_Method = Edge_Method
        self.args = [] if args is None else args

    def target_bg_lab_for_mag(self, magnification):
        if self.high_mag_target_bg_lab is not None and magnification >= 50:
            return self.high_mag_target_bg_lab
        return self.target_bg_lab

    def layer_labels_for_mag(self, magnification):
        if self.high_mag_layer_labels is not None and magnification >= 50:
            return self.high_mag_layer_labels
        return self.layer_labels
graphene_labels = {(46.6, 11, 4.2): 'monolayer', (43.9, 18.9, 1.6): 'bilayer', (40.4, 23.6, -0.6): 'trilayer', (30, 20, -20): 'fewlayer', (43.5, 8, 1): 'fewlayer', (34.4, 25.2, -19.2): 'manylayer', (50, 0, -8): 'manylayer', (53, 0, 0): 'manylayer', (75, 0, 35): 'bulk', (80, 5, 10): 'bulk', (35, 1, 36): 'bulk', (48, 10, 3.5): 'dirt', (48, 9, 2.5): 'dirt', (48.1, 6.7, -1.4): 'dirt', (53, 0, 8.5): 'dirt', (39, 8, 8): 'dirt', (0, 0, 0): 'black'}
graphene_labels_100x = {(46.0, 16.2, 0.4): 'monolayer', (43.9, 20.0, -1.2): 'bilayer', (40.4, 23.6, -0.6): 'trilayer', (30, 20, -20): 'fewlayer', (43.5, 8, 1): 'fewlayer', (24.4, 42.2, -41.5): 'manylayer', (50, 0, -8): 'manylayer', (53, 0, 0): 'manylayer', (75, 0, 35): 'bulk', (80, 5, 10): 'bulk', (35, 1, 36): 'bulk', (48, 10, 3.5): 'dirt', (48, 9, 2.5): 'dirt', (48.1, 6.7, -1.4): 'dirt', (53, 0, 8.5): 'dirt', (39, 8, 8): 'dirt', (0, 0, 0): 'black'}
graphene = Material('Graphene', [49.62793933, 12.17574073, 5.43972594], layer_labels=graphene_labels, high_mag_layer_labels=graphene_labels_100x, sigma=3, fat_threshold=0.005, Edge_Method=RankGradient, args=[[5, 3, 5, 6], [2, 2, 5, 2], [4, 3, 5, 1], [3, 3, 4, 5], [4, 3, 4, 4], [3, 3, 5, 3]])
hBN_labels = {(50, 20, -2): '0-10', (48, 15, -9): '10-20', (70.7, -31, 24.2): '20-30', (90.7, -45, 20): '30-40', (93, -54, 53): '30-40', (77, -17, 42): '40+', (85, -40, 55): '40+', (100, -30, 100): '40+', (78, 22, -7): '40+', (90, 25, 50): '40+', (60, 20, 12): '40+', (30, 20, -10): 'dirt', (0, 0, 0): 'black'}
hBN_labels_100x = {(50, 20, -2): '0-10', (48, 15, -9): '10-20', (56, -16, -15): '20-30', (81, -41, 1): '30-40', (93, -26, 18): '30-40', (93, -50, 42): '30-40', (89, -43, 39): '30-40', (100, -30, 30): '40+', (80, -57, 65): '40+', (78, 22, -7): '40+', (90, 25, 50): '40+', (60, 20, -12): '40+', (30, 20, -10): 'dirt', (0, 0, 0): 'black'}
hBN = Material('hBN', [49, 27.5, 8.5], layer_labels=hBN_labels, high_mag_layer_labels=hBN_labels_100x, sigma=0.7, fat_threshold=0.1, Edge_Method=RankGradient, args=[[2, 1.5, 6, 14], [3, 2, 5, 3], [2, 2, 5, 4], [2, 1, 5, 1]])
