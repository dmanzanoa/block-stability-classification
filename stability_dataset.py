"""
stability_dataset.py
====================

This module provides utilities for loading and preprocessing the
ShapeStacks dataset used for predicting the stable height of a stack of
blocks.  The dataset is supplied as images along with a CSV file
containing an `id` column and a `stable_height` label.  Functions are
provided to load images and labels from disk and to optionally apply
foreground segmentation to each image.  Segmentation can help focus
the downstream model on the stack of blocks rather than the
background.

The implementation is agnostic to the overall machine learning
framework; it returns NumPy arrays that can be passed to either
TensorFlow/Keras, PyTorch, or any other library.  Real-world systems
should perform further preprocessing such as data augmentation,
normalisation and batching.
"""

from __future__ import annotations

import os
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd


def load_dataset(csv_path: str, img_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and labels from a CSV and an image directory.

    The CSV file is expected to contain at least two columns:
    ``id`` which corresponds to the basename (without extension) of
    each image file, and ``stable_height`` which is an integer label.
    Image files are assumed to have the `.jpg` extension.  Images are
    loaded in RGB order and returned as floating point arrays.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing image identifiers and labels.
    img_dir : str
        Directory containing the image files.

    Returns
    -------
    tuple of ndarray
        A tuple ``(images, labels)`` where ``images`` is an array of
        shape ``(N, H, W, 3)`` and dtype ``float32``, and ``labels`` is
        an array of shape ``(N,)`` and dtype ``int``.
    """
    df = pd.read_csv(csv_path)
    images: List[np.ndarray] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        img_id = str(row["id"])
        label = int(row["stable_height"])
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{img_path}' could not be loaded")
        # Convert BGR (OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        images.append(img)
        labels.append(label)
    return np.stack(images, axis=0), np.array(labels, dtype=np.int64)


def grabcut_segmentation(image: np.ndarray, iter_count: int = 5) -> np.ndarray:
    """Apply GrabCut foreground extraction to an image.

    This function applies the GrabCut algorithm to separate the stack of
    blocks from the background.  A coarse rectangular mask is
    initialised automatically assuming the stack occupies the central
    portion of the image.  The resulting mask is returned as a binary
    array of the same height and width as the input image.  GrabCut
    requires OpenCV to be built with the ``gc`` module available.

    Parameters
    ----------
    image : ndarray
        Input image array in RGB order.
    iter_count : int, optional
        Number of iterations for the GrabCut algorithm (default 5).

    Returns
    -------
    ndarray
        Binary mask indicating the foreground region.
    """
    h, w = image.shape[:2]
    # Initialise mask: 0 = background, 2 = probable background, 1 = probable
    # foreground, 3 = foreground.  Start with all probable background.
    mask = np.zeros((h, w), np.uint8)
    # Define a rectangle covering the central region where the stack is
    # likely to appear.  The margins can be tuned.
    rect = (int(0.1 * w), int(0.05 * h), int(0.8 * w), int(0.9 * h))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR), mask, rect,
                bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
    # Convert mask to binary: foreground and probable foreground as 1, else 0
    mask_out = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    return mask_out


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to an image.

    The masked-out pixels are set to zero.  This utility is useful for
    preprocessing images prior to feeding them into a CNN.

    Parameters
    ----------
    image : ndarray
        Input RGB image.
    mask : ndarray
        Binary mask of the same height and width as ``image``.

    Returns
    -------
    ndarray
        Masked RGB image.
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must have the same spatial dimensions")
    # Broadcast mask to 3 channels and multiply
    return image * mask[:, :, np.newaxis]
