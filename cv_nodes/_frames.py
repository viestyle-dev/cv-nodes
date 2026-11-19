"""Helpers for the readers that scale frames before handing them over.

Media Foundation and AVFoundation scale on the GPU to the size from
:func:`decode_size` and hand over BGRA; :func:`cover_crop` and
:func:`bgra_to_bgr` turn that into the node's BGR output. The ffmpeg command
scales, crops and converts in one pass and uses :func:`decode_size` only.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# BGRA channels 0, 1, 2 to BGR channels 0, 1, 2
_BGRA_TO_BGR = [0, 0, 1, 1, 2, 2]


def decode_size(src_w: int, src_h: int, width: int, height: int, fit: str) -> tuple[int, int]:
    """The size to scale to for an output of width x height.

    cover: the smallest size of the source's aspect that covers the output
    (the middle is cut out afterwards); stretch: the output size; exact: the
    source size.
    """
    if fit == "exact":
        return src_w, src_h
    if fit == "stretch":
        return width, height
    scale = max(width / src_w, height / src_h)
    return max(width, math.ceil(src_w * scale - 1e-6)), max(height, math.ceil(src_h * scale - 1e-6))


def cover_crop(decoded: tuple[int, int], width: int, height: int, fit: str) -> tuple[int, int] | None:
    """Top-left corner of the output inside a decoded frame (cover), or None when it is used whole."""
    dw, dh = decoded
    if fit != "cover" or (dw, dh) == (width, height) or dw < width or dh < height:
        return None
    return (dw - width) // 2, (dh - height) // 2


def bgra_to_bgr(bgra: np.ndarray) -> np.ndarray:
    """A new BGR array from a BGRA view (rows may be strided).

    ``cv2.mixChannels`` runs on the calling thread; ``cvtColor`` would split a
    640x360 frame over OpenCV's thread pool, which costs ~2 ms of CPU per
    frame in waiting threads (Windows) for ~0.1 ms of work.
    """
    out = np.empty(bgra.shape[:2] + (3,), np.uint8)
    cv2.mixChannels([bgra], [out], _BGRA_TO_BGR)
    return out
