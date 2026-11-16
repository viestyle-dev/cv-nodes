"""Video file readers for VideoFileCapture (used by its worker thread only).

Two OpenCV backends:

- ``media_foundation`` (Windows): the OS decoders, which decode in hardware
  and convert the colors on the GPU. A 4K H.264 frame costs about a third of
  the CPU time of FFmpeg (17-26 ms against 44-54 ms, scaled to 640x360).
- ``ffmpeg``: OpenCV's bundled FFmpeg, on every platform.

Media Foundation reads nothing once it has hit the end of a file, whatever
the seek, and reopening takes ~0.3 s. :class:`VideoSource` keeps that inside:
looping rewinds right after the last frame (before the end is hit), and a
seek after the end reopens the file.
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "ffmpeg", "media_foundation")

_APIS = {"ffmpeg": cv2.CAP_FFMPEG, "media_foundation": cv2.CAP_MSMF}


def _capture(path: str, backend: str) -> cv2.VideoCapture | None:
    if backend == "media_foundation" and sys.platform != "win32":
        return None
    cap = cv2.VideoCapture(path, _APIS[backend])
    if cap.isOpened():
        return cap
    cap.release()
    return None


class VideoSource:
    """The frames of one file, in order: BGR at the file's size."""

    __slots__ = ("path", "backend", "width", "height", "fps", "frame_count", "_cap", "_next", "_ended", "_first")

    def __init__(self, path: str, backend: str, cap: cv2.VideoCapture, first: np.ndarray | None):
        self.path = path
        self.backend = backend
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cap = cap
        self._next = 0  # index of the frame the next read returns
        self._ended = first is None  # the capture has hit the end of the file
        self._first = first  # read when opening (to know the backend can decode it)

    def read(self, loop: bool) -> tuple[np.ndarray | None, bool]:
        """The next frame, and whether it wrapped to the first frame.

        Returns (None, False) at the end of the file when ``loop`` is off.
        """
        wrapped = False
        if loop and self.backend == "media_foundation" and 0 < self.frame_count <= self._next:
            # Rewind after the last frame, before the end is hit (see module doc).
            self.seek(0)
            wrapped = True
        frame = self._read()
        if frame is None and loop and not wrapped and self._next > 0:
            self.seek(0)
            wrapped = True
            frame = self._read()
        return frame, wrapped and frame is not None

    def _read(self) -> np.ndarray | None:
        first = self._first
        if first is not None:
            self._first = None
            self._next += 1
            return first
        if self._ended:
            return None
        ok, frame = self._cap.read()
        if not ok:
            self._ended = True
            return None
        self._next += 1
        return frame

    def seek(self, index: int) -> bool:
        """Continue from frame ``index``. False when the file cannot be reopened."""
        index = max(0, index)
        if self._ended and self.backend == "media_foundation":
            cap = _capture(self.path, self.backend)
            if cap is None:
                logger.error(f"Cannot reopen {self.path} with {self.backend}")
                return False
            self._cap.release()
            self._cap = cap
        self._first = None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self._next = index
        self._ended = False
        return True

    def release(self):
        self._cap.release()


def open_source(path: str, backend: str) -> VideoSource | None:
    """Open ``path``, preferring ``backend`` and falling back to FFmpeg.

    ``auto`` prefers Media Foundation on Windows and FFmpeg elsewhere. A
    preferred backend that cannot open the file or decode its first frame (a
    codec the OS lacks) gives way to FFmpeg. Returns None when nothing opens it.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    preferred = ("media_foundation" if sys.platform == "win32" else "ffmpeg") if backend == "auto" else backend
    candidates = (preferred,) if preferred == "ffmpeg" else (preferred, "ffmpeg")
    for name in candidates:
        cap = _capture(path, name)
        if cap is None:
            continue
        ok, first = cap.read()
        if ok:
            return VideoSource(path, name, cap, first)
        if name == candidates[-1]:
            return VideoSource(path, name, cap, None)  # opens but has no frame: at the end at once
        cap.release()
        logger.info(f"{name} cannot decode {path}; trying {candidates[-1]}")
    return None
