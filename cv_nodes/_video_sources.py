"""Video file readers for VideoFileCapture (used by its worker thread only).

Readers that scale before the frame reaches Python (the node's output size,
so a 4K clip costs about what its output does):

- ``media_foundation`` (Windows): Media Foundation's Source Reader; the GPU
  decodes and scales (see :mod:`._media_foundation`)
- ``avfoundation`` (macOS): an AVAssetReader; VideoToolbox decodes and scales
  (see :mod:`._avfoundation`; needs PyObjC)
- ``ffmpeg_command`` (any OS with ``ffmpeg`` on PATH; Linux and Raspberry Pi):
  ffmpeg decodes, on a hardware decoder when there is one, and scales and
  converts in one pass (see :mod:`._ffmpeg_command`)

and one that does not:

- ``opencv_ffmpeg``: OpenCV's bundled FFmpeg (LGPL 2.1), on every platform.
  Frames come at the file's size and the node fits them: a 4K frame costs a
  full BGR conversion and resize on the CPU. (Formerly ``ffmpeg``; that name
  is still accepted.)

``auto`` tries them in the order of :data:`_AUTO` for the platform; a reader
that cannot open a file or decode its first frame gives way to the next.
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "media_foundation", "avfoundation", "ffmpeg_command", "opencv_ffmpeg")
_OLD_NAMES = {"ffmpeg": "opencv_ffmpeg"}

# Readers that auto tries, best first.
_AUTO = {
    "win32": ("media_foundation", "ffmpeg_command", "opencv_ffmpeg"),
    "darwin": ("avfoundation", "ffmpeg_command", "opencv_ffmpeg"),
}
_AUTO_ELSEWHERE = ("ffmpeg_command", "opencv_ffmpeg")

Output = tuple[int, int, str]  # width, height, fit


def backend_name(name: str) -> str:
    """The current name of a backend (old names are mapped); ValueError when unknown."""
    name = _OLD_NAMES.get(name, name)
    if name not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    return name


class VideoSource:
    """The frames of one file, in order."""

    backend = ""

    def __init__(self, path: str, width: int, height: int, fps: float, frame_count: int):
        self.path = path
        self.width = width  # the file's size
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self._next = 0  # index of the frame the next read returns

    def read(self, loop: bool, output: Output) -> tuple[np.ndarray | None, bool]:
        """The next frame, and whether it wrapped to the first frame.

        Returns (None, False) at the end of the file when ``loop`` is off.
        """
        frame = self._read(output)
        if frame is None and loop and self._next > 0:
            if not self.seek(0):
                return None, False
            frame = self._read(output)
            return frame, frame is not None
        return frame, False

    def _read(self, output: Output) -> np.ndarray | None:
        raise NotImplementedError

    def seek(self, index: int) -> bool:
        """Continue from frame ``index``. False when the file cannot be read any more."""
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class _OpenCVSource(VideoSource):
    """OpenCV's FFmpeg reader: BGR frames at the file's size."""

    backend = "opencv_ffmpeg"

    def __init__(self, path: str, cap: cv2.VideoCapture, first: np.ndarray | None):
        super().__init__(
            path,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self._cap = cap
        self._first = first  # read when opening (to know it decodes)
        self._ended = first is None

    def _read(self, output: Output) -> np.ndarray | None:
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
        index = max(0, index)
        self._first = None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self._next = index
        self._ended = False
        return True

    def release(self) -> None:
        self._cap.release()


class _ScaledSource(VideoSource):
    """A reader that delivers frames of the output size (Media Foundation, AVFoundation, ffmpeg).

    The reader has the file's ``width``, ``height``, ``fps`` and
    ``frame_count``, and ``configure(width, height, fit) -> bool`` (True when
    the output changed), ``read()`` (the next frame, None at the end),
    ``seek(index)``, ``close()`` and, optionally, ``prepare_rewind()`` (get
    ready to read from the first frame: called shortly before the end when
    looping).
    """

    _REWIND_AHEAD = 3  # frames before the end at which prepare_rewind() is called

    def __init__(self, path: str, backend: str, reader, first: np.ndarray | None):
        super().__init__(path, reader.width, reader.height, reader.fps, reader.frame_count)
        self.backend = backend
        self._reader = reader
        self._first = first
        self._ended = first is None
        self._prepare_rewind = getattr(reader, "prepare_rewind", None)

    def read(self, loop: bool, output: Output) -> tuple[np.ndarray | None, bool]:
        prepare = self._prepare_rewind
        if loop and prepare is not None and 0 < self.frame_count - self._REWIND_AHEAD <= self._next:
            prepare()
        return super().read(loop, output)

    def _read(self, output: Output) -> np.ndarray | None:
        first = self._first
        if first is not None:
            self._first = None
            self._next += 1
            return first
        if self._ended:
            return None
        if self._reader.configure(*output) and self._next > 0:
            self._reader.seek(self._next)  # a new output starts where the last one ended
        frame = self._reader.read()
        if frame is None:
            self._ended = True
            return None
        self._next += 1
        return frame

    def seek(self, index: int) -> bool:
        index = max(0, index)
        self._first = None
        self._reader.seek(index)
        self._next = index
        self._ended = False
        return True

    def release(self) -> None:
        self._reader.close()


def _open_scaled(path: str, output: Output, backend: str, make_reader) -> VideoSource | None:
    try:
        reader = make_reader(path)
    except OSError as e:
        logger.info(f"{backend} cannot open {path}: {e}")
        return None
    try:
        reader.configure(*output)
        first = reader.read()
    except OSError as e:
        logger.info(f"{backend} cannot decode {path}: {e}")
        reader.close()
        return None
    if first is None:
        reader.close()  # no frame: let the next reader try
        return None
    if not reader.gpu:
        logger.info(f"{backend} decodes {path} on the CPU (no hardware decoder for it)")
    return _ScaledSource(path, backend, reader, first)


def _open_media_foundation(path: str, output: Output) -> VideoSource | None:
    if sys.platform != "win32":
        return None
    from . import _media_foundation

    return _open_scaled(path, output, "media_foundation", _media_foundation.Reader)


def _open_avfoundation(path: str, output: Output) -> VideoSource | None:
    if sys.platform != "darwin":
        return None
    try:
        from . import _avfoundation
    except ImportError as e:  # PyObjC missing
        logger.info(f"avfoundation needs PyObjC (pyobjc-framework-AVFoundation): {e}")
        return None
    return _open_scaled(path, output, "avfoundation", _avfoundation.Reader)


def _open_ffmpeg_command(path: str, output: Output) -> VideoSource | None:
    from . import _ffmpeg_command

    if _ffmpeg_command.executables() is None:
        return None
    return _open_scaled(path, output, "ffmpeg_command", _ffmpeg_command.Reader)


def _open_opencv_ffmpeg(path: str, output: Output) -> VideoSource | None:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    ok, first = cap.read()
    return _OpenCVSource(path, cap, first if ok else None)


_OPENERS = {
    "media_foundation": _open_media_foundation,
    "avfoundation": _open_avfoundation,
    "ffmpeg_command": _open_ffmpeg_command,
    "opencv_ffmpeg": _open_opencv_ffmpeg,
}


def open_source(path: str, backend: str, output: Output) -> VideoSource | None:
    """Open ``path`` for frames of ``output`` with ``backend``.

    ``auto`` tries the platform's readers best first; another backend falls
    back to OpenCV's FFmpeg when it cannot read the file. Returns None when
    nothing opens it.
    """
    backend = backend_name(backend)
    if backend == "auto":
        candidates = _AUTO.get(sys.platform, _AUTO_ELSEWHERE)
    elif backend == "opencv_ffmpeg":
        candidates = (backend,)
    else:
        candidates = (backend, "opencv_ffmpeg")
    for i, name in enumerate(candidates):
        source = _OPENERS[name](path, output)
        if source is not None:
            return source
        if i + 1 < len(candidates):
            logger.info(f"{name} cannot read {path}; trying {candidates[i + 1]}")
    return None
