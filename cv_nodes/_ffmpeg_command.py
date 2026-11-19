"""The ``ffmpeg`` command as a frame reader (used by _video_sources).

ffmpeg decodes (on the hardware decoder when there is one), scales the YUV
frames to the output size and converts them to BGR in one pass, and writes
raw frames to a pipe. Only the small frame reaches Python: 4K H.264 read for
640x360 takes ~12 ms of CPU per frame with a hardware decoder and ~32 ms
without, against ~64 ms for OpenCV's FFmpeg, which converts the full frame
before it can be resized (measured on a laptop with Windows).

The command is the user's (``sudo apt install ffmpeg``, Homebrew, winget):
it is not shipped, and runs as a separate program, so its build's license
does not extend to the application.

Decoders tried, in order, until the first frame decodes:

- Linux: HEVC with ``-hwaccel drm`` (Raspberry Pi 4 and 5), H.264 with
  ``h264_v4l2m2m`` (Raspberry Pi Zero 2 W, 3 and 4; a Pi 5 has no H.264
  decoder), ``-hwaccel vaapi`` (Intel and AMD GPUs)
- Windows ``-hwaccel d3d11va``, macOS ``-hwaccel videotoolbox``
- then software decoding

Seeking starts ffmpeg again at the frame (input seeking decodes from the key
frame before it and drops the rest). For looping, a second ffmpeg reading
from the start is opened shortly before the end, so the loop does not stall.
"""

from __future__ import annotations

import collections
import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from fractions import Fraction

import numpy as np

from ._frames import cover_crop, decode_size

logger = logging.getLogger(__name__)


def executables() -> tuple[str, str] | None:
    """Paths of ffmpeg and ffprobe, or None when either is missing."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    return (ffmpeg, ffprobe) if ffmpeg and ffprobe else None


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=30, **_popen_options()).stdout


@functools.lru_cache(maxsize=1)
def _capabilities(ffmpeg: str) -> tuple[frozenset[str], frozenset[str]]:
    """(hardware accelerations, decoders) this ffmpeg build offers."""
    hwaccels = frozenset(line.strip() for line in _run([ffmpeg, "-hide_banner", "-hwaccels"]).splitlines()[1:] if line.strip())
    decoders = frozenset(
        parts[1] for line in _run([ffmpeg, "-hide_banner", "-decoders"]).splitlines() if len(parts := line.split()) > 1
    )
    return hwaccels, decoders


def _decoder_options(ffmpeg: str, codec: str) -> list[list[str]]:
    """Decoder options to try for a codec, best first; the last is software decoding."""
    hwaccels, decoders = _capabilities(ffmpeg)
    options: list[list[str]] = []
    if sys.platform.startswith("linux"):
        if codec == "hevc" and "drm" in hwaccels:
            options.append(["-hwaccel", "drm"])
        if codec == "h264" and "h264_v4l2m2m" in decoders and os.path.exists("/dev/video10"):
            options.append(["-c:v", "h264_v4l2m2m"])
        if "vaapi" in hwaccels and os.path.exists("/dev/dri/renderD128"):
            options.append(["-hwaccel", "vaapi"])
    elif sys.platform == "win32" and "d3d11va" in hwaccels:
        options.append(["-hwaccel", "d3d11va"])
    elif sys.platform == "darwin" and "videotoolbox" in hwaccels:
        options.append(["-hwaccel", "videotoolbox"])
    options.append([])
    return options


def _popen_options() -> dict:
    # No console window per ffmpeg on Windows. If the engine ends without
    # closing it (a crash), ffmpeg's next write to the pipe fails and it exits.
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


class _Process:
    """One running ffmpeg: frames of one size from one position."""

    def __init__(self, args: list[str], frame_shape: tuple[int, int, int], start: int):
        self.frame_shape = frame_shape
        self.frame_bytes = frame_shape[0] * frame_shape[1] * frame_shape[2]
        self.start = start
        self.frames = 0  # frames read from it
        self.errors: collections.deque[str] = collections.deque(maxlen=5)
        self._proc = subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_popen_options()
        )
        self._stderr = threading.Thread(target=self._drain, daemon=True)
        self._stderr.start()

    def _drain(self) -> None:
        stderr = self._proc.stderr
        assert stderr is not None
        for line in iter(stderr.readline, b""):
            text = line.decode(errors="replace").strip()
            if text:
                self.errors.append(text)

    def read(self) -> np.ndarray | None:
        """The next frame, or None when ffmpeg has no more (end of file or an error)."""
        frame = np.empty(self.frame_shape, np.uint8)
        view = memoryview(frame).cast("B")
        stdout = self._proc.stdout
        assert stdout is not None
        got = 0
        while got < self.frame_bytes:  # a pipe may return part of a frame
            n = stdout.readinto(view[got:])  # releases the GIL while it waits
            if not n:
                return None
            got += n
        self.frames += 1
        return frame

    def failed(self) -> bool:
        """True when ffmpeg ended with an error (call after read() returned None)."""
        try:
            code = self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return False
        self._stderr.join(timeout=1.0)
        return code != 0

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None:
                stream.close()


class Reader:
    """The frames of one file in order, scaled to a requested size, as BGR."""

    def __init__(self, path: str):
        tools = executables()
        if tools is None:
            raise OSError("ffmpeg or ffprobe is not on PATH")
        self._ffmpeg, ffprobe = tools
        self._path = path
        try:
            info = json.loads(
                _run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                      "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
                      "-of", "json", path])
            )
            stream = info["streams"][0]
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as e:
            raise OSError(f"ffprobe cannot read {path}: {e}") from e
        self.codec = str(stream.get("codec_name", ""))
        self.width, self.height = int(stream["width"]), int(stream["height"])
        rate = Fraction(0)
        for key in ("avg_frame_rate", "r_frame_rate"):
            try:
                rate = Fraction(stream.get(key, "0/0"))
            except (ValueError, ZeroDivisionError):
                continue
            if rate > 0:
                break
        self.fps = float(rate)
        frames = stream.get("nb_frames")
        if frames and str(frames).isdigit():
            self.frame_count = int(frames)
        else:
            duration = float(info.get("format", {}).get("duration", 0) or 0)
            self.frame_count = int(round(duration * self.fps)) if self.fps > 0 else 0
        self._decoders = _decoder_options(self._ffmpeg, self.codec)
        self._decoder = 0
        self._decoded_any = False  # with the current decoder
        self.gpu = bool(self._decoders[0])
        self._output: tuple[int, int, str] | None = None
        self._position = 0  # index of the next frame
        self._process: _Process | None = None
        self._standby: _Process | None = None  # reading from frame 0, for the next loop

    # -- the reader protocol (see _video_sources._ScaledSource) ---------------------

    def configure(self, width: int, height: int, fit: str) -> bool:
        output = (width, height, fit)
        if output == self._output:
            return False
        self._output = output
        self._close_processes()  # started again at the current position by read()
        return True

    def read(self) -> np.ndarray | None:
        while True:
            if self._process is None:
                self._process = self._start(self._position)
            frame = self._process.read()
            if frame is not None:
                self._decoded_any = True
                self._position += 1
                return frame
            failed = self._process.failed()
            errors = list(self._process.errors)
            self._process.close()
            self._process = None
            if not failed or self._decoded_any:
                if failed:
                    logger.warning(f"ffmpeg stopped reading {self._path}: {' | '.join(errors)}")
                return None  # the end of the file
            if self._decoder + 1 >= len(self._decoders):
                raise OSError(f"ffmpeg cannot decode {self._path}: {' | '.join(errors)}")
            logger.info(f"ffmpeg {' '.join(self._decoders[self._decoder])} failed on {self._path}; trying the next decoder")
            self._decoder += 1
            self.gpu = bool(self._decoders[self._decoder])
            self._close_processes()

    def seek(self, index: int) -> None:
        index = max(0, index)
        process = self._process
        if process is not None and process.start + process.frames == index:
            return  # already there
        if process is not None:
            process.close()
            self._process = None
        if index == 0 and self._standby is not None:
            self._process, self._standby = self._standby, None
        self._position = index

    def prepare_rewind(self) -> None:
        """Open an ffmpeg reading from the first frame, for the loop that comes next."""
        if self._standby is None and self._output is not None:
            self._standby = self._start(0)

    def close(self) -> None:
        self._close_processes()

    # ------------------------------------------------------------------------------

    def _close_processes(self) -> None:
        for process in (self._process, self._standby):
            if process is not None:
                process.close()
        self._process = self._standby = None

    def _start(self, index: int) -> _Process:
        assert self._output is not None, "configure() before read()"
        width, height, fit = self._output
        dw, dh = decode_size(self.width, self.height, width, height, fit)
        filters = [] if (dw, dh) == (self.width, self.height) else [f"scale={dw}:{dh}:flags=area"]
        corner = cover_crop((dw, dh), width, height, fit)
        if corner is not None:
            filters.append(f"crop={width}:{height}:{corner[0]}:{corner[1]}")
            shape = (height, width, 3)
        else:
            shape = (dh, dw, 3)
        args = [self._ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *self._decoders[self._decoder]]
        if index > 0 and self.fps > 0:
            # Half a frame early: frame `index` is the first at or after it (input
            # seeking decodes from the key frame before and drops the rest).
            args += ["-ss", f"{(index - 0.5) / self.fps:.6f}"]
        args += ["-i", self._path, "-an", "-sn", "-dn"]
        if filters:
            args += ["-vf", ",".join(filters)]
        args += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        return _Process(args, shape, index)
