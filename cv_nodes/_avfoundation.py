"""AVFoundation as a frame reader (macOS; used by _video_sources).

An AVAssetReader decodes with VideoToolbox (the hardware decoder) and scales
to the size requested in its output settings, so only the small frame is
copied to the CPU, as with Media Foundation on Windows. Through PyObjC
(``pyobjc-framework-AVFoundation``, ``-CoreMedia``, ``-Quartz``; MIT).

A reader cannot seek once started: seeking makes a new one whose time range
starts at the frame. For looping, one reading from the start is made shortly
before the end. Each read runs in an autorelease pool (the worker thread has
none of its own).
"""

from __future__ import annotations

import numpy as np
import objc
from AVFoundation import (
    AVAssetReader,
    AVAssetReaderStatusFailed,
    AVAssetReaderTrackOutput,
    AVMediaTypeVideo,
    AVURLAsset,
)
from CoreMedia import (
    CMSampleBufferGetImageBuffer,
    CMSampleBufferGetPresentationTimeStamp,
    CMTimeGetSeconds,
    CMTimeMakeWithSeconds,
    CMTimeRangeMake,
    kCMTimePositiveInfinity,
)
from Foundation import NSURL
from Quartz import (
    CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetHeight,
    CVPixelBufferGetWidth,
    CVPixelBufferLockBaseAddress,
    CVPixelBufferUnlockBaseAddress,
    kCVPixelBufferHeightKey,
    kCVPixelBufferLock_ReadOnly,
    kCVPixelBufferPixelFormatTypeKey,
    kCVPixelBufferWidthKey,
    kCVPixelFormatType_32BGRA,
)

from ._frames import bgra_to_bgr, cover_crop, decode_size

_TIMESCALE = 600_000  # ticks per second for seek times (a multiple of common frame rates)


class _AssetRead:
    """One started AVAssetReader: frames of one size from one time on."""

    def __init__(self, asset, track, size: tuple[int, int], start_seconds: float):
        reader, error = AVAssetReader.alloc().initWithAsset_error_(asset, None)
        if reader is None:
            raise OSError(f"AVAssetReader: {error}")
        settings = {
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: size[0],
            kCVPixelBufferHeightKey: size[1],
        }
        output = AVAssetReaderTrackOutput.assetReaderTrackOutputWithTrack_outputSettings_(track, settings)
        output.setAlwaysCopiesSampleData_(False)  # the frame is copied out once, below
        if start_seconds > 0:
            start = CMTimeMakeWithSeconds(start_seconds, _TIMESCALE)
            reader.setTimeRange_(CMTimeRangeMake(start, kCMTimePositiveInfinity))
        if not reader.canAddOutput_(output):
            raise OSError("AVAssetReader cannot output BGRA frames of this track")
        reader.addOutput_(output)
        if not reader.startReading():
            raise OSError(f"AVAssetReader cannot start: {reader.error()}")
        self.reader, self.output = reader, output
        self.frames = 0

    def close(self) -> None:
        self.reader.cancelReading()


class Reader:
    """The frames of one file in order, scaled to a requested size, as BGR."""

    gpu = True  # VideoToolbox decodes and scales

    def __init__(self, path: str):
        with objc.autorelease_pool():
            self._asset = AVURLAsset.URLAssetWithURL_options_(NSURL.fileURLWithPath_(path), None)
            tracks = self._asset.tracksWithMediaType_(AVMediaTypeVideo)
            if not tracks:
                raise OSError(f"No video track in {path}")
            self._track = tracks[0]
            size = self._track.naturalSize()
            self.width, self.height = int(round(size.width)), int(round(size.height))
            self.fps = float(self._track.nominalFrameRate())
            seconds = float(CMTimeGetSeconds(self._asset.duration()))
        if self.width <= 0 or self.height <= 0:
            raise OSError(f"Cannot read the frame size of {path}")
        self.frame_count = int(round(seconds * self.fps)) if self.fps > 0 and seconds > 0 else 0
        self._path = path
        self._output: tuple[int, int, str] | None = None
        self._position = 0  # index of the next frame
        self._start = 0.0  # presentation time of the first frame
        self._started = False
        self._skip_before = -1.0  # drop frames that start before this time (seconds)
        self._read: _AssetRead | None = None
        self._standby: _AssetRead | None = None  # reading from frame 0, for the next loop

    # -- the reader protocol (see _video_sources._ScaledSource) ---------------------

    def configure(self, width: int, height: int, fit: str) -> bool:
        output = (width, height, fit)
        if output == self._output:
            return False
        self._output = output
        self._close_reads()  # made again at the current position by read()
        return True

    def read(self) -> np.ndarray | None:
        assert self._output is not None, "configure() before read()"
        with objc.autorelease_pool():
            if self._read is None:
                self._read = self._open(self._position)
            current = self._read
            while True:
                sample = current.output.copyNextSampleBuffer()
                if sample is None:
                    if current.reader.status() == AVAssetReaderStatusFailed:
                        raise OSError(f"AVFoundation cannot decode {self._path}: {current.reader.error()}")
                    return None  # the end of the file
                pts = float(CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample)))
                if not self._started:
                    self._start, self._started = pts, True
                if pts < self._skip_before:
                    continue  # before a seek target
                image = CMSampleBufferGetImageBuffer(sample)
                if image is None:
                    continue
                frame = self._copy(image)
                current.frames += 1
                self._position += 1
                return frame

    def seek(self, index: int) -> None:
        index = max(0, index)
        if self._position == index and self._read is not None:
            return  # already there
        self._close_read()
        if index == 0 and self._standby is not None:
            self._read, self._standby = self._standby, None
            self._skip_before = -1.0
        self._position = index

    def prepare_rewind(self) -> None:
        """Start a reader from the first frame, for the loop that comes next."""
        if self._standby is None and self._output is not None:
            with objc.autorelease_pool():
                self._standby = _AssetRead(self._asset, self._track, self._decode_size(), 0.0)

    def close(self) -> None:
        self._close_reads()

    # ------------------------------------------------------------------------------

    def _decode_size(self) -> tuple[int, int]:
        assert self._output is not None
        return decode_size(self.width, self.height, *self._output)

    def _open(self, index: int) -> _AssetRead:
        if index > 0 and self.fps > 0:
            # Read from half a frame before frame `index`. The reader may first return the
            # frame that covers that time, with its timestamp moved to it (macOS 15), so
            # drop what starts before a quarter frame ahead of frame `index`.
            opening = self._start + (index - 0.5) / self.fps
            self._skip_before = opening + 0.25 / self.fps
            return _AssetRead(self._asset, self._track, self._decode_size(), max(0.0, opening))
        self._skip_before = -1.0
        return _AssetRead(self._asset, self._track, self._decode_size(), 0.0)

    def _copy(self, image) -> np.ndarray:
        assert self._output is not None
        width, height, fit = self._output
        CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly)
        try:
            w, h = int(CVPixelBufferGetWidth(image)), int(CVPixelBufferGetHeight(image))
            stride = int(CVPixelBufferGetBytesPerRow(image))
            raw = CVPixelBufferGetBaseAddress(image).as_buffer(stride * h)
            bgra = np.ndarray((h, w, 4), np.uint8, raw, 0, (stride, 4, 1))
            # Whether VideoToolbox stretches or letterboxes another aspect, the size
            # asked for keeps the source's aspect: cut the middle out for cover.
            corner = cover_crop((w, h), width, height, fit)
            if corner is not None:
                x0, y0 = corner
                bgra = bgra[y0 : y0 + height, x0 : x0 + width]
            return bgra_to_bgr(bgra)  # copies out before the buffer is unlocked
        finally:
            CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly)

    def _close_read(self) -> None:
        if self._read is not None:
            self._read.close()
            self._read = None

    def _close_reads(self) -> None:
        self._close_read()
        if self._standby is not None:
            self._standby.close()
            self._standby = None
