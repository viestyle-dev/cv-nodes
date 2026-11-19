"""Media Foundation's Source Reader through ctypes (Windows; used by _video_sources).

With a Direct3D 11 device the GPU decodes (DXVA), and the video processor
scales the frames to the requested size and converts them to RGB32 there, so
only the small frame is copied to the CPU. A 4K frame read by OpenCV is
converted to BGR and resized on the CPU instead (about 16 ms of CPU against
under 1 ms here). Without a device, Media Foundation decodes and scales on the
CPU, still before the conversion. The video processor outputs RGB32 only on
the GPU path (RGB24 is refused when reading), so BGRA is copied to BGR here.

Only the methods used here are declared; the vtable indices follow the
Windows SDK headers (mfobjects.h, mfidl.h, mfreadwrite.h, d3d10misc.h).
Everything runs on the thread that opened the reader.
"""

from __future__ import annotations

import ctypes
import math
import threading
import uuid
from ctypes import POINTER, byref, c_longlong, c_uint32, c_uint64, c_void_p, wintypes

import numpy as np

from ._frames import bgra_to_bgr, cover_crop, decode_size

_HRESULT = ctypes.c_long


class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    u = uuid.UUID(text)
    g = _GUID(u.fields[0], u.fields[1], u.fields[2])
    g.d4[:] = u.bytes[8:]
    return g


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort * 3),
        ("value", c_longlong),
        ("pad", c_longlong),
    ]


def _method(index: int, *argtypes, hresult: bool = True):
    """A COM method by vtable index: call(obj, *args); raises OSError on a failed HRESULT."""
    proto = ctypes.WINFUNCTYPE(_HRESULT if hresult else wintypes.BOOL, c_void_p, *argtypes)

    def call(obj, *args):
        fn = proto(ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents[index])
        result = fn(obj, *args)
        if hresult and result < 0:
            raise OSError(f"Media Foundation error 0x{result & 0xFFFFFFFF:08X}")
        return result

    return call


_ReleaseProto = ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)


def _release(obj: c_void_p) -> None:
    if obj:
        _ReleaseProto(ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents[2])(obj)


# IUnknown
_QueryInterface = _method(0, POINTER(_GUID), POINTER(c_void_p))
# IMFAttributes (also IMFMediaType)
_GetUINT64 = _method(8, POINTER(_GUID), POINTER(c_uint64))
_SetUINT32 = _method(21, POINTER(_GUID), c_uint32)
_SetUINT64 = _method(22, POINTER(_GUID), c_uint64)
_SetGUID = _method(24, POINTER(_GUID), POINTER(_GUID))
_SetUnknown = _method(27, POINTER(_GUID), c_void_p)
# IMFSourceReader
_SetStreamSelection = _method(4, c_uint32, wintypes.BOOL)
_GetNativeMediaType = _method(5, c_uint32, c_uint32, POINTER(c_void_p))
_GetCurrentMediaType = _method(6, c_uint32, POINTER(c_void_p))
_SetCurrentMediaType = _method(7, c_uint32, c_void_p, c_void_p)
_SetCurrentPosition = _method(8, POINTER(_GUID), POINTER(_PROPVARIANT))
_ReadSample = _method(
    9, c_uint32, c_uint32, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_longlong), POINTER(c_void_p)
)
_GetPresentationAttribute = _method(12, c_uint32, POINTER(_GUID), POINTER(_PROPVARIANT))
# IMFSample
_ConvertToContiguousBuffer = _method(41, POINTER(c_void_p))
# IMFMediaBuffer
_Lock = _method(3, POINTER(c_void_p), POINTER(c_uint32), POINTER(c_uint32))
_Unlock = _method(4)
# IMFDXGIDeviceManager
_ResetDevice = _method(7, c_void_p, c_uint32)
# ID3D10Multithread (returns the previous state, not an HRESULT)
_SetMultithreadProtected = _method(5, wintypes.BOOL, hresult=False)

_MF_SOURCE_READER_D3D_MANAGER = _guid("ec822da2-e1e9-4b29-a0d8-563c719f5269")
_MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING = _guid("0f81da2c-b537-4672-a8b2-a681b17307a3")
_MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS = _guid("a634a91c-822b-41b9-a494-4de4643612b0")
_MF_MT_MAJOR_TYPE = _guid("48eba18e-f8c9-4687-bf11-0a74c9f96a8f")
_MF_MT_SUBTYPE = _guid("f7e34c9a-42e8-4714-b74b-cb29d72c35e5")
_MF_MT_FRAME_SIZE = _guid("1652c33d-d6b2-4012-b834-72030849a37d")
_MF_MT_FRAME_RATE = _guid("c459a2e8-3d2c-4e44-b132-fee5156c7bb0")
_MF_MT_PIXEL_ASPECT_RATIO = _guid("c6376a1e-8d0a-4027-be45-6d9a0ad39bb6")
_MF_PD_DURATION = _guid("6c990d33-bb8e-477a-8598-0d5d96fcd88a")
_MFMediaType_Video = _guid("73646976-0000-0010-8000-00AA00389B71")
_MFVideoFormat_RGB32 = _guid("00000016-0000-0010-8000-00AA00389B71")
_IID_ID3D10Multithread = _guid("9B7E4E00-342C-4106-A19F-4F2704F689F0")
_GUID_NULL = _GUID()

_FIRST_VIDEO_STREAM = 0xFFFFFFFC
_ALL_STREAMS = 0xFFFFFFFE
_MEDIASOURCE = 0xFFFFFFFF
_MF_VERSION = 0x00020070
_READERF_ERROR = 0x1
_READERF_ENDOFSTREAM = 0x2
_READERF_CURRENTMEDIATYPECHANGED = 0x20
_D3D_DRIVER_TYPE_HARDWARE = 1
_D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
_D3D11_CREATE_DEVICE_VIDEO_SUPPORT = 0x800
_D3D11_SDK_VERSION = 7
_VT_I8 = 20

_ole32 = ctypes.WinDLL("ole32")
_mfplat = ctypes.WinDLL("mfplat")
_mfreadwrite = ctypes.WinDLL("mfreadwrite")
_d3d11 = ctypes.WinDLL("d3d11")

_com_threads = threading.local()


def _check(hr: int, what: str) -> None:
    if hr < 0:
        raise OSError(f"{what} failed: 0x{hr & 0xFFFFFFFF:08X}")


def _init_thread() -> None:
    """COM for this thread, once (an STA thread keeps its apartment: Media Foundation works there too)."""
    if not getattr(_com_threads, "ready", False):
        _ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        _com_threads.ready = True


def _d3d_manager() -> tuple[c_void_p, c_void_p, c_void_p] | None:
    """A Direct3D 11 device for decoding and scaling on the GPU, or None (no GPU)."""
    device, context, level = c_void_p(), c_void_p(), c_uint32()
    hr = _d3d11.D3D11CreateDevice(
        None,
        _D3D_DRIVER_TYPE_HARDWARE,
        None,
        _D3D11_CREATE_DEVICE_VIDEO_SUPPORT | _D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        None,
        0,
        _D3D11_SDK_VERSION,
        byref(device),
        byref(level),
        byref(context),
    )
    if hr < 0:
        return None
    manager = c_void_p()
    try:
        multithread = c_void_p()
        _QueryInterface(device, byref(_IID_ID3D10Multithread), byref(multithread))
        _SetMultithreadProtected(multithread, True)  # the decoder uses the device from its own threads
        _release(multithread)
        token = c_uint32()
        _check(_mfplat.MFCreateDXGIDeviceManager(byref(token), byref(manager)), "MFCreateDXGIDeviceManager")
        _ResetDevice(manager, device, token)
    except OSError:
        for obj in (manager, context, device):
            _release(obj)
        return None
    return device, context, manager


class Reader:
    """The frames of one file in order, scaled to a requested size, as BGR."""

    def __init__(self, path: str):
        _init_thread()
        _check(_mfplat.MFStartup(_MF_VERSION, 0), "MFStartup")
        self._objects: list[c_void_p] = []  # released by close(), last first
        self._reader = c_void_p()
        try:
            self._open(path)
        except BaseException:
            self.close()
            raise

    def _open(self, path: str) -> None:
        attributes = c_void_p()
        _check(_mfplat.MFCreateAttributes(byref(attributes), 3), "MFCreateAttributes")
        try:
            d3d = _d3d_manager()
            self.gpu = d3d is not None
            if d3d is not None:
                self._objects.extend(d3d)
                _SetUnknown(attributes, byref(_MF_SOURCE_READER_D3D_MANAGER), d3d[2])
            _SetUINT32(attributes, byref(_MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING), 1)
            _SetUINT32(attributes, byref(_MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS), 1)
            _check(
                _mfreadwrite.MFCreateSourceReaderFromURL(ctypes.c_wchar_p(path), attributes, byref(self._reader)),
                f"Opening {path}",
            )
        finally:
            _release(attributes)
        _SetStreamSelection(self._reader, _ALL_STREAMS, False)
        _SetStreamSelection(self._reader, _FIRST_VIDEO_STREAM, True)

        native = c_void_p()
        _GetNativeMediaType(self._reader, _FIRST_VIDEO_STREAM, 0, byref(native))
        try:
            size, rate = c_uint64(), c_uint64()
            _GetUINT64(native, byref(_MF_MT_FRAME_SIZE), byref(size))
            _GetUINT64(native, byref(_MF_MT_FRAME_RATE), byref(rate))
        finally:
            _release(native)
        self.width, self.height = size.value >> 32, size.value & 0xFFFFFFFF
        numerator, denominator = rate.value >> 32, rate.value & 0xFFFFFFFF
        self.fps = numerator / denominator if numerator and denominator else 0.0
        duration = _PROPVARIANT()
        try:
            _GetPresentationAttribute(self._reader, _MEDIASOURCE, byref(_MF_PD_DURATION), byref(duration))
            self.frame_count = int(round(duration.value / 1e7 * self.fps))
        except OSError:
            self.frame_count = 0

        self._output: tuple[int, int, str] | None = None
        self._decode = (0, 0)  # size the GPU scales to
        self._crop: tuple[int, int, int, int] | None = None  # x0, y0, width, height
        self._skip_before = -1  # timestamps (100 ns) before a seek target are decoded and dropped
        self._start = 0  # timestamp of the first frame
        self._started = False
        self._index, self._flags, self._time, self._sample = c_uint32(), c_uint32(), c_longlong(), c_void_p()
        self._buffer, self._data, self._max, self._length = c_void_p(), c_void_p(), c_uint32(), c_uint32()

    def configure(self, width: int, height: int, fit: str) -> bool:
        """Deliver frames of width x height (see decode_size; exact delivers the source size).

        True when the output changed. A change drops the frames Media Foundation
        decoded ahead: seek to the next frame afterwards to keep them.
        """
        output = (width, height, fit)
        if output == self._output:
            return False
        dw, dh = decode_size(self.width, self.height, width, height, fit)
        media_type = c_void_p()
        _check(_mfplat.MFCreateMediaType(byref(media_type)), "MFCreateMediaType")
        try:
            _SetGUID(media_type, byref(_MF_MT_MAJOR_TYPE), byref(_MFMediaType_Video))
            _SetGUID(media_type, byref(_MF_MT_SUBTYPE), byref(_MFVideoFormat_RGB32))
            _SetUINT64(media_type, byref(_MF_MT_FRAME_SIZE), (dw << 32) | dh)
            # Pixels of the aspect that makes the picture fill dw x dh: the video
            # processor would add black bars to keep the source's aspect otherwise.
            numerator, denominator = self.width * dh, self.height * dw
            g = math.gcd(numerator, denominator)
            _SetUINT64(media_type, byref(_MF_MT_PIXEL_ASPECT_RATIO), ((numerator // g) << 32) | (denominator // g))
            _SetCurrentMediaType(self._reader, _FIRST_VIDEO_STREAM, None, media_type)
        finally:
            _release(media_type)
        self._output = output
        self._set_decode_size(output)
        return True

    def _set_decode_size(self, output: tuple[int, int, str]) -> None:
        current = c_void_p()
        _GetCurrentMediaType(self._reader, _FIRST_VIDEO_STREAM, byref(current))
        try:
            size = c_uint64()
            _GetUINT64(current, byref(_MF_MT_FRAME_SIZE), byref(size))
        finally:
            _release(current)
        width, height, fit = output
        self._decode = (size.value >> 32, size.value & 0xFFFFFFFF)
        corner = cover_crop(self._decode, width, height, fit)
        self._crop = None if corner is None else (corner[0], corner[1], width, height)

    def read(self) -> np.ndarray | None:
        """The next frame (BGR), or None at the end of the file."""
        while True:
            _ReadSample(
                self._reader,
                _FIRST_VIDEO_STREAM,
                0,
                byref(self._index),
                byref(self._flags),
                byref(self._time),
                byref(self._sample),
            )
            flags = self._flags.value
            sample = self._sample.value
            self._sample.value = None
            if flags & _READERF_ERROR:
                _release(c_void_p(sample))
                raise OSError("Media Foundation could not decode the stream")
            if flags & _READERF_CURRENTMEDIATYPECHANGED and self._output is not None:
                self._set_decode_size(self._output)
            if not sample:
                if flags & _READERF_ENDOFSTREAM:
                    return None
                continue  # a gap in the stream
            try:
                if not self._started:
                    self._start = self._time.value
                    self._started = True
                if self._time.value < self._skip_before:
                    continue  # decoded from the key frame before a seek target
                return self._copy(c_void_p(sample))
            finally:
                _release(c_void_p(sample))

    def _copy(self, sample: c_void_p) -> np.ndarray:
        _ConvertToContiguousBuffer(sample, byref(self._buffer))
        buffer = c_void_p(self._buffer.value)
        self._buffer.value = None
        try:
            _Lock(buffer, byref(self._data), byref(self._max), byref(self._length))
            try:
                dw, dh = self._decode
                length, address = self._length.value, self._data.value
                stride = length // dh if dh else 0
                if not address or stride < dw * 4:
                    raise OSError(f"Unexpected frame buffer: {length} bytes for {dw}x{dh}")
                raw = memoryview((ctypes.c_ubyte * length).from_address(address))
                bgra = np.ndarray((dh, dw, 4), np.uint8, raw, 0, (stride, 4, 1))
                if self._crop is not None:
                    x0, y0, w, h = self._crop
                    bgra = bgra[y0 : y0 + h, x0 : x0 + w]
                return bgra_to_bgr(bgra)  # copies out of the locked buffer
            finally:
                _Unlock(buffer)
        finally:
            _release(buffer)

    def seek(self, index: int) -> None:
        """Continue from frame ``index`` (frames from the key frame before it are dropped)."""
        target = self._start + int(round(index / self.fps * 1e7)) if self.fps > 0 else self._start
        position = _PROPVARIANT()
        position.vt = _VT_I8
        position.value = target
        _SetCurrentPosition(self._reader, byref(_GUID_NULL), byref(position))
        half_frame = int(0.5e7 / self.fps) if self.fps > 0 else 0
        self._skip_before = target - half_frame

    def close(self) -> None:
        _release(self._reader)
        self._reader = c_void_p()
        while self._objects:
            _release(self._objects.pop())
        _mfplat.MFShutdown()
