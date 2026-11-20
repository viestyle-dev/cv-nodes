# videoio_nodes — OpenCV "videoio" module (@defgroup videoio in videoio.hpp)
# VideoCapture, VideoWriter, and related I/O classes.

import asyncio
import logging
import pathlib
import queue
import sys
import threading
import time
from enum import Enum

import cv2
import numpy as np

from nonoha_core.data_type import (
    DataType,
    PortValueType,
    PropertyDataType,
)
from nonoha_core.port import (
    PortValue,
    StaticTypeInputPort,
    StaticTypeOutputPort,
)
from nonoha_core.node import BaseNode, NodeDoc, replace_node_output_port

from ._video_sources import BACKENDS, VideoSource, backend_name, open_source

logger = logging.getLogger(__name__)


class VideoFileCaptureStatus(Enum):
    NotLoaded = 0
    Loading = 1
    Loaded = 2


_FITS = ("cover", "stretch", "exact")
_MAX_QUEUED_READS = 1  # reads waiting behind the one in progress
_DROP_REPORT_SECONDS = 10.0


class VideoFileCaptureNode(BaseNode):
    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Reads the frames of a video file in order, on a background "
                "thread, when readout_count arrives (drive it from a clock for "
                "real-time playback, or from position to read every frame as "
                "fast as possible). Frames of another size are fitted to width x "
                "height (see fit). At most one read waits behind the one in "
                "progress: requests beyond that are dropped (playback slows down; "
                "no frame is skipped) and logged."
            ),
            category="Video",
            inputs={
                "readout_count": "Number of frames to read now (one batch per value)",
                "seek": "Frame index to jump to",
            },
            outputs={
                "loading_status": "0 = not loaded, 1 = loading, 2 = loaded",
                "frame": "Frames u8[height x width x 3], BGR",
                "video_info": "width, height, fps, frame_count of the loaded file",
                "eof": "True when the end of the file is reached (never, with loop on)",
                "position": "Index of the last frame read",
            },
        )

    PROPERTIES = {
        "file_path": {
            "description": (
                "Video file to read; relative paths resolve against the graph's "
                "data folder. Empty closes the file"
            ),
            "data_type": "str",
            "default_value": "",
        },
        "backend": {
            "description": (
                "How frames are read: " + ", ".join(BACKENDS) + ". auto: "
                "media_foundation on Windows, avfoundation on macOS (both "
                "decode and scale to the output on the GPU), ffmpeg_command "
                "(the ffmpeg program, if installed: hardware decoding where "
                "there is a decoder, scaled before conversion), then "
                "opencv_ffmpeg (full frames converted on the CPU). A reader "
                "that cannot read a file gives way to the next ('ffmpeg' is "
                "the old name of opencv_ffmpeg)"
            ),
            "data_type": "str",
            "default_value": "auto",
        },
        "width": {"description": "Output frame width", "data_type": "i32", "default_value": 1280},
        "height": {"description": "Output frame height", "data_type": "i32", "default_value": 720},
        "loop": {"description": "Restart from the first frame at the end of the file", "data_type": "bool", "default_value": False},
        "fit": {
            "description": (
                "Frames of another size: cover (crop the middle to the output "
                "aspect, then scale), stretch, or exact (send nothing)"
            ),
            "data_type": "str",
            "default_value": "cover",
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "VideoFileCapture")

        width = 1280
        height = 720

        # --- Input ports ---
        input_readout_count = StaticTypeInputPort(
            "readout_count",
            [PortValueType(DataType.Int32)],
        )
        input_readout_count.subscribe(id(self), self._on_input_readout_count)

        input_seek = StaticTypeInputPort("seek", [PortValueType(DataType.Int32)])
        input_seek.subscribe(id(self), self._on_input_seek)

        # --- Output ports ---
        output_loading_status = StaticTypeOutputPort(
            "loading_status", [PortValueType(DataType.Uint8)]
        )
        output_frame = StaticTypeOutputPort(
            "frame", [PortValueType(DataType.Uint8, (height, width, 3))]
        )
        output_video_info = StaticTypeOutputPort(
            "video_info",
            [
                PortValueType(DataType.Int32),  # width
                PortValueType(DataType.Int32),  # height
                PortValueType(DataType.Float32),  # fps
                PortValueType(DataType.Int32),  # frame_count
            ],
            [
                "width",
                "height",
                "fps",
                "frame_count",
            ],
        )
        output_eof = StaticTypeOutputPort("eof", [PortValueType(DataType.Bool)])
        output_position = StaticTypeOutputPort(
            "position", [PortValueType(DataType.Int32)]
        )

        # --- Store references ---
        self._input_readout_count = input_readout_count
        self._input_seek = input_seek
        self._output_loading_status = output_loading_status
        self._output_frame = output_frame
        self._output_video_info = output_video_info
        self._output_eof = output_eof
        self._output_position = output_position

        self._width = width
        self._height = height
        self._is_loaded = False
        self._resolution_mismatch = False
        self._actual_width = 0
        self._actual_height = 0
        self._file_path = ""
        self._backend = "auto"

        # Read requests: at most _MAX_QUEUED_READS wait behind the one in
        # progress; the rest are dropped and reported (see _on_input_readout_count).
        self._reads_lock = threading.Lock()
        self._queued_reads = 0
        self._dropped_reads = 0
        self._drop_report_t = 0.0
        self._read_ms = 0.0  # last read, per frame (worker thread)

        # --- Register ports ---
        self.inputs[input_readout_count.name] = input_readout_count
        self.inputs[input_seek.name] = input_seek

        self.outputs[output_loading_status.name] = output_loading_status
        self.outputs[output_frame.name] = output_frame
        self.outputs[output_video_info.name] = output_video_info
        self.outputs[output_eof.name] = output_eof
        self.outputs[output_position.name] = output_position

        # --- Properties ---
        self.properties["file_path"] = ""
        self.properties["backend"] = "auto"
        self.properties["width"] = width
        self.properties["height"] = height
        self.properties["loop"] = False
        self._loop_enabled = False  # read by the worker thread
        self.properties["fit"] = "cover"
        self._fit = "cover"  # read by the worker thread

        # --- Worker thread ---
        self._task_queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None

        # --- Initial status ---
        output_loading_status.dispatch(
            [np.array([VideoFileCaptureStatus.NotLoaded.value], dtype=np.uint8)]
        )

    def activate(self):
        self._worker_thread = threading.Thread(
            target=self._worker_loop, args=(asyncio.get_running_loop(),), daemon=True
        )
        self._worker_thread.start()

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract properties
        backend = self._backend
        if "backend" in properties:
            backend = backend_name(str(properties["backend"]))  # maps old names
        file_path = self._file_path
        if "file_path" in properties:
            file_path = str(properties["file_path"])
        # Load when the file changes, or reopen it with another backend.
        need_load = file_path != self._file_path or (backend != self._backend and file_path != "")
        width = self._width
        height = self._height
        if "width" in properties:
            width = int(properties["width"])
        if "height" in properties:
            height = int(properties["height"])

        # 2. Prepare new port (pure, no side effects)
        new_output_frame = self._output_frame
        if width != self._width or height != self._height:
            new_output_frame = StaticTypeOutputPort(
                "frame", [PortValueType(DataType.Uint8, (height, width, 3))]
            )

        # 3. Update internal state
        #   3a. Replace port
        if new_output_frame != self._output_frame:
            replace_node_output_port(self, self._output_frame, new_output_frame)
            self._output_frame = new_output_frame

        #   3b. Update state variables
        self._width = width
        self._height = height
        self._file_path = file_path
        self._backend = backend
        self.properties["file_path"] = file_path
        self.properties["backend"] = backend

        if "fit" in properties:
            fit = str(properties["fit"])
            if fit not in _FITS:
                raise ValueError(f"fit must be one of {_FITS}")
            self._fit = fit
            self.properties["fit"] = fit

        # Update resolution mismatch state if file is loaded
        if self._is_loaded:
            old_mismatch = self._resolution_mismatch
            self._resolution_mismatch = self._fit == "exact" and (
                self._actual_width != width or self._actual_height != height
            )
            if self._resolution_mismatch and not old_mismatch:
                logger.warning(
                    f"{self.name}: Resolution mismatch - file is "
                    f"{self._actual_width}x{self._actual_height}, "
                    f"but node configured for {width}x{height}. "
                    f"Frames will not be dispatched."
                )
            elif not self._resolution_mismatch and old_mismatch:
                logger.info(f"{self.name}: Resolution mismatch resolved.")

        #   3c. Update properties dict
        self.properties["width"] = width
        self.properties["height"] = height
        if "loop" in properties:
            self._loop_enabled = bool(properties["loop"])
            self.properties["loop"] = self._loop_enabled

        # 4. Notify graph
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

        # 5. Open the new file (the worker runs once activated)
        if need_load:
            self._request_load()

    # =========================================================================
    # Event loop thread — input callbacks
    # =========================================================================

    def _request_load(self):
        path = self._file_path
        self._is_loaded = False
        self._resolution_mismatch = False
        status = VideoFileCaptureStatus.Loading if path else VideoFileCaptureStatus.NotLoaded
        self._output_loading_status.dispatch([np.array([status.value], dtype=np.uint8)])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()
        self._task_queue.put(("load", (path, self._backend)))

    def _on_input_readout_count(self, values: list[PortValue]):
        if not self._is_loaded:
            return
        if self._resolution_mismatch:
            return
        with self._reads_lock:
            if self._queued_reads < _MAX_QUEUED_READS:
                self._queued_reads += 1
                self._task_queue.put(("read", (values[0],)))
                return
            # Reading is slower than the requests: drop (frames stay in order).
            self._dropped_reads += 1
            now = time.monotonic()
            if now - self._drop_report_t < _DROP_REPORT_SECONDS:
                return
            dropped, read_ms = self._dropped_reads, self._read_ms
            self._dropped_reads = 0
            self._drop_report_t = now
        logger.warning(
            f"{self.name}: {dropped} read request(s) dropped: reading is slower "
            f"than the requests ({read_ms:.0f} ms per frame)"
        )

    def _on_input_seek(self, values: list[PortValue]):
        seek_position_array = values[0]
        position = int(seek_position_array[-1])
        self._task_queue.put(("seek", (position,)))

    # =========================================================================
    # Event loop thread — callbacks from worker via call_soon_threadsafe
    # =========================================================================

    def _on_load_complete(
        self, actual_w: int, actual_h: int, fps: float, frame_count: int
    ):
        self._is_loaded = True
        self._actual_width = actual_w
        self._actual_height = actual_h

        # Resolution: other sizes are fitted unless fit is "exact"
        differs = actual_w != self._width or actual_h != self._height
        self._resolution_mismatch = differs and self._fit == "exact"
        if self._resolution_mismatch:
            logger.warning(
                f"{self.name}: Resolution mismatch - file is "
                f"{actual_w}x{actual_h}, but node configured for "
                f"{self._width}x{self._height}. Frames will not be dispatched."
            )
        elif differs:
            logger.info(
                f"{self.name}: Fitting {actual_w}x{actual_h} frames to "
                f"{self._width}x{self._height} (fit={self._fit})"
            )

        # Dispatch LOADED status
        self._output_loading_status.dispatch(
            [np.array([VideoFileCaptureStatus.Loaded.value], dtype=np.uint8)]
        )

        # Dispatch video_info
        self._output_video_info.dispatch(
            [
                np.array([actual_w], dtype=np.int32),
                np.array([actual_h], dtype=np.int32),
                np.array([fps], dtype=np.float32),
                np.array([frame_count], dtype=np.int32),
            ]
        )

        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _on_load_failed(self):
        self._is_loaded = False
        self._resolution_mismatch = False

        self._output_loading_status.dispatch(
            [np.array([VideoFileCaptureStatus.NotLoaded.value], dtype=np.uint8)]
        )

        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _dispatch_frames(
        self, batch: np.ndarray, is_eof: bool, position: int
    ):
        self._output_frame.dispatch([batch])
        self._output_position.dispatch(
            [np.array([position], dtype=np.int32)]
        )
        if is_eof:
            self._output_eof.dispatch([np.array([True])])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _dispatch_eof(self):
        self._output_eof.dispatch([np.array([True])])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    # =========================================================================
    # Worker thread
    # =========================================================================

    def _worker_loop(self, loop: asyncio.AbstractEventLoop):
        source: VideoSource | None = None
        eof: bool = False
        worker_position: int = 0

        while True:
            task = self._task_queue.get()
            if task is None:
                if source is not None:
                    source.release()
                break
            op, args = task
            try:
                if op == "load":
                    source, eof = self._do_load(loop, source, *args)
                    worker_position = 0
                elif op == "read":
                    with self._reads_lock:
                        self._queued_reads -= 1
                    eof, worker_position = self._do_read(
                        loop, source, eof, worker_position, *args
                    )
                elif op == "seek":
                    eof, worker_position = self._do_seek(source, *args)
            except Exception as e:
                logger.error(f"{self.name}: Worker error: {e}")

    def _do_load(
        self,
        loop: asyncio.AbstractEventLoop,
        old_source: VideoSource | None,
        file_name: str,
        backend: str,
    ) -> tuple[VideoSource | None, bool]:
        if old_source is not None:
            old_source.release()
        if not file_name:
            return None, False  # closed (_request_load reported "not loaded")

        file_path = pathlib.Path(file_name)
        if file_path.is_absolute():
            resolved_path = str(file_path)
        else:
            parent_graph = self.parent_graph
            if parent_graph is None or parent_graph.persist is None:
                loop.call_soon_threadsafe(self._on_load_failed)
                return None, False
            resolved_path = str(parent_graph.persist.data_dir_path / file_name)

        source = open_source(resolved_path, backend, (self._width, self._height, self._fit))
        if source is None:
            logger.error(f"{self.name}: Failed to open video file: {resolved_path}")
            loop.call_soon_threadsafe(self._on_load_failed)
            return None, False

        logger.info(
            f"{self.name}: Loaded video file: {resolved_path}, "
            f"{source.width}x{source.height}, fps={source.fps:.2f}, "
            f"frames={source.frame_count}, backend={source.backend}"
        )

        loop.call_soon_threadsafe(
            self._on_load_complete, source.width, source.height, source.fps, source.frame_count
        )
        return source, False

    def _do_read(
        self,
        loop: asyncio.AbstractEventLoop,
        source: VideoSource | None,
        eof: bool,
        worker_position: int,
        counts: np.ndarray,
    ) -> tuple[bool, int]:
        if source is None or (eof and not self._loop_enabled):
            return eof, worker_position
        started = time.perf_counter()
        try:
            return self._read_counts(loop, source, eof, worker_position, counts)
        finally:
            frames = max(1, int(np.maximum(counts, 0).sum()))
            per_frame = (time.perf_counter() - started) * 1000.0 / frames
            with self._reads_lock:
                self._read_ms = per_frame

    def _read_counts(
        self,
        loop: asyncio.AbstractEventLoop,
        source: VideoSource,
        eof: bool,
        worker_position: int,
        counts: np.ndarray,
    ) -> tuple[bool, int]:
        for count in counts:
            count = int(count)
            if count <= 0:
                continue

            if count == 1:
                # Hot path: avoid np.stack copy, use view instead
                frame, wrapped = self._read_frame(source)
                if frame is not None:
                    if wrapped:
                        worker_position = 0
                    worker_position += 1
                    loop.call_soon_threadsafe(
                        self._dispatch_frames,
                        frame[np.newaxis],
                        False,
                        worker_position - 1,
                    )
                else:
                    loop.call_soon_threadsafe(self._dispatch_eof)
                    return True, worker_position
            else:
                frames = []
                for _ in range(count):
                    frame, wrapped = self._read_frame(source)
                    if frame is None:
                        break
                    if wrapped:
                        worker_position = 0
                    worker_position += 1
                    frames.append(frame)

                is_eof = len(frames) < count
                if frames:
                    batch = np.stack(frames)
                    loop.call_soon_threadsafe(
                        self._dispatch_frames,
                        batch,
                        is_eof,
                        worker_position - 1,
                    )
                elif is_eof:
                    loop.call_soon_threadsafe(self._dispatch_eof)

                if is_eof:
                    return True, worker_position

        return eof, worker_position

    def _read_frame(self, source: VideoSource) -> tuple[np.ndarray | None, bool]:
        """Read one frame, fitted; with loop on, wrap to the first frame at the end.

        Returns (frame, or None at the end of the file; wrapped).
        """
        frame, wrapped = source.read(self._loop_enabled, (self._width, self._height, self._fit))
        if frame is None:
            return None, False
        return self._fit_frame(frame), wrapped

    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        """Bring a frame to width x height (worker thread; a no-op when it already matches,
        as Media Foundation's frames do)."""
        width, height, fit = self._width, self._height, self._fit
        frame_h, frame_w = frame.shape[:2]
        if (frame_w == width and frame_h == height) or fit == "exact":
            return frame
        if fit == "cover":
            # Crop the middle to the output aspect ratio (no distortion).
            if frame_w * height > frame_h * width:
                crop_w = frame_h * width // height
                x0 = (frame_w - crop_w) // 2
                frame = frame[:, x0 : x0 + crop_w]
            else:
                crop_h = frame_w * height // width
                y0 = (frame_h - crop_h) // 2
                frame = frame[y0 : y0 + crop_h]
        shrink = frame.shape[1] > width
        return cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
        )

    def _do_seek(self, source: VideoSource | None, position: int) -> tuple[bool, int]:
        if source is None:
            return False, position
        source.seek(position)
        return False, position  # eof reset, worker_position = seek target

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def finalize(self):
        self._task_queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None


class CameraCaptureStatus(Enum):
    Stopped = 0
    Opening = 1
    Running = 2
    Error = 3


class CameraCaptureNode(BaseNode):
    """Capture frames from a camera on a background thread.

    The reader thread paces itself on the camera. Only the newest frame is
    handed to the event loop: if the loop is busy, older frames are dropped
    instead of queuing up.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Captures frames from a camera on a background thread. Only the "
                "newest frame is delivered when the engine is busy. Frames are "
                "resized to width x height if the camera delivers another size."
            ),
            category="Video",
            outputs={
                "frame": "Frames u8[height x width x 3], BGR",
                "status": "0 = stopped, 1 = opening, 2 = running, 3 = error",
                "fps": "Measured capture rate (frames per second)",
            },
        )

    PROPERTIES = {
        "camera_index": {"description": "Camera index (0 = first camera)", "data_type": "i32", "default_value": 0},
        "width": {"description": "Frame width", "data_type": "i32", "default_value": 640},
        "height": {"description": "Frame height", "data_type": "i32", "default_value": 480},
        "fps": {"description": "Requested capture rate", "data_type": "f64", "default_value": 30.0},
        "enabled": {
            "description": "Capture while true",
            "data_type": "bool",
            "default_value": False,
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "CameraCapture")
        self._width = int(self.properties["width"])
        self._height = int(self.properties["height"])

        self._out_frame = StaticTypeOutputPort(
            "frame", [PortValueType(DataType.Uint8, (self._height, self._width, 3))]
        )
        self._out_status = StaticTypeOutputPort("status", [PortValueType(DataType.Uint8)])
        self._out_fps = StaticTypeOutputPort("fps", [PortValueType(DataType.Float32)])
        for port in (self._out_frame, self._out_status, self._out_fps):
            self.outputs[port.name] = port

        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._delivery_scheduled = False
        self._measured_fps = 0.0

    # -- lifecycle ------------------------------------------------------------

    def activate(self):
        self._main_loop = asyncio.get_running_loop()
        self._dispatch_status(CameraCaptureStatus.Stopped.value)
        if self.properties["enabled"]:
            self._start()

    def finalize(self):
        self._stop()

    def _start(self):
        if self._thread is not None or self._main_loop is None:
            return
        self._stop_event = threading.Event()
        args = (
            self._main_loop,
            int(self.properties["camera_index"]),
            self._width,
            self._height,
            float(self.properties["fps"]),
            self._stop_event,
        )
        self._thread = threading.Thread(
            target=self._reader, args=args, name=f"{self.name}-camera", daemon=True
        )
        self._thread.start()

    def _stop(self):
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=2.0)
        self._thread = None

    def set_properties(self, properties: dict[str, PropertyDataType]):
        restart = False
        if "camera_index" in properties:
            restart |= int(properties["camera_index"]) != self.properties["camera_index"]
            self.properties["camera_index"] = int(properties["camera_index"])
        if "fps" in properties:
            restart |= float(properties["fps"]) != self.properties["fps"]
            self.properties["fps"] = float(properties["fps"])

        width = max(1, int(properties.get("width", self._width)))
        height = max(1, int(properties.get("height", self._height)))
        if (width, height) != (self._width, self._height):
            new_frame = StaticTypeOutputPort(
                "frame", [PortValueType(DataType.Uint8, (height, width, 3))]
            )
            replace_node_output_port(self, self._out_frame, new_frame)
            self._out_frame = new_frame
            self._width, self._height = width, height
            self.properties.update({"width": width, "height": height})
            restart = True

        if "enabled" in properties:
            self.properties["enabled"] = bool(properties["enabled"])

        if self.is_active:
            running = self._thread is not None
            if running and (restart or not self.properties["enabled"]):
                self._stop()
            if self.properties["enabled"] and self._thread is None:
                self._start()
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    # -- reader thread ---------------------------------------------------------

    def _reader(
        self,
        loop: asyncio.AbstractEventLoop,
        index: int,
        width: int,
        height: int,
        fps: float,
        stop: threading.Event,
    ):
        post = lambda fn, *a: loop.call_soon_threadsafe(fn, *a)  # noqa: E731
        post(self._dispatch_status, CameraCaptureStatus.Opening.value)

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            logger.error(f"{self.name}: cannot open camera {index}")
            cap.release()
            post(self._dispatch_status, CameraCaptureStatus.Error.value)
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        post(self._dispatch_status, CameraCaptureStatus.Running.value)

        frames, t0 = 0, time.monotonic()
        try:
            while not stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                frames += 1
                now = time.monotonic()
                if now - t0 >= 1.0:
                    self._measured_fps = frames / (now - t0)
                    frames, t0 = 0, now

                with self._lock:
                    self._latest = frame
                    schedule = not self._delivery_scheduled
                    self._delivery_scheduled = True
                if schedule:
                    post(self._deliver_latest)
        except RuntimeError:
            pass  # event loop closed
        finally:
            cap.release()
            if not loop.is_closed():
                post(self._dispatch_status, CameraCaptureStatus.Stopped.value)

    # -- event loop thread -------------------------------------------------------

    def _deliver_latest(self):
        with self._lock:
            frame = self._latest
            self._latest = None
            self._delivery_scheduled = False
        if frame is None or frame.shape[:2] != (self._height, self._width):
            return
        self._out_frame.dispatch([frame[np.newaxis]])
        self._out_fps.dispatch([np.array([self._measured_fps], dtype=np.float32)])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _dispatch_status(self, code: int):
        self._out_status.dispatch([np.array([code], dtype=np.uint8)])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()
