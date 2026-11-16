"""Tests for the motion, resize, encode and capture nodes added in 0.3.0."""

import asyncio
import sys
import threading
import time

import cv2
import numpy as np
import pytest

from nonoha_core._testing import assert_all_ports_documented
from nonoha_core.node_registry import NodeRegistry

import cv_nodes
from cv_nodes.imgcodecs_nodes import ImdecodeNode, ImencodeNode
from cv_nodes.imgproc_geometric_nodes import ResizeNode
from cv_nodes.motion_np_nodes import FrameDiffMotionNode, MotionGridNode, OnsetDetector, grid_mean
from cv_nodes.video_nodes import BlobCounterNode, OpticalFlowFarnebackNode
from cv_nodes import _video_sources
from cv_nodes._video_sources import open_source
from cv_nodes.videoio_nodes import VideoFileCaptureNode

# Both readers on Windows; FFmpeg only elsewhere.
BACKENDS = [
    "ffmpeg",
    pytest.param(
        "media_foundation",
        marks=pytest.mark.skipif(sys.platform != "win32", reason="Media Foundation is Windows only"),
    ),
]


@pytest.fixture(scope="session")
def media_foundation_decodes(tmp_path_factory):
    """Whether Media Foundation opens the test clips (Windows Server may lack it)."""
    path = tmp_path_factory.mktemp("probe") / "clip.avi"
    _gray_clip(path)
    cap = _video_sources._capture(str(path), "media_foundation")
    if cap is None:
        return False
    cap.release()
    return True


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "media_foundation" and not request.getfixturevalue("media_foundation_decodes"):
        pytest.skip("Media Foundation cannot open the test clips here")
    return request.param

NEW_TYPES = {
    "Resize", "Imencode", "OpticalFlowFarneback", "BlobCounter",
    "MotionGrid", "FrameDiffMotion", "CameraCapture", "VideoFileCapture", "Imdecode",
}


def textured(h=90, w=160, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h // 6, w // 6), dtype=np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


# --- registration ------------------------------------------------------------

def test_portable_registration_has_no_io_nodes():
    registry = NodeRegistry()
    cv_nodes.register_portable_nodes(registry)
    names = set(registry.node_definitions)
    assert {"Resize", "Imencode", "OpticalFlowFarneback", "MotionGrid", "BlobCounter"} <= names
    assert not names & {"VideoFileCapture", "CameraCapture", "FaceDetectionYN"}


def test_full_registration_documents_new_ports():
    registry = NodeRegistry()
    cv_nodes.register_nodes(registry)
    sub = NodeRegistry()
    for name in NEW_TYPES:
        sub.register_node(name, registry.node_definitions[name].factory)
    assert_all_ports_documented(sub)


# --- Resize / Imencode --------------------------------------------------------

def test_resize_fixed_output_shape_and_channel_change():
    node = ResizeNode("r")
    assert node.outputs["dst"].value_type_list[0].shape == (180, 320, 3)
    node.inputs["src"].dispatch([np.zeros((2, 720, 1280, 3), np.uint8)])
    assert node.outputs["dst"].current_values[0].shape == (2, 180, 320, 3)

    node.set_properties({"channels": 1, "width": 64, "height": 32})
    assert node.inputs["src"].value_type_list[0].shape == (-1, -1)
    assert node.outputs["dst"].value_type_list[0].shape == (32, 64)
    node.inputs["src"].dispatch([np.zeros((1, 100, 200), np.uint8)])
    assert node.outputs["dst"].current_values[0].shape == (1, 32, 64)


def test_imencode_roundtrip():
    enc, dec = ImencodeNode("e"), ImdecodeNode("d")
    img = np.zeros((40, 60, 3), np.uint8)
    img[10:30, 20:40] = (0, 0, 255)
    enc.inputs["image"].dispatch([img[np.newaxis]])
    buf = enc.outputs["buf"].current_values[0]
    assert buf.ndim == 2 and buf.shape[0] == 1 and buf.shape[1] < img.size
    dec.inputs["buf"].dispatch([buf])
    out = dec.outputs["image"].current_values[0][0]
    assert out.shape == img.shape
    assert out[20, 30, 2] > 200 and out[20, 30, 0] < 50  # still red (BGR)


# --- MotionGrid / FrameDiffMotion --------------------------------------------------

def test_grid_mean_and_onsets():
    field = np.arange(24, dtype=np.float32).reshape(4, 6)
    assert grid_mean(field, 2, 3).shape == (2, 3)

    det = OnsetDetector((1,))
    fired = [bool(det.update(np.array([v]), 1.0, 0.5)[0]) for v in (0, 2, 2, 0.8, 2, 0.2, 2)]
    # Fires on the first rise; stays quiet while busy and after a partial dip;
    # re-arms only below 0.5.
    assert fired == [False, True, False, False, False, False, True]


def test_motion_grid_uniform_flow():
    node = MotionGridNode("g")
    node.set_properties({"smoothing": 0.0})
    flow = np.zeros((90, 160, 2), np.float32)
    flow[..., 0] = 2.0  # everything moves right
    node.inputs["flow"].dispatch([flow[np.newaxis]])

    energy = node.outputs["energy"].current_values[0][0]
    assert energy.shape == (6, 8)
    np.testing.assert_allclose(energy, 2.0, rtol=1e-5)
    assert node.outputs["onsets"].current_values[0][0].all()
    assert abs(node.outputs["direction"].current_values[0][0]) < 1e-6
    assert node.outputs["speed"].current_values[0][0] == pytest.approx(2.0)
    assert node.outputs["coherence"].current_values[0][0] == pytest.approx(1.0, abs=1e-4)

    node.inputs["flow"].dispatch([flow[np.newaxis]])
    assert not node.outputs["onsets"].current_values[0][0].any()  # still busy: no new onset


def test_motion_grid_random_flow_is_incoherent_and_resizes():
    node = MotionGridNode("g")
    rng = np.random.default_rng(1)
    flow = rng.normal(size=(90, 160, 2)).astype(np.float32)
    node.inputs["flow"].dispatch([flow[np.newaxis]])
    assert node.outputs["coherence"].current_values[0][0] < 0.1

    node.set_properties({"rows": 3, "cols": 4})
    assert node.outputs["energy"].value_type_list[0].shape == (3, 4)
    assert node.outputs["vectors"].value_type_list[0].shape == (3, 4, 2)
    node.inputs["flow"].dispatch([flow[np.newaxis]])
    assert node.outputs["vectors"].current_values[0].shape == (1, 3, 4, 2)


def test_frame_diff_motion_localizes_movement():
    node = FrameDiffMotionNode("d")
    node.set_properties({"rows": 2, "cols": 2, "smoothing": 0.0})
    a = np.zeros((60, 80), np.uint8)
    b = a.copy()
    b[5:25, 50:75] = 255  # change in the top-right cell only
    node.inputs["frame"].dispatch([a[np.newaxis]])
    assert node.outputs["energy"].current_values is None  # needs two frames
    node.inputs["frame"].dispatch([b[np.newaxis]])
    energy = node.outputs["energy"].current_values[0][0]
    assert energy[0, 1] > 0.1 and energy[0, 0] == 0 and energy[1, :].max() == 0


# --- OpenCV video nodes (inline) ---------------------------------------------------

def test_optical_flow_measures_shift_inline():
    node = OpticalFlowFarnebackNode("of")
    node.set_properties({"threaded": False})
    base = textured()
    shifted = np.roll(base, 3, axis=1)  # texture moves 3 px right
    node.inputs["frame"].dispatch([base[np.newaxis]])
    assert node.outputs["flow"].current_values is None
    node.inputs["frame"].dispatch([shifted[np.newaxis]])
    flow = node.outputs["flow"].current_values[0][0]
    assert flow.shape == (90, 160, 2)
    inner = flow[15:-15, 15:-15]
    assert np.median(inner[..., 0]) == pytest.approx(3.0, abs=0.5)
    assert abs(np.median(inner[..., 1])) < 0.5


def test_optical_flow_downscale_keeps_pixel_units():
    node = OpticalFlowFarnebackNode("of")
    node.set_properties({"threaded": False, "downscale": 2})
    base = textured(120, 200)
    node.inputs["frame"].dispatch([base[np.newaxis]])
    node.inputs["frame"].dispatch([np.roll(base, 4, axis=1)[np.newaxis]])
    flow = node.outputs["flow"].current_values[0][0]
    assert flow.shape == (60, 100, 2)
    assert np.median(flow[10:-10, 10:-10, 0]) == pytest.approx(4.0, abs=0.8)


def test_blob_counter_finds_moving_square():
    node = BlobCounterNode("b")
    node.set_properties({"threaded": False, "history": 50})
    bg = np.full((90, 160), 40, np.uint8)
    for _ in range(30):
        node.inputs["frame"].dispatch([bg[np.newaxis]])
    assert node.outputs["count"].current_values[0][0] == 0
    fg = bg.copy()
    fg[30:60, 60:100] = 220
    node.inputs["frame"].dispatch([fg[np.newaxis]])
    assert node.outputs["count"].current_values[0][0] == 1
    boxes = node.outputs["boxes"].current_values[0][0]
    x, y, w, h = boxes[0]
    assert x == pytest.approx(60 / 160, abs=0.05) and w == pytest.approx(40 / 160, abs=0.05)
    assert node.outputs["mask"].current_values[0].shape == (1, 90, 160)


# --- threaded paths ------------------------------------------------------------------

def test_optical_flow_threaded_delivers_on_loop():
    async def run():
        node = OpticalFlowFarnebackNode("of")
        node._do_activate()
        assert node._worker.running
        base = textured()
        node.inputs["frame"].dispatch([base[np.newaxis]])
        await asyncio.sleep(0.2)
        node.inputs["frame"].dispatch([np.roll(base, 2, axis=1)[np.newaxis]])
        for _ in range(100):
            await asyncio.sleep(0.02)
            if node.outputs["flow"].current_values is not None:
                break
        node._do_finalize()
        assert not node._worker.running
        return node.outputs["flow"].current_values

    values = asyncio.run(run())
    assert values is not None and values[0].shape == (1, 90, 160, 2)


class _FakeCamera:
    """Stands in for cv2.VideoCapture: 320x240 frames, about 200 fps."""

    def __init__(self, *args):
        self.n = 0

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        import time
        time.sleep(0.005)
        self.n += 1
        return True, np.full((240, 320, 3), self.n % 256, np.uint8)

    def release(self):
        pass


def test_camera_capture_resizes_and_delivers_latest(monkeypatch):
    from cv_nodes import videoio_nodes

    monkeypatch.setattr(videoio_nodes.cv2, "VideoCapture", _FakeCamera)

    async def run():
        node = videoio_nodes.CameraCaptureNode("cam")
        node.set_properties({"width": 64, "height": 48})
        node._do_activate()
        received = []
        node.outputs["frame"].subscribe(1, lambda v: received.append(v[0].shape))
        node.set_properties({"enabled": True})
        await asyncio.sleep(0.3)
        status = int(node.outputs["status"].current_values[0][0])
        node.set_properties({"enabled": False})
        await asyncio.sleep(0.05)
        node._do_finalize()
        return received, status, int(node.outputs["status"].current_values[0][0])

    received, status_running, status_after = asyncio.run(run())
    assert received and all(shape == (1, 48, 64, 3) for shape in received)
    assert status_running == 2 and status_after == 0


def test_video_file_capture_loops(tmp_path, backend):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, np.uint8))
    writer.release()

    async def run():
        node = VideoFileCaptureNode("v")
        node.set_properties({"width": 64, "height": 48, "loop": True, "backend": backend})
        node._do_activate()
        node.set_properties({"file_path": str(path)})
        for _ in range(100):
            await asyncio.sleep(0.02)
            if node._is_loaded:
                break
        positions = []
        port = node.outputs["position"]
        for _ in range(12):
            before = port.current_values
            node.inputs["readout_count"].dispatch([np.array([1], dtype=np.int32)])
            for _ in range(200):  # wait for the worker's delivery (new list object)
                await asyncio.sleep(0.005)
                if port.current_values is not before:
                    break
            positions.append(int(port.current_values[0][0]))
        eof = node.outputs["eof"].current_values
        node._do_finalize()
        return positions, eof

    positions, eof = asyncio.run(run())
    assert eof is None  # looping never reports EOF
    assert positions[:5] == [0, 1, 2, 3, 4]
    assert positions[5] == 0  # wrapped to the first frame


def _read_one(path, props, readout=1, backend="ffmpeg"):
    async def run():
        node = VideoFileCaptureNode("v")
        node.set_properties({**props, "backend": backend})
        node._do_activate()
        node.inputs["@file_path"].dispatch([np.array([str(path)])])
        for _ in range(100):
            await asyncio.sleep(0.02)
            if node._is_loaded:
                break
        port = node.outputs["frame"]
        before = port.current_values
        node.inputs["readout_count"].dispatch([np.array([readout], dtype=np.int32)])
        for _ in range(50):
            await asyncio.sleep(0.01)
            if port.current_values is not before:
                break
        values = port.current_values
        node._do_finalize()
        return values

    return asyncio.run(run())


def _write_clip(path, w, h):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    assert writer.isOpened()
    for _ in range(3):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, : w // 4] = (0, 0, 255)  # red band on the left quarter
        writer.write(frame)
    writer.release()


def test_video_file_capture_fits_other_sizes(tmp_path, backend):
    path = tmp_path / "wide.avi"
    _write_clip(path, 128, 48)  # 8:3, wider than the 4:3 output

    covered = _read_one(path, {"width": 64, "height": 48}, backend=backend)  # fit defaults to cover
    assert covered[0].shape == (1, 48, 64, 3)
    # cover crops the middle 64 px: the red left quarter (0-32 px) falls outside
    assert covered[0][0, 24, 2, 2] < 80

    stretched = _read_one(path, {"width": 64, "height": 48, "fit": "stretch"}, backend=backend)
    assert stretched[0].shape == (1, 48, 64, 3)
    assert stretched[0][0, 24, 2, 2] > 180  # red band kept at the left edge

    assert _read_one(path, {"width": 64, "height": 48, "fit": "exact"}, backend=backend) is None


# --- VideoFileCapture: file_path, readers, read requests ------------------------


def _gray_clip(path, levels=(0, 40, 80, 120, 160)):
    """MJPG clip whose frame i is a flat gray of levels[i] (both readers decode it)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for level in levels:
        writer.write(np.full((48, 64, 3), level, np.uint8))
    writer.release()


def _level(frame) -> int:
    return int(round(float(frame.mean()) / 40.0)) * 40  # MJPG is lossy


async def _until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        await asyncio.sleep(0.005)


def test_file_path_loads_once_and_empty_closes(tmp_path):
    path = tmp_path / "clip.avi"
    _gray_clip(path)

    async def run():
        node = VideoFileCaptureNode("v")
        node.set_properties({"width": 64, "height": 48, "backend": "ffmpeg"})
        statuses = []
        node.outputs["loading_status"].subscribe(1, lambda v: statuses.append(int(v[0][0])))
        node._do_activate()
        node.inputs["@file_path"].dispatch([np.array([str(path)])])
        await _until(lambda: node._is_loaded)
        node.inputs["@file_path"].dispatch([np.array([str(path)])])  # same file: no reload
        await asyncio.sleep(0.05)
        loaded = list(statuses)
        node.set_properties({"file_path": ""})
        await asyncio.sleep(0.05)
        closed = list(statuses)
        node._do_finalize()
        return loaded, closed, node.properties["file_path"]

    loaded, closed, file_path = asyncio.run(run())
    assert loaded == [1, 2]
    assert closed == [1, 2, 0]
    assert file_path == ""


def test_source_loops_without_hitting_the_end(tmp_path, backend):
    path = tmp_path / "clip.avi"
    _gray_clip(path)
    source = open_source(str(path), backend)
    assert source is not None and source.backend == backend
    levels, wraps = [], []
    for _ in range(12):
        frame, wrapped = source.read(loop=True)
        assert frame is not None
        levels.append(_level(frame))
        wraps.append(wrapped)
    assert levels == [0, 40, 80, 120, 160, 0, 40, 80, 120, 160, 0, 40]
    assert [i for i, w in enumerate(wraps) if w] == [5, 10]
    if backend == "media_foundation":
        assert not source._ended  # rewound before the end (reopening costs ~0.3 s)
    source.release()


def test_source_seeks_after_the_end(tmp_path, backend):
    path = tmp_path / "clip.avi"
    _gray_clip(path)
    source = open_source(str(path), backend)
    assert source is not None
    while source.read(loop=False)[0] is not None:
        pass
    assert source.read(loop=False) == (None, False)
    assert source.seek(2)
    frame, _ = source.read(loop=False)
    assert frame is not None and _level(frame) == 80
    source.release()


def test_source_falls_back_to_ffmpeg(tmp_path, monkeypatch):
    path = tmp_path / "clip.avi"
    _gray_clip(path)
    capture = _video_sources._capture
    monkeypatch.setattr(
        _video_sources, "_capture", lambda p, backend: None if backend == "media_foundation" else capture(p, backend)
    )
    source = open_source(str(path), "media_foundation")
    assert source is not None and source.backend == "ffmpeg"
    frame, _ = source.read(loop=False)
    assert frame is not None and _level(frame) == 0
    source.release()
    with pytest.raises(ValueError):
        open_source(str(path), "gstreamer")


def test_read_requests_beyond_one_waiting_are_dropped(tmp_path):
    path = tmp_path / "clip.avi"
    _gray_clip(path)

    async def run():
        node = VideoFileCaptureNode("v")
        node.set_properties({"width": 64, "height": 48, "backend": "ffmpeg", "file_path": str(path)})
        node._do_activate()
        await _until(lambda: node._is_loaded)
        gate = threading.Event()
        read_frame = node._read_frame

        def slow_read(source):
            gate.wait(2.0)  # the worker is busy until the test opens the gate
            return read_frame(source)

        node._read_frame = slow_read
        node._drop_report_t = time.monotonic()  # count the drops instead of logging them now
        positions = []
        node.outputs["position"].subscribe(1, lambda v: positions.append(int(v[0][0])))
        for _ in range(5):
            node.inputs["readout_count"].dispatch([np.array([1], dtype=np.int32)])
            await asyncio.sleep(0.02)  # the worker takes the first request meanwhile
        dropped = node._dropped_reads
        gate.set()
        await _until(lambda: len(positions) >= 2)
        await asyncio.sleep(0.1)
        node._do_finalize()
        return dropped, positions

    dropped, positions = asyncio.run(run())
    assert dropped == 3  # one read in progress, one waiting
    assert positions == [0, 1]  # in order, none skipped
