# video_nodes — OpenCV "video" module (@defgroup video in video.hpp):
# motion analysis (optical flow) and background subtraction.
#
# Both nodes run their OpenCV work on a LatestWinsWorker thread so the event
# loop (which also produces audio) never waits for them.

import cv2
import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc, PropertyDescription
from nonoha_core.port import PortValue, StaticTypeInputPort, StaticTypeOutputPort

from ._worker import ThreadedComputeMixin

_THREADED_PROPERTY: PropertyDescription = {
    "description": "Run OpenCV on a background thread (latest frame wins). Off = inline",
    "data_type": "bool",
    "default_value": True,
}


def _gray_input(node: BaseNode, name: str, callback) -> StaticTypeInputPort:
    port = StaticTypeInputPort(name, [PortValueType(DataType.Uint8, (-1, -1))])
    port.subscribe(id(node), callback)
    node.inputs[port.name] = port
    return port


def _blob_stats(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel counts (N,) and boxes (N, 4: x, y, w, h) of the 8-connected blobs of a binary mask.

    Blobs come in raster order of their first pixel (top row first, then left).
    The same counts and boxes as ``cv2.connectedComponentsWithStats``, which
    splits even a 320x180 mask over OpenCV's thread pool (5-20 times the CPU of
    the work, waiting threads included). ``findContours`` runs on the calling
    thread; the stats come from the borders without a Python loop:

    - box: the extremes of the outer border
    - pixels (Pick's theorem on the border through pixel centers): on or
      inside an outer border A + B/2 + 1, strictly inside a hole border
      A - B/2 + 1 (A area, B lattice points on the border); a blob is its outer
      count minus its holes' (a blob inside a hole is a blob of its own)

    The GIL is held while OpenCV builds one array per border: ~1 ms for a few
    hundred blobs, ~10 ms for 30 000 (a 4K mask of noise).
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(0, np.int64), np.zeros((0, 4), np.int64)
    lengths = np.fromiter(map(len, contours), np.intp, len(contours))
    points = np.concatenate(contours).reshape(-1, 2).astype(np.int64)
    starts = np.zeros(len(contours), np.intp)
    np.cumsum(lengths[:-1], out=starts[1:])
    x, y = points[:, 0], points[:, 1]
    following = np.arange(1, len(points) + 1)
    following[starts + lengths - 1] = starts  # each border closes on its first point
    xn, yn = x[following], y[following]
    twice_area = np.abs(np.add.reduceat(x * yn - xn * y, starts))
    # Border segments run in the 8 chain directions: max(|dx|, |dy|) lattice steps each.
    boundary = np.add.reduceat(np.maximum(np.abs(xn - x), np.abs(yn - y)), starts)
    parent = hierarchy[0][:, 3]
    outer = parent < 0
    holes = ~outer
    hole_pixels = np.bincount(
        parent[holes], weights=((twice_area[holes] - boundary[holes]) // 2 + 1), minlength=len(contours)
    ).astype(np.int64)
    pixels = ((twice_area + boundary) // 2 + 1)[outer] - hole_pixels[outer]
    x0 = np.minimum.reduceat(x, starts)[outer]
    y0 = np.minimum.reduceat(y, starts)[outer]
    boxes = np.stack(
        [x0, y0, np.maximum.reduceat(x, starts)[outer] - x0 + 1, np.maximum.reduceat(y, starts)[outer] - y0 + 1],
        axis=1,
    )
    order = np.lexsort((x[starts][outer], y[starts][outer]))  # an outer border starts at its first pixel
    return pixels[order], boxes[order]


class OpticalFlowFarnebackNode(ThreadedComputeMixin, BaseNode):
    """Dense optical flow between consecutive gray frames (Farneback)."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Dense optical flow (Farneback) between consecutive gray frames. "
                "Outputs one (dx, dy) vector per pixel, in input-pixel units. "
                "Runs on a background thread; when frames arrive faster than it "
                "can process, the newest frame wins."
            ),
            category="Motion",
            inputs={"frame": "Gray frames u8[H x W]"},
            outputs={"flow": "Flow field f32[h x w x 2], h x w = input size / downscale"},
        )

    PROPERTIES = {
        "downscale": {"description": "Compute at 1/downscale resolution (1 = full)", "data_type": "i32", "default_value": 1},
        "pyr_scale": {"description": "Pyramid scale (<1)", "data_type": "f64", "default_value": 0.5},
        "levels": {"description": "Pyramid levels", "data_type": "i32", "default_value": 3},
        "winsize": {"description": "Averaging window size", "data_type": "i32", "default_value": 15},
        "iterations": {"description": "Iterations per pyramid level", "data_type": "i32", "default_value": 3},
        "poly_n": {"description": "Pixel neighborhood for the polynomial expansion (5 or 7)", "data_type": "i32", "default_value": 5},
        "poly_sigma": {"description": "Gaussian sigma for poly_n (1.1 for 5, 1.5 for 7)", "data_type": "f64", "default_value": 1.2},
        "threaded": _THREADED_PROPERTY,
    }

    def __init__(self, name: str):
        super().__init__(name, "OpticalFlowFarneback")
        self._input = _gray_input(self, "frame", self._on_frame)
        self._output = StaticTypeOutputPort(
            "flow", [PortValueType(DataType.Float32, (-1, -1, 2))]
        )
        self.outputs[self._output.name] = self._output
        self._prev: np.ndarray | None = None
        self._mirror()
        self._init_worker()

    def _mirror(self):
        # One tuple, replaced as a whole, so the worker thread always reads
        # a consistent set of parameters.
        p = self.properties
        self._downscale = max(1, int(p["downscale"]))
        self._params = (
            float(p["pyr_scale"]),
            int(p["levels"]),
            int(p["winsize"]),
            int(p["iterations"]),
            int(p["poly_n"]),
            float(p["poly_sigma"]),
        )

    def activate(self):
        self._sync_worker()

    def finalize(self):
        self._worker.stop()

    def _on_frame(self, values: list[PortValue]):
        batch = values[0]
        if self._worker.running:
            # Copy: upstream nodes may reuse their output buffers.
            self._submit(batch[-1].copy())
            return
        for frame in batch:
            self._submit(frame)

    def _compute(self, frame: np.ndarray) -> np.ndarray | None:
        downscale = self._downscale
        if downscale > 1:
            h, w = frame.shape
            frame = cv2.resize(
                frame, (max(1, w // downscale), max(1, h // downscale)), interpolation=cv2.INTER_AREA
            )
        prev = self._prev
        self._prev = frame
        if prev is None or prev.shape != frame.shape:
            return None
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma = self._params
        # flow=None lets OpenCV allocate the result: a reused buffer would
        # overwrite flow fields already dispatched. The stubs do not allow None.
        flow = cv2.calcOpticalFlowFarneback(  # type: ignore
            prev, frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0  # type: ignore
        )
        if downscale > 1:
            flow *= downscale
        return flow

    def _emit(self, flow: np.ndarray):
        self._output.dispatch([flow[np.newaxis]])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        for key in ("downscale", "levels", "winsize", "iterations", "poly_n"):
            if key in properties:
                self.properties[key] = int(properties[key])
        for key in ("pyr_scale", "poly_sigma"):
            if key in properties:
                self.properties[key] = float(properties[key])
        if "downscale" in properties:
            self._prev = None
        self._mirror()
        if "threaded" in properties:
            self.properties["threaded"] = bool(properties["threaded"])
            self._sync_worker()
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


class BlobCounterNode(ThreadedComputeMixin, BaseNode):
    """Count moving objects with background subtraction (MOG2)."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Counts moving objects (people, cars, ...) by background "
                "subtraction (MOG2) and connected components. Runs on a "
                "background thread; the newest frame wins."
            ),
            category="Motion",
            inputs={"frame": "Gray frames u8[H x W]"},
            outputs={
                "count": "Number of moving blobs",
                "boxes": "Blob boxes f32[N x 4]: x, y, w, h normalized to 0-1, in raster order of each blob's first pixel",
                "mask": "Foreground mask u8[H x W] (255 = moving)",
            },
        )

    PROPERTIES = {
        "min_area": {
            "description": "Smallest blob as a fraction of the frame area",
            "data_type": "f64",
            "default_value": 0.002,
        },
        "history": {"description": "Background model history in frames", "data_type": "i32", "default_value": 300},
        "var_threshold": {"description": "MOG2 variance threshold", "data_type": "f64", "default_value": 25.0},
        "learning_rate": {
            "description": "Background learning rate 0-1, or -1 for automatic",
            "data_type": "f64",
            "default_value": -1.0,
        },
        "threaded": _THREADED_PROPERTY,
    }

    def __init__(self, name: str):
        super().__init__(name, "BlobCounter")
        self._input = _gray_input(self, "frame", self._on_frame)
        self._out_count = StaticTypeOutputPort("count", [PortValueType(DataType.Int32)])
        self._out_boxes = StaticTypeOutputPort(
            "boxes", [PortValueType(DataType.Float32, (-1, 4))]
        )
        self._out_mask = StaticTypeOutputPort("mask", [PortValueType(DataType.Uint8, (-1, -1))])
        for port in (self._out_count, self._out_boxes, self._out_mask):
            self.outputs[port.name] = port
        self._subtractor = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._count = np.zeros(1, dtype=np.int32)
        p = self.properties
        self._min_area = float(p["min_area"])
        self._learning_rate = float(p["learning_rate"])
        self._history = int(p["history"])
        self._var_threshold = float(p["var_threshold"])
        self._init_worker()

    def activate(self):
        self._sync_worker()

    def finalize(self):
        self._worker.stop()

    def _on_frame(self, values: list[PortValue]):
        batch = values[0]
        if self._worker.running:
            self._submit(batch[-1].copy())
            return
        for frame in batch:
            self._submit(frame)

    def _compute(self, frame: np.ndarray):
        subtractor = self._subtractor
        if subtractor is None:
            subtractor = self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self._history, varThreshold=self._var_threshold, detectShadows=True
            )
        fg = subtractor.apply(frame, learningRate=self._learning_rate)
        # MOG2 marks shadows as 127; keep only confident foreground.
        _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        h, w = mask.shape
        pixels, boxes = _blob_stats(mask)
        kept = boxes[pixels >= self._min_area * h * w]
        box_array = (kept / np.array([w, h, w, h], dtype=np.float64)).astype(np.float32)
        return len(kept), box_array, mask

    def _emit(self, result):
        count, boxes, mask = result
        self._count[0] = count  # valid during the dispatch: receivers copy what they keep
        self._out_count.dispatch([self._count])
        self._out_boxes.dispatch([boxes[np.newaxis]])
        self._out_mask.dispatch([mask[np.newaxis]])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        if "min_area" in properties:
            self.properties["min_area"] = self._min_area = float(properties["min_area"])
        if "learning_rate" in properties:
            self.properties["learning_rate"] = self._learning_rate = float(properties["learning_rate"])
        rebuild = False
        if "history" in properties:
            self.properties["history"] = self._history = int(properties["history"])
            rebuild = True
        if "var_threshold" in properties:
            self.properties["var_threshold"] = self._var_threshold = float(properties["var_threshold"])
            rebuild = True
        if rebuild:
            # Rebuilt lazily by whichever thread computes next.
            self._subtractor = None
        if "threaded" in properties:
            self.properties["threaded"] = bool(properties["threaded"])
            self._sync_worker()
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
