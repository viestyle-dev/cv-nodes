# motion_np_nodes — motion features computed with numpy only (no OpenCV).
#
# MotionGrid summarizes a dense optical-flow field into a rows x cols grid.
# FrameDiffMotion is a lighter alternative that needs no optical flow (and
# therefore no OpenCV), e.g. for builds where OpenCV is unavailable.

from typing import ClassVar

import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc, PropertyDescription, replace_node_output_port
from nonoha_core.port import PortValue, StaticTypeInputPort, StaticTypeOutputPort

_EPS = 1e-6


def grid_mean(field: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean of ``field`` (H, W, ...) over a rows x cols grid -> (rows, cols, ...).

    Edge pixels that do not fill a whole cell are dropped. Callers make sure
    the field is at least rows x cols pixels.
    """
    h, w = field.shape[:2]
    ch, cw = h // rows, w // cols
    cropped = field[: ch * rows, : cw * cols]
    shaped = cropped.reshape(rows, ch, cols, cw, *field.shape[2:])
    # Sum the rows of each cell first (contiguous adds), then the few columns
    # left: about 10x faster than mean(axis=(1, 3)), which walks the array in
    # strides.
    # Like mean(): float fields keep their dtype, integer fields give float64.
    acc = shaped.dtype if shaped.dtype.kind == "f" else np.float64
    total = shaped.sum(axis=1, dtype=acc).sum(axis=2)
    total *= 1.0 / (ch * cw)
    return total


class OnsetDetector:
    """Per-cell rising-edge detector with hysteresis.

    A cell fires when its value rises above ``threshold`` after having been
    below ``threshold * rearm_ratio`` — so a cell that stays busy fires once,
    not on every frame.
    """

    def __init__(self, shape: tuple[int, ...]):
        self.armed = np.ones(shape, dtype=bool)

    def reset(self, shape: tuple[int, ...]):
        self.armed = np.ones(shape, dtype=bool)

    def update(self, values: np.ndarray, threshold: float, rearm_ratio: float) -> np.ndarray:
        onsets = self.armed & (values > threshold)
        self.armed = (self.armed & ~onsets) | (values < threshold * rearm_ratio)
        return onsets


def _f32(shape: tuple[int, ...] = ()) -> PortValueType:
    return PortValueType(DataType.Float32, shape)


class _GridNode(BaseNode):
    """Shared grid outputs: energy, onsets, mean_energy; rows/cols properties."""

    GRID_PROPERTIES: ClassVar[dict[str, PropertyDescription]] = {
        "rows": {"description": "Grid rows", "default_value": 6},
        "cols": {"description": "Grid columns", "default_value": 8},
        "onset_threshold": {
            "description": "Cell energy above which an onset fires",
            "default_value": 1.0,
        },
        "rearm_ratio": {
            "description": "A cell re-arms when its energy falls below threshold x this ratio",
            "default_value": 0.6,
        },
        "smoothing": {
            "description": "Exponential smoothing of cell energy, 0 (none) to 0.99",
            "default_value": 0.5,
        },
    }

    def _init_grid(self):
        p = self.properties
        self._rows = int(p["rows"])
        self._cols = int(p["cols"])
        # Mirrors of the per-frame properties (no dict lookups per frame).
        self._alpha = min(0.99, max(0.0, float(p["smoothing"])))
        self._threshold = float(p["onset_threshold"])
        self._rearm = float(p["rearm_ratio"])
        self._energy_state: np.ndarray | None = None
        self._onsets = OnsetDetector((self._rows, self._cols))
        self._out_energy = StaticTypeOutputPort("energy", [_f32((self._rows, self._cols))])
        self._out_onsets = StaticTypeOutputPort(
            "onsets", [PortValueType(DataType.Bool, (self._rows, self._cols))]
        )
        self._out_mean = StaticTypeOutputPort("mean_energy", [_f32()])
        for port in (self._out_energy, self._out_onsets, self._out_mean):
            self.outputs[port.name] = port

    def _smooth(self, energy: np.ndarray) -> np.ndarray:
        # The previous state was dispatched and may still be held downstream,
        # so it is never updated in place; ``energy`` is this frame's own array.
        alpha, state = self._alpha, self._energy_state
        if state is not None and alpha > 0:
            energy *= 1.0 - alpha
            energy += alpha * state
        self._energy_state = energy
        return energy

    def _dispatch_grid(self, energy: np.ndarray):
        energy = self._smooth(energy)
        onsets = self._onsets.update(energy, self._threshold, self._rearm)
        self._out_energy.dispatch([energy[np.newaxis]])
        self._out_onsets.dispatch([onsets[np.newaxis]])
        self._out_mean.dispatch([np.array([energy.mean()], dtype=np.float32)])

    def _set_grid_properties(self, properties: dict[str, PropertyDataType]) -> bool:
        """Apply grid properties; return True if the grid shape changed."""
        if "onset_threshold" in properties:
            self.properties["onset_threshold"] = self._threshold = float(properties["onset_threshold"])
        if "rearm_ratio" in properties:
            self.properties["rearm_ratio"] = self._rearm = float(properties["rearm_ratio"])
        if "smoothing" in properties:
            self.properties["smoothing"] = self._alpha = min(0.99, max(0.0, float(properties["smoothing"])))
        rows = max(1, int(properties.get("rows", self._rows)))
        cols = max(1, int(properties.get("cols", self._cols)))
        if (rows, cols) == (self._rows, self._cols):
            return False
        self._rows, self._cols = rows, cols
        self.properties.update({"rows": rows, "cols": cols})
        self._energy_state = None
        self._onsets.reset((rows, cols))
        for attr, new_port in (
            ("_out_energy", StaticTypeOutputPort("energy", [_f32((rows, cols))])),
            ("_out_onsets", StaticTypeOutputPort("onsets", [PortValueType(DataType.Bool, (rows, cols))])),
        ):
            replace_node_output_port(self, getattr(self, attr), new_port)
            setattr(self, attr, new_port)
        return True


class MotionGridNode(_GridNode):
    """Summarize a dense optical-flow field (H, W, 2) into a motion grid."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Summarizes a dense flow field into a rows x cols grid: motion "
                "energy and mean vector per cell, cell onsets, and the overall "
                "flow direction, speed and coherence."
            ),
            category="Motion",
            inputs={"flow": "Dense flow field f32[H x W x 2] (dx, dy in pixels per frame)"},
            outputs={
                "energy": "Mean flow magnitude per cell (smoothed)",
                "vectors": "Mean flow vector (dx, dy) per cell",
                "onsets": "True for cells whose energy just rose above the threshold",
                "mean_energy": "Mean of all cell energies",
                "direction": "Direction of the mean flow in radians (0 = right, pi/2 = down)",
                "speed": "Magnitude of the mean flow vector",
                "coherence": "speed / mean magnitude: 1 = everything moves the same way, 0 = random",
            },
        )

    PROPERTIES = _GridNode.GRID_PROPERTIES

    def __init__(self, name: str):
        super().__init__(name, "MotionGrid")
        self._input = StaticTypeInputPort("flow", [_f32((-1, -1, 2))])
        self._input.subscribe(id(self), self._on_flow)
        self.inputs[self._input.name] = self._input

        self._init_grid()
        self._out_vectors = StaticTypeOutputPort("vectors", [_f32((self._rows, self._cols, 2))])
        self._out_direction = StaticTypeOutputPort("direction", [_f32()])
        self._out_speed = StaticTypeOutputPort("speed", [_f32()])
        self._out_coherence = StaticTypeOutputPort("coherence", [_f32()])
        for port in (self._out_vectors, self._out_direction, self._out_speed, self._out_coherence):
            self.outputs[port.name] = port

    def _on_flow(self, values: list[PortValue]):
        for flow in values[0]:
            if flow.shape[0] < self._rows or flow.shape[1] < self._cols:
                continue
            magnitude = np.hypot(flow[..., 0], flow[..., 1])  # one temporary
            energy = grid_mean(magnitude, self._rows, self._cols)
            vectors = grid_mean(flow, self._rows, self._cols)

            # Row sums first: reshape(-1, 2).mean(0) reduces in strides.
            mean_vec = flow.sum(axis=0).sum(axis=0) * (1.0 / (flow.shape[0] * flow.shape[1]))
            speed = float(np.hypot(mean_vec[0], mean_vec[1]))
            mean_mag = float(magnitude.mean())
            direction = float(np.arctan2(mean_vec[1], mean_vec[0]))
            coherence = speed / (mean_mag + _EPS) if mean_mag > _EPS else 0.0

            self._dispatch_grid(energy)
            self._out_vectors.dispatch([vectors[np.newaxis]])
            self._out_direction.dispatch([np.array([direction], dtype=np.float32)])
            self._out_speed.dispatch([np.array([speed], dtype=np.float32)])
            self._out_coherence.dispatch([np.array([coherence], dtype=np.float32)])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        if self._set_grid_properties(properties):
            new_vectors = StaticTypeOutputPort("vectors", [_f32((self._rows, self._cols, 2))])
            replace_node_output_port(self, self._out_vectors, new_vectors)
            self._out_vectors = new_vectors
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


class FrameDiffMotionNode(_GridNode):
    """Motion energy grid from frame differences (no optical flow, no OpenCV)."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Measures motion as the mean absolute difference between "
                "consecutive gray frames, per grid cell. Lighter than optical "
                "flow and needs no OpenCV, but gives no direction."
            ),
            category="Motion",
            inputs={"frame": "Gray frames u8[H x W]"},
            outputs={
                "energy": "Mean absolute frame difference per cell, 0-1 (smoothed)",
                "onsets": "True for cells whose energy just rose above the threshold",
                "mean_energy": "Mean of all cell energies",
            },
        )

    PROPERTIES = {
        **_GridNode.GRID_PROPERTIES,
        "onset_threshold": {
            "description": "Cell energy (0-1) above which an onset fires",
            "default_value": 0.05,
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "FrameDiffMotion")
        self._input = StaticTypeInputPort("frame", [PortValueType(DataType.Uint8, (-1, -1))])
        self._input.subscribe(id(self), self._on_frame)
        self.inputs[self._input.name] = self._input
        self._init_grid()
        self._prev: np.ndarray | None = None

    def _on_frame(self, values: list[PortValue]):
        for frame in values[0]:
            current = np.multiply(frame, 1.0 / 255.0, dtype=np.float32)
            prev = self._prev
            self._prev = current
            if prev is None or prev.shape != current.shape:
                continue
            if current.shape[0] < self._rows or current.shape[1] < self._cols:
                continue
            diff = np.subtract(current, prev, out=prev)  # prev is no longer needed
            np.abs(diff, out=diff)
            self._dispatch_grid(grid_mean(diff, self._rows, self._cols))

    def set_properties(self, properties: dict[str, PropertyDataType]):
        self._set_grid_properties(properties)
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
