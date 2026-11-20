# imgproc_geometric_nodes — OpenCV "imgproc" geometric image transformations
# (@defgroup imgproc_transform in imgproc.hpp): resize and related functions.

import cv2
import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core.node import (
    BaseNode,
    NodeDoc,
    replace_node_input_port,
    replace_node_output_port,
)
from nonoha_core.port import PortValue, StaticTypeInputPort, StaticTypeOutputPort

_INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "cubic": cv2.INTER_CUBIC,
}


def image_port_type(channels: int, height: int = -1, width: int = -1) -> PortValueType:
    """u8 image type: (H, W) for one channel, (H, W, C) otherwise."""
    if channels == 1:
        return PortValueType(DataType.Uint8, (height, width))
    return PortValueType(DataType.Uint8, (height, width, channels))


class ResizeNode(BaseNode):
    """Resize images to a fixed size, so downstream ports can use exact shapes."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Resizes each image to width x height. The output has a fixed "
                "shape, so it can feed ports that require an exact size."
            ),
            category="Image",
            inputs={"src": "Images of any size (u8, channels set by the channels property)"},
            outputs={"dst": "Resized images (u8[height x width(x channels)])"},
        )

    PROPERTIES = {
        "width": {"description": "Output width in pixels", "data_type": "i32", "default_value": 320},
        "height": {"description": "Output height in pixels", "data_type": "i32", "default_value": 180},
        "channels": {"description": "Channels: 1 (gray), 3 (BGR/RGB) or 4", "data_type": "i32", "default_value": 3},
        "interpolation": {
            "description": "area, linear, nearest or cubic (area is best for shrinking)",
            "data_type": "str",
            "default_value": "area",
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "Resize")
        self._width = int(self.properties["width"])
        self._height = int(self.properties["height"])
        self._channels = int(self.properties["channels"])
        self._interp = _INTERPOLATIONS["area"]

        self._input = StaticTypeInputPort("src", [image_port_type(self._channels)])
        self._input.subscribe(id(self), self._on_src)
        self.inputs[self._input.name] = self._input

        self._output = StaticTypeOutputPort(
            "dst", [image_port_type(self._channels, self._height, self._width)]
        )
        self.outputs[self._output.name] = self._output

    def _on_src(self, values: list[PortValue]):
        batch = values[0]
        size = (self._width, self._height)
        if batch.shape[0] == 1:
            out = cv2.resize(batch[0], size, interpolation=self._interp)
            self._output.dispatch([out[np.newaxis]])
            return
        out = np.stack([cv2.resize(img, size, interpolation=self._interp) for img in batch])
        self._output.dispatch([out])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        width = max(1, int(properties.get("width", self._width)))
        height = max(1, int(properties.get("height", self._height)))
        channels = int(properties.get("channels", self._channels))
        if channels not in (1, 3, 4):
            raise ValueError(f"channels must be 1, 3 or 4, got {channels}")
        interp_name = str(properties.get("interpolation", self.properties["interpolation"]))
        if interp_name not in _INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {sorted(_INTERPOLATIONS)}")

        if channels != self._channels:
            new_input = StaticTypeInputPort("src", [image_port_type(channels)])
            replace_node_input_port(self, self._input, new_input, self._on_src)
            self._input = new_input
        if (width, height, channels) != (self._width, self._height, self._channels):
            new_output = StaticTypeOutputPort("dst", [image_port_type(channels, height, width)])
            replace_node_output_port(self, self._output, new_output)
            self._output = new_output

        self._width, self._height, self._channels = width, height, channels
        self._interp = _INTERPOLATIONS[interp_name]
        self.properties.update(
            {"width": width, "height": height, "channels": channels, "interpolation": interp_name}
        )
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
