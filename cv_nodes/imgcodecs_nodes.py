# imgcodecs_nodes — OpenCV "imgcodecs" module (@defgroup imgcodecs in imgcodecs.hpp)
# imdecode, imencode, and related I/O functions.

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
from nonoha_core.node import BaseNode, NodeDoc, replace_node_input_port

from .imgproc_geometric_nodes import image_port_type


class ImdecodeNode(BaseNode):
    """Decode 1D uint8 encoded image bytes to uint8[H, W, 3] BGR image.

    Supports JPEG, PNG, BMP, WebP, and other formats recognized by cv2.imdecode.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description="Decodes encoded image bytes (JPEG, PNG, WebP, ...) to a BGR image",
            category="Image",
            inputs={"buf": "Encoded image bytes (one image per batch row)"},
            outputs={"image": "Decoded image (u8[H x W x 3], BGR)"},
        )

    def __init__(self, name: str):
        super().__init__(name, "Imdecode")

        # Input: encoded image bytes (1D uint8 array)
        input_buf = StaticTypeInputPort("buf", [PortValueType(DataType.Uint8, (-1,))])
        input_buf.subscribe(id(self), self._on_buf)
        self.inputs["buf"] = input_buf

        # Output: decoded image (uint8[H, W, 3])
        self._output_image = StaticTypeOutputPort(
            "image",
            [PortValueType(DataType.Uint8, (-1, -1, 3))],
        )
        self.outputs["image"] = self._output_image

    def _on_buf(self, values: list):
        buf_array = values[0]
        if buf_array is None or buf_array.size == 0:
            return

        for i in range(buf_array.shape[0]):
            img = cv2.imdecode(buf_array[i], cv2.IMREAD_COLOR)
            if img is not None:
                self._output_image.dispatch([img[np.newaxis]])


_FORMATS = {
    "jpg": (".jpg", cv2.IMWRITE_JPEG_QUALITY),
    "png": (".png", cv2.IMWRITE_PNG_COMPRESSION),
    "webp": (".webp", cv2.IMWRITE_WEBP_QUALITY),
}


class ImencodeNode(BaseNode):
    """Encode images to compressed bytes (JPEG/PNG/WebP).

    Useful before sending frames to a web app or a cloud service: a 480x270
    JPEG is tens of kilobytes instead of ~390 KB of raw pixels.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Encodes each image to JPEG, PNG or WebP bytes. Use it to send "
                "frames to apps or services without raw-pixel bandwidth."
            ),
            category="Image",
            inputs={"image": "Images to encode (u8, channels set by the channels property)"},
            outputs={"buf": "Encoded bytes, one image per dispatch (u8[?])"},
        )

    PROPERTIES = {
        "format": {"description": "jpg, png or webp", "default_value": "jpg"},
        "quality": {
            "description": "JPEG/WebP quality 1-100, or PNG compression 0-9",
            "default_value": 80,
        },
        "channels": {"description": "Input channels: 1 (gray), 3 (BGR) or 4 (BGRA)", "default_value": 3},
    }

    def __init__(self, name: str):
        super().__init__(name, "Imencode")
        self._channels = int(self.properties["channels"])
        self._ext, self._param = _FORMATS["jpg"]
        self._quality = int(self.properties["quality"])

        self._input = StaticTypeInputPort("image", [image_port_type(self._channels)])
        self._input.subscribe(id(self), self._on_image)
        self.inputs[self._input.name] = self._input

        self._output = StaticTypeOutputPort("buf", [PortValueType(DataType.Uint8, (-1,))])
        self.outputs[self._output.name] = self._output

    def _on_image(self, values: list[PortValue]):
        params = [self._param, self._quality]
        for img in values[0]:
            ok, buf = cv2.imencode(self._ext, img, params)
            if ok:
                self._output.dispatch([buf.reshape(1, -1)])

    def set_properties(self, properties: dict[str, PropertyDataType]):
        fmt = str(properties.get("format", self.properties["format"])).lower()
        if fmt not in _FORMATS:
            raise ValueError(f"format must be one of {sorted(_FORMATS)}")
        quality = int(properties.get("quality", self._quality))
        channels = int(properties.get("channels", self._channels))
        if channels not in (1, 3, 4):
            raise ValueError(f"channels must be 1, 3 or 4, got {channels}")

        if channels != self._channels:
            new_input = StaticTypeInputPort("image", [image_port_type(channels)])
            replace_node_input_port(self, self._input, new_input, self._on_image)
            self._input = new_input

        self._channels = channels
        self._ext, self._param = _FORMATS[fmt]
        self._quality = quality
        self.properties.update({"format": fmt, "quality": quality, "channels": channels})
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)
