# imgproc_color_conversions_nodes — OpenCV "imgproc" module,
# @defgroup imgproc_color_conversions (imgproc.hpp)
# cvtColor, applyColorMap, and related color space conversion functions.

import logging

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
from nonoha_core.node import BaseNode, replace_node_input_port, replace_node_output_port

logger = logging.getLogger(__name__)

# Sentinel value for planar YUV 4:2:0 shape: (H*3//2, W)
_PLANAR_420 = -420

# (cv2_code, in_channels, out_channels, batchable)
# in_channels / out_channels: positive int = interleaved channel count,
#                             _PLANAR_420 = YUV 4:2:0 planar
# batchable: True = per-pixel (batch reshape safe), False = spatial/planar
# fmt: off
CONVERSIONS: dict[str, tuple[int, int, int, bool]] = {
    # =========================================================================
    # BGR/RGB/Gray reorder & alpha
    # =========================================================================
    "BGR2BGRA":  (cv2.COLOR_BGR2BGRA,  3, 4, True),
    "BGRA2BGR":  (cv2.COLOR_BGRA2BGR,  4, 3, True),
    "BGR2RGBA":  (cv2.COLOR_BGR2RGBA,  3, 4, True),
    "RGBA2BGR":  (cv2.COLOR_RGBA2BGR,  4, 3, True),
    "BGR2RGB":   (cv2.COLOR_BGR2RGB,   3, 3, True),
    "BGRA2RGBA": (cv2.COLOR_BGRA2RGBA, 4, 4, True),
    "BGR2GRAY":  (cv2.COLOR_BGR2GRAY,  3, 1, True),
    "RGB2GRAY":  (cv2.COLOR_RGB2GRAY,  3, 1, True),
    "GRAY2BGR":  (cv2.COLOR_GRAY2BGR,  1, 3, True),
    "GRAY2BGRA": (cv2.COLOR_GRAY2BGRA, 1, 4, True),
    "BGRA2GRAY": (cv2.COLOR_BGRA2GRAY, 4, 1, True),
    "RGBA2GRAY": (cv2.COLOR_RGBA2GRAY, 4, 1, True),
    # =========================================================================
    # BGR565 (16-bit packed RGB, stored as (H,W,2) uint8)
    # =========================================================================
    "BGR2BGR565":   (cv2.COLOR_BGR2BGR565,   3, 2, True),
    "RGB2BGR565":   (cv2.COLOR_RGB2BGR565,   3, 2, True),
    "BGR5652BGR":   (cv2.COLOR_BGR5652BGR,   2, 3, True),
    "BGR5652RGB":   (cv2.COLOR_BGR5652RGB,   2, 3, True),
    "BGRA2BGR565":  (cv2.COLOR_BGRA2BGR565,  4, 2, True),
    "RGBA2BGR565":  (cv2.COLOR_RGBA2BGR565,  4, 2, True),
    "BGR5652BGRA":  (cv2.COLOR_BGR5652BGRA,  2, 4, True),
    "BGR5652RGBA":  (cv2.COLOR_BGR5652RGBA,  2, 4, True),
    "GRAY2BGR565":  (cv2.COLOR_GRAY2BGR565,  1, 2, True),
    "BGR5652GRAY":  (cv2.COLOR_BGR5652GRAY,  2, 1, True),
    # =========================================================================
    # BGR555 (16-bit packed RGB, stored as (H,W,2) uint8)
    # =========================================================================
    "BGR2BGR555":   (cv2.COLOR_BGR2BGR555,   3, 2, True),
    "RGB2BGR555":   (cv2.COLOR_RGB2BGR555,   3, 2, True),
    "BGR5552BGR":   (cv2.COLOR_BGR5552BGR,   2, 3, True),
    "BGR5552RGB":   (cv2.COLOR_BGR5552RGB,   2, 3, True),
    "BGRA2BGR555":  (cv2.COLOR_BGRA2BGR555,  4, 2, True),
    "RGBA2BGR555":  (cv2.COLOR_RGBA2BGR555,  4, 2, True),
    "BGR5552BGRA":  (cv2.COLOR_BGR5552BGRA,  2, 4, True),
    "BGR5552RGBA":  (cv2.COLOR_BGR5552RGBA,  2, 4, True),
    "GRAY2BGR555":  (cv2.COLOR_GRAY2BGR555,  1, 2, True),
    "BGR5552GRAY":  (cv2.COLOR_BGR5552GRAY,  2, 1, True),
    # =========================================================================
    # XYZ
    # =========================================================================
    "BGR2XYZ": (cv2.COLOR_BGR2XYZ, 3, 3, True),
    "RGB2XYZ": (cv2.COLOR_RGB2XYZ, 3, 3, True),
    "XYZ2BGR": (cv2.COLOR_XYZ2BGR, 3, 3, True),
    "XYZ2RGB": (cv2.COLOR_XYZ2RGB, 3, 3, True),
    # =========================================================================
    # YCrCb
    # =========================================================================
    "BGR2YCrCb": (cv2.COLOR_BGR2YCrCb, 3, 3, True),
    "RGB2YCrCb": (cv2.COLOR_RGB2YCrCb, 3, 3, True),
    "YCrCb2BGR": (cv2.COLOR_YCrCb2BGR, 3, 3, True),
    "YCrCb2RGB": (cv2.COLOR_YCrCb2RGB, 3, 3, True),
    # =========================================================================
    # HSV
    # =========================================================================
    "BGR2HSV":      (cv2.COLOR_BGR2HSV,      3, 3, True),
    "RGB2HSV":      (cv2.COLOR_RGB2HSV,      3, 3, True),
    "HSV2BGR":      (cv2.COLOR_HSV2BGR,      3, 3, True),
    "HSV2RGB":      (cv2.COLOR_HSV2RGB,      3, 3, True),
    "BGR2HSV_FULL": (cv2.COLOR_BGR2HSV_FULL, 3, 3, True),
    "RGB2HSV_FULL": (cv2.COLOR_RGB2HSV_FULL, 3, 3, True),
    "HSV2BGR_FULL": (cv2.COLOR_HSV2BGR_FULL, 3, 3, True),
    "HSV2RGB_FULL": (cv2.COLOR_HSV2RGB_FULL, 3, 3, True),
    # =========================================================================
    # HLS
    # =========================================================================
    "BGR2HLS":      (cv2.COLOR_BGR2HLS,      3, 3, True),
    "RGB2HLS":      (cv2.COLOR_RGB2HLS,      3, 3, True),
    "HLS2BGR":      (cv2.COLOR_HLS2BGR,      3, 3, True),
    "HLS2RGB":      (cv2.COLOR_HLS2RGB,      3, 3, True),
    "BGR2HLS_FULL": (cv2.COLOR_BGR2HLS_FULL, 3, 3, True),
    "RGB2HLS_FULL": (cv2.COLOR_RGB2HLS_FULL, 3, 3, True),
    "HLS2BGR_FULL": (cv2.COLOR_HLS2BGR_FULL, 3, 3, True),
    "HLS2RGB_FULL": (cv2.COLOR_HLS2RGB_FULL, 3, 3, True),
    # =========================================================================
    # Lab (CIE L*a*b*)
    # =========================================================================
    "BGR2Lab":  (cv2.COLOR_BGR2Lab,  3, 3, True),
    "RGB2Lab":  (cv2.COLOR_RGB2Lab,  3, 3, True),
    "Lab2BGR":  (cv2.COLOR_Lab2BGR,  3, 3, True),
    "Lab2RGB":  (cv2.COLOR_Lab2RGB,  3, 3, True),
    "LBGR2Lab": (cv2.COLOR_LBGR2Lab, 3, 3, True),
    "LRGB2Lab": (cv2.COLOR_LRGB2Lab, 3, 3, True),
    "Lab2LBGR": (cv2.COLOR_Lab2LBGR, 3, 3, True),
    "Lab2LRGB": (cv2.COLOR_Lab2LRGB, 3, 3, True),
    # =========================================================================
    # Luv (CIE L*u*v*)
    # =========================================================================
    "BGR2Luv":  (cv2.COLOR_BGR2Luv,  3, 3, True),
    "RGB2Luv":  (cv2.COLOR_RGB2Luv,  3, 3, True),
    "Luv2BGR":  (cv2.COLOR_Luv2BGR,  3, 3, True),
    "Luv2RGB":  (cv2.COLOR_Luv2RGB,  3, 3, True),
    "LBGR2Luv": (cv2.COLOR_LBGR2Luv, 3, 3, True),
    "LRGB2Luv": (cv2.COLOR_LRGB2Luv, 3, 3, True),
    "Luv2LBGR": (cv2.COLOR_Luv2LBGR, 3, 3, True),
    "Luv2LRGB": (cv2.COLOR_Luv2LRGB, 3, 3, True),
    # =========================================================================
    # YUV interleaved (3ch <-> 3ch)
    # =========================================================================
    "BGR2YUV": (cv2.COLOR_BGR2YUV, 3, 3, True),
    "RGB2YUV": (cv2.COLOR_RGB2YUV, 3, 3, True),
    "YUV2BGR": (cv2.COLOR_YUV2BGR, 3, 3, True),
    "YUV2RGB": (cv2.COLOR_YUV2RGB, 3, 3, True),
    # =========================================================================
    # Packed YUV 4:2:2 decode — input (H,W,2), row-independent, batchable
    # =========================================================================
    "YUV2RGB_UYVY":  (cv2.COLOR_YUV2RGB_UYVY,  2, 3, True),
    "YUV2BGR_UYVY":  (cv2.COLOR_YUV2BGR_UYVY,  2, 3, True),
    "YUV2RGBA_UYVY": (cv2.COLOR_YUV2RGBA_UYVY, 2, 4, True),
    "YUV2BGRA_UYVY": (cv2.COLOR_YUV2BGRA_UYVY, 2, 4, True),
    "YUV2RGB_YUY2":  (cv2.COLOR_YUV2RGB_YUY2,  2, 3, True),
    "YUV2BGR_YUY2":  (cv2.COLOR_YUV2BGR_YUY2,  2, 3, True),
    "YUV2RGB_YVYU":  (cv2.COLOR_YUV2RGB_YVYU,  2, 3, True),
    "YUV2BGR_YVYU":  (cv2.COLOR_YUV2BGR_YVYU,  2, 3, True),
    "YUV2RGBA_YUY2": (cv2.COLOR_YUV2RGBA_YUY2, 2, 4, True),
    "YUV2BGRA_YUY2": (cv2.COLOR_YUV2BGRA_YUY2, 2, 4, True),
    "YUV2RGBA_YVYU": (cv2.COLOR_YUV2RGBA_YVYU, 2, 4, True),
    "YUV2BGRA_YVYU": (cv2.COLOR_YUV2BGRA_YVYU, 2, 4, True),
    "YUV2GRAY_UYVY": (cv2.COLOR_YUV2GRAY_UYVY, 2, 1, True),
    "YUV2GRAY_YUY2": (cv2.COLOR_YUV2GRAY_YUY2, 2, 1, True),
    # =========================================================================
    # Packed YUV 4:2:2 encode — output (H,W,2), row-independent, batchable
    # =========================================================================
    "RGB2YUV_UYVY":  (cv2.COLOR_RGB2YUV_UYVY,  3, 2, True),
    "BGR2YUV_UYVY":  (cv2.COLOR_BGR2YUV_UYVY,  3, 2, True),
    "RGBA2YUV_UYVY": (cv2.COLOR_RGBA2YUV_UYVY, 4, 2, True),
    "BGRA2YUV_UYVY": (cv2.COLOR_BGRA2YUV_UYVY, 4, 2, True),
    "RGB2YUV_YUY2":  (cv2.COLOR_RGB2YUV_YUY2,  3, 2, True),
    "BGR2YUV_YUY2":  (cv2.COLOR_BGR2YUV_YUY2,  3, 2, True),
    "RGB2YUV_YVYU":  (cv2.COLOR_RGB2YUV_YVYU,  3, 2, True),
    "BGR2YUV_YVYU":  (cv2.COLOR_BGR2YUV_YVYU,  3, 2, True),
    "RGBA2YUV_YUY2": (cv2.COLOR_RGBA2YUV_YUY2, 4, 2, True),
    "BGRA2YUV_YUY2": (cv2.COLOR_BGRA2YUV_YUY2, 4, 2, True),
    "RGBA2YUV_YVYU": (cv2.COLOR_RGBA2YUV_YVYU, 4, 2, True),
    "BGRA2YUV_YVYU": (cv2.COLOR_BGRA2YUV_YVYU, 4, 2, True),
    # =========================================================================
    # mRGBA (premultiplied alpha)
    # =========================================================================
    "RGBA2mRGBA": (cv2.COLOR_RGBA2mRGBA, 4, 4, True),
    "mRGBA2RGBA": (cv2.COLOR_mRGBA2RGBA, 4, 4, True),
    # =========================================================================
    # Bayer demosaic — standard (bilinear interpolation)
    # NOT batchable: spatial interpolation across rows
    # =========================================================================
    "BayerBG2BGR":  (cv2.COLOR_BayerBG2BGR,  1, 3, False),
    "BayerGB2BGR":  (cv2.COLOR_BayerGB2BGR,  1, 3, False),
    "BayerRG2BGR":  (cv2.COLOR_BayerRG2BGR,  1, 3, False),
    "BayerGR2BGR":  (cv2.COLOR_BayerGR2BGR,  1, 3, False),
    # =========================================================================
    # Bayer demosaic — VNG (Variable Number of Gradients)
    # =========================================================================
    "BayerBG2BGR_VNG": (cv2.COLOR_BayerBG2BGR_VNG, 1, 3, False),
    "BayerGB2BGR_VNG": (cv2.COLOR_BayerGB2BGR_VNG, 1, 3, False),
    "BayerRG2BGR_VNG": (cv2.COLOR_BayerRG2BGR_VNG, 1, 3, False),
    "BayerGR2BGR_VNG": (cv2.COLOR_BayerGR2BGR_VNG, 1, 3, False),
    # =========================================================================
    # Bayer demosaic — EA (Edge-Aware)
    # =========================================================================
    "BayerBG2BGR_EA": (cv2.COLOR_BayerBG2BGR_EA, 1, 3, False),
    "BayerGB2BGR_EA": (cv2.COLOR_BayerGB2BGR_EA, 1, 3, False),
    "BayerRG2BGR_EA": (cv2.COLOR_BayerRG2BGR_EA, 1, 3, False),
    "BayerGR2BGR_EA": (cv2.COLOR_BayerGR2BGR_EA, 1, 3, False),
    # =========================================================================
    # Bayer demosaic — to GRAY
    # =========================================================================
    "BayerBG2GRAY": (cv2.COLOR_BayerBG2GRAY, 1, 1, False),
    "BayerGB2GRAY": (cv2.COLOR_BayerGB2GRAY, 1, 1, False),
    "BayerRG2GRAY": (cv2.COLOR_BayerRG2GRAY, 1, 1, False),
    "BayerGR2GRAY": (cv2.COLOR_BayerGR2GRAY, 1, 1, False),
    # =========================================================================
    # Bayer demosaic — to BGRA/RGBA (4-channel output)
    # =========================================================================
    "BayerBG2BGRA": (cv2.COLOR_BayerBG2BGRA, 1, 4, False),
    "BayerGB2BGRA": (cv2.COLOR_BayerGB2BGRA, 1, 4, False),
    "BayerRG2BGRA": (cv2.COLOR_BayerRG2BGRA, 1, 4, False),
    "BayerGR2BGRA": (cv2.COLOR_BayerGR2BGRA, 1, 4, False),
    # =========================================================================
    # Planar YUV 4:2:0 decode — input (H*3//2, W)
    # NOT batchable: plane layout depends on H
    # =========================================================================
    "YUV2RGB_NV12":  (cv2.COLOR_YUV2RGB_NV12,  _PLANAR_420, 3, False),
    "YUV2BGR_NV12":  (cv2.COLOR_YUV2BGR_NV12,  _PLANAR_420, 3, False),
    "YUV2RGBA_NV12": (cv2.COLOR_YUV2RGBA_NV12, _PLANAR_420, 4, False),
    "YUV2BGRA_NV12": (cv2.COLOR_YUV2BGRA_NV12, _PLANAR_420, 4, False),
    "YUV2RGB_NV21":  (cv2.COLOR_YUV2RGB_NV21,  _PLANAR_420, 3, False),
    "YUV2BGR_NV21":  (cv2.COLOR_YUV2BGR_NV21,  _PLANAR_420, 3, False),
    "YUV2RGBA_NV21": (cv2.COLOR_YUV2RGBA_NV21, _PLANAR_420, 4, False),
    "YUV2BGRA_NV21": (cv2.COLOR_YUV2BGRA_NV21, _PLANAR_420, 4, False),
    "YUV2RGB_YV12":  (cv2.COLOR_YUV2RGB_YV12,  _PLANAR_420, 3, False),
    "YUV2BGR_YV12":  (cv2.COLOR_YUV2BGR_YV12,  _PLANAR_420, 3, False),
    "YUV2RGBA_YV12": (cv2.COLOR_YUV2RGBA_YV12, _PLANAR_420, 4, False),
    "YUV2BGRA_YV12": (cv2.COLOR_YUV2BGRA_YV12, _PLANAR_420, 4, False),
    "YUV2RGB_IYUV":  (cv2.COLOR_YUV2RGB_IYUV,  _PLANAR_420, 3, False),
    "YUV2BGR_IYUV":  (cv2.COLOR_YUV2BGR_IYUV,  _PLANAR_420, 3, False),
    "YUV2RGBA_IYUV": (cv2.COLOR_YUV2RGBA_IYUV, _PLANAR_420, 4, False),
    "YUV2BGRA_IYUV": (cv2.COLOR_YUV2BGRA_IYUV, _PLANAR_420, 4, False),
    "YUV2GRAY_420":  (cv2.COLOR_YUV2GRAY_420,  _PLANAR_420, 1, False),
    # =========================================================================
    # Planar YUV 4:2:0 encode — output (H*3//2, W)
    # =========================================================================
    "RGB2YUV_I420":  (cv2.COLOR_RGB2YUV_I420,  3, _PLANAR_420, False),
    "BGR2YUV_I420":  (cv2.COLOR_BGR2YUV_I420,  3, _PLANAR_420, False),
    "RGBA2YUV_I420": (cv2.COLOR_RGBA2YUV_I420, 4, _PLANAR_420, False),
    "BGRA2YUV_I420": (cv2.COLOR_BGRA2YUV_I420, 4, _PLANAR_420, False),
    "RGB2YUV_YV12":  (cv2.COLOR_RGB2YUV_YV12,  3, _PLANAR_420, False),
    "BGR2YUV_YV12":  (cv2.COLOR_BGR2YUV_YV12,  3, _PLANAR_420, False),
    "RGBA2YUV_YV12": (cv2.COLOR_RGBA2YUV_YV12, 4, _PLANAR_420, False),
    "BGRA2YUV_YV12": (cv2.COLOR_BGRA2YUV_YV12, 4, _PLANAR_420, False),
}
# fmt: on


def _make_port_shape(width: int, height: int, channels: int) -> tuple[int, ...]:
    if channels == _PLANAR_420:
        return (height * 3 // 2 if height > 0 else -1, width)
    if channels == 1:
        return (height, width)
    return (height, width, channels)


class CvtColorNode(BaseNode):
    PROPERTIES = {
        "code": {
            "description": "Conversion code (e.g. 'BGR2RGBA')",
            "data_type": "str",
            "default_value": "BGR2RGBA",
        },
        "width": {"description": "Frame width", "data_type": "i32", "default_value": 1280},
        "height": {"description": "Frame height", "data_type": "i32", "default_value": 720},
    }

    def __init__(self, name: str):
        super().__init__(name, "CvtColor")

        code = str(self.properties["code"])
        width = int(self.properties["width"])
        height = int(self.properties["height"])

        cv2_code, in_ch, out_ch, batchable = CONVERSIONS[code]
        in_frame_shape = _make_port_shape(width, height, in_ch)
        out_frame_shape = _make_port_shape(width, height, out_ch)

        # --- Ports ---
        input_src = StaticTypeInputPort(
            "src", [PortValueType(DataType.Uint8, in_frame_shape)]
        )
        input_src.subscribe(id(self), self._on_input_src)

        output_dst = StaticTypeOutputPort(
            "dst", [PortValueType(DataType.Uint8, out_frame_shape)]
        )

        # --- Store references ---
        self._input_src = input_src
        self._output_dst = output_dst

        self._code = code
        self._cv2_code = cv2_code
        self._in_ch = in_ch
        self._out_ch = out_ch
        self._batchable = batchable
        self._in_frame_shape = in_frame_shape
        self._out_frame_shape = out_frame_shape
        self._width = width
        self._height = height
        self._dst_buffer: np.ndarray | None = None

        # --- Register ports ---
        self.inputs[input_src.name] = input_src
        self.outputs[output_dst.name] = output_dst

    def set_properties(self, properties: dict[str, PropertyDataType]):
        # 1. Extract properties
        code = self._code
        if "code" in properties:
            code = str(properties["code"])
            if code not in CONVERSIONS:
                raise ValueError(f"Unknown conversion code: {code}")

        width = self._width
        if "width" in properties:
            width = int(properties["width"])
            if width == 0:
                raise ValueError("width must be -1 (dynamic) or a positive integer")

        height = self._height
        if "height" in properties:
            height = int(properties["height"])
            if height == 0:
                raise ValueError("height must be -1 (dynamic) or a positive integer")

        cv2_code, in_ch, out_ch, batchable = CONVERSIONS[code]
        new_in_shape = _make_port_shape(width, height, in_ch)
        new_out_shape = _make_port_shape(width, height, out_ch)

        # 2. Prepare new ports (pure, no side effects)
        new_input_src = self._input_src
        if new_in_shape != self._in_frame_shape:
            new_input_src = StaticTypeInputPort(
                "src", [PortValueType(DataType.Uint8, new_in_shape)]
            )

        new_output_dst = self._output_dst
        if new_out_shape != self._out_frame_shape:
            new_output_dst = StaticTypeOutputPort(
                "dst", [PortValueType(DataType.Uint8, new_out_shape)]
            )

        # 3. Update internal state
        #   3a. Replace ports
        if new_input_src != self._input_src:
            replace_node_input_port(
                self, self._input_src, new_input_src, self._on_input_src
            )
            self._input_src = new_input_src

        if new_output_dst != self._output_dst:
            replace_node_output_port(self, self._output_dst, new_output_dst)
            self._output_dst = new_output_dst

        #   3b. Update state variables
        self._code = code
        self._cv2_code = cv2_code
        self._in_ch = in_ch
        self._out_ch = out_ch
        self._batchable = batchable
        self._in_frame_shape = new_in_shape
        self._out_frame_shape = new_out_shape
        self._width = width
        self._height = height
        self._dst_buffer = None

        #   3c. Update properties dict
        self.properties["code"] = code
        self.properties["width"] = width
        self.properties["height"] = height

        # 4. Notify graph
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_input_src(self, values: list[PortValue]):
        src = values[0]
        batch_size = src.shape[0]

        # width/height of -1: take the frame size from the input
        if self._width == -1 or self._height == -1:
            actual_h = src.shape[1]
            actual_w = src.shape[2]
            out_shape = _make_port_shape(actual_w, actual_h, self._out_ch)

            if out_shape != self._out_frame_shape:
                self._out_frame_shape = out_shape
                self._dst_buffer = None

        if self._batchable:
            result = self._process_batch(src, batch_size)
        else:
            result = self._process_per_frame(src, batch_size)

        self._output_dst.dispatch([result])

    def _process_batch(self, src: np.ndarray, batch_size: int) -> np.ndarray:
        """Batch reshape path for per-pixel conversions."""
        out_shape = self._out_frame_shape

        # Flatten: (N, H, W, C) -> (N*H, W, C), (N, H, W) -> (N*H, W)
        flat_src = src.reshape(-1, *src.shape[2:])
        flat_dst_shape = (batch_size * out_shape[0], *out_shape[1:])

        if self._dst_buffer is None or self._dst_buffer.shape != flat_dst_shape:
            self._dst_buffer = np.empty(flat_dst_shape, dtype=np.uint8)

        cv2.cvtColor(flat_src, self._cv2_code, dst=self._dst_buffer)
        return self._dst_buffer.reshape(batch_size, *out_shape)

    def _process_per_frame(self, src: np.ndarray, batch_size: int) -> np.ndarray:
        """Per-frame loop for non-batchable conversions (Bayer, planar YUV)."""
        out_shape = self._out_frame_shape
        batch_dst_shape = (batch_size, *out_shape)

        if self._dst_buffer is None or self._dst_buffer.shape != batch_dst_shape:
            self._dst_buffer = np.empty(batch_dst_shape, dtype=np.uint8)

        # Local refs to avoid attribute lookups in hot loop
        _cvtColor = cv2.cvtColor
        _code = self._cv2_code
        _dst = self._dst_buffer

        for i in range(batch_size):
            _cvtColor(src[i], _code, dst=_dst[i])

        return self._dst_buffer
