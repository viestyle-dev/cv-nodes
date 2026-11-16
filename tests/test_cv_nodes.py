"""Tests for cv-nodes plugin."""

import numpy as np
import cv2
import pytest

from cv_nodes.imgcodecs_nodes import ImdecodeNode
from cv_nodes.imgproc_color_conversions_nodes import CvtColorNode


# === Helpers ===

def make_test_image(height=100, width=160):
    """Create a BGR test image with a red rectangle."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[20:80, 30:130] = [0, 0, 255]  # Red in BGR
    return img


def encode_jpeg(img, quality=90):
    """Encode image to JPEG bytes as 1D uint8 array."""
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.flatten()


def encode_png(img):
    """Encode image to PNG bytes as 1D uint8 array."""
    _, buf = cv2.imencode('.png', img)
    return buf.flatten()


# === ImdecodeNode Tests ===

def test_imdecode_node_init():
    node = ImdecodeNode("test")
    assert node.type == "Imdecode"
    assert "buf" in node.inputs
    assert "image" in node.outputs

    buf_port = node.inputs["buf"]
    assert buf_port.value_type_list[0].data_type.name == "Uint8"
    assert buf_port.value_type_list[0].shape == (-1,)

    image_port = node.outputs["image"]
    assert image_port.value_type_list[0].data_type.name == "Uint8"
    assert image_port.value_type_list[0].shape == (-1, -1, 3)

    node.finalize()


def test_imdecode_decode_jpeg():
    node = ImdecodeNode("test")

    src_img = make_test_image(100, 160)
    jpeg_bytes = encode_jpeg(src_img)
    node.inputs["buf"].dispatch([jpeg_bytes[np.newaxis]])  # (1, N) batch dim

    image_port = node.outputs["image"]
    assert image_port.current_values is not None
    decoded = image_port.current_values[0]
    assert decoded.shape == (1, 100, 160, 3)  # batch dim included
    assert decoded.dtype == np.uint8

    node.finalize()


def test_imdecode_decode_png():
    node = ImdecodeNode("test")

    src_img = make_test_image(80, 120)
    png_bytes = encode_png(src_img)
    node.inputs["buf"].dispatch([png_bytes[np.newaxis]])  # (1, N) batch dim

    image_port = node.outputs["image"]
    assert image_port.current_values is not None
    decoded = image_port.current_values[0]
    # IMREAD_COLOR always produces 3ch BGR
    assert decoded.shape == (1, 80, 120, 3)

    node.finalize()


def test_imdecode_invalid_data():
    node = ImdecodeNode("test")

    invalid_bytes = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)  # (1, 5)
    node.inputs["buf"].dispatch([invalid_bytes])

    assert node.outputs["image"].current_values is None  # No output dispatched

    node.finalize()


def test_imdecode_empty_data():
    node = ImdecodeNode("test")

    empty_bytes = np.array([[]], dtype=np.uint8).reshape(1, 0)  # (1, 0)
    node.inputs["buf"].dispatch([empty_bytes])

    assert node.outputs["image"].current_values is None

    node.finalize()


def test_imdecode_resolution_change():
    """Different resolutions should decode correctly without port replacement."""
    node = ImdecodeNode("test")

    # First frame: 100x160
    img1 = make_test_image(100, 160)
    node.inputs["buf"].dispatch([encode_jpeg(img1)[np.newaxis]])
    assert node.outputs["image"].current_values[0].shape == (1, 100, 160, 3)

    # Second frame: 50x80 (different resolution)
    img2 = make_test_image(50, 80)
    node.inputs["buf"].dispatch([encode_jpeg(img2)[np.newaxis]])
    assert node.outputs["image"].current_values[0].shape == (1, 50, 80, 3)

    # Port shape stays as wildcard
    assert node.outputs["image"].value_type_list[0].shape == (-1, -1, 3)

    node.finalize()


# === CvtColorNode Tests ===

def test_cvtcolor_node_init():
    node = CvtColorNode("test")
    assert node.type == "CvtColor"
    assert "src" in node.inputs
    assert "dst" in node.outputs

    assert node.properties["code"] == "BGR2RGBA"
    assert node.properties["width"] == 1280
    assert node.properties["height"] == 720

    src_port = node.inputs["src"]
    assert src_port.value_type_list[0].shape == (720, 1280, 3)

    dst_port = node.outputs["dst"]
    assert dst_port.value_type_list[0].shape == (720, 1280, 4)

    node.finalize()


def test_cvtcolor_set_properties():
    node = CvtColorNode("test")

    node.set_properties({"code": "BGR2GRAY", "width": 640, "height": 480})

    assert node.properties["code"] == "BGR2GRAY"
    assert node.properties["width"] == 640
    assert node.properties["height"] == 480

    src_port = node.inputs["src"]
    assert src_port.value_type_list[0].shape == (480, 640, 3)

    dst_port = node.outputs["dst"]
    assert dst_port.value_type_list[0].shape == (480, 640)  # grayscale: (H, W)

    node.finalize()


def test_cvtcolor_bgr2rgba():
    node = CvtColorNode("test")
    node.set_properties({"code": "BGR2RGBA", "width": 4, "height": 2})

    results = []
    node.outputs["dst"].subscribe(999, lambda values: results.append(values[0]))

    # Create a 2x4 BGR image: pure blue pixel [255, 0, 0]
    src = np.full((1, 2, 4, 3), [255, 0, 0], dtype=np.uint8)  # batch=1
    node.inputs["src"].dispatch([src])

    assert len(results) == 1
    dst = results[0]
    assert dst.shape == (1, 2, 4, 4)  # RGBA
    # BGR [255, 0, 0] → RGBA [0, 0, 255, 255]
    assert dst[0, 0, 0, 0] == 0    # R
    assert dst[0, 0, 0, 1] == 0    # G
    assert dst[0, 0, 0, 2] == 255  # B
    assert dst[0, 0, 0, 3] == 255  # A

    node.finalize()


def test_cvtcolor_dynamic_shape():
    """width=-1, height=-1 should accept any resolution."""
    node = CvtColorNode("test")
    node.set_properties({"code": "BGR2RGBA", "width": -1, "height": -1})

    # Port shapes should use -1 (wildcard)
    src_port = node.inputs["src"]
    assert src_port.value_type_list[0].shape == (-1, -1, 3)

    # Dispatch a 50x80 image
    src = np.zeros((1, 50, 80, 3), dtype=np.uint8)
    node.inputs["src"].dispatch([src])

    # Output data has actual shape, but port type stays wildcard
    dst_port = node.outputs["dst"]
    assert dst_port.current_values is not None
    assert dst_port.current_values[0].shape == (1, 50, 80, 4)
    assert dst_port.value_type_list[0].shape == (-1, -1, 4)

    node.finalize()


def test_cvtcolor_reject_zero_size():
    node = CvtColorNode("test")

    with pytest.raises(ValueError, match="width"):
        node.set_properties({"width": 0})

    with pytest.raises(ValueError, match="height"):
        node.set_properties({"height": 0})

    node.finalize()


def test_cvtcolor_unknown_code():
    node = CvtColorNode("test")

    with pytest.raises(ValueError, match="Unknown conversion code"):
        node.set_properties({"code": "INVALID_CODE"})

    node.finalize()
