"""cv-nodes: OpenCV-based image and video nodes.

Two registration functions:

- ``register_nodes``: every node, including camera and video-file I/O.
  This is the entry point plugin hosts call.
- ``register_portable_nodes``: processing nodes only (no device or file
  I/O, no model files), for runtimes where the host app owns cameras and
  files and feeds frames in through graph inputs.

Node modules are imported inside the functions, so importing the package
itself stays cheap.
"""

import logging

from nonoha_core.node_registry import NodeRegistry

logger = logging.getLogger(__name__)


def register_portable_nodes(registry: NodeRegistry):
    """Register the processing nodes (no device or file I/O)."""
    from .imgcodecs_nodes import ImdecodeNode, ImencodeNode
    from .imgproc_color_conversions_nodes import CvtColorNode
    from .imgproc_geometric_nodes import ResizeNode
    from .motion_np_nodes import FrameDiffMotionNode, MotionGridNode
    from .video_nodes import BlobCounterNode, OpticalFlowFarnebackNode

    registry.register_node("CvtColor", CvtColorNode)
    registry.register_node("Imdecode", ImdecodeNode)
    registry.register_node("Imencode", ImencodeNode)
    registry.register_node("Resize", ResizeNode)
    registry.register_node("OpticalFlowFarneback", OpticalFlowFarnebackNode)
    registry.register_node("BlobCounter", BlobCounterNode)
    registry.register_node("MotionGrid", MotionGridNode)
    registry.register_node("FrameDiffMotion", FrameDiffMotionNode)


def register_nodes(registry: NodeRegistry):
    """Register all CV nodes, including camera and video-file I/O."""
    logger.info("Registering cv-nodes plugin")
    register_portable_nodes(registry)

    from .face_detection_nodes import FaceDetectionYNNode
    from .videoio_nodes import CameraCaptureNode, VideoFileCaptureNode

    registry.register_node("FaceDetectionYN", FaceDetectionYNNode)
    registry.register_node("VideoFileCapture", VideoFileCaptureNode)
    registry.register_node("CameraCapture", CameraCaptureNode)


async def finalize():
    """Optional cleanup function."""
    logger.info("Finalizing cv-nodes plugin")
