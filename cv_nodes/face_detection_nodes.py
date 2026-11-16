import cv2
import asyncio
from typing import Any
import numpy as np
import time
import logging


from nonoha_core.data_type import (
    DataType,
    PortValueType,
    PropertyDataType,
    name_to_port_value_type,
)
from nonoha_core.port import (
    PortValue,
    StaticTypeInputPort,
    StaticTypeOutputPort,
)
from nonoha_core.node import BaseNode, replace_node_input_port


logger = logging.getLogger(__name__)


class FaceDetectionYNNode(BaseNode):
    def __init__(self, name: str):
        super().__init__(name, "FaceDetectionYN")

        in_image_type = PortValueType(DataType.Uint8, (480, 640, 3))  # [H, W, 3]
        input_image = StaticTypeInputPort("image", [in_image_type])
        input_image.subscribe(id(self), self._on_input_image)

        # 0-1: x, y of bbox top left corner
        # 2-3: width, height of bbox
        # 4-5: x, y of right eye (blue point in the example image)
        # 6-7: x, y of left eye (red point in the example image)
        # 8-9: x, y of nose tip (green point in the example image)
        # 10-11: x, y of right corner of mouth (pink point in the example image)
        # 12-13: x, y of left corner of mouth (yellow point in the example image)
        # 14: face score
        output_faces = StaticTypeOutputPort(
            "faces", [PortValueType(DataType.Float32, shape=(-1, 15))]
        )

        # model_path = "face_detection_yunet_2023mar.onnx"
        model_path = ""

        self._input_image = input_image
        self._in_image_type = in_image_type
        self._output_faces = output_faces
        self._model_path = model_path
        self._face_detector: cv2.FaceDetectorYN | None = None

        self.inputs[input_image.name] = input_image
        self.outputs[output_faces.name] = output_faces

        self.properties["model_path"] = model_path
        self.properties["in_image_type"] = in_image_type.name

    def set_properties(self, properties: dict[str, PropertyDataType]):
        model_path = self._model_path
        if "model_path" in properties:
            model_path = str(properties["model_path"])

        in_image_type = self._in_image_type
        if "in_image_type" in properties:
            in_image_type = name_to_port_value_type(str(properties["in_image_type"]))
            if in_image_type.ndim != 3:
                raise ValueError("in_image_type must be 3D")
            if in_image_type.data_type != DataType.Uint8:
                raise ValueError("in_image_type must be uint8")

        new_input_image = self._input_image
        if in_image_type != self._in_image_type:
            new_input_image = StaticTypeInputPort("image", [in_image_type])
            replace_node_input_port(
                self, self._input_image, new_input_image, self._on_input_image
            )
            self._input_image = new_input_image

        persist = self.parent_graph.persist if self.parent_graph else None
        new_face_detector = self._face_detector
        if model_path != self._model_path:
            if persist is None:
                new_face_detector = None
            else:
                new_face_detector = cv2.FaceDetectorYN.create(
                    str(persist.data_dir_path / model_path),
                    "",
                    (in_image_type.shape[1], in_image_type.shape[0]),
                )

        self._in_image_type = in_image_type
        self._model_path = model_path
        self._face_detector = new_face_detector

        self.properties["model_path"] = model_path
        self.properties["in_image_type"] = in_image_type.name

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    def _on_input_image(self, values: list[PortValue]):
        face_detector = self._face_detector
        if face_detector is None:
            print("Face detector is not initialized")
            return

        image_array = values[0]

        start_time = time.perf_counter()
        for image in image_array:
            try:
                _, faces = face_detector.detect(image)
                if faces is None:
                    self._output_faces.dispatch([np.empty((1, 0, 15), dtype=np.float32)])
                else:
                    self._output_faces.dispatch([faces[np.newaxis, :]])
            except Exception as e:
                logger.warning(f"Face detection failed: {e}")

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        logger.info(f"{self.name}: {execution_time:.3f} ms")
