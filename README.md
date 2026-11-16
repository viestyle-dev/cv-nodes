# cv-nodes

A nonoha-engine plugin for computer vision with OpenCV: video file and camera capture, resizing, encoding, color conversion, optical flow, motion grids, moving-object counting and face detection.

## Overview

This plugin provides nodes for image and video processing within the nonoha-engine visual programming environment. Frames flow between nodes as batched NumPy arrays (the capture nodes send BGR `uint8[N, H, W, 3]`), so they connect directly to other plugins' nodes. File naming follows OpenCV's `@defgroup` convention (e.g., `videoio_nodes.py` corresponds to OpenCV's `videoio` module).

## Features

### Nodes Provided

**Video input**

- **VideoFileCapture**: Reads the frames of a video file in order, on a worker thread
  - Media Foundation on Windows (hardware decoding), FFmpeg elsewhere
  - Fits other sizes to the output (cover, stretch, exact); loop, seek, EOF, metadata

- **CameraCapture**: Captures frames from a camera on a worker thread
  - Delivers only the newest frame when the engine is busy
  - Reports the capture status and the measured frame rate

**Image**

- **Resize**: Resizes each image to a fixed width and height (area, linear, nearest or cubic)

- **CvtColor**: Color space conversion supporting all `cv2.cvtColor` operations
  - 100+ conversion codes (BGR/RGB/Gray/HSV/Lab/Luv/YUV/Bayer/etc.)
  - Batchable conversions use reshape optimization (single `cvtColor` call for N frames)
  - Non-batchable conversions (Bayer demosaic, planar YUV 4:2:0) use optimized per-frame loop

- **Imencode**: Encodes images to JPEG, PNG or WebP bytes, to send frames to apps or services
- **Imdecode**: Decodes JPEG, PNG, WebP, ... bytes to BGR images

**Motion**

- **OpticalFlowFarneback**: Dense optical flow between consecutive gray frames, on a worker thread
- **MotionGrid**: Summarizes a flow field into a grid: energy and mean vector per cell, onsets, and the overall direction, speed and coherence
- **FrameDiffMotion**: Motion energy per grid cell from frame differences (lighter than optical flow, no direction)
- **BlobCounter**: Counts moving objects by background subtraction (MOG2) and connected components, on a worker thread

**Detection**

- **FaceDetectionYN**: Face detection using YuNet DNN model
  - Outputs bounding boxes, facial landmarks (eyes, nose, mouth), and confidence scores
  - Configurable input image dimensions

The sections below detail VideoFileCapture, CvtColor and FaceDetectionYN. Every node's ports and properties are also described in the engine's node catalog and in the editor.

### Registration

- `register_nodes`: every node. This is the entry point plugin hosts call.
- `register_portable_nodes`: the processing nodes only (no camera, file or model I/O), for runtimes where the host app owns cameras and files and feeds frames in through graph inputs.

## Installation

### From the Repository

```sh
pip install git+https://github.com/viestyle-dev/cv-nodes.git
```

Requires Python 3.11+. `nonoha-core` is provided by the host runtime (the
nonoha engine), so it is not a runtime dependency of this package.

### As a Dependency in nonoha-engine Workspace

1. Add `cv-nodes` to your workspace's `pyproject.toml`:

   ```toml
   [project]
   dependencies = [
       "cv-nodes",
       # other dependencies...
   ]

   [tool.uv.sources]
   cv-nodes = { path = "./plugins/cv-nodes", editable = false }
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

### As a Built-in Plugin

To include cv-nodes as a built-in plugin in your nonoha-engine instance:

1. Import the plugin in your `src/main.py`:

   ```python
   import cv_nodes
   ```

2. Add to RuntimeConfig:

   ```python
   from nonoha_engine.runtime_config import RuntimeConfig

   config = RuntimeConfig(
       builtin_plugins=[
           cv_nodes,
           # other built-in plugins...
       ]
   )
   ```

## Nodes

### VideoFileCapture

Reads the frames of a video file in order on a worker thread, when
`readout_count` arrives. It never skips frames: how fast to read, and which
frames to analyze, is up to the graph (like AudioFileStream, which reads when
the player asks).

- **Real-time playback**: drive `readout_count` from a clock. At most one read
  waits behind the one in progress; requests beyond that are dropped (the
  playback slows down, the frames stay in order) and logged at most every 10 s
  with the time a read takes.
- **Every frame, as fast as possible** (offline analysis): drive the next
  `readout_count` from `position`, so a request is sent only after the previous
  frames were delivered.
- **Readers** (`backend`): `auto` uses Media Foundation on Windows (the OS
  decoders: hardware decoding and color conversion on the GPU, about a third
  of the CPU of FFmpeg for 4K H.264) and FFmpeg elsewhere. A file the preferred
  reader cannot open or decode falls back to FFmpeg. The log names the reader
  used.

#### VideoFileCapture Properties

| Property | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| file_path | str | "" | Video file; relative paths resolve against the graph's data folder. Empty closes the file. Also an input (`@file_path`): a new path loads that file and is saved with the graph |
| backend | str | "auto" | `auto`, `ffmpeg` or `media_foundation` |
| width | int | 1280 | Output frame width |
| height | int | 720 | Output frame height |
| loop | bool | false | Restart from the first frame at the end of the file |
| fit | str | "cover" | Frames of another size: `cover` (crop the middle to the output aspect, then scale), `stretch`, or `exact` (send nothing) |

#### VideoFileCapture Input Ports

| Port | Type | Description |
| ---- | ---- | ----------- |
| readout_count | Int32 | Number of frames to read now (one batch per value) |
| seek | Int32 | Frame index to continue from |
| @file_path, @backend, @fit | String | Set those properties |

#### VideoFileCapture Output Ports

| Port | Type | Description |
| ---- | ---- | ----------- |
| loading_status | Uint8 | 0 = not loaded, 1 = loading, 2 = loaded |
| frame | Uint8 [H, W, 3] | BGR frames (batched) |
| video_info | Int32 (width), Int32 (height), Float32 (fps), Int32 (frame_count) | The loaded file's metadata |
| eof | Bool | True when the end of the file is reached (never, with loop on) |
| position | Int32 | Index of the last frame read |

Before 1.0, `file_path` was the `load` input and `loading_status` was
`status`; 1.0 matches the names to AudioFileStream.

### CvtColor

Color space conversion using `cv2.cvtColor`. Supports all OpenCV conversion codes.

#### CvtColor Properties

| Property | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| code | string | "BGR2RGBA" | Conversion code name (key in CONVERSIONS table) |
| width | int | 1280 | Frame width |
| height | int | 720 | Frame height |

#### CvtColor Input Ports

| Port | Type | Description |
| ---- | ---- | ----------- |
| src | Uint8 [H, W, C] | Source image (shape varies by conversion code) |

#### CvtColor Output Ports

| Port | Type | Description |
| ---- | ---- | ----------- |
| dst | Uint8 [H, W, C'] | Converted image (shape varies by conversion code) |

#### CvtColor Port Shape Rules

| Channels | Shape |
| -------- | ----- |
| Planar YUV 4:2:0 | (H\*3//2, W) |
| 1 (grayscale) | (H, W) |
| N (interleaved) | (H, W, N) |

#### Supported Conversion Categories

| Category | Examples | Batchable |
| -------- | -------- | --------- |
| BGR/RGB/Gray reorder & alpha | BGR2BGRA, BGR2RGB, BGR2GRAY, GRAY2BGR | Yes |
| BGR565/BGR555 | BGR2BGR565, BGR5652BGR | Yes |
| XYZ / YCrCb / HSV / HLS / Lab / Luv / YUV | BGR2XYZ, BGR2HSV, BGR2Lab | Yes |
| Packed YUV 4:2:2 | YUV2RGB_UYVY, RGB2YUV_YUY2 | Yes |
| mRGBA | RGBA2mRGBA, mRGBA2RGBA | Yes |
| Bayer demosaic (bilinear/VNG/EA) | BayerBG2BGR, BayerBG2BGR_VNG | No |
| Bayer to GRAY/BGRA | BayerBG2GRAY, BayerBG2BGRA | No |
| Planar YUV 4:2:0 | YUV2RGB_NV12, RGB2YUV_I420 | No |

### FaceDetectionYN

Face detection using OpenCV's YuNet DNN model (`cv2.FaceDetectorYN`).

#### Properties

| Property | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| model_path | string | "" | Path to YuNet ONNX model file (relative to data directory) |
| in_image_type | string | "uint8[480,640,3]" | Input image port value type |

#### Input Ports

| Port | Type | Description |
| ---- | ---- | ----------- |
| image | Uint8 [H, W, 3] | Input BGR image (dimensions from in_image_type) |

#### Output Ports

| Port | Type | Description |
| ---- | ---- | ----------- |
| faces | Float32 [-1, 15] | Detected faces array |

Each row in the faces array contains:

- 0-3: bounding box (x, y, width, height)
- 4-5: right eye (x, y)
- 6-7: left eye (x, y)
- 8-9: nose tip (x, y)
- 10-11: right mouth corner (x, y)
- 12-13: left mouth corner (x, y)
- 14: confidence score

**Note:** The model is not loaded by default. You must set `model_path` to initialize detection.

## Requirements

- Python >= 3.11
- nonoha-core (nonoha-engine core library)
- numpy >= 2.0
- opencv-python >= 5.0.0.93

### Platform-Specific Requirements

**Linux (including Raspberry Pi):** the OpenCV wheels need the OpenGL
runtime library. Install it if it is missing:

```bash
sudo apt install libgl1
```

**Windows:** VideoFileCapture uses Media Foundation, which ships with
Windows. On Windows Server it is an optional feature; without it, files
are read with FFmpeg.

## Development

```bash
uv sync        # also installs the dev dependency group (pytest, nonoha-core)
uv run pytest
```

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [cv::cvtColor Color Conversions](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- [cv::FaceDetectorYN](https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html)

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Pull requests require signing the [nonoha CLA](CLA.md) (a one-time
comment on your first PR, shared across the nonoha repositories).

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
