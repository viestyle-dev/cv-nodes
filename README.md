# cv-nodes

A nonoha-engine plugin for computer vision with OpenCV: video file and camera capture, resizing, encoding, color conversion, optical flow, motion grids, moving-object counting and face detection.

## Overview

This plugin provides nodes for image and video processing within the nonoha-engine visual programming environment. Frames flow between nodes as batched NumPy arrays (the capture nodes send BGR `uint8[N, H, W, 3]`), so they connect directly to other plugins' nodes. File naming follows OpenCV's `@defgroup` convention (e.g., `videoio_nodes.py` corresponds to OpenCV's `videoio` module).

## Features

### Nodes Provided

**Video input**

- **VideoFileCapture**: Reads the frames of a video file in order, on a worker thread
  - Decodes and scales to the output before the frame reaches Python: Media Foundation (Windows),
    AVFoundation (macOS), the `ffmpeg` command (Linux, Raspberry Pi); OpenCV's FFmpeg otherwise
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
- **Readers** (`backend`). The first three scale the frames to about the
  output size before they reach Python (cover: to the smallest size that
  covers it, then the middle is cut out), so a 4K clip costs about what its
  output does:

  | backend | Where | How |
  | ------- | ----- | --- |
  | `media_foundation` | Windows | Media Foundation decodes and scales on the GPU (without a Direct3D 11 device, on the CPU) |
  | `avfoundation` | macOS | AVFoundation (VideoToolbox) decodes and scales; needs PyObjC (installed with cv-nodes on macOS) |
  | `ffmpeg_command` | any OS with `ffmpeg` and `ffprobe` on PATH | ffmpeg decodes, on the hardware decoder when there is one, and scales and converts in one pass |
  | `opencv_ffmpeg` | everywhere | OpenCV's bundled FFmpeg: full frames are converted to BGR and fitted on the CPU (formerly `ffmpeg`; that name is still accepted) |

  `auto` tries, best first: Windows `media_foundation`, `ffmpeg_command`,
  `opencv_ffmpeg`; macOS `avfoundation`, `ffmpeg_command`, `opencv_ffmpeg`;
  Linux `ffmpeg_command`, `opencv_ffmpeg`. A reader that cannot open a file
  or decode its first frame gives way to the next; the log names the reader
  used and says when it decodes on the CPU. Another backend falls back to
  `opencv_ffmpeg`.

  Reading 4K H.264 for 640x360 (CPU per frame, a Windows laptop):
  `media_foundation` ~1-2 ms, `ffmpeg_command` ~12 ms with its hardware
  decoder (d3d11va) and ~32 ms without, `opencv_ffmpeg` ~64 ms.

  With `ffmpeg_command`, ffmpeg tries these decoders before decoding on the
  CPU: HEVC with `-hwaccel drm` (Raspberry Pi 4 and 5), H.264 with
  `h264_v4l2m2m` (Raspberry Pi Zero 2 W, 3 and 4; a Pi 5 has no H.264
  decoder), `-hwaccel vaapi` (Intel and AMD GPUs on Linux), `d3d11va`
  (Windows), `videotoolbox` (macOS). On a Raspberry Pi 5, HEVC clips are
  decoded in hardware up to 4K; on a Zero 2 W, H.264 up to 1080p30 (4K and
  HEVC cannot play there in real time).

#### VideoFileCapture Properties

| Property | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| file_path | str | "" | Video file; relative paths resolve against the graph's data folder. Empty closes the file. Also an input (`@file_path`): a new path loads that file and is saved with the graph |
| backend | str | "auto" | `auto`, `media_foundation`, `avfoundation`, `ffmpeg_command` or `opencv_ffmpeg` (see Readers) |
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
- macOS only: pyobjc-framework-AVFoundation, -CoreMedia and -Quartz >= 10.3

### Platform-Specific Requirements

**Linux (including Raspberry Pi):** the OpenCV wheels need the OpenGL
runtime library. Install it if it is missing:

```bash
sudo apt install libgl1
```

**Windows:** VideoFileCapture uses Media Foundation, which ships with
Windows. On Windows Server it is an optional feature; without it, files
are read with the `ffmpeg` command when installed, else OpenCV's FFmpeg.

**macOS:** VideoFileCapture uses AVFoundation through PyObjC
(`pyobjc-framework-AVFoundation`, `-CoreMedia`, `-Quartz`; installed with
cv-nodes on macOS).

**Linux and Raspberry Pi:** install ffmpeg for the scaling reader and the
hardware decoders (`sudo apt install ffmpeg`); without it, OpenCV's FFmpeg
reads full frames on the CPU.

**Licenses of the decoders:** OpenCV's bundled FFmpeg is LGPL 2.1 (see
opencv-python). The `ffmpeg` command is the user's own installation, run as a
separate program and not shipped with cv-nodes. Media Foundation and
AVFoundation are parts of the OS.

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
