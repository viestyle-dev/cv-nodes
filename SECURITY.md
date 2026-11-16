# Security Policy

## Reporting a Vulnerability

Please use GitHub's **Private Vulnerability Reporting**
(Security tab → "Report a vulnerability") on this repository.
Do not open public issues for security reports.

## Scope

cv-nodes reads video files, cameras and model files, and decodes images.
`file_path` and `model_path` are resolved against the graph's data
directory, but an absolute path is honoured, so a graph can read any file
the engine process can access. Video files are decoded in-process by
OpenCV's FFmpeg or by Windows Media Foundation, and images sent into the
graph by OpenCV's image codecs (`Imdecode`): keep `opencv-python` and the
OS up to date. `CameraCapture` opens the camera selected by
`camera_index`. `FaceDetectionYN` loads the ONNX model at `model_path`
into OpenCV's DNN module; load only models you trust. The plugin does not
execute code from files or graph definitions.

The host runtime it plugs into (the nonoha engine) may include script
nodes with their own threat model — see the security policy of the host
project for loading untrusted graphs.

## Supported versions

Security fixes target the latest minor release.
