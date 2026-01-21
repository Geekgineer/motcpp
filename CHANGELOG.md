# Changelog

All notable changes to motcpp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release preparation

## [1.0.0] - 2026-01-21

### Added
- **9 Multi-Object Tracking Algorithms**
  - SORT (Simple Online and Realtime Tracking)
  - ByteTrack (ECCV 2022)
  - OC-SORT (CVPR 2023)
  - DeepOC-SORT
  - StrongSORT (TMM 2023)
  - BoT-SORT
  - BoostTrack (MVA 2024)
  - HybridSORT
  - UCMCTrack (AAAI 2024)

- **Core Components**
  - Kalman filter implementations (XYSR, XYAH, XYWH state spaces)
  - Linear assignment solver (Jonker-Volgenant algorithm)
  - IoU computation variants (IoU, GIoU, DIoU, CIoU)
  - Camera Motion Compensation (ECC, SOF)

- **ReID Support**
  - ONNX Runtime backend for appearance models
  - Support for OSNet, ResNet, CLIP embeddings

- **Tools**
  - `motcpp_eval` - MOT benchmark evaluation tool
  - `auto_benchmark.sh` - Automated benchmarking script

- **Documentation**
  - Comprehensive API reference
  - Getting started guide
  - Tracker selection guide
  - Architecture documentation
  - Benchmarking guide

- **Testing**
  - 46 unit tests with Google Test
  - CI/CD with GitHub Actions
  - Multi-platform support (Linux, macOS, Windows)

### Technical Details
- C++17 standard
- CMake 3.16+ build system
- Eigen3 for linear algebra
- OpenCV 4.x for image processing
- ONNX Runtime for neural network inference

[Unreleased]: https://github.com/Geekgineer/motcpp/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Geekgineer/motcpp/releases/tag/v1.0.0
