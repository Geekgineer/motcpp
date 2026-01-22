# motcpp

<div style="text-align: center; margin: 2em 0;">
  <p style="font-size: 1.2em; color: #666; margin-top: 0.5em;">
    Modern C++ Multi-Object Tracking Library
  </p>
</div>

Welcome to the official documentation for **motcpp** - a high-performance, production-ready C++ library for multi-object tracking in video sequences.

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Getting Started](guides/getting-started.md) | Installation and first steps |
| [API Reference](api/README.md) | Complete API documentation |
| [Tutorials](tutorials/README.md) | Step-by-step tutorials |
| [Examples](examples/README.md) | Code examples |
| [Trackers](guides/trackers.md) | Available tracking algorithms |
| [Architecture](guides/architecture.md) | System design |
| [Benchmarking](guides/benchmarking.md) | MOT benchmark evaluation |
| [Contributing](CONTRIBUTING.md) | How to contribute |

## Features

- **9 State-of-the-Art Trackers** — SORT, ByteTrack, OC-SORT, DeepOC-SORT, StrongSORT, BoT-SORT, BoostTrack, HybridSORT, UCMCTrack
- **High Performance** — Optimized C++17 implementation
- **Modern API** — Clean, intuitive interface with Eigen and OpenCV
- **Flexible ReID** — ONNX Runtime backend for appearance models
- **Cross-Platform** — Linux, macOS, Windows
- **Well Tested** — Comprehensive unit tests

## Quick Example

```cpp
#include <Geekgineer/motcpp.hpp>

int main() {
    // Create tracker
    motcpp::TrackerConfig config;
    config.det_thresh = 0.3f;
    config.max_age = 30;
    
    auto tracker = motcpp::create_tracker("bytetrack", config);
    
    // Process video
    cv::VideoCapture cap("video.mp4");
    cv::Mat frame;
    
    while (cap.read(frame)) {
        // Your detector: [x1, y1, x2, y2, conf, class]
        Eigen::MatrixXf dets = detector.detect(frame);
        
        // Update tracker
        Eigen::MatrixXf tracks = tracker->update(dets, frame);
        
        // tracks: [x1, y1, x2, y2, id, conf, class, det_idx]
        for (int i = 0; i < tracks.rows(); ++i) {
            int id = static_cast<int>(tracks(i, 4));
            // Use track...
        }
    }
    
    return 0;
}
```

## License

motcpp is licensed under the [GNU Affero General Public License v3.0](../LICENSE).

## Citation

```bibtex
@software{motcpp2026,
  author = {motcpp contributors},
  title = {motcpp: Modern C++ Multi-Object Tracking Library},
  year = {2026},
  url = {https://github.com/Geekgineer/motcpp},
  license = {AGPL-3.0}
}
```
