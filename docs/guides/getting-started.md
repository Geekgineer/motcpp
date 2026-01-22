# Getting Started

This guide will help you install motcpp and run your first multi-object tracker.

## Prerequisites

Before installing motcpp, ensure you have:

- **C++17 compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **CMake**: 3.16 or higher
- **OpenCV**: 4.x
- **Eigen3**: 3.3+

### Installing Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libopencv-dev \
    libyaml-cpp-dev
```

#### macOS

```bash
brew install cmake eigen opencv yaml-cpp
```

#### Windows (vcpkg)

```powershell
vcpkg install eigen3:x64-windows opencv4:x64-windows yaml-cpp:x64-windows
```

## Installation

### Option 1: Build from Source

```bash
# Clone the repository
git clone https://github.com/Geekgineer/motcpp.git
cd motcpp

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Install (optional)
sudo cmake --install build
```

### Option 2: CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    motcpp
    GIT_REPOSITORY https://github.com/Geekgineer/motcpp.git
    GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(motcpp)

target_link_libraries(your_target PRIVATE motcpp::motcpp)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `MOTCPP_BUILD_TESTS` | ON | Build unit tests |
| `MOTCPP_BUILD_EXAMPLES` | ON | Build example applications |
| `MOTCPP_BUILD_TOOLS` | ON | Build CLI tools |
| `MOTCPP_ENABLE_ONNX` | ON | Enable ONNX Runtime for ReID |
| `MOTCPP_COVERAGE` | OFF | Enable code coverage |

Example with custom options:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DMOTCPP_BUILD_TESTS=OFF \
    -DMOTCPP_ENABLE_ONNX=ON
```

## First Tracker {#first-tracker}

Here's a minimal example to get you started:

```cpp
#include <motcpp/trackers/bytetrack.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Create ByteTrack tracker with default parameters
    motcpp::trackers::ByteTrack tracker;
    
    // Create a dummy frame
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // Simulated detections: [x1, y1, x2, y2, confidence, class_id]
    Eigen::MatrixXf detections(2, 6);
    detections << 100, 100, 200, 200, 0.9, 0,
                  300, 300, 400, 400, 0.8, 0;
    
    // Update tracker
    Eigen::MatrixXf tracks = tracker.update(detections, frame);
    
    // Print results
    std::cout << "Tracked " << tracks.rows() << " objects\n";
    for (int i = 0; i < tracks.rows(); ++i) {
        std::cout << "  Track ID: " << static_cast<int>(tracks(i, 4))
                  << " at [" << tracks(i, 0) << ", " << tracks(i, 1)
                  << ", " << tracks(i, 2) << ", " << tracks(i, 3) << "]\n";
    }
    
    return 0;
}
```

Compile with:

```bash
g++ -std=c++17 first_tracker.cpp -o first_tracker \
    -I/path/to/motcpp/include \
    -L/path/to/motcpp/build -lmotcpp \
    $(pkg-config --cflags --libs opencv4 eigen3)
```

## Next Steps

- [Choose a Tracker](trackers.md) — Compare tracking algorithms
- [API Reference](../api/README.md) — Detailed API documentation
- [Tutorials](../tutorials/README.md) — In-depth tutorials
- [Examples](../examples/README.md) — More code examples
