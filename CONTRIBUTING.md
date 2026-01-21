# Contributing to motcpp

First off, thank you for considering contributing to motcpp! It's people like you that make motcpp such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what you expected**
- **Include your environment details** (OS, compiler, OpenCV version, etc.)

### Suggesting Enhancements

If you have a suggestion for the project:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain the desired behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows our style guidelines
6. Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/Geekgineer/motcpp.git
cd motcpp

# Create a branch
git checkout -b feature/my-awesome-feature

# Install dependencies (Ubuntu)
sudo apt-get install -y cmake libeigen3-dev libopencv-dev libyaml-cpp-dev

# Build with tests
cmake -B build -DMOTCPP_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure
```

## Style Guidelines

### C++ Code Style

We follow a modern C++ style based on the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with some modifications:

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters max
- **Braces**: K&R style for functions, same-line for control structures
- **Naming**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Variables: `snake_case`
  - Member variables: `snake_case_` (trailing underscore)
  - Constants: `kPascalCase` or `UPPER_SNAKE_CASE`
  - Namespaces: `lowercase`

```cpp
// Example
namespace motcpp::trackers {

class MyTracker : public BaseTracker {
public:
    explicit MyTracker(const TrackerConfig& config);
    
    Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                          const cv::Mat& img,
                          const Eigen::MatrixXf& embs) override;

private:
    float threshold_;
    std::vector<Track> active_tracks_;
};

} // namespace motcpp::trackers
```

### Code Formatting

We use `clang-format` for automatic formatting:

```bash
# Format all files
find include src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

### Documentation

- All public APIs must be documented with Doxygen-style comments
- Include `@brief`, `@param`, `@return`, and `@example` where appropriate

```cpp
/**
 * @brief Update tracker with new detections
 * 
 * @param dets Detection matrix (N, 6): [x1, y1, x2, y2, conf, cls]
 * @param img Current frame image
 * @param embs Optional embeddings for ReID (N, D)
 * @return Tracked objects (M, 8): [x1, y1, x2, y2, id, conf, cls, det_ind]
 * 
 * @example
 * @code
 * Eigen::MatrixXf detections = detector.detect(frame);
 * Eigen::MatrixXf tracks = tracker.update(detections, frame);
 * @endcode
 */
virtual Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                               const cv::Mat& img,
                               const Eigen::MatrixXf& embs = Eigen::MatrixXf()) = 0;
```

## Testing

### Writing Tests

- Use Google Test framework
- Test files should be named `test_*.cpp`
- Each test file should focus on one component
- Use descriptive test names

```cpp
TEST(SortTest, TracksConsistentlyAcrossFrames) {
    Sort tracker(0.3f, 30, 50, 1);
    
    // Create detection
    Eigen::MatrixXf det(1, 6);
    det << 100, 100, 200, 200, 0.9, 0;
    
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // Track across multiple frames
    auto tracks1 = tracker.update(det, img);
    auto tracks2 = tracker.update(det, img);
    
    // ID should be consistent
    EXPECT_EQ(tracks1(0, 4), tracks2(0, 4));
}
```

### Running Tests

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run specific test suite
./motcpp_tests --gtest_filter=SortTest.*

# Run with verbose output
./motcpp_tests --gtest_filter=* --gtest_print_time=1
```

## Adding a New Tracker

1. Create header in `include/motcpp/trackers/mytracker.hpp`
2. Create implementation in `src/trackers/mytracker.cpp`
3. Add to `CMakeLists.txt`
4. Add tests in `tests/test_mytracker.cpp`
5. Update `include/Geekgineer/motcpp.hpp` to include new tracker
6. Document in README.md

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

```
feat: add UCMCTrack tracker implementation

- Implement ground-plane Kalman filter
- Add camera coordinate mapping
- Add unit tests for UCMCTrack

Fixes #123
```

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 License.
