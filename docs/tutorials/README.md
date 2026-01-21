# Tutorials

Step-by-step tutorials for common tracking tasks.

## Available Tutorials

| Tutorial | Description | Difficulty |
|----------|-------------|------------|
| [Basic Tracking](#basic-tracking) | Track objects in a video | Beginner |
| [ReID Integration](#reid-integration) | Add appearance features | Intermediate |
| [Custom Detector](#custom-detector-integration) | Integrate your detector | Intermediate |
| [MOT Evaluation](#mot-benchmark-evaluation) | Benchmark on MOT17 | Advanced |
| [Real-time Pipeline](#real-time-pipeline) | Optimize for speed | Advanced |

---

## Basic Tracking

Learn to track objects in a video using ByteTrack.

### Prerequisites

- Compiled motcpp library
- OpenCV 4.x
- A video file with objects to track
- Pre-computed detections (or a detector)

### Step 1: Setup

```cpp
#include <motcpp/trackers/bytetrack.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>\n";
        return 1;
    }
    
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << argv[1] << "\n";
        return 1;
    }
```

### Step 2: Create Tracker

```cpp
    // Create ByteTrack with tuned parameters
    motcpp::trackers::ByteTrack tracker(
        0.25f,  // det_thresh: filter low confidence
        30,     // max_age: keep tracks for 30 frames
        50,     // max_obs: observation history
        3,      // min_hits: confirm after 3 detections
        0.3f,   // iou_threshold: matching threshold
        false,  // per_class: track all classes together
        80,     // nr_classes
        "iou"   // association function
    );
```

### Step 3: Process Video

```cpp
    cv::Mat frame;
    int frame_id = 0;
    
    while (cap.read(frame)) {
        // Simulate detections (replace with your detector)
        // Format: [x1, y1, x2, y2, confidence, class_id]
        Eigen::MatrixXf dets = get_detections(frame, frame_id);
        
        // Update tracker
        Eigen::MatrixXf tracks = tracker.update(dets, frame);
        
        // Process results
        for (int i = 0; i < tracks.rows(); ++i) {
            int x1 = static_cast<int>(tracks(i, 0));
            int y1 = static_cast<int>(tracks(i, 1));
            int x2 = static_cast<int>(tracks(i, 2));
            int y2 = static_cast<int>(tracks(i, 3));
            int track_id = static_cast<int>(tracks(i, 4));
            float conf = tracks(i, 5);
            
            // Draw bounding box
            cv::Scalar color = motcpp::BaseTracker::id_to_color(track_id);
            cv::rectangle(frame, {x1, y1}, {x2, y2}, color, 2);
            
            // Draw label
            std::string label = "ID:" + std::to_string(track_id);
            cv::putText(frame, label, {x1, y1 - 5},
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
        
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;  // ESC to exit
        
        ++frame_id;
    }
    
    return 0;
}
```

### Complete Example

See [examples/basic_tracking.cpp](../../examples/basic_tracking.cpp).

---

## ReID Integration

Add appearance features for better re-identification.

### Download ReID Model

```bash
# Download OSNet model
wget https://github.com/Geekgineer/motcpp/releases/download/v1.0/osnet_x1_0.onnx
```

### Create ReID Tracker

```cpp
#include <motcpp/trackers/strongsort.hpp>

// StrongSORT with ReID
motcpp::trackers::StrongSORT tracker(
    "osnet_x1_0.onnx",  // ReID model path
    false,              // use_half (FP16)
    false,              // use_gpu
    0.3f,               // det_thresh
    30,                 // max_age
    50,                 // max_obs
    1,                  // min_hits
    0.3f,               // iou_threshold
    false,              // per_class
    80,                 // nr_classes
    "iou",              // asso_func
    false               // is_obb
);
```

### Update with Embeddings

For custom ReID, provide embeddings directly:

```cpp
// Your custom ReID model
Eigen::MatrixXf embeddings = reid_model.extract(frame, dets);

// Update with embeddings
Eigen::MatrixXf tracks = tracker.update(dets, frame, embeddings);
```

---

## Custom Detector Integration

Integrate YOLO or other detectors.

### YOLO Format Conversion

```cpp
// YOLO outputs: [cx, cy, w, h, conf, class_id]
// motcpp expects: [x1, y1, x2, y2, conf, class_id]

Eigen::MatrixXf convert_yolo_detections(
    const Eigen::MatrixXf& yolo_dets) {
    
    Eigen::MatrixXf dets(yolo_dets.rows(), 6);
    
    for (int i = 0; i < yolo_dets.rows(); ++i) {
        float cx = yolo_dets(i, 0);
        float cy = yolo_dets(i, 1);
        float w = yolo_dets(i, 2);
        float h = yolo_dets(i, 3);
        
        dets(i, 0) = cx - w / 2;  // x1
        dets(i, 1) = cy - h / 2;  // y1
        dets(i, 2) = cx + w / 2;  // x2
        dets(i, 3) = cy + h / 2;  // y2
        dets(i, 4) = yolo_dets(i, 4);  // conf
        dets(i, 5) = yolo_dets(i, 5);  // class
    }
    
    return dets;
}
```

### ONNX Detector Example

```cpp
#include <onnxruntime_cxx_api.h>

class YOLODetector {
public:
    YOLODetector(const std::string& model_path) {
        // Initialize ONNX Runtime
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLODetector");
        session_ = Ort::Session(env_, model_path.c_str(), Ort::SessionOptions{});
    }
    
    Eigen::MatrixXf detect(const cv::Mat& frame) {
        // Preprocess
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, {640, 640});
        
        // Run inference
        // ...
        
        // Postprocess and return detections
        return detections;
    }

private:
    Ort::Env env_;
    Ort::Session session_;
};
```

---

## MOT Benchmark Evaluation

Evaluate tracker on MOT17 dataset.

### Download Dataset

```bash
# Download MOT17
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
```

### Run Evaluation

```bash
# Build evaluation tool
cmake -B build -DMOTCPP_BUILD_TOOLS=ON
cmake --build build

# Run tracker on MOT17
./build/tools/motcpp_eval \
    --tracker bytetrack \
    --dataset MOT17/train \
    --dets yolox \
    --output results/

# Evaluate with TrackEval
python -m trackeval.eval_mot \
    --GT_FOLDER MOT17/train \
    --TRACKERS_FOLDER results/ \
    --BENCHMARK MOT17
```

### Results Format

Output files follow MOT format:
```
<frame>,<id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1
```

---

## Real-time Pipeline

Optimize for maximum throughput.

### Parallel Processing

```cpp
#include <thread>
#include <queue>
#include <mutex>

class AsyncTracker {
public:
    AsyncTracker(const std::string& tracker_type) {
        tracker_ = motcpp::create_tracker(tracker_type);
        
        // Start processing thread
        running_ = true;
        thread_ = std::thread(&AsyncTracker::process_loop, this);
    }
    
    void submit(const cv::Mat& frame, const Eigen::MatrixXf& dets) {
        std::lock_guard<std::mutex> lock(mutex_);
        input_queue_.push({frame.clone(), dets});
    }
    
    bool get_result(Eigen::MatrixXf& tracks) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (output_queue_.empty()) return false;
        tracks = output_queue_.front();
        output_queue_.pop();
        return true;
    }

private:
    void process_loop() {
        while (running_) {
            cv::Mat frame;
            Eigen::MatrixXf dets;
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (input_queue_.empty()) continue;
                auto& item = input_queue_.front();
                frame = item.first;
                dets = item.second;
                input_queue_.pop();
            }
            
            auto tracks = tracker_->update(dets, frame);
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                output_queue_.push(tracks);
            }
        }
    }
    
    std::unique_ptr<motcpp::BaseTracker> tracker_;
    std::queue<std::pair<cv::Mat, Eigen::MatrixXf>> input_queue_;
    std::queue<Eigen::MatrixXf> output_queue_;
    std::mutex mutex_;
    std::thread thread_;
    std::atomic<bool> running_;
};
```

### Memory Optimization

```cpp
// Pre-allocate buffers
Eigen::MatrixXf dets_buffer(100, 6);  // Max 100 detections
Eigen::MatrixXf tracks_buffer(50, 8); // Max 50 tracks

// Reuse buffers
dets_buffer.conservativeResize(num_dets, 6);
// Fill dets_buffer...

auto tracks = tracker.update(dets_buffer, frame);
```

### Profiling

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

auto tracks = tracker.update(dets, frame);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

std::cout << "Tracking time: " << duration.count() << " µs\n";
std::cout << "FPS: " << 1e6 / duration.count() << "\n";
```
