# Examples

Code examples demonstrating motcpp features.

## Quick Examples

### Minimal Tracking

```cpp
#include <motcpp/trackers/bytetrack.hpp>

int main() {
    motcpp::trackers::ByteTrack tracker;
    
    cv::Mat frame = cv::imread("frame.jpg");
    
    // Detections: [x1, y1, x2, y2, conf, class]
    Eigen::MatrixXf dets(2, 6);
    dets << 100, 100, 200, 200, 0.9, 0,
            300, 300, 400, 400, 0.8, 0;
    
    // Track
    auto tracks = tracker.update(dets, frame);
    
    std::cout << "Tracked " << tracks.rows() << " objects\n";
    return 0;
}
```

### Video Processing

```cpp
#include <motcpp/trackers/ocsort.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::VideoCapture cap(argv[1]);
    motcpp::trackers::OCSort tracker;
    
    cv::Mat frame;
    while (cap.read(frame)) {
        Eigen::MatrixXf dets = detect(frame);  // Your detector
        auto tracks = tracker.update(dets, frame);
        
        visualize(frame, tracks);
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}
```

### Per-Class Tracking

```cpp
#include <motcpp/trackers/bytetrack.hpp>

int main() {
    // Enable per-class tracking
    motcpp::trackers::ByteTrack tracker(
        0.3f,   // det_thresh
        30,     // max_age
        50,     // max_obs
        3,      // min_hits
        0.3f,   // iou_threshold
        true,   // per_class = TRUE
        80      // nr_classes
    );
    
    // Detections with different classes
    Eigen::MatrixXf dets(3, 6);
    dets << 100, 100, 200, 200, 0.9, 0,   // Person
            300, 300, 400, 400, 0.8, 2,   // Car
            500, 100, 600, 200, 0.7, 0;   // Person
    
    auto tracks = tracker.update(dets, frame);
    // Each class tracked independently
    return 0;
}
```

### Oriented Bounding Boxes

```cpp
#include <motcpp/trackers/sort.hpp>

int main() {
    // Enable OBB tracking
    motcpp::trackers::Sort tracker(
        0.3f, 1, 50, 3, 0.3f,
        false, 80, "iou_obb",
        true  // is_obb = TRUE
    );
    
    // OBB format: [cx, cy, w, h, angle, conf, class]
    Eigen::MatrixXf obb_dets(1, 7);
    obb_dets << 150, 150, 100, 50, 0.785, 0.9, 0;  // 45 degrees
    
    auto tracks = tracker.update(obb_dets, frame);
    return 0;
}
```

---

## Complete Examples

### 1. Pedestrian Tracking

Full example for pedestrian tracking with visualization.

```cpp
// examples/pedestrian_tracking.cpp
#include <motcpp/trackers/bytetrack.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>

// Load detections from file (MOT format)
Eigen::MatrixXf load_detections(const std::string& path, int frame_id) {
    std::vector<std::vector<float>> dets;
    std::ifstream file(path);
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> det;
        
        while (std::getline(ss, token, ',')) {
            det.push_back(std::stof(token));
        }
        
        if (static_cast<int>(det[0]) == frame_id) {
            // Convert [frame,id,x,y,w,h,conf,...] to [x1,y1,x2,y2,conf,cls]
            dets.push_back({
                det[2],              // x1
                det[3],              // y1
                det[2] + det[4],     // x2
                det[3] + det[5],     // y2
                det[6],              // conf
                0                    // class (pedestrian)
            });
        }
    }
    
    Eigen::MatrixXf result(dets.size(), 6);
    for (size_t i = 0; i < dets.size(); ++i) {
        for (int j = 0; j < 6; ++j) {
            result(i, j) = dets[i][j];
        }
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video> <detections>\n";
        return 1;
    }
    
    cv::VideoCapture cap(argv[1]);
    std::string det_path = argv[2];
    
    // Optimized ByteTrack for pedestrians
    motcpp::trackers::ByteTrack tracker(
        0.25f,  // Lower threshold for partial occlusions
        50,     // Longer max_age for crowded scenes
        100,    // More observation history
        2,      // Quick confirmation
        0.3f,   // Standard IoU threshold
        false,  // Single class
        1,      // Only pedestrians
        "iou",  // Standard IoU
        false,  // Axis-aligned boxes
        0.1f,   // Include low-confidence detections
        0.4f,   // Track threshold
        0.85f,  // Match threshold
        50,     // Track buffer
        30.0f   // 30 FPS
    );
    
    cv::Mat frame;
    int frame_id = 1;
    
    // Track statistics
    std::set<int> unique_ids;
    int total_tracks = 0;
    
    while (cap.read(frame)) {
        auto dets = load_detections(det_path, frame_id);
        auto tracks = tracker.update(dets, frame);
        
        // Update statistics
        for (int i = 0; i < tracks.rows(); ++i) {
            unique_ids.insert(static_cast<int>(tracks(i, 4)));
        }
        total_tracks += tracks.rows();
        
        // Visualize
        for (int i = 0; i < tracks.rows(); ++i) {
            int x1 = tracks(i, 0), y1 = tracks(i, 1);
            int x2 = tracks(i, 2), y2 = tracks(i, 3);
            int id = tracks(i, 4);
            
            auto color = motcpp::BaseTracker::id_to_color(id);
            cv::rectangle(frame, {x1, y1}, {x2, y2}, color, 2);
            
            std::string label = "ID:" + std::to_string(id);
            int baseline;
            auto size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                        0.5, 1, &baseline);
            cv::rectangle(frame, {x1, y1 - size.height - 4}, 
                         {x1 + size.width, y1}, color, -1);
            cv::putText(frame, label, {x1, y1 - 2}, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
        }
        
        // Display stats
        std::string stats = "Frame: " + std::to_string(frame_id) + 
                           " | Tracks: " + std::to_string(tracks.rows()) +
                           " | Unique IDs: " + std::to_string(unique_ids.size());
        cv::putText(frame, stats, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, {0, 255, 0}, 2);
        
        cv::imshow("Pedestrian Tracking", frame);
        if (cv::waitKey(1) == 27) break;
        
        ++frame_id;
    }
    
    std::cout << "\n=== Tracking Summary ===\n";
    std::cout << "Total frames: " << frame_id - 1 << "\n";
    std::cout << "Unique IDs: " << unique_ids.size() << "\n";
    std::cout << "Avg tracks/frame: " << total_tracks / (frame_id - 1.0) << "\n";
    
    return 0;
}
```

### 2. Multi-Camera Tracking

Track objects across multiple camera views.

```cpp
// examples/multi_camera.cpp
#include <motcpp/trackers/strongsort.hpp>
#include <thread>
#include <map>

struct CameraTracker {
    std::unique_ptr<motcpp::trackers::StrongSORT> tracker;
    int camera_id;
    std::map<int, Eigen::VectorXf> track_embeddings;
};

// Match tracks across cameras using embeddings
std::map<std::pair<int,int>, int> match_cross_camera(
    const std::vector<CameraTracker>& cameras,
    float threshold = 0.5f) {
    
    std::map<std::pair<int,int>, int> global_ids;
    int next_global_id = 1;
    
    // For each pair of cameras
    for (size_t i = 0; i < cameras.size(); ++i) {
        for (size_t j = i + 1; j < cameras.size(); ++j) {
            // Compare embeddings
            for (auto& [id1, emb1] : cameras[i].track_embeddings) {
                for (auto& [id2, emb2] : cameras[j].track_embeddings) {
                    float dist = (emb1 - emb2).norm();
                    if (dist < threshold) {
                        // Same person across cameras
                        auto key_i = std::make_pair(cameras[i].camera_id, id1);
                        auto key_j = std::make_pair(cameras[j].camera_id, id2);
                        
                        if (global_ids.count(key_i)) {
                            global_ids[key_j] = global_ids[key_i];
                        } else if (global_ids.count(key_j)) {
                            global_ids[key_i] = global_ids[key_j];
                        } else {
                            global_ids[key_i] = next_global_id;
                            global_ids[key_j] = next_global_id;
                            ++next_global_id;
                        }
                    }
                }
            }
        }
    }
    
    return global_ids;
}
```

### 3. Streaming Integration

Integrate with video streaming protocols.

```cpp
// examples/streaming.cpp
#include <motcpp/trackers/bytetrack.hpp>

int main() {
    // RTSP stream
    cv::VideoCapture cap("rtsp://camera_ip:554/stream");
    
    // Low-latency tracker settings
    motcpp::trackers::ByteTrack tracker(
        0.4f,   // Higher threshold for cleaner tracks
        10,     // Short max_age for responsiveness
        20,     // Limited history
        1,      // Immediate confirmation
        0.4f,   // Higher IoU threshold
        false, 80, "iou", false,
        0.2f, 0.5f, 0.9f, 10, 25.0f
    );
    
    cv::Mat frame;
    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();
        
        Eigen::MatrixXf dets = fast_detector(frame);
        auto tracks = tracker.update(dets, frame);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count();
        
        // Ensure low latency
        if (latency > 33) {  // > 30 FPS
            std::cerr << "Warning: High latency " << latency << "ms\n";
        }
        
        // Send tracks downstream
        publish_tracks(tracks);
    }
    return 0;
}
```

---

## File Index

| File | Description |
|------|-------------|
| `simple_tracking.cpp` | Minimal tracking example |
| `pedestrian_tracking.cpp` | Full pedestrian tracker |
| `multi_camera.cpp` | Cross-camera tracking |
| `streaming.cpp` | RTSP stream integration |
| `benchmark.cpp` | Performance benchmarking |
| `custom_tracker.cpp` | Implementing custom tracker |
