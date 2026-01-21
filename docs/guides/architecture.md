# Architecture

This document describes the internal architecture of motcpp.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              motcpp                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Input     │    │   Tracker   │    │   Output    │                 │
│  │  Detections │───▶│   Engine    │───▶│   Tracks    │                 │
│  │  (N×6)      │    │             │    │   (M×8)     │                 │
│  └─────────────┘    └──────┬──────┘    └─────────────┘                 │
│                            │                                            │
│         ┌──────────────────┼──────────────────┐                        │
│         │                  │                  │                        │
│         ▼                  ▼                  ▼                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Motion    │    │ Association │    │ Appearance  │                 │
│  │   Model     │    │   Module    │    │   Model     │                 │
│  │  (Kalman)   │    │ (Hungarian) │    │   (ReID)    │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. BaseTracker

The abstract base class that all trackers inherit from.

```cpp
namespace motcpp {

class BaseTracker {
public:
    // Main interface
    virtual Eigen::MatrixXf update(
        const Eigen::MatrixXf& dets,
        const cv::Mat& img,
        const Eigen::MatrixXf& embs = Eigen::MatrixXf()) = 0;
    
    virtual void reset() = 0;
    
protected:
    TrackerConfig config_;
    int frame_count_;
    std::vector<Track> tracks_;
};

}
```

### 2. Motion Models

Kalman filters for state estimation and prediction.

```
┌────────────────────────────────────────────────────────┐
│                    Motion Models                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  KalmanFilterXYSR          KalmanFilterXYAH            │
│  ├─ State: [x,y,s,r,       ├─ State: [x,y,a,h,         │
│  │          vx,vy,vs]      │          vx,vy,va,vh]     │
│  ├─ Obs: [x,y,s,r]         ├─ Obs: [x,y,a,h]           │
│  └─ Used by: SORT,         └─ Used by: StrongSORT,     │
│              OC-SORT                   ByteTrack        │
│                                                        │
│  UCMCKalmanFilter                                      │
│  ├─ State: [x,vx,y,vy] (ground plane)                  │
│  └─ Used by: UCMCTrack                                 │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 3. Association Module

Data association using various cost functions.

```cpp
namespace motcpp::utils {

// Cost matrices
Eigen::MatrixXf iou_batch(bboxes1, bboxes2);      // Standard IoU
Eigen::MatrixXf giou_batch(bboxes1, bboxes2);     // Generalized IoU
Eigen::MatrixXf diou_batch(bboxes1, bboxes2);     // Distance IoU
Eigen::MatrixXf ciou_batch(bboxes1, bboxes2);     // Complete IoU

// Linear assignment (Hungarian algorithm)
LinearAssignmentResult linear_assignment(
    const Eigen::MatrixXf& cost_matrix,
    float threshold
);

}
```

### 4. Appearance Model (ReID)

ONNX-based appearance feature extraction.

```
┌────────────────────────────────────────────────────────┐
│                  Appearance Pipeline                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Input Image ──▶ Crop Detections ──▶ ReID Model ──▶   │
│                                                        │
│                  ┌─────────────┐                       │
│                  │  OSNet      │                       │
│                  │  ResNet     │                       │
│                  │  CLIP       │  ──▶ Embeddings       │
│                  │  ...        │      (N × 512)        │
│                  └─────────────┘                       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Data Flow

### Input Format

Detections: `Eigen::MatrixXf` with shape `(N, 6)`

| Column | Description |
|--------|-------------|
| 0 | x1 (left) |
| 1 | y1 (top) |
| 2 | x2 (right) |
| 3 | y2 (bottom) |
| 4 | confidence |
| 5 | class_id |

### Output Format

Tracks: `Eigen::MatrixXf` with shape `(M, 8)`

| Column | Description |
|--------|-------------|
| 0 | x1 (left) |
| 1 | y1 (top) |
| 2 | x2 (right) |
| 3 | y2 (bottom) |
| 4 | track_id |
| 5 | confidence |
| 6 | class_id |
| 7 | detection_index |

## Tracking Pipeline

```
Frame t
   │
   ▼
┌──────────────────┐
│ 1. Predict       │  ← Kalman filter prediction
│    tracks[t-1]   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. Associate     │  ← Hungarian matching
│    dets[t] ↔     │
│    tracks[t-1]   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. Update        │  ← Kalman filter update
│    matched       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Initialize    │  ← New tracks for
│    unmatched     │     unmatched dets
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. Delete        │  ← Remove lost
│    old tracks    │     tracks
└────────┬─────────┘
         │
         ▼
    tracks[t]
```

## Track Lifecycle

```
                    ┌─────────┐
                    │   NEW   │
                    └────┬────┘
                         │
                         │ hits >= min_hits
                         ▼
    ┌─────────────────────────────────────┐
    │                                     │
    │              TRACKED                │◀──┐
    │                                     │   │
    └─────────────────┬───────────────────┘   │
                      │                       │
                      │ time_since_update > 0 │ matched
                      ▼                       │
                ┌─────────┐                   │
                │  LOST   │───────────────────┘
                └────┬────┘
                     │
                     │ age > max_age
                     ▼
                ┌─────────┐
                │ REMOVED │
                └─────────┘
```

## Memory Management

motcpp uses modern C++ memory management:

- **Unique pointers** for tracker ownership
- **Eigen matrices** for efficient linear algebra
- **Move semantics** for zero-copy data transfer
- **RAII** for resource management

```cpp
// Example: Efficient track creation
auto tracker = std::make_unique<ByteTrack>(config);

// Zero-copy update
Eigen::MatrixXf tracks = tracker->update(dets, frame);

// Move semantics
process_tracks(std::move(tracks));
```

## Thread Safety

- Single tracker instances are **NOT** thread-safe
- Create separate tracker instances for parallel processing
- Shared data (e.g., ReID model) uses internal synchronization

```cpp
// Parallel video processing
#pragma omp parallel for
for (int i = 0; i < num_videos; ++i) {
    auto tracker = create_tracker("bytetrack");  // Thread-local
    process_video(videos[i], tracker.get());
}
```

## Extension Points

### Custom Tracker

```cpp
class MyTracker : public motcpp::BaseTracker {
public:
    MyTracker(const TrackerConfig& config)
        : BaseTracker(config) {}
    
    Eigen::MatrixXf update(
        const Eigen::MatrixXf& dets,
        const cv::Mat& img,
        const Eigen::MatrixXf& embs) override {
        // Your tracking logic here
    }
    
    void reset() override {
        // Reset state
    }
};
```

### Custom Association Function

```cpp
Eigen::MatrixXf my_distance(
    const Eigen::MatrixXf& bboxes1,
    const Eigen::MatrixXf& bboxes2) {
    // Custom distance metric
    return distance_matrix;
}

// Register with AssociationFunction
```
