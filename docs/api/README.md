# API Reference

Complete API documentation for motcpp.

## Quick Links

| Module | Description |
|--------|-------------|
| [Core](#core) | Base classes and configuration |
| [Trackers](#trackers) | Tracking algorithms |
| [Motion](#motion) | Kalman filters and CMC |
| [Appearance](#appearance) | ReID backends |
| [Utils](#utils) | Utility functions |

---

## Core

### TrackerConfig

Configuration structure for tracker initialization.

```cpp
namespace motcpp {

struct TrackerConfig {
    float det_thresh = 0.3f;        // Detection confidence threshold
    int max_age = 30;               // Frames before track deletion
    int max_obs = 50;               // Max observations to store
    int min_hits = 3;               // Hits before track confirmation
    float iou_threshold = 0.3f;     // IoU threshold for matching
    bool per_class = false;         // Per-class tracking
    int nr_classes = 80;            // Number of object classes
    std::string asso_func = "iou";  // Association function
    bool is_obb = false;            // Oriented bounding boxes
    
    // ReID specific
    std::string reid_weights;       // ReID model path
    bool use_half = false;          // FP16 inference
    bool use_gpu = false;           // GPU inference
    
    float frame_rate = 30.0f;       // Video frame rate
};

}
```

### BaseTracker

Abstract base class for all trackers.

```cpp
namespace motcpp {

class BaseTracker {
public:
    explicit BaseTracker(const TrackerConfig& config = TrackerConfig{});
    virtual ~BaseTracker() = default;
    
    // Main interface
    virtual Eigen::MatrixXf update(
        const Eigen::MatrixXf& dets,  // (N, 6): [x1,y1,x2,y2,conf,cls]
        const cv::Mat& img,
        const Eigen::MatrixXf& embs = Eigen::MatrixXf()  // (N, D)
    ) = 0;
    
    virtual void reset() = 0;
    
    // Visualization
    cv::Mat draw(const cv::Mat& img, 
                 bool show_trajectory = false,
                 int thickness = 2) const;
    
    static cv::Scalar id_to_color(int id);
    
protected:
    TrackerConfig config_;
    int frame_count_;
    std::vector<Track> tracks_;
};

}
```

### Track

Single track representation.

```cpp
namespace motcpp {

struct Track {
    int id;                         // Unique track ID
    Eigen::Vector4f bbox;           // [x1, y1, x2, y2]
    float confidence;               // Detection confidence
    int class_id;                   // Object class
    int det_index;                  // Detection index
    TrackState state;               // NEW, TRACKED, LOST, REMOVED
    int age;                        // Track age in frames
    int hits;                       // Total detection hits
    int time_since_update;          // Frames since last update
    Eigen::VectorXf embedding;      // Appearance embedding
};

}
```

### Factory Function

```cpp
namespace motcpp {

std::unique_ptr<BaseTracker> create_tracker(
    const std::string& name,          // "sort", "bytetrack", etc.
    const TrackerConfig& config = {}
);

}
```

---

## Trackers

### Sort

```cpp
namespace motcpp::trackers {

class Sort : public BaseTracker {
public:
    Sort(float det_thresh = 0.3f,
         int max_age = 1,
         int max_obs = 50,
         int min_hits = 3,
         float iou_threshold = 0.3f,
         bool per_class = false,
         int nr_classes = 80,
         const std::string& asso_func = "iou",
         bool is_obb = false);
    
    Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                          const cv::Mat& img,
                          const Eigen::MatrixXf& embs = {}) override;
    void reset() override;
};

}
```

### ByteTrack

```cpp
namespace motcpp::trackers {

class ByteTrack : public BaseTracker {
public:
    ByteTrack(float det_thresh = 0.3f,
              int max_age = 30,
              int max_obs = 50,
              int min_hits = 3,
              float iou_threshold = 0.3f,
              bool per_class = false,
              int nr_classes = 80,
              const std::string& asso_func = "iou",
              bool is_obb = false,
              float min_conf = 0.1f,
              float track_thresh = 0.45f,
              float match_thresh = 0.8f,
              int track_buffer = 30,
              float frame_rate = 30.0f);
    
    Eigen::MatrixXf update(...) override;
    void reset() override;
};

}
```

### OCSort

```cpp
namespace motcpp::trackers {

class OCSort : public BaseTracker {
public:
    OCSort(float det_thresh = 0.2f,
           int max_age = 30,
           int max_obs = 50,
           int min_hits = 3,
           float iou_threshold = 0.3f,
           bool per_class = false,
           int nr_classes = 80,
           const std::string& asso_func = "iou",
           bool is_obb = false,
           float min_conf = 0.1f,
           int delta_t = 3,
           float inertia = 0.2f,
           bool use_byte = false,
           float Q_xy_scaling = 0.01f,
           float Q_s_scaling = 0.0001f);
    
    Eigen::MatrixXf update(...) override;
    void reset() override;
};

}
```

---

## Motion

### KalmanFilterXYSR

7D Kalman filter with state `[x, y, s, r, vx, vy, vs]`.

```cpp
namespace motcpp::motion {

class KalmanFilterXYSR {
public:
    KalmanFilterXYSR(int dim_x = 7, int dim_z = 4, int max_obs = 50);
    
    void predict();
    void update(const Eigen::VectorXf& z);
    void apply_affine_correction(const Eigen::Matrix2f& m,
                                 const Eigen::Vector2f& t);
    
    // State
    Eigen::VectorXf x;              // State vector
    Eigen::MatrixXf P;              // Covariance
    Eigen::MatrixXf F;              // Transition matrix
    Eigen::MatrixXf H;              // Observation matrix
    std::deque<Eigen::VectorXf> history_obs;
};

}
```

---

## Utils

### IoU Functions

```cpp
namespace motcpp::utils {

// Batch IoU computation
Eigen::MatrixXf iou_batch(
    const Eigen::MatrixXf& bboxes1,  // (N, 4)
    const Eigen::MatrixXf& bboxes2   // (M, 4)
);  // Returns (N, M)

// Variants
Eigen::MatrixXf giou_batch(...);  // Generalized IoU
Eigen::MatrixXf diou_batch(...);  // Distance IoU
Eigen::MatrixXf ciou_batch(...);  // Complete IoU
Eigen::MatrixXf hmiou_batch(...); // Height-modified IoU

// IoU distance (1 - IoU)
Eigen::MatrixXf iou_distance(
    const Eigen::MatrixXf& tracks,
    const Eigen::MatrixXf& dets
);

}
```

### Linear Assignment

```cpp
namespace motcpp::utils {

struct LinearAssignmentResult {
    std::vector<std::array<int, 2>> matches;
    std::vector<int> unmatched_a;
    std::vector<int> unmatched_b;
};

LinearAssignmentResult linear_assignment(
    const Eigen::MatrixXf& cost_matrix,
    float threshold
);

}
```

### Coordinate Conversions

```cpp
namespace motcpp::utils {

// [x1,y1,x2,y2] ↔ [cx,cy,s,r]
Eigen::Vector4f xyxy2xysr(const Eigen::Vector4f& xyxy);
Eigen::Vector4f xysr2xyxy(const Eigen::Vector4f& xysr);

// [x1,y1,x2,y2] ↔ [cx,cy,a,h]
Eigen::Vector4f xyxy2xyah(const Eigen::Vector4f& xyxy);
Eigen::Vector4f xyah2xyxy(const Eigen::Vector4f& xyah);

// [x1,y1,x2,y2] ↔ [x1,y1,w,h]
Eigen::Vector4f xyxy2xywh(const Eigen::Vector4f& xyxy);
Eigen::Vector4f xywh2xyxy(const Eigen::Vector4f& xywh);

}
```

---

## Error Handling

motcpp uses exceptions for error handling:

```cpp
try {
    auto tracker = motcpp::create_tracker("invalid_name");
} catch (const std::invalid_argument& e) {
    std::cerr << "Invalid tracker: " << e.what() << std::endl;
}

try {
    tracker->update(invalid_dets, frame);
} catch (const std::runtime_error& e) {
    std::cerr << "Update failed: " << e.what() << std::endl;
}
```

---

## See Also

- [Getting Started](../guides/getting-started.md)
- [Trackers Guide](../guides/trackers.md)
- [Architecture](../guides/architecture.md)
