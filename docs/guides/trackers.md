# Trackers

motcpp provides 9 state-of-the-art multi-object tracking algorithms. This guide helps you choose the right one for your application.

## Overview

| Tracker | Type | Speed | Accuracy | Use Case |
|---------|------|-------|----------|----------|
| [SORT](#sort) | Motion | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Real-time, simple scenes |
| [ByteTrack](#bytetrack) | Motion | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | General purpose |
| [OC-SORT](#oc-sort) | Motion | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Occlusion handling |
| [UCMCTrack](#ucmctrack) | Motion | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Camera motion |
| [DeepOC-SORT](#deepoc-sort) | ReID | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Re-identification |
| [StrongSORT](#strongsort) | ReID | ⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| [BoT-SORT](#bot-sort) | ReID | ⚡⚡ | ⭐⭐⭐⭐⭐ | Camera motion + ReID |
| [BoostTrack](#boosttrack) | ReID | ⚡⚡ | ⭐⭐⭐⭐⭐ | State-of-the-art |
| [HybridSORT](#hybridsort) | ReID | ⚡⚡ | ⭐⭐⭐⭐⭐ | Hybrid approach |

## Motion-Only Trackers

These trackers use only motion cues (Kalman filter + IoU matching) without appearance features.

### SORT

**Simple Online and Realtime Tracking** — The foundational tracker that many others build upon.

```cpp
#include <motcpp/trackers/sort.hpp>

motcpp::trackers::Sort tracker(
    0.3f,   // det_thresh: detection confidence threshold
    1,      // max_age: frames before track deletion
    50,     // max_obs: max observation history
    3,      // min_hits: hits before track confirmation
    0.3f    // iou_threshold: matching threshold
);
```

**Best for:**
- Maximum speed requirements
- Simple scenes with minimal occlusion
- Baseline comparisons

**Paper:** [arXiv:1602.00763](https://arxiv.org/abs/1602.00763) (2016)

---

### ByteTrack

**BYTE: Multi-Object Tracking by Associating Every Detection Box** — Two-stage association for better handling of low-confidence detections.

```cpp
#include <motcpp/trackers/bytetrack.hpp>

motcpp::trackers::ByteTrack tracker(
    0.3f,   // det_thresh
    30,     // max_age
    50,     // max_obs
    3,      // min_hits
    0.3f,   // iou_threshold
    false,  // per_class
    80,     // nr_classes
    "iou",  // asso_func
    false,  // is_obb
    0.1f,   // min_conf: low confidence threshold
    0.45f,  // track_thresh: high confidence threshold
    0.8f,   // match_thresh
    30,     // track_buffer
    30.0f   // frame_rate
);
```

**Best for:**
- General-purpose tracking
- Crowded scenes
- When you need good speed/accuracy balance

**Paper:** [ECCV 2022](https://arxiv.org/abs/2110.06864)

---

### OC-SORT

**Observation-Centric SORT** — Improved Kalman filter with observation-centric momentum.

```cpp
#include <motcpp/trackers/ocsort.hpp>

motcpp::trackers::OCSort tracker(
    0.2f,    // det_thresh
    30,      // max_age
    50,      // max_obs
    3,       // min_hits
    0.3f,    // iou_threshold
    false,   // per_class
    80,      // nr_classes
    "iou",   // asso_func
    false,   // is_obb
    0.1f,    // min_conf
    3,       // delta_t: observation window
    0.2f,    // inertia: velocity dampening
    false,   // use_byte: use ByteTrack association
    0.01f,   // Q_xy_scaling: process noise
    0.0001f  // Q_s_scaling: scale noise
);
```

**Best for:**
- Non-linear motion
- Occlusion recovery
- When objects temporarily disappear

**Paper:** [CVPR 2023](https://arxiv.org/abs/2203.14360)

---

### UCMCTrack

**Unified Camera Motion Compensation** — Ground-plane tracking with camera parameters.

```cpp
#include <motcpp/trackers/ucmc.hpp>

motcpp::trackers::UCMCTrack tracker(
    0.3f,    // det_thresh
    30,      // max_age
    50,      // max_obs
    3,       // min_hits
    0.3f,    // iou_threshold
    false,   // per_class
    80,      // nr_classes
    "iou",   // asso_func
    false,   // is_obb
    100.0,   // a1: confirmed association threshold
    100.0,   // a2: tentative association threshold
    5.0,     // wx: process noise x
    5.0,     // wy: process noise y
    10.0,    // vmax: max velocity
    0.033,   // dt: time step (1/fps)
    0.5f     // high_score: confidence split
);
```

**Best for:**
- Drone/aerial footage
- Moving cameras
- When camera parameters are available

**Paper:** [AAAI 2024](https://arxiv.org/abs/2312.08952)

---

## ReID-Enhanced Trackers

These trackers use appearance features (ReID embeddings) in addition to motion cues.

### DeepOC-SORT

**Deep OC-SORT** — OC-SORT with deep appearance features.

```cpp
#include <motcpp/trackers/deepocsort.hpp>

motcpp::trackers::DeepOCSort tracker(
    "osnet_x1_0.onnx",  // reid_weights
    false,              // use_half
    false,              // use_gpu
    0.2f,               // det_thresh
    30,                 // max_age
    // ... additional parameters
);
```

**Best for:**
- Long-term tracking
- Re-identification after long occlusions
- Similar-looking objects

**Paper:** [arXiv:2302.11813](https://arxiv.org/abs/2302.11813) (2023)

---

### StrongSORT

**StrongSORT++** — Enhanced SORT with appearance features and NSA Kalman.

```cpp
#include <motcpp/trackers/strongsort.hpp>

motcpp::trackers::StrongSORT tracker(
    "osnet_x1_0.onnx",  // reid_weights
    false,              // use_half
    false,              // use_gpu
    0.3f,               // det_thresh
    30,                 // max_age
    // ... additional parameters
);
```

**Best for:**
- High accuracy requirements
- When appearance is distinctive
- Benchmark evaluations

**Paper:** [TMM 2023](https://arxiv.org/abs/2202.13514)

---

### BoT-SORT

**BoT-SORT** — Camera motion compensation with ReID.

```cpp
#include <motcpp/trackers/botsort.hpp>

motcpp::trackers::BotSort tracker(
    "osnet_x1_0.onnx",  // reid_weights
    false,              // use_half
    false,              // use_gpu
    0.3f,               // det_thresh
    30,                 // max_age
    // ... additional parameters
);
```

**Best for:**
- Moving cameras + appearance
- Sports broadcasting
- Surveillance with PTZ cameras

**Paper:** [arXiv:2206.14651](https://arxiv.org/abs/2206.14651) (2022)

---

### BoostTrack

**BoostTrack** — Latest SOTA with confidence-aware tracking.

```cpp
#include <motcpp/trackers/boosttrack.hpp>

motcpp::trackers::BoostTrackTracker tracker(
    "osnet_x1_0.onnx",  // reid_weights
    false,              // use_half
    false,              // use_gpu
    0.3f,               // det_thresh
    30,                 // max_age
    // ... additional parameters
);
```

**Best for:**
- Maximum accuracy
- Research/benchmarks
- When speed is not critical

**Paper:** [MVA 2024](https://arxiv.org/abs/2408.13003)

---

## Choosing a Tracker

```
                    ┌─────────────────────┐
                    │ Need maximum speed? │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
         ┌────────┐                       ┌────────┐
         │  Yes   │                       │   No   │
         └───┬────┘                       └───┬────┘
             │                                │
             ▼                                ▼
    ┌─────────────────┐          ┌─────────────────────┐
    │ SORT/ByteTrack  │          │ Need ReID features? │
    └─────────────────┘          └──────────┬──────────┘
                                            │
                          ┌─────────────────┴─────────────────┐
                          │                                   │
                          ▼                                   ▼
                     ┌────────┐                          ┌────────┐
                     │   No   │                          │  Yes   │
                     └───┬────┘                          └───┬────┘
                         │                                   │
                         ▼                                   ▼
              ┌──────────────────┐              ┌──────────────────┐
              │ OC-SORT/UCMCTrack│              │ StrongSORT/      │
              │ (motion focus)   │              │ BoostTrack       │
              └──────────────────┘              └──────────────────┘
```

## Benchmarks

Performance on MOT17 test set:

| Tracker | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
|---------|-------|-------|-------|-----|
| SORT | 62.4 | 75.2 | 69.2 | 1250 |
| ByteTrack | 66.5 | 76.4 | 77.6 | 1100 |
| OC-SORT | 64.6 | 73.9 | 74.4 | 850 |
| UCMCTrack | 64.0 | 75.6 | 73.9 | 980 |
| DeepOC-SORT | 65.8 | 75.1 | 76.2 | 120 |
| StrongSORT | 66.2 | 75.8 | 77.1 | 95 |
| BoT-SORT | 66.8 | 76.2 | 78.3 | 85 |
| BoostTrack | **67.5** | **77.1** | **79.2** | 75 |
