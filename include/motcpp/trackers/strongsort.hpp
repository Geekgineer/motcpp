// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/appearance/reid_backend.hpp>
#include <motcpp/motion/cmc/ecc.hpp>
#include <motcpp/motion/kalman_filters/xyah_kf.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>

namespace motcpp::trackers::strongsort {

// Forward declarations
class Track;
class Detection;

/**
 * Track state enumeration
 * C++ implementation based on trackers/strongsort/sort/track.py
 */
enum class TrackState {
    Tentative = 1,
    Confirmed = 2,
    Deleted = 3
};

/**
 * Detection class representing a bounding box detection
 * C++ implementation based on trackers/strongsort/sort/detection.py
 */
class Detection {
public:
    Detection(const Eigen::Vector4f& tlwh, float conf, int cls, int det_ind, 
              const Eigen::VectorXf& feat = Eigen::VectorXf());
    
    /**
     * Convert bounding box to format (center x, center y, aspect ratio, height)
     * From [x, y, w, h] to [cx, cy, w/h, h]
     */
    Eigen::Vector4f to_xyah() const;
    
    // Public members matching Python structure
    Eigen::Vector4f tlwh;  // top-left x, top-left y, width, height
    float conf;
    int cls;
    int det_ind;
    Eigen::VectorXf feat;  // appearance feature vector
};

/**
 * Single target track with state space (x, y, a, h) and associated velocities
 * C++ implementation based on trackers/strongsort/sort/track.py
 */
class Track {
public:
    Track(const Detection& detection, int id, int n_init, int max_age, float ema_alpha);
    
    /**
     * Get current position in bounding box format (top left x, top left y, width, height)
     */
    Eigen::Vector4f to_tlwh() const;
    
    /**
     * Get current position in bounding box format (min x, min y, max x, max y)
     */
    Eigen::Vector4f to_tlbr() const;
    
    /**
     * Apply camera motion compensation (warp matrix)
     */
    void camera_update(const Eigen::Matrix<float, 2, 3>& warp_matrix);
    
    /**
     * Increment age and time_since_update
     */
    void increment_age();
    
    /**
     * Propagate state distribution to current time step
     */
    void predict();
    
    /**
     * Perform Kalman filter measurement update and update feature cache
     */
    void update(const Detection& detection);
    
    /**
     * Mark this track as missed (no association at current time step)
     */
    void mark_missed();
    
    // State queries
    bool is_tentative() const { return state_ == TrackState::Tentative; }
    bool is_confirmed() const { return state_ == TrackState::Confirmed; }
    bool is_deleted() const { return state_ == TrackState::Deleted; }
    
    // Public members matching Python structure
    int id;
    Eigen::VectorXf mean;           // Mean vector of state distribution
    Eigen::MatrixXf covariance;     // Covariance matrix of state distribution
    motion::KalmanFilterXYAH kf;    // Kalman filter instance
    std::vector<Eigen::VectorXf> features;  // Feature cache (stores only smoothed EMA feature)
    float conf;
    int cls;
    int det_ind;
    Eigen::Vector4f bbox;           // Current bbox in xyah format
    
    // Track statistics
    int hits;
    int age;
    int time_since_update;
    
private:
    TrackState state_;
    int n_init_;
    int max_age_;
    float ema_alpha_;
};

/**
 * Nearest neighbor distance metric
 * C++ implementation based on trackers/strongsort/sort/linear_assignment.py
 */
class NearestNeighborDistanceMetric {
public:
    NearestNeighborDistanceMetric(const std::string& metric, float matching_threshold, 
                                   int budget = -1);
    
    /**
     * Update the distance metric with new data
     */
    void partial_fit(const Eigen::MatrixXf& features, const std::vector<int>& targets,
                     const std::vector<int>& active_targets);
    
    /**
     * Compute distance between features and targets
     * Returns cost matrix of shape (len(targets), len(features))
     */
    Eigen::MatrixXf distance(const Eigen::MatrixXf& features, const std::vector<int>& targets) const;
    
    /**
     * Reset metric state (clear samples)
     */
    void reset();
    
    float matching_threshold;
    
private:
    std::string metric_type_;
    int budget_;
    std::unordered_map<int, std::vector<Eigen::VectorXf>> samples_;  // target_id -> features
    
    // Helper functions
    Eigen::VectorXf nn_cosine_distance(const std::vector<Eigen::VectorXf>& x_samples,
                                       const Eigen::MatrixXf& y) const;
    Eigen::MatrixXf cosine_distance(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b,
                                     bool data_is_normalized = false) const;
};

/**
 * Matching cascade and linear assignment functions
 * C++ implementation based on trackers/strongsort/sort/linear_assignment.py
 */
namespace linear_assignment {

constexpr float INFTY_COST = 1e5f;

/**
 * Solve linear assignment problem
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
min_cost_matching(
    std::function<Eigen::MatrixXf(const std::vector<Track>&, const std::vector<Detection>&,
                                   const std::vector<int>&, const std::vector<int>&)> distance_metric,
    float max_distance,
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices = {},
    const std::vector<int>& detection_indices = {});

/**
 * Run matching cascade (simple wrapper calling min_cost_matching once)
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
matching_cascade(
    std::function<Eigen::MatrixXf(const std::vector<Track>&, const std::vector<Detection>&,
                                   const std::vector<int>&, const std::vector<int>&)> distance_metric,
    float max_distance,
    int cascade_depth,
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices = {},
    const std::vector<int>& detection_indices = {});

/**
 * Invalidate infeasible entries in cost matrix based on Kalman filter gating
 */
Eigen::MatrixXf gate_cost_matrix(
    const Eigen::MatrixXf& cost_matrix,
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices,
    float mc_lambda,
    float gated_cost = INFTY_COST,
    bool only_position = false);

} // namespace linear_assignment

/**
 * IoU matching functions
 */
namespace iou_matching {

/**
 * Compute intersection over union between bbox and candidates
 * Returns vector of IoU values (one per candidate)
 */
Eigen::VectorXf iou(const Eigen::Vector4f& bbox, const Eigen::MatrixXf& candidates);

/**
 * IoU-based cost metric
 * Returns cost matrix of shape (len(track_indices), len(detection_indices))
 */
Eigen::MatrixXf iou_cost(const std::vector<Track>& tracks,
                         const std::vector<Detection>& detections,
                         const std::vector<int>& track_indices = {},
                         const std::vector<int>& detection_indices = {});

} // namespace iou_matching

/**
 * Internal Tracker class for StrongSORT
 * C++ implementation based on trackers/strongsort/sort/tracker.py
 */
class Tracker {
public:
    Tracker(std::shared_ptr<NearestNeighborDistanceMetric> metric,
            float max_iou_dist = 0.9f,
            int max_age = 30,
            int n_init = 3,
            float mc_lambda = 0.995f,
            float ema_alpha = 0.9f);
    
    /**
     * Propagate track state distributions one time step forward
     */
    void predict();
    
    /**
     * Perform measurement update and track management
     */
    void update(const std::vector<Detection>& detections);
    
    /**
     * Reset tracker state (clear tracks and reset ID counter)
     */
    void reset();
    
    std::vector<Track> tracks;
    
private:
    /**
     * Run matching cascade and IoU matching
     */
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    match(const std::vector<Detection>& detections);
    
    /**
     * Initiate new track
     */
    void initiate_track(const Detection& detection);
    
    std::shared_ptr<NearestNeighborDistanceMetric> metric_;
    float max_iou_dist_;
    int max_age_;
    int n_init_;
    float mc_lambda_;
    float ema_alpha_;
    int next_id_;
    std::unique_ptr<motion::ECC> cmc_;  // Unused but present for compatibility
};

} // namespace motcpp::trackers::strongsort

// ============================================================================
// Public StrongSORT API
// ============================================================================

namespace motcpp::trackers {

/**
 * StrongSORT tracker implementation
 * C++ implementation based on trackers/strongsort/strongsort.py
 */
class StrongSORT : public BaseTracker {
public:
    StrongSORT(const std::string& reid_weights,
               bool use_half = false,
               bool use_gpu = false,
               float det_thresh = 0.3f,
               int max_age = 30,
               int max_obs = 50,
               int min_hits = 3,
               float iou_threshold = 0.3f,
               bool per_class = false,
               int nr_classes = 80,
               const std::string& asso_func = "iou",
               bool is_obb = false,
               float min_conf = 0.1f,
               float max_cos_dist = 0.2f,
               float max_iou_dist = 0.7f,
               int n_init = 3,
               int nn_budget = 100,
               float mc_lambda = 0.98f,
               float ema_alpha = 0.9f);
    
    ~StrongSORT() override;
    
    /**
     * Update tracker with new detections
     */
    Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                          const cv::Mat& img,
                          const Eigen::MatrixXf& embs = Eigen::MatrixXf()) override;
    
    /**
     * Reset tracker state
     */
    void reset() override;

private:
    float min_conf_;
    float max_cos_dist_;
    float max_iou_dist_;
    int n_init_;
    int nn_budget_;
    float mc_lambda_;
    float ema_alpha_;
    
    std::unique_ptr<appearance::ReIDBackend> reid_backend_;  // nullptr if using pre-generated embeddings
    std::unique_ptr<motion::ECC> cmc_;
    std::unique_ptr<strongsort::Tracker> tracker_;
    
    // Pre-allocated buffers for performance
    mutable Eigen::MatrixXf dets_buffer_;
    mutable std::vector<strongsort::Detection> detections_buffer_;
};

} // namespace motcpp::trackers
