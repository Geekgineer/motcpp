// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/motion/kalman_filters/xysr_kf.hpp>
#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/ops.hpp>
#include <motcpp/utils/iou.hpp>
#include <deque>
#include <vector>
#include <memory>
#include <unordered_map>

namespace motcpp::trackers {

/**
 * Track state for OCSort
 */
enum class OCSortState {
    Tentative = 0,
    Confirmed = 1,
    Deleted = 2
};

/**
 * Single track representation for OCSort (KalmanBoxTracker)
 */
class KalmanBoxTracker {
public:
    static int next_id() {
        static int count = 0;
        return ++count;
    }
    
    static void clear_count() {
        // Reset handled by static initialization
    }
    
    KalmanBoxTracker(const Eigen::VectorXf& bbox, int cls, int det_ind,
                     int delta_t = 3, int max_obs = 50,
                     float Q_xy_scaling = 0.01f, float Q_s_scaling = 0.0001f);
    
    void update(const Eigen::VectorXf& bbox, int cls, int det_ind);
    Eigen::Vector4f predict();
    Eigen::Vector4f get_state() const;
    
    // Get observation from k frames ago
    Eigen::VectorXf k_previous_obs(int k) const;
    
    int id() const { return id_; }
    float conf() const { return conf_; }
    int cls() const { return cls_; }
    int det_ind() const { return det_ind_; }
    int age() const { return age_; }
    int hits() const { return hits_; }
    int hit_streak() const { return hit_streak_; }
    int time_since_update() const { return time_since_update_; }
    Eigen::Vector2f velocity() const { return velocity_; }
    Eigen::VectorXf last_observation() const { return last_observation_; }
    
    motion::KalmanFilterXYSR kf;
    
private:
    int id_;
    int age_;
    int hits_;
    int hit_streak_;
    int time_since_update_;
    float conf_;
    int cls_;
    int det_ind_;
    int delta_t_;
    int max_obs_;
    
    Eigen::VectorXf last_observation_;  // placeholder: [-1, -1, -1, -1, -1]
    std::unordered_map<int, Eigen::VectorXf> observations_;  // age -> bbox
    std::deque<Eigen::VectorXf> history_observations_;
    Eigen::Vector2f velocity_;  // velocity direction (dy, dx)
};

/**
 * OCSort tracker implementation
 */
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
    
    Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                          const cv::Mat& img,
                          const Eigen::MatrixXf& embs = Eigen::MatrixXf()) override;
    
    void reset() override;
    
private:
    float min_conf_;
    float asso_threshold_;
    int delta_t_;
    float inertia_;
    bool use_byte_;
    float Q_xy_scaling_;
    float Q_s_scaling_;
    int frame_id_;
    
    std::vector<KalmanBoxTracker> active_tracks_;
    
    // Pre-allocated buffers for zero-allocation hot path
    mutable Eigen::MatrixXf cost_matrix_buffer_;
    mutable Eigen::MatrixXf track_xyxy_buffer_;
    mutable Eigen::MatrixXf det_xyxy_buffer_;
    mutable Eigen::VectorXf det_confs_buffer_;
    mutable Eigen::MatrixXf velocity_buffer_;
    mutable Eigen::MatrixXf k_obs_buffer_;
    
    // Helper functions
    Eigen::Vector2f speed_direction(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const;
    Eigen::Vector4f convert_x_to_bbox(const Eigen::VectorXf& x) const;
};

} // namespace motcpp::trackers

