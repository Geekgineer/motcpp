// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/motion/kalman_filters/xysr_kf.hpp>
#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/ops.hpp>
#include <motcpp/utils/iou.hpp>
#include <vector>
#include <memory>

namespace motcpp::trackers {

/**
 * Single track representation for SORT
 * Uses XYSR (center x, center y, scale, aspect ratio) state space
 */
class SortTrack {
public:
    SortTrack(const Eigen::VectorXf& det);
    
    void predict();
    void update(const Eigen::VectorXf& det);
    
    Eigen::Vector4f get_state() const;  // Returns [x1, y1, x2, y2]
    
    int id() const { return id_; }
    float conf() const { return conf_; }
    int cls() const { return cls_; }
    int det_ind() const { return det_ind_; }
    int hits() const { return hits_; }
    int time_since_update() const { return time_since_update_; }
    int age() const { return age_; }
    
    void set_det_ind(int ind) { det_ind_ = ind; }

private:
    static int next_id();
    
    int id_;
    float conf_;
    int cls_;
    int det_ind_;
    int hits_;
    int time_since_update_;
    int age_;
    
    motion::KalmanFilterXYSR kf_;
    Eigen::VectorXf mean_;
    Eigen::MatrixXf covariance_;
};

/**
 * SORT (Simple Online and Realtime Tracking)
 * 
 * Original paper: "Simple Online and Realtime Tracking"
 * by Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft
 * https://arxiv.org/abs/1602.00763
 * 
 * This is the foundational tracker that many others build upon:
 * - Uses Kalman filter for motion prediction
 * - Hungarian algorithm with IoU for data association
 * - No appearance features (motion-only)
 */
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
                          const Eigen::MatrixXf& embs = Eigen::MatrixXf()) override;
    
    void reset() override;
    
private:
    int frame_count_;
    std::vector<SortTrack> trackers_;
};

} // namespace motcpp::trackers
