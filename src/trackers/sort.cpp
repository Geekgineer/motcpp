// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/sort.hpp>
#include <motcpp/utils/iou.hpp>
#include <motcpp/utils/matching.hpp>
#include <algorithm>
#include <numeric>

namespace motcpp::trackers {

// ============================================================================
// SortTrack Implementation
// ============================================================================

int SortTrack::next_id() {
    static int count = 0;
    return ++count;
}

SortTrack::SortTrack(const Eigen::VectorXf& det)
    : id_(next_id())
    , conf_(det.size() > 4 ? det(4) : 1.0f)
    , cls_(det.size() > 5 ? static_cast<int>(det(5)) : 0)
    , det_ind_(det.size() > 6 ? static_cast<int>(det(6)) : -1)
    , hits_(1)
    , time_since_update_(0)
    , age_(1)
    , kf_(7, 4, 50)
{
    // Convert [x1, y1, x2, y2] to [cx, cy, s, r] and initialize KF
    Eigen::Vector4f xyxy = det.head<4>();
    Eigen::Vector4f xysr = utils::xyxy2xysr(xyxy);
    
    // Initialize Kalman filter state
    mean_ = Eigen::VectorXf::Zero(7);
    mean_.head<4>() = xysr;
    
    kf_.x = mean_;
    covariance_ = kf_.P;
}

void SortTrack::predict() {
    // Predict next state
    kf_.predict();
    mean_ = kf_.x;
    covariance_ = kf_.P;
    
    age_++;
    time_since_update_++;
}

void SortTrack::update(const Eigen::VectorXf& det) {
    // Update with new detection
    conf_ = det.size() > 4 ? det(4) : 1.0f;
    cls_ = det.size() > 5 ? static_cast<int>(det(5)) : cls_;
    det_ind_ = det.size() > 6 ? static_cast<int>(det(6)) : det_ind_;
    
    // Convert detection to XYSR format
    Eigen::Vector4f xyxy = det.head<4>();
    Eigen::Vector4f xysr = utils::xyxy2xysr(xyxy);
    
    // Update Kalman filter
    kf_.update(xysr);
    mean_ = kf_.x;
    covariance_ = kf_.P;
    
    hits_++;
    time_since_update_ = 0;
}

Eigen::Vector4f SortTrack::get_state() const {
    // Convert from XYSR to XYXY
    Eigen::Vector4f xysr = mean_.head<4>();
    return utils::xysr2xyxy(xysr);
}

// ============================================================================
// Sort Implementation
// ============================================================================

Sort::Sort(float det_thresh,
           int max_age,
           int max_obs,
           int min_hits,
           float iou_threshold,
           bool per_class,
           int nr_classes,
           const std::string& asso_func,
           bool is_obb)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                  per_class, nr_classes, asso_func, is_obb)
    , frame_count_(0)
{
}

void Sort::reset() {
    trackers_.clear();
    frame_count_ = 0;
}

Eigen::MatrixXf Sort::update(const Eigen::MatrixXf& dets,
                             const cv::Mat& img,
                             const Eigen::MatrixXf& embs) {
    (void)img;   // Not used in SORT
    (void)embs;  // Not used in SORT
    
    frame_count_++;
    
    // Filter detections by confidence threshold
    std::vector<int> valid_indices;
    for (int i = 0; i < dets.rows(); ++i) {
        if (dets(i, 4) >= det_thresh_) {
            valid_indices.push_back(i);
        }
    }
    
    Eigen::MatrixXf filtered_dets(valid_indices.size(), dets.cols());
    for (size_t i = 0; i < valid_indices.size(); ++i) {
        filtered_dets.row(i) = dets.row(valid_indices[i]);
    }
    
    // Get predicted locations from existing trackers
    std::vector<int> to_del;
    Eigen::MatrixXf trks(trackers_.size(), 4);
    
    for (size_t t = 0; t < trackers_.size(); ++t) {
        trackers_[t].predict();
        Eigen::Vector4f pos = trackers_[t].get_state();
        
        // Check for NaN values
        if (std::isnan(pos.sum())) {
            to_del.push_back(static_cast<int>(t));
        }
        trks.row(t) = pos.transpose();
    }
    
    // Remove invalid trackers
    for (auto it = to_del.rbegin(); it != to_del.rend(); ++it) {
        trackers_.erase(trackers_.begin() + *it);
    }
    
    // Resize trks if we removed any
    if (!to_del.empty()) {
        Eigen::MatrixXf new_trks(trackers_.size(), 4);
        for (size_t t = 0; t < trackers_.size(); ++t) {
            new_trks.row(t) = trackers_[t].get_state().transpose();
        }
        trks = new_trks;
    }
    
    // Association using IoU
    std::vector<std::array<int, 2>> matched;
    std::vector<int> unmatched_dets;
    std::vector<int> unmatched_trks;
    
    if (trackers_.empty()) {
        // All detections are unmatched
        for (int i = 0; i < filtered_dets.rows(); ++i) {
            unmatched_dets.push_back(i);
        }
    } else if (filtered_dets.rows() == 0) {
        // All trackers are unmatched
        for (size_t t = 0; t < trackers_.size(); ++t) {
            unmatched_trks.push_back(static_cast<int>(t));
        }
    } else {
        // Compute IoU distance matrix (1 - IoU)
        Eigen::MatrixXf det_boxes = filtered_dets.leftCols(4);
        Eigen::MatrixXf cost_matrix = utils::iou_distance(trks, det_boxes);
        
        // Run Hungarian algorithm (threshold is on distance, so 1 - iou_threshold)
        auto result = utils::linear_assignment(cost_matrix, 1.0f - iou_threshold_);
        
        matched = result.matches;
        unmatched_dets = result.unmatched_b;
        unmatched_trks = result.unmatched_a;
    }
    
    // Update matched trackers with assigned detections
    for (const auto& m : matched) {
        int trk_idx = m[0];
        int det_idx = m[1];
        
        Eigen::VectorXf det = filtered_dets.row(det_idx);
        // Add original detection index
        Eigen::VectorXf det_with_ind(det.size() + 1);
        det_with_ind.head(det.size()) = det;
        det_with_ind(det.size()) = static_cast<float>(valid_indices[det_idx]);
        
        trackers_[trk_idx].update(det_with_ind);
        trackers_[trk_idx].set_det_ind(valid_indices[det_idx]);
    }
    
    // Create new trackers for unmatched detections
    for (int det_idx : unmatched_dets) {
        Eigen::VectorXf det = filtered_dets.row(det_idx);
        // Add original detection index
        Eigen::VectorXf det_with_ind(det.size() + 1);
        det_with_ind.head(det.size()) = det;
        det_with_ind(det.size()) = static_cast<float>(valid_indices[det_idx]);
        
        trackers_.emplace_back(det_with_ind);
    }
    
    // Remove dead trackers
    std::vector<SortTrack> new_trackers;
    new_trackers.reserve(trackers_.size());
    
    for (const auto& trk : trackers_) {
        if (trk.time_since_update() <= max_age_) {
            new_trackers.push_back(trk);
        }
    }
    trackers_ = std::move(new_trackers);
    
    // Build output: tracks that have been updated recently and meet min_hits requirement
    std::vector<Eigen::VectorXf> outputs;
    outputs.reserve(trackers_.size());
    
    for (const auto& trk : trackers_) {
        // Output only if:
        // 1. Track was just updated (time_since_update == 0)
        // 2. Track has enough hits OR we're still in early frames
        if (trk.time_since_update() == 0 && 
            (trk.hits() >= min_hits_ || frame_count_ <= min_hits_)) {
            
            Eigen::Vector4f state = trk.get_state();
            
            Eigen::VectorXf output(8);
            output(0) = state(0);  // x1
            output(1) = state(1);  // y1
            output(2) = state(2);  // x2
            output(3) = state(3);  // y2
            output(4) = static_cast<float>(trk.id());
            output(5) = trk.conf();
            output(6) = static_cast<float>(trk.cls());
            output(7) = static_cast<float>(trk.det_ind());
            
            outputs.push_back(output);
        }
    }
    
    // Convert to matrix
    if (outputs.empty()) {
        return Eigen::MatrixXf(0, 8);
    }
    
    Eigen::MatrixXf result(outputs.size(), 8);
    for (size_t i = 0; i < outputs.size(); ++i) {
        result.row(i) = outputs[i].transpose();
    }
    
    return result;
}

} // namespace motcpp::trackers
