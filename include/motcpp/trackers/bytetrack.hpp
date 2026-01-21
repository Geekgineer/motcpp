// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/motion/kalman_filters/xyah_kf.hpp>
#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/ops.hpp>
#include <motcpp/utils/iou.hpp>
#include <deque>
#include <vector>
#include <memory>
#include <unordered_set>

namespace motcpp::trackers {

/**
 * Track state for ByteTrack
 */
enum class ByteTrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

/**
 * Single track representation for ByteTrack
 */
class STrack {
public:
    static int next_id() {
        static int count = 0;
        return ++count;
    }
    
    static void clear_count() {
        // Reset is handled by static initialization
    }
    
    STrack(const Eigen::VectorXf& det, int max_obs);
    
    void activate(motion::KalmanFilterXYAH& kalman_filter, int frame_id);
    void re_activate(const STrack& new_track, int frame_id, bool new_id = false);
    void update(const STrack& new_track, int frame_id);
    void predict();
    
    static void multi_predict(std::vector<STrack>& stracks);
    
    void mark_lost() { state_ = ByteTrackState::Lost; }
    void mark_removed() { state_ = ByteTrackState::Removed; }
    
    Eigen::Vector4f xyxy() const;
    Eigen::Vector4f to_xyah() const { return xyah_; }
    
    int id() const { return id_; }
    float conf() const { return conf_; }
    int cls() const { return cls_; }
    int det_ind() const { return det_ind_; }
    ByteTrackState state() const { return state_; }
    bool is_activated() const { return is_activated_; }
    int frame_id() const { return frame_id_; }
    int start_frame() const { return start_frame_; }
    int end_frame() const { return frame_id_; }
    
    Eigen::VectorXf mean;
    Eigen::MatrixXf covariance;
    
private:
    Eigen::Vector4f xywh_;
    Eigen::Vector4f tlwh_;
    Eigen::Vector4f xyah_;
    float conf_;
    int cls_;
    int det_ind_;
    int max_obs_;
    
    motion::KalmanFilterXYAH* kalman_filter_;
    int id_;
    ByteTrackState state_;
    bool is_activated_;
    int tracklet_len_;
    int frame_id_;
    int start_frame_;
    
    std::deque<Eigen::Vector4f> history_observations_;
    
    static motion::KalmanFilterXYAH shared_kalman_;
};

/**
 * ByteTrack tracker implementation
 */
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
              int track_buffer = 25,
              int frame_rate = 30);
    
    Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                          const cv::Mat& img,
                          const Eigen::MatrixXf& embs = Eigen::MatrixXf()) override;
    
    void reset() override;
    
private:
    float min_conf_;
    float track_thresh_;
    float match_thresh_;
    int track_buffer_;
    int buffer_size_;
    int max_time_lost_;
    int frame_id_;
    
    motion::KalmanFilterXYAH kalman_filter_;
    std::vector<STrack> active_tracks_;
    std::vector<STrack> lost_stracks_;
    std::vector<STrack> removed_stracks_;
    
    // Pre-allocated buffers for zero-allocation hot path
    mutable Eigen::MatrixXf cost_matrix_buffer_;
    mutable Eigen::MatrixXf track_xyxy_buffer_;
    mutable Eigen::MatrixXf det_xyxy_buffer_;
    mutable Eigen::VectorXf det_confs_buffer_;
    mutable std::vector<STrack> strack_pool_buffer_;
    mutable std::vector<int> index_buffer_;
    mutable std::vector<std::pair<int, bool>> track_index_map_buffer_;
    
    // Helper functions
    std::vector<STrack> joint_stracks(const std::vector<STrack>& tlista,
                                      const std::vector<STrack>& tlistb) const;
    std::vector<STrack> sub_stracks(const std::vector<STrack>& tlista,
                                    const std::vector<STrack>& tlistb) const;
    std::pair<std::vector<STrack>, std::vector<STrack>> remove_duplicate_stracks(
        const std::vector<STrack>& stracksa,
        const std::vector<STrack>& stracksb) const;
};

} // namespace motcpp::trackers

