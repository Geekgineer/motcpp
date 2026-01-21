// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/motion/kalman_filters/xywh_kf.hpp>
#include <motcpp/motion/cmc/ecc.hpp>
#include <motcpp/motion/cmc/cmc.hpp>
#include <motcpp/appearance/onnx_backend.hpp>
#include <deque>
#include <unordered_map>

namespace motcpp::trackers {

// Track state enum for BoTSORT
enum class BotTrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

/**
 * @brief Single track for BoTSORT with appearance features
 */
class BotSTrack {
public:
    static int next_id_;
    static std::shared_ptr<KalmanFilterXYWH> shared_kalman_;
    
    BotSTrack(const Eigen::VectorXf& det, int max_obs);
    BotSTrack(const Eigen::VectorXf& det, const Eigen::VectorXf& feat, int max_obs);
    
    void predict();
    static void multi_predict(std::vector<BotSTrack*>& tracks);
    static void multi_gmc(std::vector<BotSTrack*>& tracks, const Eigen::Matrix3f& warp_matrix);
    
    void activate(std::shared_ptr<KalmanFilterXYWH> kalman_filter, int frame_id);
    void re_activate(const BotSTrack& new_track, int frame_id, bool new_id = false);
    void update(const BotSTrack& new_track, int frame_id);
    
    void mark_lost() { state_ = BotTrackState::Lost; }
    void mark_removed() { state_ = BotTrackState::Removed; }
    
    // Getters
    int id() const { return id_; }
    int frame_id() const { return frame_id_; }
    int start_frame() const { return start_frame_; }
    int end_frame() const { return end_frame_; }
    int tracklet_len() const { return tracklet_len_; }
    float conf() const { return conf_; }
    int cls() const { return cls_; }
    int det_ind() const { return det_ind_; }
    BotTrackState state() const { return state_; }
    bool is_activated() const { return is_activated_; }
    
    Eigen::Vector4f xyxy() const;
    Eigen::Vector4f xywh() const;
    Eigen::Vector4f tlwh() const;
    
    Eigen::VectorXf mean() const { return mean_; }
    Eigen::MatrixXf covariance() const { return covariance_; }
    Eigen::VectorXf curr_feat() const { return curr_feat_; }
    Eigen::VectorXf smooth_feat() const { return smooth_feat_; }
    
    void update_features(const Eigen::VectorXf& feat);
    
private:
    static int next_id();
    
    int id_ = 0;
    int frame_id_ = 0;
    int start_frame_ = 0;
    int end_frame_ = 0;
    int tracklet_len_ = 0;
    
    Eigen::Vector4f xywh_;
    float conf_ = 0.0f;
    int cls_ = 0;
    int det_ind_ = -1;
    
    Eigen::VectorXf mean_;
    Eigen::MatrixXf covariance_;
    
    std::shared_ptr<KalmanFilterXYWH> kalman_filter_;
    
    BotTrackState state_ = BotTrackState::New;
    bool is_activated_ = false;
    
    // Appearance features
    Eigen::VectorXf curr_feat_;
    Eigen::VectorXf smooth_feat_;
    float alpha_ = 0.9f;
    
    std::deque<Eigen::Vector4f> history_observations_;
    int max_obs_;
};

/**
 * @brief BoTSORT tracker - ByteTrack with CMC and ReID
 * 
 * Based on: "BoT-SORT: Robust Associations Multi-Pedestrian Tracking"
 * https://arxiv.org/abs/2206.14651
 */
class BotSort : public BaseTracker {
public:
    BotSort(
        const std::string& reid_weights = "",
        bool use_half = false,
        bool use_gpu = false,
        // BaseTracker parameters
        float det_thresh = 0.3f,
        int max_age = 30,
        int max_obs = 50,
        int min_hits = 3,
        float iou_threshold = 0.3f,
        bool per_class = false,
        int nr_classes = 80,
        const std::string& asso_func = "iou",
        bool is_obb = false,
        // BotSort specific
        float track_high_thresh = 0.5f,
        float track_low_thresh = 0.1f,
        float new_track_thresh = 0.6f,
        int track_buffer = 30,
        float match_thresh = 0.8f,
        float proximity_thresh = 0.5f,
        float appearance_thresh = 0.25f,
        const std::string& cmc_method = "ecc",
        int frame_rate = 30,
        bool fuse_first_associate = false,
        bool with_reid = true
    );
    
    ~BotSort() override = default;
    
    Eigen::MatrixXf update(
        const Eigen::MatrixXf& dets,
        const cv::Mat& img,
        const Eigen::MatrixXf& embs = Eigen::MatrixXf()
    ) override;
    
    void reset() override;
    
private:
    // Split detections by confidence
    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
    split_detections(const Eigen::MatrixXf& dets, const Eigen::MatrixXf& embs);
    
    // Create detection tracks
    std::vector<BotSTrack> create_detections(
        const Eigen::MatrixXf& dets_first,
        const Eigen::MatrixXf& features_high
    );
    
    // First association step
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    first_association(
        const Eigen::MatrixXf& dets,
        const Eigen::MatrixXf& dets_first,
        std::vector<BotSTrack*>& active_tracks,
        std::vector<BotSTrack*>& unconfirmed,
        const cv::Mat& img,
        std::vector<BotSTrack>& detections,
        std::vector<BotSTrack*>& activated_stracks,
        std::vector<BotSTrack*>& refind_stracks,
        std::vector<BotSTrack*>& strack_pool
    );
    
    // Second association step
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    second_association(
        const Eigen::MatrixXf& dets_second,
        std::vector<BotSTrack*>& activated_stracks,
        std::vector<BotSTrack*>& lost_stracks,
        std::vector<BotSTrack*>& refind_stracks,
        const std::vector<int>& u_track_first,
        std::vector<BotSTrack*>& strack_pool
    );
    
    // Handle unconfirmed tracks
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    handle_unconfirmed_tracks(
        const std::vector<int>& u_detection,
        std::vector<BotSTrack>& detections,
        std::vector<BotSTrack*>& activated_stracks,
        std::vector<BotSTrack*>& removed_stracks,
        std::vector<BotSTrack*>& unconfirmed
    );
    
    // Initialize new tracks
    void initialize_new_tracks(
        const std::vector<int>& u_detections,
        std::vector<BotSTrack*>& activated_stracks,
        std::vector<BotSTrack>& detections
    );
    
    // Update track states
    void update_track_states(std::vector<BotSTrack*>& removed_stracks);
    
    // Prepare output
    Eigen::MatrixXf prepare_output(
        std::vector<BotSTrack*>& activated_stracks,
        std::vector<BotSTrack*>& refind_stracks,
        std::vector<BotSTrack*>& lost_stracks,
        std::vector<BotSTrack*>& removed_stracks
    );
    
    // BotSort specific parameters
    float track_high_thresh_;
    float track_low_thresh_;
    float new_track_thresh_;
    int track_buffer_;
    float match_thresh_;
    float proximity_thresh_;
    float appearance_thresh_;
    bool fuse_first_associate_;
    bool with_reid_;
    
    int buffer_size_;
    int max_time_lost_;
    
    std::shared_ptr<KalmanFilterXYWH> kalman_filter_;
    std::unique_ptr<motcpp::motion::ECC> cmc_;
    std::unique_ptr<appearance::ONNXBackend> reid_model_;
    
    std::vector<BotSTrack> active_tracks_;
    std::vector<BotSTrack> lost_stracks_;
    std::vector<BotSTrack> removed_stracks_;
};

// Helper functions
std::vector<BotSTrack*> joint_stracks(
    std::vector<BotSTrack*>& tlista,
    std::vector<BotSTrack*>& tlistb
);

std::vector<BotSTrack*> sub_stracks(
    std::vector<BotSTrack*>& tlista,
    const std::vector<BotSTrack*>& tlistb
);

std::pair<std::vector<BotSTrack*>, std::vector<BotSTrack*>>
remove_duplicate_stracks(
    std::vector<BotSTrack*>& stracksa,
    std::vector<BotSTrack*>& stracksb
);

} // namespace motcpp::trackers
