// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/motion/cmc/ecc.hpp>
#include <motcpp/appearance/onnx_backend.hpp>
#include <deque>
#include <vector>
#include <memory>
#include <unordered_map>

namespace motcpp::trackers {

/**
 * @brief Custom 9D Kalman Filter for HybridSORT
 * State: [u, v, s, c, r, du, dv, ds, dc]
 * Measurement: [u, v, s, c, r]
 */
class HybridKalmanFilter {
public:
    HybridKalmanFilter();
    
    void init(const Eigen::VectorXf& z);
    void predict();
    void update(const Eigen::VectorXf& z);
    void camera_update(const Eigen::Matrix3f& transform);
    
    Eigen::VectorXf get_state() const;
    Eigen::VectorXf x;  // State vector [u, v, s, c, r, du, dv, ds, dc]
    Eigen::MatrixXf P;  // Covariance matrix
    
private:
    Eigen::MatrixXf F;  // State transition matrix (9x9)
    Eigen::MatrixXf H;  // Measurement matrix (5x9)
    Eigen::MatrixXf Q;  // Process noise
    Eigen::MatrixXf R;  // Measurement noise
};

/**
 * @brief Single track for HybridSORT
 */
class HybridKalmanBoxTracker {
public:
    static int next_id_;
    
    HybridKalmanBoxTracker(
        const Eigen::VectorXf& bbox,
        const Eigen::VectorXf& temp_feat,
        int delta_t = 3,
        bool use_custom_kf = true,
        int longterm_bank_length = 30,
        float alpha = 0.9f,
        bool adapfs = false,
        float track_thresh = 0.5f,
        int cls = 0,
        int det_ind = -1
    );
    
    void predict();
    void update(const Eigen::VectorXf& bbox, const Eigen::VectorXf& id_feature, 
                bool update_feature = true, int cls = -1, int det_ind = -1);
    void camera_update(const Eigen::Matrix3f& warp_matrix);
    void update_features(const Eigen::VectorXf& feat, float score = -1.0f);
    
    Eigen::Vector4f get_bbox() const;
    float get_kalman_score() const;
    float get_simple_score() const;
    
    Eigen::VectorXf convert_bbox_to_z(const Eigen::VectorXf& bbox) const;
    Eigen::Vector4f convert_x_to_bbox(const Eigen::VectorXf& x) const;
    
    int id() const { return id_; }
    float conf() const { return conf_; }
    int cls() const { return cls_; }
    int det_ind() const { return det_ind_; }
    int age() const { return age_; }
    int hits() const { return hits_; }
    int hit_streak() const { return hit_streak_; }
    int time_since_update() const { return time_since_update_; }
    
    Eigen::VectorXf last_observation() const { return last_observation_; }
    Eigen::VectorXf smooth_feat() const { return smooth_feat_; }
    const std::deque<Eigen::VectorXf>& features() const { return features_; }
    Eigen::VectorXf k_previous_obs(int cur_age, int k) const;
    
    Eigen::Vector2f velocity_lt, velocity_rt, velocity_lb, velocity_rb;
    std::unordered_map<int, Eigen::VectorXf> observations;
    
    int longterm_bank_length_;
    
private:
    static int next_id();
    
    int id_;
    int age_;
    int hits_;
    int hit_streak_;
    int time_since_update_;
    float conf_;
    float confidence_pre_;
    int cls_;
    int det_ind_;
    int delta_t_;
    float track_thresh_;
    float alpha_;
    bool adapfs_;
    
    HybridKalmanFilter kf_;
    Eigen::VectorXf last_observation_;
    std::deque<Eigen::VectorXf> history_observations_;
    std::deque<Eigen::VectorXf> features_;
    Eigen::VectorXf smooth_feat_;
    Eigen::VectorXf curr_feat_;
    
    std::vector<Eigen::Vector4f> history_;
};

/**
 * @brief HybridSORT tracker implementation
 * 
 * Hybrid SORT + ReID with ECC CMC
 */
class HybridSort : public BaseTracker {
public:
    HybridSort(
        const std::string& reid_weights = "",
        bool use_half = false,
        bool use_gpu = false,
        // BaseTracker parameters
        float det_thresh = 0.7f,
        int max_age = 30,
        int max_obs = 50,
        int min_hits = 3,
        float iou_threshold = 0.15f,
        bool per_class = false,
        int nr_classes = 80,
        const std::string& asso_func = "hmiou",
        bool is_obb = false,
        // HybridSORT-specific parameters
        float low_thresh = 0.1f,
        int delta_t = 3,
        float inertia = 0.05f,
        bool use_byte = true,
        bool use_custom_kf = true,
        int longterm_bank_length = 30,
        float alpha = 0.9f,
        bool adapfs = false,
        float track_thresh = 0.5f,
        float EG_weight_high_score = 4.6f,
        float EG_weight_low_score = 1.3f,
        bool TCM_first_step = true,
        bool TCM_byte_step = true,
        float TCM_byte_step_weight = 1.0f,
        float high_score_matching_thresh = 0.7f,
        bool with_longterm_reid = true,
        float longterm_reid_weight = 0.0f,
        bool with_longterm_reid_correction = true,
        float longterm_reid_correction_thresh = 0.4f,
        float longterm_reid_correction_thresh_low = 0.4f,
        const std::string& cmc_method = "ecc",
        bool with_reid = true
    );
    
    ~HybridSort() override = default;
    
    Eigen::MatrixXf update(
        const Eigen::MatrixXf& dets,
        const cv::Mat& img,
        const Eigen::MatrixXf& embs = Eigen::MatrixXf()
    ) override;
    
    void reset() override;
    
private:
    // Helper functions
    Eigen::VectorXf convert_bbox_to_z(const Eigen::VectorXf& bbox) const;
    Eigen::Vector4f convert_x_to_bbox(const Eigen::VectorXf& x) const;
    Eigen::Vector2f speed_direction_lt(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const;
    Eigen::Vector2f speed_direction_rt(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const;
    Eigen::Vector2f speed_direction_lb(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const;
    Eigen::Vector2f speed_direction_rb(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const;
    Eigen::VectorXf k_previous_obs(const std::unordered_map<int, Eigen::VectorXf>& observations, int cur_age, int k) const;
    
    // Association functions
    Eigen::MatrixXf iou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const;
    Eigen::MatrixXf hmiou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const;
    Eigen::MatrixXf giou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const;
    Eigen::MatrixXf ciou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const;
    Eigen::MatrixXf diou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const;
    Eigen::MatrixXf ct_dist_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const;
    Eigen::MatrixXf cal_score_dif_batch_two_score(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const;
    
    // Association
    std::tuple<Eigen::MatrixXi, Eigen::VectorXi, Eigen::VectorXi>
    associate_4_points_with_score(
        const Eigen::MatrixXf& detections,
        const Eigen::MatrixXf& trackers,
        float iou_threshold,
        const Eigen::MatrixXf& velocities_lt,
        const Eigen::MatrixXf& velocities_rt,
        const Eigen::MatrixXf& velocities_lb,
        const Eigen::MatrixXf& velocities_rb,
        const Eigen::MatrixXf& k_observations,
        float inertia,
        const std::string& asso_func
    ) const;
    
    std::tuple<Eigen::MatrixXi, Eigen::VectorXi, Eigen::VectorXi>
    associate_4_points_with_score_with_reid(
        const Eigen::MatrixXf& detections,
        const Eigen::MatrixXf& trackers,
        float iou_threshold,
        const Eigen::MatrixXf& velocities_lt,
        const Eigen::MatrixXf& velocities_rt,
        const Eigen::MatrixXf& velocities_lb,
        const Eigen::MatrixXf& velocities_rb,
        const Eigen::MatrixXf& k_observations,
        float inertia,
        const std::string& asso_func,
        const Eigen::MatrixXf& emb_cost,
        const std::pair<float, float>& weights,
        float thresh,
        const Eigen::MatrixXf& long_emb_dists = Eigen::MatrixXf(),
        bool with_longterm_reid = false,
        float longterm_reid_weight = 0.0f,
        bool with_longterm_reid_correction = false,
        float longterm_reid_correction_thresh = 0.4f,
        const std::string& dataset = ""
    ) const;
    
    // HybridSORT-specific parameters
    float low_thresh_;
    int delta_t_;
    float inertia_;
    bool use_byte_;
    bool use_custom_kf_;
    int longterm_bank_length_;
    float alpha_;
    bool adapfs_;
    float track_thresh_;
    float EG_weight_high_score_;
    float EG_weight_low_score_;
    bool TCM_first_step_;
    bool TCM_byte_step_;
    float TCM_byte_step_weight_;
    float high_score_matching_thresh_;
    bool with_longterm_reid_;
    float longterm_reid_weight_;
    bool with_longterm_reid_correction_;
    float longterm_reid_correction_thresh_;
    float longterm_reid_correction_thresh_low_;
    bool with_reid_;
    std::string cmc_method_;
    
    std::unique_ptr<motcpp::motion::ECC> cmc_;
    std::unique_ptr<appearance::ONNXBackend> reid_model_;
    
    std::vector<HybridKalmanBoxTracker> active_tracks_;
};

} // namespace motcpp::trackers
