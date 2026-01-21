// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/motion/cmc/ecc.hpp>
#include <motcpp/appearance/onnx_backend.hpp>
#include <deque>
#include <vector>
#include <memory>

namespace motcpp::trackers {

/**
 * @brief Kalman filter for BoostTrack
 * State: [x, y, h, r, vx, vy, vh, vr] (8D)
 * Measurement: [x, y, h, r] (4D) where r = w/h
 */
class BoostKalmanFilter {
public:
    BoostKalmanFilter(const Eigen::Vector4f& z);
    
    void predict();
    void update(const Eigen::Vector4f& z);
    void camera_update(const Eigen::Matrix3f& transform);
    
    Eigen::Vector4f get_state() const;
    float get_confidence(float coef = 0.9f) const;
    
    Eigen::VectorXf x;  // State vector [x, y, h, r, vx, vy, vh, vr]
    Eigen::MatrixXf covariance;
    
    int time_since_update = 0;
    int age = 0;
    int hit_streak = 0;
    
private:
    Eigen::MatrixXf motion_mat_;  // 8x8
    Eigen::MatrixXf update_mat_;  // 4x8
    Eigen::MatrixXf process_noise_;
    Eigen::MatrixXf measurement_noise_;
};

/**
 * @brief Single track for BoostTrack
 */
class BoostTrack {
public:
    static int next_id_;
    
    BoostTrack(const Eigen::VectorXf& det, int max_obs, const Eigen::VectorXf& emb = Eigen::VectorXf());
    
    void predict();
    void update(const Eigen::VectorXf& det);
    void update_emb(const Eigen::VectorXf& emb, float alpha = 0.9f);
    void camera_update(const Eigen::Matrix3f& transform);
    
    Eigen::Vector4f get_state() const;
    float get_confidence() const;
    Eigen::VectorXf get_emb() const { return emb_; }
    
    int id() const { return id_; }
    float conf() const { return conf_; }
    int cls() const { return cls_; }
    int det_ind() const { return det_ind_; }
    int time_since_update() const { return kf_.time_since_update; }
    int hit_streak() const { return kf_.hit_streak; }
    
    // Access to Kalman filter for Mahalanobis distance computation
    const BoostKalmanFilter& kalman_filter() const { return kf_; }
    
private:
    static int next_id();
    
    int id_;
    float conf_;
    int cls_;
    int det_ind_;
    
    BoostKalmanFilter kf_;  // Must be initialized before max_obs_
    Eigen::VectorXf emb_;
    
    std::deque<Eigen::Vector4f> history_observations_;
    int max_obs_;
};

/**
 * @brief BoostTrack tracker implementation
 * 
 * Based on: "Boosting Multi-Object Tracking"
 */
class BoostTrackTracker : public BaseTracker {
public:
    BoostTrackTracker(
        const std::string& reid_weights = "",
        bool use_half = false,
        bool use_gpu = false,
        // BaseTracker parameters
        float det_thresh = 0.6f,
        int max_age = 60,
        int max_obs = 50,
        int min_hits = 3,
        float iou_threshold = 0.3f,
        bool per_class = false,
        int nr_classes = 80,
        const std::string& asso_func = "iou",
        bool is_obb = false,
        // BoostTrack-specific parameters
        bool use_ecc = true,
        int min_box_area = 10,
        float aspect_ratio_thresh = 1.6f,
        const std::string& cmc_method = "ecc",
        float lambda_iou = 0.5f,
        float lambda_mhd = 0.25f,
        float lambda_shape = 0.25f,
        bool use_dlo_boost = true,
        bool use_duo_boost = true,
        float dlo_boost_coef = 0.65f,
        bool s_sim_corr = false,
        bool use_rich_s = false,
        bool use_sb = false,
        bool use_vt = false,
        bool with_reid = false
    );
    
    ~BoostTrackTracker() override = default;
    
    Eigen::MatrixXf update(
        const Eigen::MatrixXf& dets,
        const cv::Mat& img,
        const Eigen::MatrixXf& embs = Eigen::MatrixXf()
    ) override;
    
    void reset() override;
    
private:
    // Helper functions
    Eigen::Vector4f convert_bbox_to_z(const Eigen::Vector4f& bbox) const;
    Eigen::Vector4f convert_x_to_bbox(const Eigen::VectorXf& x) const;
    
    Eigen::MatrixXf get_iou_matrix(const Eigen::MatrixXf& detections, bool buffered = false) const;
    Eigen::MatrixXf get_mh_dist_matrix(const Eigen::MatrixXf& detections, int n_dims = 4) const;
    
    Eigen::MatrixXf dlo_confidence_boost(const Eigen::MatrixXf& detections);
    Eigen::MatrixXf duo_confidence_boost(const Eigen::MatrixXf& detections);
    Eigen::MatrixXf filter_outputs(const Eigen::MatrixXf& outputs) const;
    
    // BoostTrack-specific parameters
    bool use_ecc_;
    int min_box_area_;
    float aspect_ratio_thresh_;
    std::string cmc_method_;
    float lambda_iou_;
    float lambda_mhd_;
    float lambda_shape_;
    bool use_dlo_boost_;
    bool use_duo_boost_;
    float dlo_boost_coef_;
    bool s_sim_corr_;
    bool use_rich_s_;
    bool use_sb_;
    bool use_vt_;
    bool with_reid_;
    
    std::unique_ptr<motcpp::motion::ECC> cmc_;
    std::unique_ptr<appearance::ONNXBackend> reid_model_;
    
    std::vector<BoostTrack> trackers_;
    std::vector<BoostTrack*> active_tracks_;
};

} // namespace motcpp::trackers
