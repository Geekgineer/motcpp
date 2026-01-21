// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/motion/kalman_filters/xysr_kf.hpp>
#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/ops.hpp>
#include <motcpp/utils/iou.hpp>
#include <motcpp/appearance/reid_backend.hpp>
#include <motcpp/motion/cmc/sof.hpp>
#include <deque>
#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Dense>

namespace motcpp::trackers {

/**
 * Extended KalmanBoxTracker for DeepOCSort with embedding support
 */
class DeepOCSortKalmanBoxTracker {
public:
    static int next_id() {
        static int count = 0;
        return ++count;
    }
    
    static void clear_count() {
        // Reset handled by static initialization
    }
    
    DeepOCSortKalmanBoxTracker(const Eigen::VectorXf& bbox, int cls, int det_ind,
                                const Eigen::VectorXf& emb = Eigen::VectorXf(),
                                int delta_t = 3, int max_obs = 50,
                                float Q_xy_scaling = 0.01f, float Q_s_scaling = 0.0001f);
    
    void update(const Eigen::VectorXf& bbox, int cls, int det_ind);
    void update_emb(const Eigen::VectorXf& emb, float alpha);
    Eigen::Vector4f predict();
    Eigen::Vector4f get_state() const;
    
    // Get observation from k frames ago
    Eigen::VectorXf k_previous_obs(int k) const;
    
    // Apply affine correction for CMC
    void apply_affine_correction(const Eigen::Matrix2f& m, const Eigen::Vector2f& t);
    
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
    Eigen::VectorXf get_emb() const { return emb_; }
    bool frozen() const { return frozen_; }
    
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
    
    // Embedding support
    Eigen::VectorXf emb_;  // Normalized embedding vector
    bool frozen_;  // True when track is not updated
};

/**
 * DeepOCSort tracker implementation with ReID and CMC
 */
class DeepOCSort : public BaseTracker {
public:
    DeepOCSort(const std::string& reid_weights,
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
                int delta_t = 3,
                float inertia = 0.2f,
                float w_association_emb = 0.5f,
                float alpha_fixed_emb = 0.95f,
                float aw_param = 0.5f,
                bool embedding_off = false,
                bool cmc_off = false,
                bool aw_off = false,
                float Q_xy_scaling = 0.01f,
                float Q_s_scaling = 0.0001f);
    
    ~DeepOCSort() override;
    
    Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                          const cv::Mat& img,
                          const Eigen::MatrixXf& embs = Eigen::MatrixXf()) override;
    
    void reset() override;
    
private:
    // DeepOCSort-specific parameters
    int delta_t_;
    float inertia_;
    float w_association_emb_;
    float alpha_fixed_emb_;
    float aw_param_;
    bool embedding_off_;
    bool cmc_off_;
    bool aw_off_;
    float Q_xy_scaling_;
    float Q_s_scaling_;
    
    // ReID backend
    std::unique_ptr<appearance::ReIDBackend> reid_backend_;
    
    // CMC (Camera Motion Compensation)
    std::unique_ptr<motion::SOF> cmc_;
    
    std::vector<DeepOCSortKalmanBoxTracker> active_tracks_;
    
    // Pre-allocated buffers for zero-allocation hot path
    mutable Eigen::MatrixXf cost_matrix_buffer_;
    mutable Eigen::MatrixXf track_xyxy_buffer_;
    mutable Eigen::MatrixXf det_xyxy_buffer_;
    mutable Eigen::VectorXf det_confs_buffer_;
    mutable Eigen::MatrixXf velocity_buffer_;
    mutable Eigen::MatrixXf k_obs_buffer_;
    mutable Eigen::MatrixXf dets_embs_buffer_;
    mutable Eigen::MatrixXf trk_embs_buffer_;
    mutable Eigen::MatrixXf emb_cost_buffer_;
    mutable Eigen::MatrixXf final_cost_buffer_;
    
    // Helper functions
    Eigen::Vector2f speed_direction(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const;
    Eigen::Vector4f convert_x_to_bbox(const Eigen::VectorXf& x) const;
};

} // namespace motcpp::trackers

