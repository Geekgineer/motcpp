// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/tracker.hpp>
#include <motcpp/utils/matching.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace motcpp::trackers {

/**
 * Track states for UCMCTrack
 */
enum class UCMCTrackState {
    Tentative = 0,
    Confirmed = 1,
    Coasted = 2,
    Deleted = 3
};

/**
 * Kalman Filter for ground-plane tracking (x, vx, y, vy)
 * Used by UCMCTrack for position/velocity estimation
 */
class UCMCKalmanFilter {
public:
    UCMCKalmanFilter(int dim_x, int dim_z);
    
    void predict();
    void update(const Eigen::Vector2d& z, const Eigen::Matrix2d& R);
    
    Eigen::Vector4d getState() const { return x_; }
    Eigen::Matrix4d getCovariance() const { return P_; }
    
    // State transition and observation matrices
    Eigen::Matrix4d F_;  // State transition
    Eigen::Matrix<double, 2, 4> H_;  // Observation matrix
    Eigen::Matrix4d Q_;  // Process noise
    Eigen::Matrix4d P_;  // State covariance
    Eigen::Vector4d x_;  // State [x, vx, y, vy]
    
private:
    int dim_x_;
    int dim_z_;
};

/**
 * Camera mapper for ground-plane projection
 * Converts image coordinates (u, v) to ground-plane coordinates (x, y)
 */
class CameraMapper {
public:
    CameraMapper();
    CameraMapper(const std::vector<double>& Ki, const std::vector<double>& Ko);
    
    bool isValid() const { return valid_; }
    
    // Map bounding box to ground-plane coordinates
    void mapToGroundPlane(float bbox_cx, float bbox_bottom, float bbox_w, float bbox_h,
                          Eigen::Vector2d& y, Eigen::Matrix2d& R) const;
    
    // Fallback: estimate ground-plane position from image coordinates
    void mapToImageSpace(float bbox_cx, float bbox_bottom, float bbox_w, float bbox_h,
                         Eigen::Vector2d& y, Eigen::Matrix2d& R) const;
    
private:
    bool valid_;
    Eigen::Matrix3d A_;
    Eigen::Matrix3d InvA_;
    Eigen::Matrix<double, 3, 4> KiKo_;
    
    std::vector<double> uvError(float bbox_w, float bbox_h) const;
    void uv2xy(const Eigen::Vector2d& uv, const Eigen::Matrix2d& sigma_uv,
               Eigen::Vector2d& xy, Eigen::Matrix2d& sigma_xy) const;
};

/**
 * Single track representation for UCMCTrack
 */
class UCMCSingleTrack {
public:
    UCMCSingleTrack(const Eigen::Vector2d& y, const Eigen::Matrix2d& R,
                    double wx, double wy, double vmax, float w, float h,
                    double dt, int track_id);
    
    Eigen::Vector2d predict();
    void update(const Eigen::Vector2d& y, const Eigen::Matrix2d& R);
    
    // Compute Mahalanobis distance to a detection
    double distance(const Eigen::Vector2d& y, const Eigen::Matrix2d& R) const;
    
    // Getters
    int id() const { return id_; }
    int age() const { return age_; }
    int deathCount() const { return death_count_; }
    int birthCount() const { return birth_count_; }
    int detIdx() const { return det_idx_; }
    float width() const { return w_; }
    float height() const { return h_; }
    UCMCTrackState state() const { return state_; }
    Eigen::Vector4d getState() const { return kf_.getState(); }
    Eigen::Matrix4d getCovariance() const { return kf_.getCovariance(); }
    const Eigen::Matrix<double, 2, 4>& H() const { return kf_.H_; }
    
    // Setters
    void setDetIdx(int idx) { det_idx_ = idx; }
    void setDeathCount(int c) { death_count_ = c; }
    void setBirthCount(int c) { birth_count_ = c; }
    void setState(UCMCTrackState s) { state_ = s; }
    void setWidth(float w) { w_ = w; }
    void setHeight(float h) { h_ = h; }
    void incrementDeathCount() { death_count_++; }
    void incrementBirthCount() { birth_count_++; }
    void incrementAge() { age_++; }
    
private:
    int id_;
    int age_;
    int death_count_;
    int birth_count_;
    int det_idx_;
    float w_;
    float h_;
    UCMCTrackState state_;
    UCMCKalmanFilter kf_;
};

/**
 * UCMCTrack - Unified Confidence-based Multi-object tracker with Camera motion compensation
 * 
 * Paper: "UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation"
 * https://arxiv.org/abs/2312.08952
 * 
 * Key features:
 * - Ground-plane tracking via camera calibration (optional)
 * - Mahalanobis distance with covariance for association
 * - Two-stage association: high confidence then low confidence
 * - Track states: Tentative, Confirmed, Coasted, Deleted
 */
class UCMCTrack : public BaseTracker {
public:
    UCMCTrack(float det_thresh = 0.3f,
              int max_age = 30,
              int max_obs = 50,
              int min_hits = 3,
              float iou_threshold = 0.3f,
              bool per_class = false,
              int nr_classes = 80,
              const std::string& asso_func = "iou",
              bool is_obb = false,
              // UCMC-specific parameters
              double a1 = 100.0,           // Association threshold for high-confidence
              double a2 = 100.0,           // Association threshold for low-confidence
              double wx = 5.0,             // Process noise x
              double wy = 5.0,             // Process noise y
              double vmax = 10.0,          // Max velocity for initialization
              double dt = 1.0/30.0,        // Time step
              float high_score = 0.5f,     // High/low confidence threshold
              const std::vector<double>& Ki = {},  // Camera intrinsic matrix (3x4)
              const std::vector<double>& Ko = {}); // Camera extrinsic matrix (4x4)
    
    Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                          const cv::Mat& img,
                          const Eigen::MatrixXf& embs = Eigen::MatrixXf()) override;
    
    void reset() override;
    
private:
    // UCMC parameters
    double a1_;          // High-confidence association threshold
    double a2_;          // Low-confidence association threshold
    double wx_;          // Process noise x
    double wy_;          // Process noise y
    double vmax_;        // Max velocity
    double dt_;          // Time step
    float high_score_;   // High/low confidence split
    
    int frame_count_;
    int tracker_count_;
    
    std::vector<UCMCSingleTrack> trackers_;
    std::vector<int> confirmed_idx_;
    std::vector<int> coasted_idx_;
    std::vector<int> tentative_idx_;
    
    CameraMapper mapper_;
    
    // Internal structures for detection mapping
    struct MappedDetection {
        int original_idx;
        Eigen::Vector2d y;      // Ground-plane position
        Eigen::Matrix2d R;      // Measurement covariance
        float conf;
        int cls;
        float x1, y1, x2, y2;   // Original bbox
        float w, h;
    };
    
    // Association methods
    void dataAssociation(std::vector<MappedDetection>& dets);
    void associateTentative(std::vector<MappedDetection>& dets,
                           std::vector<int>& det_remain);
    void initTentative(std::vector<MappedDetection>& dets,
                      const std::vector<int>& det_remain);
    void deleteOldTrackers();
    void updateStatus();
};

} // namespace motcpp::trackers
