// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <Eigen/Dense>
#include <tuple>

namespace motcpp {

/**
 * @brief Kalman Filter for bounding boxes in XYWH (center x, center y, width, height) format
 * 
 * State vector: [x, y, w, h, vx, vy, vw, vh]
 * Measurement vector: [x, y, w, h]
 */
class KalmanFilterXYWH {
public:
    KalmanFilterXYWH() {
        // Initialize motion and observation matrices
        motion_mat_ = Eigen::MatrixXf::Identity(8, 8);
        for (int i = 0; i < 4; ++i) {
            motion_mat_(i, i + 4) = 1.0f;  // Position is affected by velocity
        }
        
        update_mat_ = Eigen::MatrixXf::Zero(4, 8);
        for (int i = 0; i < 4; ++i) {
            update_mat_(i, i) = 1.0f;  // We observe positions only
        }
        
        // Weight for measurement uncertainty
        std_weight_position_ = 1.0f / 20.0f;
        std_weight_velocity_ = 1.0f / 160.0f;
    }
    
    /**
     * @brief Create new track from measurement
     * @param measurement [cx, cy, w, h]
     * @return tuple of mean and covariance
     */
    std::tuple<Eigen::VectorXf, Eigen::MatrixXf> initiate(const Eigen::Vector4f& measurement) {
        Eigen::VectorXf mean = Eigen::VectorXf::Zero(8);
        mean.head<4>() = measurement;  // Position
        // Velocity initialized to zero
        
        // Initialize covariance
        float h = measurement(3);  // height
        Eigen::VectorXf std(8);
        std << 2.0f * std_weight_position_ * h,
               2.0f * std_weight_position_ * h,
               2.0f * std_weight_position_ * h,
               2.0f * std_weight_position_ * h,
               10.0f * std_weight_velocity_ * h,
               10.0f * std_weight_velocity_ * h,
               10.0f * std_weight_velocity_ * h,
               10.0f * std_weight_velocity_ * h;
        
        Eigen::MatrixXf covariance = Eigen::MatrixXf::Zero(8, 8);
        covariance.diagonal() = std.array().square();
        
        return {mean, covariance};
    }
    
    /**
     * @brief Predict next state
     * @param mean Current state mean
     * @param covariance Current state covariance
     * @return tuple of predicted mean and covariance
     */
    std::tuple<Eigen::VectorXf, Eigen::MatrixXf> predict(
        const Eigen::VectorXf& mean,
        const Eigen::MatrixXf& covariance
    ) {
        // Process noise
        float h = mean(3);  // height
        Eigen::VectorXf std(8);
        std << std_weight_position_ * h,
               std_weight_position_ * h,
               std_weight_position_ * h,
               std_weight_position_ * h,
               std_weight_velocity_ * h,
               std_weight_velocity_ * h,
               std_weight_velocity_ * h,
               std_weight_velocity_ * h;
        
        Eigen::MatrixXf motion_cov = Eigen::MatrixXf::Zero(8, 8);
        motion_cov.diagonal() = std.array().square();
        
        // Predict
        Eigen::VectorXf mean_pred = motion_mat_ * mean;
        Eigen::MatrixXf covariance_pred = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;
        
        return {mean_pred, covariance_pred};
    }
    
    /**
     * @brief Update state with measurement
     * @param mean Current state mean
     * @param covariance Current state covariance
     * @param measurement [cx, cy, w, h]
     * @return tuple of updated mean and covariance
     */
    std::tuple<Eigen::VectorXf, Eigen::MatrixXf> update(
        const Eigen::VectorXf& mean,
        const Eigen::MatrixXf& covariance,
        const Eigen::Vector4f& measurement
    ) {
        // Measurement noise
        float h = mean(3);  // height
        Eigen::VectorXf std(4);
        std << std_weight_position_ * h,
               std_weight_position_ * h,
               std_weight_position_ * h,
               std_weight_position_ * h;
        
        Eigen::MatrixXf innovation_cov = Eigen::MatrixXf::Zero(4, 4);
        innovation_cov.diagonal() = std.array().square();
        
        // Project state to measurement space
        Eigen::VectorXf projected_mean = update_mat_ * mean;
        Eigen::MatrixXf projected_cov = update_mat_ * covariance * update_mat_.transpose();
        
        // Kalman gain
        Eigen::MatrixXf S = projected_cov + innovation_cov;
        Eigen::MatrixXf K = covariance * update_mat_.transpose() * S.inverse();
        
        // Innovation
        Eigen::VectorXf innovation = measurement - projected_mean;
        
        // Update
        Eigen::VectorXf mean_updated = mean + K * innovation;
        Eigen::MatrixXf covariance_updated = covariance - K * S * K.transpose();
        
        return {mean_updated, covariance_updated};
    }
    
    /**
     * @brief Compute gating distance for data association
     */
    Eigen::VectorXf gating_distance(
        const Eigen::VectorXf& mean,
        const Eigen::MatrixXf& covariance,
        const Eigen::MatrixXf& measurements,
        bool only_position = false
    ) {
        int n = measurements.rows();
        Eigen::VectorXf distances(n);
        
        // Project mean and covariance
        Eigen::VectorXf projected_mean = update_mat_ * mean;
        Eigen::MatrixXf projected_cov = update_mat_ * covariance * update_mat_.transpose();
        
        // Add measurement noise
        float h = mean(3);
        Eigen::VectorXf std(4);
        std << std_weight_position_ * h,
               std_weight_position_ * h,
               std_weight_position_ * h,
               std_weight_position_ * h;
        Eigen::MatrixXf innovation_cov = Eigen::MatrixXf::Zero(4, 4);
        innovation_cov.diagonal() = std.array().square();
        
        Eigen::MatrixXf S = projected_cov + innovation_cov;
        Eigen::MatrixXf S_inv = S.inverse();
        
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXf diff = measurements.row(i).transpose() - projected_mean;
            if (only_position) {
                diff = diff.head<2>();
                distances(i) = diff.transpose() * S_inv.topLeftCorner<2, 2>() * diff;
            } else {
                distances(i) = diff.transpose() * S_inv * diff;
            }
        }
        
        return distances;
    }
    
private:
    Eigen::MatrixXf motion_mat_;   // State transition matrix (8x8)
    Eigen::MatrixXf update_mat_;   // Observation matrix (4x8)
    
    float std_weight_position_;
    float std_weight_velocity_;
};

} // namespace motcpp
