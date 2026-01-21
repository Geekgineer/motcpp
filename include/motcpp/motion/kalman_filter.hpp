// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace motcpp::motion {

/**
 * Base Kalman filter for tracking bounding boxes
 */
class BaseKalmanFilter {
public:
    explicit BaseKalmanFilter(int ndim);
    virtual ~BaseKalmanFilter() = default;
    
    /**
     * Initialize track from measurement
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> initiate(const Eigen::VectorXf& measurement);
    
    /**
     * Prediction step
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> predict(const Eigen::VectorXf& mean,
                                                         const Eigen::MatrixXf& covariance);
    
    /**
     * Project state to measurement space
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> project(const Eigen::VectorXf& mean,
                                                        const Eigen::MatrixXf& covariance,
                                                        float confidence = 0.0f) const;
    
    /**
     * Update step (correction)
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> update(const Eigen::VectorXf& mean,
                                                       const Eigen::MatrixXf& covariance,
                                                       const Eigen::VectorXf& measurement,
                                                       float confidence = 0.0f);
    
    /**
     * Batch prediction (vectorized)
     */
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> multi_predict(const Eigen::MatrixXf& mean,
                                                              const Eigen::MatrixXf& covariance);
    
    /**
     * Gating distance (Mahalanobis or Gaussian)
     */
    Eigen::VectorXf gating_distance(const Eigen::VectorXf& mean,
                                    const Eigen::MatrixXf& covariance,
                                    const Eigen::MatrixXf& measurements,
                                    bool only_position = false,
                                    const std::string& metric = "maha") const;
    
protected:
    int ndim_;
    float dt_;
    Eigen::MatrixXf motion_mat_;  // State transition matrix
    Eigen::MatrixXf update_mat_;  // Observation matrix
    
    float std_weight_position_;
    float std_weight_velocity_;
    
    // Pure virtual methods for subclasses
    virtual Eigen::VectorXf get_initial_covariance_std(const Eigen::VectorXf& measurement) = 0;
    virtual std::pair<Eigen::VectorXf, Eigen::VectorXf> get_process_noise_std(
        const Eigen::VectorXf& mean) = 0;
    virtual Eigen::VectorXf get_measurement_noise_std(const Eigen::VectorXf& mean,
                                                      float confidence) const = 0;
    virtual std::pair<Eigen::MatrixXf, Eigen::MatrixXf> get_multi_process_noise_std(
        const Eigen::MatrixXf& mean) = 0;
};

} // namespace motcpp::motion

