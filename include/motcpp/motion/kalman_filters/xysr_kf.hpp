// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/motion/kalman_filter.hpp>
#include <Eigen/Dense>
#include <deque>

namespace motcpp::motion {

/**
 * Kalman filter for tracking with state space: (x, y, scale, ratio) + velocities
 * State: [x, y, s, r, vx, vy, vs] (7D)
 * Observation: [x, y, s, r] (4D)
 * 
 * This is used by OCSort tracker
 */
class KalmanFilterXYSR {
public:
    KalmanFilterXYSR(int dim_x = 7, int dim_z = 4, int max_obs = 50);
    
    // State transition matrix F (7x7)
    Eigen::MatrixXf F;
    
    // Measurement matrix H (4x7)
    Eigen::MatrixXf H;
    
    // State vector x (7x1)
    Eigen::VectorXf x;
    
    // Covariance matrix P (7x7)
    Eigen::MatrixXf P;
    
    // Process noise Q (7x7)
    Eigen::MatrixXf Q;
    
    // Measurement noise R (4x4)
    Eigen::MatrixXf R;
    
    // Predict next state
    void predict();
    
    // Update with measurement
    void update(const Eigen::VectorXf& z);
    
    // Get current state
    Eigen::VectorXf get_state() const { return x; }
    
    // Apply affine correction for camera motion compensation
    // m: 2x2 rotation/scaling matrix, t: 2x1 translation vector
    void apply_affine_correction(const Eigen::Matrix2f& m, const Eigen::Vector2f& t);
    
    // Get observation history
    std::deque<Eigen::VectorXf> history_obs;
    
private:
    int dim_x_;
    int dim_z_;
    int max_obs_;
    
    // Kalman gain
    Eigen::MatrixXf K;
    
    // Innovation (residual)
    Eigen::VectorXf y;
    
    // System uncertainty
    Eigen::MatrixXf S;
    
    // Identity matrix
    Eigen::MatrixXf I_;
};

} // namespace motcpp::motion

