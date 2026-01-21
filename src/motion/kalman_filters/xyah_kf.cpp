// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/motion/kalman_filters/xyah_kf.hpp>
#include <cmath>

namespace motcpp::motion {

KalmanFilterXYAH::KalmanFilterXYAH()
    : BaseKalmanFilter(4)  // 4 dimensions: x, y, a (aspect ratio), h (height)
{
}

Eigen::VectorXf KalmanFilterXYAH::get_initial_covariance_std(
    const Eigen::VectorXf& measurement) {
    float h = measurement(3);  // height
    Eigen::VectorXf std(8);  // 4 positions + 4 velocities
    
    std(0) = 2.0f * std_weight_position_ * h;  // x
    std(1) = 2.0f * std_weight_position_ * h;  // y
    std(2) = 1e-2f;                            // a (aspect ratio)
    std(3) = 2.0f * std_weight_position_ * h; // h
    std(4) = 10.0f * std_weight_velocity_ * h; // vx
    std(5) = 10.0f * std_weight_velocity_ * h; // vy
    std(6) = 1e-5f;                            // va
    std(7) = 10.0f * std_weight_velocity_ * h; // vh
    
    return std;
}

std::pair<Eigen::VectorXf, Eigen::VectorXf> KalmanFilterXYAH::get_process_noise_std(
    const Eigen::VectorXf& mean) {
    float h = mean(3);  // height
    Eigen::VectorXf std_pos(4);
    Eigen::VectorXf std_vel(4);
    
    std_pos(0) = std_weight_position_ * h;
    std_pos(1) = std_weight_position_ * h;
    std_pos(2) = 1e-2f;
    std_pos(3) = std_weight_position_ * h;
    
    std_vel(0) = std_weight_velocity_ * h;
    std_vel(1) = std_weight_velocity_ * h;
    std_vel(2) = 1e-5f;
    std_vel(3) = std_weight_velocity_ * h;
    
    return {std_pos, std_vel};
}

Eigen::VectorXf KalmanFilterXYAH::get_measurement_noise_std(
    const Eigen::VectorXf& mean,
    float /* confidence */) const {
    float h = mean(3);
    Eigen::VectorXf std(4);
    
    std(0) = std_weight_position_ * h;
    std(1) = std_weight_position_ * h;
    std(2) = 1e-1f;
    std(3) = std_weight_position_ * h;
    
    return std;
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> KalmanFilterXYAH::get_multi_process_noise_std(
    const Eigen::MatrixXf& mean) {
    int n = mean.rows();
    Eigen::MatrixXf std_pos(n, 4);
    Eigen::MatrixXf std_vel(n, 4);
    
    Eigen::VectorXf heights = mean.col(3);
    
    std_pos.col(0) = std_weight_position_ * heights;
    std_pos.col(1) = std_weight_position_ * heights;
    std_pos.col(2) = Eigen::VectorXf::Constant(n, 1e-2f);
    std_pos.col(3) = std_weight_position_ * heights;
    
    std_vel.col(0) = std_weight_velocity_ * heights;
    std_vel.col(1) = std_weight_velocity_ * heights;
    std_vel.col(2) = Eigen::VectorXf::Constant(n, 1e-5f);
    std_vel.col(3) = std_weight_velocity_ * heights;
    
    return {std_pos, std_vel};
}

} // namespace motcpp::motion

