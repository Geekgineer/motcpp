// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/motion/kalman_filter.hpp>

namespace motcpp::motion {

/**
 * Kalman filter for tracking with state space: (x, y, aspect_ratio, height) + velocities
 */
class KalmanFilterXYAH : public BaseKalmanFilter {
public:
    KalmanFilterXYAH();
    
protected:
    Eigen::VectorXf get_initial_covariance_std(const Eigen::VectorXf& measurement) override;
    std::pair<Eigen::VectorXf, Eigen::VectorXf> get_process_noise_std(
        const Eigen::VectorXf& mean) override;
    Eigen::VectorXf get_measurement_noise_std(const Eigen::VectorXf& mean,
                                              float confidence) const override;
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> get_multi_process_noise_std(
        const Eigen::MatrixXf& mean) override;
};

} // namespace motcpp::motion

