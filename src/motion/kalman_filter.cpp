// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/motion/kalman_filter.hpp>
#include <Eigen/Cholesky>
#include <cmath>

namespace motcpp::motion {

BaseKalmanFilter::BaseKalmanFilter(int ndim)
    : ndim_(ndim)
    , dt_(1.0f)
    , std_weight_position_(1.0f / 20.0f)
    , std_weight_velocity_(1.0f / 160.0f)
{
    // Initialize motion matrix: [I, dt*I; 0, I]
    motion_mat_ = Eigen::MatrixXf::Identity(2 * ndim_, 2 * ndim_);
    for (int i = 0; i < ndim_; ++i) {
        motion_mat_(i, ndim_ + i) = dt_;
    }
    
    // Initialize update matrix: [I, 0]
    update_mat_ = Eigen::MatrixXf::Zero(ndim_, 2 * ndim_);
    for (int i = 0; i < ndim_; ++i) {
        update_mat_(i, i) = 1.0f;
    }
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> BaseKalmanFilter::initiate(
    const Eigen::VectorXf& measurement) {
    Eigen::VectorXf mean_pos = measurement;
    Eigen::VectorXf mean_vel = Eigen::VectorXf::Zero(ndim_);
    
    Eigen::VectorXf mean(2 * ndim_);
    mean.head(ndim_) = mean_pos;
    mean.tail(ndim_) = mean_vel;
    
    Eigen::VectorXf std = get_initial_covariance_std(measurement);
    Eigen::MatrixXf covariance = std.array().square().matrix().asDiagonal();
    
    return {mean, covariance};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> BaseKalmanFilter::predict(
    const Eigen::VectorXf& mean,
    const Eigen::MatrixXf& covariance) {
    auto [std_pos, std_vel] = get_process_noise_std(mean);
    
    Eigen::VectorXf std_full(2 * ndim_);
    std_full.head(ndim_) = std_pos;
    std_full.tail(ndim_) = std_vel;
    Eigen::MatrixXf motion_cov = std_full.array().square().matrix().asDiagonal();
    
    Eigen::VectorXf new_mean = motion_mat_ * mean;
    Eigen::MatrixXf new_covariance = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;
    
    return {new_mean, new_covariance};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> BaseKalmanFilter::project(
    const Eigen::VectorXf& mean,
    const Eigen::MatrixXf& covariance,
    float confidence) const {
    Eigen::VectorXf std = get_measurement_noise_std(mean, confidence);
    
    // NSA Kalman: Rk = (1 - ck) * Rk
    std = std.array() * (1.0f - confidence);
    
    Eigen::MatrixXf innovation_cov = std.array().square().matrix().asDiagonal();
    
    Eigen::VectorXf projected_mean = update_mat_ * mean;
    Eigen::MatrixXf projected_cov = update_mat_ * covariance * update_mat_.transpose() + innovation_cov;
    
    return {projected_mean, projected_cov};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> BaseKalmanFilter::update(
    const Eigen::VectorXf& mean,
    const Eigen::MatrixXf& covariance,
    const Eigen::VectorXf& measurement,
    float confidence) {
    auto [projected_mean, projected_cov] = project(mean, covariance, confidence);
    
    // Cholesky decomposition for solving
    Eigen::LLT<Eigen::MatrixXf> chol(projected_cov);
    if (chol.info() != Eigen::Success) {
        // Fallback: use pseudo-inverse if Cholesky fails
        Eigen::MatrixXf kalman_gain = covariance * update_mat_.transpose() * 
                                      projected_cov.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::VectorXf innovation = measurement - projected_mean;
        Eigen::VectorXf new_mean = mean + kalman_gain * innovation;
        Eigen::MatrixXf new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();
        return {new_mean, new_covariance};
    }
    
    // Solve: kalman_gain * projected_cov = covariance * H^T
    // So: kalman_gain = (covariance * H^T) * projected_cov^-1
    // projected_cov is (ndim_ x ndim_), covariance * H^T is (2*ndim_ x ndim_)
    Eigen::MatrixXf cov_h_t = covariance * update_mat_.transpose();  // (2*ndim_ x ndim_)
    Eigen::MatrixXf kalman_gain(2 * ndim_, ndim_);
    // Solve each row: kalman_gain[i] * projected_cov = cov_h_t[i]
    // This means: kalman_gain[i] = cov_h_t[i] * projected_cov^-1
    for (int i = 0; i < 2 * ndim_; ++i) {
        kalman_gain.row(i) = chol.solve(cov_h_t.row(i).transpose()).transpose();
    }
    
    Eigen::VectorXf innovation = measurement - projected_mean;
    Eigen::VectorXf new_mean = mean + kalman_gain * innovation;
    Eigen::MatrixXf new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();
    
    return {new_mean, new_covariance};
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> BaseKalmanFilter::multi_predict(
    const Eigen::MatrixXf& mean,
    const Eigen::MatrixXf& covariance) {
    int n = mean.rows();
    auto [std_pos, std_vel] = get_multi_process_noise_std(mean);
    
    Eigen::MatrixXf new_mean = mean * motion_mat_.transpose();
    
    // covariance is (n, 2*ndim_, 2*ndim_) stored as (n, 2*ndim_*2*ndim_)
    Eigen::MatrixXf new_covariance(n, 2 * ndim_ * 2 * ndim_);
    
    for (int i = 0; i < n; ++i) {
        // Extract covariance matrix for this track
        Eigen::Map<const Eigen::MatrixXf> P(
            covariance.data() + i * 2 * ndim_ * 2 * ndim_, 2 * ndim_, 2 * ndim_);
        
        // Build process noise covariance Q
        Eigen::VectorXf std_full(2 * ndim_);
        std_full.head(ndim_) = std_pos.row(i).transpose();
        std_full.tail(ndim_) = std_vel.row(i).transpose();
        Eigen::MatrixXf Q = std_full.array().square().matrix().asDiagonal();
        
        // Predict: P_new = F * P * F^T + Q
        Eigen::MatrixXf new_P = motion_mat_ * P * motion_mat_.transpose() + Q;
        
        // Store back
        Eigen::Map<Eigen::RowVectorXf> cov_row(
            new_covariance.data() + i * 2 * ndim_ * 2 * ndim_, 2 * ndim_ * 2 * ndim_);
        cov_row = Eigen::Map<const Eigen::RowVectorXf>(new_P.data(), 2 * ndim_ * 2 * ndim_);
    }
    
    return {new_mean, new_covariance};
}

Eigen::VectorXf BaseKalmanFilter::gating_distance(
    const Eigen::VectorXf& mean,
    const Eigen::MatrixXf& covariance,
    const Eigen::MatrixXf& measurements,
    bool only_position,
    const std::string& metric) const {
    auto [projected_mean, projected_cov] = project(mean, covariance);
    
    int dim = only_position ? 2 : ndim_;
    Eigen::VectorXf mean_sub = projected_mean.head(dim);
    Eigen::MatrixXf cov_sub = projected_cov.topLeftCorner(dim, dim);
    Eigen::MatrixXf measurements_sub = measurements.leftCols(dim);
    
    Eigen::MatrixXf d = measurements_sub.rowwise() - mean_sub.transpose();
    
    if (metric == "gaussian") {
        return d.rowwise().squaredNorm();
    } else if (metric == "maha") {
        Eigen::LLT<Eigen::MatrixXf> chol(cov_sub);
        if (chol.info() != Eigen::Success) {
            // Fallback
            return d.rowwise().squaredNorm();
        }
        Eigen::MatrixXf z = chol.solve(d.transpose()).transpose();
        return z.rowwise().squaredNorm();
    } else {
        throw std::invalid_argument("Invalid metric: " + metric);
    }
}

} // namespace motcpp::motion

