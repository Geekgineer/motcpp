// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/motion/kalman_filters/xysr_kf.hpp>
#include <Eigen/Cholesky>
#include <cmath>

namespace motcpp::motion {

KalmanFilterXYSR::KalmanFilterXYSR(int dim_x, int dim_z, int max_obs)
    : F(dim_x, dim_x)
    , H(dim_z, dim_x)
    , x(dim_x)
    , P(dim_x, dim_x)
    , Q(dim_x, dim_x)
    , R(dim_z, dim_z)
    , K(dim_x, dim_z)
    , y(dim_z)
    , S(dim_z, dim_z)
    , dim_x_(dim_x)
    , dim_z_(dim_z)
    , max_obs_(max_obs)
    , I_(dim_x, dim_x)
{
    // Initialize F matrix: constant velocity model
    // [1, 0, 0, 0, 1, 0, 0]  // x = x + vx
    // [0, 1, 0, 0, 0, 1, 0]  // y = y + vy
    // [0, 0, 1, 0, 0, 0, 1]  // s = s + vs
    // [0, 0, 0, 1, 0, 0, 0]  // r = r
    // [0, 0, 0, 0, 1, 0, 0]  // vx = vx
    // [0, 0, 0, 0, 0, 1, 0]  // vy = vy
    // [0, 0, 0, 0, 0, 0, 1]  // vs = vs
    F.setIdentity();
    F(0, 4) = 1.0f;  // x += vx
    F(1, 5) = 1.0f;  // y += vy
    F(2, 6) = 1.0f;  // s += vs
    
    // Initialize H matrix: observe position, scale, ratio
    // [1, 0, 0, 0, 0, 0, 0]  // observe x
    // [0, 1, 0, 0, 0, 0, 0]  // observe y
    // [0, 0, 1, 0, 0, 0, 0]  // observe s
    // [0, 0, 0, 1, 0, 0, 0]  // observe r
    H.setZero();
    H(0, 0) = 1.0f;
    H(1, 1) = 1.0f;
    H(2, 2) = 1.0f;
    H(3, 3) = 1.0f;
    
    // Initialize state
    x.setZero();
    
    // Initialize covariance P (high uncertainty for velocities)
    P.setIdentity();
    P *= 10.0f;
    P.block(4, 4, 3, 3) *= 100.0f;  // High uncertainty for velocities
    
    // Initialize process noise Q
    Q.setIdentity();
    Q(4, 4) = 0.01f;  // Q_xy_scaling will be applied
    Q(5, 5) = 0.01f;
    Q(6, 6) = 0.0001f;  // Q_s_scaling will be applied
    
    // Initialize measurement noise R
    R.setIdentity();
    R.block(2, 2, 2, 2) *= 10.0f;  // Higher noise for scale/ratio
    
    // Identity matrix
    I_.setIdentity();
}

void KalmanFilterXYSR::predict() {
    // x = F * x
    x = F * x;
    
    // P = F * P * F^T + Q
    P = F * P * F.transpose() + Q;
}

void KalmanFilterXYSR::update(const Eigen::VectorXf& z) {
    if (z.size() != dim_z_) {
        return;  // Invalid measurement
    }
    
    // Append to history
    history_obs.push_back(z);
    if (history_obs.size() > static_cast<size_t>(max_obs_)) {
        history_obs.pop_front();
    }
    
    // Innovation: y = z - H * x
    y = z - H * x;
    
    // System uncertainty: S = H * P * H^T + R
    S = H * P * H.transpose() + R;
    
    // Kalman gain: K = P * H^T * S^-1
    Eigen::LLT<Eigen::MatrixXf> chol(S);
    if (chol.info() == Eigen::Success) {
        K = P * H.transpose() * chol.solve(Eigen::MatrixXf::Identity(dim_z_, dim_z_));
    } else {
        // Fallback: use pseudo-inverse
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXf> cod(S);
        K = P * H.transpose() * cod.pseudoInverse();
    }
    
    // Update state: x = x + K * y
    x = x + K * y;
    
    // Update covariance: P = (I - K*H) * P * (I - K*H)^T + K*R*K^T
    Eigen::MatrixXf I_KH = I_ - K * H;
    P = I_KH * P * I_KH.transpose() + K * R * K.transpose();
}

void KalmanFilterXYSR::apply_affine_correction(const Eigen::Matrix2f& m, const Eigen::Vector2f& t) {
    // Transform center coordinates (x, y) using affine transformation
    // x[:2] = m @ x[:2] + t
    Eigen::Vector2f center = x.head<2>();
    Eigen::Vector2f transformed_center = m * center + t;
    x(0) = transformed_center(0);
    x(1) = transformed_center(1);
    
    // Transform velocity components (vx, vy) using rotation/scaling only (no translation)
    // x[4:6] = m @ x[4:6]
    Eigen::Vector2f velocity = x.segment<2>(4);
    Eigen::Vector2f transformed_velocity = m * velocity;
    x(4) = transformed_velocity(0);
    x(5) = transformed_velocity(1);
    
    // Transform covariance P for positions: P[:2, :2] = m @ P[:2, :2] @ m^T
    Eigen::Matrix2f P_pos = P.block<2, 2>(0, 0);
    P.block<2, 2>(0, 0) = m * P_pos * m.transpose();
    
    // Transform covariance P for velocities: P[4:6, 4:6] = m @ P[4:6, 4:6] @ m^T
    Eigen::Matrix2f P_vel = P.block<2, 2>(4, 4);
    P.block<2, 2>(4, 4) = m * P_vel * m.transpose();
    
    // Cross-covariance terms P[:2, 4:6] and P[4:6, :2] also need transformation
    Eigen::Matrix2f P_pos_vel = P.block<2, 2>(0, 4);
    P.block<2, 2>(0, 4) = m * P_pos_vel * m.transpose();
    P.block<2, 2>(4, 0) = P.block<2, 2>(0, 4).transpose();
}

} // namespace motcpp::motion

