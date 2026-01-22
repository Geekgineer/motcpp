// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/ucmc.hpp>
#include <motcpp/utils/matching.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace motcpp::trackers {

// ============================================================================
// UCMCKalmanFilter Implementation
// ============================================================================

UCMCKalmanFilter::UCMCKalmanFilter(int dim_x, int dim_z)
    : dim_x_(dim_x), dim_z_(dim_z)
{
    x_ = Eigen::Vector4d::Zero();
    P_ = Eigen::Matrix4d::Identity();
    Q_ = Eigen::Matrix4d::Identity();
    F_ = Eigen::Matrix4d::Identity();
    H_ = Eigen::Matrix<double, 2, 4>::Zero();
    H_(0, 0) = 1.0;
    H_(1, 2) = 1.0;
}

void UCMCKalmanFilter::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void UCMCKalmanFilter::update(const Eigen::Vector2d& z, const Eigen::Matrix2d& R) {
    // Innovation
    Eigen::Vector2d y = z - H_ * x_;
    
    // Innovation covariance
    Eigen::Matrix2d S = H_ * P_ * H_.transpose() + R;
    
    // Kalman gain
    Eigen::Matrix<double, 4, 2> K = P_ * H_.transpose() * S.inverse();
    
    // State update
    x_ = x_ + K * y;
    
    // Covariance update (Joseph form for numerical stability)
    Eigen::Matrix4d I_KH = Eigen::Matrix4d::Identity() - K * H_;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
}

// ============================================================================
// CameraMapper Implementation
// ============================================================================

CameraMapper::CameraMapper() : valid_(false) {}

CameraMapper::CameraMapper(const std::vector<double>& Ki, const std::vector<double>& Ko)
    : valid_(false)
{
    if (Ki.size() != 12 || Ko.size() != 16) {
        return;
    }
    
    // Map Ki (3x4) and Ko (4x4) from column-major vectors
    Eigen::Map<const Eigen::Matrix<double, 4, 3>> KiT(Ki.data());
    Eigen::Map<const Eigen::Matrix4d> KoT(Ko.data());
    
    Eigen::Matrix<double, 3, 4> KiMat = KiT.transpose();
    Eigen::Matrix4d KoMat = KoT.transpose();
    
    KiKo_ = KiMat * KoMat;
    
    A_.setZero();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 2; ++col) {
            A_(row, col) = KiKo_(row, col);
        }
        A_(row, 2) = KiKo_(row, 3);
    }
    
    InvA_ = A_.inverse();
    valid_ = true;
}

std::vector<double> CameraMapper::uvError(float bbox_w, float bbox_h) const {
    std::vector<double> uv(2);
    uv[0] = std::max(2.0, std::min(13.0, 0.05 * bbox_w));
    uv[1] = std::max(2.0, std::min(10.0, 0.05 * bbox_h));
    return uv;
}

void CameraMapper::uv2xy(const Eigen::Vector2d& uv, const Eigen::Matrix2d& sigma_uv,
                         Eigen::Vector2d& xy, Eigen::Matrix2d& sigma_xy) const {
    Eigen::Vector3d uv1(uv(0), uv(1), 1.0);
    Eigen::Vector3d b = InvA_ * uv1;
    
    double gamma = 1.0 / b(2);
    
    // Jacobian for error propagation
    Eigen::Matrix2d C = gamma * InvA_.block<2, 2>(0, 0)
                      - (gamma * gamma) * b.head<2>() * InvA_.block<1, 2>(2, 0);
    
    xy = b.head<2>() * gamma;
    sigma_xy = C * sigma_uv * C.transpose();
}

void CameraMapper::mapToGroundPlane(float bbox_cx, float bbox_bottom, float bbox_w, float bbox_h,
                                     Eigen::Vector2d& y, Eigen::Matrix2d& R) const {
    if (!valid_) {
        mapToImageSpace(bbox_cx, bbox_bottom, bbox_w, bbox_h, y, R);
        return;
    }
    
    Eigen::Vector2d uv(bbox_cx, bbox_bottom);
    std::vector<double> uv_err = uvError(bbox_w, bbox_h);
    
    Eigen::Matrix2d sigma_uv = Eigen::Matrix2d::Identity();
    sigma_uv(0, 0) = uv_err[0] * uv_err[0];
    sigma_uv(1, 1) = uv_err[1] * uv_err[1];
    
    uv2xy(uv, sigma_uv, y, R);
}

void CameraMapper::mapToImageSpace(float bbox_cx, float bbox_bottom, float bbox_w, float bbox_h,
                                    Eigen::Vector2d& y, Eigen::Matrix2d& R) const {
    // Fallback: use image coordinates directly (scaled)
    // Scale factor to normalize coordinates (assume typical image ~1000px)
    constexpr double scale = 0.01;
    
    y(0) = bbox_cx * scale;
    y(1) = bbox_bottom * scale;
    
    // Measurement noise based on bbox size
    double err_x = std::max(0.02, std::min(0.13, 0.0005 * bbox_w));
    double err_y = std::max(0.02, std::min(0.10, 0.0005 * bbox_h));
    
    R = Eigen::Matrix2d::Identity();
    R(0, 0) = err_x * err_x;
    R(1, 1) = err_y * err_y;
}

// ============================================================================
// UCMCSingleTrack Implementation
// ============================================================================

UCMCSingleTrack::UCMCSingleTrack(const Eigen::Vector2d& y, const Eigen::Matrix2d& /* R */,
                                  double wx, double wy, double vmax, float w, float h,
                                  double dt, int track_id)
    : id_(track_id)
    , age_(0)
    , death_count_(0)
    , birth_count_(0)
    , det_idx_(-1)
    , w_(w)
    , h_(h)
    , state_(UCMCTrackState::Tentative)
    , kf_(4, 2)
{
    // State transition matrix F
    kf_.F_ = Eigen::Matrix4d::Identity();
    kf_.F_(0, 1) = dt;  // x += vx * dt
    kf_.F_(2, 3) = dt;  // y += vy * dt
    
    // Initial state: position from detection, zero velocity
    kf_.x_(0) = y(0);
    kf_.x_(1) = 0.0;
    kf_.x_(2) = y(1);
    kf_.x_(3) = 0.0;
    
    // Initial covariance: high uncertainty on velocity
    kf_.P_.setZero();
    kf_.P_(0, 0) = 1.0;
    kf_.P_(1, 1) = vmax * vmax / 3.0;
    kf_.P_(2, 2) = 1.0;
    kf_.P_(3, 3) = vmax * vmax / 3.0;
    
    // Process noise Q = G * Q0 * G^T
    Eigen::Matrix<double, 4, 2> G;
    G << 0.5 * dt * dt, 0.0,
         dt, 0.0,
         0.0, 0.5 * dt * dt,
         0.0, dt;
    
    Eigen::Matrix2d Q0;
    Q0.setZero();
    Q0(0, 0) = wx;
    Q0(1, 1) = wy;
    
    kf_.Q_ = G * Q0 * G.transpose();
}

Eigen::Vector2d UCMCSingleTrack::predict() {
    kf_.predict();
    age_++;
    return kf_.H_ * kf_.x_;
}

void UCMCSingleTrack::update(const Eigen::Vector2d& y, const Eigen::Matrix2d& R) {
    kf_.update(y, R);
}

double UCMCSingleTrack::distance(const Eigen::Vector2d& y, const Eigen::Matrix2d& R) const {
    // Mahalanobis distance with log-determinant term
    Eigen::Vector2d diff = y - kf_.H_ * kf_.x_;
    Eigen::Matrix2d S = kf_.H_ * kf_.P_ * kf_.H_.transpose() + R;
    Eigen::Matrix2d SI = S.inverse();
    
    double mahalanobis = diff.transpose() * SI * diff;
    double logdet = std::log(S.determinant());
    
    return mahalanobis + logdet;
}

// ============================================================================
// UCMCTrack (Main Tracker) Implementation
// ============================================================================

UCMCTrack::UCMCTrack(float det_thresh,
                     int max_age,
                     int max_obs,
                     int min_hits,
                     float iou_threshold,
                     bool per_class,
                     int nr_classes,
                     const std::string& asso_func,
                     bool is_obb,
                     double a1,
                     double a2,
                     double wx,
                     double wy,
                     double vmax,
                     double dt,
                     float high_score,
                     const std::vector<double>& Ki,
                     const std::vector<double>& Ko)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                  per_class, nr_classes, asso_func, is_obb)
    , a1_(a1)
    , a2_(a2)
    , wx_(wx)
    , wy_(wy)
    , vmax_(vmax)
    , dt_(dt)
    , high_score_(high_score)
    , frame_count_(0)
    , tracker_count_(0)
{
    if (!Ki.empty() && !Ko.empty()) {
        mapper_ = CameraMapper(Ki, Ko);
    }
}

void UCMCTrack::reset() {
    trackers_.clear();
    confirmed_idx_.clear();
    coasted_idx_.clear();
    tentative_idx_.clear();
    frame_count_ = 0;
    tracker_count_ = 0;
}

Eigen::MatrixXf UCMCTrack::update(const Eigen::MatrixXf& dets,
                                  const cv::Mat& img,
                                  const Eigen::MatrixXf& embs) {
    (void)img;
    (void)embs;
    
    frame_count_++;
    
    // Map detections to ground-plane coordinates
    std::vector<MappedDetection> mapped_dets;
    mapped_dets.reserve(dets.rows());
    
    for (int i = 0; i < dets.rows(); ++i) {
        float conf = dets(i, 4);
        if (conf < det_thresh_) continue;
        
        MappedDetection md;
        md.original_idx = i;
        md.x1 = dets(i, 0);
        md.y1 = dets(i, 1);
        md.x2 = dets(i, 2);
        md.y2 = dets(i, 3);
        md.conf = conf;
        md.cls = dets.cols() > 5 ? static_cast<int>(dets(i, 5)) : 0;
        md.w = md.x2 - md.x1;
        md.h = md.y2 - md.y1;
        
        float cx = (md.x1 + md.x2) / 2.0f;
        float bottom = md.y2;
        
        mapper_.mapToGroundPlane(cx, bottom, md.w, md.h, md.y, md.R);
        
        mapped_dets.push_back(md);
    }
    
    // Run UCMC tracking pipeline
    dataAssociation(mapped_dets);
    
    std::vector<int> det_remain;
    associateTentative(mapped_dets, det_remain);
    initTentative(mapped_dets, det_remain);
    deleteOldTrackers();
    updateStatus();
    
    // Build output
    std::vector<Eigen::VectorXf> outputs;
    outputs.reserve(trackers_.size());
    
    for (const auto& trk : trackers_) {
        if (trk.state() == UCMCTrackState::Confirmed && trk.detIdx() >= 0) {
            int det_idx = trk.detIdx();
            
            // Find the mapped detection
            for (const auto& md : mapped_dets) {
                if (md.original_idx == det_idx) {
                    Eigen::VectorXf output(8);
                    output(0) = md.x1;
                    output(1) = md.y1;
                    output(2) = md.x2;
                    output(3) = md.y2;
                    output(4) = static_cast<float>(trk.id());
                    output(5) = md.conf;
                    output(6) = static_cast<float>(md.cls);
                    output(7) = static_cast<float>(md.original_idx);
                    outputs.push_back(output);
                    break;
                }
            }
        }
    }
    
    if (outputs.empty()) {
        return Eigen::MatrixXf(0, 8);
    }
    
    Eigen::MatrixXf result(outputs.size(), 8);
    for (size_t i = 0; i < outputs.size(); ++i) {
        result.row(i) = outputs[i].transpose();
    }
    
    return result;
}

void UCMCTrack::dataAssociation(std::vector<MappedDetection>& dets) {
    // Split detections by confidence
    std::vector<int> det_high, det_low;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (dets[i].conf >= high_score_) {
            det_high.push_back(static_cast<int>(i));
        } else {
            det_low.push_back(static_cast<int>(i));
        }
    }
    
    // Predict all trackers
    for (auto& trk : trackers_) {
        trk.predict();
        trk.setDetIdx(-1);
    }
    
    // First association: high-confidence detections with confirmed + coasted tracks
    std::vector<int> track_idx = confirmed_idx_;
    track_idx.insert(track_idx.end(), coasted_idx_.begin(), coasted_idx_.end());
    
    std::vector<int> det_remain = det_high;
    std::vector<int> track_remain;
    
    int num_det = det_high.size();
    int num_trk = track_idx.size();
    
    if (num_det > 0 && num_trk > 0) {
        // Build cost matrix using Mahalanobis distance
        Eigen::MatrixXd cost_matrix(num_trk, num_det);
        
        for (int i = 0; i < num_trk; ++i) {
            for (int j = 0; j < num_det; ++j) {
                cost_matrix(i, j) = trackers_[track_idx[i]].distance(
                    dets[det_high[j]].y, dets[det_high[j]].R);
            }
        }
        
        // Linear assignment with Mahalanobis threshold
        auto result = utils::linear_assignment(cost_matrix.cast<float>(), static_cast<float>(a1_));
        
        // Process matches
        for (const auto& m : result.matches) {
            int trk_local = m[0];
            int det_local = m[1];
            int trk_idx_val = track_idx[trk_local];
            int det_idx_val = det_high[det_local];
            
            trackers_[trk_idx_val].update(dets[det_idx_val].y, dets[det_idx_val].R);
            trackers_[trk_idx_val].setDeathCount(0);
            trackers_[trk_idx_val].setDetIdx(dets[det_idx_val].original_idx);
            trackers_[trk_idx_val].setState(UCMCTrackState::Confirmed);
        }
        
        // Unmatched detections
        det_remain.clear();
        for (int i : result.unmatched_b) {
            det_remain.push_back(det_high[i]);
        }
        
        // Unmatched tracks
        for (int i : result.unmatched_a) {
            track_remain.push_back(track_idx[i]);
        }
    } else {
        track_remain = track_idx;
    }
    
    // Second association: low-confidence detections with remaining tracks
    int num_det_low = det_low.size();
    int num_trk_remain = track_remain.size();
    
    if (num_det_low > 0 && num_trk_remain > 0) {
        Eigen::MatrixXd cost_matrix(num_trk_remain, num_det_low);
        
        for (int i = 0; i < num_trk_remain; ++i) {
            for (int j = 0; j < num_det_low; ++j) {
                cost_matrix(i, j) = trackers_[track_remain[i]].distance(
                    dets[det_low[j]].y, dets[det_low[j]].R);
            }
        }
        
        auto result = utils::linear_assignment(cost_matrix.cast<float>(), static_cast<float>(a2_));
        
        for (const auto& m : result.matches) {
            int trk_local = m[0];
            int det_local = m[1];
            int trk_idx_val = track_remain[trk_local];
            int det_idx_val = det_low[det_local];
            
            trackers_[trk_idx_val].update(dets[det_idx_val].y, dets[det_idx_val].R);
            trackers_[trk_idx_val].setDeathCount(0);
            trackers_[trk_idx_val].setDetIdx(dets[det_idx_val].original_idx);
            trackers_[trk_idx_val].setState(UCMCTrackState::Confirmed);
        }
        
        // Mark unmatched tracks as coasted
        for (int i : result.unmatched_a) {
            trackers_[track_remain[i]].setState(UCMCTrackState::Coasted);
        }
    } else {
        // Mark all remaining tracks as coasted
        for (int idx : track_remain) {
            trackers_[idx].setState(UCMCTrackState::Coasted);
        }
    }
    
    // Store remaining detections for tentative association
    // (will be passed via parameter in associateTentative)
}

void UCMCTrack::associateTentative(std::vector<MappedDetection>& dets,
                                    std::vector<int>& det_remain) {
    // Find unassigned high-confidence detections
    det_remain.clear();
    for (size_t i = 0; i < dets.size(); ++i) {
        bool assigned = false;
        for (const auto& trk : trackers_) {
            if (trk.detIdx() == dets[i].original_idx) {
                assigned = true;
                break;
            }
        }
        if (!assigned && dets[i].conf >= high_score_) {
            det_remain.push_back(static_cast<int>(i));
        }
    }
    
    int num_det = det_remain.size();
    int num_trk = tentative_idx_.size();
    
    if (num_det == 0 || num_trk == 0) {
        return;
    }
    
    Eigen::MatrixXd cost_matrix(num_trk, num_det);
    
    for (int i = 0; i < num_trk; ++i) {
        for (int j = 0; j < num_det; ++j) {
            cost_matrix(i, j) = trackers_[tentative_idx_[i]].distance(
                dets[det_remain[j]].y, dets[det_remain[j]].R);
        }
    }
    
    auto result = utils::linear_assignment(cost_matrix.cast<float>(), static_cast<float>(a1_));
    
    for (const auto& m : result.matches) {
        int trk_local = m[0];
        int det_local = m[1];
        int trk_idx_val = tentative_idx_[trk_local];
        int det_idx_val = det_remain[det_local];
        
        trackers_[trk_idx_val].update(dets[det_idx_val].y, dets[det_idx_val].R);
        trackers_[trk_idx_val].setDeathCount(0);
        trackers_[trk_idx_val].incrementBirthCount();
        trackers_[trk_idx_val].setDetIdx(dets[det_idx_val].original_idx);
        
        if (trackers_[trk_idx_val].birthCount() >= 2) {
            trackers_[trk_idx_val].setBirthCount(0);
            trackers_[trk_idx_val].setState(UCMCTrackState::Confirmed);
        }
    }
    
    // Update det_remain to only unmatched detections
    std::vector<int> new_det_remain;
    for (int i : result.unmatched_b) {
        new_det_remain.push_back(det_remain[i]);
    }
    det_remain = std::move(new_det_remain);
}

void UCMCTrack::initTentative(std::vector<MappedDetection>& dets,
                               const std::vector<int>& det_remain) {
    for (int i : det_remain) {
        UCMCSingleTrack new_track(
            dets[i].y, dets[i].R,
            wx_, wy_, vmax_,
            dets[i].w, dets[i].h,
            dt_, ++tracker_count_
        );
        new_track.setState(UCMCTrackState::Tentative);
        new_track.setDetIdx(dets[i].original_idx);
        trackers_.push_back(std::move(new_track));
    }
}

void UCMCTrack::deleteOldTrackers() {
    std::vector<UCMCSingleTrack> remaining;
    remaining.reserve(trackers_.size());
    
    for (auto& trk : trackers_) {
        trk.incrementDeathCount();
        
        bool should_delete = 
            (trk.state() == UCMCTrackState::Coasted && trk.deathCount() >= max_age_) ||
            (trk.state() == UCMCTrackState::Tentative && trk.deathCount() >= 2);
        
        if (!should_delete) {
            remaining.push_back(std::move(trk));
        }
    }
    
    trackers_ = std::move(remaining);
}

void UCMCTrack::updateStatus() {
    confirmed_idx_.clear();
    coasted_idx_.clear();
    tentative_idx_.clear();
    
    for (size_t i = 0; i < trackers_.size(); ++i) {
        (void)trackers_[i].detIdx();  // Check detIdx but don't use it
        
        switch (trackers_[i].state()) {
            case UCMCTrackState::Confirmed:
                confirmed_idx_.push_back(static_cast<int>(i));
                break;
            case UCMCTrackState::Coasted:
                coasted_idx_.push_back(static_cast<int>(i));
                break;
            case UCMCTrackState::Tentative:
                tentative_idx_.push_back(static_cast<int>(i));
                break;
            default:
                break;
        }
    }
}

} // namespace motcpp::trackers
