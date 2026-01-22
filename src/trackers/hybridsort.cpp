// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/hybridsort.hpp>
#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/ops.hpp>
#include <motcpp/utils/iou.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <numeric>

namespace motcpp::trackers {

// Helper functions for bbox conversion
// Static members
int HybridKalmanBoxTracker::next_id_ = 0;

int HybridKalmanBoxTracker::next_id() {
    return ++next_id_;
}

// HybridKalmanFilter implementation
HybridKalmanFilter::HybridKalmanFilter() {
    // Initialize state transition matrix F (9x9)
    F = Eigen::MatrixXf::Identity(9, 9);
    F(0, 5) = 1.0f;  // u += du
    F(1, 6) = 1.0f;  // v += dv
    F(2, 7) = 1.0f;  // s += ds
    F(3, 8) = 1.0f;  // c += dc
    
    // Measurement matrix H (5x9)
    H = Eigen::MatrixXf::Zero(5, 9);
    H(0, 0) = 1.0f;  // observe u
    H(1, 1) = 1.0f;  // observe v
    H(2, 2) = 1.0f;  // observe s
    H(3, 3) = 1.0f;  // observe c
    H(4, 4) = 1.0f;  // observe r
    
    // Process noise Q
    Q = Eigen::MatrixXf::Identity(9, 9) * 0.1f;
    Q(8, 8) = 0.01f;  // dc noise
    Q(7, 7) = 0.01f;  // ds noise
    Q(5, 5) = 0.01f;  // du noise
    Q(6, 6) = 0.01f;  // dv noise
    
    // Measurement noise R
    R = Eigen::MatrixXf::Identity(5, 5);
    R(2, 2) = 10.0f;  // s noise
    R(3, 3) = 0.01f;  // c noise
    
    // Initialize state and covariance
    x = Eigen::VectorXf::Zero(9);
    P = Eigen::MatrixXf::Identity(9, 9) * 10.0f;
    P.block<4, 4>(5, 5) *= 1000.0f;  // High uncertainty for velocities
}

void HybridKalmanFilter::init(const Eigen::VectorXf& z) {
    if (z.size() >= 5) {
        x.head<5>() = z.head<5>();
    }
    x.tail<4>().setZero();  // Initialize velocities to zero
}

void HybridKalmanFilter::predict() {
    x = F * x;
    P = F * P * F.transpose() + Q;
}

void HybridKalmanFilter::update(const Eigen::VectorXf& z) {
    if (z.size() < 5) return;
    // Project to measurement space
    Eigen::VectorXf projected_mean = H * x;
    Eigen::MatrixXf projected_cov = H * P * H.transpose() + R;
    
    // Kalman gain
    Eigen::MatrixXf K = P * H.transpose() * projected_cov.inverse();
    
    // Innovation
    Eigen::VectorXf z_5 = z.head<5>();
    Eigen::VectorXf innovation = z_5 - projected_mean;
    
    // Update
    x = x + K * innovation;
    P = (Eigen::MatrixXf::Identity(9, 9) - K * H) * P;
}

void HybridKalmanFilter::camera_update(const Eigen::Matrix3f& transform) {
    // Get current bbox from state
    float u = x(0), v = x(1), s = x(2), c = x(3), r = x(4);
    float w = std::sqrt(s * r);
    float h = s / w;
    float x1 = u - w / 2.0f, y1 = v - h / 2.0f;
    float x2 = u + w / 2.0f, y2 = v + h / 2.0f;
    
    // Transform corners
    Eigen::Vector3f p1(x1, y1, 1.0f);
    Eigen::Vector3f p2(x2, y2, 1.0f);
    
    Eigen::Vector3f p1_new = transform * p1;
    Eigen::Vector3f p2_new = transform * p2;
    
    float x1_new = p1_new(0) / p1_new(2);
    float y1_new = p1_new(1) / p1_new(2);
    float x2_new = p2_new(0) / p2_new(2);
    float y2_new = p2_new(1) / p2_new(2);
    
    // Rebuild state
    float w_new = x2_new - x1_new;
    float h_new = y2_new - y1_new;
    float u_new = x1_new + w_new / 2.0f;
    float v_new = y1_new + h_new / 2.0f;
    float s_new = w_new * h_new;
    float r_new = (h_new > 1e-6f) ? w_new / h_new : 0.0f;
    
    Eigen::VectorXf z_new(5);
    z_new << u_new, v_new, s_new, c, r_new;
    init(z_new);
}

Eigen::VectorXf HybridKalmanFilter::get_state() const {
    return x.head<5>();
}

// HybridKalmanBoxTracker implementation
HybridKalmanBoxTracker::HybridKalmanBoxTracker(
    const Eigen::VectorXf& bbox,
    const Eigen::VectorXf& temp_feat,
    int delta_t,
    bool /* use_custom_kf */,
    int longterm_bank_length,
    float alpha,
    bool adapfs,
    float track_thresh,
    int cls,
    int det_ind
)
    : id_(next_id())
    , age_(0)
    , hits_(0)
    , hit_streak_(0)
    , time_since_update_(0)
    , conf_(bbox.size() > 4 ? bbox(4) : 0.0f)
    , confidence_pre_(0.0f)
    , cls_(cls)
    , det_ind_(det_ind)
    , delta_t_(delta_t)
    , longterm_bank_length_(longterm_bank_length)
    , track_thresh_(track_thresh)
    , alpha_(alpha)
    , adapfs_(adapfs)
    , last_observation_(5)
{
    velocity_lt = Eigen::Vector2f::Zero();
    velocity_rt = Eigen::Vector2f::Zero();
    velocity_lb = Eigen::Vector2f::Zero();
    velocity_rb = Eigen::Vector2f::Zero();
    last_observation_ << -1, -1, -1, -1, -1;
    
    // Initialize Kalman filter
    Eigen::VectorXf z(5);
    float w = bbox(2) - bbox(0);
    float h = bbox(3) - bbox(1);
    float u = bbox(0) + w / 2.0f;
    float v = bbox(1) + h / 2.0f;
    float s = w * h;
    float r = (h > 1e-6f) ? w / h : 0.0f;
    float c = bbox.size() > 4 ? bbox(4) : 0.0f;
    z << u, v, s, c, r;
    kf_.init(z);
    
    // Initialize features
    if (temp_feat.size() > 0) {
        update_features(temp_feat);
    }
}

namespace {
    Eigen::VectorXf convert_bbox_to_z(const Eigen::VectorXf& bbox) {
        float w = bbox(2) - bbox(0);
        float h = bbox(3) - bbox(1);
        float u = bbox(0) + w / 2.0f;
        float v = bbox(1) + h / 2.0f;
        float s = w * h;
        float r = (h > 1e-6f) ? w / h : 0.0f;
        float c = bbox.size() > 4 ? bbox(4) : 0.0f;
        Eigen::VectorXf z(5);
        z << u, v, s, c, r;
        return z;
    }
    
    Eigen::Vector4f convert_x_to_bbox(const Eigen::VectorXf& x) {
        if (x.size() < 5) return Eigen::Vector4f::Zero();
        float u = x(0), v = x(1), s = x(2), r = x(4);
        float w = std::sqrt(s * r);
        float h = s / w;
        return Eigen::Vector4f(u - w/2, v - h/2, u + w/2, v + h/2);
    }
    
    Eigen::Vector2f speed_direction_lt(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) {
        float cx1 = bbox1(0), cy1 = bbox1(1);
        float cx2 = bbox2(0), cy2 = bbox2(1);
        Eigen::Vector2f speed(cy2 - cy1, cx2 - cx1);
        float norm = speed.norm() + 1e-6f;
        return speed / norm;
    }
    
    Eigen::Vector2f speed_direction_rt(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) {
        float cx1 = bbox1(0), cy1 = bbox1(3);
        float cx2 = bbox2(0), cy2 = bbox2(3);
        Eigen::Vector2f speed(cy2 - cy1, cx2 - cx1);
        float norm = speed.norm() + 1e-6f;
        return speed / norm;
    }
    
    Eigen::Vector2f speed_direction_lb(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) {
        float cx1 = bbox1(2), cy1 = bbox1(1);
        float cx2 = bbox2(2), cy2 = bbox2(1);
        Eigen::Vector2f speed(cy2 - cy1, cx2 - cx1);
        float norm = speed.norm() + 1e-6f;
        return speed / norm;
    }
    
    Eigen::Vector2f speed_direction_rb(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) {
        float cx1 = bbox1(2), cy1 = bbox1(3);
        float cx2 = bbox2(2), cy2 = bbox2(3);
        Eigen::Vector2f speed(cy2 - cy1, cx2 - cx1);
        float norm = speed.norm() + 1e-6f;
        return speed / norm;
    }
}

Eigen::VectorXf HybridKalmanBoxTracker::convert_bbox_to_z(const Eigen::VectorXf& bbox) const {
    float w = bbox(2) - bbox(0);
    float h = bbox(3) - bbox(1);
    float u = bbox(0) + w / 2.0f;
    float v = bbox(1) + h / 2.0f;
    float s = w * h;
    float r = (h > 1e-6f) ? w / h : 0.0f;
    float c = bbox.size() > 4 ? bbox(4) : 0.0f;
    Eigen::VectorXf z(5);
    z << u, v, s, c, r;
    return z;
}

Eigen::Vector4f HybridKalmanBoxTracker::convert_x_to_bbox(const Eigen::VectorXf& x) const {
    if (x.size() < 5) return Eigen::Vector4f::Zero();
    float u = x(0), v = x(1), s = x(2), r = x(4);
    float w = std::sqrt(s * r);
    float h = s / w;
    return Eigen::Vector4f(u - w/2, v - h/2, u + w/2, v + h/2);
}

void HybridKalmanBoxTracker::predict() {
    if (kf_.x(7) + kf_.x(2) <= 0) {
        kf_.x(7) = 0.0f;
    }
    
    kf_.predict();
    age_++;
    if (time_since_update_ > 0) {
        hit_streak_ = 0;
    }
    time_since_update_++;
    
    Eigen::Vector4f bbox = convert_x_to_bbox(kf_.x);
    history_.push_back(bbox);
}

void HybridKalmanBoxTracker::update(const Eigen::VectorXf& bbox, const Eigen::VectorXf& id_feature,
                                    bool update_feature, int cls, int det_ind) {
    if (bbox.size() >= 4) {
        // Calculate velocities
        if (last_observation_.head<4>().sum() >= 0) {
            Eigen::VectorXf previous_box = k_previous_obs(age_, delta_t_);
            if (previous_box.size() >= 4 && previous_box.head<4>().sum() >= 0) {
                Eigen::VectorXf prev_4 = previous_box.head<4>();
                Eigen::VectorXf curr_4 = bbox.head<4>();
                velocity_lt = speed_direction_lt(prev_4, curr_4);
                velocity_rt = speed_direction_rt(prev_4, curr_4);
                velocity_lb = speed_direction_lb(prev_4, curr_4);
                velocity_rb = speed_direction_rb(prev_4, curr_4);
            } else {
                Eigen::VectorXf last_4 = last_observation_.head<4>();
                Eigen::VectorXf curr_4 = bbox.head<4>();
                velocity_lt = speed_direction_lt(last_4, curr_4);
                velocity_rt = speed_direction_rt(last_4, curr_4);
                velocity_lb = speed_direction_lb(last_4, curr_4);
                velocity_rb = speed_direction_rb(last_4, curr_4);
            }
        }
        
        Eigen::VectorXf obs(5);
        obs.head<4>() = bbox.head<4>();
        obs(4) = bbox.size() > 4 ? bbox(4) : 0.0f;
        last_observation_ = obs;
        observations[age_] = obs;
        history_observations_.push_back(obs);
        
        time_since_update_ = 0;
        history_.clear();
        hits_++;
        hit_streak_++;
        
        Eigen::VectorXf z = convert_bbox_to_z(bbox);
        kf_.update(z);
        
        if (cls >= 0) cls_ = cls;
        if (det_ind >= 0) det_ind_ = det_ind;
        
        if (update_feature && id_feature.size() > 0) {
            float score = bbox.size() > 4 ? bbox(4) : -1.0f;
            update_features(id_feature, score);
        }
        
        confidence_pre_ = conf_;
        conf_ = bbox.size() > 4 ? bbox(4) : 0.0f;
    } else {
        // No bbox update
        Eigen::VectorXf z = Eigen::VectorXf::Zero(5);
        kf_.update(z);
        confidence_pre_ = 0.0f;
    }
}

void HybridKalmanBoxTracker::camera_update(const Eigen::Matrix3f& warp_matrix) {
    kf_.camera_update(warp_matrix);
}

void HybridKalmanBoxTracker::update_features(const Eigen::VectorXf& feat, float score) {
    if (feat.size() == 0) return;
    
    Eigen::VectorXf normalized_feat = feat;
    float norm = normalized_feat.norm() + 1e-12f;
    normalized_feat /= norm;
    
    curr_feat_ = normalized_feat;
    
    if (smooth_feat_.size() == 0) {
        smooth_feat_ = normalized_feat;
    } else {
        if (adapfs_ && score > 0) {
            float pre_w = alpha_ * (conf_ / (conf_ + score));
            float cur_w = (1.0f - alpha_) * (score / (conf_ + score));
            float s = pre_w + cur_w;
            pre_w /= s;
            cur_w /= s;
            smooth_feat_ = pre_w * smooth_feat_ + cur_w * normalized_feat;
        } else {
            smooth_feat_ = alpha_ * smooth_feat_ + (1.0f - alpha_) * normalized_feat;
        }
        float smooth_norm = smooth_feat_.norm() + 1e-12f;
        smooth_feat_ /= smooth_norm;
    }
    
    features_.push_back(normalized_feat);
    while (features_.size() > static_cast<size_t>(longterm_bank_length_)) {
        features_.pop_front();
    }
}

Eigen::Vector4f HybridKalmanBoxTracker::get_bbox() const {
    if (last_observation_.head<4>().sum() < 0) {
        return convert_x_to_bbox(kf_.x);
    }
    return last_observation_.head<4>();
}

float HybridKalmanBoxTracker::get_kalman_score() const {
    float x3 = kf_.x(3);
    return std::clamp(x3, track_thresh_, 1.0f);
}

float HybridKalmanBoxTracker::get_simple_score() const {
    if (confidence_pre_ == 0.0f) {
        return std::clamp(conf_, 0.1f, track_thresh_);
    }
    return std::clamp(conf_ - (confidence_pre_ - conf_), 0.1f, track_thresh_);
}

Eigen::VectorXf HybridKalmanBoxTracker::k_previous_obs(int cur_age, int k) const {
    if (observations.empty()) {
        Eigen::VectorXf result(5);
        result << -1, -1, -1, -1, -1;
        return result;
    }
    
    for (int i = 0; i < k; ++i) {
        int dt = k - i;
        int age_key = cur_age - dt;
        auto it = observations.find(age_key);
        if (it != observations.end()) {
            return it->second;
        }
    }
    
    int max_age = std::max_element(observations.begin(), observations.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; })->first;
    return observations.at(max_age);
}


// HybridSort implementation
HybridSort::HybridSort(
    const std::string& reid_weights,
    bool use_half,
    bool use_gpu,
    float det_thresh,
    int max_age,
    int max_obs,
    int min_hits,
    float iou_threshold,
    bool per_class,
    int nr_classes,
    const std::string& asso_func,
    bool is_obb,
    float low_thresh,
    int delta_t,
    float inertia,
    bool use_byte,
    bool use_custom_kf,
    int longterm_bank_length,
    float alpha,
    bool adapfs,
    float track_thresh,
    float EG_weight_high_score,
    float EG_weight_low_score,
    bool TCM_first_step,
    bool TCM_byte_step,
    float TCM_byte_step_weight,
    float high_score_matching_thresh,
    bool with_longterm_reid,
    float longterm_reid_weight,
    bool with_longterm_reid_correction,
    float longterm_reid_correction_thresh,
    float longterm_reid_correction_thresh_low,
    const std::string& cmc_method,
    bool with_reid
)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                  per_class, nr_classes, asso_func, is_obb)
    , low_thresh_(low_thresh)
    , delta_t_(delta_t)
    , inertia_(inertia)
    , use_byte_(use_byte)
    , use_custom_kf_(use_custom_kf)
    , longterm_bank_length_(longterm_bank_length)
    , alpha_(alpha)
    , adapfs_(adapfs)
    , track_thresh_(track_thresh)
    , EG_weight_high_score_(EG_weight_high_score)
    , EG_weight_low_score_(EG_weight_low_score)
    , TCM_first_step_(TCM_first_step)
    , TCM_byte_step_(TCM_byte_step)
    , TCM_byte_step_weight_(TCM_byte_step_weight)
    , high_score_matching_thresh_(high_score_matching_thresh)
    , with_longterm_reid_(with_longterm_reid)
    , longterm_reid_weight_(longterm_reid_weight)
    , with_longterm_reid_correction_(with_longterm_reid_correction)
    , longterm_reid_correction_thresh_(longterm_reid_correction_thresh)
    , longterm_reid_correction_thresh_low_(longterm_reid_correction_thresh_low)
    , with_reid_(with_reid)
    , cmc_method_(cmc_method)
{
    if (cmc_method_ == "ecc") {
        cmc_ = std::make_unique<motcpp::motion::ECC>();
    }
    
    if (with_reid_ && !reid_weights.empty()) {
        reid_model_ = std::make_unique<appearance::ONNXBackend>(reid_weights, "", use_half, use_gpu);
    }
    
    HybridKalmanBoxTracker::next_id_ = 0;
}

void HybridSort::reset() {
    BaseTracker::reset();
    active_tracks_.clear();
    HybridKalmanBoxTracker::next_id_ = 0;
}

Eigen::VectorXf HybridSort::convert_bbox_to_z(const Eigen::VectorXf& bbox) const {
    return motcpp::trackers::convert_bbox_to_z(bbox);
}

Eigen::Vector4f HybridSort::convert_x_to_bbox(const Eigen::VectorXf& x) const {
    return motcpp::trackers::convert_x_to_bbox(x);
}

Eigen::Vector2f HybridSort::speed_direction_lt(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const {
    return motcpp::trackers::speed_direction_lt(bbox1, bbox2);
}

Eigen::Vector2f HybridSort::speed_direction_rt(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const {
    return motcpp::trackers::speed_direction_rt(bbox1, bbox2);
}

Eigen::Vector2f HybridSort::speed_direction_lb(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const {
    return motcpp::trackers::speed_direction_lb(bbox1, bbox2);
}

Eigen::Vector2f HybridSort::speed_direction_rb(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const {
    return motcpp::trackers::speed_direction_rb(bbox1, bbox2);
}

Eigen::VectorXf HybridSort::k_previous_obs(const std::unordered_map<int, Eigen::VectorXf>& observations, int cur_age, int k) const {
    if (observations.empty()) {
        Eigen::VectorXf result(5);
        result << -1, -1, -1, -1, -1;
        return result;
    }
    
    for (int i = 0; i < k; ++i) {
        int dt = k - i;
        int age_key = cur_age - dt;
        auto it = observations.find(age_key);
        if (it != observations.end()) {
            return it->second;
        }
    }
    
    int max_age = std::max_element(observations.begin(), observations.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; })->first;
    return observations.at(max_age);
}

Eigen::MatrixXf HybridSort::iou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const {
    if (bboxes1.rows() == 0 || bboxes2.rows() == 0) {
        return Eigen::MatrixXf(0, 0);
    }
    
    Eigen::MatrixXf iou_matrix(bboxes1.rows(), bboxes2.rows());
    
    for (int i = 0; i < bboxes1.rows(); ++i) {
        Eigen::Vector4f b1(bboxes1(i, 0), bboxes1(i, 1), bboxes1(i, 2), bboxes1(i, 3));
        for (int j = 0; j < bboxes2.rows(); ++j) {
            Eigen::Vector4f b2(bboxes2(j, 0), bboxes2(j, 1), bboxes2(j, 2), bboxes2(j, 3));
            // Compute IoU manually
            float xx1 = std::max(b1(0), b2(0));
            float yy1 = std::max(b1(1), b2(1));
            float xx2 = std::min(b1(2), b2(2));
            float yy2 = std::min(b1(3), b2(3));
            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float intersection = w * h;
            float area1 = (b1(2) - b1(0)) * (b1(3) - b1(1));
            float area2 = (b2(2) - b2(0)) * (b2(3) - b2(1));
            float union_area = area1 + area2 - intersection;
            iou_matrix(i, j) = (union_area > 1e-6f) ? intersection / union_area : 0.0f;
        }
    }
    
    return iou_matrix;
}

Eigen::MatrixXf HybridSort::hmiou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const {
    // Height-Modulated IoU - simplified version
    Eigen::MatrixXf iou = iou_batch(bboxes1, bboxes2);
    
    // Add height modulation
    for (int i = 0; i < bboxes1.rows(); ++i) {
        (void)(bboxes1(i, 3) - bboxes1(i, 1));  // h1 unused
        for (int j = 0; j < bboxes2.rows(); ++j) {
            (void)(bboxes2(j, 3) - bboxes2(j, 1));  // h2 unused
            float yy1 = std::max(bboxes1(i, 1), bboxes2(j, 1));
            float yy2 = std::min(bboxes1(i, 3), bboxes2(j, 3));
            float yy3 = std::min(bboxes1(i, 1), bboxes2(j, 1));
            float yy4 = std::max(bboxes1(i, 3), bboxes2(j, 3));
            float h_overlap = std::max(0.0f, yy2 - yy1) / (yy4 - yy3 + 1e-6f);
            iou(i, j) *= h_overlap;
        }
    }
    
    return iou;
}

Eigen::MatrixXf HybridSort::giou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const {
    // Simplified GIoU - can be enhanced later
    return iou_batch(bboxes1, bboxes2);
}

Eigen::MatrixXf HybridSort::ciou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const {
    // Simplified CIoU - can be enhanced later
    return iou_batch(bboxes1, bboxes2);
}

Eigen::MatrixXf HybridSort::diou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const {
    // Simplified DIoU - can be enhanced later
    return iou_batch(bboxes1, bboxes2);
}

Eigen::MatrixXf HybridSort::ct_dist_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const {
    // Center distance - simplified
    Eigen::MatrixXf dist_matrix(bboxes1.rows(), bboxes2.rows());
    
    for (int i = 0; i < bboxes1.rows(); ++i) {
        float cx1 = (bboxes1(i, 0) + bboxes1(i, 2)) / 2.0f;
        float cy1 = (bboxes1(i, 1) + bboxes1(i, 3)) / 2.0f;
        for (int j = 0; j < bboxes2.rows(); ++j) {
            float cx2 = (bboxes2(j, 0) + bboxes2(j, 2)) / 2.0f;
            float cy2 = (bboxes2(j, 1) + bboxes2(j, 3)) / 2.0f;
            float dist = std::sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2));
            dist_matrix(i, j) = dist;
        }
    }
    
    // Normalize
    float max_dist = dist_matrix.maxCoeff();
    if (max_dist > 1e-6f) {
        dist_matrix = dist_matrix / max_dist;
        dist_matrix = Eigen::MatrixXf::Ones(dist_matrix.rows(), dist_matrix.cols()) * max_dist - dist_matrix;
    }
    
    return dist_matrix;
}

Eigen::MatrixXf HybridSort::cal_score_dif_batch_two_score(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const {
    Eigen::MatrixXf score_diff(bboxes1.rows(), bboxes2.rows());
    
    for (int i = 0; i < bboxes1.rows(); ++i) {
        float score1 = bboxes1(i, 4);
        for (int j = 0; j < bboxes2.rows(); ++j) {
            float score2 = bboxes2.cols() > 5 ? bboxes2(j, 5) : bboxes2(j, 4);
            score_diff(i, j) = std::abs(score2 - score1);
        }
    }
    
    return score_diff;
}

std::tuple<Eigen::MatrixXi, Eigen::VectorXi, Eigen::VectorXi>
HybridSort::associate_4_points_with_score(
    const Eigen::MatrixXf& detections,
    const Eigen::MatrixXf& trackers,
    float iou_threshold,
    const Eigen::MatrixXf& /* velocities_lt */,
    const Eigen::MatrixXf& /* velocities_rt */,
    const Eigen::MatrixXf& /* velocities_lb */,
    const Eigen::MatrixXf& /* velocities_rb */,
    const Eigen::MatrixXf& /* k_observations */,
    float /* inertia */,
    const std::string& asso_func
) const {
    // Simplified association - can be enhanced with 4-point matching later
    Eigen::MatrixXf iou_matrix;
    
    if (asso_func == "hmiou") {
        iou_matrix = hmiou_batch(detections.leftCols(4), trackers.leftCols(4));
    } else if (asso_func == "giou") {
        iou_matrix = giou_batch(detections.leftCols(4), trackers.leftCols(4));
    } else if (asso_func == "ciou") {
        iou_matrix = ciou_batch(detections.leftCols(4), trackers.leftCols(4));
    } else if (asso_func == "diou") {
        iou_matrix = diou_batch(detections.leftCols(4), trackers.leftCols(4));
    } else if (asso_func == "ct_dist") {
        iou_matrix = ct_dist_batch(detections.leftCols(4), trackers.leftCols(4));
    } else {
        iou_matrix = iou_batch(detections.leftCols(4), trackers.leftCols(4));
    }
    
    // Convert to cost matrix (1 - iou)
    Eigen::MatrixXf cost_matrix = Eigen::MatrixXf::Ones(iou_matrix.rows(), iou_matrix.cols()) - iou_matrix;
    
    // Linear assignment - threshold is on COST (1 - IoU), so pass (1 - iou_threshold)
    // This allows matches with IoU >= iou_threshold
    auto result = utils::linear_assignment(cost_matrix, 1.0f - iou_threshold);
    
    // Filter matches by IoU threshold
    std::vector<std::array<int, 2>> filtered_matches;
    std::vector<int> unmatched_dets, unmatched_trks;
    
    std::unordered_set<int> matched_det_set, matched_trk_set;
    for (const auto& m : result.matches) {
        if (iou_matrix(m[0], m[1]) >= iou_threshold) {
            filtered_matches.push_back(m);
            matched_det_set.insert(m[0]);
            matched_trk_set.insert(m[1]);
        } else {
            unmatched_dets.push_back(m[0]);
            unmatched_trks.push_back(m[1]);
        }
    }
    
    for (int i = 0; i < detections.rows(); ++i) {
        if (matched_det_set.find(i) == matched_det_set.end()) {
            unmatched_dets.push_back(i);
        }
    }
    
    for (int i = 0; i < trackers.rows(); ++i) {
        if (matched_trk_set.find(i) == matched_trk_set.end()) {
            unmatched_trks.push_back(i);
        }
    }
    
    Eigen::MatrixXi matches(filtered_matches.size(), 2);
    for (size_t i = 0; i < filtered_matches.size(); ++i) {
        matches(i, 0) = filtered_matches[i][0];
        matches(i, 1) = filtered_matches[i][1];
    }
    
    Eigen::VectorXi unmatched_dets_vec(unmatched_dets.size());
    for (size_t i = 0; i < unmatched_dets.size(); ++i) {
        unmatched_dets_vec(i) = unmatched_dets[i];
    }
    
    Eigen::VectorXi unmatched_trks_vec(unmatched_trks.size());
    for (size_t i = 0; i < unmatched_trks.size(); ++i) {
        unmatched_trks_vec(i) = unmatched_trks[i];
    }
    
    return {matches, unmatched_dets_vec, unmatched_trks_vec};
}

std::tuple<Eigen::MatrixXi, Eigen::VectorXi, Eigen::VectorXi>
HybridSort::associate_4_points_with_score_with_reid(
    const Eigen::MatrixXf& detections,
    const Eigen::MatrixXf& trackers,
    float iou_threshold,
    const Eigen::MatrixXf& /* velocities_lt */,
    const Eigen::MatrixXf& /* velocities_rt */,
    const Eigen::MatrixXf& /* velocities_lb */,
    const Eigen::MatrixXf& /* velocities_rb */,
    const Eigen::MatrixXf& /* k_observations */,
    float /* inertia */,
    const std::string& asso_func,
    const Eigen::MatrixXf& emb_cost,
    const std::pair<float, float>& weights,
    float /* thresh */,
    const Eigen::MatrixXf& /* long_emb_dists */,
    bool /* with_longterm_reid */,
    float /* longterm_reid_weight */,
    bool with_longterm_reid_correction,
    float /* longterm_reid_correction_thresh */,
        const std::string& /* dataset */
) const {
    // Simplified version with embedding cost
    Eigen::MatrixXf iou_matrix;
    
    if (asso_func == "hmiou") {
        iou_matrix = hmiou_batch(detections.leftCols(4), trackers.leftCols(4));
    } else {
        iou_matrix = iou_batch(detections.leftCols(4), trackers.leftCols(4));
    }
    
    // Combine IoU and embedding costs
    Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(iou_matrix.rows(), iou_matrix.cols());
    Eigen::MatrixXf cost_matrix = (ones - iou_matrix) * weights.first;
    if (emb_cost.rows() > 0 && emb_cost.cols() > 0) {
        cost_matrix += emb_cost * weights.second;
    }
    
    // Linear assignment - use a high cost threshold to allow more matches through
    // The actual filtering is done later based on IoU and embedding thresholds
    float max_cost = (1.0f - iou_threshold) * weights.first + weights.second;
    auto result = utils::linear_assignment(cost_matrix, max_cost);
    
    // Filter matches
    std::vector<std::array<int, 2>> filtered_matches;
    std::vector<int> unmatched_dets, unmatched_trks;
    
    std::unordered_set<int> matched_det_set, matched_trk_set;
    for (const auto& m : result.matches) {
        // emb_cost is a DISTANCE (1 - similarity), so LOWER values mean more similar
        bool valid = iou_matrix(m[0], m[1]) >= iou_threshold;
        if (with_longterm_reid_correction && emb_cost.rows() > 0 && emb_cost.cols() > 0) {
            // Accept match if: IoU is decent OR embedding similarity is high (distance is low)
            valid = valid || (iou_matrix(m[0], m[1]) >= iou_threshold / 2.0f && 
                              emb_cost(m[0], m[1]) <= 0.3f);  // similarity >= 0.7
        }
        if (valid) {
            filtered_matches.push_back(m);
            matched_det_set.insert(m[0]);
            matched_trk_set.insert(m[1]);
        } else {
            unmatched_dets.push_back(m[0]);
            unmatched_trks.push_back(m[1]);
        }
    }
    
    for (int i = 0; i < detections.rows(); ++i) {
        if (matched_det_set.find(i) == matched_det_set.end()) {
            unmatched_dets.push_back(i);
        }
    }
    
    for (int i = 0; i < trackers.rows(); ++i) {
        if (matched_trk_set.find(i) == matched_trk_set.end()) {
            unmatched_trks.push_back(i);
        }
    }
    
    Eigen::MatrixXi matches(filtered_matches.size(), 2);
    for (size_t i = 0; i < filtered_matches.size(); ++i) {
        matches(i, 0) = filtered_matches[i][0];
        matches(i, 1) = filtered_matches[i][1];
    }
    
    Eigen::VectorXi unmatched_dets_vec(unmatched_dets.size());
    for (size_t i = 0; i < unmatched_dets.size(); ++i) {
        unmatched_dets_vec(i) = unmatched_dets[i];
    }
    
    Eigen::VectorXi unmatched_trks_vec(unmatched_trks.size());
    for (size_t i = 0; i < unmatched_trks.size(); ++i) {
        unmatched_trks_vec(i) = unmatched_trks[i];
    }
    
    return {matches, unmatched_dets_vec, unmatched_trks_vec};
}

Eigen::MatrixXf HybridSort::update(
    const Eigen::MatrixXf& dets,
    const cv::Mat& img,
    const Eigen::MatrixXf& embs
) {
    check_inputs(dets, img);
    
    frame_count_++;
    
    int n_dets_full = dets.rows();
    if (n_dets_full == 0) {
        // Predict all tracks
        for (auto& trk : active_tracks_) {
            trk.predict();
        }
        
        // Remove dead tracks
        active_tracks_.erase(
            std::remove_if(active_tracks_.begin(), active_tracks_.end(),
                [this](const HybridKalmanBoxTracker& trk) {
                    return trk.time_since_update() > max_age_;
                }),
            active_tracks_.end()
        );
        
        return Eigen::MatrixXf(0, 8);
    }
    
    // Add detection indices
    Eigen::MatrixXf dets_idx(n_dets_full, 7);
    dets_idx.leftCols(6) = dets;
    for (int i = 0; i < n_dets_full; ++i) {
        dets_idx(i, 6) = static_cast<float>(i);
    }
    
    // Apply CMC
    Eigen::Matrix3f warp = Eigen::Matrix3f::Identity();
    if (cmc_ && dets_idx.rows() > 0) {
        auto warp_2x3 = cmc_->apply(img, dets_idx);
        warp.topRows<2>() = warp_2x3;
        for (auto& trk : active_tracks_) {
            trk.camera_update(warp);
        }
    }
    
    // Get ReID features
    Eigen::MatrixXf id_features;
    if (with_reid_) {
        if (embs.rows() > 0 && embs.cols() > 0) {
            id_features = embs;
        } else if (reid_model_ && dets_idx.rows() > 0) {
            id_features = reid_model_->get_features(dets_idx.leftCols(4), img);
        } else {
            id_features = Eigen::MatrixXf(n_dets_full, 128);
            id_features.setZero();
        }
    } else {
        id_features = Eigen::MatrixXf(n_dets_full, 1);
        id_features.setOnes();
    }
    
    // Split detections by confidence
    Eigen::VectorXf confs = dets.col(4);
    Eigen::Array<bool, Eigen::Dynamic, 1> inds_second = (confs.array() > low_thresh_) && (confs.array() < det_thresh_);
    Eigen::Array<bool, Eigen::Dynamic, 1> remain_inds = confs.array() > det_thresh_;
    
    Eigen::MatrixXf dets_second, dets_keep;
    Eigen::MatrixXf id_feature_second, id_feature_keep;
    Eigen::VectorXi det_inds_second, det_inds_keep;
    Eigen::VectorXi cls_second, cls_keep;
    
    // Extract high confidence detections
    int n_keep = remain_inds.cast<int>().sum();
    if (n_keep > 0) {
        dets_keep.resize(n_keep, 7);
        id_feature_keep.resize(n_keep, id_features.cols());
        det_inds_keep.resize(n_keep);
        cls_keep.resize(n_keep);
        
        int idx = 0;
        for (int i = 0; i < n_dets_full; ++i) {
            if (remain_inds(i)) {
                dets_keep.row(idx) = dets_idx.row(i);
                id_feature_keep.row(idx) = id_features.row(i);
                det_inds_keep(idx) = i;
                cls_keep(idx) = static_cast<int>(dets(i, 5));
                idx++;
            }
        }
    } else {
        dets_keep = Eigen::MatrixXf(0, 7);
        id_feature_keep = Eigen::MatrixXf(0, id_features.cols());
        det_inds_keep = Eigen::VectorXi(0);
        cls_keep = Eigen::VectorXi(0);
    }
    
    // Extract low confidence detections
    int n_second = inds_second.cast<int>().sum();
    if (n_second > 0) {
        dets_second.resize(n_second, 7);
        id_feature_second.resize(n_second, id_features.cols());
        det_inds_second.resize(n_second);
        cls_second.resize(n_second);
        
        int idx = 0;
        for (int i = 0; i < n_dets_full; ++i) {
            if (inds_second(i)) {
                dets_second.row(idx) = dets_idx.row(i);
                id_feature_second.row(idx) = id_features.row(i);
                det_inds_second(idx) = i;
                cls_second(idx) = static_cast<int>(dets(i, 5));
                idx++;
            }
        }
    } else {
        dets_second = Eigen::MatrixXf(0, 7);
        id_feature_second = Eigen::MatrixXf(0, id_features.cols());
        det_inds_second = Eigen::VectorXi(0);
        cls_second = Eigen::VectorXi(0);
    }
    
    // Predict all tracks
    Eigen::MatrixXf trks;
    if (!active_tracks_.empty()) {
        trks.resize(active_tracks_.size(), 6);
        for (size_t t = 0; t < active_tracks_.size(); ++t) {
            active_tracks_[t].predict();
            Eigen::Vector4f pos = active_tracks_[t].get_bbox();
            float kal_score = active_tracks_[t].get_kalman_score();
            float simple_score = active_tracks_[t].get_simple_score();
            trks(t, 0) = pos(0);
            trks(t, 1) = pos(1);
            trks(t, 2) = pos(2);
            trks(t, 3) = pos(3);
            trks(t, 4) = kal_score;
            trks(t, 5) = simple_score;
        }
    } else {
        trks = Eigen::MatrixXf(0, 6);
    }
    
    // Prepare motion cues
    Eigen::MatrixXf velocities_lt, velocities_rt, velocities_lb, velocities_rb;
    Eigen::MatrixXf last_boxes;
    Eigen::MatrixXf k_observations;
    
    if (!active_tracks_.empty()) {
        int n_trks = active_tracks_.size();
        velocities_lt.resize(n_trks, 2);
        velocities_rt.resize(n_trks, 2);
        velocities_lb.resize(n_trks, 2);
        velocities_rb.resize(n_trks, 2);
        last_boxes.resize(n_trks, 5);
        k_observations.resize(n_trks, 5);
        
        for (size_t i = 0; i < active_tracks_.size(); ++i) {
            velocities_lt.row(i) = active_tracks_[i].velocity_lt.transpose();
            velocities_rt.row(i) = active_tracks_[i].velocity_rt.transpose();
            velocities_lb.row(i) = active_tracks_[i].velocity_lb.transpose();
            velocities_rb.row(i) = active_tracks_[i].velocity_rb.transpose();
            last_boxes.row(i) = active_tracks_[i].last_observation().transpose();
            k_observations.row(i) = k_previous_obs(active_tracks_[i].observations, active_tracks_[i].age(), delta_t_).transpose();
        }
    }
    
    // First association
    Eigen::MatrixXi matched;
    Eigen::VectorXi unmatched_dets, unmatched_trks;
    
    if (TCM_first_step_ && dets_keep.rows() > 0 && trks.rows() > 0) {
        if (EG_weight_high_score_ > 0 && with_reid_) {
            // Get track features
            Eigen::MatrixXf track_features(active_tracks_.size(), id_feature_keep.cols());
            for (size_t i = 0; i < active_tracks_.size(); ++i) {
                Eigen::VectorXf feat = active_tracks_[i].smooth_feat();
                if (feat.size() == id_feature_keep.cols()) {
                    track_features.row(i) = feat.transpose();
                } else {
                    track_features.row(i).setZero();
                }
            }
            
            // Compute embedding distance
            Eigen::MatrixXf emb_dists = track_features * id_feature_keep.transpose();
            emb_dists = Eigen::MatrixXf::Ones(emb_dists.rows(), emb_dists.cols()) - emb_dists;  // Convert to distance
            
            std::tie(matched, unmatched_dets, unmatched_trks) = associate_4_points_with_score_with_reid(
                dets_keep.leftCols(5),
                trks,
                iou_threshold_,
                velocities_lt,
                velocities_rt,
                velocities_lb,
                velocities_rb,
                k_observations,
                inertia_,
                asso_func_name_,
                emb_dists,
                {1.0f, EG_weight_high_score_},
                high_score_matching_thresh_
            );
        } else {
            std::tie(matched, unmatched_dets, unmatched_trks) = associate_4_points_with_score(
                dets_keep.leftCols(5),
                trks,
                iou_threshold_,
                velocities_lt,
                velocities_rt,
                velocities_lb,
                velocities_rb,
                k_observations,
                inertia_,
                asso_func_name_
            );
        }
    } else {
        matched = Eigen::MatrixXi(0, 2);
        unmatched_dets.resize(dets_keep.rows());
        std::iota(unmatched_dets.data(), unmatched_dets.data() + unmatched_dets.size(), 0);
        unmatched_trks.resize(trks.rows());
        std::iota(unmatched_trks.data(), unmatched_trks.data() + unmatched_trks.size(), 0);
    }
    
    // Update matched tracks
    for (int i = 0; i < matched.rows(); ++i) {
        int det_i = matched(i, 0);
        int trk_i = matched(i, 1);
        if (det_i < dets_keep.rows() && trk_i < static_cast<int>(active_tracks_.size())) {
            int cls_val = (cls_keep.size() > det_i) ? cls_keep(det_i) : 0;
            int det_ind_val = (det_inds_keep.size() > det_i) ? det_inds_keep(det_i) : -1;
            active_tracks_[trk_i].update(
                dets_keep.row(det_i).head<5>().transpose(),
                id_feature_keep.row(det_i).transpose(),
                true,
                cls_val,
                det_ind_val
            );
        }
    }
    
    // BYTE association (low score)
    if (use_byte_ && dets_second.rows() > 0 && unmatched_trks.size() > 0) {
        Eigen::MatrixXf u_trks(unmatched_trks.size(), 6);
        for (int i = 0; i < unmatched_trks.size(); ++i) {
            u_trks.row(i) = trks.row(unmatched_trks(i));
        }
        
        Eigen::MatrixXf iou_left = iou_batch(dets_second.leftCols(4), u_trks.leftCols(4));
        
        if (TCM_byte_step_) {
            Eigen::MatrixXf score_diff = cal_score_dif_batch_two_score(dets_second.leftCols(5), u_trks);
            iou_left -= score_diff * TCM_byte_step_weight_;
        }
        
        if (iou_left.maxCoeff() > iou_threshold_) {
            Eigen::MatrixXf cost_matrix = Eigen::MatrixXf::Ones(iou_left.rows(), iou_left.cols()) - iou_left;
            
            if (EG_weight_low_score_ > 0 && with_reid_ && id_feature_second.rows() > 0) {
                // Add embedding cost
                // Note: iou_left is (dets_second.rows(), unmatched_trks.size())
                // We need emb_dists_low to have the same shape
                Eigen::MatrixXf u_track_features(unmatched_trks.size(), id_feature_second.cols());
                for (int i = 0; i < unmatched_trks.size(); ++i) {
                    Eigen::VectorXf feat = active_tracks_[unmatched_trks(i)].smooth_feat();
                    if (feat.size() == id_feature_second.cols()) {
                        u_track_features.row(i) = feat.transpose();
                    } else {
                        u_track_features.row(i).setZero();
                    }
                }
                // emb_dists_low = id_feature_second @ u_track_features.T -> (dets_second.rows(), unmatched_trks.size())
                Eigen::MatrixXf emb_dists_low = id_feature_second * u_track_features.transpose();
                emb_dists_low = Eigen::MatrixXf::Ones(emb_dists_low.rows(), emb_dists_low.cols()) - emb_dists_low;
                
                // Make sure dimensions match before adding
                if (emb_dists_low.rows() == cost_matrix.rows() && emb_dists_low.cols() == cost_matrix.cols()) {
                    cost_matrix += emb_dists_low * EG_weight_low_score_;
                }
            }
            
            // BYTE pass uses cost threshold (1 - IoU threshold)
            auto byte_result = utils::linear_assignment(cost_matrix, 1.0f - iou_threshold_);
            
            std::vector<int> to_remove_trk_indices;
            for (const auto& m : byte_result.matches) {
                int det_rel = m[0];
                int trk_rel = m[1];
                int trk_ind = unmatched_trks(trk_rel);
                
                if (iou_left(det_rel, trk_rel) >= iou_threshold_) {
                    int cls_val = (cls_second.size() > det_rel) ? cls_second(det_rel) : 0;
                    int det_ind_val = (det_inds_second.size() > det_rel) ? det_inds_second(det_rel) : -1;
                    active_tracks_[trk_ind].update(
                        dets_second.row(det_rel).head<5>().transpose(),
                        id_feature_second.row(det_rel).transpose(),
                        false,  // Don't update features in BYTE pass
                        cls_val,
                        det_ind_val
                    );
                    to_remove_trk_indices.push_back(trk_ind);
                }
            }
            
            // Remove matched tracks from unmatched list
            std::vector<int> new_unmatched_trks;
            std::unordered_set<int> remove_set(to_remove_trk_indices.begin(), to_remove_trk_indices.end());
            for (int i = 0; i < unmatched_trks.size(); ++i) {
                if (remove_set.find(unmatched_trks(i)) == remove_set.end()) {
                    new_unmatched_trks.push_back(unmatched_trks(i));
                }
            }
            unmatched_trks.resize(new_unmatched_trks.size());
            for (size_t i = 0; i < new_unmatched_trks.size(); ++i) {
                unmatched_trks(i) = new_unmatched_trks[i];
            }
        }
    }
    
    // Final chance: IoU vs last boxes
    if (unmatched_dets.size() > 0 && unmatched_trks.size() > 0) {
        Eigen::MatrixXf left_dets(unmatched_dets.size(), 4);
        Eigen::MatrixXf left_trks(unmatched_trks.size(), 4);
        
        for (int i = 0; i < unmatched_dets.size(); ++i) {
            left_dets.row(i) = dets_keep.row(unmatched_dets(i)).head<4>();
        }
        for (int i = 0; i < unmatched_trks.size(); ++i) {
            left_trks.row(i) = last_boxes.row(unmatched_trks(i)).head<4>();
        }
        
        Eigen::MatrixXf iou_left = iou_batch(left_dets, left_trks);
        if (iou_left.maxCoeff() > iou_threshold_) {
            Eigen::MatrixXf cost_matrix = Eigen::MatrixXf::Ones(iou_left.rows(), iou_left.cols()) - iou_left;
            // Rematch pass uses cost threshold (1 - IoU threshold)
            auto rematch_result = utils::linear_assignment(cost_matrix, 1.0f - iou_threshold_);
            
            std::vector<int> rematched_dets, rematched_trks;
            for (const auto& m : rematch_result.matches) {
                if (iou_left(m[0], m[1]) >= iou_threshold_) {
                    int det_abs = unmatched_dets(m[0]);
                    int trk_abs = unmatched_trks(m[1]);
                    int cls_val = (cls_keep.size() > det_abs) ? cls_keep(det_abs) : 0;
                    int det_ind_val = (det_inds_keep.size() > det_abs) ? det_inds_keep(det_abs) : -1;
                    active_tracks_[trk_abs].update(
                        dets_keep.row(det_abs).head<5>().transpose(),
                        id_feature_keep.row(det_abs).transpose(),
                        false,  // Don't update features
                        cls_val,
                        det_ind_val
                    );
                    rematched_dets.push_back(det_abs);
                    rematched_trks.push_back(trk_abs);
                }
            }
            
            // Update unmatched lists
            std::vector<int> new_unmatched_dets, new_unmatched_trks;
            std::unordered_set<int> rematched_det_set(rematched_dets.begin(), rematched_dets.end());
            std::unordered_set<int> rematched_trk_set(rematched_trks.begin(), rematched_trks.end());
            
            for (int i = 0; i < unmatched_dets.size(); ++i) {
                if (rematched_det_set.find(unmatched_dets(i)) == rematched_det_set.end()) {
                    new_unmatched_dets.push_back(unmatched_dets(i));
                }
            }
            for (int i = 0; i < unmatched_trks.size(); ++i) {
                if (rematched_trk_set.find(unmatched_trks(i)) == rematched_trk_set.end()) {
                    new_unmatched_trks.push_back(unmatched_trks(i));
                }
            }
            
            unmatched_dets.resize(new_unmatched_dets.size());
            unmatched_trks.resize(new_unmatched_trks.size());
            for (size_t i = 0; i < new_unmatched_dets.size(); ++i) {
                unmatched_dets(i) = new_unmatched_dets[i];
            }
            for (size_t i = 0; i < new_unmatched_trks.size(); ++i) {
                unmatched_trks(i) = new_unmatched_trks[i];
            }
        }
    }
    
    // Mark unmatched tracks
    for (int i = 0; i < unmatched_trks.size(); ++i) {
        int trk_idx = unmatched_trks(i);
        if (trk_idx < static_cast<int>(active_tracks_.size())) {
            Eigen::VectorXf empty_bbox, empty_feat;
            active_tracks_[trk_idx].update(empty_bbox, empty_feat, false);
        }
    }
    
    // Create new trackers for unmatched high-score detections
    for (int i = 0; i < unmatched_dets.size(); ++i) {
        int det_idx = unmatched_dets(i);
        if (det_idx < dets_keep.rows()) {
            int cls_val = (cls_keep.size() > det_idx) ? cls_keep(det_idx) : 0;
            int det_ind_val = (det_inds_keep.size() > det_idx) ? det_inds_keep(det_idx) : -1;
            active_tracks_.emplace_back(
                dets_keep.row(det_idx).head<5>().transpose(),
                id_feature_keep.row(det_idx).transpose(),
                delta_t_,
                use_custom_kf_,
                longterm_bank_length_,
                alpha_,
                adapfs_,
                track_thresh_,
                cls_val,
                det_ind_val
            );
        }
    }
    
    // Collect outputs
    std::vector<Eigen::VectorXf> outputs;
    for (auto it = active_tracks_.rbegin(); it != active_tracks_.rend(); ++it) {
        const auto& trk = *it;
        if (trk.time_since_update() < 1 && 
            (trk.hit_streak() >= min_hits_ || frame_count_ <= min_hits_)) {
            Eigen::Vector4f d = trk.get_bbox();
            Eigen::VectorXf output(8);
            output << d(0), d(1), d(2), d(3),
                     static_cast<float>(trk.id() + 1),
                     trk.conf(),
                     static_cast<float>(trk.cls()),
                     static_cast<float>(trk.det_ind());
            outputs.push_back(output);
        }
    }
    
    // Remove dead tracks
    active_tracks_.erase(
        std::remove_if(active_tracks_.begin(), active_tracks_.end(),
            [this](const HybridKalmanBoxTracker& trk) {
                return trk.time_since_update() > max_age_;
            }),
        active_tracks_.end()
    );
    
    if (outputs.empty()) {
        return Eigen::MatrixXf(0, 8);
    }
    
    Eigen::MatrixXf result(outputs.size(), 8);
    for (size_t i = 0; i < outputs.size(); ++i) {
        result.row(i) = outputs[i].transpose();
    }
    
    return result;
}

} // namespace motcpp::trackers
