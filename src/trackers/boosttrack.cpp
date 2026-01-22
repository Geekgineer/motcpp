// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/boosttrack.hpp>
#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/ops.hpp>
#include <motcpp/utils/iou.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace motcpp::trackers {

// Static members
int BoostTrack::next_id_ = 0;

int BoostTrack::next_id() {
    return ++next_id_;
}

// BoostKalmanFilter implementation
BoostKalmanFilter::BoostKalmanFilter(const Eigen::Vector4f& z) {
    // Initialize state: [x, y, h, r, vx, vy, vh, vr]
    x = Eigen::VectorXf::Zero(8);
    x.head<4>() = z;
    
    // Motion matrix: [I, dt*I; 0, I]
    motion_mat_ = Eigen::MatrixXf::Identity(8, 8);
    for (int i = 0; i < 4; ++i) {
        motion_mat_(i, i + 4) = 1.0f;  // dt = 1
    }
    
    // Update matrix: [I, 0]
    update_mat_ = Eigen::MatrixXf::Zero(4, 8);
    for (int i = 0; i < 4; ++i) {
        update_mat_(i, i) = 1.0f;
    }
    
    // Process noise Q
    process_noise_ = Eigen::MatrixXf::Identity(8, 8);
    process_noise_.block<4, 4>(0, 0) *= 10.0f;
    process_noise_.block<4, 4>(4, 4) *= 0.01f;
    
    // Measurement noise R
    measurement_noise_ = Eigen::MatrixXf::Identity(4, 4);
    measurement_noise_(0, 0) = 1.0f;
    measurement_noise_(1, 1) = 1.0f;
    measurement_noise_(2, 2) = 10.0f;
    measurement_noise_(3, 3) = 0.01f;
    
    // Initial covariance
    covariance = Eigen::MatrixXf::Identity(8, 8) * 10.0f;
    covariance.block<4, 4>(4, 4) *= 1000.0f;
}

void BoostKalmanFilter::predict() {
    x = motion_mat_ * x;
    covariance = motion_mat_ * covariance * motion_mat_.transpose() + process_noise_;
}

void BoostKalmanFilter::update(const Eigen::Vector4f& z) {
    // Project to measurement space
    Eigen::VectorXf projected_mean = update_mat_ * x;
    Eigen::MatrixXf projected_cov = update_mat_ * covariance * update_mat_.transpose() + measurement_noise_;
    
    // Kalman gain
    Eigen::MatrixXf K = covariance * update_mat_.transpose() * projected_cov.inverse();
    
    // Innovation
    Eigen::VectorXf innovation = z - projected_mean;
    
    // Update
    x = x + K * innovation;
    covariance = covariance - K * projected_cov * K.transpose();
}

void BoostKalmanFilter::camera_update(const Eigen::Matrix3f& transform) {
    // Get current bbox
    Eigen::Vector4f bbox = get_state();
    float x1 = bbox(0), y1 = bbox(1), x2 = bbox(2), y2 = bbox(3);
    
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
    float w = x2_new - x1_new;
    float h = y2_new - y1_new;
    float cx = x1_new + w / 2.0f;
    float cy = y1_new + h / 2.0f;
    float r = (h > 1e-6f) ? w / h : 0.0f;
    
    x(0) = cx;
    x(1) = cy;
    x(2) = h;
    x(3) = r;
}

Eigen::Vector4f BoostKalmanFilter::get_state() const {
    float cx = x(0);
    float cy = x(1);
    float h = x(2);
    float r = x(3);
    float w = r * h;
    
    return Eigen::Vector4f(cx - w/2, cy - h/2, cx + w/2, cy + h/2);
}

float BoostKalmanFilter::get_confidence(float coef) const {
    int n = 7;
    if (age < n) {
        return std::pow(coef, n - age);
    }
    return std::pow(coef, time_since_update - 1);
}

// BoostTrack implementation
namespace {
    Eigen::Vector4f convert_bbox_to_z(const Eigen::Vector4f& bbox) {
        float w = bbox(2) - bbox(0);
        float h = bbox(3) - bbox(1);
        float cx = bbox(0) + w / 2.0f;
        float cy = bbox(1) + h / 2.0f;
        float r = (h > 1e-6f) ? w / h : 0.0f;
        return Eigen::Vector4f(cx, cy, h, r);
    }
}

BoostTrack::BoostTrack(const Eigen::VectorXf& det, int max_obs, const Eigen::VectorXf& emb)
    : kf_(convert_bbox_to_z(Eigen::Vector4f(det(0), det(1), det(2), det(3))))
    , max_obs_(max_obs) {
    id_ = next_id();
    
    // det format: [x1, y1, x2, y2, conf, cls, det_ind]
    conf_ = det(4);
    cls_ = static_cast<int>(det(5));
    det_ind_ = (det.size() > 6) ? static_cast<int>(det(6)) : -1;
    
    if (emb.size() > 0) {
        emb_ = emb;
        // Normalize
        if (emb_.norm() > 0) {
            emb_ /= emb_.norm();
        }
    }
}

void BoostTrack::predict() {
    kf_.predict();
    kf_.age++;
    if (kf_.time_since_update > 0) {
        kf_.hit_streak = 0;
    }
    kf_.time_since_update++;
}

void BoostTrack::update(const Eigen::VectorXf& det) {
    kf_.time_since_update = 0;
    kf_.hit_streak++;
    
    Eigen::Vector4f bbox(det(0), det(1), det(2), det(3));
    Eigen::Vector4f z = convert_bbox_to_z(bbox);
    kf_.update(z);
    
    history_observations_.push_back(get_state());
    while (history_observations_.size() > static_cast<size_t>(max_obs_)) {
        history_observations_.pop_front();
    }
    
    conf_ = det(4);
    cls_ = static_cast<int>(det(5));
    det_ind_ = (det.size() > 6) ? static_cast<int>(det(6)) : -1;
}

void BoostTrack::update_emb(const Eigen::VectorXf& emb, float alpha) {
    if (emb.size() == 0) return;
    
    Eigen::VectorXf normalized_emb = emb;
    if (normalized_emb.norm() > 0) {
        normalized_emb /= normalized_emb.norm();
    }
    
    if (emb_.size() == 0) {
        emb_ = normalized_emb;
    } else {
        emb_ = alpha * emb_ + (1.0f - alpha) * normalized_emb;
        if (emb_.norm() > 0) {
            emb_ /= emb_.norm();
        }
    }
}

void BoostTrack::camera_update(const Eigen::Matrix3f& transform) {
    kf_.camera_update(transform);
}

Eigen::Vector4f BoostTrack::get_state() const {
    return kf_.get_state();
}

float BoostTrack::get_confidence() const {
    return kf_.get_confidence();
}

// BoostTrackTracker implementation
BoostTrackTracker::BoostTrackTracker(
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
    bool use_ecc,
    int min_box_area,
    float aspect_ratio_thresh,
    const std::string& cmc_method,
    float lambda_iou,
    float lambda_mhd,
    float lambda_shape,
    bool use_dlo_boost,
    bool use_duo_boost,
    float dlo_boost_coef,
    bool s_sim_corr,
    bool use_rich_s,
    bool use_sb,
    bool use_vt,
    bool with_reid
)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                  per_class, nr_classes, asso_func, is_obb)
    , use_ecc_(use_ecc)
    , min_box_area_(min_box_area)
    , aspect_ratio_thresh_(aspect_ratio_thresh)
    , cmc_method_(cmc_method)
    , lambda_iou_(lambda_iou)
    , lambda_mhd_(lambda_mhd)
    , lambda_shape_(lambda_shape)
    , use_dlo_boost_(use_dlo_boost)
    , use_duo_boost_(use_duo_boost)
    , dlo_boost_coef_(dlo_boost_coef)
    , s_sim_corr_(s_sim_corr)
    , use_rich_s_(use_rich_s)
    , use_sb_(use_sb)
    , use_vt_(use_vt)
    , with_reid_(with_reid)
{
    if (use_ecc_ && cmc_method_ == "ecc") {
        cmc_ = std::make_unique<motcpp::motion::ECC>();
    }
    
    if (with_reid_ && !reid_weights.empty()) {
        reid_model_ = std::make_unique<appearance::ONNXBackend>(reid_weights, "", use_half, use_gpu);
    }
    
    BoostTrack::next_id_ = 0;
}

void BoostTrackTracker::reset() {
    BaseTracker::reset();
    trackers_.clear();
    active_tracks_.clear();
    BoostTrack::next_id_ = 0;
}

Eigen::Vector4f BoostTrackTracker::convert_bbox_to_z(const Eigen::Vector4f& bbox) const {
    float w = bbox(2) - bbox(0);
    float h = bbox(3) - bbox(1);
    float cx = bbox(0) + w / 2.0f;
    float cy = bbox(1) + h / 2.0f;
    float r = (h > 1e-6f) ? w / h : 0.0f;
    return Eigen::Vector4f(cx, cy, h, r);
}

Eigen::Vector4f BoostTrackTracker::convert_x_to_bbox(const Eigen::VectorXf& x) const {
    float cx = x(0);
    float cy = x(1);
    float h = x(2);
    float r = x(3);
    float w = r * h;
    return Eigen::Vector4f(cx - w/2, cy - h/2, cx + w/2, cy + h/2);
}

Eigen::MatrixXf BoostTrackTracker::get_iou_matrix(const Eigen::MatrixXf& detections, bool /* buffered */) const {
    if (trackers_.empty() || detections.rows() == 0) {
        return Eigen::MatrixXf(0, 0);
    }
    
    // Compute IoU matrix manually
    Eigen::MatrixXf iou_matrix(detections.rows(), trackers_.size());
    
    for (int i = 0; i < detections.rows(); ++i) {
        Eigen::Vector4f det_bbox(detections(i, 0), detections(i, 1), 
                                 detections(i, 2), detections(i, 3));
        
        for (size_t j = 0; j < trackers_.size(); ++j) {
            Eigen::Vector4f trk_bbox = trackers_[j].get_state();
            
            // Compute IoU
            float x1 = std::max(det_bbox(0), trk_bbox(0));
            float y1 = std::max(det_bbox(1), trk_bbox(1));
            float x2 = std::min(det_bbox(2), trk_bbox(2));
            float y2 = std::min(det_bbox(3), trk_bbox(3));
            
            float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float det_area = (det_bbox(2) - det_bbox(0)) * (det_bbox(3) - det_bbox(1));
            float trk_area = (trk_bbox(2) - trk_bbox(0)) * (trk_bbox(3) - trk_bbox(1));
            float union_area = det_area + trk_area - inter_area;
            
            float iou = (union_area > 1e-6f) ? inter_area / union_area : 0.0f;
            iou_matrix(i, j) = 1.0f - iou;  // Convert to distance
        }
    }
    
    return iou_matrix;
}

Eigen::MatrixXf BoostTrackTracker::get_mh_dist_matrix(const Eigen::MatrixXf& detections, int n_dims) const {
    if (trackers_.empty() || detections.rows() == 0) {
        return Eigen::MatrixXf(0, 0);
    }
    
    int n_dets = detections.rows();
    int n_trks = trackers_.size();
    Eigen::MatrixXf mh_dist(n_dets, n_trks);
    
        for (int i = 0; i < n_dets; ++i) {
        Eigen::Vector4f bbox(detections(i, 0), detections(i, 1), detections(i, 2), detections(i, 3));
        Eigen::Vector4f z = convert_bbox_to_z(bbox);
        
        for (size_t j = 0; j < trackers_.size(); ++j) {
            const auto& trk = trackers_[j];
            const auto& kf = trk.kalman_filter();
            Eigen::VectorXf diff = z.head(n_dims) - kf.x.head(n_dims);
            
            // Get inverse covariance (diagonal approximation)
            Eigen::VectorXf sigma_inv = kf.covariance.diagonal().head(n_dims).cwiseInverse();
            
            // Mahalanobis distance
            float dist = (diff.array() * diff.array() * sigma_inv.array()).sum();
            mh_dist(i, j) = dist;
        }
    }
    
    return mh_dist;
}

Eigen::MatrixXf BoostTrackTracker::dlo_confidence_boost(const Eigen::MatrixXf& detections) {
    // Detection-level optimized confidence boost
    if (detections.rows() == 0 || trackers_.empty()) {
        return detections;
    }
    
    // Get tracker states
    Eigen::MatrixXf trk_boxes(trackers_.size(), 4);
    std::vector<int> time_since_update(trackers_.size());
    for (size_t i = 0; i < trackers_.size(); ++i) {
        Eigen::Vector4f state = trackers_[i].get_state();
        trk_boxes.row(i) = state.transpose();
        time_since_update[i] = trackers_[i].time_since_update() - 1;
    }
    
    // Compute IoU matrix between detections and trackers
    Eigen::MatrixXf det_boxes = detections.leftCols(4);
    Eigen::MatrixXf S = motcpp::utils::iou_batch(det_boxes, trk_boxes);
    
    if (S.rows() == 0 || S.cols() == 0) {
        return detections;
    }
    
    Eigen::MatrixXf boosted = detections;
    
    if (!use_sb_ && !use_vt_) {
        // Basic DLO boost: conf = max(conf, max_iou * dlo_boost_coef)
        Eigen::VectorXf max_s = S.rowwise().maxCoeff();
        for (int i = 0; i < boosted.rows(); ++i) {
            boosted(i, 4) = std::max(boosted(i, 4), max_s(i) * dlo_boost_coef_);
        }
        return boosted;
    }
    
    if (use_sb_) {
        // Soft-BIoU: conf = max(conf, alpha * conf + (1-alpha) * max_iou^1.5)
        float alpha = 0.65f;
        Eigen::VectorXf max_s = S.rowwise().maxCoeff();
        for (int i = 0; i < boosted.rows(); ++i) {
            float boosted_conf = alpha * boosted(i, 4) + (1.0f - alpha) * std::pow(max_s(i), 1.5f);
            boosted(i, 4) = std::max(boosted(i, 4), boosted_conf);
        }
    }
    
    if (use_vt_) {
        // Visual tracking: boost confidence if high IoU with recent tracks
        float threshold_s = 0.95f;
        float threshold_e = 0.8f;
        
        for (int i = 0; i < boosted.rows(); ++i) {
            bool should_boost = false;
            for (int j = 0; j < S.cols(); ++j) {
                float thresh = std::max(threshold_s - static_cast<float>(time_since_update[j]), threshold_e);
                if (S(i, j) > thresh) {
                    should_boost = true;
                    break;
                }
            }
            if (should_boost) {
                boosted(i, 4) = std::max(boosted(i, 4), det_thresh_ + 1e-5f);
            }
        }
    }
    
    return boosted;
}

Eigen::MatrixXf BoostTrackTracker::duo_confidence_boost(const Eigen::MatrixXf& detections) {
    // Detection-unmatched-object level confidence boost (similar to DLO but uses different matching)
    // For now, return unchanged as the main boost is done in dlo_confidence_boost
    return detections;
}

Eigen::MatrixXf BoostTrackTracker::filter_outputs(const Eigen::MatrixXf& outputs) const {
    if (outputs.rows() == 0) {
        return outputs;
    }
    
    Eigen::VectorXf w = outputs.col(2) - outputs.col(0);
    Eigen::VectorXf h = outputs.col(3) - outputs.col(1);
    Eigen::VectorXf area = w.array() * h.array();
    Eigen::VectorXf aspect_ratio = w.array() / (h.array() + 1e-6f);
    
    std::vector<int> keep_indices;
    for (int i = 0; i < outputs.rows(); ++i) {
        bool vertical_ok = aspect_ratio(i) <= aspect_ratio_thresh_;
        bool area_ok = area(i) > min_box_area_;
        if (vertical_ok && area_ok) {
            keep_indices.push_back(i);
        }
    }
    
    if (keep_indices.empty()) {
        return Eigen::MatrixXf(0, 8);
    }
    
    Eigen::MatrixXf filtered(keep_indices.size(), outputs.cols());
    for (size_t i = 0; i < keep_indices.size(); ++i) {
        filtered.row(i) = outputs.row(keep_indices[i]);
    }
    
    return filtered;
}

Eigen::MatrixXf BoostTrackTracker::update(
    const Eigen::MatrixXf& dets,
    const cv::Mat& img,
    const Eigen::MatrixXf& embs
) {
    check_inputs(dets, img);
    
    frame_count_++;
    
    // Add detection indices
    Eigen::MatrixXf dets_with_ind;
    if (dets.rows() > 0) {
        dets_with_ind.resize(dets.rows(), dets.cols() + 1);
        dets_with_ind.leftCols(dets.cols()) = dets;
        for (int i = 0; i < dets.rows(); ++i) {
            dets_with_ind(i, dets.cols()) = static_cast<float>(i);
        }
    } else {
        dets_with_ind = Eigen::MatrixXf(0, 7);
    }
    
    // Apply CMC if enabled
    if (cmc_ && dets_with_ind.rows() > 0) {
        auto warp_2x3 = cmc_->apply(img, dets_with_ind);
        Eigen::Matrix3f warp = Eigen::Matrix3f::Identity();
        warp.topRows<2>() = warp_2x3;
        
        for (auto& trk : trackers_) {
            trk.camera_update(warp);
        }
    }
    
    // Predict all trackers
    Eigen::MatrixXf trks;
    Eigen::VectorXf confs;
    if (!trackers_.empty()) {
        trks.resize(trackers_.size(), 5);
        confs.resize(trackers_.size());
        
        for (size_t i = 0; i < trackers_.size(); ++i) {
            trackers_[i].predict();
            Eigen::Vector4f pos = trackers_[i].get_state();
            float conf = trackers_[i].get_confidence();
            confs(i) = conf;
            trks(i, 0) = pos(0);
            trks(i, 1) = pos(1);
            trks(i, 2) = pos(2);
            trks(i, 3) = pos(3);
            trks(i, 4) = conf;
        }
    } else {
        trks = Eigen::MatrixXf(0, 5);
        confs = Eigen::VectorXf(0);
    }
    
    // Apply confidence boosts (simplified for now)
    Eigen::MatrixXf boosted_dets = dets_with_ind;
    if (use_dlo_boost_ && dets_with_ind.rows() > 0) {
        boosted_dets = dlo_confidence_boost(boosted_dets);
    }
    if (use_duo_boost_ && dets_with_ind.rows() > 0) {
        boosted_dets = duo_confidence_boost(boosted_dets);
    }
    
    // Filter detections by threshold
    Eigen::MatrixXf filtered_dets;
    Eigen::MatrixXf dets_embs;
    if (boosted_dets.rows() > 0) {
        std::vector<int> keep_indices;
        for (int i = 0; i < boosted_dets.rows(); ++i) {
            if (boosted_dets(i, 4) >= det_thresh_) {
                keep_indices.push_back(i);
            }
        }
        
        if (!keep_indices.empty()) {
            filtered_dets.resize(keep_indices.size(), boosted_dets.cols());
            for (size_t i = 0; i < keep_indices.size(); ++i) {
                filtered_dets.row(i) = boosted_dets.row(keep_indices[i]);
            }
            
            // Get embeddings
            if (with_reid_) {
                if (embs.rows() > 0 && embs.cols() > 0) {
                    dets_embs.resize(keep_indices.size(), embs.cols());
                    for (size_t i = 0; i < keep_indices.size(); ++i) {
                        dets_embs.row(i) = embs.row(keep_indices[i]);
                    }
                } else if (reid_model_) {
                    Eigen::MatrixXf dets_bbox = filtered_dets.leftCols(4);
                    dets_embs = reid_model_->get_features(dets_bbox, img);
                }
            }
        } else {
            filtered_dets = Eigen::MatrixXf(0, boosted_dets.cols());
            dets_embs = Eigen::MatrixXf(0, 0);
        }
    } else {
        filtered_dets = Eigen::MatrixXf(0, boosted_dets.cols());
        dets_embs = Eigen::MatrixXf(0, 0);
    }
    
    // Association
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_dets, unmatched_trks;
    
    if (filtered_dets.rows() > 0 && trks.rows() > 0) {
        // Compute IoU matrix
        Eigen::MatrixXf iou_matrix = get_iou_matrix(filtered_dets);
        
        // Compute Mahalanobis distance matrix
        Eigen::MatrixXf mh_dist_matrix = get_mh_dist_matrix(filtered_dets);
        
        // Compute embedding cost if using ReID
        Eigen::MatrixXf emb_cost;
        if (with_reid_ && dets_embs.rows() > 0 && !trackers_.empty()) {
            Eigen::MatrixXf tracker_embs(trackers_.size(), dets_embs.cols());
            for (size_t i = 0; i < trackers_.size(); ++i) {
                Eigen::VectorXf trk_emb = trackers_[i].get_emb();
                if (trk_emb.size() == dets_embs.cols()) {
                    tracker_embs.row(i) = trk_emb.transpose();
                } else {
                    tracker_embs.row(i).setZero();
                }
            }
            emb_cost = dets_embs * tracker_embs.transpose();
        }
        
        // Build cost matrix (lower = better match)
        // iou_matrix is already a distance (1 - IoU)
        // mh_sim and emb_cost are similarities (higher = better), so we SUBTRACT them
        Eigen::MatrixXf cost_matrix = iou_matrix;
        if (mh_dist_matrix.rows() > 0 && mh_dist_matrix.cols() > 0) {
            // Normalize Mahalanobis distance to similarity (limit - dist)
            Eigen::MatrixXf mh_sim = mh_dist_matrix;
            float limit = 13.2767f;  // 99% confidence interval
            for (int i = 0; i < mh_sim.rows(); ++i) {
                for (int j = 0; j < mh_sim.cols(); ++j) {
                    if (mh_sim(i, j) > limit) mh_sim(i, j) = limit;
                    mh_sim(i, j) = (limit - mh_sim(i, j)) / limit;  // Normalize to [0,1]
                }
            }
            cost_matrix -= lambda_mhd_ * mh_sim;  // Subtract similarity
        }
        
        if (emb_cost.rows() > 0 && emb_cost.cols() > 0) {
            // emb_cost is cosine similarity in [-1, 1], normalize to [0, 1]
            float lambda_emb = (1.0f + lambda_iou_ + lambda_shape_ + lambda_mhd_) * 1.5f;
            Eigen::MatrixXf emb_sim = (emb_cost.array() + 1.0f) / 2.0f;  // Map to [0, 1]
            cost_matrix -= lambda_emb * emb_sim;  // Subtract similarity
        }
        
        // Linear assignment
        // Cost matrix shape: (detections, trackers) -> rows=dets, cols=trks
        auto result = utils::linear_assignment(cost_matrix, iou_threshold_);
        for (const auto& m : result.matches) {
            matches.emplace_back(m[0], m[1]);  // (det_idx, trk_idx)
        }
        unmatched_dets = result.unmatched_a;  // unmatched rows = detections
        unmatched_trks = result.unmatched_b;  // unmatched cols = trackers
    } else if (filtered_dets.rows() > 0) {
        // All detections unmatched
        for (int i = 0; i < filtered_dets.rows(); ++i) {
            unmatched_dets.push_back(i);
        }
    } else if (trks.rows() > 0) {
        // All trackers unmatched
        for (int i = 0; i < trks.rows(); ++i) {
            unmatched_trks.push_back(i);
        }
    }
    
    // Update matched trackers
    Eigen::VectorXf scores = filtered_dets.col(4);
    Eigen::VectorXf trust = (scores.array() - det_thresh_) / (1.0f - det_thresh_);
    float af = 0.95f;
    Eigen::VectorXf dets_alpha = af + (1.0f - af) * (1.0f - trust.array());
    
    for (const auto& [idet, itrk] : matches) {
        if (idet < filtered_dets.rows() && itrk < static_cast<int>(trackers_.size())) {
            trackers_[itrk].update(filtered_dets.row(idet).transpose());
            if (dets_embs.rows() > idet && dets_embs.cols() > 0) {
                trackers_[itrk].update_emb(dets_embs.row(idet).transpose(), dets_alpha(idet));
            }
        }
    }
    
    // Create new trackers for unmatched detections
    for (int idet : unmatched_dets) {
        if (idet < filtered_dets.rows() && filtered_dets(idet, 4) >= det_thresh_) {
            Eigen::VectorXf emb;
            if (dets_embs.rows() > idet && dets_embs.cols() > 0) {
                emb = dets_embs.row(idet).transpose();
            }
            trackers_.emplace_back(filtered_dets.row(idet).transpose(), max_obs_, emb);
        }
    }
    
    // Prepare output
    std::vector<Eigen::VectorXf> outputs;
    active_tracks_.clear();
    
    for (auto& trk : trackers_) {
        if (trk.time_since_update() < 1 && 
            (trk.hit_streak() >= min_hits_ || frame_count_ <= min_hits_)) {
            Eigen::Vector4f bbox = trk.get_state();
            Eigen::VectorXf output(8);
            output << bbox(0), bbox(1), bbox(2), bbox(3),
                     static_cast<float>(trk.id()),
                     trk.conf(),
                     static_cast<float>(trk.cls()),
                     static_cast<float>(trk.det_ind());
            outputs.push_back(output);
            active_tracks_.push_back(&trk);
        }
    }
    
    // Remove old trackers
    trackers_.erase(
        std::remove_if(trackers_.begin(), trackers_.end(),
            [this](const BoostTrack& trk) { return trk.time_since_update() > max_age_; }),
        trackers_.end()
    );
    
    if (outputs.empty()) {
        return Eigen::MatrixXf(0, 8);
    }
    
    Eigen::MatrixXf result(outputs.size(), 8);
    for (size_t i = 0; i < outputs.size(); ++i) {
        result.row(i) = outputs[i].transpose();
    }
    
    return filter_outputs(result);
}

} // namespace motcpp::trackers
