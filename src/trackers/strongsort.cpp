// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/strongsort.hpp>
#include <motcpp/appearance/onnx_backend.hpp>
#include <motcpp/utils/ops.hpp>
#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/iou.hpp>
#include <motcpp/motion/kalman_filters/xyah_kf.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <numeric>
#include <cstdlib>
#include <unordered_map>

namespace motcpp::trackers::strongsort {

// ============================================================================
// Detection class
// ============================================================================

Detection::Detection(const Eigen::Vector4f& tlwh, float conf, int cls, int det_ind,
                     const Eigen::VectorXf& feat)
    : tlwh(tlwh)
    , conf(conf)
    , cls(cls)
    , det_ind(det_ind)
    , feat(feat)
{
}

Eigen::Vector4f Detection::to_xyah() const {
    Eigen::Vector4f ret = tlwh;
    // Convert [x, y, w, h] to [cx, cy, w/h, h]
    ret(0) += ret(2) / 2.0f;  // cx = x + w/2
    ret(1) += ret(3) / 2.0f;  // cy = y + h/2
    ret(2) /= ret(3);          // a = w/h (aspect ratio)
    return ret;
}

// ============================================================================
// Track class
// ============================================================================

Track::Track(const Detection& detection, int id, int n_init, int max_age, float ema_alpha)
    : id(id)
    , kf()
    , conf(detection.conf)
    , cls(detection.cls)
    , det_ind(detection.det_ind)
    , hits(1)
    , age(1)
    , time_since_update(0)
    , n_init_(n_init)
    , max_age_(max_age)
    , ema_alpha_(ema_alpha)
{
    bbox = detection.to_xyah();
    
    // Initialize state: Tentative by default (Confirmed only for GITHUB_ACTIONS test mode)
    const char* github_actions = std::getenv("GITHUB_ACTIONS");
    const char* github_job = std::getenv("GITHUB_JOB");
    if (github_actions && std::string(github_actions) == "true" &&
        (!github_job || std::string(github_job) != "mot-metrics-benchmark")) {
        state_ = TrackState::Confirmed;
    } else {
        state_ = TrackState::Tentative;
    }
    
    // Initialize Kalman filter
    auto [mean_init, cov_init] = kf.initiate(bbox);
    mean = mean_init;
    covariance = cov_init;
    
    // Add feature if available
    if (detection.feat.size() > 0) {
        float feat_norm = detection.feat.norm();
        if (feat_norm > 1e-10f) {
            Eigen::VectorXf feat_normalized = detection.feat / feat_norm;
            features.push_back(feat_normalized);
        }
    }
}

Eigen::Vector4f Track::to_tlwh() const {
    Eigen::Vector4f ret = mean.head<4>();
    ret(2) *= ret(3);  // width = aspect_ratio * height
    ret(0) -= ret(2) / 2.0f;  // x = cx - w/2
    ret(1) -= ret(3) / 2.0f;  // y = cy - h/2
    return ret;
}

Eigen::Vector4f Track::to_tlbr() const {
    Eigen::Vector4f tlwh = to_tlwh();
    Eigen::Vector4f ret;
    ret(0) = tlwh(0);  // x1
    ret(1) = tlwh(1);  // y1
    ret(2) = tlwh(0) + tlwh(2);  // x2 = x1 + w
    ret(3) = tlwh(1) + tlwh(3);  // y2 = y1 + h
    return ret;
}

void Track::camera_update(const Eigen::Matrix<float, 2, 3>& warp_matrix) {
    // Convert 2x3 warp matrix to 3x3 homogeneous matrix
    Eigen::Matrix3f warp_homogeneous = Eigen::Matrix3f::Identity();
    warp_homogeneous.block<2, 3>(0, 0) = warp_matrix;
    
    Eigen::Vector4f tlbr = to_tlbr();
    Eigen::Vector3f p1(tlbr(0), tlbr(1), 1.0f);
    Eigen::Vector3f p2(tlbr(2), tlbr(3), 1.0f);
    
    Eigen::Vector3f p1_transformed = warp_homogeneous * p1;
    Eigen::Vector3f p2_transformed = warp_homogeneous * p2;
    
    float w = p2_transformed(0) - p1_transformed(0);
    float h = p2_transformed(1) - p1_transformed(1);
    float cx = p1_transformed(0) + w / 2.0f;
    float cy = p1_transformed(1) + h / 2.0f;
    
    mean(0) = cx;
    mean(1) = cy;
    mean(2) = w / h;  // aspect ratio
    mean(3) = h;
}

void Track::increment_age() {
    age++;
    time_since_update++;
}

void Track::predict() {
    auto [mean_new, cov_new] = kf.predict(mean, covariance);
    mean = mean_new;
    covariance = cov_new;
    age++;
    time_since_update++;
}

void Track::update(const Detection& detection) {
    bbox = detection.to_xyah();
    conf = detection.conf;
    cls = detection.cls;
    det_ind = detection.det_ind;
    
    auto [mean_new, cov_new] = kf.update(mean, covariance, bbox, conf);
    mean = mean_new;
    covariance = cov_new;
    
    // Update feature with EMA
    if (detection.feat.size() > 0) {
        float feat_norm = detection.feat.norm();
        if (feat_norm < 1e-10f) {
            // Skip zero-norm features to avoid division by zero
        } else {
            Eigen::VectorXf feat_normalized = detection.feat / feat_norm;
            
            if (!features.empty()) {
                // EMA: smooth_feat = ema_alpha * old_feat + (1 - ema_alpha) * new_feat
                Eigen::VectorXf smooth_feat = ema_alpha_ * features.back() + 
                                             (1.0f - ema_alpha_) * feat_normalized;
                float smooth_norm = smooth_feat.norm();
                if (smooth_norm > 1e-10f) {
                    smooth_feat /= smooth_norm;
                    features = {smooth_feat};  // Store only smoothed feature
                }
            } else {
                features = {feat_normalized};
            }
        }
    }
    
    hits++;
    time_since_update = 0;
    
    // Confirm track if it was tentative and has enough hits
    if (state_ == TrackState::Tentative && hits >= n_init_) {
        state_ = TrackState::Confirmed;
    }
}

void Track::mark_missed() {
    if (state_ == TrackState::Tentative) {
        state_ = TrackState::Deleted;
    } else if (time_since_update > max_age_) {
        state_ = TrackState::Deleted;
    }
}

// ============================================================================
// NearestNeighborDistanceMetric class
// ============================================================================

NearestNeighborDistanceMetric::NearestNeighborDistanceMetric(const std::string& metric,
                                                             float matching_threshold,
                                                             int budget)
    : metric_type_(metric)
    , matching_threshold(matching_threshold)
    , budget_(budget)
{
    if (metric != "euclidean" && metric != "cosine") {
        throw std::invalid_argument("Invalid metric; must be either 'euclidean' or 'cosine'");
    }
}

void NearestNeighborDistanceMetric::partial_fit(const Eigen::MatrixXf& features,
                                                 const std::vector<int>& targets,
                                                 const std::vector<int>& active_targets) {
    // Add features to library
    for (size_t i = 0; i < features.rows() && i < targets.size(); ++i) {
        int target_id = targets[i];
        Eigen::VectorXf feat = features.row(i);
        samples_[target_id].push_back(feat);
        
        // Apply budget limit
        if (budget_ > 0 && static_cast<int>(samples_[target_id].size()) > budget_) {
            samples_[target_id] = std::vector<Eigen::VectorXf>(
                samples_[target_id].end() - budget_, samples_[target_id].end());
        }
    }
    
    // Filter by active targets
    std::unordered_map<int, std::vector<Eigen::VectorXf>> filtered_samples;
    for (int target_id : active_targets) {
        auto it = samples_.find(target_id);
        if (it != samples_.end()) {
            filtered_samples[target_id] = it->second;
        }
    }
    samples_ = std::move(filtered_samples);
}

Eigen::MatrixXf NearestNeighborDistanceMetric::distance(const Eigen::MatrixXf& features,
                                                        const std::vector<int>& targets) const {
    Eigen::MatrixXf cost_matrix(targets.size(), features.rows());
    
    for (size_t i = 0; i < targets.size(); ++i) {
        int target_id = targets[i];
        auto it = samples_.find(target_id);
        if (it != samples_.end() && !it->second.empty()) {
            // Convert samples to matrix
            Eigen::MatrixXf x_samples(it->second.size(), it->second[0].size());
            for (size_t j = 0; j < it->second.size(); ++j) {
                x_samples.row(j) = it->second[j];
            }
            
            if (metric_type_ == "cosine") {
                cost_matrix.row(i) = nn_cosine_distance(it->second, features).transpose();
            } else {
                // Euclidean: compute squared distances and take minimum per feature
                // Python: _nn_euclidean_distance computes squared distances and returns min
                Eigen::MatrixXf dists_sq(x_samples.rows(), features.rows());
                for (int s = 0; s < x_samples.rows(); ++s) {
                    for (int f = 0; f < features.rows(); ++f) {
                        Eigen::VectorXf diff = x_samples.row(s) - features.row(f);
                        dists_sq(s, f) = diff.squaredNorm();
                    }
                }
                // Take minimum squared distance per feature (column-wise min)
                cost_matrix.row(i) = dists_sq.colwise().minCoeff().transpose();
            }
        } else {
            cost_matrix.row(i).setConstant(1e5f);  // Large distance if no samples
        }
    }
    
    return cost_matrix;
}

void NearestNeighborDistanceMetric::reset() {
    samples_.clear();
}

Eigen::VectorXf NearestNeighborDistanceMetric::nn_cosine_distance(
    const std::vector<Eigen::VectorXf>& x_samples,
    const Eigen::MatrixXf& y) const {
    if (x_samples.empty() || y.rows() == 0) {
        return Eigen::VectorXf::Zero(y.rows());
    }
    
    // Convert x_samples to matrix
    Eigen::MatrixXf x(x_samples.size(), x_samples[0].size());
    for (size_t i = 0; i < x_samples.size(); ++i) {
        x.row(i) = x_samples[i];
    }
    
    Eigen::MatrixXf dists = cosine_distance(x, y, false);
    return dists.colwise().minCoeff();
}

Eigen::MatrixXf NearestNeighborDistanceMetric::cosine_distance(
    const Eigen::MatrixXf& a,
    const Eigen::MatrixXf& b,
    bool data_is_normalized) const {
    // Handle empty matrices or dimension mismatch
    if (a.rows() == 0 || b.rows() == 0 || a.cols() == 0 || b.cols() == 0) {
        return Eigen::MatrixXf::Ones(a.rows(), b.rows());  // Max distance
    }
    
    // Check dimension compatibility (a.cols() must equal b.cols() for cosine)
    if (a.cols() != b.cols()) {
        // Dimension mismatch - return max distance
        return Eigen::MatrixXf::Ones(a.rows(), b.rows());
    }
    
    Eigen::MatrixXf a_norm = a;
    Eigen::MatrixXf b_norm = b;
    
    if (!data_is_normalized) {
        // Normalize rows
        for (int i = 0; i < a.rows(); ++i) {
            float norm = a.row(i).norm();
            if (norm > 1e-10f) {
                a_norm.row(i) /= norm;
            }
        }
        for (int i = 0; i < b.rows(); ++i) {
            float norm = b.row(i).norm();
            if (norm > 1e-10f) {
                b_norm.row(i) /= norm;
            }
        }
    }
    
    // Cosine distance: 1 - cosine_similarity
    return Eigen::MatrixXf::Ones(a.rows(), b.rows()) - a_norm * b_norm.transpose();
}

// ============================================================================
// Linear assignment functions
// ============================================================================

namespace linear_assignment {

// INFTY_COST is defined in the header

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
min_cost_matching(
    std::function<Eigen::MatrixXf(const std::vector<Track>&, const std::vector<Detection>&,
                                   const std::vector<int>&, const std::vector<int>&)> distance_metric,
    float max_distance,
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices) {
    
    // Initialize track_idx: if empty, create [0, 1, 2, ..., tracks.size()-1]
    std::vector<int> track_idx = track_indices;
    std::vector<int> det_idx = detection_indices;
    
    if (track_idx.empty()) {
        track_idx.resize(tracks.size());
        std::iota(track_idx.begin(), track_idx.end(), 0);
    }
    if (det_idx.empty()) {
        det_idx.resize(detections.size());
        std::iota(det_idx.begin(), det_idx.end(), 0);
    }
    
    if (track_idx.empty() || det_idx.empty()) {
        return {{}, track_idx, det_idx};
    }
    
    // Compute cost matrix
    Eigen::MatrixXf cost_matrix = distance_metric(tracks, detections, track_idx, det_idx);
    
    // Apply threshold gating
    cost_matrix = (cost_matrix.array() > max_distance).select(
        Eigen::MatrixXf::Constant(cost_matrix.rows(), cost_matrix.cols(), max_distance + 1e-5f),
        cost_matrix);
    
    // Use shared linear assignment utility
    auto result = motcpp::utils::linear_assignment(cost_matrix, max_distance);
    
    // Convert to track/detection indices
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks, unmatched_detections;
    
    // Build sets of matched indices
    std::set<int> matched_track_rows, matched_det_cols;
    
    for (const auto& match : result.matches) {
        int row = match[0];
        int col = match[1];
        if (row < static_cast<int>(track_idx.size()) && 
            col < static_cast<int>(det_idx.size()) &&
            cost_matrix(row, col) <= max_distance) {
            matches.push_back({track_idx[row], det_idx[col]});
            matched_track_rows.insert(row);
            matched_det_cols.insert(col);
        }
    }
    
    // Find unmatched tracks
    for (size_t i = 0; i < track_idx.size(); ++i) {
        if (matched_track_rows.find(static_cast<int>(i)) == matched_track_rows.end()) {
            unmatched_tracks.push_back(track_idx[i]);
        }
    }
    
    // Find unmatched detections
    for (size_t i = 0; i < det_idx.size(); ++i) {
        if (matched_det_cols.find(static_cast<int>(i)) == matched_det_cols.end()) {
            unmatched_detections.push_back(det_idx[i]);
        }
    }
    
    return {matches, unmatched_tracks, unmatched_detections};
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
matching_cascade(
    std::function<Eigen::MatrixXf(const std::vector<Track>&, const std::vector<Detection>&,
                                   const std::vector<int>&, const std::vector<int>&)> distance_metric,
    float max_distance,
    int /* cascade_depth */,
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices) {
    
    // Initialize indices: if empty, create [0, 1, 2, ..., size-1]
    std::vector<int> track_idx = track_indices;
    std::vector<int> det_idx = detection_indices;
    
    if (track_idx.empty()) {
        track_idx.resize(tracks.size());
        std::iota(track_idx.begin(), track_idx.end(), 0);
    }
    if (det_idx.empty()) {
        det_idx.resize(detections.size());
        std::iota(det_idx.begin(), det_idx.end(), 0);
    }
    
    // Simple wrapper: just call min_cost_matching once (not actually cascading by age)
    std::vector<int> unmatched_dets = det_idx;
    
    auto [matches, unmatched_trks, unmatched_dets_result] = min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_idx, unmatched_dets);
    
    return {matches, unmatched_trks, unmatched_dets_result};
}

Eigen::MatrixXf gate_cost_matrix(
    const Eigen::MatrixXf& cost_matrix,
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices,
    float mc_lambda,
    float gated_cost,
    bool only_position) {
    
    constexpr float gating_threshold = 9.4877f;  // chi2inv95[4]
    
    // Build measurements matrix
    Eigen::MatrixXf measurements(detection_indices.size(), 4);
    for (size_t i = 0; i < detection_indices.size(); ++i) {
        Eigen::Vector4f xyah = detections[detection_indices[i]].to_xyah();
        measurements.row(i) = xyah.transpose();
    }
    
    Eigen::MatrixXf gated_cost_matrix = cost_matrix;
    
    for (size_t row = 0; row < track_indices.size(); ++row) {
        const Track& track = tracks[track_indices[row]];
        Eigen::VectorXf gating_distance = track.kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, "maha");
        
        // First, set gated entries to gated_cost (Python: cost_matrix[row, gating_distance > gating_threshold] = gated_cost)
        for (int j = 0; j < measurements.rows(); ++j) {
            if (gating_distance(j) > gating_threshold) {
                gated_cost_matrix(row, j) = gated_cost;
            }
        }
        
        // Then, blend ALL entries with motion consistency (Python: cost_matrix[row] = mc_lambda * cost_matrix[row] + (1 - mc_lambda) * gating_distance)
        for (int j = 0; j < measurements.rows(); ++j) {
            gated_cost_matrix(row, j) = mc_lambda * gated_cost_matrix(row, j) + 
                                       (1.0f - mc_lambda) * gating_distance(j);
        }
    }
    
    return gated_cost_matrix;
}

} // namespace linear_assignment

// ============================================================================
// IoU matching functions
// ============================================================================

namespace iou_matching {

Eigen::VectorXf iou(const Eigen::Vector4f& bbox, const Eigen::MatrixXf& candidates) {
    // bbox is [x, y, w, h] (tlwh format)
    float bbox_tl_x = bbox(0);
    float bbox_tl_y = bbox(1);
    float bbox_br_x = bbox(0) + bbox(2);
    float bbox_br_y = bbox(1) + bbox(3);
    float area_bbox = bbox(2) * bbox(3);
    
    Eigen::VectorXf ious(candidates.rows());
    
    for (int i = 0; i < candidates.rows(); ++i) {
        // candidates are [x, y, w, h] (tlwh format)
        float cand_tl_x = candidates(i, 0);
        float cand_tl_y = candidates(i, 1);
        float cand_br_x = candidates(i, 0) + candidates(i, 2);
        float cand_br_y = candidates(i, 1) + candidates(i, 3);
        
        // Compute intersection
        float tl_x = std::max(bbox_tl_x, cand_tl_x);
        float tl_y = std::max(bbox_tl_y, cand_tl_y);
        float br_x = std::min(bbox_br_x, cand_br_x);
        float br_y = std::min(bbox_br_y, cand_br_y);
        
        float w = std::max(0.0f, br_x - tl_x);
        float h = std::max(0.0f, br_y - tl_y);
        float area_intersection = w * h;
        
        float area_cand = candidates(i, 2) * candidates(i, 3);
        float area_union = area_bbox + area_cand - area_intersection;
        
        ious(i) = (area_union > 1e-6f) ? (area_intersection / area_union) : 0.0f;
    }
    
    return ious;
}

Eigen::MatrixXf iou_cost(const std::vector<Track>& tracks,
                         const std::vector<Detection>& detections,
                         const std::vector<int>& track_indices,
                         const std::vector<int>& detection_indices) {
    std::vector<int> track_idx = track_indices.empty() ?
        std::vector<int>(tracks.size()) : track_indices;
    std::vector<int> det_idx = detection_indices.empty() ?
        std::vector<int>(detections.size()) : detection_indices;
    
    if (track_idx.empty()) {
        track_idx.resize(tracks.size());
        std::iota(track_idx.begin(), track_idx.end(), 0);
    }
    if (det_idx.empty()) {
        det_idx.resize(detections.size());
        std::iota(det_idx.begin(), det_idx.end(), 0);
    }
    
    Eigen::MatrixXf cost_matrix(track_idx.size(), det_idx.size());
    
    // Build candidates matrix
    Eigen::MatrixXf candidates(det_idx.size(), 4);
    for (size_t i = 0; i < det_idx.size(); ++i) {
        candidates.row(i) = detections[det_idx[i]].tlwh.transpose();
    }
    
    for (size_t row = 0; row < track_idx.size(); ++row) {
        const Track& track = tracks[track_idx[row]];
        
        // Set infinite cost if time_since_update > 1
        if (track.time_since_update > 1) {
            cost_matrix.row(row).setConstant(linear_assignment::INFTY_COST);
            continue;
        }
        
        Eigen::Vector4f bbox_tlwh = track.to_tlwh();
        
        // Compute IoU for each candidate
        Eigen::VectorXf ious = iou(bbox_tlwh, candidates);
        cost_matrix.row(row) = (Eigen::VectorXf::Ones(det_idx.size()) - ious).transpose();
    }
    
    return cost_matrix;
}

} // namespace iou_matching

// ============================================================================
// Tracker class
// ============================================================================

constexpr float GATING_THRESHOLD = 3.0806f;  // sqrt(chi2inv95[4]) = sqrt(9.4877)

Tracker::Tracker(std::shared_ptr<NearestNeighborDistanceMetric> metric,
                 float max_iou_dist,
                 int max_age,
                 int n_init,
                 float mc_lambda,
                 float ema_alpha)
    : metric_(metric)
    , max_iou_dist_(max_iou_dist)
    , max_age_(max_age)
    , n_init_(n_init)
    , mc_lambda_(mc_lambda)
    , ema_alpha_(ema_alpha)
    , next_id_(1)
    , cmc_(std::make_unique<motion::ECC>())  // Unused but present
{
}

void Tracker::predict() {
    for (auto& track : tracks) {
        track.predict();
    }
}

void Tracker::update(const std::vector<Detection>& detections) {
    // Run matching
    auto [matches, unmatched_tracks, unmatched_detections] = match(detections);
    
    // Update matched tracks
    for (const auto& [track_idx, det_idx] : matches) {
        tracks[track_idx].update(detections[det_idx]);
    }
    
    // Mark unmatched tracks as missed
    for (int track_idx : unmatched_tracks) {
        tracks[track_idx].mark_missed();
    }
    
    // Initiate new tracks for unmatched detections
    for (int det_idx : unmatched_detections) {
        initiate_track(detections[det_idx]);
    }
    
    // Remove deleted tracks
    tracks.erase(
        std::remove_if(tracks.begin(), tracks.end(),
                      [](const Track& t) { return t.is_deleted(); }),
        tracks.end());
    
    // Update distance metric with all features from all confirmed tracks
    std::vector<int> active_targets;
    std::vector<Eigen::VectorXf> all_features;
    std::vector<int> all_targets;
    
    for (const auto& track : tracks) {
        if (track.is_confirmed()) {
            active_targets.push_back(track.id);
            // Collect all features from this track
            for (const auto& feat : track.features) {
                all_features.push_back(feat);
                all_targets.push_back(track.id);
            }
        }
    }
    
    if (!all_features.empty()) {
        Eigen::MatrixXf features_matrix(all_features.size(), all_features[0].size());
        for (size_t i = 0; i < all_features.size(); ++i) {
            features_matrix.row(i) = all_features[i];
        }
        metric_->partial_fit(features_matrix, all_targets, active_targets);
    }
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
Tracker::match(const std::vector<Detection>& detections) {
    // Gated metric function for confirmed tracks
    auto gated_metric = [this](const std::vector<Track>& tracks,
                               const std::vector<Detection>& dets,
                               const std::vector<int>& track_indices,
                               const std::vector<int>& detection_indices) -> Eigen::MatrixXf {
        if (detection_indices.empty() || track_indices.empty()) {
            return Eigen::MatrixXf(0, 0);
        }
        
        // Extract features - handle empty features
        if (detection_indices.empty()) {
            return Eigen::MatrixXf(0, 0);
        }
        
        // Determine feature dimension from first detection
        int feat_dim = 0;
        for (int det_idx : detection_indices) {
            if (det_idx >= 0 && det_idx < static_cast<int>(dets.size())) {
                int dim = dets[det_idx].feat.size();
                if (dim > 0) {
                    feat_dim = dim;
                    break;
                }
            }
        }
        
        // If no features found, return empty matrix (will use IoU matching)
        if (feat_dim == 0) {
            return Eigen::MatrixXf(track_indices.size(), detection_indices.size()).setConstant(1e5f);
        }
        
        Eigen::MatrixXf features(detection_indices.size(), feat_dim);
        for (size_t i = 0; i < detection_indices.size(); ++i) {
            int det_idx = detection_indices[i];
            if (det_idx >= 0 && det_idx < static_cast<int>(dets.size()) && 
                dets[det_idx].feat.size() == feat_dim) {
                features.row(i) = dets[det_idx].feat;
            } else {
                features.row(i).setZero();  // Empty or mismatched feature
            }
        }
        
        std::vector<int> targets;
        for (int track_idx : track_indices) {
            targets.push_back(tracks[track_idx].id);
        }
        
        Eigen::MatrixXf cost_matrix = metric_->distance(features, targets);
        
        // Apply gating
        return linear_assignment::gate_cost_matrix(
            cost_matrix, tracks, dets, track_indices, detection_indices, mc_lambda_);
    };
    
    // Split track set into confirmed and unconfirmed tracks
    std::vector<int> confirmed_tracks, unconfirmed_tracks;
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (tracks[i].is_confirmed()) {
            confirmed_tracks.push_back(static_cast<int>(i));
        } else {
            unconfirmed_tracks.push_back(static_cast<int>(i));
        }
    }
    
    // Associate confirmed tracks using appearance features
    std::vector<int> unmatched_detections;
    auto [matches_a, unmatched_tracks_a, unmatched_detections_a] = 
        linear_assignment::matching_cascade(
            gated_metric,
            metric_->matching_threshold,
            max_age_,
            tracks,
            detections,
            confirmed_tracks);
    
    unmatched_detections = unmatched_detections_a;
    
    // Associate remaining tracks together with unconfirmed tracks using IoU
    std::vector<int> iou_track_candidates = unconfirmed_tracks;
    for (int k : unmatched_tracks_a) {
        if (tracks[k].time_since_update == 1) {
            iou_track_candidates.push_back(k);
        }
    }
    
    std::vector<int> unmatched_tracks_a_filtered;
    for (int k : unmatched_tracks_a) {
        if (tracks[k].time_since_update != 1) {
            unmatched_tracks_a_filtered.push_back(k);
        }
    }
    
    auto iou_metric = [](const std::vector<Track>& tracks,
                         const std::vector<Detection>& dets,
                         const std::vector<int>& track_indices,
                         const std::vector<int>& detection_indices) -> Eigen::MatrixXf {
        return iou_matching::iou_cost(tracks, dets, track_indices, detection_indices);
    };
    
    auto [matches_b, unmatched_tracks_b, unmatched_detections_b] =
        linear_assignment::min_cost_matching(
            iou_metric,
            max_iou_dist_,
            tracks,
            detections,
            iou_track_candidates,
            unmatched_detections);
    
    unmatched_detections = unmatched_detections_b;
    
    // Combine matches - validate no duplicates
    std::vector<std::pair<int, int>> matches = matches_a;
    std::set<int> matched_tracks_set, matched_dets_set;
    for (const auto& [t, d] : matches_a) {
        matched_tracks_set.insert(t);
        matched_dets_set.insert(d);
    }
    
    // Add matches_b, but skip any that would create duplicates
    for (const auto& [t, d] : matches_b) {
        if (matched_tracks_set.find(t) == matched_tracks_set.end() &&
            matched_dets_set.find(d) == matched_dets_set.end()) {
            matches.push_back({t, d});
            matched_tracks_set.insert(t);
            matched_dets_set.insert(d);
        }
    }
    
    // Combine unmatched tracks
    std::set<int> unmatched_set(unmatched_tracks_a_filtered.begin(), 
                                unmatched_tracks_a_filtered.end());
    unmatched_set.insert(unmatched_tracks_b.begin(), unmatched_tracks_b.end());
    std::vector<int> unmatched_tracks(unmatched_set.begin(), unmatched_set.end());
    
    return {matches, unmatched_tracks, unmatched_detections};
}

void Tracker::initiate_track(const Detection& detection) {
    tracks.emplace_back(detection, next_id_++, n_init_, max_age_, ema_alpha_);
}

void Tracker::reset() {
    tracks.clear();
    next_id_ = 1;
    if (metric_) {
        metric_->reset();
    }
}

} // namespace motcpp::trackers::strongsort

// ============================================================================
// StrongSORT tracker (public API)
// ============================================================================

namespace motcpp::trackers {

StrongSORT::StrongSORT(const std::string& reid_weights,
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
                       float min_conf,
                       float max_cos_dist,
                       float max_iou_dist,
                       int n_init,
                       int nn_budget,
                       float mc_lambda,
                       float ema_alpha)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                  per_class, nr_classes, asso_func, is_obb)
    , min_conf_(min_conf)
    , max_cos_dist_(max_cos_dist)
    , max_iou_dist_(max_iou_dist)
    , n_init_(n_init)
    , nn_budget_(nn_budget)
    , mc_lambda_(mc_lambda)
    , ema_alpha_(ema_alpha)
{
    // Initialize ReID backend only if weights are provided
    if (!reid_weights.empty()) {
        reid_backend_ = std::make_unique<appearance::ONNXBackend>(
            reid_weights, "", use_half, use_gpu);
    } else {
        reid_backend_ = nullptr;  // Will use pre-generated embeddings
    }
    
    // Initialize CMC
    cmc_ = std::make_unique<motion::ECC>();
    
    // Initialize tracker with metric
    auto metric = std::make_shared<strongsort::NearestNeighborDistanceMetric>(
        "cosine", max_cos_dist_, nn_budget_);
    tracker_ = std::make_unique<strongsort::Tracker>(
        metric, max_iou_dist_, max_age, n_init_, mc_lambda_, ema_alpha_);
}

StrongSORT::~StrongSORT() = default;

Eigen::MatrixXf StrongSORT::update(const Eigen::MatrixXf& dets,
                                   const cv::Mat& img,
                                   const Eigen::MatrixXf& embs) {
    // Check inputs
    check_inputs(dets, img, embs);
    
    if (dets.rows() == 0) {
        tracker_->predict();
        tracker_->update({});
        return Eigen::MatrixXf(0, 8);
    }
    
    // Add det_ind column: dets = hstack([dets, arange(len(dets))])
    Eigen::MatrixXf dets_with_ind(dets.rows(), dets.cols() + 1);
    dets_with_ind.leftCols(dets.cols()) = dets;
    for (int i = 0; i < dets.rows(); ++i) {
        dets_with_ind(i, dets.cols()) = static_cast<float>(i);
    }
    
    // Filter detections by min_conf (not det_thresh)
    std::vector<int> remain_inds;
    for (int i = 0; i < dets_with_ind.rows(); ++i) {
        if (dets_with_ind(i, 4) >= min_conf_) {  // conf column
            remain_inds.push_back(i);
        }
    }
    
    if (remain_inds.empty()) {
        tracker_->predict();
        tracker_->update({});
        return Eigen::MatrixXf(0, 8);
    }
    
    Eigen::MatrixXf dets_filtered(remain_inds.size(), dets_with_ind.cols());
    for (size_t i = 0; i < remain_inds.size(); ++i) {
        dets_filtered.row(i) = dets_with_ind.row(remain_inds[i]);
    }
    
    Eigen::MatrixXf xyxy = dets_filtered.leftCols(4);
    Eigen::VectorXf confs = dets_filtered.col(4);
    Eigen::VectorXi clss = dets_filtered.col(5).cast<int>();
    Eigen::VectorXi det_ind = dets_filtered.col(6).cast<int>();
    
    // Apply CMC if tracks exist
    if (!tracker_->tracks.empty()) {
        Eigen::Matrix<float, 2, 3> warp_matrix = cmc_->apply(img, xyxy);
        for (auto& track : tracker_->tracks) {
            track.camera_update(warp_matrix);
        }
    }
    
    // Extract appearance features
    Eigen::MatrixXf features;
    if (embs.rows() > 0) {
        // We have pre-computed embeddings.
        // Python StrongSort applies the same remain_inds mask to both dets and embs:
        //   dets = dets[remain_inds]
        //   features = embs[remain_inds]
        //
        // In C++ we receive embs aligned with the *unfiltered* detections for this frame,
        // so we must sub-select rows using remain_inds.
        //
        // The embeddings should have the same number of rows as the unfiltered detections.
        if (embs.rows() != dets_with_ind.rows()) {
            // Mismatch: embeddings don't match detections - create empty features
            features = Eigen::MatrixXf(remain_inds.size(), 0);
        } else {
            // Filter embeddings using remain_inds (same indices as filtered detections)
            features = Eigen::MatrixXf(remain_inds.size(), embs.cols());
            for (size_t i = 0; i < remain_inds.size(); ++i) {
                int src_idx = remain_inds[i];
                if (src_idx >= 0 && src_idx < embs.rows()) {
                    features.row(i) = embs.row(src_idx);
                } else {
                    features.row(i).setZero();
                }
            }
        }
    } else if (reid_backend_) {
        // No pre-computed embeddings: extract features using ReID backend
        features = reid_backend_->get_features(xyxy, img);
    } else {
        // No embeddings and no ReID backend - create empty features matrix.
        // This degrades to IoU-only association but keeps dimensions consistent.
        features = Eigen::MatrixXf(remain_inds.size(), 0);
    }
    
    // Convert xyxy to tlwh and create Detection objects
    Eigen::MatrixXf tlwh(xyxy.rows(), 4);
    for (int i = 0; i < xyxy.rows(); ++i) {
        float x1 = xyxy(i, 0), y1 = xyxy(i, 1);
        float x2 = xyxy(i, 2), y2 = xyxy(i, 3);
        tlwh(i, 0) = x1;  // x (top-left)
        tlwh(i, 1) = y1;  // y (top-left)
        tlwh(i, 2) = x2 - x1;  // w
        tlwh(i, 3) = y2 - y1;  // h
    }
    
    detections_buffer_.clear();
    detections_buffer_.reserve(tlwh.rows());
    for (int i = 0; i < tlwh.rows(); ++i) {
        Eigen::VectorXf feat = (features.rows() > i) ? features.row(i) : Eigen::VectorXf();
        detections_buffer_.emplace_back(
            tlwh.row(i).transpose(), confs(i), clss(i), det_ind(i), feat);
    }
    
    // Update tracker
    tracker_->predict();
    tracker_->update(detections_buffer_);
    
    // Build output: only confirmed tracks with time_since_update < 1
    std::vector<Eigen::VectorXf> outputs;
    outputs.reserve(tracker_->tracks.size());
    
    for (const auto& track : tracker_->tracks) {
        if (!track.is_confirmed() || track.time_since_update >= 1) {
            continue;
        }
        
        Eigen::Vector4f tlbr = track.to_tlbr();
        Eigen::VectorXf output(8);
        output(0) = tlbr(0);  // x1
        output(1) = tlbr(1);  // y1
        output(2) = tlbr(2);  // x2
        output(3) = tlbr(3);  // y2
        output(4) = static_cast<float>(track.id);
        output(5) = track.conf;
        output(6) = static_cast<float>(track.cls);
        output(7) = static_cast<float>(track.det_ind);
        outputs.push_back(output);
    }
    
    if (outputs.empty()) {
        return Eigen::MatrixXf(0, 8);
    }
    
    Eigen::MatrixXf result(outputs.size(), 8);
    for (size_t i = 0; i < outputs.size(); ++i) {
        result.row(i) = outputs[i];
    }
    
    return result;
}

void StrongSORT::reset() {
    // Reset tracker state: clear tracks and reset ID counter
    if (tracker_) {
        tracker_->reset();
    }
}

} // namespace motcpp::trackers
