// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/botsort.hpp>
#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/ops.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace motcpp::trackers {

// Static members
int BotSTrack::next_id_ = 0;
std::shared_ptr<KalmanFilterXYWH> BotSTrack::shared_kalman_ = std::make_shared<KalmanFilterXYWH>();

int BotSTrack::next_id() {
    return ++next_id_;
}

// BotSTrack implementation
BotSTrack::BotSTrack(const Eigen::VectorXf& det, int max_obs)
    : max_obs_(max_obs) {
    // det format: [x1, y1, x2, y2, conf, cls, det_ind]
    float x1 = det(0), y1 = det(1), x2 = det(2), y2 = det(3);
    float w = x2 - x1;
    float h = y2 - y1;
    float cx = x1 + w / 2.0f;
    float cy = y1 + h / 2.0f;
    
    xywh_ = Eigen::Vector4f(cx, cy, w, h);
    conf_ = det(4);
    cls_ = static_cast<int>(det(5));
    det_ind_ = (det.size() > 6) ? static_cast<int>(det(6)) : -1;
}

BotSTrack::BotSTrack(const Eigen::VectorXf& det, const Eigen::VectorXf& feat, int max_obs)
    : BotSTrack(det, max_obs) {
    curr_feat_ = feat;
    smooth_feat_ = feat;
    // Normalize feature
    if (smooth_feat_.norm() > 0) {
        smooth_feat_ /= smooth_feat_.norm();
    }
}

void BotSTrack::predict() {
    auto [mean, cov] = shared_kalman_->predict(mean_, covariance_);
    mean_ = mean;
    covariance_ = cov;
}

void BotSTrack::multi_predict(std::vector<BotSTrack*>& tracks) {
    for (auto* track : tracks) {
        track->predict();
    }
}

void BotSTrack::multi_gmc(std::vector<BotSTrack*>& tracks, const Eigen::Matrix3f& warp_matrix) {
    for (auto* track : tracks) {
        if (track->mean_.size() < 4) continue;
        
        // Get current bbox
        Eigen::Vector4f bbox = track->xyxy();
        float x1 = bbox(0), y1 = bbox(1), x2 = bbox(2), y2 = bbox(3);
        
        // Transform corners
        Eigen::Vector3f p1(x1, y1, 1.0f);
        Eigen::Vector3f p2(x2, y2, 1.0f);
        
        Eigen::Vector3f p1_new = warp_matrix * p1;
        Eigen::Vector3f p2_new = warp_matrix * p2;
        
        float x1_new = p1_new(0) / p1_new(2);
        float y1_new = p1_new(1) / p1_new(2);
        float x2_new = p2_new(0) / p2_new(2);
        float y2_new = p2_new(1) / p2_new(2);
        
        // Update state with new xywh
        float w = x2_new - x1_new;
        float h = y2_new - y1_new;
        float cx = x1_new + w / 2.0f;
        float cy = y1_new + h / 2.0f;
        
        track->mean_(0) = cx;
        track->mean_(1) = cy;
        track->mean_(2) = w;
        track->mean_(3) = h;
    }
}

void BotSTrack::activate(std::shared_ptr<KalmanFilterXYWH> kalman_filter, int frame_id) {
    kalman_filter_ = kalman_filter;
    id_ = next_id();
    
    auto [mean, cov] = kalman_filter_->initiate(xywh_);
    mean_ = mean;
    covariance_ = cov;
    
    tracklet_len_ = 0;
    state_ = BotTrackState::Tracked;
    if (frame_id == 1) {
        is_activated_ = true;
    }
    frame_id_ = frame_id;
    end_frame_ = frame_id;  // Initialize end_frame
    start_frame_ = frame_id;
}

void BotSTrack::re_activate(const BotSTrack& new_track, int frame_id, bool new_id) {
    auto [mean, cov] = shared_kalman_->update(mean_, covariance_, new_track.xywh_);
    mean_ = mean;
    covariance_ = cov;
    
    if (new_track.curr_feat_.size() > 0) {
        update_features(new_track.curr_feat_);
    }
    
    tracklet_len_ = 0;
    state_ = BotTrackState::Tracked;
    is_activated_ = true;
    frame_id_ = frame_id;
    end_frame_ = frame_id;  // Update end_frame when track is re-activated
    if (new_id) {
        id_ = next_id();
    }
    conf_ = new_track.conf_;
    cls_ = new_track.cls_;
    det_ind_ = new_track.det_ind_;
}

void BotSTrack::update(const BotSTrack& new_track, int frame_id) {
    frame_id_ = frame_id;
    end_frame_ = frame_id;  // Update end_frame when track is updated
    tracklet_len_++;
    
    history_observations_.push_back(xyxy());
    while (history_observations_.size() > static_cast<size_t>(max_obs_)) {
        history_observations_.pop_front();
    }
    
    auto [mean, cov] = shared_kalman_->update(mean_, covariance_, new_track.xywh_);
    mean_ = mean;
    covariance_ = cov;
    
    if (new_track.curr_feat_.size() > 0) {
        update_features(new_track.curr_feat_);
    }
    
    state_ = BotTrackState::Tracked;
    is_activated_ = true;
    conf_ = new_track.conf_;
    cls_ = new_track.cls_;
    det_ind_ = new_track.det_ind_;
}

void BotSTrack::update_features(const Eigen::VectorXf& feat) {
    curr_feat_ = feat;
    if (smooth_feat_.size() == 0) {
        smooth_feat_ = feat;
    } else {
        smooth_feat_ = alpha_ * smooth_feat_ + (1.0f - alpha_) * feat;
    }
    // Normalize
    if (smooth_feat_.norm() > 0) {
        smooth_feat_ /= smooth_feat_.norm();
    }
}

Eigen::Vector4f BotSTrack::xyxy() const {
    if (mean_.size() >= 4) {
        float cx = mean_(0);
        float cy = mean_(1);
        float w = mean_(2);
        float h = mean_(3);
        return Eigen::Vector4f(cx - w/2, cy - h/2, cx + w/2, cy + h/2);
    }
    float cx = xywh_(0), cy = xywh_(1), w = xywh_(2), h = xywh_(3);
    return Eigen::Vector4f(cx - w/2, cy - h/2, cx + w/2, cy + h/2);
}

Eigen::Vector4f BotSTrack::xywh() const {
    if (mean_.size() >= 4) {
        return Eigen::Vector4f(mean_(0), mean_(1), mean_(2), mean_(3));
    }
    return xywh_;
}

Eigen::Vector4f BotSTrack::tlwh() const {
    Eigen::Vector4f bbox = xyxy();
    return Eigen::Vector4f(bbox(0), bbox(1), bbox(2) - bbox(0), bbox(3) - bbox(1));
}

// BotSort implementation
BotSort::BotSort(
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
    float track_high_thresh,
    float track_low_thresh,
    float new_track_thresh,
    int track_buffer,
    float match_thresh,
    float proximity_thresh,
    float appearance_thresh,
    const std::string& cmc_method,
    int frame_rate,
    bool fuse_first_associate,
    bool with_reid
)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                  per_class, nr_classes, asso_func, is_obb)
    , track_high_thresh_(track_high_thresh)
    , track_low_thresh_(track_low_thresh)
    , new_track_thresh_(new_track_thresh)
    , track_buffer_(track_buffer)
    , match_thresh_(match_thresh)
    , proximity_thresh_(proximity_thresh)
    , appearance_thresh_(appearance_thresh)
    , fuse_first_associate_(fuse_first_associate)
    , with_reid_(with_reid)
{
    buffer_size_ = static_cast<int>(frame_rate / 30.0f * track_buffer);
    max_time_lost_ = buffer_size_;
    
    kalman_filter_ = std::make_shared<KalmanFilterXYWH>();
    BotSTrack::shared_kalman_ = kalman_filter_;
    
    // Initialize CMC
    if (cmc_method == "ecc") {
        cmc_ = std::make_unique<motcpp::motion::ECC>();
    }
    
    // Initialize ReID model
    if (with_reid_ && !reid_weights.empty()) {
        reid_model_ = std::make_unique<appearance::ONNXBackend>(reid_weights, "", use_half, use_gpu);
    }
    
    BotSTrack::next_id_ = 0;
}

void BotSort::reset() {
    BaseTracker::reset();
    active_tracks_.clear();
    lost_stracks_.clear();
    removed_stracks_.clear();
    BotSTrack::next_id_ = 0;
}

Eigen::MatrixXf BotSort::update(
    const Eigen::MatrixXf& dets,
    const cv::Mat& img,
    const Eigen::MatrixXf& embs
) {
    check_inputs(dets, img);
    
    if (dets.rows() == 0) {
        return Eigen::MatrixXf(0, 8);
    }
    
    frame_count_++;
    
    std::vector<BotSTrack*> activated_stracks, refind_stracks, lost_stracks, removed_stracks;
    
    // Split detections by confidence
    auto [dets_with_ind, dets_first, embs_first, dets_second] = split_detections(dets, embs);
    
    // Extract appearance features
    Eigen::MatrixXf features_high;
    if (with_reid_ && embs.cols() == 0 && reid_model_) {
        features_high = reid_model_->get_features(dets_first.leftCols(4), img);
    } else if (embs_first.rows() > 0) {
        features_high = embs_first;
    }
    
    // Create detection objects
    std::vector<BotSTrack> detections = create_detections(dets_first, features_high);
    
    // IMPORTANT: Reserve capacity in active_tracks_ to prevent reallocation
    // when adding new tracks later. This keeps existing pointers valid.
    active_tracks_.reserve(active_tracks_.size() + detections.size() + 10);
    
    // Separate unconfirmed and active tracks
    std::vector<BotSTrack*> unconfirmed, active_tracks_ptrs;
    for (auto& track : active_tracks_) {
        if (!track.is_activated()) {
            unconfirmed.push_back(&track);
        } else {
            active_tracks_ptrs.push_back(&track);
        }
    }
    
    // Create strack pool
    // Reserve capacity for lost_stracks_ to prevent reallocation
    lost_stracks_.reserve(lost_stracks_.size() + active_tracks_.size() + 10);
    
    std::vector<BotSTrack*> lost_stracks_ptrs;
    for (auto& track : lost_stracks_) {
        lost_stracks_ptrs.push_back(&track);
    }
    std::vector<BotSTrack*> strack_pool = joint_stracks(active_tracks_ptrs, lost_stracks_ptrs);
    
    // Predict new locations
    BotSTrack::multi_predict(strack_pool);
    
    // Apply CMC
    if (cmc_) {
        auto warp_2x3 = cmc_->apply(img, dets);
        // Convert 2x3 to 3x3 affine matrix
        Eigen::Matrix3f warp = Eigen::Matrix3f::Identity();
        warp.topRows<2>() = warp_2x3;
        BotSTrack::multi_gmc(strack_pool, warp);
        BotSTrack::multi_gmc(unconfirmed, warp);
    }
    
    // First association
    auto [matches_first, u_track_first, u_detection_first] = first_association(
        dets_with_ind, dets_first, active_tracks_ptrs, unconfirmed, img,
        detections, activated_stracks, refind_stracks, strack_pool
    );
    
    // Second association
    auto [matches_second, u_track_second, u_detection_second] = second_association(
        dets_second, activated_stracks, lost_stracks, refind_stracks,
        u_track_first, strack_pool
    );
    
    // Handle unconfirmed tracks
    auto [matches_unc, u_unconfirmed, u_detection_unc] = handle_unconfirmed_tracks(
        u_detection_first, detections, activated_stracks, removed_stracks, unconfirmed
    );
    
    // Initialize new tracks
    // u_detection_unc contains indices into the filtered detections list [detections[i] for i in u_detection_first]
    // So we need to create this filtered list and use u_detection_unc as indices into it
    std::vector<BotSTrack> filtered_detections;
    for (int i : u_detection_first) {
        if (i >= 0 && i < static_cast<int>(detections.size())) {
            filtered_detections.push_back(detections[i]);
        }
    }
    initialize_new_tracks(u_detection_unc, activated_stracks, filtered_detections);
    
    // Update track states
    update_track_states(removed_stracks);
    
    // Prepare output
    return prepare_output(activated_stracks, refind_stracks, lost_stracks, removed_stracks);
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
BotSort::split_detections(const Eigen::MatrixXf& dets, const Eigen::MatrixXf& embs) {
    int n = dets.rows();
    
    // Add detection indices
    Eigen::MatrixXf dets_with_ind(n, dets.cols() + 1);
    dets_with_ind.leftCols(dets.cols()) = dets;
    for (int i = 0; i < n; ++i) {
        dets_with_ind(i, dets.cols()) = static_cast<float>(i);
    }
    
    std::vector<int> first_indices, second_indices;
    for (int i = 0; i < n; ++i) {
        float conf = dets(i, 4);
        if (conf > track_high_thresh_) {
            first_indices.push_back(i);
        } else if (conf > track_low_thresh_) {
            second_indices.push_back(i);
        }
    }
    
    Eigen::MatrixXf dets_first(first_indices.size(), dets_with_ind.cols());
    Eigen::MatrixXf dets_second(second_indices.size(), dets_with_ind.cols());
    Eigen::MatrixXf embs_first;
    
    for (size_t i = 0; i < first_indices.size(); ++i) {
        dets_first.row(i) = dets_with_ind.row(first_indices[i]);
    }
    for (size_t i = 0; i < second_indices.size(); ++i) {
        dets_second.row(i) = dets_with_ind.row(second_indices[i]);
    }
    
    if (embs.rows() > 0 && embs.cols() > 0) {
        embs_first.resize(first_indices.size(), embs.cols());
        for (size_t i = 0; i < first_indices.size(); ++i) {
            if (first_indices[i] < embs.rows()) {
                embs_first.row(i) = embs.row(first_indices[i]);
            }
        }
    }
    
    return {dets_with_ind, dets_first, embs_first, dets_second};
}

std::vector<BotSTrack> BotSort::create_detections(
    const Eigen::MatrixXf& dets_first,
    const Eigen::MatrixXf& features_high
) {
    std::vector<BotSTrack> detections;
    
    for (int i = 0; i < dets_first.rows(); ++i) {
        Eigen::VectorXf det = dets_first.row(i).transpose();
        
        if (with_reid_ && features_high.rows() > i && features_high.cols() > 0) {
            Eigen::VectorXf feat = features_high.row(i).transpose();
            detections.emplace_back(det, feat, max_obs_);
        } else {
            detections.emplace_back(det, max_obs_);
        }
    }
    
    return detections;
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
BotSort::first_association(
    const Eigen::MatrixXf& dets,
    const Eigen::MatrixXf& dets_first,
    std::vector<BotSTrack*>& active_tracks,
    std::vector<BotSTrack*>& unconfirmed,
    const cv::Mat& img,
    std::vector<BotSTrack>& detections,
    std::vector<BotSTrack*>& activated_stracks,
    std::vector<BotSTrack*>& refind_stracks,
    std::vector<BotSTrack*>& strack_pool
) {
    // Compute IoU distance
    Eigen::MatrixXf ious_dists = utils::iou_distance(strack_pool, detections);
    Eigen::MatrixXf ious_dists_mask = (ious_dists.array() > proximity_thresh_).cast<float>();
    
    if (fuse_first_associate_) {
        ious_dists = utils::fuse_score(ious_dists, detections);
    }
    
    Eigen::MatrixXf dists = ious_dists;
    
    // Combine with embedding distance if using ReID
    if (with_reid_) {
        Eigen::MatrixXf emb_dists = utils::embedding_distance(strack_pool, detections);
        emb_dists = emb_dists / 2.0f;
        
        // Mask high embedding distances
        for (int i = 0; i < emb_dists.rows(); ++i) {
            for (int j = 0; j < emb_dists.cols(); ++j) {
                if (emb_dists(i, j) > appearance_thresh_) {
                    emb_dists(i, j) = 1.0f;
                }
                if (ious_dists_mask(i, j) > 0.5f) {
                    emb_dists(i, j) = 1.0f;
                }
            }
        }
        
        // Take minimum of IoU and embedding distances
        dists = dists.cwiseMin(emb_dists);
    }
    
    // Linear assignment
    auto result = utils::linear_assignment(dists, match_thresh_);
    auto& matches_arr = result.matches;
    auto& u_track = result.unmatched_a;
    auto& u_detection = result.unmatched_b;
    
    // Convert matches from array to pair
    std::vector<std::pair<int, int>> matches;
    for (const auto& m : matches_arr) {
        matches.emplace_back(m[0], m[1]);
    }
    
    // Update matched tracks
    for (const auto& [itracked, idet] : matches) {
        BotSTrack* track = strack_pool[itracked];
        BotSTrack& det = detections[idet];
        
        if (track->state() == BotTrackState::Tracked) {
            track->update(det, frame_count_);
            activated_stracks.push_back(track);
        } else {
            track->re_activate(det, frame_count_, false);
            refind_stracks.push_back(track);
        }
    }
    
    return std::make_tuple(matches, u_track, u_detection);
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
BotSort::second_association(
    const Eigen::MatrixXf& dets_second,
    std::vector<BotSTrack*>& activated_stracks,
    std::vector<BotSTrack*>& lost_stracks,
    std::vector<BotSTrack*>& refind_stracks,
    const std::vector<int>& u_track_first,
    std::vector<BotSTrack*>& strack_pool
) {
    // Create detections from second stage
    std::vector<BotSTrack> detections_second;
    for (int i = 0; i < dets_second.rows(); ++i) {
        Eigen::VectorXf det = dets_second.row(i).transpose();
        detections_second.emplace_back(det, max_obs_);
    }
    
    // Get remaining tracks
    std::vector<BotSTrack*> r_tracked_stracks;
    for (int i : u_track_first) {
        if (i < static_cast<int>(strack_pool.size()) && 
            strack_pool[i]->state() == BotTrackState::Tracked) {
            r_tracked_stracks.push_back(strack_pool[i]);
        }
    }
    
    if (r_tracked_stracks.empty() || detections_second.empty()) {
        return std::make_tuple(std::vector<std::pair<int, int>>(), std::vector<int>(), std::vector<int>());
    }
    
    // IoU distance
    Eigen::MatrixXf dists = utils::iou_distance(r_tracked_stracks, detections_second);
    auto result = utils::linear_assignment(dists, 0.5f);
    auto& matches_arr = result.matches;
    auto& u_track = result.unmatched_a;
    auto& u_detection = result.unmatched_b;
    
    // Convert matches from array to pair
    std::vector<std::pair<int, int>> matches;
    for (const auto& m : matches_arr) {
        matches.emplace_back(m[0], m[1]);
    }
    
    // Update matched tracks
    for (const auto& [itracked, idet] : matches) {
        BotSTrack* track = r_tracked_stracks[itracked];
        BotSTrack& det = detections_second[idet];
        
        if (track->state() == BotTrackState::Tracked) {
            track->update(det, frame_count_);
            activated_stracks.push_back(track);
        } else {
            track->re_activate(det, frame_count_, false);
            refind_stracks.push_back(track);
        }
    }
    
    // Mark remaining unmatched as lost
    for (int it : u_track) {
        BotSTrack* track = r_tracked_stracks[it];
        if (track->state() != BotTrackState::Lost) {
            track->mark_lost();
            lost_stracks.push_back(track);
        }
    }
    
    return std::make_tuple(matches, u_track, u_detection);
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
BotSort::handle_unconfirmed_tracks(
    const std::vector<int>& u_detection,
    std::vector<BotSTrack>& detections,
    std::vector<BotSTrack*>& activated_stracks,
    std::vector<BotSTrack*>& removed_stracks,
    std::vector<BotSTrack*>& unconfirmed
) {
    if (unconfirmed.empty() || u_detection.empty()) {
        std::vector<int> u_det_range;
        for (size_t i = 0; i < u_detection.size(); ++i) {
            u_det_range.push_back(static_cast<int>(i));
        }
        return {{}, {}, u_det_range};
    }
    
    // Filter detections to only those in u_detection (matching Python: detections = [detections[i] for i in u_detection])
    std::vector<BotSTrack> remaining_dets;
    for (int i : u_detection) {
        if (i >= 0 && i < static_cast<int>(detections.size())) {
            remaining_dets.push_back(detections[i]);
        }
    }
    
    if (remaining_dets.empty()) {
        std::vector<int> u_det_range;
        for (size_t i = 0; i < u_detection.size(); ++i) {
            u_det_range.push_back(static_cast<int>(i));
        }
        return {{}, {}, u_det_range};
    }
    
    // IoU distance
    Eigen::MatrixXf ious_dists = utils::iou_distance(unconfirmed, remaining_dets);
    Eigen::MatrixXf ious_dists_mask = (ious_dists.array() > proximity_thresh_).cast<float>();
    ious_dists = utils::fuse_score(ious_dists, remaining_dets);
    
    Eigen::MatrixXf dists = ious_dists;
    
    if (with_reid_) {
        Eigen::MatrixXf emb_dists = utils::embedding_distance(unconfirmed, remaining_dets);
        emb_dists = emb_dists / 2.0f;
        
        for (int i = 0; i < emb_dists.rows(); ++i) {
            for (int j = 0; j < emb_dists.cols(); ++j) {
                if (emb_dists(i, j) > appearance_thresh_) {
                    emb_dists(i, j) = 1.0f;
                }
                if (ious_dists_mask(i, j) > 0.5f) {
                    emb_dists(i, j) = 1.0f;
                }
            }
        }
        
        dists = dists.cwiseMin(emb_dists);
    }
    
    auto result = utils::linear_assignment(dists, 0.7f);
    auto& matches_arr = result.matches;
    auto& u_unconfirmed = result.unmatched_a;
    auto& u_detection_out = result.unmatched_b;
    
    // Convert matches from array to pair
    std::vector<std::pair<int, int>> matches;
    for (const auto& m : matches_arr) {
        matches.emplace_back(m[0], m[1]);
    }
    
    // Update matched
    for (const auto& [itracked, idet] : matches) {
        unconfirmed[itracked]->update(remaining_dets[idet], frame_count_);
        activated_stracks.push_back(unconfirmed[itracked]);
    }
    
    // Mark unmatched as removed
    for (int it : u_unconfirmed) {
        BotSTrack* track = unconfirmed[it];
        track->mark_removed();
        removed_stracks.push_back(track);
    }
    
    return std::make_tuple(matches, u_unconfirmed, u_detection_out);
}

void BotSort::initialize_new_tracks(
    const std::vector<int>& u_detections,
    std::vector<BotSTrack*>& activated_stracks,
    std::vector<BotSTrack>& detections
) {
    for (int idx : u_detections) {
        if (idx < 0 || idx >= static_cast<int>(detections.size())) {
            continue;
        }
        BotSTrack& track = detections[idx];
        if (track.conf() < new_track_thresh_) {
            continue;
        }
        
        active_tracks_.push_back(track);
        active_tracks_.back().activate(kalman_filter_, frame_count_);
        activated_stracks.push_back(&active_tracks_.back());
    }
}

void BotSort::update_track_states(std::vector<BotSTrack*>& removed_stracks) {
    for (auto& track : lost_stracks_) {
        if (frame_count_ - track.end_frame() > max_time_lost_) {
            track.mark_removed();
            removed_stracks.push_back(&track);
        }
    }
}

Eigen::MatrixXf BotSort::prepare_output(
    std::vector<BotSTrack*>& activated_stracks,
    std::vector<BotSTrack*>& refind_stracks,
    std::vector<BotSTrack*>& lost_stracks,
    std::vector<BotSTrack*>& removed_stracks
) {
    // IMPORTANT: We must process lost_stracks BEFORE modifying active_tracks_
    // because lost_stracks contains pointers to tracks in active_tracks_
    
    // First, collect IDs of active tracks
    std::unordered_set<int> active_ids;
    for (auto* track : activated_stracks) {
        if (track->state() == BotTrackState::Tracked) {
            active_ids.insert(track->id());
        }
    }
    for (auto* track : refind_stracks) {
        if (track->state() == BotTrackState::Tracked) {
            active_ids.insert(track->id());
        }
    }
    
    // Build new lost list from existing lost_stracks_ FIRST
    std::vector<BotSTrack> new_lost;
    new_lost.reserve(lost_stracks_.size() + lost_stracks.size());
    
    std::unordered_set<int> lost_id_set;
    for (auto& track : lost_stracks_) {
        if (active_ids.find(track.id()) == active_ids.end() && 
            track.state() != BotTrackState::Removed) {
            new_lost.push_back(std::move(track));
            lost_id_set.insert(new_lost.back().id());
        }
    }
    
    // Add newly lost tracks from the parameter (pointers to active_tracks_)
    // Copy their data BEFORE we modify active_tracks_
    for (auto* track : lost_stracks) {
        if (track && 
            active_ids.find(track->id()) == active_ids.end() &&
            lost_id_set.find(track->id()) == lost_id_set.end()) {
            new_lost.push_back(*track);  // Copy the track data
            lost_id_set.insert(track->id());
        }
    }
    
    // NOW we can safely modify active_tracks_
    std::vector<BotSTrack> new_active;
    new_active.reserve(active_tracks_.size());
    
    for (auto& track : active_tracks_) {
        if (track.state() == BotTrackState::Tracked) {
            new_active.push_back(std::move(track));
        }
    }
    
    active_tracks_ = std::move(new_active);
    lost_stracks_ = std::move(new_lost);
    
    // Prepare output from active_tracks_
    std::vector<Eigen::VectorXf> outputs;
    outputs.reserve(active_tracks_.size());
    
    for (const auto& track : active_tracks_) {
        if (track.is_activated()) {
            Eigen::Vector4f bbox = track.xyxy();
            Eigen::VectorXf output(8);
            output << bbox(0), bbox(1), bbox(2), bbox(3),
                     static_cast<float>(track.id()),
                     track.conf(),
                     static_cast<float>(track.cls()),
                     static_cast<float>(track.det_ind());
            outputs.push_back(output);
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

// Helper functions
std::vector<BotSTrack*> joint_stracks(
    std::vector<BotSTrack*>& tlista,
    std::vector<BotSTrack*>& tlistb
) {
    std::unordered_map<int, bool> exists;
    std::vector<BotSTrack*> res;
    
    for (auto* t : tlista) {
        exists[t->id()] = true;
        res.push_back(t);
    }
    
    for (auto* t : tlistb) {
        if (exists.find(t->id()) == exists.end()) {
            exists[t->id()] = true;
            res.push_back(t);
        }
    }
    
    return res;
}

std::vector<BotSTrack*> sub_stracks(
    std::vector<BotSTrack*>& tlista,
    const std::vector<BotSTrack*>& tlistb
) {
    std::unordered_map<int, BotSTrack*> stracks;
    for (auto* t : tlista) {
        stracks[t->id()] = t;
    }
    
    for (auto* t : tlistb) {
        stracks.erase(t->id());
    }
    
    std::vector<BotSTrack*> res;
    for (auto& [id, t] : stracks) {
        res.push_back(t);
    }
    return res;
}

std::pair<std::vector<BotSTrack*>, std::vector<BotSTrack*>>
remove_duplicate_stracks(
    std::vector<BotSTrack*>& stracksa,
    std::vector<BotSTrack*>& stracksb
) {
    // Compute pairwise IoU distances
    Eigen::MatrixXf pdist = utils::iou_distance(stracksa, stracksb);
    
    std::vector<int> dupa, dupb;
    for (int i = 0; i < pdist.rows(); ++i) {
        for (int j = 0; j < pdist.cols(); ++j) {
            if (pdist(i, j) < 0.15f) {
                int timep = stracksa[i]->frame_id() - stracksa[i]->start_frame();
                int timeq = stracksb[j]->frame_id() - stracksb[j]->start_frame();
                if (timep > timeq) {
                    dupb.push_back(j);
                } else {
                    dupa.push_back(i);
                }
            }
        }
    }
    
    std::vector<BotSTrack*> resa, resb;
    for (int i = 0; i < static_cast<int>(stracksa.size()); ++i) {
        if (std::find(dupa.begin(), dupa.end(), i) == dupa.end()) {
            resa.push_back(stracksa[i]);
        }
    }
    for (int i = 0; i < static_cast<int>(stracksb.size()); ++i) {
        if (std::find(dupb.begin(), dupb.end(), i) == dupb.end()) {
            resb.push_back(stracksb[i]);
        }
    }
    
    return {resa, resb};
}

} // namespace motcpp::trackers
