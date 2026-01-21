// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/ocsort.hpp>
#include <motcpp/utils/iou.hpp>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <limits>
#include <numeric>

namespace motcpp::trackers {

// Forward declarations
namespace {
Eigen::Vector2f speed_direction_impl(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2);
Eigen::Vector4f convert_x_to_bbox_impl(const Eigen::VectorXf& x);
} // anonymous namespace

Eigen::Vector2f speed_direction(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2);
Eigen::Vector4f convert_x_to_bbox(const Eigen::VectorXf& x);

// Helper function: k_previous_obs implementation
Eigen::VectorXf k_previous_obs(const std::unordered_map<int, Eigen::VectorXf>& observations,
                                int cur_age, int k, bool is_obb) {
    if (observations.empty()) {
        if (is_obb) {
            Eigen::VectorXf result(6);
            result << -1, -1, -1, -1, -1, -1;
            return result;
        } else {
            Eigen::VectorXf result(5);
            result << -1, -1, -1, -1, -1;
            return result;
        }
    }
    
    for (int i = 0; i < k; ++i) {
        int dt = k - i;
        int age_key = cur_age - dt;
        auto it = observations.find(age_key);
        if (it != observations.end()) {
            return it->second;
        }
    }
    
    // Return observation with max age
    int max_age = std::max_element(observations.begin(), observations.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; })->first;
    return observations.at(max_age);
}

KalmanBoxTracker::KalmanBoxTracker(const Eigen::VectorXf& bbox, int cls, int det_ind,
                                   int delta_t, int max_obs,
                                   float Q_xy_scaling, float Q_s_scaling)
    : kf(7, 4, max_obs)
    , id_(next_id())
    , age_(0)
    , hits_(0)
    , hit_streak_(0)
    , time_since_update_(0)
    , conf_(bbox.size() > 4 ? bbox(4) : 0.0f)
    , cls_(cls)
    , det_ind_(det_ind)
    , delta_t_(delta_t)
    , max_obs_(max_obs)
    , last_observation_(5)
    , velocity_(Eigen::Vector2f::Zero())
{
    // Initialize last_observation placeholder: [-1, -1, -1, -1, -1]
    last_observation_ << -1, -1, -1, -1, -1;
    
    // Configure Kalman filter matrices (same as Python)
    // F matrix is already set in constructor
    
    // Set Q scaling
    kf.Q(4, 4) *= Q_xy_scaling;
    kf.Q(5, 5) *= Q_xy_scaling;
    kf.Q(6, 6) *= Q_s_scaling;
    
    // Initialize state with bbox
    Eigen::Vector4f xyxy = bbox.head<4>();
    Eigen::Vector4f xysr = utils::xyxy2xysr(xyxy);
    kf.x.head<4>() = xysr;
}

void KalmanBoxTracker::update(const Eigen::VectorXf& bbox, int cls, int det_ind) {
    det_ind_ = det_ind;
    
    if (bbox.size() >= 4) {
        conf_ = (bbox.size() > 4) ? bbox(4) : 0.0f;
        cls_ = cls;
        
        // Calculate velocity direction if we have previous observation
        float last_sum = last_observation_.head<4>().sum();
        if (last_sum >= 0) {  // Has previous observation
            Eigen::VectorXf previous_box = motcpp::trackers::k_previous_obs(observations_, age_, delta_t_, false);
            
            // Check if previous_box is valid (not placeholder)
            if (previous_box.size() >= 4 && previous_box.head<4>().sum() >= 0) {
                velocity_ = speed_direction(previous_box.head<4>(), bbox.head<4>());
            } else {
                velocity_ = speed_direction(last_observation_.head<4>(), bbox.head<4>());
            }
        }
        
        // Update observations
        last_observation_.head<4>() = bbox.head<4>();
        if (last_observation_.size() > 4) {
            last_observation_(4) = conf_;
        }
        observations_[age_] = last_observation_;
        history_observations_.push_back(last_observation_);
        if (history_observations_.size() > static_cast<size_t>(max_obs_)) {
            history_observations_.pop_front();
        }
        
        time_since_update_ = 0;
        hits_++;
        hit_streak_++;
        
        // Update Kalman filter
        Eigen::Vector4f xyxy = bbox.head<4>();
        Eigen::Vector4f xysr = utils::xyxy2xysr(xyxy);
        kf.update(xysr);
    } else {
        // None/null update
        kf.update(Eigen::VectorXf());
    }
}

Eigen::Vector4f KalmanBoxTracker::predict() {
    // Check if scale + ratio velocity would make scale negative
    if ((kf.x(6) + kf.x(2)) <= 0) {
        kf.x(6) = 0.0f;  // Zero scale velocity
    }
    
    kf.predict();
    age_++;
    
    if (time_since_update_ > 0) {
        hit_streak_ = 0;
    }
    time_since_update_++;
    
    Eigen::Vector4f bbox = convert_x_to_bbox(kf.x);
    return bbox;
}

Eigen::Vector4f KalmanBoxTracker::get_state() const {
    return convert_x_to_bbox(kf.x);
}

Eigen::VectorXf KalmanBoxTracker::k_previous_obs(int k) const {
    return motcpp::trackers::k_previous_obs(observations_, age_, k, false);
}

// Helper function implementations
namespace {
Eigen::Vector2f speed_direction_impl(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) {
    float cx1 = (bbox1(0) + bbox1(2)) / 2.0f;
    float cy1 = (bbox1(1) + bbox1(3)) / 2.0f;
    float cx2 = (bbox2(0) + bbox2(2)) / 2.0f;
    float cy2 = (bbox2(1) + bbox2(3)) / 2.0f;
    
    float dy = cy2 - cy1;
    float dx = cx2 - cx1;
    float norm = std::sqrt(dy * dy + dx * dx) + 1e-6f;
    
    return Eigen::Vector2f(dy / norm, dx / norm);
}

Eigen::Vector4f convert_x_to_bbox_impl(const Eigen::VectorXf& x) {
    float xc = x(0), yc = x(1), s = x(2), r = x(3);
    float w = std::sqrt(s * r);
    float h = s / w;
    float x1 = xc - w * 0.5f;
    float y1 = yc - h * 0.5f;
    float x2 = xc + w * 0.5f;
    float y2 = yc + h * 0.5f;
    return Eigen::Vector4f(x1, y1, x2, y2);
}
} // anonymous namespace

Eigen::Vector2f speed_direction(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) {
    return speed_direction_impl(bbox1, bbox2);
}

Eigen::Vector4f convert_x_to_bbox(const Eigen::VectorXf& x) {
    return convert_x_to_bbox_impl(x);
}

OCSort::OCSort(float det_thresh, int max_age, int max_obs, int min_hits,
               float iou_threshold, bool per_class, int nr_classes,
               const std::string& asso_func, bool is_obb,
               float min_conf, int delta_t, float inertia, bool use_byte,
               float Q_xy_scaling, float Q_s_scaling)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                 per_class, nr_classes, asso_func, is_obb)
    , min_conf_(min_conf)
    , asso_threshold_(iou_threshold)
    , delta_t_(delta_t)
    , inertia_(inertia)
    , use_byte_(use_byte)
    , Q_xy_scaling_(Q_xy_scaling)
    , Q_s_scaling_(Q_s_scaling)
    , frame_id_(0)
{
    // Pre-allocate buffers
    cost_matrix_buffer_.resize(200, 200);
    track_xyxy_buffer_.resize(200, 4);
    det_xyxy_buffer_.resize(200, 4);
    det_confs_buffer_.resize(200);
    velocity_buffer_.resize(200, 2);
    k_obs_buffer_.resize(200, 5);
    
    KalmanBoxTracker::clear_count();
}

void OCSort::reset() {
    BaseTracker::reset();
    frame_id_ = 0;
    active_tracks_.clear();
    KalmanBoxTracker::clear_count();
}

Eigen::Vector2f OCSort::speed_direction(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const {
    // Compute velocity direction between two bounding boxes
    // bbox format: [x1, y1, x2, y2] or [x1, y1, x2, y2, ...]
    // Returns normalized direction vector (dy, dx)
    float cx1 = (bbox1(0) + bbox1(2)) / 2.0f;
    float cy1 = (bbox1(1) + bbox1(3)) / 2.0f;
    float cx2 = (bbox2(0) + bbox2(2)) / 2.0f;
    float cy2 = (bbox2(1) + bbox2(3)) / 2.0f;
    
    float dy = cy2 - cy1;
    float dx = cx2 - cx1;
    
    float norm = std::sqrt(dy * dy + dx * dx) + 1e-6f;
    return Eigen::Vector2f(dy / norm, dx / norm);
}

Eigen::Vector4f OCSort::convert_x_to_bbox(const Eigen::VectorXf& x) const {
    // Convert Kalman filter state (x, y, s, r) to bounding box (x1, y1, x2, y2)
    // x = [cx, cy, scale, ratio] where scale = w*h, ratio = w/h
    float cx = x(0);
    float cy = x(1);
    float s = x(2);  // scale (area)
    float r = x(3);  // aspect ratio (w/h)
    
    // Prevent negative or zero values
    s = std::max(s, 1e-6f);
    r = std::max(r, 1e-6f);
    
    // w = sqrt(s * r), h = s / w
    float w = std::sqrt(s * r);
    float h = s / w;
    
    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    float x2 = cx + w / 2.0f;
    float y2 = cy + h / 2.0f;
    
    return Eigen::Vector4f(x1, y1, x2, y2);
}

// Forward declaration for associate function
namespace ocsort_assoc {
    struct AssociateResult {
        std::vector<std::array<int, 2>> matches;
        std::vector<int> unmatched_dets;
        std::vector<int> unmatched_trks;
    };
    
    AssociateResult associate(const Eigen::MatrixXf& detections,
                              const Eigen::MatrixXf& trackers,
                              const utils::AssociationFunction& asso_func,
                              float iou_threshold,
                              const Eigen::MatrixXf& velocities,
                              const Eigen::MatrixXf& previous_obs,
                              float vdc_weight,
                              int w, int h);
}

Eigen::MatrixXf OCSort::update(const Eigen::MatrixXf& dets,
                                const cv::Mat& img,
                                const Eigen::MatrixXf& embs) {
    check_inputs(dets, img, embs);
    setup_detection_format(dets);
    setup_association_function(img);
    
    frame_id_++;
    frame_count_++;
    
    int img_h = img.rows;
    int img_w = img.cols;
    
    // Add detection indices
    Eigen::MatrixXf dets_with_ind = Eigen::MatrixXf(dets.rows(), dets.cols() + 1);
    dets_with_ind.leftCols(dets.cols()) = dets;
    for (int i = 0; i < dets.rows(); ++i) {
        dets_with_ind(i, dets.cols()) = static_cast<float>(i);
    }
    
    // Filter detections by confidence
    Eigen::VectorXf confs = dets_with_ind.col(4);
    
    // inds_low: conf > min_conf
    // inds_high: conf < det_thresh
    // inds_second: min_conf < conf < det_thresh (for second matching)
    std::vector<int> second_indices;
    std::vector<int> remain_indices;
    
    for (int i = 0; i < dets_with_ind.rows(); ++i) {
        if (confs(i) > min_conf_ && confs(i) < det_thresh_) {
            second_indices.push_back(i);
        }
        if (confs(i) > det_thresh_) {
            remain_indices.push_back(i);
        }
    }
    
    Eigen::MatrixXf dets_high(remain_indices.size(), dets_with_ind.cols());
    Eigen::MatrixXf dets_second(second_indices.size(), dets_with_ind.cols());
    
    int high_count = 0;
    for (int idx : remain_indices) {
        dets_high.row(high_count++) = dets_with_ind.row(idx);
    }
    
    int second_count = 0;
    for (int idx : second_indices) {
        dets_second.row(second_count++) = dets_with_ind.row(idx);
    }
    
    // Predict all trackers
    size_t n_tracks = active_tracks_.size();
    if (track_xyxy_buffer_.rows() < static_cast<int>(n_tracks)) {
        track_xyxy_buffer_.conservativeResize(n_tracks * 2, 4);
    }
    
    Eigen::MatrixXf trks(n_tracks, 5);
    std::vector<int> to_del;
    
    for (size_t t = 0; t < n_tracks; ++t) {
        Eigen::Vector4f pos = active_tracks_[t].predict();
        trks(t, 0) = pos(0);
        trks(t, 1) = pos(1);
        trks(t, 2) = pos(2);
        trks(t, 3) = pos(3);
        trks(t, 4) = 0.0f;
        
        // Check for NaN
        if (std::isnan(pos(0)) || std::isnan(pos(1)) || std::isnan(pos(2)) || std::isnan(pos(3))) {
            to_del.push_back(t);
        }
    }
    
    // Remove invalid tracks
    for (auto it = to_del.rbegin(); it != to_del.rend(); ++it) {
        active_tracks_.erase(active_tracks_.begin() + *it);
        n_tracks--;
    }
    trks.conservativeResize(n_tracks, 5);
    
    if (n_tracks == 0) {
        // No tracks, create new ones from all detections
        std::vector<Eigen::Vector4f> outputs;
        outputs.reserve(high_count);
        
        for (int i = 0; i < high_count; ++i) {
            Eigen::VectorXf bbox = dets_high.row(i).head<5>();
            int cls = static_cast<int>(dets_high(i, 5));
            int det_ind = static_cast<int>(dets_high(i, 6));
            
            KalmanBoxTracker trk(bbox, cls, det_ind, delta_t_, max_obs_,
                                 Q_xy_scaling_, Q_s_scaling_);
            active_tracks_.push_back(trk);
        }
        
        // Return empty for now (tracks need min_hits before output)
        return Eigen::MatrixXf(0, 8);
    }
    
    // Get velocities and k_observations
    if (velocity_buffer_.rows() < static_cast<int>(n_tracks)) {
        velocity_buffer_.conservativeResize(n_tracks * 2, 2);
    }
    if (k_obs_buffer_.rows() < static_cast<int>(n_tracks)) {
        k_obs_buffer_.conservativeResize(n_tracks * 2, 5);
    }
    
    Eigen::MatrixXf velocities = velocity_buffer_.topRows(n_tracks);
    Eigen::MatrixXf k_observations = k_obs_buffer_.topRows(n_tracks);
    
    for (size_t t = 0; t < n_tracks; ++t) {
        Eigen::Vector2f vel = active_tracks_[t].velocity();
        velocities(t, 0) = vel(0);  // dy
        velocities(t, 1) = vel(1);   // dx
        
        Eigen::VectorXf k_obs = active_tracks_[t].k_previous_obs(delta_t_);
        if (k_obs.size() >= 5) {
            k_observations.row(t) = k_obs.head<5>().transpose();
        } else {
            k_observations.row(t) << -1, -1, -1, -1, -1;
        }
    }
    
    // First association
    Eigen::MatrixXf dets_for_assoc = dets_high.leftCols(5);
    Eigen::MatrixXf trks_for_assoc = trks.leftCols(5);
    // Create association function
    utils::AssociationFunction asso_func(img_w, img_h, asso_func_name_);
    ocsort_assoc::AssociateResult assoc_result = ocsort_assoc::associate(dets_for_assoc, trks_for_assoc,
                                            asso_func, asso_threshold_,
                                            velocities, k_observations,
                                            inertia_, img_w, img_h);
    
    // Update matched tracks
    for (const auto& match : assoc_result.matches) {
        int det_idx = match[0];
        int trk_idx = match[1];
        Eigen::VectorXf bbox = dets_high.row(det_idx).head<5>();
        int cls = static_cast<int>(dets_high(det_idx, 5));
        int det_ind = static_cast<int>(dets_high(det_idx, 6));
        active_tracks_[trk_idx].update(bbox, cls, det_ind);
    }
    
    // Second association (BYTE-style with low confidence detections)
    if (use_byte_ && second_count > 0 && !assoc_result.unmatched_trks.empty()) {
        Eigen::MatrixXf u_trks(assoc_result.unmatched_trks.size(), 5);
        for (size_t i = 0; i < assoc_result.unmatched_trks.size(); ++i) {
            int trk_idx = assoc_result.unmatched_trks[i];
            u_trks.row(i) = trks.row(trk_idx).head<5>();
        }
        
        Eigen::MatrixXf dets_second_for_assoc = dets_second.leftCols(5);
        utils::AssociationFunction asso_func_byte(img_w, img_h, asso_func_name_);
        Eigen::MatrixXf iou_left = asso_func_byte(dets_second_for_assoc, u_trks.leftCols(5));
        
        float max_iou = iou_left.maxCoeff();
        if (max_iou > asso_threshold_) {
            Eigen::MatrixXf cost_matrix = -iou_left;  // Negative for minimization
            auto assignment = utils::linear_assignment(cost_matrix, -asso_threshold_);
            
            std::vector<int> to_remove_trk_indices;
            for (const auto& match : assignment.matches) {
                int det_idx = match[0];
                int u_trk_idx = match[1];
                int trk_idx = assoc_result.unmatched_trks[u_trk_idx];
                
                if (iou_left(det_idx, u_trk_idx) < asso_threshold_) {
                    continue;
                }
                
                Eigen::VectorXf bbox = dets_second.row(det_idx).head<5>();
                int cls = static_cast<int>(dets_second(det_idx, 5));
                int det_ind = static_cast<int>(dets_second(det_idx, 6));
                active_tracks_[trk_idx].update(bbox, cls, det_ind);
                to_remove_trk_indices.push_back(trk_idx);
            }
            
            // Remove matched tracks from unmatched list
            std::unordered_set<int> to_remove_set(to_remove_trk_indices.begin(),
                                                  to_remove_trk_indices.end());
            assoc_result.unmatched_trks.erase(
                std::remove_if(assoc_result.unmatched_trks.begin(),
                              assoc_result.unmatched_trks.end(),
                              [&](int idx) { return to_remove_set.find(idx) != to_remove_set.end(); }),
                assoc_result.unmatched_trks.end());
        }
    }
    
    // Rematch unmatched detections and unmatched tracks using last_observation
    if (!assoc_result.unmatched_dets.empty() && !assoc_result.unmatched_trks.empty()) {
        Eigen::MatrixXf left_dets(assoc_result.unmatched_dets.size(), 5);
        Eigen::MatrixXf left_trks(assoc_result.unmatched_trks.size(), 5);
        
        for (size_t i = 0; i < assoc_result.unmatched_dets.size(); ++i) {
            int det_idx = assoc_result.unmatched_dets[i];
            left_dets.row(i) = dets_high.row(det_idx).head<5>();
        }
        
        for (size_t i = 0; i < assoc_result.unmatched_trks.size(); ++i) {
            int trk_idx = assoc_result.unmatched_trks[i];
            Eigen::VectorXf last_obs = active_tracks_[trk_idx].last_observation();
            if (last_obs.size() >= 4) {
                left_trks.row(i) = last_obs.head<5>().transpose();
            } else {
                left_trks.row(i) << 0, 0, 0, 0, 0;
            }
        }
        
        utils::AssociationFunction asso_func_rematch(img_w, img_h, asso_func_name_);
        Eigen::MatrixXf iou_left = asso_func_rematch(left_dets.leftCols(4), left_trks.leftCols(4));
        float max_iou = iou_left.maxCoeff();
        
        if (max_iou > asso_threshold_) {
            Eigen::MatrixXf cost_matrix = -iou_left;
            auto assignment = utils::linear_assignment(cost_matrix, -asso_threshold_);
            
            std::vector<int> to_remove_det_indices, to_remove_trk_indices;
            for (const auto& match : assignment.matches) {
                int left_det_idx = match[0];
                int left_trk_idx = match[1];
                int det_idx = assoc_result.unmatched_dets[left_det_idx];
                int trk_idx = assoc_result.unmatched_trks[left_trk_idx];
                
                if (iou_left(left_det_idx, left_trk_idx) < asso_threshold_) {
                    continue;
                }
                
                Eigen::VectorXf bbox = dets_high.row(det_idx).head<5>();
                int cls = static_cast<int>(dets_high(det_idx, 5));
                int det_ind = static_cast<int>(dets_high(det_idx, 6));
                active_tracks_[trk_idx].update(bbox, cls, det_ind);
                
                to_remove_det_indices.push_back(det_idx);
                to_remove_trk_indices.push_back(trk_idx);
            }
            
            // Remove matched from unmatched lists
            std::unordered_set<int> to_remove_det_set(to_remove_det_indices.begin(),
                                                      to_remove_det_indices.end());
            std::unordered_set<int> to_remove_trk_set(to_remove_trk_indices.begin(),
                                                      to_remove_trk_indices.end());
            
            assoc_result.unmatched_dets.erase(
                std::remove_if(assoc_result.unmatched_dets.begin(),
                              assoc_result.unmatched_dets.end(),
                              [&](int idx) { return to_remove_det_set.find(idx) != to_remove_det_set.end(); }),
                assoc_result.unmatched_dets.end());
            
            assoc_result.unmatched_trks.erase(
                std::remove_if(assoc_result.unmatched_trks.begin(),
                              assoc_result.unmatched_trks.end(),
                              [&](int idx) { return to_remove_trk_set.find(idx) != to_remove_trk_set.end(); }),
                assoc_result.unmatched_trks.end());
        }
    }
    
    // Update unmatched tracks with None
    for (int trk_idx : assoc_result.unmatched_trks) {
        active_tracks_[trk_idx].update(Eigen::VectorXf(), 0, 0);
    }
    
    // Create new trackers for unmatched detections
    for (int det_idx : assoc_result.unmatched_dets) {
        Eigen::VectorXf bbox = dets_high.row(det_idx).head<5>();
        int cls = static_cast<int>(dets_high(det_idx, 5));
        int det_ind = static_cast<int>(dets_high(det_idx, 6));
        
        KalmanBoxTracker trk(bbox, cls, det_ind, delta_t_, max_obs_,
                            Q_xy_scaling_, Q_s_scaling_);
        active_tracks_.push_back(trk);
    }
    
    // Build output (only confirmed tracks)
    std::vector<Eigen::Vector4f> output_bboxes;
    std::vector<float> output_ids, output_confs, output_clss, output_det_inds;
    
    for (auto it = active_tracks_.rbegin(); it != active_tracks_.rend(); ++it) {
        auto& trk = *it;
        
        Eigen::Vector4f d;
        Eigen::VectorXf last_obs = trk.last_observation();
        if (last_obs.size() >= 4 && last_obs.head<4>().sum() < 0) {
            d = trk.get_state();
        } else {
            d = last_obs.head<4>();
        }
        
        if ((trk.time_since_update() < 1) &&
            (trk.hit_streak() >= min_hits_ || frame_count_ <= min_hits_)) {
            output_bboxes.push_back(d);
            output_ids.push_back(static_cast<float>(trk.id() + 1));  // +1 for MOT format
            output_confs.push_back(trk.conf());
            output_clss.push_back(static_cast<float>(trk.cls()));
            output_det_inds.push_back(static_cast<float>(trk.det_ind()));
        }
        
        // Remove dead tracklets
        if (trk.time_since_update() > max_age_) {
            size_t idx = std::distance(it, active_tracks_.rend()) - 1;
            active_tracks_.erase(active_tracks_.begin() + idx);
        }
    }
    
    if (output_bboxes.empty()) {
        return Eigen::MatrixXf(0, 8);
    }
    
    Eigen::MatrixXf outputs(output_bboxes.size(), 8);
    for (size_t i = 0; i < output_bboxes.size(); ++i) {
        outputs(i, 0) = output_bboxes[i](0);  // x1
        outputs(i, 1) = output_bboxes[i](1);   // y1
        outputs(i, 2) = output_bboxes[i](2);  // x2
        outputs(i, 3) = output_bboxes[i](3);   // y2
        outputs(i, 4) = output_ids[i];
        outputs(i, 5) = output_confs[i];
        outputs(i, 6) = output_clss[i];
        outputs(i, 7) = output_det_inds[i];
    }
    
    return outputs;
}

// Associate function implementation
namespace ocsort_assoc {
    AssociateResult associate(const Eigen::MatrixXf& detections,
                              const Eigen::MatrixXf& trackers,
                              const utils::AssociationFunction& asso_func,
                              float iou_threshold,
                              const Eigen::MatrixXf& velocities,
                              const Eigen::MatrixXf& previous_obs,
                              float vdc_weight,
                              int /* w */, int /* h */) {
        AssociateResult result;
        
        if (trackers.rows() == 0) {
            result.unmatched_dets.resize(detections.rows());
            for (int i = 0; i < detections.rows(); ++i) {
                result.unmatched_dets[i] = i;
            }
            return result;
        }
        
        // Compute speed direction batch: Y, X from detections to previous_obs
        int n_dets = detections.rows();
        int n_trks = trackers.rows();
        
        Eigen::MatrixXf Y(n_trks, n_dets), X(n_trks, n_dets);
        
        for (int i = 0; i < n_trks; ++i) {
            Eigen::VectorXf prev_obs = previous_obs.row(i).head<4>();
            for (int j = 0; j < n_dets; ++j) {
                Eigen::Vector4f det_xyxy = detections.row(j).head<4>();
                float cx1 = (det_xyxy(0) + det_xyxy(2)) / 2.0f;
                float cy1 = (det_xyxy(1) + det_xyxy(3)) / 2.0f;
                float cx2 = (prev_obs(0) + prev_obs(2)) / 2.0f;
                float cy2 = (prev_obs(1) + prev_obs(3)) / 2.0f;
                
                float dx = cx1 - cx2;
                float dy = cy1 - cy2;
                float norm = std::sqrt(dx * dx + dy * dy) + 1e-6f;
                Y(i, j) = dy / norm;
                X(i, j) = dx / norm;
            }
        }
        
        // Compute angle difference cost
        // Note: Y and X are (n_trks, n_dets), velocities is (n_trks, 2)
        Eigen::MatrixXf inertia_Y = velocities.col(0).replicate(1, n_dets);  // (n_trks, n_dets)
        Eigen::MatrixXf inertia_X = velocities.col(1).replicate(1, n_dets);  // (n_trks, n_dets)
        
        Eigen::MatrixXf diff_angle_cos = inertia_X.cwiseProduct(X) + inertia_Y.cwiseProduct(Y);  // (n_trks, n_dets)
        diff_angle_cos = diff_angle_cos.cwiseMax(-1.0f).cwiseMin(1.0f);
        
        Eigen::MatrixXf diff_angle = diff_angle_cos.unaryExpr([](float cos_val) {
            const float PI = 3.14159265358979323846f;
            return (PI / 2.0f - std::abs(std::acos(cos_val))) / PI;
        });  // (n_trks, n_dets)
        
        // Valid mask (previous_obs not placeholder)
        Eigen::VectorXf valid_mask = (previous_obs.col(4).array() >= 0).cast<float>();  // (n_trks,)
        Eigen::MatrixXf valid_mask_matrix = valid_mask.replicate(1, n_dets);  // (n_trks, n_dets)
        
        // IoU matrix: asso_func returns (n_dets, n_trks)
        Eigen::MatrixXf iou_matrix = asso_func(detections, trackers);  // (n_dets, n_trks)
        
        // Detection scores: Python does np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
        // This creates (n_dets, n_trks)
        Eigen::VectorXf det_scores = detections.col(4);  // (n_dets,)
        Eigen::MatrixXf scores_matrix = det_scores.replicate(1, n_trks);  // (n_dets, n_trks)
        
        // Angle difference cost: Python computes (n_trks, n_dets) then transposes to (n_dets, n_trks)
        Eigen::MatrixXf angle_diff_cost = (valid_mask_matrix.cwiseProduct(diff_angle)) * vdc_weight;  // (n_trks, n_dets)
        angle_diff_cost = angle_diff_cost.transpose().eval();  // (n_dets, n_trks) - CRITICAL: Match Python transpose, use eval() to avoid aliasing
        angle_diff_cost = angle_diff_cost.cwiseProduct(scores_matrix);  // (n_dets, n_trks) * (n_dets, n_trks)
        
        if (iou_matrix.rows() > 0 && iou_matrix.cols() > 0) {
            // Check for trivial case (one-to-one matching)
            // iou_matrix is (n_dets, n_trks)
            Eigen::MatrixXi a = (iou_matrix.array() > iou_threshold).cast<int>();
            int max_row_sum = a.rowwise().sum().maxCoeff();  // max det matches
            int max_col_sum = a.colwise().sum().maxCoeff();  // max trk matches
            
            if (max_row_sum == 1 && max_col_sum == 1) {
                // Trivial matching: each detection matches at most one track, each track matches at most one detection
                for (int i = 0; i < a.rows(); ++i) {  // detections
                    for (int j = 0; j < a.cols(); ++j) {  // tracks
                        if (a(i, j) == 1) {
                            result.matches.push_back({{i, j}});  // (det_idx, trk_idx)
                        }
                    }
                }
            } else {
                // Use linear assignment
                // iou_matrix is (n_dets, n_trks), angle_diff_cost is (n_dets, n_trks)
                Eigen::MatrixXf final_cost = -(iou_matrix + angle_diff_cost);  // (n_dets, n_trks)
                auto assignment = utils::linear_assignment(final_cost, -iou_threshold);
                
                // Filter matches by IoU threshold
                for (const auto& match : assignment.matches) {
                    int det_idx = match[0];
                    int trk_idx = match[1];
                    if (iou_matrix(det_idx, trk_idx) >= iou_threshold) {
                        result.matches.push_back({{det_idx, trk_idx}});
                    } else {
                        result.unmatched_dets.push_back(det_idx);
                        result.unmatched_trks.push_back(trk_idx);
                    }
                }
            }
        }
        
        // Find unmatched detections and trackers
        std::unordered_set<int> matched_dets, matched_trks;
        for (const auto& match : result.matches) {
            matched_dets.insert(match[0]);
            matched_trks.insert(match[1]);
        }
        
        for (int i = 0; i < n_dets; ++i) {
            if (matched_dets.find(i) == matched_dets.end()) {
                result.unmatched_dets.push_back(i);
            }
        }
        
        for (int i = 0; i < n_trks; ++i) {
            if (matched_trks.find(i) == matched_trks.end()) {
                result.unmatched_trks.push_back(i);
            }
        }
        
        return result;
    }
}

} // namespace motcpp::trackers

