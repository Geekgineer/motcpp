// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/trackers/deepocsort.hpp>
#include <motcpp/utils/iou.hpp>
#include <motcpp/appearance/onnx_backend.hpp>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <limits>
#include <numeric>

namespace motcpp::trackers {

// Forward declarations - use anonymous namespace to avoid multiple definition
namespace {
Eigen::Vector2f speed_direction_impl(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2);
Eigen::Vector4f convert_x_to_bbox_impl(const Eigen::VectorXf& x);
} // anonymous namespace

// Use anonymous namespace versions directly - these are local to this file
// Note: ocsort.cpp also defines these in motcpp::trackers namespace, but we use local versions

// Helper function: k_previous_obs implementation (for internal use)
namespace {
Eigen::VectorXf k_previous_obs_impl(const std::unordered_map<int, Eigen::VectorXf>& observations,
                                    int cur_age, int k) {
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
    
    // Return observation with max age
    int max_age = std::max_element(observations.begin(), observations.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; })->first;
    return observations.at(max_age);
}
} // anonymous namespace

DeepOCSortKalmanBoxTracker::DeepOCSortKalmanBoxTracker(
    const Eigen::VectorXf& bbox, int cls, int det_ind,
    const Eigen::VectorXf& emb,
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
    , emb_(emb)
    , frozen_(false)
{
    // Initialize last_observation placeholder: [-1, -1, -1, -1, -1]
    last_observation_ << -1, -1, -1, -1, -1;
    
    // Normalize embedding if provided
    if (emb_.size() > 0) {
        float norm = emb_.norm();
        if (norm > 1e-6f) {
            emb_ /= norm;
        }
    }
    
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

void DeepOCSortKalmanBoxTracker::update(const Eigen::VectorXf& bbox, int cls, int det_ind) {
    det_ind_ = det_ind;
    frozen_ = false;
    
    if (bbox.size() >= 4) {
        conf_ = (bbox.size() > 4) ? bbox(4) : 0.0f;
        cls_ = cls;
        
        // Calculate velocity direction if we have previous observation
        float last_sum = last_observation_.head<4>().sum();
        if (last_sum >= 0) {  // Has previous observation
            Eigen::VectorXf previous_box = k_previous_obs_impl(observations_, age_, delta_t_);
            
            // Check if previous_box is valid (not placeholder)
            if (previous_box.size() >= 4 && previous_box.head<4>().sum() >= 0) {
                velocity_ = speed_direction_impl(previous_box.head<4>(), bbox.head<4>());
            } else {
                velocity_ = speed_direction_impl(last_observation_.head<4>(), bbox.head<4>());
            }
        }
        
        // Update observations
        Eigen::VectorXf new_obs(5);
        new_obs.head<4>() = bbox.head<4>();
        new_obs(4) = conf_;
        last_observation_ = new_obs;
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
        frozen_ = true;
    }
}

void DeepOCSortKalmanBoxTracker::update_emb(const Eigen::VectorXf& emb, float alpha) {
    if (emb.size() == 0) {
        return;
    }
    
    if (emb_.size() == 0) {
        // Initialize with provided embedding
        emb_ = emb;
    } else {
        // Exponential moving average: emb = alpha * emb + (1 - alpha) * new_emb
        emb_ = alpha * emb_ + (1.0f - alpha) * emb;
    }
    
    // Normalize
    float norm = emb_.norm();
    if (norm > 1e-6f) {
        emb_ /= norm;
    }
}

Eigen::Vector4f DeepOCSortKalmanBoxTracker::predict() {
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
    
    Eigen::Vector4f bbox = convert_x_to_bbox_impl(kf.x);
    return bbox;
}

Eigen::Vector4f DeepOCSortKalmanBoxTracker::get_state() const {
    return convert_x_to_bbox_impl(kf.x);
}

Eigen::VectorXf DeepOCSortKalmanBoxTracker::k_previous_obs(int k) const {
    return k_previous_obs_impl(observations_, age_, k);
}

void DeepOCSortKalmanBoxTracker::apply_affine_correction(const Eigen::Matrix2f& m, const Eigen::Vector2f& t) {
    // For OCR: transform last_observation
    if (last_observation_.head<4>().sum() > 0) {
        // Reshape bbox corners: [x1,y1,x2,y2] -> [[x1,x2],[y1,y2]]
        Eigen::Matrix2f corners;
        corners(0, 0) = last_observation_(0);  // x1
        corners(1, 0) = last_observation_(1);  // y1
        corners(0, 1) = last_observation_(2);  // x2
        corners(1, 1) = last_observation_(3);  // y2
        
        // Transform: m @ corners + t
        Eigen::Matrix2f transformed = m * corners;
        transformed.col(0) += t;
        transformed.col(1) += t;
        
        last_observation_(0) = transformed(0, 0);  // x1
        last_observation_(1) = transformed(1, 0);  // y1
        last_observation_(2) = transformed(0, 1);  // x2
        last_observation_(3) = transformed(1, 1);  // y2
    }
    
    // Apply to each box in the range of velocity computation
    for (int dt = delta_t_; dt >= 0; --dt) {
        int age_key = age_ - dt;
        auto it = observations_.find(age_key);
        if (it != observations_.end()) {
            Eigen::VectorXf& obs = it->second;
            if (obs.head<4>().sum() > 0) {
                Eigen::Matrix2f corners;
                corners(0, 0) = obs(0);
                corners(1, 0) = obs(1);
                corners(0, 1) = obs(2);
                corners(1, 1) = obs(3);
                
                Eigen::Matrix2f transformed = m * corners;
                transformed.col(0) += t;
                transformed.col(1) += t;
                
                obs(0) = transformed(0, 0);
                obs(1) = transformed(1, 0);
                obs(2) = transformed(0, 1);
                obs(3) = transformed(1, 1);
            }
        }
    }
    
    // Also need to change kf state
    kf.apply_affine_correction(m, t);
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

// Forward declaration for associate function
namespace deepocsort_assoc {
    struct AssociateResult {
        std::vector<std::array<int, 2>> matches;
        std::vector<int> unmatched_dets;
        std::vector<int> unmatched_trks;
    };
    
    // Compute adaptive weight for embedding cost
    Eigen::MatrixXf compute_aw_max_metric(const Eigen::MatrixXf& emb_cost,
                                         float w_association_emb,
                                         float bottom);
    
    AssociateResult associate(const Eigen::MatrixXf& detections,
                              const Eigen::MatrixXf& trackers,
                              const utils::AssociationFunction& asso_func,
                              float iou_threshold,
                              const Eigen::MatrixXf& velocities,
                              const Eigen::MatrixXf& previous_obs,
                              float vdc_weight,
                              int w, int h,
                              const Eigen::MatrixXf& emb_cost = Eigen::MatrixXf(),
                              float w_assoc_emb = 0.5f,
                              bool aw_off = false,
                              float aw_param = 0.5f);
}

// Compute adaptive weight for embedding cost
Eigen::MatrixXf deepocsort_assoc::compute_aw_max_metric(const Eigen::MatrixXf& emb_cost,
                                                         float w_association_emb,
                                                         float bottom) {
    Eigen::MatrixXf w_emb = Eigen::MatrixXf::Constant(emb_cost.rows(), emb_cost.cols(), w_association_emb);
    
    // Process rows (detections)
    for (int idx = 0; idx < emb_cost.rows(); ++idx) {
        // Get sorted indices (descending order)
        std::vector<std::pair<float, int>> sorted;
        for (int j = 0; j < emb_cost.cols(); ++j) {
            sorted.push_back({emb_cost(idx, j), j});
        }
        std::sort(sorted.rbegin(), sorted.rend());
        
        if (sorted.size() < 2) {
            continue;
        }
        
        float max_val = sorted[0].first;
        float second_max_val = sorted[1].first;
        
        if (max_val == 0.0f) {
            w_emb.row(idx).setZero();
        } else {
            float row_weight = 1.0f - std::max((second_max_val / max_val) - bottom, 0.0f) / (1.0f - bottom);
            w_emb.row(idx) *= row_weight;
        }
    }
    
    // Process columns (tracks)
    for (int idj = 0; idj < emb_cost.cols(); ++idj) {
        // Get sorted indices (descending order)
        std::vector<std::pair<float, int>> sorted;
        for (int i = 0; i < emb_cost.rows(); ++i) {
            sorted.push_back({emb_cost(i, idj), i});
        }
        std::sort(sorted.rbegin(), sorted.rend());
        
        if (sorted.size() < 2) {
            continue;
        }
        
        float max_val = sorted[0].first;
        float second_max_val = sorted[1].first;
        
        if (max_val == 0.0f) {
            w_emb.col(idj).setZero();
        } else {
            float col_weight = 1.0f - std::max((second_max_val / max_val) - bottom, 0.0f) / (1.0f - bottom);
            w_emb.col(idj) *= col_weight;
        }
    }
    
    return w_emb.cwiseProduct(emb_cost);
}

// Associate function implementation for DeepOCSort
deepocsort_assoc::AssociateResult deepocsort_assoc::associate(
    const Eigen::MatrixXf& detections,
    const Eigen::MatrixXf& trackers,
    const utils::AssociationFunction& asso_func,
    float iou_threshold,
    const Eigen::MatrixXf& velocities,
    const Eigen::MatrixXf& previous_obs,
    float vdc_weight,
    int w, int h,
    const Eigen::MatrixXf& emb_cost,
    float w_assoc_emb,
    bool aw_off,
    float aw_param) {
    
    AssociateResult result;
    
    if (trackers.rows() == 0) {
        result.unmatched_dets.resize(detections.rows());
        std::iota(result.unmatched_dets.begin(), result.unmatched_dets.end(), 0);
        return result;
    }
    
    int n_dets = detections.rows();
    int n_trks = trackers.rows();
    
    // Compute speed direction (Y, X) for angle difference
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
    Eigen::MatrixXf inertia_Y = velocities.col(0).replicate(1, n_dets);
    Eigen::MatrixXf inertia_X = velocities.col(1).replicate(1, n_dets);
    
    Eigen::MatrixXf diff_angle_cos = inertia_X.cwiseProduct(X) + inertia_Y.cwiseProduct(Y);
    diff_angle_cos = diff_angle_cos.cwiseMax(-1.0f).cwiseMin(1.0f);
    
    Eigen::MatrixXf diff_angle = diff_angle_cos.unaryExpr([](float cos_val) {
        const float PI = 3.14159265358979323846f;
        return (PI / 2.0f - std::abs(std::acos(cos_val))) / PI;
    });
    
    Eigen::VectorXf valid_mask = (previous_obs.col(4).array() >= 0).cast<float>();
    Eigen::MatrixXf valid_mask_matrix = valid_mask.replicate(1, n_dets);
    
    Eigen::MatrixXf iou_matrix = asso_func(detections, trackers);
    
    Eigen::VectorXf det_scores = detections.col(4);
    Eigen::MatrixXf scores_matrix = det_scores.replicate(1, n_trks);
    
    Eigen::MatrixXf angle_diff_cost = (valid_mask_matrix.cwiseProduct(diff_angle)) * vdc_weight;
    angle_diff_cost = angle_diff_cost.transpose().eval();
    angle_diff_cost = angle_diff_cost.cwiseProduct(scores_matrix);
    
    // Handle embedding cost
    Eigen::MatrixXf final_emb_cost;
    if (emb_cost.rows() == 0 || emb_cost.cols() == 0) {
        final_emb_cost = Eigen::MatrixXf::Zero(n_dets, n_trks);
    } else {
        final_emb_cost = emb_cost;
        // Zero out embedding cost where IoU is zero
        for (int i = 0; i < n_dets; ++i) {
            for (int j = 0; j < n_trks; ++j) {
                if (iou_matrix(i, j) <= 0.0f) {
                    final_emb_cost(i, j) = 0.0f;
                }
            }
        }
        
        // Apply adaptive weights or fixed weight
        if (!aw_off) {
            final_emb_cost = compute_aw_max_metric(final_emb_cost, w_assoc_emb, aw_param);
        } else {
            final_emb_cost *= w_assoc_emb;
        }
    }
    
    if (iou_matrix.rows() > 0 && iou_matrix.cols() > 0) {
        // Check for trivial case (one-to-one matching)
        Eigen::MatrixXi a = (iou_matrix.array() > iou_threshold).cast<int>();
        int max_row_sum = a.rowwise().sum().maxCoeff();
        int max_col_sum = a.colwise().sum().maxCoeff();
        
        if (max_row_sum == 1 && max_col_sum == 1) {
            // Trivial matching
            for (int i = 0; i < a.rows(); ++i) {
                for (int j = 0; j < a.cols(); ++j) {
                    if (a(i, j) == 1) {
                        result.matches.push_back({{i, j}});
                    }
                }
            }
        } else {
            // Use linear assignment
            Eigen::MatrixXf final_cost = -(iou_matrix + angle_diff_cost + final_emb_cost);
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
            
            // Add unmatched from assignment
            for (int idx : assignment.unmatched_a) {
                result.unmatched_dets.push_back(idx);
            }
            for (int idx : assignment.unmatched_b) {
                result.unmatched_trks.push_back(idx);
            }
        }
    }
    
    // Build unmatched lists
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

// DeepOCSort constructor
DeepOCSort::DeepOCSort(const std::string& reid_weights,
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
                       int delta_t,
                       float inertia,
                       float w_association_emb,
                       float alpha_fixed_emb,
                       float aw_param,
                       bool embedding_off,
                       bool cmc_off,
                       bool aw_off,
                       float Q_xy_scaling,
                       float Q_s_scaling)
    : BaseTracker(det_thresh, max_age, max_obs, min_hits, iou_threshold,
                 per_class, nr_classes, asso_func, is_obb)
    , delta_t_(delta_t)
    , inertia_(inertia)
    , w_association_emb_(w_association_emb)
    , alpha_fixed_emb_(alpha_fixed_emb)
    , aw_param_(aw_param)
    , embedding_off_(embedding_off)
    , cmc_off_(cmc_off)
    , aw_off_(aw_off)
    , Q_xy_scaling_(Q_xy_scaling)
    , Q_s_scaling_(Q_s_scaling)
{
    // Initialize ReID backend (matches Python: ReidAutoBackend(...).model)
    // Extract model name from path if possible
    std::string model_name = "";
    size_t last_slash = reid_weights.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        model_name = reid_weights.substr(last_slash + 1);
    }
    
    reid_backend_ = std::make_unique<appearance::ONNXBackend>(
        reid_weights, model_name, use_half, use_gpu);
    
    // Initialize CMC (SOF)
    if (!cmc_off_) {
        cmc_ = std::make_unique<motion::SOF>(0.15f);  // Default scale 0.15
    }
    
    // Pre-allocate buffers
    cost_matrix_buffer_.resize(200, 200);
    track_xyxy_buffer_.resize(200, 4);
    det_xyxy_buffer_.resize(200, 4);
    det_confs_buffer_.resize(200);
    velocity_buffer_.resize(200, 2);
    k_obs_buffer_.resize(200, 5);
    dets_embs_buffer_.resize(200, 512);  // Typical embedding dimension
    trk_embs_buffer_.resize(200, 512);
    emb_cost_buffer_.resize(200, 200);
    final_cost_buffer_.resize(200, 200);
    
    DeepOCSortKalmanBoxTracker::clear_count();
}

DeepOCSort::~DeepOCSort() = default;

void DeepOCSort::reset() {
    BaseTracker::reset();
    active_tracks_.clear();
    DeepOCSortKalmanBoxTracker::clear_count();
}

Eigen::Vector2f DeepOCSort::speed_direction(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) const {
    return speed_direction_impl(bbox1, bbox2);
}

Eigen::Vector4f DeepOCSort::convert_x_to_bbox(const Eigen::VectorXf& x) const {
    return convert_x_to_bbox_impl(x);
}

Eigen::MatrixXf DeepOCSort::update(const Eigen::MatrixXf& dets,
                                   const cv::Mat& img,
                                   const Eigen::MatrixXf& embs) {
    check_inputs(dets, img, embs);
    setup_detection_format(dets);
    setup_association_function(img);
    
    frame_count_++;
    int img_h = img.rows;
    int img_w = img.cols;
    
    // Filter detections by det_thresh and add det_ind column
    Eigen::VectorXf scores = dets.col(4);
    Eigen::MatrixXf dets_with_ind(dets.rows(), dets.cols() + 1);
    dets_with_ind.leftCols(dets.cols()) = dets;
    for (int i = 0; i < dets.rows(); ++i) {
        dets_with_ind(i, dets.cols()) = static_cast<float>(i);
    }
    
    // Filter by det_thresh
    std::vector<int> remain_indices;
    for (int i = 0; i < dets_with_ind.rows(); ++i) {
        if (scores(i) > det_thresh_) {
            remain_indices.push_back(i);
        }
    }
    
    Eigen::MatrixXf dets_filtered(remain_indices.size(), dets_with_ind.cols());
    for (size_t i = 0; i < remain_indices.size(); ++i) {
        dets_filtered.row(i) = dets_with_ind.row(remain_indices[i]);
    }
    
    // Extract embeddings (matches Python exactly)
    Eigen::MatrixXf dets_embs;
    if (embedding_off_ || dets_filtered.rows() == 0) {
        dets_embs = Eigen::MatrixXf::Ones(dets_filtered.rows(), 1);
    } else if (embs.rows() > 0) {
        // Use provided embeddings (filtered by remain_indices)
        dets_embs.resize(remain_indices.size(), embs.cols());
        for (size_t i = 0; i < remain_indices.size(); ++i) {
            dets_embs.row(i) = embs.row(remain_indices[i]);
        }
    } else {
        // Extract via ReID backend
        Eigen::MatrixXf xyxys = dets_filtered.leftCols<4>();
        dets_embs = reid_backend_->get_features(xyxys, img);
    }
    
    // CMC (Camera Motion Compensation)
    if (!cmc_off_ && cmc_) {
        Eigen::Matrix<float, 2, 3> transform = cmc_->apply(img, dets_filtered.leftCols<4>());
        // Extract m (2x2) and t (2x1) from transform (2x3)
        Eigen::Matrix2f m = transform.leftCols<2>();
        Eigen::Vector2f t = transform.col(2);
        
        // Apply affine correction to all tracks
        for (auto& trk : active_tracks_) {
            trk.apply_affine_correction(m, t);
        }
    }
    
    // Compute adaptive alpha from detection confidence
    Eigen::VectorXf trust = (dets_filtered.col(4).array() - det_thresh_) / (1.0f - det_thresh_);
    Eigen::VectorXf dets_alpha = Eigen::VectorXf::Constant(dets_filtered.rows(), alpha_fixed_emb_) +
                                 (1.0f - alpha_fixed_emb_) * (Eigen::VectorXf::Ones(dets_filtered.rows()) - trust);
    
    // Predict all tracks
    size_t n_tracks = active_tracks_.size();
    if (n_tracks == 0) {
        // No tracks, create new ones
        for (int i = 0; i < dets_filtered.rows(); ++i) {
            Eigen::VectorXf bbox = dets_filtered.row(i).head<5>();
            int cls = static_cast<int>(dets_filtered(i, 5));
            int det_ind = static_cast<int>(dets_filtered(i, 6));
            Eigen::VectorXf emb = dets_embs.row(i);
            
            DeepOCSortKalmanBoxTracker trk(bbox, cls, det_ind, emb, delta_t_, max_obs_,
                                           Q_xy_scaling_, Q_s_scaling_);
            active_tracks_.push_back(trk);
        }
        return Eigen::MatrixXf(0, 8);  // No confirmed tracks yet
    }
    
    // Prepare track predictions and embeddings
    Eigen::MatrixXf trks(n_tracks, 5);
    std::vector<Eigen::VectorXf> trk_embs;
    std::vector<int> to_del;
    
    for (size_t t = 0; t < n_tracks; ++t) {
        Eigen::Vector4f pos = active_tracks_[t].predict();
        trks(t, 0) = pos(0);
        trks(t, 1) = pos(1);
        trks(t, 2) = pos(2);
        trks(t, 3) = pos(3);
        trks(t, 4) = 0.0f;
        
        if (std::isnan(pos(0)) || std::isnan(pos(1)) || std::isnan(pos(2)) || std::isnan(pos(3))) {
            to_del.push_back(t);
        } else {
            trk_embs.push_back(active_tracks_[t].get_emb());
        }
    }
    
    // Remove invalid tracks
    for (auto it = to_del.rbegin(); it != to_del.rend(); ++it) {
        active_tracks_.erase(active_tracks_.begin() + *it);
        n_tracks--;
    }
    trks.conservativeResize(n_tracks, 5);
    
    if (n_tracks == 0) {
        // Create new tracks from all detections
        for (int i = 0; i < dets_filtered.rows(); ++i) {
            Eigen::VectorXf bbox = dets_filtered.row(i).head<5>();
            int cls = static_cast<int>(dets_filtered(i, 5));
            int det_ind = static_cast<int>(dets_filtered(i, 6));
            Eigen::VectorXf emb = dets_embs.row(i);
            
            DeepOCSortKalmanBoxTracker trk(bbox, cls, det_ind, emb, delta_t_, max_obs_,
                                           Q_xy_scaling_, Q_s_scaling_);
            active_tracks_.push_back(trk);
        }
        return Eigen::MatrixXf(0, 8);
    }
    
    // Prepare velocities and k_observations
    Eigen::MatrixXf velocities(n_tracks, 2);
    Eigen::MatrixXf k_observations(n_tracks, 5);
    
    for (size_t t = 0; t < n_tracks; ++t) {
        Eigen::Vector2f vel = active_tracks_[t].velocity();
        velocities(t, 0) = vel(0);
        velocities(t, 1) = vel(1);
        
        Eigen::VectorXf k_obs = active_tracks_[t].k_previous_obs(delta_t_);
        if (k_obs.size() >= 5) {
            k_observations.row(t) = k_obs.head<5>();
        } else {
            k_observations.row(t) << -1, -1, -1, -1, -1;
        }
    }
    
    // Build track embeddings matrix
    Eigen::MatrixXf trk_embs_matrix;
    if (!trk_embs.empty() && trk_embs[0].size() > 0) {
        int emb_dim = trk_embs[0].size();
        trk_embs_matrix.resize(n_tracks, emb_dim);
        for (size_t t = 0; t < n_tracks; ++t) {
            if (t < trk_embs.size() && trk_embs[t].size() == emb_dim) {
                trk_embs_matrix.row(t) = trk_embs[t];
            } else {
                // Pad with zeros if dimension mismatch
                trk_embs_matrix.row(t).setZero();
            }
        }
    } else {
        // If no track embeddings, create dummy matrix matching dets_embs dimension
        int emb_dim = (dets_embs.rows() > 0) ? dets_embs.cols() : 1;
        trk_embs_matrix = Eigen::MatrixXf(n_tracks, emb_dim);
        trk_embs_matrix.setOnes();
    }
    
    // First stage association with embeddings
    Eigen::MatrixXf emb_cost;
    if (!embedding_off_ && dets_embs.rows() > 0 && trk_embs_matrix.rows() > 0) {
        // Ensure dimensions match
        if (dets_embs.cols() == trk_embs_matrix.cols()) {
            // Cosine similarity: dets_embs @ trk_embs.T
            // dets_embs: (n_dets, emb_dim), trk_embs_matrix: (n_trks, emb_dim)
            // Result: (n_dets, n_trks)
            emb_cost = dets_embs * trk_embs_matrix.transpose();
        } else {
            // Dimension mismatch - create zero cost matrix
            emb_cost = Eigen::MatrixXf::Zero(dets_embs.rows(), trk_embs_matrix.rows());
        }
    } else {
        emb_cost = Eigen::MatrixXf();  // Empty
    }
    
    // Create association function
    utils::AssociationFunction asso_func(img_w, img_h, asso_func_name_);
    
    auto assoc_result = deepocsort_assoc::associate(
        dets_filtered.leftCols<5>(),
        trks,
        asso_func,
        iou_threshold_,
        velocities,
        k_observations,
        inertia_,
        img_w,
        img_h,
        emb_cost,
        w_association_emb_,
        aw_off_,
        aw_param_
    );
    
    // Update matched tracks
    for (const auto& match : assoc_result.matches) {
        int det_idx = match[0];
        int trk_idx = match[1];
        
        Eigen::VectorXf bbox = dets_filtered.row(det_idx).head<5>();
        int cls = static_cast<int>(dets_filtered(det_idx, 5));
        int det_ind = static_cast<int>(dets_filtered(det_idx, 6));
        
        active_tracks_[trk_idx].update(bbox, cls, det_ind);
        active_tracks_[trk_idx].update_emb(dets_embs.row(det_idx), dets_alpha(det_idx));
    }
    
    // Second stage association (OCR) for unmatched detections/tracks
    if (assoc_result.unmatched_dets.size() > 0 && assoc_result.unmatched_trks.size() > 0) {
        // Get unmatched detections and tracks
        Eigen::MatrixXf left_dets(assoc_result.unmatched_dets.size(), 5);
        Eigen::MatrixXf left_dets_embs(assoc_result.unmatched_dets.size(), dets_embs.cols());
        Eigen::MatrixXf left_trks(assoc_result.unmatched_trks.size(), 4);
        Eigen::MatrixXf left_trks_embs(assoc_result.unmatched_trks.size(), trk_embs_matrix.cols());
        
        for (size_t i = 0; i < assoc_result.unmatched_dets.size(); ++i) {
            int det_idx = assoc_result.unmatched_dets[i];
            left_dets.row(i) = dets_filtered.row(det_idx).head<5>();
            left_dets_embs.row(i) = dets_embs.row(det_idx);
        }
        
        for (size_t i = 0; i < assoc_result.unmatched_trks.size(); ++i) {
            int trk_idx = assoc_result.unmatched_trks[i];
            Eigen::VectorXf last_obs = active_tracks_[trk_idx].last_observation();
            left_trks.row(i) = last_obs.head<4>();
            left_trks_embs.row(i) = trk_embs_matrix.row(trk_idx);
        }
        
        // Compute IoU for OCR
        utils::AssociationFunction asso_func_ocr(img_w, img_h, asso_func_name_);
        Eigen::MatrixXf iou_left = asso_func_ocr(left_dets, left_trks);
        
        // Compute embedding cost for OCR
        Eigen::MatrixXf emb_cost_left;
        if (!embedding_off_ && left_dets_embs.rows() > 0 && left_trks_embs.rows() > 0) {
            // Ensure dimensions match
            if (left_dets_embs.cols() == left_trks_embs.cols()) {
                emb_cost_left = left_dets_embs * left_trks_embs.transpose();
            } else {
                // Dimension mismatch - create zero cost matrix
                emb_cost_left = Eigen::MatrixXf::Zero(left_dets.rows(), left_trks.rows());
            }
        } else {
            emb_cost_left = Eigen::MatrixXf::Zero(left_dets.rows(), left_trks.rows());
        }
        
        // Rematch if IoU is high enough
        if (iou_left.maxCoeff() > iou_threshold_) {
            Eigen::MatrixXf ocr_cost = -iou_left;
            auto ocr_assignment = utils::linear_assignment(ocr_cost, -iou_threshold_);
            
            std::vector<int> to_remove_det, to_remove_trk;
            for (const auto& match : ocr_assignment.matches) {
                int det_idx = assoc_result.unmatched_dets[match[0]];
                int trk_idx = assoc_result.unmatched_trks[match[1]];
                
                if (iou_left(match[0], match[1]) >= iou_threshold_) {
                    Eigen::VectorXf bbox = dets_filtered.row(det_idx).head<5>();
                    int cls = static_cast<int>(dets_filtered(det_idx, 5));
                    int det_ind = static_cast<int>(dets_filtered(det_idx, 6));
                    
                    active_tracks_[trk_idx].update(bbox, cls, det_ind);
                    active_tracks_[trk_idx].update_emb(dets_embs.row(det_idx), dets_alpha(det_idx));
                    
                    to_remove_det.push_back(det_idx);
                    to_remove_trk.push_back(trk_idx);
                }
            }
            
            // Remove rematched from unmatched lists
            std::unordered_set<int> remove_det_set(to_remove_det.begin(), to_remove_det.end());
            std::unordered_set<int> remove_trk_set(to_remove_trk.begin(), to_remove_trk.end());
            
            assoc_result.unmatched_dets.erase(
                std::remove_if(assoc_result.unmatched_dets.begin(), assoc_result.unmatched_dets.end(),
                              [&remove_det_set](int idx) { return remove_det_set.find(idx) != remove_det_set.end(); }),
                assoc_result.unmatched_dets.end());
            
            assoc_result.unmatched_trks.erase(
                std::remove_if(assoc_result.unmatched_trks.begin(), assoc_result.unmatched_trks.end(),
                              [&remove_trk_set](int idx) { return remove_trk_set.find(idx) != remove_trk_set.end(); }),
                assoc_result.unmatched_trks.end());
        }
    }
    
    // Update unmatched tracks (no detection)
    for (int trk_idx : assoc_result.unmatched_trks) {
        active_tracks_[trk_idx].update(Eigen::VectorXf(), 0, 0);
    }
    
    // Create new tracks for unmatched detections
    for (int det_idx : assoc_result.unmatched_dets) {
        Eigen::VectorXf bbox = dets_filtered.row(det_idx).head<5>();
        int cls = static_cast<int>(dets_filtered(det_idx, 5));
        int det_ind = static_cast<int>(dets_filtered(det_idx, 6));
        Eigen::VectorXf emb = dets_embs.row(det_idx);
        
        DeepOCSortKalmanBoxTracker trk(bbox, cls, det_ind, emb, delta_t_, max_obs_,
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
            output_ids.push_back(static_cast<float>(trk.id()));
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
    
    // Build output matrix: [x1, y1, x2, y2, id, conf, cls, det_ind]
    Eigen::MatrixXf output(output_bboxes.size(), 8);
    for (size_t i = 0; i < output_bboxes.size(); ++i) {
        output(i, 0) = output_bboxes[i](0);
        output(i, 1) = output_bboxes[i](1);
        output(i, 2) = output_bboxes[i](2);
        output(i, 3) = output_bboxes[i](3);
        output(i, 4) = output_ids[i];
        output(i, 5) = output_confs[i];
        output(i, 6) = output_clss[i];
        output(i, 7) = output_det_inds[i];
    }
    
    return output;
}

} // namespace motcpp::trackers

