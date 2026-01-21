// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/tracker.hpp>
#include <motcpp/utils/iou.hpp>
#include <motcpp/utils/ops.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace motcpp {

BaseTracker::BaseTracker(float det_thresh, int max_age, int max_obs, int min_hits,
                       float iou_threshold, bool per_class, int nr_classes,
                       const std::string& asso_func, bool is_obb)
    : det_thresh_(det_thresh)
    , max_age_(max_age)
    , max_obs_(max_obs)
    , min_hits_(min_hits)
    , iou_threshold_(iou_threshold)
    , per_class_(per_class)
    , nr_classes_(nr_classes)
    , asso_func_name_(asso_func)
    , is_obb_(is_obb)
    , frame_count_(0)
    , first_frame_processed_(false)
    , first_dets_processed_(false)
    , last_emb_size_(-1)
    , frame_width_(0)
    , frame_height_(0)
    , target_id_(-1)
{
    if (max_age_ >= max_obs_) {
        max_obs_ = max_age_ + 5;
    }
    
    if (per_class_) {
        for (int i = 0; i < nr_classes_; ++i) {
            per_class_active_tracks_[i] = std::vector<void*>();
        }
    }
}

void BaseTracker::reset() {
    frame_count_ = 0;
    active_tracks_.clear();
    for (auto& [cls, tracks] : per_class_active_tracks_) {
        tracks.clear();
    }
    first_frame_processed_ = false;
    first_dets_processed_ = false;
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> BaseTracker::get_class_dets_n_embs(
    const Eigen::MatrixXf& dets,
    const Eigen::MatrixXf& embs,
    int cls_id) {
    
    Eigen::MatrixXf class_dets(0, dets.cols());
    Eigen::MatrixXf class_embs;
    
    if (dets.rows() == 0) {
        if (last_emb_size_ > 0 && embs.rows() > 0) {
            class_embs = Eigen::MatrixXf(0, last_emb_size_);
        }
        return {class_dets, class_embs};
    }
    
    // Filter by class
    std::vector<int> class_indices;
    int cls_col = dets.cols() - 1; // Last column is class
    for (int i = 0; i < dets.rows(); ++i) {
        if (std::abs(dets(i, cls_col) - cls_id) < 1e-5f) {
            class_indices.push_back(i);
        }
    }
    
    if (class_indices.empty()) {
        if (last_emb_size_ > 0 && embs.rows() > 0) {
            class_embs = Eigen::MatrixXf(0, last_emb_size_);
        }
        return {class_dets, class_embs};
    }
    
    class_dets = Eigen::MatrixXf(class_indices.size(), dets.cols());
    for (size_t i = 0; i < class_indices.size(); ++i) {
        class_dets.row(i) = dets.row(class_indices[i]);
    }
    
    if (embs.rows() > 0) {
        if (embs.rows() != dets.rows()) {
            throw std::runtime_error("Detections and embeddings must have same number of rows");
        }
        class_embs = Eigen::MatrixXf(class_indices.size(), embs.cols());
        for (size_t i = 0; i < class_indices.size(); ++i) {
            class_embs.row(i) = embs.row(class_indices[i]);
        }
        last_emb_size_ = embs.cols();
    }
    
    return {class_dets, class_embs};
}

void BaseTracker::check_inputs(const Eigen::MatrixXf& dets, const cv::Mat& img,
                              const Eigen::MatrixXf& embs) const {
    if (dets.rows() > 0 && dets.cols() != 6 && dets.cols() != 7) {
        throw std::invalid_argument("Detections must have 6 (AABB) or 7 (OBB) columns");
    }
    
    if (img.empty()) {
        throw std::invalid_argument("Image cannot be empty");
    }
    
    if (embs.rows() > 0 && dets.rows() != embs.rows()) {
        throw std::invalid_argument("Detections and embeddings must have same number of rows");
    }
    
    if (is_obb_ && dets.rows() > 0 && dets.cols() != 7) {
        throw std::invalid_argument("OBB mode requires 7 columns in detections");
    }
}

cv::Scalar BaseTracker::id_to_color(int id, float saturation, float value) const {
    if (target_id_ >= 0 && id == target_id_) {
        return cv::Scalar(0, 255, 0); // Green for target
    }
    
    // Hash-based color generation
    std::hash<int> hasher;
    size_t hash = hasher(id);
    float hue = (hash % 360) / 360.0f;
    
    // HSV to BGR conversion
    float c = value * saturation;
    float x = c * (1.0f - std::abs(std::fmod(hue * 6.0f, 2.0f) - 1.0f));
    float m = value - c;
    
    float r, g, b;
    if (hue < 1.0f/6.0f) {
        r = c; g = x; b = 0;
    } else if (hue < 2.0f/6.0f) {
        r = x; g = c; b = 0;
    } else if (hue < 3.0f/6.0f) {
        r = 0; g = c; b = x;
    } else if (hue < 4.0f/6.0f) {
        r = 0; g = x; b = c;
    } else if (hue < 5.0f/6.0f) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }
    
    return cv::Scalar((b + m) * 255, (g + m) * 255, (r + m) * 255);
}

cv::Mat BaseTracker::plot_results(const cv::Mat& img, bool /* show_trajectories */,
                                 int /* thickness */, float /* fontscale */) const {
    // This is a placeholder - subclasses should override with track-specific visualization
    return img.clone();
}

void BaseTracker::setup_association_function(const cv::Mat& img) {
    if (!first_frame_processed_ && !img.empty()) {
        frame_height_ = img.rows;
        frame_width_ = img.cols;
        first_frame_processed_ = true;
    }
}

void BaseTracker::setup_detection_format(const Eigen::MatrixXf& dets) {
    if (!first_dets_processed_ && dets.rows() > 0) {
        if (dets.cols() == 6) {
            is_obb_ = false;
        } else if (dets.cols() == 7) {
            is_obb_ = true;
        }
        first_dets_processed_ = true;
    }
}

} // namespace motcpp

