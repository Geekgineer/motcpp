// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <memory>
#include <unordered_map>
#include <deque>
#include <cstdint>
#include <string>
#include <utility>

namespace motcpp {

/**
 * Track state enumeration
 */
enum class TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

/**
 * Base tracker interface
 * All trackers should inherit from this class
 */
class BaseTracker {
public:
    /**
     * Constructor
     * @param det_thresh Detection threshold
     * @param max_age Maximum age before track is considered lost
     * @param max_obs Maximum number of observations to store
     * @param min_hits Minimum hits before track is confirmed
     * @param iou_threshold IoU threshold for matching
     * @param per_class Enable per-class tracking
     * @param nr_classes Number of classes (if per_class=true)
     * @param asso_func Association function name
     * @param is_obb Use oriented bounding boxes
     */
    BaseTracker(float det_thresh = 0.3f,
                int max_age = 30,
                int max_obs = 50,
                int min_hits = 3,
                float iou_threshold = 0.3f,
                bool per_class = false,
                int nr_classes = 80,
                const std::string& asso_func = "iou",
                bool is_obb = false);
    
    virtual ~BaseTracker() = default;
    
    /**
     * Update tracker with new detections
     * @param dets Detection matrix: (N, 6) for AABB [x1, y1, x2, y2, conf, cls]
     *             or (N, 7) for OBB [cx, cy, w, h, angle, conf, cls]
     * @param img Current frame image
     * @param embs Optional embeddings matrix: (N, emb_dim)
     * @return Tracked objects matrix: (M, 8) [x1, y1, x2, y2, id, conf, cls, det_ind]
     */
    virtual Eigen::MatrixXf update(const Eigen::MatrixXf& dets,
                                   const cv::Mat& img,
                                   const Eigen::MatrixXf& embs = Eigen::MatrixXf()) = 0;
    
    /**
     * Reset tracker state
     */
    virtual void reset();
    
    /**
     * Get class-specific detections and embeddings
     */
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> get_class_dets_n_embs(
        const Eigen::MatrixXf& dets,
        const Eigen::MatrixXf& embs,
        int cls_id);
    
    /**
     * Check input validity
     */
    void check_inputs(const Eigen::MatrixXf& dets, const cv::Mat& img, 
                     const Eigen::MatrixXf& embs = Eigen::MatrixXf()) const;
    
    /**
     * Plot results on image
     */
    cv::Mat plot_results(const cv::Mat& img, bool show_trajectories = false,
                        int thickness = 2, float fontscale = 0.5f) const;
    
    /**
     * Generate color for track ID
     */
    cv::Scalar id_to_color(int id, float saturation = 0.75f, float value = 0.95f) const;
    
protected:
    // Parameters
    float det_thresh_;
    int max_age_;
    int max_obs_;
    int min_hits_;
    float iou_threshold_;
    bool per_class_;
    int nr_classes_;
    std::string asso_func_name_;
    bool is_obb_;
    
    // State
    int frame_count_;
    std::vector<void*> active_tracks_;  // Type-erased, subclasses handle concrete types
    
    // Per-class tracking
    std::unordered_map<int, std::vector<void*>> per_class_active_tracks_;
    bool first_frame_processed_;
    bool first_dets_processed_;
    int last_emb_size_;
    
    // Frame dimensions (set on first frame)
    int frame_width_;
    int frame_height_;
    
    // Optional target ID for highlighting
    int target_id_;
    
    /**
     * Setup association function on first frame
     */
    void setup_association_function(const cv::Mat& img);
    
    /**
     * Setup detection format on first detection
     */
    void setup_detection_format(const Eigen::MatrixXf& dets);
};

} // namespace motcpp

