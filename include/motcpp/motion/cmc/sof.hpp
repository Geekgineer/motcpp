// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/motion/cmc/cmc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <vector>

namespace motcpp::motion {

/**
 * Sparse Optical Flow (SOF) based Camera Motion Compensation
 * Uses Lucas-Kanade optical flow to track features and estimate affine transformation
 */
class SOF : public CMC {
public:
    /**
     * Constructor
     * @param scale Scale factor for image preprocessing (default 0.15 for speed)
     */
    explicit SOF(float scale = 0.15f);
    
    /**
     * Apply SOF to estimate affine transformation between frames
     * @param img Current frame image (BGR format)
     * @param dets Optional detections (not used)
     * @return 2x3 affine transformation matrix
     */
    Eigen::Matrix<float, 2, 3> apply(const cv::Mat& img, 
                                     const Eigen::MatrixXf& dets = Eigen::MatrixXf()) override;
    
private:
    float scale_;
    bool initialized_;
    cv::Mat prev_frame_;
    std::vector<cv::Point2f> prev_keypoints_;
    
    // Pre-allocated buffers
    std::vector<cv::Point2f> next_keypoints_;
    std::vector<uchar> status_;
    std::vector<float> err_;
    
    // Feature detection parameters (matching Python)
    static constexpr int max_corners_ = 1000;
    static constexpr double quality_level_ = 0.01;
    static constexpr double min_distance_ = 1.0;
    static constexpr int block_size_ = 3;
    static constexpr bool use_harris_ = false;
    static constexpr double k_ = 0.04;
    
    // Optical flow parameters (matching Python)
    static const cv::Size win_size_;
    static constexpr int max_level_ = 3;
    static const cv::TermCriteria term_criteria_;
};

} // namespace motcpp::motion

