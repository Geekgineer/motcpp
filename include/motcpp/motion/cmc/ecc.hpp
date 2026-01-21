// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/motion/cmc/cmc.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace motcpp::motion {

/**
 * Enhanced Correlation Coefficient (ECC) CMC
 * C++ implementation based on motion/cmc/ecc.py
 */
class ECC : public CMC {
public:
    ECC(int warp_mode = cv::MOTION_TRANSLATION,
        float eps = 1e-5f,
        int max_iter = 100,
        float scale = 0.15f,
        bool align = false,
        bool grayscale = true);
    
    /**
     * Apply ECC to compute warp matrix
     * @param img Current frame image
     * @param dets Optional detections (not used)
     * @return 2x3 warp matrix (or 3x3 for homography)
     */
    Eigen::Matrix<float, 2, 3> apply(const cv::Mat& img,
                                     const Eigen::MatrixXf& dets = Eigen::MatrixXf()) override;

private:
    int warp_mode_;
    float eps_;
    int max_iter_;
    float scale_;
    bool align_;
    bool grayscale_;
    cv::TermCriteria termination_criteria_;
    cv::Mat prev_img_;
};

} // namespace motcpp::motion

