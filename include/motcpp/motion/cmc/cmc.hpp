// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace motcpp::motion {

/**
 * Base class for Camera Motion Compensation (CMC)
 * Estimates affine transformation between consecutive frames
 */
class CMC {
public:
    virtual ~CMC() = default;
    
    /**
     * Apply CMC to estimate affine transformation
     * @param img Current frame image (BGR format)
     * @param dets Optional detections (not used by all methods)
     * @return 2x3 affine transformation matrix
     */
    virtual Eigen::Matrix<float, 2, 3> apply(const cv::Mat& img, 
                                              const Eigen::MatrixXf& dets = Eigen::MatrixXf()) = 0;
    
protected:
    /**
     * Preprocess image (convert to grayscale, resize if needed)
     */
    cv::Mat preprocess(const cv::Mat& img, float scale = 1.0f, bool grayscale = true);
};

} // namespace motcpp::motion

