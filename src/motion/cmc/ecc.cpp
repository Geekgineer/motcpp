// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/motion/cmc/ecc.hpp>
#include <motcpp/utils/matching.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

namespace motcpp::motion {

ECC::ECC(int warp_mode, float eps, int max_iter, float scale, bool align, bool grayscale)
    : warp_mode_(warp_mode)
    , eps_(eps)
    , max_iter_(max_iter)
    , scale_(scale)
    , align_(align)
    , grayscale_(grayscale)
    , termination_criteria_(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, max_iter, eps)
{
}

Eigen::Matrix<float, 2, 3> ECC::apply(const cv::Mat& img, const Eigen::MatrixXf& /* dets */) {
    Eigen::Matrix<float, 2, 3> warp_matrix = Eigen::Matrix<float, 2, 3>::Identity();
    
    if (warp_mode_ == cv::MOTION_HOMOGRAPHY) {
        // For homography, return 3x3 identity (but we return 2x3 for compatibility)
        return warp_matrix;
    }
    
    cv::Mat img_processed = preprocess(img, scale_, grayscale_);
    
    if (prev_img_.empty()) {
        prev_img_ = img_processed.clone();
        return warp_matrix;  // Return identity for first frame
    }
    
    // Initialize warp matrix
    cv::Mat warp_mat_cv;
    if (warp_mode_ == cv::MOTION_HOMOGRAPHY) {
        warp_mat_cv = cv::Mat::eye(3, 3, CV_32F);
    } else {
        warp_mat_cv = cv::Mat::eye(2, 3, CV_32F);
    }
    
    try {
        // Find transform using ECC
        (void)cv::findTransformECC(
            prev_img_,
            img_processed,
            warp_mat_cv,
            warp_mode_,
            termination_criteria_,
            cv::noArray(),
            1
        );
        
        // Convert OpenCV Mat to Eigen Matrix
        if (warp_mode_ == cv::MOTION_HOMOGRAPHY) {
            // Extract 2x3 from 3x3 homography (first 2 rows, first 3 cols)
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    warp_matrix(i, j) = warp_mat_cv.at<float>(i, j);
                }
            }
        } else {
            // Direct copy for 2x3 matrix
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    warp_matrix(i, j) = warp_mat_cv.at<float>(i, j);
                }
            }
        }
        
        // Upscale translation if image was scaled down
        if (scale_ < 1.0f && scale_ > 0.0f) {
            warp_matrix(0, 2) /= scale_;
            warp_matrix(1, 2) /= scale_;
        }
        
    } catch (const cv::Exception& e) {
        // Error 7 is StsNoConv (no convergence)
        if (e.code == cv::Error::StsNoConv) {
            // Return identity matrix
            return Eigen::Matrix<float, 2, 3>::Identity();
        } else {
            // Re-throw other errors
            throw;
        }
    }
    
    // Update previous image
    prev_img_ = img_processed.clone();
    
    return warp_matrix;
}

} // namespace motcpp::motion

