// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/motion/cmc/sof.hpp>
#include <algorithm>

namespace motcpp::motion {

// Static member definitions
const cv::Size SOF::win_size_(21, 21);
const cv::TermCriteria SOF::term_criteria_(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

SOF::SOF(float scale)
    : scale_(scale)
    , initialized_(false)
{
    // Pre-allocate buffers
    prev_keypoints_.reserve(max_corners_);
    next_keypoints_.reserve(max_corners_);
    status_.reserve(max_corners_);
    err_.reserve(max_corners_);
}

Eigen::Matrix<float, 2, 3> SOF::apply(const cv::Mat& img, const Eigen::MatrixXf& /* dets */) {
    // Default identity transformation
    Eigen::Matrix<float, 2, 3> transform = Eigen::Matrix<float, 2, 3>::Identity();
    
    // Preprocess image
    cv::Mat frame_gray = preprocess(img, scale_, true);
    
    if (!initialized_) {
        // First frame: detect keypoints
        cv::goodFeaturesToTrack(
            frame_gray,
            prev_keypoints_,
            max_corners_,
            quality_level_,
            min_distance_,
            cv::Mat(),
            block_size_,
            use_harris_,
            k_
        );
        
        if (!prev_keypoints_.empty()) {
            // Refine keypoints to sub-pixel accuracy
            cv::cornerSubPix(
                frame_gray,
                prev_keypoints_,
                cv::Size(5, 5),
                cv::Size(-1, -1),
                term_criteria_
            );
            
            prev_frame_ = frame_gray.clone();
            initialized_ = true;
        }
        
        return transform;  // Return identity on first frame
    }
    
    // Compute optical flow
    next_keypoints_.clear();
    status_.clear();
    err_.clear();
    
    cv::calcOpticalFlowPyrLK(
        prev_frame_,
        frame_gray,
        prev_keypoints_,
        next_keypoints_,
        status_,
        err_,
        win_size_,
        max_level_,
        term_criteria_,
        cv::OPTFLOW_LK_GET_MIN_EIGENVALS,
        0.001
    );
    
    // Filter valid points
    std::vector<cv::Point2f> valid_prev, valid_next;
    for (size_t i = 0; i < prev_keypoints_.size(); ++i) {
        if (status_[i] == 1 && err_[i] < 50.0f) {
            valid_prev.push_back(prev_keypoints_[i]);
            valid_next.push_back(next_keypoints_[i]);
        }
    }
    
    // Need at least 4 points to estimate affine transformation (matching Python)
    if (valid_prev.size() < 4) {
        // Re-detect keypoints
        cv::goodFeaturesToTrack(
            frame_gray,
            prev_keypoints_,
            max_corners_,
            quality_level_,
            min_distance_,
            cv::Mat(),
            block_size_,
            use_harris_,
            k_
        );
        if (!prev_keypoints_.empty()) {
            cv::cornerSubPix(
                frame_gray,
                prev_keypoints_,
                cv::Size(5, 5),
                cv::Size(-1, -1),
                term_criteria_
            );
        }
        prev_frame_ = frame_gray.clone();
        return transform;  // Return identity
    }
    
    // Convert to OpenCV format
    std::vector<cv::Point2f> cv_valid_prev(valid_prev.begin(), valid_prev.end());
    std::vector<cv::Point2f> cv_valid_next(valid_next.begin(), valid_next.end());
    
    // Estimate affine transformation using RANSAC (matching Python)
    cv::Mat affine_mat, inliers;
    affine_mat = cv::estimateAffinePartial2D(
        cv_valid_prev, cv_valid_next, inliers, cv::RANSAC
    );
    
    if (affine_mat.empty()) {
        // Transformation estimation failed, return identity
        return transform;
    }
    
    // Convert OpenCV Mat to Eigen Matrix
    // OpenCV returns 2x3 matrix: [m00, m01, m02; m10, m11, m12]
    transform(0, 0) = affine_mat.at<double>(0, 0);
    transform(0, 1) = affine_mat.at<double>(0, 1);
    transform(0, 2) = affine_mat.at<double>(0, 2);
    transform(1, 0) = affine_mat.at<double>(1, 0);
    transform(1, 1) = affine_mat.at<double>(1, 1);
    transform(1, 2) = affine_mat.at<double>(1, 2);
    
    // Upscale translation if image was scaled down
    if (scale_ < 1.0f && scale_ > 0.0f) {
        transform(0, 2) /= scale_;
        transform(1, 2) /= scale_;
    }
    
    // Update for next frame: re-detect keypoints (matching Python behavior)
    cv::goodFeaturesToTrack(
        frame_gray,
        prev_keypoints_,
        max_corners_,
        quality_level_,
        min_distance_,
        cv::Mat(),
        block_size_,
        use_harris_,
        k_
    );
    
    if (prev_keypoints_.empty()) {
        // If detection fails, use tracked points
        prev_keypoints_ = valid_next;
    } else {
        // Refine new keypoints
        cv::cornerSubPix(
            frame_gray,
            prev_keypoints_,
            cv::Size(5, 5),
            cv::Size(-1, -1),
            term_criteria_
        );
    }
    
    prev_frame_ = frame_gray.clone();
    
    return transform;
}

} // namespace motcpp::motion

