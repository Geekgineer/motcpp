// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>

namespace motcpp::appearance {

/**
 * Base class for ReID backends
 * Provides common interface for feature extraction from image crops
 */
class ReIDBackend {
public:
    virtual ~ReIDBackend() = default;

    /**
     * Extract features from bounding box crops
     * @param xyxys Bounding boxes in format [x1, y1, x2, y2] per row
     * @param img Input image (BGR format)
     * @return Feature matrix (N x feature_dim) where N is number of boxes
     */
    virtual Eigen::MatrixXf get_features(const Eigen::MatrixXf& xyxys, const cv::Mat& img) = 0;

    /**
     * Get input shape required by the model
     * @return (height, width) tuple
     */
    virtual std::pair<int, int> get_input_shape() const = 0;

    /**
     * Warmup the model by running inference on dummy data
     */
    virtual void warmup() = 0;

protected:
    /**
     * Extract and preprocess crops from bounding boxes
     * @param xyxys Bounding boxes in format [x1, y1, x2, y2] per row
     * @param img Input image (BGR format)
     * @return Preprocessed crops as float array (N, C, H, W) where C=3
     */
    Eigen::MatrixXf get_crops(const Eigen::MatrixXf& xyxys, const cv::Mat& img);

    /**
     * Normalize features using L2 norm
     * @param features Feature matrix (N x feature_dim)
     * @return Normalized features
     */
    Eigen::MatrixXf normalize_features(const Eigen::MatrixXf& features);

    // Input shape (height, width)
    std::pair<int, int> input_shape_;

    // Normalization parameters
    Eigen::Vector3f mean_;  // [0.485, 0.456, 0.406] or [0.5, 0.5, 0.5] for CLIP
    Eigen::Vector3f std_;   // [0.229, 0.224, 0.225] or [0.5, 0.5, 0.5] for CLIP

    // Use half precision (FP16)
    bool use_half_;

    /**
     * Determine input shape based on model name
     * @param model_name Model name (e.g., "osnet_x1_0_dukemtmcreid")
     * @return (height, width) tuple
     */
    static std::pair<int, int> determine_input_shape(const std::string& model_name);

    /**
     * Determine normalization parameters based on model name
     * @param model_name Model name
     * @return (mean, std) tuple
     */
    static std::pair<Eigen::Vector3f, Eigen::Vector3f> determine_normalization(const std::string& model_name);
};

} // namespace motcpp::appearance
