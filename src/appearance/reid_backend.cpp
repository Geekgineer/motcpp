// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/appearance/reid_backend.hpp>
#include <algorithm>
#include <cmath>

namespace motcpp::appearance {

Eigen::MatrixXf ReIDBackend::get_crops(const Eigen::MatrixXf& xyxys, const cv::Mat& img) {
    if (xyxys.rows() == 0) {
        return Eigen::MatrixXf(0, 0);
    }

    int h = img.rows;
    int w = img.cols;
    int num_crops = xyxys.rows();
    int crop_h = input_shape_.first;
    int crop_w = input_shape_.second;

    // Preallocate output: (N, C, H, W) = (N, 3, H, W)
    // Flattened to (N, 3*H*W) for easier processing
    Eigen::MatrixXf crops(num_crops, 3 * crop_h * crop_w);

    for (int i = 0; i < num_crops; ++i) {
        // Extract bounding box coordinates
        int x1 = static_cast<int>(std::round(xyxys(i, 0)));
        int y1 = static_cast<int>(std::round(xyxys(i, 1)));
        int x2 = static_cast<int>(std::round(xyxys(i, 2)));
        int y2 = static_cast<int>(std::round(xyxys(i, 3)));

        // Clamp to image boundaries
        x1 = std::max(0, std::min(x1, w));
        y1 = std::max(0, std::min(y1, h));
        x2 = std::max(0, std::min(x2, w));
        y2 = std::max(0, std::min(y2, h));

        // Extract crop
        cv::Mat crop = img(cv::Range(y1, y2), cv::Range(x1, x2));

        // Resize to model input size
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(crop_w, crop_h), 0, 0, cv::INTER_LINEAR);

        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // Convert to float and normalize to [0, 1]
        rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

        // Rearrange to CHW format and normalize
        // OpenCV stores as (H, W, C), we need (C, H, W)
        float* crop_data = crops.row(i).data();
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < crop_h; ++y) {
                for (int x = 0; x < crop_w; ++x) {
                    float pixel = rgb.at<cv::Vec3f>(y, x)[c];
                    // Normalize: (pixel - mean) / std
                    pixel = (pixel - mean_(c)) / std_(c);
                    crop_data[c * crop_h * crop_w + y * crop_w + x] = pixel;
                }
            }
        }
    }

    return crops;
}

Eigen::MatrixXf ReIDBackend::normalize_features(const Eigen::MatrixXf& features) {
    if (features.rows() == 0) {
        return features;
    }

    Eigen::MatrixXf normalized = features;

    // L2 normalize each row (each feature vector)
    for (int i = 0; i < normalized.rows(); ++i) {
        float norm = normalized.row(i).norm();
        if (norm > 1e-6f) {
            normalized.row(i) /= norm;
        }
    }

    return normalized;
}

std::pair<int, int> ReIDBackend::determine_input_shape(const std::string& model_name) {
    // Check for vehicle datasets (typically square input)
    if (model_name.find("vehicleid") != std::string::npos || 
        model_name.find("veri") != std::string::npos) {
        return {256, 256};
    }
    
    // Check for lmbn models
    if (model_name.find("lmbn") != std::string::npos) {
        return {384, 128};
    }
    
    // Check for hacnn models
    if (model_name.find("hacnn") != std::string::npos) {
        return {160, 64};
    }
    
    // Default: 256x128 (most common for person ReID)
    return {256, 128};
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> ReIDBackend::determine_normalization(
    const std::string& model_name) {
    // CLIP models use different normalization
    if (model_name.find("clip") != std::string::npos) {
        Eigen::Vector3f mean(0.5f, 0.5f, 0.5f);
        Eigen::Vector3f std(0.5f, 0.5f, 0.5f);
        return {mean, std};
    }
    
    // Standard ImageNet normalization
    Eigen::Vector3f mean(0.485f, 0.456f, 0.406f);
    Eigen::Vector3f std(0.229f, 0.224f, 0.225f);
    return {mean, std};
}

} // namespace motcpp::appearance

