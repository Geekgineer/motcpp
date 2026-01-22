// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <motcpp/appearance/reid_backend.hpp>
#include <string>
#include <memory>

#ifdef MOTCPP_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace motcpp::appearance {

/**
 * ONNX Runtime backend for ReID inference
 * Uses ONNX Runtime for optimized inference
 */
class ONNXBackend : public ReIDBackend {
public:
    /**
     * Constructor
     * @param model_path Path to ONNX model file
     * @param model_name Model name (for determining input shape and normalization)
     * @param use_half Use FP16 precision (if supported)
     * @param use_gpu Use GPU execution provider (if available)
     */
    ONNXBackend(const std::string& model_path,
                const std::string& model_name = "",
                bool use_half = false,
                bool use_gpu = false);

    ~ONNXBackend() override;

    /**
     * Extract features from bounding box crops
     */
    Eigen::MatrixXf get_features(const Eigen::MatrixXf& xyxys, const cv::Mat& img) override;

    /**
     * Get input shape required by the model
     */
    std::pair<int, int> get_input_shape() const override {
#ifdef MOTCPP_HAS_ONNX
        // Convert from model input shape (N, C, H, W) to (H, W)
        if (model_input_shape_.size() >= 4) {
            return {static_cast<int>(model_input_shape_[2]), static_cast<int>(model_input_shape_[3])};
        }
#endif
        return input_shape_;  // Fallback to base class value
    }

    /**
     * Warmup the model
     */
    void warmup() override;

private:
#ifdef MOTCPP_HAS_ONNX
    // ONNX Runtime session
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    
    // Model metadata
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> model_input_shape_;  // Full model input shape (N, C, H, W)
    std::vector<int64_t> output_shape_;
    
    // Memory allocator
    Ort::AllocatorWithDefaultOptions allocator_;
    
    // Pre-allocated buffers for performance
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    size_t max_batch_size_;
    bool supports_dynamic_batch_;
    int fixed_batch_size_;
    
    /**
     * Run inference on preprocessed crops
     * @param crops Preprocessed crops (N, C, H, W) as Eigen matrix
     * @return Features (N, feature_dim)
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& crops);
    
    /**
     * Preprocess crops for inference (handle half precision, NHWC conversion if needed)
     * @param crops Crops in NCHW format
     * @return Preprocessed crops
     */
    Eigen::MatrixXf inference_preprocess(const Eigen::MatrixXf& crops);
#else
    // Dummy implementation when ONNX Runtime is not available
    Eigen::MatrixXf forward(const Eigen::MatrixXf& /* crops */) {
        throw std::runtime_error("ONNX Runtime not available. Compile with MOTCPP_HAS_ONNX.");
    }
#endif

    std::string model_path_;
    std::string model_name_;
    bool use_gpu_;
};

} // namespace motcpp::appearance
