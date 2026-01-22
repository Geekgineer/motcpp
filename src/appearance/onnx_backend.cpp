// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/appearance/onnx_backend.hpp>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#ifdef MOTCPP_HAS_ONNX

namespace motcpp::appearance {

ONNXBackend::ONNXBackend(const std::string& model_path,
                         const std::string& model_name,
                         bool use_half,
                         bool use_gpu)
    : ReIDBackend()
    , env_(ORT_LOGGING_LEVEL_WARNING, "motcppReID")
    , max_batch_size_(32)  // Pre-allocate for up to 32 crops
    , model_path_(model_path)
    , model_name_(model_name.empty() ? model_path : model_name)
    , use_gpu_(use_gpu)
{
    // Determine input shape and normalization from model name
    input_shape_ = determine_input_shape(model_name_);
    auto [mean, std] = determine_normalization(model_name_);
    mean_ = mean;
    std_ = std;
    use_half_ = use_half;  // Set inherited protected member

    // Configure session options
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Set execution providers
    std::vector<const char*> providers;
    if (use_gpu) {
        providers.push_back("CUDAExecutionProvider");
    }
    providers.push_back("CPUExecutionProvider");

    // Create session
    try {
        session_ = std::make_unique<Ort::Session>(
            env_, model_path_.c_str(), session_options_);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
    }

    // Get input/output names and shapes
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Input info
    size_t num_input_nodes = session_->GetInputCount();
    if (num_input_nodes != 1) {
        throw std::runtime_error("Expected single input node, got " + 
                                std::to_string(num_input_nodes));
    }
    
    input_name_ = session_->GetInputNameAllocated(0, allocator).get();
    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> temp_input_shape = input_tensor_info.GetShape();
    
    // Store model input shape (handle dynamic batch size)
    model_input_shape_.clear();
    for (size_t i = 0; i < temp_input_shape.size(); ++i) {
        if (temp_input_shape[i] == -1) {
            model_input_shape_.push_back(1);  // Use 1 for dynamic dimensions
        } else {
            model_input_shape_.push_back(temp_input_shape[i]);
        }
    }

    // Output info
    size_t num_output_nodes = session_->GetOutputCount();
    if (num_output_nodes != 1) {
        throw std::runtime_error("Expected single output node, got " + 
                                std::to_string(num_output_nodes));
    }
    
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
    Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape_ = output_tensor_info.GetShape();
    
    // Check if model supports dynamic batch size
    supports_dynamic_batch_ = (output_shape_[0] == -1);
    if (!supports_dynamic_batch_) {
        fixed_batch_size_ = static_cast<int>(output_shape_[0]);
    } else {
        fixed_batch_size_ = -1;
        output_shape_[0] = 1;  // Use 1 for pre-allocation
    }

    // Pre-allocate buffers
    int batch_size = max_batch_size_;
    int channels = static_cast<int>(model_input_shape_[1]);
    int height = static_cast<int>(model_input_shape_[2]);
    int width = static_cast<int>(model_input_shape_[3]);
    
    input_buffer_.resize(batch_size * channels * height * width);
    
    int feature_dim = output_shape_.back();
    output_buffer_.resize(batch_size * feature_dim);
}

ONNXBackend::~ONNXBackend() = default;

Eigen::MatrixXf ONNXBackend::get_features(const Eigen::MatrixXf& xyxys, const cv::Mat& img) {
    if (xyxys.rows() == 0) {
        return Eigen::MatrixXf(0, 0);
    }

    // If model has fixed batch size of 1, process one at a time
    if (!supports_dynamic_batch_ && fixed_batch_size_ == 1) {
        int feature_dim = static_cast<int>(output_shape_.back());
        Eigen::MatrixXf all_features(xyxys.rows(), feature_dim);
        for (int i = 0; i < xyxys.rows(); ++i) {
            Eigen::MatrixXf single_xyxy(1, 4);
            single_xyxy.row(0) = xyxys.row(i);
            Eigen::MatrixXf crops = get_crops(single_xyxy, img);
            crops = inference_preprocess(crops);
            Eigen::MatrixXf feat = forward(crops);
            if (feat.rows() > 0 && feat.cols() == feature_dim) {
                all_features.row(i) = feat.row(0);
            } else {
                // Fallback: zero feature if inference fails
                all_features.row(i).setZero();
            }
        }
        return normalize_features(all_features);
    }

    // Extract and preprocess crops
    Eigen::MatrixXf crops = get_crops(xyxys, img);
    
    // Preprocess for inference
    crops = inference_preprocess(crops);
    
    // Run inference
    Eigen::MatrixXf features = forward(crops);
    
    // Postprocess (L2 normalize)
    features = normalize_features(features);
    
    return features;
}

Eigen::MatrixXf ONNXBackend::inference_preprocess(const Eigen::MatrixXf& crops) {
    // For ONNX Runtime, we keep NCHW format (standard)
    // If half precision is requested, we'd convert here, but ONNX Runtime
    // typically expects float32, so we keep float32 for now
    // (Half precision support can be added later if needed)
    
    return crops;  // Already in correct format
}

Eigen::MatrixXf ONNXBackend::forward(const Eigen::MatrixXf& crops) {
    int batch_size = crops.rows();
    int channels = static_cast<int>(model_input_shape_[1]);
    int height = static_cast<int>(model_input_shape_[2]);
    int width = static_cast<int>(model_input_shape_[3]);
    
    // Prepare input tensor
    std::vector<int64_t> input_tensor_shape = {
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(channels),
        static_cast<int64_t>(height),
        static_cast<int64_t>(width)
    };
    
    // Copy crops to input buffer (ONNX Runtime expects contiguous memory)
    // Crops is (N, C*H*W), we need to ensure it's properly laid out
    std::memcpy(input_buffer_.data(), crops.data(), 
                batch_size * channels * height * width * sizeof(float));
    
    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_buffer_.data(),
        batch_size * channels * height * width,
        input_tensor_shape.data(),
        input_tensor_shape.size());
    
    // Run inference - let ONNX Runtime allocate output tensor
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                       input_names, &input_tensor, 1,
                                       output_names, 1);
    } catch (const std::exception& e) {
        throw std::runtime_error("ONNX Runtime inference failed: " + 
                                std::string(e.what()));
    }
    
    if (output_tensors.empty()) {
        throw std::runtime_error("ONNX Runtime returned no output tensors");
    }
    
    Ort::Value& output_tensor = output_tensors[0];
    
    // Get output shape
    auto output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();
    
    int actual_batch_size = static_cast<int>(output_dims[0]);
    int feature_dim = static_cast<int>(output_dims.back());
    
    // Extract features from output tensor
    float* output_data = output_tensor.GetTensorMutableData<float>();
    Eigen::MatrixXf features(actual_batch_size, feature_dim);
    std::memcpy(features.data(), output_data, 
                actual_batch_size * feature_dim * sizeof(float));
    
    return features;
}

void ONNXBackend::warmup() {
    // Create dummy crops for warmup
    int batch_size = 2;
    int channels = static_cast<int>(model_input_shape_[1]);
    int height = static_cast<int>(model_input_shape_[2]);
    int width = static_cast<int>(model_input_shape_[3]);
    
    Eigen::MatrixXf dummy_crops(batch_size, channels * height * width);
    dummy_crops.setRandom();  // Random values for warmup
    
    // Run inference
    try {
        forward(dummy_crops);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Warmup failed: " << e.what() << std::endl;
    }
}

} // namespace motcpp::appearance

#else  // MOTCPP_HAS_ONNX not defined

namespace motcpp::appearance {

ONNXBackend::ONNXBackend(const std::string& model_path,
                         const std::string& model_name,
                         bool use_half,
                         bool use_gpu)
    : ReIDBackend()
    , model_path_(model_path)
    , model_name_(model_name.empty() ? model_path : model_name)
    , use_gpu_(use_gpu)
{
    // Set input shape and normalization even without ONNX Runtime
    input_shape_ = determine_input_shape(model_name_);
    auto [mean, std] = determine_normalization(model_name_);
    mean_ = mean;
    std_ = std;
    use_half_ = use_half;
    
    throw std::runtime_error("ONNX Runtime not available. "
                            "Compile with MOTCPP_HAS_ONNX.");
}

ONNXBackend::~ONNXBackend() = default;

Eigen::MatrixXf ONNXBackend::get_features(const Eigen::MatrixXf& /* xyxys */, const cv::Mat& /* img */) {
    throw std::runtime_error("ONNX Runtime not available.");
}

void ONNXBackend::warmup() {
    throw std::runtime_error("ONNX Runtime not available.");
}

} // namespace motcpp::appearance

#endif  // MOTCPP_HAS_ONNX
