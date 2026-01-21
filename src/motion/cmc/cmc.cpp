// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/motion/cmc/cmc.hpp>

namespace motcpp::motion {

cv::Mat CMC::preprocess(const cv::Mat& img, float scale, bool grayscale) {
    cv::Mat processed = img;
    
    // Convert to grayscale if needed
    if (grayscale && img.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
    }
    
    // Resize if scale < 1.0
    if (scale < 1.0f && scale > 0.0f) {
        cv::Size new_size(static_cast<int>(img.cols * scale), 
                         static_cast<int>(img.rows * scale));
        cv::resize(processed, processed, new_size, 0, 0, cv::INTER_LINEAR);
    }
    
    return processed;
}

} // namespace motcpp::motion

