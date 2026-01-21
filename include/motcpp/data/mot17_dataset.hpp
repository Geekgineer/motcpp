// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <memory>
#include <unordered_map>

namespace motcpp::data {

/**
 * MOT17 sequence information
 */
struct SequenceInfo {
    std::string name;
    std::filesystem::path seq_dir;
    std::filesystem::path img_dir;
    std::filesystem::path det_path;
    std::filesystem::path gt_path;
    std::vector<int> frame_ids;
    std::vector<std::filesystem::path> frame_paths;
    int fps;
};

/**
 * Frame data with detections
 */
struct FrameData {
    int frame_id;
    cv::Mat image;
    Eigen::MatrixXf detections;  // (N, 5) as [x1, y1, x2, y2, conf] (converted from tlwh)
    Eigen::MatrixXf embeddings;  // (N, emb_dim) optional
};

/**
 * MOT17 Dataset loader
 */
class MOT17Dataset {
public:
    MOT17Dataset(const std::string& mot_root, 
                 const std::string& det_emb_root = "",
                 const std::string& model_name = "",
                 const std::string& reid_name = "");
    
    /**
     * Get list of sequence names
     */
    std::vector<std::string> sequence_names() const;
    
    /**
     * Get sequence info
     */
    SequenceInfo get_sequence_info(const std::string& seq_name) const;
    
    /**
     * Load detections for a sequence from det.txt file
     * Format options:
     *   - MOT17 format: frame_id, -1, x1, y1, w, h, conf
     *   - Pre-generated: frame_id x1 y1 x2 y2 conf cls (space-separated)
     * Returns: map of frame_id -> detections (xyxy format: [x1, y1, x2, y2, conf, cls])
     */
    std::unordered_map<int, Eigen::MatrixXf> load_detections(
        const std::filesystem::path& det_path) const;
    
    /**
     * Load embeddings for a sequence from embeddings file
     * Format: one embedding vector per line (space-separated floats)
     * Returns: map of frame_id -> embeddings (must match detection order)
     */
    std::unordered_map<int, Eigen::MatrixXf> load_embeddings(
        const std::filesystem::path& emb_path,
        const std::unordered_map<int, Eigen::MatrixXf>& detections) const;
    
    /**
     * Convert tlwh to xyxy format
     */
    static Eigen::Vector4f tlwh2xyxy(const Eigen::Vector4f& tlwh);
    
    /**
     * Convert xyxy to tlwh format
     */
    static Eigen::Vector4f xyxy2tlwh(const Eigen::Vector4f& xyxy);
    
    /**
     * Get frame data for a specific frame
     */
    FrameData get_frame(const SequenceInfo& seq_info, int frame_id,
                       const std::unordered_map<int, Eigen::MatrixXf>& detections,
                       const std::unordered_map<int, Eigen::MatrixXf>& embeddings = {}) const;
    
private:
    std::filesystem::path mot_root_;
    std::filesystem::path det_path_;
    std::vector<SequenceInfo> sequences_;
    
    void index_sequences();
    int read_seq_fps(const std::filesystem::path& seq_dir) const;
};

} // namespace motcpp::data

