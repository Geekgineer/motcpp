// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <filesystem>
#include <iomanip>

namespace motcpp::utils {

/**
 * Convert tracking results to MOT challenge format
 * Input: tracks (N, 8) as [x1, y1, x2, y2, id, conf, cls, det_ind]
 * Output: MOT format matrix (N, 10) as [frame_id, track_id, x1, y1, w, h, conf, x, y, z]
 * MOT Challenge format requires 10 fields: frame, id, x1, y1, w, h, conf, x, y, z
 * For 2D tracking, x, y, z are set to -1
 */
inline Eigen::MatrixXf convert_to_mot_format(const Eigen::MatrixXf& tracks, int frame_id) {
    if (tracks.rows() == 0) {
        return Eigen::MatrixXf(0, 10);
    }
    
    Eigen::MatrixXf mot_results(tracks.rows(), 10);
    
    for (int i = 0; i < tracks.rows(); ++i) {
        float x1 = tracks(i, 0);
        float y1 = tracks(i, 1);
        float x2 = tracks(i, 2);
        float y2 = tracks(i, 3);
        float w = x2 - x1;
        float h = y2 - y1;
        
        mot_results(i, 0) = static_cast<float>(frame_id);
        mot_results(i, 1) = tracks(i, 4);  // track_id
        mot_results(i, 2) = x1;            // x1 (top-left)
        mot_results(i, 3) = y1;            // y1 (top-left)
        mot_results(i, 4) = w;             // width
        mot_results(i, 5) = h;             // height
        mot_results(i, 6) = tracks(i, 5);  // confidence
        mot_results(i, 7) = -1.0f;         // x (world coordinate, -1 for 2D)
        mot_results(i, 8) = -1.0f;         // y (world coordinate, -1 for 2D)
        mot_results(i, 9) = -1.0f;         // z (world coordinate, -1 for 2D)
    }
    
    return mot_results;
}

/**
 * Write MOT format results to file
 * Format: frame_id,track_id,x1,y1,w,h,conf,x,y,z (10 fields)
 * MOT Challenge format requires 10 fields
 */
inline void write_mot_results(const std::filesystem::path& output_path,
                              const Eigen::MatrixXf& mot_results) {
    std::filesystem::create_directories(output_path.parent_path());
    
    std::ofstream file(output_path, std::ios::app);
    file << std::fixed << std::setprecision(6);
    
    for (int i = 0; i < mot_results.rows(); ++i) {
        file << static_cast<int>(mot_results(i, 0)) << ","  // frame_id
             << static_cast<int>(mot_results(i, 1)) << ","  // track_id
             << static_cast<int>(mot_results(i, 2)) << ","  // x1
             << static_cast<int>(mot_results(i, 3)) << ","  // y1
             << static_cast<int>(mot_results(i, 4)) << ","  // w
             << static_cast<int>(mot_results(i, 5)) << ","  // h
             << mot_results(i, 6) << ","                   // conf
             << static_cast<int>(mot_results(i, 7)) << ","  // x (world coord, -1 for 2D)
             << static_cast<int>(mot_results(i, 8)) << ","  // y (world coord, -1 for 2D)
             << static_cast<int>(mot_results(i, 9)) << "\n"; // z (world coord, -1 for 2D)
    }
}

} // namespace motcpp::utils

