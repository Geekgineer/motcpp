// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <limits>
#include <algorithm>
#include <cmath>

namespace motcpp::utils {

// Chi-square distribution 0.95 quantile with N degrees of freedom
constexpr float chi2inv95[] = {
    3.8415f,  // 1
    5.9915f,  // 2
    7.8147f,  // 3
    9.4877f,  // 4
    11.070f,  // 5
    12.592f,  // 6
    14.067f,  // 7
    15.507f,  // 8
    16.919f   // 9
};

/**
 * Simple Hungarian algorithm implementation for linear assignment
 * Returns: matches (vector of [row, col] pairs), unmatched_a, unmatched_b
 */
struct LinearAssignmentResult {
    std::vector<std::array<int, 2>> matches;
    std::vector<int> unmatched_a;
    std::vector<int> unmatched_b;
};

/**
 * Hungarian algorithm (Jonker-Volgenant) for linear assignment
 * Cost matrix should be non-negative. Higher values mean higher cost.
 * Threshold: maximum cost allowed for a match
 */
LinearAssignmentResult linear_assignment(const Eigen::MatrixXf& cost_matrix, float thresh);

/**
 * Compute IoU-based distance between tracks and detections
 */
Eigen::MatrixXf iou_distance(const Eigen::MatrixXf& atracks, const Eigen::MatrixXf& btracks);

/**
 * Compute embedding distance (cosine or euclidean)
 */
Eigen::MatrixXf embedding_distance(const Eigen::MatrixXf& track_features, 
                                   const Eigen::MatrixXf& det_features,
                                   const std::string& metric = "cosine");

/**
 * Fuse motion information into cost matrix using Kalman filter gating
 */
template<typename KFType, typename TrackType>
Eigen::MatrixXf fuse_motion(KFType& kf,
                            const Eigen::MatrixXf& cost_matrix,
                            const std::vector<TrackType>& tracks,
                            const Eigen::MatrixXf& measurements,
                            bool only_position = false,
                            float lambda = 0.98f) {
    if (cost_matrix.size() == 0) {
        return cost_matrix;
    }
    
    int gating_dim = only_position ? 2 : 4;
    float gating_threshold = chi2inv95[gating_dim - 1];
    
    Eigen::MatrixXf fused_cost = cost_matrix;
    
    for (size_t row = 0; row < tracks.size(); ++row) {
        const auto& track = tracks[row];
        Eigen::MatrixXf gating_dist = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, "maha"
        );
        
        // Set infinite cost for gated detections
        for (int j = 0; j < cost_matrix.cols(); ++j) {
            if (gating_dist(j) > gating_threshold) {
                fused_cost(row, j) = std::numeric_limits<float>::infinity();
            } else {
                fused_cost(row, j) = lambda * cost_matrix(row, j) + 
                                    (1.0f - lambda) * gating_dist(j);
            }
        }
    }
    
    return fused_cost;
}

/**
 * Fuse IoU with ReID similarity
 */
Eigen::MatrixXf fuse_iou(const Eigen::MatrixXf& reid_cost_matrix,
                         const Eigen::MatrixXf& tracks_xyxy,
                         const Eigen::MatrixXf& detections_xyxy,
                         const Eigen::VectorXf& det_confs);

/**
 * Fuse IoU with detection confidence scores
 */
Eigen::MatrixXf fuse_score(const Eigen::MatrixXf& iou_cost_matrix,
                           const Eigen::VectorXf& det_confs);

/**
 * Return linear assignment result as tuple for convenient unpacking
 */
inline std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
linear_assignment(const Eigen::MatrixXf& cost_matrix, float thresh, bool return_tuple) {
    auto result = linear_assignment(cost_matrix, thresh);
    
    std::vector<std::pair<int, int>> matches;
    for (const auto& m : result.matches) {
        matches.emplace_back(m[0], m[1]);
    }
    
    return {matches, result.unmatched_a, result.unmatched_b};
}

/**
 * IoU distance for two track pointer vectors
 */
template<typename TrackType>
Eigen::MatrixXf iou_distance(const std::vector<TrackType*>& tracks_a, 
                              const std::vector<TrackType*>& tracks_b) {
    int m = static_cast<int>(tracks_a.size());
    int n = static_cast<int>(tracks_b.size());
    
    if (m == 0 || n == 0) {
        return Eigen::MatrixXf::Ones(m, n);
    }
    
    Eigen::MatrixXf atracks(m, 4);
    Eigen::MatrixXf btracks(n, 4);
    
    for (int i = 0; i < m; ++i) {
        Eigen::Vector4f bbox = tracks_a[i]->xyxy();
        atracks.row(i) = bbox.transpose();
    }
    
    for (int j = 0; j < n; ++j) {
        Eigen::Vector4f bbox = tracks_b[j]->xyxy();
        btracks.row(j) = bbox.transpose();
    }
    
    return iou_distance(atracks, btracks);
}

/**
 * IoU distance for track pointer vectors (for BoTSORT, etc.)
 */
template<typename TrackType>
Eigen::MatrixXf iou_distance(const std::vector<TrackType*>& tracks, 
                              const std::vector<TrackType>& detections) {
    int m = static_cast<int>(tracks.size());
    int n = static_cast<int>(detections.size());
    
    if (m == 0 || n == 0) {
        return Eigen::MatrixXf::Ones(m, n);
    }
    
    // Extract bboxes
    Eigen::MatrixXf atracks(m, 4);
    Eigen::MatrixXf btracks(n, 4);
    
    for (int i = 0; i < m; ++i) {
        Eigen::Vector4f bbox = tracks[i]->xyxy();
        atracks.row(i) = bbox.transpose();
    }
    
    for (int j = 0; j < n; ++j) {
        Eigen::Vector4f bbox = detections[j].xyxy();
        btracks.row(j) = bbox.transpose();
    }
    
    return iou_distance(atracks, btracks);
}

/**
 * Embedding distance for track pointer vectors
 */
template<typename TrackType>
Eigen::MatrixXf embedding_distance(const std::vector<TrackType*>& tracks,
                                    const std::vector<TrackType>& detections,
                                    const std::string& metric = "cosine") {
    int m = static_cast<int>(tracks.size());
    int n = static_cast<int>(detections.size());
    
    if (m == 0 || n == 0) {
        return Eigen::MatrixXf::Ones(m, n);
    }
    
    // Extract features
    Eigen::MatrixXf track_features(m, tracks[0]->smooth_feat().size());
    Eigen::MatrixXf det_features(n, detections[0].smooth_feat().size());
    
    for (int i = 0; i < m; ++i) {
        Eigen::VectorXf feat = tracks[i]->smooth_feat();
        if (feat.size() > 0) {
            track_features.row(i) = feat.transpose();
        } else {
            track_features.row(i).setZero();
        }
    }
    
    for (int j = 0; j < n; ++j) {
        Eigen::VectorXf feat = detections[j].smooth_feat();
        if (feat.size() > 0) {
            det_features.row(j) = feat.transpose();
        } else {
            det_features.row(j).setZero();
        }
    }
    
    return embedding_distance(track_features, det_features, metric);
}

/**
 * Fuse IoU cost with detection scores for detection vectors
 */
template<typename TrackType>
Eigen::MatrixXf fuse_score(const Eigen::MatrixXf& iou_cost_matrix,
                            const std::vector<TrackType>& detections) {
    Eigen::VectorXf det_confs(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        det_confs(i) = detections[i].conf();
    }
    return fuse_score(iou_cost_matrix, det_confs);
}

} // namespace motcpp::utils

