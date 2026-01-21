// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/utils/matching.hpp>
#include <motcpp/utils/iou.hpp>
#include <motcpp/association/lap_solver.hpp>
#include <numeric>
#include <cassert>

namespace motcpp::utils {

// Linear assignment using LAP Solver (Jonker-Volgenant algorithm)
// OPTIMIZED: Direct float processing to avoid MatrixXf->MatrixXd conversion
LinearAssignmentResult linear_assignment(const Eigen::MatrixXf& cost_matrix, float thresh) {
    LinearAssignmentResult result;
    
    int n = cost_matrix.rows();
    int m = cost_matrix.cols();
    
    if (n == 0 || m == 0) {
        // All unmatched
        result.unmatched_a.reserve(n);
        result.unmatched_b.reserve(m);
        for (int i = 0; i < n; ++i) result.unmatched_a.push_back(i);
        for (int j = 0; j < m; ++j) result.unmatched_b.push_back(j);
        return result;
    }
    
    // OPTIMIZED: Convert to double only for LAP solver (necessary for numerical stability)
    // Use map-based conversion to avoid full copy if possible, but LAP needs double precision
    Eigen::MatrixXd cost_matrix_d = cost_matrix.cast<double>();
    
    // Solve using LAP Solver
    std::vector<std::vector<int>> matches_vec;
    std::vector<int> unmatched_a_vec, unmatched_b_vec;
    matches_vec.reserve(std::min(n, m));
    unmatched_a_vec.reserve(n);
    unmatched_b_vec.reserve(m);
    
    trackers::association::LAPSolver::linearAssignment(
        cost_matrix_d, 
        static_cast<double>(thresh),
        matches_vec,
        unmatched_a_vec,
        unmatched_b_vec
    );
    
    // OPTIMIZED: Pre-allocate and use emplace_back
    result.matches.reserve(matches_vec.size());
    for (const auto& match : matches_vec) {
        if (match.size() >= 2) {
            result.matches.emplace_back(std::array<int, 2>{{match[0], match[1]}});
        }
    }
    
    result.unmatched_a = std::move(unmatched_a_vec);
    result.unmatched_b = std::move(unmatched_b_vec);
    
    return result;
}

Eigen::MatrixXf iou_distance(const Eigen::MatrixXf& atracks, const Eigen::MatrixXf& btracks) {
    Eigen::MatrixXf ious = iou_batch(atracks, btracks);
    return Eigen::MatrixXf::Ones(ious.rows(), ious.cols()) - ious;
}

Eigen::MatrixXf embedding_distance(const Eigen::MatrixXf& track_features,
                                   const Eigen::MatrixXf& det_features,
                                   const std::string& metric) {
    int n_tracks = track_features.rows();
    int n_dets = det_features.rows();
    
    if (n_tracks == 0 || n_dets == 0) {
        return Eigen::MatrixXf::Zero(n_tracks, n_dets);
    }
    
    Eigen::MatrixXf cost_matrix(n_tracks, n_dets);
    
    if (metric == "cosine") {
        // Cosine distance: 1 - cosine_similarity
        for (int i = 0; i < n_tracks; ++i) {
            Eigen::VectorXf track_feat = track_features.row(i);
            float track_norm = track_feat.norm();
            
            for (int j = 0; j < n_dets; ++j) {
                Eigen::VectorXf det_feat = det_features.row(j);
                float det_norm = det_feat.norm();
                
                float cosine_sim = track_feat.dot(det_feat) / (track_norm * det_norm + 1e-10f);
                cost_matrix(i, j) = std::max(0.0f, 1.0f - cosine_sim);
            }
        }
    } else if (metric == "euclidean") {
        // Euclidean distance
        for (int i = 0; i < n_tracks; ++i) {
            Eigen::VectorXf track_feat = track_features.row(i);
            for (int j = 0; j < n_dets; ++j) {
                Eigen::VectorXf det_feat = det_features.row(j);
                cost_matrix(i, j) = (track_feat - det_feat).norm();
            }
        }
    } else {
        throw std::invalid_argument("Unknown metric: " + metric);
    }
    
    return cost_matrix;
}

Eigen::MatrixXf fuse_iou(const Eigen::MatrixXf& reid_cost_matrix,
                        const Eigen::MatrixXf& tracks_xyxy,
                        const Eigen::MatrixXf& detections_xyxy,
                        const Eigen::VectorXf& /* det_confs */) {
    if (reid_cost_matrix.size() == 0) {
        return reid_cost_matrix;
    }
    
    Eigen::MatrixXf reid_sim = Eigen::MatrixXf::Ones(reid_cost_matrix.rows(), 
                                                     reid_cost_matrix.cols()) - reid_cost_matrix;
    Eigen::MatrixXf iou_dist = iou_distance(tracks_xyxy, detections_xyxy);
    Eigen::MatrixXf iou_sim = Eigen::MatrixXf::Ones(iou_dist.rows(), iou_dist.cols()) - iou_dist;
    
    Eigen::MatrixXf fuse_sim = reid_sim.cwiseProduct(
        (Eigen::MatrixXf::Ones(iou_sim.rows(), iou_sim.cols()) + iou_sim) / 2.0f
    );
    
    Eigen::MatrixXf fuse_cost = Eigen::MatrixXf::Ones(fuse_sim.rows(), fuse_sim.cols()) - fuse_sim;
    return fuse_cost;
}

Eigen::MatrixXf fuse_score(const Eigen::MatrixXf& iou_cost_matrix,
                           const Eigen::VectorXf& det_confs) {
    if (iou_cost_matrix.size() == 0) {
        return iou_cost_matrix;
    }
    
    Eigen::MatrixXf iou_sim = Eigen::MatrixXf::Ones(iou_cost_matrix.rows(), 
                                                    iou_cost_matrix.cols()) - iou_cost_matrix;
    
    Eigen::MatrixXf det_confs_matrix = det_confs.transpose().replicate(iou_cost_matrix.rows(), 1);
    Eigen::MatrixXf fuse_sim = iou_sim.cwiseProduct(det_confs_matrix);
    
    return Eigen::MatrixXf::Ones(fuse_sim.rows(), fuse_sim.cols()) - fuse_sim;
}

} // namespace motcpp::utils

