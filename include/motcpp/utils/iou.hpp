// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <functional>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>

namespace motcpp::utils {

/**
 * Compute IoU for a pair of oriented bounding boxes
 */
inline float iou_obb_pair(const Eigen::VectorXf& bbox1, const Eigen::VectorXf& bbox2) {
    float cx1 = bbox1(0), cy1 = bbox1(1), w1 = bbox1(2), h1 = bbox1(3), angle1 = bbox1(4);
    float cx2 = bbox2(0), cy2 = bbox2(1), w2 = bbox2(2), h2 = bbox2(3), angle2 = bbox2(4);
    
    cv::RotatedRect r1(cv::Point2f(cx1, cy1), cv::Size2f(w1, h1), angle1 * 180.0f / M_PI);
    cv::RotatedRect r2(cv::Point2f(cx2, cy2), cv::Size2f(w2, h2), angle2 * 180.0f / M_PI);
    
    std::vector<cv::Point2f> intersect;
    int ret = cv::rotatedRectangleIntersection(r1, r2, intersect);
    
    if (ret == 0 || intersect.empty()) {
        return 0.0f; // No intersection
    }
    
    float intersection_area = cv::contourArea(intersect);
    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - intersection_area;
    
    return (union_area > 0.0f) ? (intersection_area / union_area) : 0.0f;
}

/**
 * Batch IoU computation for axis-aligned bounding boxes
 * Input: bboxes1 (N, 4), bboxes2 (M, 4) as [x1, y1, x2, y2]
 * Output: (N, M) matrix of IoU values
 */
inline Eigen::MatrixXf iou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) {
    int N = bboxes1.rows();
    int M = bboxes2.rows();
    
    if (N == 0 || M == 0) {
        return Eigen::MatrixXf::Zero(N, M);
    }
    
    Eigen::MatrixXf iou_matrix(N, M);
    
    // Compute areas
    Eigen::VectorXf area1 = (bboxes1.col(2) - bboxes1.col(0)).cwiseProduct(
                            bboxes1.col(3) - bboxes1.col(1));
    Eigen::VectorXf area2 = (bboxes2.col(2) - bboxes2.col(0)).cwiseProduct(
                            bboxes2.col(3) - bboxes2.col(1));
    
    // Vectorized computation
    for (int i = 0; i < N; ++i) {
        Eigen::VectorXf bbox1 = bboxes1.row(i).transpose();
        for (int j = 0; j < M; ++j) {
            Eigen::VectorXf bbox2 = bboxes2.row(j).transpose();
            
            float xx1 = std::max(bbox1(0), bbox2(0));
            float yy1 = std::max(bbox1(1), bbox2(1));
            float xx2 = std::min(bbox1(2), bbox2(2));
            float yy2 = std::min(bbox1(3), bbox2(3));
            
            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float intersection = w * h;
            
            float union_area = area1(i) + area2(j) - intersection;
            iou_matrix(i, j) = (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
        }
    }
    
    return iou_matrix;
}

/**
 * Batch IoU computation for oriented bounding boxes
 */
inline Eigen::MatrixXf iou_batch_obb(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) {
    int N = bboxes1.rows();
    int M = bboxes2.rows();
    Eigen::MatrixXf iou_matrix = Eigen::MatrixXf::Zero(N, M);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            iou_matrix(i, j) = iou_obb_pair(bboxes1.row(i).transpose(), bboxes2.row(j).transpose());
        }
    }
    
    return iou_matrix;
}

/**
 * Height-modified IoU (hIoU)
 */
inline Eigen::MatrixXf hmiou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) {
    int N = bboxes1.rows();
    int M = bboxes2.rows();
    
    if (N == 0 || M == 0) {
        return Eigen::MatrixXf::Zero(N, M);
    }
    
    // Compute vertical overlap ratio
    Eigen::MatrixXf intersect_y1 = bboxes1.col(1).replicate(1, M).cwiseMax(
                                    bboxes2.col(1).replicate(N, 1));
    Eigen::MatrixXf intersect_y2 = bboxes1.col(3).replicate(1, M).cwiseMin(
                                  bboxes2.col(3).replicate(N, 1));
    Eigen::MatrixXf intersection_height = (intersect_y2 - intersect_y1).cwiseMax(0.0f);
    
    Eigen::MatrixXf union_y1 = bboxes1.col(1).replicate(1, M).cwiseMin(
                               bboxes2.col(1).replicate(N, 1));
    Eigen::MatrixXf union_y2 = bboxes1.col(3).replicate(1, M).cwiseMax(
                               bboxes2.col(3).replicate(N, 1));
    Eigen::MatrixXf union_height = (union_y2 - union_y1).cwiseMax(1e-10f);
    
    Eigen::MatrixXf o = intersection_height.cwiseQuotient(union_height);
    
    // Compute standard IoU
    Eigen::MatrixXf iou = iou_batch(bboxes1, bboxes2);
    
    // Modify IoU with vertical overlap ratio
    return iou.cwiseProduct(o);
}

/**
 * Generalized IoU (GIoU)
 */
inline Eigen::MatrixXf giou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) {
    int N = bboxes1.rows();
    int M = bboxes2.rows();
    
    if (N == 0 || M == 0) {
        return Eigen::MatrixXf::Zero(N, M);
    }
    
    // Compute standard IoU
    Eigen::MatrixXf iou = iou_batch(bboxes1, bboxes2);
    
    // Compute smallest enclosing box
    Eigen::MatrixXf xxc1 = bboxes1.col(0).replicate(1, M).cwiseMin(
                           bboxes2.col(0).replicate(N, 1));
    Eigen::MatrixXf yyc1 = bboxes1.col(1).replicate(1, M).cwiseMin(
                          bboxes2.col(1).replicate(N, 1));
    Eigen::MatrixXf xxc2 = bboxes1.col(2).replicate(1, M).cwiseMax(
                           bboxes2.col(2).replicate(N, 1));
    Eigen::MatrixXf yyc2 = bboxes1.col(3).replicate(1, M).cwiseMax(
                          bboxes2.col(3).replicate(N, 1));
    
    Eigen::MatrixXf wc = xxc2 - xxc1;
    Eigen::MatrixXf hc = yyc2 - yyc1;
    Eigen::MatrixXf area_enclose = wc.cwiseProduct(hc);
    
    Eigen::MatrixXf area1 = (bboxes1.col(2) - bboxes1.col(0)).cwiseProduct(
                            (bboxes1.col(3) - bboxes1.col(1))).replicate(1, M);
    Eigen::MatrixXf area2 = (bboxes2.col(2) - bboxes2.col(0)).cwiseProduct(
                            (bboxes2.col(3) - bboxes2.col(1))).replicate(N, 1);
    
    Eigen::MatrixXf intersection = (iou.array() * (area1 + area2).array() / (iou.array() + 1e-10f)).matrix();
    Eigen::MatrixXf union_area = area1 + area2 - intersection;
    
    Eigen::MatrixXf giou = iou - (area_enclose - union_area).cwiseQuotient((area_enclose.array() + 1e-10f).matrix());
    giou = (giou.array() + 1.0f) / 2.0f; // Resize from (-1,1) to (0,1)
    
    return giou;
}

/**
 * Complete IoU (CIoU)
 */
inline Eigen::MatrixXf ciou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) {
    int N = bboxes1.rows();
    int M = bboxes2.rows();
    
    if (N == 0 || M == 0) {
        return Eigen::MatrixXf::Zero(N, M);
    }
    
    const float epsilon = 1e-7f;
    
    // Compute IoU
    Eigen::MatrixXf iou = iou_batch(bboxes1, bboxes2);
    
    // Calculate center points
    Eigen::MatrixXf centerx1 = (bboxes1.col(0) + bboxes1.col(2)) / 2.0f;
    Eigen::MatrixXf centery1 = (bboxes1.col(1) + bboxes1.col(3)) / 2.0f;
    Eigen::MatrixXf centerx2 = (bboxes2.col(0) + bboxes2.col(2)) / 2.0f;
    Eigen::MatrixXf centery2 = (bboxes2.col(1) + bboxes2.col(3)) / 2.0f;
    
    // Squared center distance
    Eigen::MatrixXf inner_diag = (centerx1.replicate(1, M) - centerx2.replicate(N, 1)).array().square() +
                                 (centery1.replicate(1, M) - centery2.replicate(N, 1)).array().square();
    
    // Smallest enclosing box diagonal
    Eigen::MatrixXf xxc1 = bboxes1.col(0).replicate(1, M).cwiseMin(
                           bboxes2.col(0).replicate(N, 1));
    Eigen::MatrixXf yyc1 = bboxes1.col(1).replicate(1, M).cwiseMin(
                          bboxes2.col(1).replicate(N, 1));
    Eigen::MatrixXf xxc2 = bboxes1.col(2).replicate(1, M).cwiseMax(
                           bboxes2.col(2).replicate(N, 1));
    Eigen::MatrixXf yyc2 = bboxes1.col(3).replicate(1, M).cwiseMax(
                          bboxes2.col(3).replicate(N, 1));
    
    Eigen::MatrixXf outer_diag = (xxc2 - xxc1).array().square() + 
                                (yyc2 - yyc1).array().square() + epsilon;
    
    // Aspect ratio consistency
    Eigen::VectorXf w1 = bboxes1.col(2) - bboxes1.col(0);
    Eigen::VectorXf h1 = bboxes1.col(3) - bboxes1.col(1);
    Eigen::VectorXf w2 = bboxes2.col(2) - bboxes2.col(0);
    Eigen::VectorXf h2 = bboxes2.col(3) - bboxes2.col(1);
    
    Eigen::MatrixXf arctan_diff = (w2.replicate(N, 1).array() / (h2.replicate(N, 1).array() + epsilon)).atan() -
                                  (w1.replicate(1, M).array() / (h1.replicate(1, M).array() + epsilon)).atan();
    Eigen::MatrixXf v = (4.0f / (M_PI * M_PI)) * arctan_diff.array().square();
    
    // Alpha
    Eigen::MatrixXf S = Eigen::MatrixXf::Ones(N, M) - iou;
    Eigen::MatrixXf alpha = v.cwiseQuotient((S.array() + v.array() + epsilon).matrix());
    
    // CIoU
    Eigen::MatrixXf ciou = iou - inner_diag.cwiseQuotient(outer_diag) + alpha.cwiseProduct(v);
    
    // Scale to [0, 1]
    return (ciou.array() + 1.0f) / 2.0f;
}

/**
 * Distance IoU (DIoU)
 */
inline Eigen::MatrixXf diou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) {
    int N = bboxes1.rows();
    int M = bboxes2.rows();
    
    if (N == 0 || M == 0) {
        return Eigen::MatrixXf::Zero(N, M);
    }
    
    // Compute IoU
    Eigen::MatrixXf iou = iou_batch(bboxes1, bboxes2);
    
    // Center points
    Eigen::MatrixXf centerx1 = (bboxes1.col(0) + bboxes1.col(2)) / 2.0f;
    Eigen::MatrixXf centery1 = (bboxes1.col(1) + bboxes1.col(3)) / 2.0f;
    Eigen::MatrixXf centerx2 = (bboxes2.col(0) + bboxes2.col(2)) / 2.0f;
    Eigen::MatrixXf centery2 = (bboxes2.col(1) + bboxes2.col(3)) / 2.0f;
    
    // Inner diagonal
    Eigen::MatrixXf inner_diag = (centerx1.replicate(1, M) - centerx2.replicate(N, 1)).array().square() +
                                 (centery1.replicate(1, M) - centery2.replicate(N, 1)).array().square();
    
    // Outer diagonal
    Eigen::MatrixXf xxc1 = bboxes1.col(0).replicate(1, M).cwiseMin(
                           bboxes2.col(0).replicate(N, 1));
    Eigen::MatrixXf yyc1 = bboxes1.col(1).replicate(1, M).cwiseMin(
                          bboxes2.col(1).replicate(N, 1));
    Eigen::MatrixXf xxc2 = bboxes1.col(2).replicate(1, M).cwiseMax(
                           bboxes2.col(2).replicate(N, 1));
    Eigen::MatrixXf yyc2 = bboxes1.col(3).replicate(1, M).cwiseMax(
                          bboxes2.col(3).replicate(N, 1));
    
    Eigen::MatrixXf outer_diag = (xxc2 - xxc1).array().square() + 
                                (yyc2 - yyc1).array().square();
    
    Eigen::MatrixXf diou = iou - inner_diag.cwiseQuotient((outer_diag.array() + 1e-10f).matrix());
    
    return (diou.array() + 1.0f) / 2.0f;
}

/**
 * Centroid distance (normalized) for axis-aligned boxes
 */
inline Eigen::MatrixXf centroid_batch(const Eigen::MatrixXf& bboxes1, 
                                      const Eigen::MatrixXf& bboxes2,
                                      int frame_width, int frame_height) {
    int N = bboxes1.rows();
    int M = bboxes2.rows();
    
    if (N == 0 || M == 0) {
        return Eigen::MatrixXf::Zero(N, M);
    }
    
    // Compute centroids
    Eigen::MatrixXf centroids1(N, 2);
    centroids1.col(0) = (bboxes1.col(0) + bboxes1.col(2)) / 2.0f;
    centroids1.col(1) = (bboxes1.col(1) + bboxes1.col(3)) / 2.0f;
    
    Eigen::MatrixXf centroids2(M, 2);
    centroids2.col(0) = (bboxes2.col(0) + bboxes2.col(2)) / 2.0f;
    centroids2.col(1) = (bboxes2.col(1) + bboxes2.col(3)) / 2.0f;
    
    // Compute distances
    Eigen::MatrixXf distances(N, M);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float dx = centroids1(i, 0) - centroids2(j, 0);
            float dy = centroids1(i, 1) - centroids2(j, 1);
            distances(i, j) = std::sqrt(dx * dx + dy * dy);
        }
    }
    
    float norm_factor = std::sqrt(frame_width * frame_width + frame_height * frame_height);
    Eigen::MatrixXf normalized_distances = distances / norm_factor;
    
    return Eigen::MatrixXf::Ones(N, M) - normalized_distances;
}

/**
 * Centroid distance for oriented bounding boxes
 */
inline Eigen::MatrixXf centroid_batch_obb(const Eigen::MatrixXf& bboxes1,
                                          const Eigen::MatrixXf& bboxes2,
                                          int frame_width, int frame_height) {
    int N = bboxes1.rows();
    int M = bboxes2.rows();
    
    if (N == 0 || M == 0) {
        return Eigen::MatrixXf::Zero(N, M);
    }
    
    // Centroids are directly (cx, cy)
    Eigen::MatrixXf centroids1 = bboxes1.block(0, 0, N, 2);
    Eigen::MatrixXf centroids2 = bboxes2.block(0, 0, M, 2);
    
    // Compute distances
    Eigen::MatrixXf distances(N, M);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float dx = centroids1(i, 0) - centroids2(j, 0);
            float dy = centroids1(i, 1) - centroids2(j, 1);
            distances(i, j) = std::sqrt(dx * dx + dy * dy);
        }
    }
    
    float norm_factor = std::sqrt(frame_width * frame_width + frame_height * frame_height);
    Eigen::MatrixXf normalized_distances = distances / norm_factor;
    
    return Eigen::MatrixXf::Ones(N, M) - normalized_distances;
}

/**
 * Association function selector
 */
class AssociationFunction {
public:
    using AssoFunc = std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)>;
    
    AssociationFunction(int w, int h, const std::string& asso_mode = "iou")
        : frame_width_(w), frame_height_(h) {
        asso_func_ = get_asso_func(asso_mode);
    }
    
    Eigen::MatrixXf operator()(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2) const {
        return asso_func_(bboxes1, bboxes2);
    }
    
private:
    AssoFunc get_asso_func(const std::string& mode) {
        if (mode == "iou") {
            return iou_batch;
        } else if (mode == "iou_obb") {
            return iou_batch_obb;
        } else if (mode == "hmiou") {
            return hmiou_batch;
        } else if (mode == "giou") {
            return giou_batch;
        } else if (mode == "ciou") {
            return ciou_batch;
        } else if (mode == "diou") {
            return diou_batch;
        } else if (mode == "centroid") {
            return [this](const Eigen::MatrixXf& b1, const Eigen::MatrixXf& b2) {
                return centroid_batch(b1, b2, frame_width_, frame_height_);
            };
        } else if (mode == "centroid_obb") {
            return [this](const Eigen::MatrixXf& b1, const Eigen::MatrixXf& b2) {
                return centroid_batch_obb(b1, b2, frame_width_, frame_height_);
            };
        } else {
            throw std::invalid_argument("Invalid association mode: " + mode);
        }
    }
    
    int frame_width_;
    int frame_height_;
    AssoFunc asso_func_;
};

} // namespace motcpp::utils

