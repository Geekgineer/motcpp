// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>

namespace motcpp::utils {

/**
 * Convert (x1, y1, x2, y2) to (xc, yc, w, h)
 */
inline Eigen::Vector4f xyxy2xywh(const Eigen::Vector4f& xyxy) {
    float x1 = xyxy(0), y1 = xyxy(1), x2 = xyxy(2), y2 = xyxy(3);
    float w = x2 - x1;
    float h = y2 - y1;
    float xc = x1 + w * 0.5f;
    float yc = y1 + h * 0.5f;
    return Eigen::Vector4f(xc, yc, w, h);
}

/**
 * Convert (xc, yc, w, h) to (x1, y1, x2, y2)
 */
inline Eigen::Vector4f xywh2xyxy(const Eigen::Vector4f& xywh) {
    float xc = xywh(0), yc = xywh(1), w = xywh(2), h = xywh(3);
    float x1 = xc - w * 0.5f;
    float y1 = yc - h * 0.5f;
    float x2 = xc + w * 0.5f;
    float y2 = yc + h * 0.5f;
    return Eigen::Vector4f(x1, y1, x2, y2);
}

/**
 * Convert (xc, yc, w, h) to (t, l, w, h) where (t, l) is top-left
 */
inline Eigen::Vector4f xywh2tlwh(const Eigen::Vector4f& xywh) {
    float xc = xywh(0), yc = xywh(1), w = xywh(2), h = xywh(3);
    float t = xc - w * 0.5f;
    float l = yc - h * 0.5f;
    return Eigen::Vector4f(t, l, w, h);
}

/**
 * Convert (t, l, w, h) to (xc, yc, w, h)
 */
inline Eigen::Vector4f tlwh2xywh(const Eigen::Vector4f& tlwh) {
    float t = tlwh(0), l = tlwh(1), w = tlwh(2), h = tlwh(3);
    float xc = t + w * 0.5f;
    float yc = l + h * 0.5f;
    return Eigen::Vector4f(xc, yc, w, h);
}

/**
 * Convert (t, l, w, h) to (x1, y1, x2, y2)
 */
inline Eigen::Vector4f tlwh2xyxy(const Eigen::Vector4f& tlwh) {
    float t = tlwh(0), l = tlwh(1), w = tlwh(2), h = tlwh(3);
    return Eigen::Vector4f(t, l, t + w, l + h);
}

/**
 * Convert (x1, y1, x2, y2) to (t, l, w, h)
 */
inline Eigen::Vector4f xyxy2tlwh(const Eigen::Vector4f& xyxy) {
    float x1 = xyxy(0), y1 = xyxy(1), x2 = xyxy(2), y2 = xyxy(3);
    float t = x1;
    float l = y1;
    float w = x2 - x1;
    float h = y2 - y1;
    return Eigen::Vector4f(t, l, w, h);
}

/**
 * Convert (t, l, w, h) to (xc, yc, a, h) where a = aspect ratio
 */
inline Eigen::Vector4f tlwh2xyah(const Eigen::Vector4f& tlwh) {
    float t = tlwh(0), l = tlwh(1), w = tlwh(2), h = tlwh(3);
    float xc = t + w * 0.5f;
    float yc = l + h * 0.5f;
    float a = (h > 0.0f) ? (w / h) : 0.0f;
    return Eigen::Vector4f(xc, yc, a, h);
}

/**
 * Convert (xc, yc, a, h) to (t, l, w, h)
 */
inline Eigen::Vector4f xyah2tlwh(const Eigen::Vector4f& xyah) {
    float xc = xyah(0), yc = xyah(1), a = xyah(2), h = xyah(3);
    float w = a * h;
    float t = xc - w * 0.5f;
    float l = yc - h * 0.5f;
    return Eigen::Vector4f(t, l, w, h);
}

/**
 * Convert (xc, yc, w, h) to (xc, yc, a, h)
 */
inline Eigen::Vector4f xywh2xyah(const Eigen::Vector4f& xywh) {
    float xc = xywh(0), yc = xywh(1), w = xywh(2), h = xywh(3);
    float a = (h > 0.0f) ? (w / h) : 0.0f;
    return Eigen::Vector4f(xc, yc, a, h);
}

/**
 * Convert (xc, yc, a, h) to (xc, yc, w, h)
 */
inline Eigen::Vector4f xyah2xywh(const Eigen::Vector4f& xyah) {
    float xc = xyah(0), yc = xyah(1), a = xyah(2), h = xyah(3);
    float w = a * h;
    return Eigen::Vector4f(xc, yc, w, h);
}

/**
 * Batch conversion: xyxy to xywh
 */
inline Eigen::MatrixXf xyxy2xywh_batch(const Eigen::MatrixXf& xyxy) {
    Eigen::MatrixXf xywh(xyxy.rows(), 4);
    for (int i = 0; i < xyxy.rows(); ++i) {
        xywh.row(i) = xyxy2xywh(xyxy.row(i).transpose()).transpose();
    }
    return xywh;
}

/**
 * Batch conversion: xywh to xyxy
 */
inline Eigen::MatrixXf xywh2xyxy_batch(const Eigen::MatrixXf& xywh) {
    Eigen::MatrixXf xyxy(xywh.rows(), 4);
    for (int i = 0; i < xywh.rows(); ++i) {
        xyxy.row(i) = xywh2xyxy(xywh.row(i).transpose()).transpose();
    }
    return xyxy;
}

/**
 * Batch conversion: xyxy to tlwh
 */
inline Eigen::MatrixXf xyxy2tlwh_batch(const Eigen::MatrixXf& xyxy) {
    Eigen::MatrixXf tlwh(xyxy.rows(), 4);
    for (int i = 0; i < xyxy.rows(); ++i) {
        tlwh.row(i) = xyxy2tlwh(xyxy.row(i).transpose()).transpose();
    }
    return tlwh;
}

/**
 * Batch conversion: tlwh to xyxy
 */
inline Eigen::MatrixXf tlwh2xyxy_batch(const Eigen::MatrixXf& tlwh) {
    Eigen::MatrixXf xyxy(tlwh.rows(), 4);
    for (int i = 0; i < tlwh.rows(); ++i) {
        xyxy.row(i) = tlwh2xyxy(tlwh.row(i).transpose()).transpose();
    }
    return xyxy;
}

/**
 * Batch conversion: tlwh to xyah
 */
inline Eigen::MatrixXf tlwh2xyah_batch(const Eigen::MatrixXf& tlwh) {
    Eigen::MatrixXf xyah(tlwh.rows(), 4);
    for (int i = 0; i < tlwh.rows(); ++i) {
        xyah.row(i) = tlwh2xyah(tlwh.row(i).transpose()).transpose();
    }
    return xyah;
}

/**
 * Batch conversion: xyah to tlwh
 */
inline Eigen::MatrixXf xyah2tlwh_batch(const Eigen::MatrixXf& xyah) {
    Eigen::MatrixXf tlwh(xyah.rows(), 4);
    for (int i = 0; i < xyah.rows(); ++i) {
        tlwh.row(i) = xyah2tlwh(xyah.row(i).transpose()).transpose();
    }
    return tlwh;
}

/**
 * Convert (x1, y1, x2, y2) to (x, y, s, r) where:
 *   x, y = center coordinates
 *   s = scale (area = w * h)
 *   r = aspect ratio (w / h)
 */
inline Eigen::Vector4f xyxy2xysr(const Eigen::Vector4f& xyxy) {
    float x1 = xyxy(0), y1 = xyxy(1), x2 = xyxy(2), y2 = xyxy(3);
    float w = x2 - x1;
    float h = y2 - y1;
    float xc = x1 + w * 0.5f;
    float yc = y1 + h * 0.5f;
    float s = w * h;  // scale (area)
    float r = (h > 1e-6f) ? (w / h) : 0.0f;  // aspect ratio
    return Eigen::Vector4f(xc, yc, s, r);
}

/**
 * Convert (x, y, s, r) to (x1, y1, x2, y2)
 */
inline Eigen::Vector4f xysr2xyxy(const Eigen::Vector4f& xysr) {
    float xc = xysr(0), yc = xysr(1), s = xysr(2), r = xysr(3);
    float w = std::sqrt(s * r);
    float h = s / w;
    float x1 = xc - w * 0.5f;
    float y1 = yc - h * 0.5f;
    float x2 = xc + w * 0.5f;
    float y2 = yc + h * 0.5f;
    return Eigen::Vector4f(x1, y1, x2, y2);
}

} // namespace motcpp::utils

