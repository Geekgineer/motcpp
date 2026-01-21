// SPDX-License-Identifier: AGPL-3.0
// Copyright (c) 2026 motcpp contributors

#include <gtest/gtest.h>
#include <motcpp/utils/iou.hpp>
#include <motcpp/utils/ops.hpp>

namespace motcpp::utils::test {

class IoUTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test boxes [x1, y1, x2, y2]
        box1_ = Eigen::MatrixXf(1, 4);
        box1_ << 0, 0, 100, 100;
        
        box2_ = Eigen::MatrixXf(1, 4);
        box2_ << 50, 50, 150, 150;
        
        box3_ = Eigen::MatrixXf(1, 4);
        box3_ << 200, 200, 300, 300;
    }
    
    Eigen::MatrixXf box1_, box2_, box3_;
};

TEST_F(IoUTest, IdenticalBoxesHaveIoUOne) {
    Eigen::MatrixXf iou = iou_batch(box1_, box1_);
    EXPECT_FLOAT_EQ(iou(0, 0), 1.0f);
}

TEST_F(IoUTest, NonOverlappingBoxesHaveIoUZero) {
    Eigen::MatrixXf iou = iou_batch(box1_, box3_);
    EXPECT_FLOAT_EQ(iou(0, 0), 0.0f);
}

TEST_F(IoUTest, OverlappingBoxesHaveCorrectIoU) {
    Eigen::MatrixXf iou = iou_batch(box1_, box2_);
    
    // Intersection: 50*50 = 2500
    // Union: 100*100 + 100*100 - 2500 = 17500
    // IoU: 2500/17500 ≈ 0.143
    EXPECT_NEAR(iou(0, 0), 0.143f, 0.01f);
}

TEST_F(IoUTest, BatchIoUComputation) {
    Eigen::MatrixXf boxes_a(2, 4);
    boxes_a << 0, 0, 100, 100,
               50, 50, 150, 150;
    
    Eigen::MatrixXf boxes_b(2, 4);
    boxes_b << 0, 0, 100, 100,
               200, 200, 300, 300;
    
    Eigen::MatrixXf iou_matrix = iou_batch(boxes_a, boxes_b);
    
    EXPECT_EQ(iou_matrix.rows(), 2);
    EXPECT_EQ(iou_matrix.cols(), 2);
    
    // boxes_a[0] vs boxes_b[0] should be 1.0
    EXPECT_FLOAT_EQ(iou_matrix(0, 0), 1.0f);
    
    // boxes_a[0] vs boxes_b[1] should be 0.0 (no overlap)
    EXPECT_FLOAT_EQ(iou_matrix(0, 1), 0.0f);
}

TEST_F(IoUTest, EmptyBatchReturnsEmptyMatrix) {
    Eigen::MatrixXf empty(0, 4);
    Eigen::MatrixXf result = iou_batch(empty, box1_);
    
    EXPECT_EQ(result.rows(), 0);
    EXPECT_EQ(result.cols(), 1);
}

TEST_F(IoUTest, GIoUComputation) {
    Eigen::MatrixXf giou = giou_batch(box1_, box2_);
    
    // GIoU should be in [0, 1] after scaling
    EXPECT_GE(giou(0, 0), 0.0f);
    EXPECT_LE(giou(0, 0), 1.0f);
}

TEST_F(IoUTest, DIoUComputation) {
    Eigen::MatrixXf diou = diou_batch(box1_, box2_);
    
    // DIoU should be in [0, 1] after scaling
    EXPECT_GE(diou(0, 0), 0.0f);
    EXPECT_LE(diou(0, 0), 1.0f);
}

TEST_F(IoUTest, CIoUComputation) {
    Eigen::MatrixXf ciou = ciou_batch(box1_, box2_);
    
    // CIoU should be in [0, 1] after scaling
    EXPECT_GE(ciou(0, 0), 0.0f);
    EXPECT_LE(ciou(0, 0), 1.0f);
}

TEST_F(IoUTest, AssociationFunctionIoU) {
    AssociationFunction asso(640, 480, "iou");
    Eigen::MatrixXf result = asso(box1_, box2_);
    
    EXPECT_EQ(result.rows(), 1);
    EXPECT_EQ(result.cols(), 1);
    EXPECT_NEAR(result(0, 0), 0.143f, 0.01f);
}

TEST_F(IoUTest, CentroidDistance) {
    Eigen::MatrixXf dist = centroid_batch(box1_, box3_, 640, 480);
    
    // Centroid of box1: (50, 50)
    // Centroid of box3: (250, 250)
    // Distance should be non-zero
    EXPECT_GT(dist(0, 0), 0.0f);
    EXPECT_LT(dist(0, 0), 1.0f);
}

} // namespace motcpp::utils::test
