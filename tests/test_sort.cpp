// SPDX-License-Identifier: AGPL-3.0
// Copyright (c) 2026 motcpp contributors

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <motcpp/trackers/sort.hpp>

namespace motcpp::trackers::test {

class SortTest : public ::testing::Test {
protected:
    void SetUp() override {
        img_ = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Single detection
        single_det_ = Eigen::MatrixXf(1, 6);
        single_det_ << 100, 100, 200, 200, 0.9, 0;
        
        // Multiple detections
        multi_det_ = Eigen::MatrixXf(3, 6);
        multi_det_ << 100, 100, 200, 200, 0.9, 0,
                      300, 300, 400, 400, 0.8, 0,
                      500, 100, 600, 200, 0.7, 1;
    }
    
    cv::Mat img_;
    Eigen::MatrixXf single_det_;
    Eigen::MatrixXf multi_det_;
};

TEST_F(SortTest, TrackerInitialization) {
    Sort tracker;
    // Tracker initializes without crash
    EXPECT_TRUE(true);
}

TEST_F(SortTest, SingleDetectionTracking) {
    Sort tracker(0.3f, 1, 50, 1);  // min_hits=1, max_age=1
    
    auto tracks = tracker.update(single_det_, img_);
    
    EXPECT_EQ(tracks.cols(), 8);
    EXPECT_EQ(tracks.rows(), 1);
    
    // Check bounding box is reasonable
    EXPECT_GT(tracks(0, 2), tracks(0, 0));  // x2 > x1
    EXPECT_GT(tracks(0, 3), tracks(0, 1));  // y2 > y1
}

TEST_F(SortTest, MultipleFramesTracking) {
    Sort tracker(0.3f, 3, 50, 1);  // max_age=3, min_hits=1
    
    // Frame 1: Single detection
    tracker.update(single_det_, img_);
    
    // Frame 2: Same position
    tracker.update(single_det_, img_);
    
    // Frame 3: Moved slightly
    Eigen::MatrixXf moved_det(1, 6);
    moved_det << 110, 110, 210, 210, 0.9, 0;
    auto tracks = tracker.update(moved_det, img_);
    
    EXPECT_EQ(tracks.rows(), 1);
    
    // Track ID should be 1 (first assigned)
    EXPECT_EQ(static_cast<int>(tracks(0, 4)), 1);
}

TEST_F(SortTest, TrackDeletion) {
    Sort tracker(0.3f, 2, 50, 1);  // max_age=2
    
    // Frame 1: Detection
    tracker.update(single_det_, img_);
    
    // Frame 2: No detection (empty)
    Eigen::MatrixXf empty(0, 6);
    tracker.update(empty, img_);
    
    // Frame 3: Still no detection
    auto tracks = tracker.update(empty, img_);
    
    // Track should be deleted after max_age
    EXPECT_EQ(tracks.rows(), 0);
}

TEST_F(SortTest, MultiClassTracking) {
    Sort tracker(0.3f, 3, 50, 1, 0.3f, true, 80);  // per_class=true
    
    auto tracks = tracker.update(multi_det_, img_);
    
    // Should track objects from different classes separately
    EXPECT_EQ(tracks.cols(), 8);
}

TEST_F(SortTest, IoUThreshold) {
    Sort tracker(0.3f, 3, 50, 1, 0.9f);  // Very high IoU threshold
    
    // First frame
    tracker.update(single_det_, img_);
    
    // Second frame: Significantly moved detection
    Eigen::MatrixXf far_det(1, 6);
    far_det << 300, 300, 400, 400, 0.9, 0;
    auto tracks = tracker.update(far_det, img_);
    
    // Should not match due to low IoU, creates new track
    // Both tracks exist but old one not output (not updated)
}

TEST_F(SortTest, ConfidenceFiltering) {
    Sort tracker(0.5f);  // det_thresh=0.5
    
    Eigen::MatrixXf mixed_conf(2, 6);
    mixed_conf << 100, 100, 200, 200, 0.3, 0,  // Below threshold
                  300, 300, 400, 400, 0.7, 0;  // Above threshold
    
    // Even with min_hits=1, only high conf should be considered
    Sort high_thresh_tracker(0.5f, 3, 50, 1);
    auto tracks = high_thresh_tracker.update(mixed_conf, img_);
    
    // At most 1 track (the high confidence one)
    EXPECT_LE(tracks.rows(), 1);
}

TEST_F(SortTest, KalmanPrediction) {
    Sort tracker(0.3f, 5, 50, 1);  // max_age=5
    
    // Moving object
    for (int i = 0; i < 5; ++i) {
        Eigen::MatrixXf det(1, 6);
        det << 100 + i*10, 100 + i*10, 200 + i*10, 200 + i*10, 0.9, 0;
        tracker.update(det, img_);
    }
    
    // Miss one frame
    Eigen::MatrixXf empty(0, 6);
    tracker.update(empty, img_);
    
    // Object reappears at predicted position
    Eigen::MatrixXf det(1, 6);
    det << 160, 160, 260, 260, 0.9, 0;  // Approximate predicted position
    auto tracks = tracker.update(det, img_);
    
    // Should maintain same track ID
    EXPECT_EQ(tracks.rows(), 1);
    EXPECT_EQ(static_cast<int>(tracks(0, 4)), 1);
}

} // namespace motcpp::trackers::test
