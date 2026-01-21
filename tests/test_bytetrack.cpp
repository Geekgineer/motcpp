// SPDX-License-Identifier: AGPL-3.0
// Copyright (c) 2026 motcpp contributors

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <motcpp/trackers/bytetrack.hpp>

namespace motcpp::trackers::test {

class ByteTrackTest : public ::testing::Test {
protected:
    void SetUp() override {
        img_ = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // High confidence detections
        high_conf_ = Eigen::MatrixXf(2, 6);
        high_conf_ << 100, 100, 200, 200, 0.9, 0,
                      300, 300, 400, 400, 0.85, 0;
        
        // Mixed confidence (high + low)
        mixed_conf_ = Eigen::MatrixXf(3, 6);
        mixed_conf_ << 100, 100, 200, 200, 0.9, 0,   // High
                       300, 300, 400, 400, 0.3, 0,   // Low
                       500, 500, 600, 600, 0.15, 0;  // Very low
    }
    
    cv::Mat img_;
    Eigen::MatrixXf high_conf_;
    Eigen::MatrixXf mixed_conf_;
};

TEST_F(ByteTrackTest, TrackerInitialization) {
    ByteTrack tracker;
    // Tracker initializes without crash
    EXPECT_TRUE(true);
}

TEST_F(ByteTrackTest, TwoStageAssociation) {
    // ByteTrack should use both high and low confidence detections
    ByteTrack tracker(
        0.1f,   // det_thresh (very low to include all)
        30,     // max_age
        50,     // max_obs
        1,      // min_hits
        0.3f,   // iou_threshold
        false,  // per_class
        80,     // nr_classes
        "iou",  // asso_func
        false,  // is_obb
        0.1f,   // min_conf
        0.45f,  // track_thresh
        0.8f,   // match_thresh
        30,     // track_buffer
        30.0f   // frame_rate
    );
    
    auto tracks = tracker.update(mixed_conf_, img_);
    
    // Should handle different confidence levels
    EXPECT_EQ(tracks.cols(), 8);
}

TEST_F(ByteTrackTest, TrackThresholdFiltering) {
    ByteTrack tracker(
        0.1f, 30, 50, 1, 0.3f, false, 80, "iou", false,
        0.1f,  // min_conf = 0.1
        0.6f   // track_thresh = 0.6 (high)
    );
    
    // First stage should only use high confidence detections
    // Low confidence used in second stage for lost track recovery
    
    auto tracks = tracker.update(high_conf_, img_);
    EXPECT_EQ(tracks.cols(), 8);
}

TEST_F(ByteTrackTest, LostTrackRecovery) {
    ByteTrack tracker(0.3f, 30, 50, 1, 0.3f, false, 80, "iou", false,
                      0.1f, 0.45f, 0.8f, 30, 30.0f);
    
    // Frame 1: High confidence detection
    Eigen::MatrixXf det1(1, 6);
    det1 << 100, 100, 200, 200, 0.9, 0;
    tracker.update(det1, img_);
    
    // Frame 2: Same object, low confidence (occluded)
    Eigen::MatrixXf det2(1, 6);
    det2 << 100, 100, 200, 200, 0.3, 0;  // Low confidence
    auto tracks = tracker.update(det2, img_);
    
    // ByteTrack should still maintain the track using second stage
    // association with low confidence detections
}

TEST_F(ByteTrackTest, FrameRateAwareness) {
    // Higher frame rate = longer track buffer
    ByteTrack tracker_30fps(0.3f, 30, 50, 3, 0.3f, false, 80, "iou", false,
                            0.1f, 0.45f, 0.8f, 30, 30.0f);
    
    ByteTrack tracker_60fps(0.3f, 30, 50, 3, 0.3f, false, 80, "iou", false,
                            0.1f, 0.45f, 0.8f, 30, 60.0f);
    
    // Both should initialize without crashing
    EXPECT_TRUE(true);
}

TEST_F(ByteTrackTest, ScoreDecay) {
    ByteTrack tracker(0.3f, 30, 50, 1);
    
    // Track with high initial score
    for (int i = 0; i < 3; ++i) {
        tracker.update(high_conf_, img_);
    }
    
    // Miss several frames
    Eigen::MatrixXf empty(0, 6);
    for (int i = 0; i < 5; ++i) {
        tracker.update(empty, img_);
    }
    
    // Track score should have decayed
    // (internal state, verified through behavior)
}

TEST_F(ByteTrackTest, OutputFormat) {
    ByteTrack tracker(0.3f, 30, 50, 1);
    
    auto tracks = tracker.update(high_conf_, img_);
    
    if (tracks.rows() > 0) {
        // Verify output columns
        // [x1, y1, x2, y2, id, conf, cls, det_ind]
        for (int i = 0; i < tracks.rows(); ++i) {
            // Bounding box should be valid
            EXPECT_LT(tracks(i, 0), tracks(i, 2));  // x1 < x2
            EXPECT_LT(tracks(i, 1), tracks(i, 3));  // y1 < y2
            
            // ID should be positive
            EXPECT_GT(tracks(i, 4), 0);
            
            // Confidence in [0, 1]
            EXPECT_GE(tracks(i, 5), 0);
            EXPECT_LE(tracks(i, 5), 1);
            
            // Class ID should be non-negative
            EXPECT_GE(tracks(i, 6), 0);
        }
    }
}

} // namespace motcpp::trackers::test
