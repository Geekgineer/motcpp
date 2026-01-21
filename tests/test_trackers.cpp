// SPDX-License-Identifier: AGPL-3.0
// Copyright (c) 2026 motcpp contributors

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <motcpp/trackers/ocsort.hpp>

namespace motcpp::trackers::test {

class TrackerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create dummy image
        img_ = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Create sample detections: [x1, y1, x2, y2, conf, cls]
        dets_ = Eigen::MatrixXf(3, 6);
        dets_ << 100, 100, 200, 200, 0.9, 0,
                 300, 300, 400, 400, 0.8, 0,
                 500, 100, 600, 200, 0.7, 1;
    }
    
    cv::Mat img_;
    Eigen::MatrixXf dets_;
};

TEST_F(TrackerTest, ByteTrackInitialization) {
    ByteTrack tracker;
    // Tracker initializes without crash
    EXPECT_TRUE(true);
}

TEST_F(TrackerTest, ByteTrackUpdateReturnsValidOutput) {
    ByteTrack tracker;
    
    auto tracks = tracker.update(dets_, img_);
    
    // Should have 8 columns: x1, y1, x2, y2, id, conf, cls, det_ind
    EXPECT_EQ(tracks.cols(), 8);
    
    // First frame with min_hits=3, might not output tracks
    // But structure should be correct
    if (tracks.rows() > 0) {
        // IDs should be positive
        for (int i = 0; i < tracks.rows(); ++i) {
            EXPECT_GT(tracks(i, 4), 0);  // ID > 0
        }
    }
}

TEST_F(TrackerTest, ByteTrackConsistentTracking) {
    ByteTrack tracker(0.3f, 30, 50, 1);  // min_hits=1 for immediate output
    
    // Process same detections multiple times
    auto tracks1 = tracker.update(dets_, img_);
    auto tracks2 = tracker.update(dets_, img_);
    auto tracks3 = tracker.update(dets_, img_);
    
    // After 3 frames, should have consistent IDs
    EXPECT_GT(tracks3.rows(), 0);
    
    // IDs should be stable (same detections = same IDs)
    if (tracks2.rows() > 0 && tracks3.rows() > 0) {
        std::set<int> ids2, ids3;
        for (int i = 0; i < tracks2.rows(); ++i) {
            ids2.insert(static_cast<int>(tracks2(i, 4)));
        }
        for (int i = 0; i < tracks3.rows(); ++i) {
            ids3.insert(static_cast<int>(tracks3(i, 4)));
        }
        
        // At least some IDs should persist
        size_t common = 0;
        for (int id : ids3) {
            if (ids2.count(id)) common++;
        }
        EXPECT_GT(common, 0);
    }
}

TEST_F(TrackerTest, TrackerReset) {
    ByteTrack tracker;
    
    tracker.update(dets_, img_);
    tracker.update(dets_, img_);
    
    tracker.reset();
    
    // After reset, tracks should start fresh
    auto tracks = tracker.update(dets_, img_);
    
    // New IDs should start from beginning (implementation dependent)
    // Just verify it doesn't crash and returns valid output
    EXPECT_EQ(tracks.cols(), 8);
}

TEST_F(TrackerTest, EmptyDetections) {
    ByteTrack tracker;
    
    Eigen::MatrixXf empty_dets(0, 6);
    auto tracks = tracker.update(empty_dets, img_);
    
    EXPECT_EQ(tracks.rows(), 0);
}

TEST_F(TrackerTest, OCSortInitialization) {
    OCSort tracker;
    // Tracker initializes without crash
    EXPECT_TRUE(true);
}

TEST_F(TrackerTest, OCSortUpdateReturnsValidOutput) {
    OCSort tracker;
    
    auto tracks = tracker.update(dets_, img_);
    
    EXPECT_EQ(tracks.cols(), 8);
}

TEST_F(TrackerTest, TrackerWithLowConfidenceDetections) {
    ByteTrack tracker(0.5f);  // Higher threshold
    
    Eigen::MatrixXf low_conf_dets(2, 6);
    low_conf_dets << 100, 100, 200, 200, 0.3, 0,  // Below threshold
                     300, 300, 400, 400, 0.6, 0;  // Above threshold
    
    auto tracks = tracker.update(low_conf_dets, img_);
    
    // Only high confidence detection should potentially be tracked
    // (may still be 0 if min_hits not met)
    EXPECT_LE(tracks.rows(), 1);
}

} // namespace motcpp::trackers::test
