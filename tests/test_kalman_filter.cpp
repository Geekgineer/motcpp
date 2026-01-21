// SPDX-License-Identifier: AGPL-3.0
// Copyright (c) 2026 motcpp contributors

#include <gtest/gtest.h>
#include <motcpp/motion/kalman_filters/xysr_kf.hpp>

namespace motcpp::motion::test {

class KalmanFilterXYSRTest : public ::testing::Test {
protected:
    void SetUp() override {
        kf_ = std::make_unique<KalmanFilterXYSR>(7, 4, 50);
    }
    
    std::unique_ptr<KalmanFilterXYSR> kf_;
};

TEST_F(KalmanFilterXYSRTest, InitializesWithCorrectDimensions) {
    EXPECT_EQ(kf_->x.size(), 7);
    EXPECT_EQ(kf_->F.rows(), 7);
    EXPECT_EQ(kf_->F.cols(), 7);
    EXPECT_EQ(kf_->H.rows(), 4);
    EXPECT_EQ(kf_->H.cols(), 7);
}

TEST_F(KalmanFilterXYSRTest, StateTransitionMatrixIsCorrect) {
    // F should have 1s on diagonal and velocity terms
    EXPECT_FLOAT_EQ(kf_->F(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(kf_->F(0, 4), 1.0f);  // x += vx
    EXPECT_FLOAT_EQ(kf_->F(1, 5), 1.0f);  // y += vy
    EXPECT_FLOAT_EQ(kf_->F(2, 6), 1.0f);  // s += vs
}

TEST_F(KalmanFilterXYSRTest, PredictUpdatesState) {
    // Set initial state
    kf_->x << 100, 100, 1000, 0.5, 10, 10, 0;
    
    Eigen::VectorXf initial_state = kf_->x;
    kf_->predict();
    
    // Position should change by velocity
    EXPECT_FLOAT_EQ(kf_->x(0), initial_state(0) + initial_state(4));
    EXPECT_FLOAT_EQ(kf_->x(1), initial_state(1) + initial_state(5));
}

TEST_F(KalmanFilterXYSRTest, UpdateCorrectsMeasurement) {
    kf_->x << 100, 100, 1000, 0.5, 0, 0, 0;
    
    Eigen::VectorXf z(4);
    z << 110, 110, 1100, 0.5;  // Measurement slightly different
    
    kf_->update(z);
    
    // State should move towards measurement
    EXPECT_GT(kf_->x(0), 100);
    EXPECT_LT(kf_->x(0), 110);
}

TEST_F(KalmanFilterXYSRTest, HistoryStoresObservations) {
    EXPECT_TRUE(kf_->history_obs.empty());
    
    Eigen::VectorXf z(4);
    z << 100, 100, 1000, 0.5;
    
    kf_->update(z);
    EXPECT_EQ(kf_->history_obs.size(), 1);
    
    kf_->update(z);
    EXPECT_EQ(kf_->history_obs.size(), 2);
}

TEST_F(KalmanFilterXYSRTest, AffineCorrection) {
    kf_->x << 100, 100, 1000, 0.5, 10, 10, 0;
    
    Eigen::Matrix2f m;
    m << 1, 0, 0, 1;  // Identity rotation
    Eigen::Vector2f t(50, 50);  // Translation
    
    kf_->apply_affine_correction(m, t);
    
    EXPECT_FLOAT_EQ(kf_->x(0), 150);  // x + tx
    EXPECT_FLOAT_EQ(kf_->x(1), 150);  // y + ty
}

} // namespace motcpp::motion::test
