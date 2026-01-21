// SPDX-License-Identifier: AGPL-3.0
// Copyright (c) 2026 motcpp contributors

#include <gtest/gtest.h>
#include <motcpp/utils/matching.hpp>

namespace motcpp::utils::test {

class LinearAssignmentTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(LinearAssignmentTest, EmptyCostMatrix) {
    Eigen::MatrixXf cost(0, 0);
    auto result = linear_assignment(cost, 0.5f);
    
    EXPECT_TRUE(result.matches.empty());
    EXPECT_TRUE(result.unmatched_a.empty());
    EXPECT_TRUE(result.unmatched_b.empty());
}

TEST_F(LinearAssignmentTest, SingleMatch) {
    Eigen::MatrixXf cost(1, 1);
    cost << 0.1f;
    
    auto result = linear_assignment(cost, 0.5f);
    
    EXPECT_EQ(result.matches.size(), 1);
    EXPECT_EQ(result.matches[0][0], 0);
    EXPECT_EQ(result.matches[0][1], 0);
    EXPECT_TRUE(result.unmatched_a.empty());
    EXPECT_TRUE(result.unmatched_b.empty());
}

TEST_F(LinearAssignmentTest, MatchAboveThreshold) {
    Eigen::MatrixXf cost(1, 1);
    cost << 0.9f;  // Above threshold
    
    auto result = linear_assignment(cost, 0.5f);
    
    EXPECT_TRUE(result.matches.empty());
    EXPECT_EQ(result.unmatched_a.size(), 1);
    EXPECT_EQ(result.unmatched_b.size(), 1);
}

TEST_F(LinearAssignmentTest, MultipleMatches) {
    Eigen::MatrixXf cost(3, 3);
    cost << 0.1f, 0.9f, 0.9f,
            0.9f, 0.1f, 0.9f,
            0.9f, 0.9f, 0.1f;
    
    auto result = linear_assignment(cost, 0.5f);
    
    EXPECT_EQ(result.matches.size(), 3);
    EXPECT_TRUE(result.unmatched_a.empty());
    EXPECT_TRUE(result.unmatched_b.empty());
    
    // Check diagonal matches
    std::set<std::pair<int, int>> matches_set;
    for (const auto& m : result.matches) {
        matches_set.insert({m[0], m[1]});
    }
    EXPECT_TRUE(matches_set.count({0, 0}));
    EXPECT_TRUE(matches_set.count({1, 1}));
    EXPECT_TRUE(matches_set.count({2, 2}));
}

TEST_F(LinearAssignmentTest, MoreTracksThanDetections) {
    Eigen::MatrixXf cost(3, 2);
    cost << 0.1f, 0.9f,
            0.9f, 0.1f,
            0.9f, 0.9f;
    
    auto result = linear_assignment(cost, 0.5f);
    
    EXPECT_EQ(result.matches.size(), 2);
    EXPECT_EQ(result.unmatched_a.size(), 1);
    EXPECT_TRUE(result.unmatched_b.empty());
}

TEST_F(LinearAssignmentTest, MoreDetectionsThanTracks) {
    Eigen::MatrixXf cost(2, 3);
    cost << 0.1f, 0.9f, 0.9f,
            0.9f, 0.1f, 0.9f;
    
    auto result = linear_assignment(cost, 0.5f);
    
    EXPECT_EQ(result.matches.size(), 2);
    EXPECT_TRUE(result.unmatched_a.empty());
    EXPECT_EQ(result.unmatched_b.size(), 1);
}

TEST_F(LinearAssignmentTest, OptimalAssignment) {
    // Test that we get the optimal (minimum cost) assignment
    Eigen::MatrixXf cost(2, 2);
    cost << 0.1f, 0.2f,
            0.3f, 0.1f;
    
    auto result = linear_assignment(cost, 0.5f);
    
    // Optimal: (0,0)=0.1 + (1,1)=0.1 = 0.2
    // Not: (0,1)=0.2 + (1,0)=0.3 = 0.5
    std::set<std::pair<int, int>> matches_set;
    for (const auto& m : result.matches) {
        matches_set.insert({m[0], m[1]});
    }
    EXPECT_TRUE(matches_set.count({0, 0}));
    EXPECT_TRUE(matches_set.count({1, 1}));
}

} // namespace motcpp::utils::test
