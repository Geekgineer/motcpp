// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors
// https://github.com/Geekgineer/motcpp

#pragma once

/**
 * @file motcpp.hpp
 * @brief Main header file for motcpp - Modern C++ Multi-Object Tracking Library
 * 
 * motcpp is a high-performance, header-friendly C++ library for multi-object tracking.
 * It provides state-of-the-art tracking algorithms with a clean, modern C++ API.
 * 
 * @example
 * @code
 * #include <Geekgineer/motcpp.hpp>
 * 
 * // Create a tracker
 * motcpp::trackers::ByteTrack tracker;
 * 
 * // Process detections
 * auto tracks = tracker.update(detections, frame);
 * @endcode
 */

#include <motcpp/version.hpp>
#include <motcpp/tracker.hpp>
#include <motcpp/trackers/sort.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <motcpp/trackers/ocsort.hpp>
#include <motcpp/trackers/deepocsort.hpp>
#include <motcpp/trackers/strongsort.hpp>
#include <motcpp/trackers/botsort.hpp>
#include <motcpp/trackers/boosttrack.hpp>
#include <motcpp/trackers/hybridsort.hpp>
#include <motcpp/trackers/ucmc.hpp>

namespace motcpp {

/**
 * @brief Library version information
 */
constexpr const char* version() noexcept {
    return MOTCPP_VERSION_STRING;
}

} // namespace motcpp
