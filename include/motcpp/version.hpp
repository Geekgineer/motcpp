// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#define MOTCPP_VERSION_MAJOR 1
#define MOTCPP_VERSION_MINOR 0
#define MOTCPP_VERSION_PATCH 0
#define MOTCPP_VERSION_STRING "1.0.0"

// Feature flags
#define MOTCPP_HAS_ONNX 1
#define MOTCPP_HAS_OPENCV 1

namespace motcpp {

struct Version {
    static constexpr int major = MOTCPP_VERSION_MAJOR;
    static constexpr int minor = MOTCPP_VERSION_MINOR;
    static constexpr int patch = MOTCPP_VERSION_PATCH;
    static constexpr const char* string = MOTCPP_VERSION_STRING;
};

} // namespace motcpp
