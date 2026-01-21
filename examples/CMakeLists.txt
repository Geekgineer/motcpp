# SPDX-License-Identifier: AGPL-3.0
# Copyright (c) 2026 motcpp contributors

# Simple tracking example
add_executable(simple_tracking simple_tracking.cpp)
target_link_libraries(simple_tracking PRIVATE motcpp)

# Add ONNX Runtime include for ReID support
if(MOTCPP_ENABLE_ONNX AND ONNXRUNTIME_FOUND)
    target_include_directories(simple_tracking PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
endif()
