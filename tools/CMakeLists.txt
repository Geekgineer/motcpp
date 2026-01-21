# SPDX-License-Identifier: AGPL-3.0
# Copyright (c) 2026 motcpp contributors

# MOT Evaluation Tool
add_executable(motcpp_eval motcpp_eval.cpp)
target_link_libraries(motcpp_eval PRIVATE motcpp)

# Add ONNX Runtime include for ReID support
if(MOTCPP_ENABLE_ONNX AND ONNXRUNTIME_FOUND)
    target_include_directories(motcpp_eval PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
endif()

# Install
install(TARGETS motcpp_eval RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
