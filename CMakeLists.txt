# SPDX-License-Identifier: AGPL-3.0
# Copyright (c) 2026 motcpp contributors

cmake_minimum_required(VERSION 3.20)

# Extract version from version.hpp
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/motcpp/version.hpp" VERSION_FILE)
string(REGEX MATCH "MOTCPP_VERSION_MAJOR ([0-9]+)" _ ${VERSION_FILE})
set(VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "MOTCPP_VERSION_MINOR ([0-9]+)" _ ${VERSION_FILE})
set(VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "MOTCPP_VERSION_PATCH ([0-9]+)" _ ${VERSION_FILE})
set(VERSION_PATCH ${CMAKE_MATCH_1})

project(motcpp
    VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
    DESCRIPTION "Modern C++ Multi-Object Tracking Library"
    HOMEPAGE_URL "https://github.com/Geekgineer/motcpp"
    LANGUAGES CXX
)

# ============================================================================
# Build Options
# ============================================================================
option(MOTCPP_BUILD_TESTS "Build unit tests" ON)
option(MOTCPP_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(MOTCPP_BUILD_EXAMPLES "Build examples" ON)
option(MOTCPP_BUILD_TOOLS "Build command-line tools" ON)
option(MOTCPP_ENABLE_ONNX "Enable ONNX Runtime for ReID" ON)
option(MOTCPP_INSTALL "Generate install target" ON)
option(MOTCPP_COVERAGE "Enable code coverage" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  motcpp v${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
message(STATUS "")

# ============================================================================
# Dependencies
# ============================================================================
find_package(OpenCV REQUIRED)

# Eigen3 - accept version 3.3+ or 5.x (Eigen changed versioning scheme)
find_package(Eigen3 CONFIG REQUIRED)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 REQUIRED NO_MODULE)
endif()

# yaml-cpp - try new target first, fallback to old
find_package(yaml-cpp REQUIRED)
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp target not found")
endif()

# Optional: ONNX Runtime for ReID models
if(MOTCPP_ENABLE_ONNX)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DownloadONNXRuntime.cmake)
endif()

# Optional: spdlog for logging
find_package(spdlog QUIET)
if(spdlog_FOUND)
    message(STATUS "Found spdlog: logging enabled")
else()
    message(STATUS "spdlog not found: logging disabled")
endif()

# ============================================================================
# Source Files
# ============================================================================
set(MOTCPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MOTCPP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(MOTCPP_SOURCES
    # Core
    ${MOTCPP_SOURCE_DIR}/config.cpp
    ${MOTCPP_SOURCE_DIR}/tracker.cpp
    
    # Utils
    ${MOTCPP_SOURCE_DIR}/utils/ops.cpp
    ${MOTCPP_SOURCE_DIR}/utils/iou.cpp
    ${MOTCPP_SOURCE_DIR}/utils/association.cpp
    ${MOTCPP_SOURCE_DIR}/utils/matching.cpp
    
    # Motion models
    ${MOTCPP_SOURCE_DIR}/motion/kalman_filter.cpp
    ${MOTCPP_SOURCE_DIR}/motion/kalman_filters/xyah_kf.cpp
    ${MOTCPP_SOURCE_DIR}/motion/kalman_filters/xysr_kf.cpp
    ${MOTCPP_SOURCE_DIR}/motion/cmc/cmc.cpp
    ${MOTCPP_SOURCE_DIR}/motion/cmc/sof.cpp
    ${MOTCPP_SOURCE_DIR}/motion/cmc/ecc.cpp
    
    # Appearance (ReID)
    ${MOTCPP_SOURCE_DIR}/appearance/reid_backend.cpp
    ${MOTCPP_SOURCE_DIR}/appearance/onnx_backend.cpp
    
    # Trackers
    ${MOTCPP_SOURCE_DIR}/trackers/sort.cpp
    ${MOTCPP_SOURCE_DIR}/trackers/bytetrack.cpp
    ${MOTCPP_SOURCE_DIR}/trackers/ocsort.cpp
    ${MOTCPP_SOURCE_DIR}/trackers/deepocsort.cpp
    ${MOTCPP_SOURCE_DIR}/trackers/strongsort.cpp
    ${MOTCPP_SOURCE_DIR}/trackers/botsort.cpp
    ${MOTCPP_SOURCE_DIR}/trackers/boosttrack.cpp
    ${MOTCPP_SOURCE_DIR}/trackers/hybridsort.cpp
    ${MOTCPP_SOURCE_DIR}/trackers/ucmc.cpp
    
    # Data
    ${MOTCPP_SOURCE_DIR}/data/mot17_dataset.cpp
)

set(MOTCPP_HEADERS
    ${MOTCPP_INCLUDE_DIR}/motcpp/motcpp.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/version.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/tracker.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/config.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/utils/ops.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/utils/iou.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/utils/association.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/utils/matching.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/utils/mot_format.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/motion/kalman_filter.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/motion/kalman_filters/xyah_kf.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/motion/kalman_filters/xysr_kf.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/motion/kalman_filters/xywh_kf.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/motion/cmc/cmc.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/motion/cmc/sof.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/motion/cmc/ecc.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/appearance/reid_backend.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/appearance/onnx_backend.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/sort.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/bytetrack.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/ocsort.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/deepocsort.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/strongsort.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/botsort.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/boosttrack.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/hybridsort.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/trackers/ucmc.hpp
    ${MOTCPP_INCLUDE_DIR}/motcpp/data/mot17_dataset.hpp
)

# ============================================================================
# Library Target
# ============================================================================
add_library(motcpp ${MOTCPP_SOURCES} ${MOTCPP_HEADERS})
add_library(motcpp::motcpp ALIAS motcpp)

target_include_directories(motcpp
    PUBLIC
        $<BUILD_INTERFACE:${MOTCPP_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
# OpenCV must be PUBLIC as it's used in public headers
# Eigen3 must be PUBLIC as it's used in public headers
# Use OpenCV target if available (handles paths correctly), otherwise use legacy variables
if(TARGET opencv_core)
    target_link_libraries(motcpp
        PUBLIC
            opencv_core
            opencv_imgproc
            opencv_imgcodecs
            Eigen3::Eigen
            ${YAML_CPP_TARGET}
    )
else()
    target_link_libraries(motcpp
        PUBLIC
            ${OpenCV_LIBS}
            Eigen3::Eigen
            ${YAML_CPP_TARGET}
    )
endif()

# ONNX Runtime (optional)
if(MOTCPP_ENABLE_ONNX AND ONNXRUNTIME_FOUND)
    target_compile_definitions(motcpp PUBLIC MOTCPP_HAS_ONNX=1)
    target_link_libraries(motcpp PRIVATE ${ONNXRUNTIME_LIBRARY})
    target_include_directories(motcpp PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    
    # RPATH for runtime linking
    if(NOT WIN32)
        get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
        set_target_properties(motcpp PROPERTIES
            INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()

# spdlog (optional)
if(spdlog_FOUND)
    target_link_libraries(motcpp PUBLIC spdlog::spdlog)
    target_compile_definitions(motcpp PUBLIC MOTCPP_HAS_SPDLOG=1)
endif()

# Compiler options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(motcpp PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O2>
        $<$<CONFIG:Debug>:-g -O0>
    )
    
    if(MOTCPP_COVERAGE)
        target_compile_options(motcpp PRIVATE --coverage)
        target_link_options(motcpp PRIVATE --coverage)
    endif()
elseif(MSVC)
    target_compile_options(motcpp PRIVATE /W4 /permissive-)
endif()

# ============================================================================
# Tools
# ============================================================================
if(MOTCPP_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ============================================================================
# Examples
# ============================================================================
if(MOTCPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Testing
# ============================================================================
if(MOTCPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Installation
# ============================================================================
if(MOTCPP_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    
    # Install library
    install(TARGETS motcpp
        EXPORT motcppTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    
    # Install headers
    install(DIRECTORY ${MOTCPP_INCLUDE_DIR}/motcpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    # Export targets
    install(EXPORT motcppTargets
        FILE motcppTargets.cmake
        NAMESPACE motcpp::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/motcpp
    )
    
    # Package config
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/motcppConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/motcppConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/motcpp
    )
    
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/motcppConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/motcppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/motcppConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/motcpp
    )
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build tests:      ${MOTCPP_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${MOTCPP_BUILD_BENCHMARKS}")
message(STATUS "  Build examples:   ${MOTCPP_BUILD_EXAMPLES}")
message(STATUS "  Build tools:      ${MOTCPP_BUILD_TOOLS}")
message(STATUS "  ONNX Runtime:     ${MOTCPP_ENABLE_ONNX}")
message(STATUS "  OpenCV:           ${OpenCV_VERSION}")
message(STATUS "  Eigen3:           ${Eigen3_VERSION}")
message(STATUS "")
