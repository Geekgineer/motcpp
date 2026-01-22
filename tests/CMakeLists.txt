# SPDX-License-Identifier: AGPL-3.0
# Copyright (c) 2026 motcpp contributors

include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Test sources
set(TEST_SOURCES
    test_main.cpp
    test_kalman_filter.cpp
    test_iou.cpp
    test_matching.cpp
    test_trackers.cpp
    test_sort.cpp
    test_bytetrack.cpp
)

# Create test executable
add_executable(motcpp_tests ${TEST_SOURCES})

target_link_libraries(motcpp_tests
    PRIVATE
        motcpp
        GTest::gtest
        GTest::gmock
)

target_include_directories(motcpp_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Discover tests
gtest_discover_tests(motcpp_tests)

# Coverage report target
if(MOTCPP_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)
    
    if(LCOV AND GENHTML)
        add_custom_target(coverage
            COMMAND ${LCOV} --directory . --zerocounters
            COMMAND $<TARGET_FILE:motcpp_tests>
            COMMAND ${LCOV} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/tests/*' '*/googletest/*' --output-file coverage.info --ignore-errors unused
            COMMAND ${GENHTML} coverage.info --output-directory coverage_report
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating coverage report"
        )
    endif()
endif()
