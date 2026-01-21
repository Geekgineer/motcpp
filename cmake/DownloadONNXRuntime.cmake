# DownloadONNXRuntime.cmake
# Automatically downloads ONNX Runtime pre-built binaries for the current platform

include(FetchContent)

# Default version
if(NOT DEFINED ONNXRUNTIME_VERSION)
    set(ONNXRUNTIME_VERSION "1.20.1" CACHE STRING "ONNX Runtime version to download")
endif()

# Default GPU support
if(NOT DEFINED ONNXRUNTIME_GPU)
    set(ONNXRUNTIME_GPU OFF CACHE BOOL "Enable GPU support for ONNX Runtime")
endif()

# Detect platform
if(APPLE)
    set(ONNXRUNTIME_PLATFORM "osx")
    set(ONNXRUNTIME_ARCHIVE_EXTENSION "tgz")
elseif(UNIX)
    set(ONNXRUNTIME_PLATFORM "linux")
    set(ONNXRUNTIME_ARCHIVE_EXTENSION "tgz")
elseif(WIN32)
    set(ONNXRUNTIME_PLATFORM "win")
    set(ONNXRUNTIME_ARCHIVE_EXTENSION "zip")
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime download")
endif()

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(ONNXRUNTIME_ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(ONNXRUNTIME_ARCH "x64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(ONNXRUNTIME_ARCH "arm")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i[3-6]86|X86")
    set(ONNXRUNTIME_ARCH "x86")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Construct filename and directory
set(ONNXRUNTIME_FILE "onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}")
set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/third_party/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}")

if(ONNXRUNTIME_GPU)
    set(ONNXRUNTIME_FILE "${ONNXRUNTIME_FILE}-gpu")
    set(ONNXRUNTIME_DIR "${ONNXRUNTIME_DIR}-gpu")
endif()

set(ONNXRUNTIME_FILE "${ONNXRUNTIME_FILE}-${ONNXRUNTIME_VERSION}.${ONNXRUNTIME_ARCHIVE_EXTENSION}")
set(ONNXRUNTIME_DIR "${ONNXRUNTIME_DIR}-${ONNXRUNTIME_VERSION}")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}")

# Set download directory
set(ONNXRUNTIME_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads")
set(ONNXRUNTIME_ARCHIVE_PATH "${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_FILE}")

# Check if already extracted
if(NOT EXISTS "${ONNXRUNTIME_DIR}")
    message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION} from ${ONNXRUNTIME_URL}")
    
    # Create download directory
    file(MAKE_DIRECTORY "${ONNXRUNTIME_DOWNLOAD_DIR}")
    
    # Download archive
    if(NOT EXISTS "${ONNXRUNTIME_ARCHIVE_PATH}")
        file(DOWNLOAD
            "${ONNXRUNTIME_URL}"
            "${ONNXRUNTIME_ARCHIVE_PATH}"
            SHOW_PROGRESS
            STATUS download_status
        )
        
        list(GET download_status 0 status_code)
        if(NOT status_code EQUAL 0)
            list(GET download_status 1 error_msg)
            message(FATAL_ERROR "Failed to download ONNX Runtime: ${error_msg}")
        endif()
    else()
        message(STATUS "ONNX Runtime archive already exists, skipping download")
    endif()
    
    # Extract archive
    message(STATUS "Extracting ONNX Runtime to ${ONNXRUNTIME_DIR}")
    file(MAKE_DIRECTORY "${ONNXRUNTIME_DIR}")
    
    if(ONNXRUNTIME_ARCHIVE_EXTENSION STREQUAL "tgz")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${ONNXRUNTIME_ARCHIVE_PATH}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/third_party"
            RESULT_VARIABLE extract_result
        )
    elseif(ONNXRUNTIME_ARCHIVE_EXTENSION STREQUAL "zip")
        if(WIN32)
            # Use PowerShell on Windows
            execute_process(
                COMMAND powershell -Command "Expand-Archive -Path '${ONNXRUNTIME_ARCHIVE_PATH}' -DestinationPath '${CMAKE_BINARY_DIR}/third_party' -Force"
                RESULT_VARIABLE extract_result
            )
        else()
            # Use unzip on Unix-like systems
            find_program(UNZIP_EXECUTABLE unzip)
            if(NOT UNZIP_EXECUTABLE)
                message(FATAL_ERROR "unzip not found. Please install unzip to extract ONNX Runtime.")
            endif()
            execute_process(
                COMMAND ${UNZIP_EXECUTABLE} -q "${ONNXRUNTIME_ARCHIVE_PATH}" -d "${CMAKE_BINARY_DIR}/third_party"
                RESULT_VARIABLE extract_result
            )
        endif()
    endif()
    
    if(NOT extract_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract ONNX Runtime archive")
    endif()
    
    # Verify extraction
    if(NOT EXISTS "${ONNXRUNTIME_DIR}")
        message(FATAL_ERROR "ONNX Runtime extraction failed. Expected directory: ${ONNXRUNTIME_DIR}")
    endif()
    
    message(STATUS "ONNX Runtime extracted successfully to ${ONNXRUNTIME_DIR}")
else()
    message(STATUS "ONNX Runtime already exists at ${ONNXRUNTIME_DIR}, skipping download")
endif()

# Find headers and libraries
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Verify paths exist
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found at ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

# Find library file
if(WIN32)
    set(ONNXRUNTIME_LIB_NAME "onnxruntime")
    set(ONNXRUNTIME_LIB_PREFIX "")
    set(ONNXRUNTIME_LIB_EXT ".lib")
    # Also check for DLL
    file(GLOB ONNXRUNTIME_DLL_FILES "${ONNXRUNTIME_LIB_DIR}/*${ONNXRUNTIME_LIB_NAME}*.dll")
else()
    set(ONNXRUNTIME_LIB_NAME "onnxruntime")
    set(ONNXRUNTIME_LIB_PREFIX "lib")
    if(APPLE)
        set(ONNXRUNTIME_LIB_EXT ".dylib")
    else()
        set(ONNXRUNTIME_LIB_EXT ".so")
    endif()
endif()

# Find the actual library file (prefer shared library)
# Try with prefix first (libonnxruntime.so on Linux)
file(GLOB ONNXRUNTIME_LIB_FILES "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_PREFIX}${ONNXRUNTIME_LIB_NAME}${ONNXRUNTIME_LIB_EXT}*")
if(NOT ONNXRUNTIME_LIB_FILES)
    # Try without prefix
    file(GLOB ONNXRUNTIME_LIB_FILES "${ONNXRUNTIME_LIB_DIR}/*${ONNXRUNTIME_LIB_NAME}*${ONNXRUNTIME_LIB_EXT}*")
endif()

if(ONNXRUNTIME_LIB_FILES)
    # Prefer the actual library file (not symlink) if possible
    set(ONNXRUNTIME_LIBRARY "")
    foreach(lib_file ${ONNXRUNTIME_LIB_FILES})
        get_filename_component(lib_name "${lib_file}" NAME)
        # Prefer the main .so file (not versioned symlinks like .so.1.20.1)
        if(lib_name MATCHES "^${ONNXRUNTIME_LIB_PREFIX}${ONNXRUNTIME_LIB_NAME}${ONNXRUNTIME_LIB_EXT}$" AND NOT ONNXRUNTIME_LIBRARY)
            set(ONNXRUNTIME_LIBRARY "${lib_file}")
        endif()
    endforeach()
    # If no exact match found, use the first one (usually the main library)
    if(NOT ONNXRUNTIME_LIBRARY)
        list(GET ONNXRUNTIME_LIB_FILES 0 ONNXRUNTIME_LIBRARY)
    endif()
else()
    # Try without extension (for static libs or different naming)
    file(GLOB ONNXRUNTIME_LIB_FILES "${ONNXRUNTIME_LIB_DIR}/*${ONNXRUNTIME_LIB_NAME}*")
    if(ONNXRUNTIME_LIB_FILES)
        list(GET ONNXRUNTIME_LIB_FILES 0 ONNXRUNTIME_LIBRARY)
    else()
        message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIR}")
        message(FATAL_ERROR "  Searched for: ${ONNXRUNTIME_LIB_PREFIX}${ONNXRUNTIME_LIB_NAME}${ONNXRUNTIME_LIB_EXT}*")
        message(FATAL_ERROR "  Directory contents:")
        file(GLOB dir_contents "${ONNXRUNTIME_LIB_DIR}/*")
        foreach(item ${dir_contents})
            message(FATAL_ERROR "    ${item}")
        endforeach()
    endif()
endif()

message(STATUS "ONNX Runtime found:")
message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "  Library: ${ONNXRUNTIME_LIBRARY}")

# Export variables
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_INCLUDE_DIR}" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LIBRARY}" CACHE FILEPATH "ONNX Runtime library")
set(ONNXRUNTIME_FOUND TRUE CACHE BOOL "ONNX Runtime found")

