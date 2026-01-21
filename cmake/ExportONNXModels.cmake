# CMake module to export ReID models to ONNX format
# (Currently not used - export is handled by Python scripts)

# Find Python3 (optional)
find_package(Python3 COMPONENTS Interpreter QUIET)

# Function to export a ReID model to ONNX
# (Placeholder - actual export is done via Python scripts)
function(export_reid_onnx MODEL_NAME OUTPUT_DIR)
    set(EXPORT_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/export_reid_onnx.py")
    set(VENV_DIR "${CMAKE_SOURCE_DIR}/scripts/.venv")
    set(PYTHON_EXE "${Python3_EXECUTABLE}")
    
    # Check if venv exists, if not set it up
    if(NOT EXISTS "${VENV_DIR}/bin/activate")
        message(STATUS "Setting up Python virtual environment for ONNX export...")
        execute_process(
            COMMAND ${PYTHON_EXE} -m venv "${VENV_DIR}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts"
            RESULT_VARIABLE VENV_RESULT
        )
        if(VENV_RESULT)
            message(WARNING "Failed to create venv. Trying to use system Python...")
            set(VENV_PYTHON "${PYTHON_EXE}")
        else
            if(UNIX)
                set(VENV_PYTHON "${VENV_DIR}/bin/python")
            else()
                set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
            endif()
            
            # Install dependencies
            execute_process(
                COMMAND ${VENV_PYTHON} -m pip install --upgrade pip -q
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts"
            )
            execute_process(
                COMMAND ${VENV_PYTHON} -m pip install -q torch onnx onnxruntime numpy pandas loguru filelock gdown
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts"
            )
        endif()
    else()
        if(UNIX)
            set(VENV_PYTHON "${VENV_DIR}/bin/python")
        else()
            set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
        endif()
    endif()
    
    # Export model
    set(OUTPUT_FILE "${OUTPUT_DIR}/${MODEL_NAME}.onnx")
    
    if(EXISTS "${OUTPUT_FILE}")
        message(STATUS "ONNX model already exists: ${OUTPUT_FILE}")
        set(EXPORTED_MODEL "${OUTPUT_FILE}" PARENT_SCOPE)
        return()
    endif()
    
    message(STATUS "Exporting ${MODEL_NAME} to ONNX format...")
    execute_process(
        COMMAND ${VENV_PYTHON} ${EXPORT_SCRIPT} --model ${MODEL_NAME} --output-dir ${OUTPUT_DIR}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts"
        RESULT_VARIABLE EXPORT_RESULT
        OUTPUT_VARIABLE EXPORT_OUTPUT
        ERROR_VARIABLE EXPORT_ERROR
    )
    
    if(EXPORT_RESULT EQUAL 0 AND EXISTS "${OUTPUT_FILE}")
        message(STATUS "✅ Successfully exported: ${OUTPUT_FILE}")
        set(EXPORTED_MODEL "${OUTPUT_FILE}" PARENT_SCOPE)
    else()
        message(WARNING "Failed to export model: ${EXPORT_ERROR}")
        set(EXPORTED_MODEL "" PARENT_SCOPE)
    endif()
endfunction()

# Custom target to export default ReID model
add_custom_target(export_reid_models
    COMMAND ${CMAKE_COMMAND} -DEXPORT_MODEL=osnet_x1_0_dukemtmcreid
            -DOUTPUT_DIR=${CMAKE_SOURCE_DIR}/scripts/models
            -P ${CMAKE_CURRENT_LIST_DIR}/ExportONNXModels.cmake
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
    COMMENT "Exporting ReID models to ONNX format"
)

