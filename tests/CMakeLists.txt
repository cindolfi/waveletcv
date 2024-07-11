
set(VENV_PATH "${CMAKE_BINARY_DIR}/venv")
set(VENV_PYTHON_EXECUTABLE "${VENV_PATH}/bin/python")
# set(VENV_PYTHON_EXECUTABLE "python")
set(VENV_PIP_EXECUTABLE "${VENV_PATH}/bin/pip")
set(WTCV_GENERATE_TEST_DATA_COMMAND ${VENV_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_cases.py)

include(FindPython)
if (Python_FOUND)
    message(STATUS "Creating python virtual environment")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m venv "${VENV_PATH}"
        RESULT_VARIABLE venv_created
    )
    if (venv_created GREATER 0)
        message(FATAL_ERROR "Failed to create virtual environment")
    endif()

    # if (NOT EXISTS VENV_PYTHON_EXECUTABLE)
    #     message(FATAL_ERROR "Python virutal environment executable does not exist: ${VENV_PYTHON_EXECUTABLE}")
    # endif()
    # if (NOT EXISTS "${VENV_PIP_EXECUTABLE}")
    #     message(FATAL_ERROR "Python virutal environment pip does not exist: ${VENV_PIP_EXECUTABLE}")
    # endif()

    message(STATUS "Installing numpy")
    execute_process(
        COMMAND
        ${VENV_PIP_EXECUTABLE} install numpy
        RESULT_VARIABLE numpy_installed
    )
    if (numpy_installed GREATER 0)
        message(FATAL_ERROR "Failed to install numpy")
    endif()

    message(STATUS "Installing PyWavelets")
    execute_process(
        COMMAND
        ${VENV_PIP_EXECUTABLE} install PyWavelets
        RESULT_VARIABLE pywt_installed
    )
    if (pywt_installed GREATER 0)
        message(FATAL_ERROR "Failed to install PyWavelets")
    endif()

    message(STATUS "Installing opencv-python")
    execute_process(
        COMMAND
        ${VENV_PIP_EXECUTABLE} install opencv-python
        RESULT_VARIABLE opencv_python_installed
    )
    if (opencv_python_installed GREATER 0)
        message(FATAL_ERROR "Failed to install opencv-python")
    endif()
else()
    message(FATAL_ERROR "Python not found")
endif()

#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
set(INSTALL_GTEST OFF)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG        v1.14.0
)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(googletest json)

#   ----------------------------------------------------------------------------
#   Build Tests
#   ----------------------------------------------------------------------------
add_executable(
    wtcv-test
    EXCLUDE_FROM_ALL
    common.hpp
    common.cpp
    json.hpp
    json.cpp
    base_dwt2d.hpp
    base_dwt2d.cpp
    wavelet_test.cpp
    filterbank_test.cpp
    dwt2d_test.cpp
    dwt2d_coeffsexpr_test.cpp
    shrink_test.cpp
    array_test.cpp
    compare_test.cpp
    statistics_test.cpp
)
target_link_libraries (
    wtcv-test
    PRIVATE
    wtcv
    gtest_main
    gmock_main
    nlohmann_json::nlohmann_json
)
target_compile_definitions(
    wtcv-test
    PRIVATE
    WAVELET_TEST_DATA_PATH="${CMAKE_CURRENT_BINARY_DIR}/wavelet_test_data.json"
    DWT2D_TEST_DATA_PATH="${CMAKE_CURRENT_BINARY_DIR}/dwt2d_test_data.json"
    SHRINK_THRESHOLDS_TEST_DATA_PATH="${CMAKE_CURRENT_BINARY_DIR}/shrink_thresholds_test_data.json"
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/images/lena.png
    ${CMAKE_CURRENT_BINARY_DIR}/images/lena.png
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/images/lena_gray.png
    ${CMAKE_CURRENT_BINARY_DIR}/images/lena_gray.png
    COPYONLY
)
add_custom_target(
    build-test-data
    ${WTCV_GENERATE_TEST_DATA_COMMAND} --build
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building test data"
)
add_dependencies(wtcv-test build-test-data)

add_custom_target(
    clean-test-data
    ${WTCV_GENERATE_TEST_DATA_COMMAND} --clean
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Cleaning test data"
)

add_custom_target(
    rebuild-test-data
    ${WTCV_GENERATE_TEST_DATA_COMMAND} --clean --build
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Rebuilding test data"
)

#   ----------------------------------------------------------------------------
#   Add Tests
#   ----------------------------------------------------------------------------
add_test(
    NAME wtcv_unittests
    COMMAND wtcv-test
)
