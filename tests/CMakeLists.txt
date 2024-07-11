include(python_venv)

if (NOT VENV_CREATED)
    message(FATAL_ERROR "Failed to create python virtual environment - cannot generate test data")
endif()

pip_install_requirements("${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt")
set(WTCV_GENERATE_TEST_DATA_COMMAND ${VENV_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_cases.py)

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
