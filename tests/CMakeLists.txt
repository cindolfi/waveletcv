enable_testing()

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
    cvwt-test
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
    cvwt-test
    PRIVATE
    cvwt
    gtest_main
    gmock_main
    nlohmann_json::nlohmann_json
)
target_compile_definitions(
    cvwt-test
    PRIVATE
    WAVELET_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/wavelet_test_data.json"
    DWT2D_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/dwt2d_test_data.json"
    SHRINK_THRESHOLDS_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/shrink_thresholds_test_data.json"
)

#   ----------------------------------------------------------------------------
#   Add Tests
#   ----------------------------------------------------------------------------
add_test(
    NAME testcvwt
    COMMAND cvwt-test
)
