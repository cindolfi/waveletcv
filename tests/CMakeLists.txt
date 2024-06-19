#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
FetchContent_Declare(
    googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

#   ----------------------------------------------------------------------------
#   Build Tests
#   ----------------------------------------------------------------------------
add_executable(
    cvwt_test
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
    shrinkage_test.cpp
    array_test.cpp
    compare_test.cpp
    statistics_test.cpp
)

target_link_libraries (
    cvwt_test
    PRIVATE
    cvwt
    gtest_main
    gmock_main
    nlohmann_json::nlohmann_json
)
target_compile_features(cvwt_test PRIVATE cxx_std_20)
target_compile_definitions(
    cvwt_test
    PRIVATE
    WAVELET_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/wavelet_test_data.json"
    DWT2D_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/dwt2d_test_data.json"
    SHRINKAGE_THRESHOLD_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/shrinkage_thresholds_test_data.json"
)

#   ----------------------------------------------------------------------------
#   Add Tests
#   ----------------------------------------------------------------------------
add_test(
    NAME testcvwt_test COMMAND cvwt_test
)

