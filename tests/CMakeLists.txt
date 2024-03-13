#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
FetchContent_Declare(
    googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# set(ABSL_PROPAGATE_CXX_STD ON)
# FetchContent_Declare(
#     abseil
#     GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
#     GIT_TAG        20240116.1
# )
# FetchContent_MakeAvailable(abseil)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

#   ----------------------------------------------------------------------------
#   Add Test Exceutable
#   ----------------------------------------------------------------------------
add_executable(
    wavelet_test
    # utils_test.cpp
    # wavelet_test.hpp
    # wavelet_test_cases.hpp
    wavelet_test.cpp
    # dwt2d_test.hpp
    # dwt2d_test_cases.hpp
    dwt2d_test.cpp
    # shrinkage_test.cpp
    common.cpp
    common.hpp
)

#   ----------------------------------------------------------------------------
#   Test Target Settings
#   ----------------------------------------------------------------------------
target_link_libraries (
    wavelet_test
    PRIVATE
    wavelet_library
    gtest_main
    gmock_main
    # absl::strings
    nlohmann_json::nlohmann_json
)



target_compile_features(wavelet_test PRIVATE cxx_std_20)
target_compile_definitions(
    wavelet_test
    PRIVATE
    WAVELET_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/wavelet_test_data.json"
    DWT2D_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/dwt2d_test_data.json"
)
#   ----------------------------------------------------------------------------
#   Add Tests
#   ----------------------------------------------------------------------------
# add_test(NAME testwavelet_test COMMAND wavelet_test)
set(GTEST_OPTIONS, --gtest_break_on_failure)

add_test(
    NAME testwavelet_test
    # COMMAND wavelet_test
    # COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:MyTest> -P ${CMAKE_SOURCE_DIR}/cmake/RunTests.cmake
    COMMAND wavelet_test \${GTEST_OPTIONS}
)



