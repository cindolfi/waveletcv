#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
FetchContent_Declare(
    googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# FetchContent_Declare(
#     json
#     URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
# )
# FetchContent_MakeAvailable(json)

#   ----------------------------------------------------------------------------
#   Add Test Exceutable
#   ----------------------------------------------------------------------------
add_executable(wavelet_test wavelet_test.cpp)

#   ----------------------------------------------------------------------------
#   Test Target Settings
#   ----------------------------------------------------------------------------
target_link_libraries (
    wavelet_test
    PRIVATE
    wavelet_library
    gtest_main
    gmock_main
    # nlohmann_json::nlohmann_json
)

target_compile_features(wavelet_test PRIVATE cxx_std_20)

#   ----------------------------------------------------------------------------
#   Add Tests
#   ----------------------------------------------------------------------------
add_test(NAME testwavelet_test COMMAND wavelet_test)
