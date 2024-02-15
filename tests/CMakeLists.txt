#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
# Fetch GoogleTest testing framework
FetchContent_Declare(
    googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG        main
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    json 
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

#   ----------------------------------------------------------------------------
#   Add Test Exceutable
#   ----------------------------------------------------------------------------
# Tests need to be added as executables first
add_executable(wavelet_test wavelet_test.cpp)

#   ----------------------------------------------------------------------------
#   Test Target Settings
#   ----------------------------------------------------------------------------
# Should be linked to the main library, as well as the GoogleTest testing library
target_link_libraries (
    wavelet_test 
    PRIVATE 
    wavelet_library 
    gtest_main 
    nlohmann_json::nlohmann_json
)

# I'm using C++17 in the test
target_compile_features(wavelet_test PRIVATE cxx_std_17)

#   ----------------------------------------------------------------------------
#   Add Tests
#   ----------------------------------------------------------------------------
add_test(NAME testwavelet_test COMMAND wavelet_test)


