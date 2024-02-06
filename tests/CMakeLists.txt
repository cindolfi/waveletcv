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
    # URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
    # URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip

#   ----------------------------------------------------------------------------
#   Add Test Exceutable
#   ----------------------------------------------------------------------------
# Tests need to be added as executables first
add_executable(wavelet_test wavelet_test.cpp)

#   ----------------------------------------------------------------------------
#   Test Target Settings
#   ----------------------------------------------------------------------------
# Should be linked to the main library, as well as the GoogleTest testing library
target_link_libraries (wavelet_test PRIVATE wavelet_library gtest_main)

# I'm using C++17 in the test
target_compile_features(wavelet_test PRIVATE cxx_std_17)

#   ----------------------------------------------------------------------------
#   Add Tests
#   ----------------------------------------------------------------------------
add_test(NAME testwavelet_test COMMAND wavelet_test)

