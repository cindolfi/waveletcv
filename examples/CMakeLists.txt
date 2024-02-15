#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        eb787304d67ec22f7c3a184ee8b4c481d04357fd # release-3.1.1
)
FetchContent_MakeAvailable(cxxopts)

#   ----------------------------------------------------------------------------
#   Add Test Exceutable
#   ----------------------------------------------------------------------------
# Tests need to be added as executables first
add_executable(denoise noise.cpp)

#   ----------------------------------------------------------------------------
#   Executable Target Settings
#   ----------------------------------------------------------------------------
target_link_libraries (
    denoise
    PRIVATE
    wavelet_library
    cxxopts
)

target_compile_features(denoise PRIVATE cxx_std_20)
