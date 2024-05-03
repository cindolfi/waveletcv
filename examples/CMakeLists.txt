#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.0
)
FetchContent_MakeAvailable(cxxopts)

#   ----------------------------------------------------------------------------
#   Build dwt2d
#   ----------------------------------------------------------------------------
add_executable(
    dwt2d
    dwt2d.cpp
    common.hpp
    common.cpp
)
target_link_libraries (
    dwt2d
    PRIVATE
    cvwt
    cxxopts
)
target_compile_features(dwt2d PRIVATE cxx_std_20)

#   ----------------------------------------------------------------------------
#   Build denoise
#   ----------------------------------------------------------------------------
add_executable(
    dwt2d-denoise
    denoise.cpp
    common.hpp
    common.cpp
)
target_link_libraries (
    dwt2d-denoise
    PRIVATE
    cvwt
    cxxopts
)
target_compile_features(dwt2d-denoise PRIVATE cxx_std_20)

