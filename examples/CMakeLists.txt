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
#   Build wtcv-dwt2d
#   ----------------------------------------------------------------------------
add_executable(
    wtcv-dwt2d
    EXCLUDE_FROM_ALL
    dwt2d.cpp
    common.hpp
    common.cpp
)
target_link_libraries (
    wtcv-dwt2d
    PRIVATE
    wtcv
    cxxopts
)
get_cmake_property(DWT2D_PROGRAM_NAME wtcv-dwt2d OUTPUT_NAME)
target_compile_definitions(
    wtcv-dwt2d
    PRIVATE
    PROGRAM_NAME="${DWT2D_PROGRAM_NAME}"
)

install(
    TARGETS
    wtcv-dwt2d
    COMPONENT wtcv-dwt2d
    EXCLUDE_FROM_ALL
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#   ----------------------------------------------------------------------------
#   Build wtcv-denoise
#   ----------------------------------------------------------------------------
add_executable(
    wtcv-denoise
    EXCLUDE_FROM_ALL
    denoise.cpp
    common.hpp
    common.cpp
)
target_link_libraries (
    wtcv-denoise
    PRIVATE
    wtcv
    cxxopts
)
get_cmake_property(DENOISE_PROGRAM_NAME wtcv-denoise OUTPUT_NAME)
target_compile_definitions(
    wtcv-denoise
    PRIVATE
    PROGRAM_NAME="${DENOISE_PROGRAM_NAME}"
)

install(
    TARGETS
    wtcv-denoise
    COMPONENT wtcv-denoise
    EXCLUDE_FROM_ALL
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#   ----------------------------------------------------------------------------
#   Group Examples Into Single Target
#   ----------------------------------------------------------------------------
if(NOT TARGET examples)
  add_custom_target(examples)
  add_dependencies(examples wtcv-dwt2d wtcv-denoise)
endif()

install(
    TARGETS
    wtcv-denoise
    wtcv-dwt2d
    COMPONENT examples
    EXCLUDE_FROM_ALL
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)


