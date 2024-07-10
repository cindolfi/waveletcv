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
#   Build cvwt-dwt2d
#   ----------------------------------------------------------------------------
add_executable(
    cvwt-dwt2d
    EXCLUDE_FROM_ALL
    dwt2d.cpp
    common.hpp
    common.cpp
)
target_link_libraries (
    cvwt-dwt2d
    PRIVATE
    cvwt
    cxxopts
)
get_cmake_property(DWT2D_PROGRAM_NAME cvwt-dwt2d OUTPUT_NAME)
target_compile_definitions(
    cvwt-dwt2d
    PRIVATE
    PROGRAM_NAME="${DWT2D_PROGRAM_NAME}"
)

install(
    TARGETS
    cvwt-dwt2d
    COMPONENT cvwt-dwt2d
    EXCLUDE_FROM_ALL
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#   ----------------------------------------------------------------------------
#   Build cvwt-denoise
#   ----------------------------------------------------------------------------
add_executable(
    cvwt-denoise
    EXCLUDE_FROM_ALL
    denoise.cpp
    common.hpp
    common.cpp
)
target_link_libraries (
    cvwt-denoise
    PRIVATE
    cvwt
    cxxopts
)
get_cmake_property(DENOISE_PROGRAM_NAME cvwt-denoise OUTPUT_NAME)
target_compile_definitions(
    cvwt-denoise
    PRIVATE
    PROGRAM_NAME="${DENOISE_PROGRAM_NAME}"
)

install(
    TARGETS
    cvwt-denoise
    COMPONENT cvwt-denoise
    EXCLUDE_FROM_ALL
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#   ----------------------------------------------------------------------------
#   Group Examples Into Single Target
#   ----------------------------------------------------------------------------
if(NOT TARGET examples)
  add_custom_target(examples)
  add_dependencies(examples cvwt-denoise cvwt-denoise)
endif()

install(
    TARGETS
    cvwt-denoise
    cvwt-dwt2d
    COMPONENT examples
    EXCLUDE_FROM_ALL
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)


