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

if(CVWT_INSTALL_EXAMPLES)
    install(
        TARGETS
        cvwt-dwt2d
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

#   ----------------------------------------------------------------------------
#   Build cvwt-denoise
#   ----------------------------------------------------------------------------
add_executable(
    cvwt-denoise
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

if(CVWT_INSTALL_EXAMPLES)
    install(
        TARGETS
        cvwt-denoise
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
