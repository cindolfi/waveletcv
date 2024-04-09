cmake_minimum_required(VERSION 3.22)
project(
    WaveletCV
    VERSION 0.1.0
    DESCRIPTION "OpenCV Wavelets Library"
    LANGUAGES CXX
)

#   ----------------------------------------------------------------------------
#   CMake Includes
#   ----------------------------------------------------------------------------
include(FetchContent)

#   ----------------------------------------------------------------------------
#   Options
#   ----------------------------------------------------------------------------
option(CVWT_BUILD_TESTS "Build tests" ON)
option(CVWT_BUILD_EXAMPLES "Build examples" ON)
option(CVWT_BUILD_DOCS "Build documentation" ON)

message(STATUS "CVWT_BUILD_TESTS ${CVWT_BUILD_TESTS}")
message(STATUS "CVWT_BUILD_EXAMPLES ${CVWT_BUILD_EXAMPLES}")
message(STATUS "CVWT_BUILD_DOCS ${CVWT_BUILD_DOCS}")

#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

#   ----------------------------------------------------------------------------
#   Define Project File Structure
#   ----------------------------------------------------------------------------
set(CVWT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

add_subdirectory(src)
if (CVWT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if (CVWT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if (CVWT_BUILD_DOCS)
    add_subdirectory(docs)
endif()

