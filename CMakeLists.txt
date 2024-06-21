cmake_minimum_required(VERSION 3.24)

project(
    WaveletCV
    VERSION 0.1.0
    DESCRIPTION "OpenCV Wavelets Library"
    LANGUAGES CXX
)
message(STATUS "WaveletCV version ${WaveletCV_VERSION}")

#   ----------------------------------------------------------------------------
#   CMake Includes
#   ----------------------------------------------------------------------------
include(FetchContent)

#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

#   ----------------------------------------------------------------------------
#   Options
#   ----------------------------------------------------------------------------
option(CVWT_BUILD_TESTS "Build tests" OFF)
option(CVWT_BUILD_EXAMPLES "Build examples" OFF)
option(CVWT_INSTALL_EXAMPLES "Install examples" OFF)
option(CVWT_ENABLE_BUILD_IN_PLACE "Allow building in the project directory" OFF)

#   Guard against accidental in place builds.
if((CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR) AND NOT CVWT_ENABLE_BUILD_IN_PLACE)
    message(
        FATAL_ERROR [=[
Attempting to build inside the source directory.
Create a separate build directory and use:
    cmake -S <source_directory> and -B <build_directory>
Or to enable building in place use:
    cmake -DCVWT_ENABLE_BUILD_IN_PLACE=ON]=]
    )
endif()

if (CVWT_INSTALL_EXAMPLES AND NOT CVWT_BUILD_EXAMPLES)
    message(
        WARNING
        "CVWT_INSTALL_EXAMPLES is ${CVWT_INSTALL_EXAMPLES} "
        "but CVWT_BUILD_TESTS is ${CVWT_BUILD_TESTS}"
    )
endif()

message(VERBOSE "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")
message(VERBOSE "CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")
message(VERBOSE "CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR}")
message(VERBOSE "CVWT_BUILD_TESTS ${CVWT_BUILD_TESTS}")
message(VERBOSE "CVWT_BUILD_EXAMPLES ${CVWT_BUILD_EXAMPLES}")
message(VERBOSE "CVWT_INSTALL_EXAMPLES ${CVWT_INSTALL_EXAMPLES}")
message(VERBOSE "CVWT_ENABLE_BUILD_IN_PLACE ${CVWT_ENABLE_BUILD_IN_PLACE}")

#   ----------------------------------------------------------------------------
#   Global Compiler Settings
#   ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#   ----------------------------------------------------------------------------
#   Define Project File Structure
#   ----------------------------------------------------------------------------
set(CVWT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

add_subdirectory(src)
if (CVWT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if (CVWT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
add_subdirectory(docs)

#   ----------------------------------------------------------------------------
#   Uninstall Target
#   ----------------------------------------------------------------------------
if(NOT TARGET uninstall)
    add_custom_target(
        uninstall
        "${CMAKE_COMMAND}" -P "${CMAKE_MODULE_PATH}/uninstall.cmake"
    )
endif()
