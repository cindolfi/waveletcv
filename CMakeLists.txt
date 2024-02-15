cmake_minimum_required(VERSION 3.14)
project(Wavelet)

enable_testing()

#   ----------------------------------------------------------------------------
#   CMake Includes
#   ----------------------------------------------------------------------------
include(FetchContent)

#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

#   ----------------------------------------------------------------------------
#   Define Project File Structure
#   ----------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
