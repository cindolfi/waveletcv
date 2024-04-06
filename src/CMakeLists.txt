# Optionally glob, but only for CMake 3.12 or later:
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${Wavelet_SOURCE_DIR}/include/wavelet/*.hpp")
set(CVWT_INCLUDE_DIR "${Wavelet_SOURCE_DIR}/include/cvwt")
set(
    HEADER_LIST
    "${CVWT_INCLUDE_DIR}/wavelet.hpp"
    "${CVWT_INCLUDE_DIR}/filterbank.hpp"
    "${CVWT_INCLUDE_DIR}/dwt2d.hpp"
    "${CVWT_INCLUDE_DIR}/shrinkage.hpp"
    "${CVWT_INCLUDE_DIR}/utils.hpp"
    "${CVWT_INCLUDE_DIR}/daubechies.hpp"
    "${CVWT_INCLUDE_DIR}/symlets.hpp"
    "${CVWT_INCLUDE_DIR}/coiflets.hpp"
    "${CVWT_INCLUDE_DIR}/biorthogonal.hpp"
)

FetchContent_Declare(
    nlopt
    GIT_REPOSITORY "https://github.com/stevengj/nlopt.git"
    GIT_TAG        v2.7.1
)
FetchContent_MakeAvailable(nlopt)

#   ----------------------------------------------------------------------------
#   Add Library Target
#   ----------------------------------------------------------------------------
add_library(
    wavelet_library
    wavelet.cpp
    filterbank.cpp
    dwt2d.cpp
    shrinkage.cpp
    utils.cpp
    ${HEADER_LIST}
)

#   ----------------------------------------------------------------------------
#   Target Settings
#   ----------------------------------------------------------------------------
option(CVWT_ARGUMENT_CHECKING "Build exception throwing for bad function arguments" ON)
option(CVWT_DWT_WARNINGS "Build log warnings when performing DWT with border effects" ON)

target_include_directories(wavelet_library PUBLIC ../include)
target_link_libraries(
    wavelet_library
    ${OpenCV_LIBS}
    nlopt
)
target_compile_features(wavelet_library PUBLIC cxx_std_20)

message(STATUS "CVWT_ARGUMENT_CHECKING ${CVWT_ARGUMENT_CHECKING}")
message(STATUS "CVWT_DWT_WARNINGS ${CVWT_DWT_WARNINGS}")

if(CVWT_ARGUMENT_CHECKING)
    target_compile_definitions(
        wavelet_library
        PUBLIC
        CVWT_ARGUMENT_CHECKING_ENABLED=true
    )
else()
    target_compile_definitions(
        wavelet_library
        PUBLIC
        CVWT_ARGUMENT_CHECKING_ENABLED=false
    )
endif()

if(CVWT_DWT_WARNINGS)
    target_compile_definitions(
        wavelet_library
        PUBLIC
        CVWT_DISABLE_DWT_WARNINGS_ENABLED=true
    )
else()
    target_compile_definitions(
        wavelet_library
        PUBLIC
        CVWT_DISABLE_DWT_WARNINGS_ENABLED=false
    )
endif()

message(STATUS ${COMPILE_DEFINITIONS})

#   ----------------------------------------------------------------------------
#   IDE Settings
#   ----------------------------------------------------------------------------
# IDEs should put the headers in a nice place
source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
