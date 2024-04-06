#   ----------------------------------------------------------------------------
#   Options
#   ----------------------------------------------------------------------------
option(CVWT_ARGUMENT_CHECKING "Build exception throwing for bad function arguments" ON)
option(CVWT_DWT_WARNINGS "Build log warnings when performing DWT with border effects" ON)

message(STATUS "CVWT_ARGUMENT_CHECKING ${CVWT_ARGUMENT_CHECKING}")
message(STATUS "CVWT_DWT_WARNINGS ${CVWT_DWT_WARNINGS}")

#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
FetchContent_Declare(
    nlopt
    GIT_REPOSITORY "https://github.com/stevengj/nlopt.git"
    GIT_TAG        v2.7.1
)
FetchContent_MakeAvailable(nlopt)

#   ----------------------------------------------------------------------------
#   Build Library
#   ----------------------------------------------------------------------------
set(CVWT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

add_library(cvwt_library)
target_sources (
    cvwt_library
    PUBLIC
        ${CVWT_INCLUDE_DIR}/cvwt/wavelet.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/filterbank.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/dwt2d.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/shrinkage.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/utils.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/daubechies.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/symlets.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/coiflets.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/biorthogonal.hpp
    PRIVATE
        wavelet.cpp
        filterbank.cpp
        dwt2d.cpp
        shrinkage.cpp
        utils.cpp
)
target_include_directories(
    cvwt_library
    PUBLIC
    ${CVWT_INCLUDE_DIR}
)
target_link_libraries(
    cvwt_library
    ${OpenCV_LIBS}
    nlopt
)
target_compile_features(
    cvwt_library
    PRIVATE
    cxx_std_20
)

if(CVWT_ARGUMENT_CHECKING)
    target_compile_definitions(
        cvwt_library
        PRIVATE
        CVWT_ARGUMENT_CHECKING_ENABLED=true
    )
else()
    target_compile_definitions(
        cvwt_library
        PRIVATE
        CVWT_ARGUMENT_CHECKING_ENABLED=false
    )
endif()

if(CVWT_DWT_WARNINGS)
    target_compile_definitions(
        cvwt_library
        PRIVATE
        CVWT_DISABLE_DWT_WARNINGS_ENABLED=true
    )
else()
    target_compile_definitions(
        cvwt_library
        PRIVATE
        CVWT_DISABLE_DWT_WARNINGS_ENABLED=false
    )
endif()

