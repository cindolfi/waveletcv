# Optionally glob, but only for CMake 3.12 or later:
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${Wavelet_SOURCE_DIR}/include/wavelet/*.hpp")
set(
    HEADER_LIST
    "${Wavelet_SOURCE_DIR}/include/wavelet/wavelet.hpp"
    "${Wavelet_SOURCE_DIR}/include/wavelet/filterbank.hpp"
    "${Wavelet_SOURCE_DIR}/include/wavelet/dwt2d.hpp"
    "${Wavelet_SOURCE_DIR}/include/wavelet/shrinkage.hpp"
    "${Wavelet_SOURCE_DIR}/include/wavelet/utils.hpp"
    "${Wavelet_SOURCE_DIR}/include/wavelet/daubechies.hpp"
    "${Wavelet_SOURCE_DIR}/include/wavelet/symlets.hpp"
    "${Wavelet_SOURCE_DIR}/include/wavelet/coiflets.hpp"
    "${Wavelet_SOURCE_DIR}/include/wavelet/biorthogonal.hpp"
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
target_include_directories(wavelet_library PUBLIC ../include)
target_link_libraries(
    wavelet_library
    ${OpenCV_LIBS}
    nlopt
)
target_compile_features(wavelet_library PUBLIC cxx_std_20)

#   ----------------------------------------------------------------------------
#   IDE Settings
#   ----------------------------------------------------------------------------
# IDEs should put the headers in a nice place
source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
