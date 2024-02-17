# Optionally glob, but only for CMake 3.12 or later:
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${Wavelet_SOURCE_DIR}/include/wavelet/*.hpp")
set(
    HEADER_LIST
    "${Wavelet_SOURCE_DIR}/include/wavelet/wavelet.h"
    "${Wavelet_SOURCE_DIR}/include/wavelet/dwt2d.hpp"
)

#   ----------------------------------------------------------------------------
#   Add Library Target
#   ----------------------------------------------------------------------------
add_library(wavelet_library wavelet.cpp ${HEADER_LIST})

#   ----------------------------------------------------------------------------
#   Target Settings
#   ----------------------------------------------------------------------------
target_include_directories(wavelet_library PUBLIC ../include)
target_link_libraries(wavelet_library ${OpenCV_LIBS})
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
