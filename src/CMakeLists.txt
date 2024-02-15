# Optionally glob, but only for CMake 3.12 or later:
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${Wavelet_SOURCE_DIR}/include/wavelet/*.hpp")
set(HEADER_LIST "${Wavelet_SOURCE_DIR}/include/wavelet/wavelet.h")

#   ----------------------------------------------------------------------------
#   Add Library Target
#   ----------------------------------------------------------------------------
# Make an automatic library - will be static or dynamic based on user setting
add_library(wavelet_library wavelet.cpp ${HEADER_LIST})

#   ----------------------------------------------------------------------------
#   Target Settings
#   ----------------------------------------------------------------------------
# We need this directory, and users of our library will need it too
target_include_directories(wavelet_library PUBLIC ../include /usr/local/include/opencv4)
target_link_libraries(wavelet_library gtest_main ${OpenCV_LIBS})

# All users of this library will need at least C++20
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
