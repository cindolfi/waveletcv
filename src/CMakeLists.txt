#   ----------------------------------------------------------------------------
#   Options
#   ----------------------------------------------------------------------------
option(CVWT_ENABLE_EXCEPTIONS "Enable exception throwing globally" ON)
option(CVWT_ENABLE_DWT2D_EXCEPTIONS "Enable exception throwing by DWT2D" ON)
option(CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS "Enable exception throwing by DWT2D::Coeffs" ON)
option(CVWT_ENABLE_FILTER_BANK_EXCEPTIONS "Enable exception throwing by FilterBank" ON)
option(CVWT_ENABLE_WAVELET_EXCEPTIONS "Enable exception throwing by Wavelet and related functions" ON)
option(CVWT_ENABLE_DWT2D_WARNINGS "Enable warnings logged by DWT2D" ON)
set(CVWT_USE_NLOPT_VERSION "2.7.1" CACHE STRING "The version of NLopt library")

message(VERBOSE "CVWT_ENABLE_EXCEPTIONS ${CVWT_ENABLE_EXCEPTIONS}")
message(VERBOSE "CVWT_ENABLE_DWT2D_EXCEPTIONS ${CVWT_ENABLE_DWT2D_EXCEPTIONS}")
message(VERBOSE "CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS ${CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS}")
message(VERBOSE "CVWT_ENABLE_FILTER_BANK_EXCEPTIONS ${CVWT_ENABLE_FILTER_BANK_EXCEPTIONS}")
message(VERBOSE "CVWT_ENABLE_WAVELET_EXCEPTIONS ${CVWT_ENABLE_WAVELET_EXCEPTIONS}")
message(VERBOSE "CVWT_ENABLE_DWT_WARNINGS ${CVWT_DWT_WARNINGS}")
message(VERBOSE "CVWT_FORCE_BUILD_NLOPT ${CVWT_FORCE_BUILD_NLOPT}")

#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
option(NLOPT_FORTRAN "enable fortran" OFF)
option(NLOPT_PYTHON "build python bindings" OFF)
option(NLOPT_OCTAVE "build octave bindings" OFF)
option(NLOPT_MATLAB "build matlab bindings" OFF)
option(NLOPT_GUILE "build guile bindings" OFF)
option(NLOPT_SWIG "use SWIG to build bindings" OFF)
FetchContent_Declare(
    nlopt
    GIT_REPOSITORY "https://github.com/stevengj/nlopt.git"
    GIT_TAG        "v${CVWT_USE_NLOPT_VERSION}"
    FIND_PACKAGE_ARGS
        "${CVWT_USE_NLOPT_VERSION}"
        CONFIGS NLoptConfig.cmake
)
FetchContent_MakeAvailable(nlopt)

#   ----------------------------------------------------------------------------
#   Build Library
#   ----------------------------------------------------------------------------
add_library(cvwt)
target_sources (
    cvwt
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS ${CVWT_INCLUDE_DIR}
        FILES
            ${CVWT_INCLUDE_DIR}/cvwt/wavelet.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/filterbank.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/dwt2d.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/utils.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/dispatch.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/array.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/array/compare.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/array/array.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/array/statistics.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/filters/daubechies.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/filters/symlets.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/filters/coiflets.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/filters/biorthogonal.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/exception.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/shrink.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/shrink/shrink.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/shrink/bayes.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/shrink/sure.hpp
            ${CVWT_INCLUDE_DIR}/cvwt/shrink/universal.hpp
    PRIVATE
        wavelet.cpp
        filterbank.cpp
        dwt2d.cpp
        dwt2d_coeffsexpr.cpp
        utils.cpp
        array/compare.cpp
        array/statistics.cpp
        array/array.cpp
        shrink/shrink.cpp
        shrink/bayes.cpp
        shrink/sure.cpp
        shrink/universal.cpp
)
target_link_libraries(
    cvwt
    ${OpenCV_LIBS}
    nlopt
)

if(CVWT_ENABLE_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC CVWT_EXCEPTIONS_ENABLED=true
        PUBLIC CVWT_NOEXCEPT=
        PUBLIC CVWT_NORETURN=[[noreturn]]
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_EXCEPTIONS_ENABLED=false
        CVWT_NOEXCEPT=
        CVWT_NORETURN=
    )
endif()

if(CVWT_ENABLE_DWT2D_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_EXCEPTIONS_ENABLED=CVWT_EXCEPTIONS_ENABLED
        CVWT_DWT2D_NOEXCEPT=CVWT_NOEXCEPT
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_EXCEPTIONS_ENABLED=false
        CVWT_DWT2D_NOEXCEPT=noexcept
    )
endif()

if(CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_COEFFS_EXCEPTIONS_ENABLED=CVWT_EXCEPTIONS_ENABLED
        CVWT_DWT2D_COEFFS_NOEXCEPT=CVWT_NOEXCEPT
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_COEFFS_EXCEPTIONS_ENABLED=false
        CVWT_DWT2D_COEFFS_NOEXCEPT=noexcept
    )
endif()

if(CVWT_ENABLE_FILTER_BANK_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_FILTER_BANK_EXCEPTIONS_ENABLED=CVWT_EXCEPTIONS_ENABLED
        CVWT_FILTER_BANK_NOEXCEPT=CVWT_NOEXCEPT
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_FILTER_BANK_EXCEPTIONS_ENABLED=false
        CVWT_FILTER_BANK_NOEXCEPT=noexcept
    )
endif()

if(CVWT_ENABLE_WAVELET_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_WAVELET_EXCEPTIONS_ENABLED=CVWT_EXCEPTIONS_ENABLED
        CVWT_WAVELET_NOEXCEPT=CVWT_NOEXCEPT
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_WAVELET_EXCEPTIONS_ENABLED=false
        CVWT_WAVELET_NOEXCEPT=noexcept
    )
endif()

if(CVWT_ENABLE_DWT2D_WARNINGS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_WARNINGS_ENABLED=true
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_WARNINGS_ENABLED=false
    )
endif()

#   ----------------------------------------------------------------------------
#   Install Library
#   ----------------------------------------------------------------------------
install(
    TARGETS cvwt
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET HEADERS
)

