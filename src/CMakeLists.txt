#   ----------------------------------------------------------------------------
#   Options
#   ----------------------------------------------------------------------------
option(CVWT_ENABLE_EXCEPTIONS "Enable exception throwing globally" ON)
option(CVWT_ENABLE_DWT2D_EXCEPTIONS "Enable exception throwing by DWT2D" ON)
option(CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS "Enable exception throwing by DWT2D::Coeffs" ON)
option(CVWT_ENABLE_FILTER_BANK_EXCEPTIONS "Enable exception throwing by FilterBank" ON)
option(CVWT_ENABLE_WAVELET_EXCEPTIONS "Enable exception throwing by FilterBank" ON)
option(CVWT_ENABLE_DWT_WARNINGS "Enable warnings when performing DWT with border effects" ON)

message(STATUS "CVWT_ENABLE_EXCEPTIONS ${CVWT_ENABLE_EXCEPTIONS}")
message(STATUS "CVWT_ENABLE_DWT2D_EXCEPTIONS ${CVWT_ENABLE_DWT2D_EXCEPTIONS}")
message(STATUS "CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS ${CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS}")
message(STATUS "CVWT_ENABLE_FILTER_BANK_EXCEPTIONS ${CVWT_ENABLE_FILTER_BANK_EXCEPTIONS}")
message(STATUS "CVWT_ENABLE_WAVELET_EXCEPTIONS ${CVWT_ENABLE_WAVELET_EXCEPTIONS}")
message(STATUS "CVWT_ENABLE_DWT_WARNINGS ${CVWT_DWT_WARNINGS}")

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
add_library(cvwt)
target_sources (
    cvwt
    PUBLIC
        ${CVWT_INCLUDE_DIR}/cvwt/wavelet.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/filterbank.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/dwt2d.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/shrinkage.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/utils.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/dispatch.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/compare.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/array.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/statistics.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/filters/daubechies.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/filters/symlets.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/filters/coiflets.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/filters/biorthogonal.hpp
        ${CVWT_INCLUDE_DIR}/cvwt/exception.hpp
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
        compare.cpp
        statistics.cpp
        array.cpp
        shrink/shrink.cpp
        shrink/bayes.cpp
        shrink/sure.cpp
        shrink/universal.cpp
)
target_include_directories(
    cvwt
    PUBLIC
    ${CVWT_INCLUDE_DIR}
)
target_link_libraries(
    cvwt
    ${OpenCV_LIBS}
    nlopt
)
target_compile_features(
    cvwt
    PRIVATE
    cxx_std_20
)

if(CVWT_ENABLE_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_EXCEPTIONS_ENABLED=true
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_NOEXCEPT=
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_NORETURN=[[noreturn]]
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_EXCEPTIONS_ENABLED=false
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_NOEXCEPT=
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_NORETURN=
    )
endif()

if(CVWT_ENABLE_DWT2D_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_EXCEPTIONS_ENABLED=CVWT_EXCEPTIONS_ENABLED
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_NOEXCEPT=CVWT_NOEXCEPT
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_EXCEPTIONS_ENABLED=false
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_NOEXCEPT=noexcept
    )
endif()

if(CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_COEFFS_EXCEPTIONS_ENABLED=CVWT_EXCEPTIONS_ENABLED
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_COEFFS_NOEXCEPT=CVWT_NOEXCEPT
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_COEFFS_EXCEPTIONS_ENABLED=false
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_DWT2D_COEFFS_NOEXCEPT=noexcept
    )
endif()

if(CVWT_ENABLE_FILTER_BANK_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_FILTER_BANK_EXCEPTIONS_ENABLED=CVWT_EXCEPTIONS_ENABLED
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_FILTER_BANK_NOEXCEPT=CVWT_NOEXCEPT
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_FILTER_BANK_EXCEPTIONS_ENABLED=false
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_FILTER_BANK_NOEXCEPT=noexcept
    )
endif()

if(CVWT_ENABLE_WAVELET_EXCEPTIONS)
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_WAVELET_EXCEPTIONS_ENABLED=CVWT_EXCEPTIONS_ENABLED
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_WAVELET_NOEXCEPT=CVWT_NOEXCEPT
    )
else()
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_WAVELET_EXCEPTIONS_ENABLED=false
    )
    target_compile_definitions(
        cvwt
        PUBLIC
        CVWT_WAVELET_NOEXCEPT=noexcept
    )
endif()

if(CVWT_ENABLE_DWT_WARNINGS)
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

