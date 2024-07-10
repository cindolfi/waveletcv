
#include "wtcv/shrink/universal.hpp"

#include "wtcv/utils.hpp"

namespace wtcv
{
//  ============================================================================
//  Low Level API
//  ============================================================================
//  ----------------------------------------------------------------------------
//  Universal
//  ----------------------------------------------------------------------------
cv::Scalar UniversalShrinker::compute_universal_threshold(
    int num_elements,
    const cv::Scalar& stdev
)
{
    return stdev * std::sqrt(2.0 * std::log(num_elements));
}

cv::Scalar UniversalShrinker::compute_universal_threshold(
    const DWT2D::Coeffs& coeffs,
    const cv::Scalar& stdev
)
{
    return compute_universal_threshold(
        coeffs.total_details(),
        internal::set_unused_channels(stdev, coeffs.channels(), 0.0)
    );
}

cv::Scalar UniversalShrinker::compute_universal_threshold(
    cv::InputArray detail_coeffs,
    const cv::Scalar& stdev
)
{
    return compute_universal_threshold(
        detail_coeffs.total(),
        internal::set_unused_channels(stdev, detail_coeffs.channels(), 0.0)
    );
}

cv::Scalar UniversalShrinker::compute_universal_threshold(
    cv::InputArray detail_coeffs,
    cv::InputArray mask,
    const cv::Scalar& stdev
)
{
    throw_if_bad_mask_for_array(detail_coeffs, mask, AllowedMaskChannels::SINGLE_OR_SAME);

    return compute_universal_threshold(
        cv::countNonZero(mask),
        internal::set_unused_channels(stdev, detail_coeffs.channels(), 0.0)
    );
}


//  ----------------------------------------------------------------------------
//  VisuShrink Functional API
//  ----------------------------------------------------------------------------
DWT2D::Coeffs visu_shrink(const DWT2D::Coeffs& coeffs)
{
    VisuShrinker shrink;
    return shrink(coeffs);
}

void visu_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
{
    VisuShrinker shrink;
    shrink(coeffs, shrunk_coeffs);
}

DWT2D::Coeffs visu_shrink(const DWT2D::Coeffs& coeffs, int levels)
{
    VisuShrinker shrink;
    return shrink(coeffs, levels);
}

void visu_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels)
{
    VisuShrinker shrink;
    shrink(coeffs, shrunk_coeffs, levels);
}
}   // namespace wtcv

