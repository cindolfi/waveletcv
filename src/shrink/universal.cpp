
#include "cvwt/shrink/universal.hpp"

#include "cvwt/utils.hpp"
#include "cvwt/exception.hpp"

namespace cvwt
{
//  ============================================================================
//  Low Level API
//  ============================================================================
//  ----------------------------------------------------------------------------
//  Universal
//  ----------------------------------------------------------------------------
cv::Scalar UniversalShrink::compute_universal_threshold(
    int num_elements,
    const cv::Scalar& stdev
)
{
    return stdev * std::sqrt(2.0 * std::log(num_elements));
}

cv::Scalar UniversalShrink::compute_universal_threshold(
    const DWT2D::Coeffs& coeffs,
    const cv::Scalar& stdev
)
{
    return compute_universal_threshold(
        coeffs.total_details(),
        internal::set_unused_channels(stdev, coeffs.channels(), 0.0)
    );
}

cv::Scalar UniversalShrink::compute_universal_threshold(
    cv::InputArray detail_coeffs,
    const cv::Scalar& stdev
)
{
    return compute_universal_threshold(
        detail_coeffs.total(),
        internal::set_unused_channels(stdev, detail_coeffs.channels(), 0.0)
    );
}

cv::Scalar UniversalShrink::compute_universal_threshold(
    cv::InputArray detail_coeffs,
    cv::InputArray mask,
    const cv::Scalar& stdev
)
{
    internal::throw_if_bad_mask_depth(mask);
    internal::throw_if_bad_mask_for_array(detail_coeffs, mask, internal::AllowedMaskChannels::SINGLE_OR_SAME);

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
    VisuShrink shrink;
    return shrink(coeffs);
}

void visu_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
{
    VisuShrink shrink;
    shrink(coeffs, shrunk_coeffs);
}

DWT2D::Coeffs visu_shrink(DWT2D::Coeffs& coeffs, int levels)
{
    VisuShrink shrink;
    return shrink(coeffs, levels);
}

void visu_shrink(DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels)
{
    VisuShrink shrink;
    shrink(coeffs, shrunk_coeffs, levels);
}
}   // namespace cvwt

