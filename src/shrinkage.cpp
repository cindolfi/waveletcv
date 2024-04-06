
#include "cvwt/shrinkage.hpp"

#include <ranges>
#include <algorithm>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace cvwt
{
cv::Scalar estimate_stdev(cv::InputArray x)
{
    return median(cv::abs(x.getMat())) / 0.675;
}

cv::Scalar estimate_stdev(cv::InputArray x, cv::InputArray mask)
{
    return median(cv::abs(x.getMat()), mask) / 0.675;
}

/**
 * -----------------------------------------------------------------------------
 * Thresholding
 * -----------------------------------------------------------------------------
*/
using ThresholdFunction = void(cv::InputOutputArray, cv::Scalar);
using MaskedThresholdFunction = void(cv::InputOutputArray, cv::Scalar, cv::InputArray);

void soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold)
{
    internal::dispatch_on_pixel_type<internal::soft_threshold>(
        input.type(),
        input,
        output,
        threshold
    );
}

void soft_threshold(
    cv::InputArray input,
    cv::OutputArray output,
    cv::Scalar threshold,
    cv::InputArray mask
)
{
    internal::dispatch_on_pixel_type<internal::soft_threshold>(
        input.type(),
        input,
        output,
        threshold,
        mask
    );
}

void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold)
{
    internal::dispatch_on_pixel_type<internal::soft_threshold>(
        array.type(),
        array,
        threshold
    );
}

void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask)
{
    internal::dispatch_on_pixel_type<internal::soft_threshold>(
        array.type(),
        array,
        threshold,
        mask
    );
}

void hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold)
{
    internal::dispatch_on_pixel_type<internal::hard_threshold>(
        input.type(),
        input,
        output,
        threshold
    );
}

void hard_threshold(
    cv::InputArray input,
    cv::OutputArray output,
    cv::Scalar threshold,
    cv::InputArray mask
)
{
    internal::dispatch_on_pixel_type<internal::hard_threshold>(
        input.type(),
        input,
        output,
        threshold,
        mask
    );
}

void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold)
{
    internal::dispatch_on_pixel_type<internal::hard_threshold>(
        array.type(),
        array,
        threshold
    );
}

void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask)
{
    internal::dispatch_on_pixel_type<internal::hard_threshold>(
        array.type(),
        array,
        threshold,
        mask
    );
}


/**
 * -----------------------------------------------------------------------------
 * Shrink Coefficients
 * -----------------------------------------------------------------------------
*/
void shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    MaskedThresholdFunction threshold_function,
    int lower_level,
    int upper_level
)
{
    auto detail_mask = coeffs.detail_mask(lower_level, upper_level);
    threshold_function(coeffs, threshold, detail_mask);
}

void soft_shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    int lower_level,
    int upper_level
)
{
    shrink_details(coeffs, threshold, soft_threshold, lower_level, upper_level);
}

void hard_shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    int lower_level,
    int upper_level
)
{
    shrink_details(coeffs, threshold, hard_threshold, lower_level, upper_level);
}

//  ----------------------------------------------------------------------------
void shrink_detail_levels(
    DWT2D::Coeffs& coeffs,
    cv::InputArray level_thresholds,
    ThresholdFunction threshold_function
)
{
    auto level_thresholds_matrix = level_thresholds.getMat();
    if (level_thresholds_matrix.rows > level_thresholds_matrix.cols)
        level_thresholds_matrix = level_thresholds_matrix.t();

    assert(level_thresholds.channels() == 4);
    assert(level_thresholds_matrix.rows == 1);

    level_thresholds_matrix.forEach<cv::Scalar>(
        [&](const auto& threshold, const auto position) {
            int level = position[1];
            for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL}) {
                auto subband_detail = coeffs.detail(level, subband);
                threshold_function(subband_detail, threshold);
            }
        }
    );
}

void soft_shrink_detail_levels(DWT2D::Coeffs& coeffs, cv::InputArray thresholds)
{
    shrink_detail_levels(coeffs, thresholds, soft_threshold);
}

void hard_shrink_detail_levels(DWT2D::Coeffs& coeffs, cv::InputArray thresholds)
{
    shrink_detail_levels(coeffs, thresholds, hard_threshold);
}

//  ----------------------------------------------------------------------------
void shrink_detail_subbands(
    DWT2D::Coeffs& coeffs,
    cv::InputArray subband_thresholds,
    ThresholdFunction threshold_function
)
{
    assert(subband_thresholds.channels() == 4);

    subband_thresholds.getMat().forEach<cv::Scalar>(
        [&](const auto& threshold, auto position) {
            int level = position[0];
            int subband = position[1];
            auto subband_detail = coeffs.detail(level, subband);
            threshold_function(subband_detail, threshold);
        }
    );
}

void soft_shrink_detail_subbands(DWT2D::Coeffs& coeffs, cv::InputArray thresholds)
{
    shrink_detail_subbands(coeffs, thresholds, soft_threshold);
}

void hard_shrink_detail_subbands(DWT2D::Coeffs& coeffs, cv::InputArray thresholds)
{
    shrink_detail_subbands(coeffs, thresholds, hard_threshold);
}


/**
 * -----------------------------------------------------------------------------
 * Universal / VisuShrink
 * -----------------------------------------------------------------------------
*/
cv::Scalar universal_threshold(const DWT2D::Coeffs& coeffs, const cv::Scalar& stdev)
{
    return universal_threshold(coeffs.total() - coeffs.approx().total(), stdev);
}

cv::Scalar universal_threshold(cv::InputArray details, const cv::Scalar& stdev)
{
    return universal_threshold(details.total(), stdev);
}

cv::Scalar universal_threshold(cv::InputArray details, cv::InputArray mask, const cv::Scalar& stdev)
{
    assert(details.size() == mask.size());
    return universal_threshold(cv::countNonZero(mask), stdev);
}

cv::Scalar universal_threshold(int num_elements, const cv::Scalar& stdev)
{
    return stdev * std::sqrt(2.0 * std::log(num_elements));
}

cv::Scalar visu_shrink_threshold(const DWT2D::Coeffs& coeffs)
{
    return universal_threshold(coeffs, estimate_stdev(coeffs, coeffs.detail_mask()));
}

cv::Scalar visu_shrink_threshold(cv::InputArray details)
{
    return universal_threshold(details, estimate_stdev(details));
}

cv::Scalar visu_shrink_threshold(cv::InputArray details, cv::InputArray mask)
{
    return universal_threshold(details, mask, estimate_stdev(details, mask));
}


void visu_soft_shrink(DWT2D::Coeffs& coeffs)
{
    auto threshold = visu_shrink_threshold(coeffs);
    soft_shrink_details(coeffs, threshold);
}

void visu_hard_shrink(DWT2D::Coeffs& coeffs)
{
    auto threshold = visu_shrink_threshold(coeffs);
    hard_shrink_details(coeffs, threshold);
}


/**
 * -----------------------------------------------------------------------------
 * SureShrink
 * -----------------------------------------------------------------------------
*/
cv::Scalar compute_sure_threshold(
    cv::InputArray input,
    const cv::Scalar& stdev,
    SureShrinkVariant variant,
    nlopt::algorithm algorithm
)
{
    cv::Scalar result;
    internal::dispatch_on_pixel_type<internal::compute_sure_threshold>(
        input.type(),
        input,
        stdev,
        algorithm,
        variant,
        (variant == HYBRID_SURE_SHRINK) ? universal_threshold(input, stdev) : cv::Scalar(),
        result
    );

    return result;
}

cv::Scalar compute_sure_threshold(
    cv::InputArray input,
    cv::InputArray mask,
    const cv::Scalar& stdev,
    SureShrinkVariant variant,
    nlopt::algorithm algorithm
)
{
    cv::Scalar result;
    internal::dispatch_on_pixel_type<internal::compute_sure_threshold>(
        input.type(),
        input,
        mask,
        stdev,
        algorithm,
        variant,
        (variant == HYBRID_SURE_SHRINK) ? universal_threshold(input, stdev) : cv::Scalar(),
        result
    );

    return result;
}

/**
 * Returns a matrix of thresholds where rows corresond to levels and cols correspond to subbands
 * i.e. levels x 3 matrix where
 * row k <=> level k
 * column 0 <=> horizontal, column 1 <=> vertical, column 2 <=> diaganal
*/
cv::Mat4d sure_shrink_subband_thresholds(
    const DWT2D::Coeffs& coeffs,
    int levels,
    SureShrinkVariant variant,
    nlopt::algorithm algorithm
)
{
    cv::Mat4d thresholds(levels, 3);
    for (int level = 0; level < levels; ++level) {
        for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL}) {
            auto detail_coeffs = coeffs.detail(level, subband);
            auto stdev = estimate_stdev(detail_coeffs);
            thresholds(level, subband) = compute_sure_threshold(detail_coeffs, stdev, variant, algorithm);
        }
    }

    return thresholds;
}

cv::Mat4d sure_shrink_level_thresholds(
    const DWT2D::Coeffs& coeffs,
    int levels,
    SureShrinkVariant variant,
    nlopt::algorithm algorithm
)
{
    cv::Mat4d thresholds(levels, 1);
    for (int level = 0; level < levels; ++level) {
        cv::Mat detail_coeffs;
        collect_masked(coeffs, detail_coeffs, coeffs.detail_mask(level));
        auto stdev = estimate_stdev(detail_coeffs);
        thresholds(level, 1) = compute_sure_threshold(detail_coeffs, stdev, variant, algorithm);
    }

    return thresholds;
}

void sure_shrink(DWT2D::Coeffs& coeffs)
{
    auto thresholds = sure_shrink_subband_thresholds(
        coeffs,
        coeffs.levels(),
        NORMAL_SURE_SHRINK,
        DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM
    );
    soft_shrink_detail_subbands(coeffs, thresholds);
}

void sure_shrink(DWT2D::Coeffs& coeffs, int levels)
{
    auto thresholds = sure_shrink_subband_thresholds(
        coeffs,
        levels,
        NORMAL_SURE_SHRINK,
        DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM
    );
    soft_shrink_detail_subbands(coeffs, thresholds);
}

void sure_shrink(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm)
{
    auto thresholds = sure_shrink_subband_thresholds(
        coeffs,
        levels,
        NORMAL_SURE_SHRINK,
        algorithm
    );
    soft_shrink_detail_subbands(coeffs, thresholds);
}

void sure_shrink_levelwise(DWT2D::Coeffs& coeffs)
{
    auto thresholds = sure_shrink_level_thresholds(
        coeffs,
        coeffs.levels(),
        NORMAL_SURE_SHRINK,
        DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

void sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels)
{
    auto thresholds = sure_shrink_level_thresholds(
        coeffs,
        levels,
        NORMAL_SURE_SHRINK,
        DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

void sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm)
{
    auto thresholds = sure_shrink_level_thresholds(
        coeffs,
        levels,
        NORMAL_SURE_SHRINK,
        algorithm
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

void hybrid_sure_shrink(DWT2D::Coeffs& coeffs)
{
    auto thresholds = sure_shrink_subband_thresholds(
        coeffs,
        coeffs.levels(),
        HYBRID_SURE_SHRINK,
        DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

void hybrid_sure_shrink(DWT2D::Coeffs& coeffs, int levels)
{
    auto thresholds = sure_shrink_subband_thresholds(
        coeffs,
        levels,
        HYBRID_SURE_SHRINK,
        DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

void hybrid_sure_shrink(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm)
{
    auto thresholds = sure_shrink_subband_thresholds(
        coeffs,
        levels,
        HYBRID_SURE_SHRINK,
        algorithm
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs)
{
    auto thresholds = sure_shrink_level_thresholds(
        coeffs,
        coeffs.levels(),
        HYBRID_SURE_SHRINK,
        DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels)
{
    auto thresholds = sure_shrink_level_thresholds(
        coeffs,
        levels,
        HYBRID_SURE_SHRINK,
        DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm)
{
    auto thresholds = sure_shrink_level_thresholds(
        coeffs,
        levels,
        HYBRID_SURE_SHRINK,
        algorithm
    );
    soft_shrink_detail_levels(coeffs, thresholds);
}

/**
 * -----------------------------------------------------------------------------
 * BayesShrink
 * -----------------------------------------------------------------------------
*/
cv::Scalar bayes_shrink_threshold(const DWT2D::Coeffs& coeffs)
{
    return cv::Scalar();
}

void bayes_shrink(DWT2D::Coeffs& coeffs)
{
}
}   // namespace cvwt

