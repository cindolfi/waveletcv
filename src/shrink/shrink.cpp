
#include "cvwt/shrink/shrink.hpp"

#include <algorithm>
#include <numeric>
// #include <opencv2/imgproc.hpp>

namespace cvwt
{
//  ============================================================================
//  Low Level API
//  ============================================================================
//  ----------------------------------------------------------------------------
//  Thresholding
//  ----------------------------------------------------------------------------
void soft_threshold(
    cv::InputArray array,
    cv::OutputArray result,
    cv::Scalar threshold,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_type<internal::SoftThreshold>(
            array.type(), array, result, threshold
        );
    else
        internal::dispatch_on_pixel_type<internal::SoftThreshold>(
            array.type(), array, result, threshold, mask
        );
}

void hard_threshold(
    cv::InputArray array,
    cv::OutputArray result,
    cv::Scalar threshold,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_type<internal::HardThreshold>(
            array.type(), array, result, threshold
        );
    else
        internal::dispatch_on_pixel_type<internal::HardThreshold>(
            array.type(), array, result, threshold, mask
        );
}

//  ----------------------------------------------------------------------------
//  Shrink
//  ----------------------------------------------------------------------------
void shrink_globally(
    DWT2D::Coeffs& coeffs,
    const cv::Scalar& threshold,
    ShrinkFunction threshold_function,
    const cv::Range& levels
)
{
    auto detail_mask = coeffs.detail_mask(levels);
    threshold_function(coeffs, coeffs, threshold, detail_mask);
}

void shrink_levels(
    DWT2D::Coeffs& coeffs,
    cv::InputArray level_thresholds,
    ShrinkFunction threshold_function,
    const cv::Range& levels
)
{
    auto resolved_levels = coeffs.resolve_level_range(levels);
    assert(level_thresholds.rows() == resolved_levels.size());
    assert(level_thresholds.cols() == 1);
    assert(level_thresholds.channels() == 4);

    level_thresholds.getMat().forEach<cv::Scalar>(
        [&](const auto& threshold, const auto index) {
            int level = resolved_levels.start + index[0];
            for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL}) {
                auto subband_detail = coeffs.detail(level, subband);
                threshold_function(subband_detail, subband_detail, threshold, cv::noArray());
            }
        }
    );
}

void shrink_subbands(
    DWT2D::Coeffs& coeffs,
    cv::InputArray subband_thresholds,
    ShrinkFunction threshold_function,
    const cv::Range& levels
)
{
    auto resolved_levels = coeffs.resolve_level_range(levels);
    assert(subband_thresholds.rows() == resolved_levels.size());
    assert(subband_thresholds.cols() == 3);
    assert(subband_thresholds.channels() == 4);

    subband_thresholds.getMat().forEach<cv::Scalar>(
        [&](const auto& threshold, auto index) {
            int level = resolved_levels.start + index[0];
            int subband = index[1];
            auto subband_detail = coeffs.detail(level, subband);
            threshold_function(subband_detail, subband_detail, threshold, cv::noArray());
        }
    );
}



//  ============================================================================
//  High Level API
//  ============================================================================
void Shrink::shrink(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& shrunk_coeffs,
    const cv::Range& levels,
    const cv::Scalar& stdev,
    cv::OutputArray thresholds
) const
{
    cv::Mat4d subset_thresholds;
    shrunk_coeffs = coeffs.clone();
    PartitioningContext context(this, coeffs, levels, stdev, &shrunk_coeffs, &subset_thresholds);

    subset_thresholds = compute_partition_thresholds(coeffs, levels, stdev);
    if (!is_not_array(thresholds))
        thresholds.assign(subset_thresholds);

    switch (partition()) {
    case Shrink::GLOBALLY:
        cvwt::shrink_globally(
            shrunk_coeffs,
            subset_thresholds.at<cv::Scalar>(0, 0),
            shrink_function(),
            levels
        );
        break;
    case Shrink::LEVELS:
        cvwt::shrink_levels(shrunk_coeffs, subset_thresholds, shrink_function(), levels);
        break;
    case Shrink::SUBBANDS:
        cvwt::shrink_subbands(shrunk_coeffs, subset_thresholds, shrink_function(), levels);
        break;
    case Shrink::SUBSETS:
        shrink_subsets(shrunk_coeffs, subset_thresholds, levels);
        break;
    }
}



void Shrink::expand_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Mat4d& subset_thresholds,
    cv::OutputArray expanded_thresholds,
    const cv::Range& levels
) const
{
    expanded_thresholds.create(
        coeffs.size(),
        CV_MAKE_TYPE(subset_thresholds.depth(), coeffs.channels())
    );
    auto expanded_thresholds_matrix = expanded_thresholds.getMat();
    expanded_thresholds_matrix = 0.0;

    auto resolved_levels = (levels == cv::Range::all()) ? cv::Range(0, coeffs.levels())
                                                        : levels;
    switch (partition()) {
    case Shrink::GLOBALLY:
        expanded_thresholds_matrix.setTo(
            subset_thresholds.at<cv::Scalar>(0, 0),
            coeffs.detail_mask(levels)
        );
        break;
    case Shrink::LEVELS:
        for (int level = resolved_levels.start; level < resolved_levels.end; ++level)
            for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL})
                expanded_thresholds_matrix(coeffs.detail_rect(level, subband)) = subset_thresholds.at<cv::Scalar>(level);
        break;
    case Shrink::SUBBANDS:
        for (int level = resolved_levels.start; level < resolved_levels.end; ++level)
            for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL})
                expanded_thresholds_matrix(coeffs.detail_rect(level, subband)) = subset_thresholds.at<cv::Scalar>(level, subband);
        break;
    case Shrink::SUBSETS:
        expand_subset_thresholds(coeffs, subset_thresholds, levels, expanded_thresholds_matrix);
        break;
    }
}

cv::Mat4d Shrink::compute_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Range& levels,
    const cv::Scalar& stdev
) const
{
    cv::Mat4d subset_thresholds;
    PartitioningContext context(this, coeffs, levels, stdev, nullptr, &subset_thresholds);

    subset_thresholds = compute_partition_thresholds(coeffs, levels, stdev);

    return subset_thresholds;
}

cv::Mat4d Shrink::compute_partition_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Range& levels,
    const cv::Scalar& stdev
) const
{
    switch (partition()) {
    case Shrink::GLOBALLY:
        return cv::Mat4d(
            cv::Size(1, 1),
            compute_global_threshold(coeffs, levels, stdev)
        );
    case Shrink::LEVELS:
        return compute_level_thresholds(coeffs, levels, stdev);
    case Shrink::SUBBANDS:
        return compute_subband_thresholds(coeffs, levels, stdev);
    case Shrink::SUBSETS:
        return compute_subset_thresholds(coeffs, levels, stdev);
    }

    return cv::Mat4d();
}

void Shrink::shrink_subsets(
    DWT2D::Coeffs& coeffs,
    const cv::Mat4d& subset_thresholds,
    const cv::Range& levels
) const
{
    throw_member_not_implemented(
        typeid(*this).name(),
        "shrink_subsets()"
    );
}

cv::Scalar Shrink::compute_global_threshold(
    const DWT2D::Coeffs& coeffs,
    const cv::Range& levels,
    const cv::Scalar& stdev
) const
{
    throw_member_not_implemented(
        typeid(*this).name(),
        "compute_global_threshold()"
    );
}

cv::Scalar Shrink::compute_level_threshold(
    const cv::Mat& detail_coeffs,
    int level,
    const cv::Scalar& stdev
) const
{
    throw_member_not_implemented(
        typeid(*this).name(),
        "compute_level_threshold()"
    );
}

cv::Scalar Shrink::compute_subband_threshold(
    const cv::Mat& detail_coeffs,
    int level,
    int subband,
    const cv::Scalar& stdev
) const
{
    throw_member_not_implemented(
        typeid(*this).name(),
        "compute_subband_threshold()"
    );
}

cv::Mat4d Shrink::compute_subset_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Range& levels,
    const cv::Scalar& stdev
) const
{
    throw_member_not_implemented(
        typeid(*this).name(),
        "compute_subset_thresholds()"
    );
}

void Shrink::expand_subset_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Mat4d& subset_thresholds,
    const cv::Range& levels,
    cv::Mat& expanded_thresholds
) const
{
    throw_member_not_implemented(
        typeid(*this).name(),
        "expand_subset_thresholds()"
    );
}

cv::Mat4d Shrink::compute_level_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Range& levels,
    const cv::Scalar& stdev
) const
{
    cv::Range resolved_levels = coeffs.resolve_level_range(levels);
    cv::Mat4d thresholds(resolved_levels.size(), 1);
    for (int level = resolved_levels.start; level < resolved_levels.end; ++level) {
        cv::Mat detail_coeffs;
        collect_masked(coeffs, detail_coeffs, coeffs.detail_mask(level));
        auto threshold = compute_level_threshold(detail_coeffs, level, stdev);
        thresholds(level - resolved_levels.start) = threshold;
    }

    return thresholds;
}

cv::Mat4d Shrink::compute_subband_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Range& levels,
    const cv::Scalar& stdev
) const
{
    cv::Range resolved_levels = coeffs.resolve_level_range(levels);
    cv::Mat4d thresholds(resolved_levels.size(), 3, cv::Scalar(0.0));
    for (int level = resolved_levels.start; level < resolved_levels.end; ++level) {
        for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL}) {
            auto detail_coeffs = coeffs.detail(level, subband);
            auto threshold = compute_subband_threshold(detail_coeffs, level, subband, stdev);
            thresholds(level - resolved_levels.start, subband) = threshold;
        }
    }

    return thresholds;
}
}   // namespace cvwt

