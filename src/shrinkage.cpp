
#include "cvwt/shrinkage.hpp"

#include <ranges>
#include <algorithm>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace cvwt
{
//  ============================================================================
//  Low Level API
//  ============================================================================
//  ----------------------------------------------------------------------------
//  Thresholding
//  ----------------------------------------------------------------------------
void soft_threshold(
    cv::InputArray input,
    cv::OutputArray output,
    cv::Scalar threshold,
    cv::InputArray mask
)
{
    if (is_no_array(mask))
        internal::dispatch_on_pixel_type<internal::SoftThreshold>(
            input.type(), input, output, threshold
        );
    else
        internal::dispatch_on_pixel_type<internal::SoftThreshold>(
            input.type(), input, output, threshold, mask
        );
}

void hard_threshold(
    cv::InputArray input,
    cv::OutputArray output,
    cv::Scalar threshold,
    cv::InputArray mask
)
{
    if (is_no_array(mask))
        internal::dispatch_on_pixel_type<internal::HardThreshold>(
            input.type(), input, output, threshold
        );
    else
        internal::dispatch_on_pixel_type<internal::HardThreshold>(
            input.type(), input, output, threshold, mask
        );
}

//  ----------------------------------------------------------------------------
//  Shrink
//  ----------------------------------------------------------------------------
void shrink_globally(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    ThresholdFunction threshold_function,
    const cv::Range& levels
)
{
    auto detail_mask = coeffs.detail_mask(levels);
    threshold_function(coeffs, coeffs, threshold, detail_mask);
}

void shrink_levels(
    DWT2D::Coeffs& coeffs,
    cv::InputArray level_thresholds,
    ThresholdFunction threshold_function,
    const cv::Range& levels
)
{
    auto resolved_levels = coeffs.resolve_level_range(levels);
    assert(level_thresholds.rows() == resolved_levels.size());
    assert(level_thresholds.cols() == 1);
    assert(level_thresholds.channels() == 4);

    int first_level = resolved_levels.start;
    level_thresholds.getMat().forEach<cv::Scalar>(
        [&](const auto& threshold, const auto position) {
            int level = resolved_levels.start + position[0];
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
    ThresholdFunction threshold_function,
    const cv::Range& levels
)
{
    auto resolved_levels = coeffs.resolve_level_range(levels);
    assert(subband_thresholds.rows() == resolved_levels.size());
    assert(subband_thresholds.cols() == 3);
    assert(subband_thresholds.channels() == 4);

    subband_thresholds.getMat().forEach<cv::Scalar>(
        [&](const auto& threshold, auto position) {
            int level = resolved_levels.start + position[0];
            int subband = position[1];
            auto subband_detail = coeffs.detail(level, subband);
            threshold_function(subband_detail, subband_detail, threshold, cv::noArray());
        }
    );
}

//  ----------------------------------------------------------------------------
//  Universal
//  ----------------------------------------------------------------------------
cv::Scalar compute_universal_threshold(int num_elements, const cv::Scalar& stdev)
{
    return stdev * std::sqrt(2.0 * std::log(num_elements));
}

cv::Scalar compute_universal_threshold(const DWT2D::Coeffs& coeffs, const cv::Scalar& stdev)
{
    return compute_universal_threshold(
        coeffs.total_details(),
        internal::set_unused_channels(stdev, coeffs.channels(), 0.0)
    );
}

cv::Scalar compute_universal_threshold(cv::InputArray details, const cv::Scalar& stdev)
{
    return compute_universal_threshold(
        details.total(),
        internal::set_unused_channels(stdev, details.channels(), 0.0)
    );
}

cv::Scalar compute_universal_threshold(cv::InputArray details, cv::InputArray mask, const cv::Scalar& stdev)
{
    internal::throw_if_bad_mask_type(mask);
    internal::throw_if_bad_mask_for_array(details, mask, internal::AllowedMaskChannels::SINGLE_OR_SAME);

    return compute_universal_threshold(
        cv::countNonZero(mask),
        internal::set_unused_channels(stdev, details.channels(), 0.0)
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
    const std::lock_guard<std::mutex> lock(_mutex);
    start(coeffs, levels, stdev);
    cv::Mat4d subset_thresholds;
    try {
        if (&coeffs != &shrunk_coeffs)
            shrunk_coeffs = coeffs.clone();

        subset_thresholds = compute_partition_thresholds(coeffs, levels, stdev);
        if (!is_no_array(thresholds))
            thresholds.assign(subset_thresholds);

        switch (partition()) {
        case Shrink::GLOBALLY:
            cvwt::shrink_globally(
                shrunk_coeffs,
                subset_thresholds.at<cv::Scalar>(0, 0),
                threshold_function(),
                levels
            );
            break;
        case Shrink::LEVELS:
            cvwt::shrink_levels(shrunk_coeffs, subset_thresholds, threshold_function(), levels);
            break;
        case Shrink::SUBBANDS:
            cvwt::shrink_subbands(shrunk_coeffs, subset_thresholds, threshold_function(), levels);
            break;
        case Shrink::SUBSETS:
            shrink_subsets(shrunk_coeffs, subset_thresholds, levels);
            break;
        }
    } catch (...) {
        finish(coeffs, levels, stdev, shrunk_coeffs, subset_thresholds);
        throw;
    }

    finish(coeffs, levels, stdev, shrunk_coeffs, subset_thresholds);
}

cv::Mat Shrink::expand_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Mat4d& thresholds,
    const cv::Range& levels
) const
{
    cv::Mat expanded_thresholds(
        coeffs.size(),
        CV_MAKE_TYPE(thresholds.depth(), coeffs.channels()),
        cv::Scalar::all(0.0)
    );
    auto resolved_levels = (levels == cv::Range::all()) ? cv::Range(0, coeffs.levels())
                                                        : levels;
    switch (partition()) {
    case Shrink::GLOBALLY:
        expanded_thresholds.setTo(thresholds.at<cv::Scalar>(0, 0), coeffs.detail_mask(levels));
        break;
    case Shrink::LEVELS:
        for (int level = resolved_levels.start; level < resolved_levels.end; ++level)
            for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL})
                expanded_thresholds(coeffs.detail_rect(level, subband)) = thresholds.at<cv::Scalar>(level);
        break;
    case Shrink::SUBBANDS:
        for (int level = resolved_levels.start; level < resolved_levels.end; ++level)
            for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL})
                expanded_thresholds(coeffs.detail_rect(level, subband)) = thresholds.at<cv::Scalar>(level, subband);
        break;
    case Shrink::SUBSETS:
        expand_subset_thresholds(coeffs, thresholds, levels, expanded_thresholds);
        break;
    }

    return expanded_thresholds;
}

cv::Mat4d Shrink::compute_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Range& levels,
    const cv::Scalar& stdev
) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    start(coeffs, levels, stdev);
    auto thresholds = compute_partition_thresholds(coeffs, levels, stdev);
    finish(coeffs, levels, stdev, DWT2D::Coeffs(), thresholds);

    return thresholds;
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
    const cv::Mat4d& thresholds,
    const cv::Range& levels
) const
{
    internal::throw_member_not_implemented(
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
    internal::throw_member_not_implemented(
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
    internal::throw_member_not_implemented(
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
    internal::throw_member_not_implemented(
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
    internal::throw_member_not_implemented(
        typeid(*this).name(),
        "compute_subset_thresholds()"
    );
}

void Shrink::expand_subset_thresholds(
    const DWT2D::Coeffs& coeffs,
    const cv::Mat4d& thresholds,
    const cv::Range& levels,
    cv::Mat& expanded_thresholds
) const
{
    internal::throw_member_not_implemented(
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
    cv::Range resolved_levels = resolve_levels(levels, coeffs);
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
    cv::Range resolved_levels = resolve_levels(levels, coeffs);
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


//  ----------------------------------------------------------------------------
//  SureShrink
//  ----------------------------------------------------------------------------
int SureShrink::AUTO_BRUTE_FORCE_SIZE_LIMIT = 32 * 32 * 3;
SureShrink::Optimizer SureShrink::AUTO_OPTIMIZER = SureShrink::Optimizer::SBPLX;

cv::Scalar SureShrink::compute_sure_threshold(
    cv::InputArray detail_coeffs,
    const cv::Scalar& stdev
) const
{
    cv::Scalar threshold;
    return internal::dispatch_on_pixel_type<internal::ComputeSureThreshold>(
        detail_coeffs.type(),
        detail_coeffs,
        stdev,
        resolve_optimizer(detail_coeffs),
        variant(),
        _stop_conditions
    );

    return threshold;
}

cv::Scalar SureShrink::compute_sure_threshold(
    cv::InputArray detail_coeffs,
    cv::InputArray mask,
    const cv::Scalar& stdev
) const
{
    return internal::dispatch_on_pixel_type<internal::ComputeSureThreshold>(
        detail_coeffs.type(),
        detail_coeffs,
        mask,
        stdev,
        resolve_optimizer(detail_coeffs),
        variant(),
        _stop_conditions
    );
}

SureShrink::Optimizer SureShrink::resolve_optimizer(cv::InputArray detail_coeffs) const
{
    SureShrink::Optimizer resolved_optimizer = optimizer();
    if (optimizer() == SureShrink::AUTO) {
        if (detail_coeffs.total() * detail_coeffs.channels() <= AUTO_BRUTE_FORCE_SIZE_LIMIT)
            resolved_optimizer = SureShrink::BRUTE_FORCE;
        else
            resolved_optimizer = AUTO_OPTIMIZER;
    }

    return resolved_optimizer;
}

//  ----------------------------------------------------------------------------
//  SureShrink Functional API
//  ----------------------------------------------------------------------------
DWT2D::Coeffs sure_shrink(const DWT2D::Coeffs& coeffs)
{
    SureShrink shrink;
    return shrink(coeffs);
}

void sure_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
{
    SureShrink shrink;
    shrink(coeffs, shrunk_coeffs);
}

DWT2D::Coeffs sure_shrink(DWT2D::Coeffs& coeffs, int levels)
{
    SureShrink shrink;
    return shrink(coeffs, levels);
}

void sure_shrink(DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels)
{
    SureShrink shrink;
    shrink(coeffs, shrunk_coeffs, levels);
}

DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs)
{
    SureShrink shrink(Shrink::LEVELS);
    return shrink(coeffs);
}

void sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
{
    SureShrink shrink(Shrink::LEVELS);
    shrink(coeffs, shrunk_coeffs);
}

DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, int levels)
{
    SureShrink shrink(Shrink::LEVELS);
    return shrink(coeffs, levels);
}

void sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels)
{
    SureShrink shrink(Shrink::LEVELS);
    shrink(coeffs, shrunk_coeffs, levels);
}


//  ----------------------------------------------------------------------------
//  BayesShrink
//  ----------------------------------------------------------------------------
cv::Scalar BayesShrink::compute_bayes_threshold(
    cv::InputArray detail_coeffs,
    const cv::Scalar& stdev
) const
{
    return internal::dispatch_on_pixel_type<internal::ComputeBayesThreshold>(
        detail_coeffs.type(),
        detail_coeffs,
        stdev
    );
}

cv::Scalar BayesShrink::compute_bayes_threshold(
    cv::InputArray detail_coeffs,
    cv::InputArray mask,
    const cv::Scalar& stdev
) const
{
    return internal::dispatch_on_pixel_type<internal::ComputeBayesThreshold>(
        detail_coeffs.type(),
        detail_coeffs,
        mask,
        stdev
    );
}

DWT2D::Coeffs bayes_shrink(const DWT2D::Coeffs& coeffs)
{
    BayesShrink shrink;
    return shrink(coeffs);
}

void bayes_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
{
    BayesShrink shrink;
    shrink(coeffs, shrunk_coeffs);
}

}   // namespace cvwt

