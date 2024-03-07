#ifndef WAVELET_SHRINKAGE_HPP
#define WAVELET_SHRINKAGE_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <nlopt.hpp>
#include "wavelet/wavelet.hpp"
#include "wavelet/dwt2d.hpp"
#include "wavelet/utils.hpp"

namespace wavelet
{
cv::Scalar estimate_std(cv::InputArray x);
cv::Scalar estimate_std(cv::InputArray x, cv::InputArray mask);

/**
 * -----------------------------------------------------------------------------
 * Thresholding
 * -----------------------------------------------------------------------------
*/
using ThresholdFunction = void(cv::InputOutputArray, cv::Scalar);
using MaskedThresholdFunction = void(cv::InputOutputArray, cv::Scalar, cv::InputArray);

void soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold);
void soft_threshold(
    cv::InputArray input,
    cv::OutputArray output,
    cv::Scalar threshold,
    cv::InputArray mask
);
void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold);
void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask);

void hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold);
void hard_threshold(
    cv::InputArray input,
    cv::OutputArray output,
    cv::Scalar threshold,
    cv::InputArray mask
);
void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold);
void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask);

/**
 * -----------------------------------------------------------------------------
 * Shrink Coefficients
 * -----------------------------------------------------------------------------
*/
void shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    MaskedThresholdFunction threshold_function,
    int lower_level=0,
    int upper_level=-1
);
void soft_shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    int lower_level=0,
    int upper_level=-1
);
void hard_shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    int lower_level=0,
    int upper_level=-1
);

//  ----------------------------------------------------------------------------
void shrink_detail_levels(
    DWT2D::Coeffs& coeffs,
    cv::InputArray level_thresholds,
    ThresholdFunction threshold_function
);
void soft_shrink_detail_levels(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);
void hard_shrink_detail_levels(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);

//  ----------------------------------------------------------------------------
void shrink_detail_subbands(
    DWT2D::Coeffs& coeffs,
    cv::InputArray subband_thresholds,
    ThresholdFunction threshold_function
);
void soft_shrink_detail_subbands(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);
void hard_shrink_detail_subbands(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);

/**
 * -----------------------------------------------------------------------------
 * Universal / VisuShrink
 * -----------------------------------------------------------------------------
*/
cv::Scalar universal_threshold(const DWT2D::Coeffs& coeffs, cv::Scalar stdev=1.0);
cv::Scalar visu_shrink_threshold(const DWT2D::Coeffs& coeffs);
void visu_soft_shrink(DWT2D::Coeffs& coeffs);
void visu_hard_shrink(DWT2D::Coeffs& coeffs);

/**
 * -----------------------------------------------------------------------------
 * SureShrink
 * -----------------------------------------------------------------------------
*/
enum SureShrinkVariant {
    NORMAL_SURE_SHRINK = 0,
    HYBRID_SURE_SHRINK,
};
const nlopt::algorithm DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM = nlopt::LN_NELDERMEAD;

cv::Scalar compute_sure_threshold(
    const cv::Mat& input,
    const cv::Scalar& stdev = cv::Scalar::all(1.0),
    SureShrinkVariant variant = NORMAL_SURE_SHRINK,
    nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
);
cv::Scalar compute_sure_threshold(
    const cv::Mat& input,
    cv::InputArray mask,
    const cv::Scalar& stdev = cv::Scalar::all(1.0),
    SureShrinkVariant variant = NORMAL_SURE_SHRINK,
    nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
);

cv::Mat4d sure_shrink_thresholds(
    const DWT2D::Coeffs& coeffs,
    int levels = 0,
    SureShrinkVariant variant = NORMAL_SURE_SHRINK,
    nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
);
cv::Mat4d sure_shrink_level_thresholds(
    const DWT2D::Coeffs& coeffs,
    int levels = 0,
    SureShrinkVariant variant = NORMAL_SURE_SHRINK,
    nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
);

void sure_shrink(DWT2D::Coeffs& coeffs);
void sure_shrink(DWT2D::Coeffs& coeffs, int levels);
void sure_shrink(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

void sure_shrink_levelwise(DWT2D::Coeffs& coeffs);
void sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels);
void sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

void hybrid_sure_shrink(DWT2D::Coeffs& coeffs);
void hybrid_sure_shrink(DWT2D::Coeffs& coeffs, int levels);
void hybrid_sure_shrink(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs);
void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels);
void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

/**
 * -----------------------------------------------------------------------------
 * Bayes Shrink
 * -----------------------------------------------------------------------------
*/
cv::Scalar bayes_shrink_threshold(const DWT2D::Coeffs& coeffs);
void bayes_shrink(DWT2D::Coeffs& coeffs);


//  ----------------------------------------------------------------------------
//  ----------------------------------------------------------------------------
//  ----------------------------------------------------------------------------
namespace internal
{
template <typename T, int N, typename Thresholder>
struct threshold
{
    using Pixel = cv::Vec<T, N>;

    void operator()(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold) const
    {
        assert(input.channels() == N);

        Thresholder thresholder;
        output.create(input.size(), input.type());
        auto result = output.getMat();
        input.getMat().forEach<Pixel>(
            [&](const auto& pixel, auto position) {
                auto& result_pixel = result.at<Pixel>(position);
                for (int i = 0; i < N; ++i)
                    result_pixel[i] = thresholder(pixel[i], threshold[i]);
            }
        );
    }

    void operator()(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask) const
    {
        assert(input.channels() == N);

        Thresholder thresholder;
        cv::Mat mask_mat;
        if (mask.type() == CV_8U)
            mask_mat = mask.getMat();
        else
            mask.getMat().convertTo(mask_mat, CV_8U);

        output.create(input.size(), input.type());
        auto result = output.getMat();
        input.getMat().forEach<Pixel>(
            [&](const auto& pixel, auto position) {
                if (mask_mat.at<uchar>(position)) {
                    auto& result_pixel = result.at<Pixel>(position);
                    for (int i = 0; i < N; ++i)
                        result_pixel[i] = thresholder(pixel[i], threshold[i]);
                } else {
                    result.at<Pixel>(position) = pixel;
                }
            }
        );
    }

    void operator()(cv::InputOutputArray array, cv::Scalar threshold) const
    {
        assert(array.channels() == N);

        Thresholder thresholder;
        array.getMat().forEach<Pixel>(
            [&](auto& pixel, auto position) {
                for (int i = 0; i < N; ++i)
                    pixel[i] = thresholder(pixel[i], threshold[i]);
            }
        );
    }

    void operator()(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask) const
    {
        assert(array.channels() == N);

        Thresholder thresholder;
        cv::Mat mask_mat;
        if (mask.type() == CV_8U)
            mask_mat = mask.getMat();
        else
            mask.getMat().convertTo(mask_mat, CV_8U);
        array.getMat().forEach<Pixel>(
            [&](auto& pixel, auto position) {
                if (mask_mat.at<uchar>(position))
                    for (int i = 0; i < N; ++i)
                        pixel[i] = thresholder(pixel[i], threshold[i]);
            }
        );
    }
};

struct SoftThresholder
{
    template <typename T1, typename T2>
    constexpr auto operator()(T1 x, T2 threshold) const
    {
        auto abs_x = std::fabs(x);
        return (abs_x > threshold) * std::copysign(1.0, x) * (abs_x - threshold);
    }
};

template <typename T, int N>
struct soft_threshold : public threshold<T, N, SoftThresholder> {};

struct HardThresholder
{
    template <typename T1, typename T2>
    constexpr auto operator()(T1 x, T2 threshold) const
    {
        auto abs_x = std::fabs(x);
        return (abs_x > threshold) * x;
    }
};

template <typename T, int N>
struct hard_threshold : public threshold<T, N, HardThresholder> {};


template <typename T, int N>
double nlopt_sure_threshold_objective(const std::vector<double>& x, std::vector<double>& grad, void* f_data);

template <typename T, int N>
struct compute_sure_threshold
{
    struct SureThresholdStopConditions
    {
        double threshold_rel_tol = 1e-8;
        double threshold_abs_tol = 0.0;
        double risk_rel_tol = 1e-8;
        double risk_abs_tol = 0.0;
        double max_time = 10.0;
        int max_evals = 0;
    };

    void operator()(
        cv::InputArray input,
        const cv::Scalar& stdev,
        nlopt::algorithm algorithm,
        SureShrinkVariant variant,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);
        cv::Mat channels[N];
        cv::split(input.getMat(), channels);

        for (int i = 0; i < N; ++i) {
            CV_LOG_INFO(NULL, "computing channel " << i);
            switch (variant) {
            case NORMAL_SURE_SHRINK:
                result[i] = single_channel_normal_using_nlopt(channels[i], stdev[i], algorithm);
                break;
            case HYBRID_SURE_SHRINK:
                result[i] = single_channel_hybrid_using_nlopt(channels[i], stdev[i], algorithm);
                break;
            }
        }
    }

    void operator()(
        cv::InputArray input,
        cv::InputArray mask,
        const cv::Scalar& stdev,
        nlopt::algorithm algorithm,
        SureShrinkVariant variant,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);
        cv::Mat masked_input;
        collect_masked<T, N>()(input, masked_input, mask);
        this->operator()(masked_input, stdev, algorithm, variant, result);
    }

    void operator()(
        cv::InputArray input,
        const cv::Scalar& stdev,
        SureShrinkVariant variant,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);
        cv::Mat channels[N];
        cv::split(input.getMat(), channels);
        for (int i = 0; i < N; ++i) {
            CV_LOG_INFO(NULL, "computing channel " << i);
            switch (variant) {
            case NORMAL_SURE_SHRINK:
                result[i] = single_channel_normal_using_brute_force(channels[i], stdev[i]);
                break;
            case HYBRID_SURE_SHRINK:
                result[i] = single_channel_hybrid_using_brute_force(channels[i], stdev[i]);
                break;
            }
        }
    }

    void operator()(
        cv::InputArray input,
        cv::InputArray mask,
        const cv::Scalar& stdev,
        SureShrinkVariant variant,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);
        cv::Mat masked_input;
        collect_masked<T, N>()(input, masked_input, mask);
        this->operator()(masked_input, stdev, variant, result);
    }

    void sure_risk(
        cv::InputArray input,
        const cv::Scalar& threshold,
        const cv::Scalar& stdev,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);

        cv::Mat channels[N];
        cv::split(input.getMat(), channels);
        for (int i = 0; i < N; ++i)
            result[i] = single_channel_sure_risk(channels[i], threshold[i], stdev[i]);
    }

    friend double nlopt_sure_threshold_objective<T, N>(const std::vector<double>& x, std::vector<double>& grad, void* f_data);

private:
    using nlopt_objective_data_type = std::tuple<compute_sure_threshold<T, N>*, cv::Mat, T>;

    T single_channel_normal_using_nlopt(
        const cv::Mat& channel,
        T stdev,
        nlopt::algorithm algorithm
    ) const
    {
        auto data = std::make_tuple(this, channel, stdev);

        nlopt::opt optimizer(algorithm, 1);
        optimizer.set_min_objective(nlopt_sure_threshold_objective<T, N>, &data);

        SureThresholdStopConditions stop_conditions;
        optimizer.set_maxtime(stop_conditions.max_time);
        optimizer.set_maxeval(stop_conditions.max_evals);
        optimizer.set_xtol_abs(stop_conditions.threshold_abs_tol);
        optimizer.set_xtol_rel(stop_conditions.threshold_rel_tol);
        optimizer.set_ftol_abs(stop_conditions.risk_abs_tol);
        optimizer.set_ftol_rel(stop_conditions.risk_rel_tol);

        double min_threshold;
        double max_threshold;
        cv::minMaxIdx(cv::abs(channel), &min_threshold, &max_threshold);
        optimizer.set_lower_bounds({min_threshold});
        optimizer.set_upper_bounds({max_threshold});
        std::vector<double> threshold = {0.8 * min_threshold + 0.2 * max_threshold};

        double optimal_risk;
        auto result = optimizer.optimize(threshold, optimal_risk);
        switch (result) {
            case nlopt::SUCCESS:
                CV_LOG_INFO(NULL, "nlopt success");
                break;
            case nlopt::STOPVAL_REACHED:
                CV_LOG_WARNING(NULL, "nlopt stop value reached");
                break;
            case nlopt::FTOL_REACHED:
                CV_LOG_INFO(NULL, "nlopt risk tolerance reached");
                break;
            case nlopt::XTOL_REACHED:
                CV_LOG_INFO(NULL, "nlopt threshold tolerance reached");
                break;
            case nlopt::MAXEVAL_REACHED:
                CV_LOG_WARNING(NULL, "nlopt max evals reached");
                break;
            case nlopt::MAXTIME_REACHED:
                CV_LOG_WARNING(NULL, "nlopt max time reached");
                break;
            case nlopt::FAILURE:
                CV_LOG_ERROR(NULL, "nlopt failed");
                break;
            case nlopt::INVALID_ARGS:
                CV_LOG_ERROR(NULL, "nlopt invalid args");
                break;
            case nlopt::OUT_OF_MEMORY:
                CV_LOG_ERROR(NULL, "nlopt out of memory");
                break;
            case nlopt::ROUNDOFF_LIMITED:
                CV_LOG_ERROR(NULL, "nlopt round off limited completion");
                break;
            case nlopt::FORCED_STOP:
                CV_LOG_ERROR(NULL, "nlopt forced stop");
                break;
        }
        CV_LOG_IF_INFO(NULL, result > 0, "optimal threshold = " << threshold[0] << ", optimal risk = " << optimal_risk);

        return threshold[0];
    }

    T single_channel_hybrid_using_nlopt(const cv::Mat& channel, T stdev, nlopt::algorithm algorithm) const
    {
        if (use_universal_threshold(channel, stdev))
            return universal_threshold(channel, stdev)[0];

        return single_channel_normal_using_nlopt(channel, stdev, algorithm);
    }

    T single_channel_normal_using_brute_force(const cv::Mat& channel, T stdev) const
    {
        cv::Mat flattened_channel;
        flatten(channel, flattened_channel);
        flattened_channel = flattened_channel / stdev;

        std::vector<T> risks(flattened_channel.total());
        flattened_channel.forEach<T>(
            [&](const auto& pixel, auto index) {
                risks[index[1]] = single_channel_sure_risk(flattened_channel, pixel / stdev);
            }
        );

        auto threshold_index = std::ranges::distance(
            risks.begin(),
            std::ranges::min_element(risks)
        );

        return std::fabs(flattened_channel.at<T>(threshold_index));
    }

    T single_channel_hybrid_using_brute_force(const cv::Mat& channel, T stdev) const
    {
        if (use_universal_threshold(channel, stdev))
            return universal_threshold(channel, stdev)[0];

        return single_channel_normal_using_brute_force(channel, stdev);
    }

    bool use_universal_threshold(const cv::Mat& channel, T stdev) const
    {
        int n = channel.total();
        auto mse = cv::sum(channel * channel) / (n * stdev * stdev);
        auto universal_test_statistic = 1 + std::pow(std::log2(n), 1.5) / std::sqrt(n);
        auto result = mse[0] < universal_test_statistic;
        CV_LOG_DEBUG(
            NULL,
            (result ? "using universal threshold" : "using SURE threshold")
            << "  mse = " << mse[0]
            << "  universal_test_statistic = " << universal_test_statistic
            << "  stdev = " << stdev
        );

        return result;
    }

    T single_channel_sure_risk(const cv::Mat& x, T threshold) const
    {
        // https://computing.llnl.gov/sites/default/files/jei2001.pdf
        assert(x.channels() == 1);
        auto abs_x = cv::abs(x);
        auto clamped_abs_x = cv::min(abs_x, threshold);
        return x.total()
            + cv::sum(clamped_abs_x.mul(clamped_abs_x))[0]
            - 2 * cv::countNonZero(abs_x <= threshold);
    }

    T single_channel_sure_risk(const cv::Mat& x, T threshold, T stdev) const
    {
        // https://computing.llnl.gov/sites/default/files/jei2001.pdf
        return single_channel_sure_risk(x / stdev, threshold / stdev);
    }
};

template <typename T, int N>
double nlopt_sure_threshold_objective(const std::vector<double>& x, std::vector<double>& grad, void* f_data)
{
    auto data = static_cast<compute_sure_threshold<T, N>::nlopt_objective_data_type*>(f_data);
    auto compute_sure_object = std::get<0>(*data);
    auto channel = std::get<1>(*data);
    auto stdev = std::get<2>(*data);
    return compute_sure_object->single_channel_sure_risk(channel, x[0], stdev);
};
}   // namespace internal
} // namespace wavelet

#endif  // WAVELET_SHRINKAGE_HPP






















/**
 * weight = step(abs(detail_coeffs) - threshold)
 * weight = sigmoid(abs(detail_coeffs) - threshold)
 * weight = ramp_step(abs(detail_coeffs) - threshold)
 * coeffs.details() * weight
 *
 * https://www.ijert.org/research/adaptive-wavelet-thresholding-for-image-denoising-using-various-shrinkage-under-different-noise-conditions-IJERTV1IS8439.pdf
 * Sure Shrink
 *  - sub-band adaptive threshold
 *  - separate threshold for each detail sub-band based on SURE (Stein's unbiased estimator for risk)
 *  - Goal of sure is to minimize MSE
 *  - threshold: t* = min(t, sigma * sqrt(2 * log(n)))
 *      where
 *          - t denotes values that minimizes SURE
 *          - sigma is noise variance, estimate based on ..
 *          - n is size of image
 *  - Smoothness is adaptive - if function contains abrupt changes/boundaries
 *      the reconstructed image also does
 *  - Bivariate Shrinkage
 *      - Depends on both coeff and parent
 *      - Let w2 = parent of w1 (i.e. w2 is the wavelet coeff at the same position
 *          as w1, but at next coarser scale)
 *
 *  Let y1 = w1 + n1
 *      y2 = w2 + n2
 *  where y's are noisy observations of w's
 *  Standard MAP estimators for w given y is
 *      w^(y) = argmax Pw/y(w/ y)
 *  Assumiung gaussian noise,
 *      z_w1 = ((sqrt(y1^2 + y2^2) - sqrt(3) * (sigma^2 * n) / sigma) + y1) / sqrt(y1^2+y2^2)
 *
 *
 *
 * https://wseas.com/journals/sp/2013/125714-170.pdf
 *
 * 3.1 Universal Threshold
 * T = sigma * sqrt(2 * log(N))
 *
 * 3.2 Visu Shrink
 * sigma^ = ( (median(abs(X)) / 0.675 )^2
 * T = sigma^ * sqrt(2 * log(N))
 *  - Does not deal with minimizing MSE
 *  - Cannot remove speckle noise
 *  - Exhibits better denoising than universal thresholding
 *
 * 3.3 Sure Shrink
 *  - Based on SURE
 *  - Combination of Universal threshold and SURE threshold
 *  - Distinct advantage of analytic unbiased estimator
 *
 * SURE(t; X) = d - 2#{i : |X_i <= t} + sum(min(X_i), 1, d)|^2
 * Ts = argmin SURE(t; X)
 *
 * 3.4 Bayes Shrink
 *  - Sets different threshold at each subband
 *  - Noise assumed to be gaussian
 *  - Assume Y = X + V
 *  - sigma_y^2 = sigma_x^2 + sigma_v^2
 *  - noise variance sigma_v^2 calculated using robust estimator
*/