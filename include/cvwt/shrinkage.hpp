#ifndef CVWT_SHRINKAGE_HPP
#define CVWT_SHRINKAGE_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <nlopt.hpp>
#include "cvwt/wavelet.hpp"
#include "cvwt/dwt2d.hpp"
#include "cvwt/utils.hpp"

namespace cvwt
{
/**
 * @brief Robust estimation of multichannel standard deviation.
 *
 * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
 * \f{equation}{
 *     \hat{\sigma_k} = \frac{\median(|x_k|)}{0.675}
 * \f}
 *
 * @param data The multichannel data.
 * @return cv::Scalar The estimated standard deviation of each channel.
 */
cv::Scalar estimate_stdev(cv::InputArray data);

/**
 * @brief Masked robust estimation of multichannel standard deviation.
 *
 * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
 * \f{equation}{
 *     \hat{\sigma_k} = \frac{\median(|x_k|)}{0.675}
 * \f}
 * where the median is taken over locations where the mask is nonzero.
 *
 * @param data The multichannel data.
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which data locations are used.
 * @return cv::Scalar The estimated standard deviation of each channel.
 */
cv::Scalar estimate_stdev(cv::InputArray data, cv::InputArray mask);


//  ----------------------------------------------------------------------------
//  Thresholding
//  ----------------------------------------------------------------------------
using ThresholdFunction = void(cv::InputOutputArray, cv::Scalar);
using MaskedThresholdFunction = void(cv::InputOutputArray, cv::Scalar, cv::InputArray);

/**
 * @brief Multichannel soft threshold
 *
 * Given input \f$x\f$ and threshold \f$\lambda\f$ the output at
 * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
 * \f{equation}{
 *     y_{ijk} = \indicator{|x_{ijk}| > \lambda_k} \, \sgn(x_{ijk}) \, (|x_{ijk}| - \lambda_k)
 * \f}
 * where \f$i\f$ is the row, \f$j\f$ is the column and \f$k\f$ is the channel.
 *
 * @param input
 * @param output
 * @param threshold
 */
void soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold);

/**
 * @brief Multichannel masked soft threshold
 *
 * Given input \f$x\f$, mask \f$m\f$, and threshold \f$\lambda\f$ the output at
 * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
 * \f{equation}{
 *     y_{ijk} =
 *     \begin{cases}
 *         \indicator{|x_{ijk}| > \lambda_k} \, \sgn(x_{ijk}) \, (|x_{ijk}| - \lambda_k) & \textrm{if} \,\, m_{ij} \ne 0 \\
 *         |x_{ijk}|                                                                     & \textrm{if} \,\, m_{ij} = 0
 *     \end{cases}
 * \f}
 *
 * @param input
 * @param output
 * @param threshold
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are thresholded.  The identity
 *             function is applied to inputs at corresponding zero entries.
 */
void soft_threshold(
    cv::InputArray input,
    cv::OutputArray output,
    cv::Scalar threshold,
    cv::InputArray mask
);

/**
 * @brief Multichannel inplace soft threshold
 *
 * This is equivalent to
 * @code{cpp}
 * soft_threshold(array, array, threshold);
 * @endcode
 *
 * @see soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold)
 *
 * @param array
 * @param threshold
 */
void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold);

/**
 * @brief Multichannel inplace masked soft threshold
 *
 * This is equivalent to
 * @code{cpp}
 * soft_threshold(array, array, threshold, mask);
 * @endcode
 *
 * @see soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask)
 *
 * @param array
 * @param threshold
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are thresholded.  The identity
 *             function is applied to inputs at corresponding zero entries.
 */
void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask);

/**
 * @brief Multichannel hard threshold
 *
 * Given input \f$x\f$ and threshold \f$\lambda\f$ the output at
 * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
 * \f{equation}{
 *     y_{ijk} = \indicator{|x_{ijk}| > \lambda_{k}} \, x_{ijk}
 * \f}
 * where \f$i\f$ is the row, \f$j\f$ is the column and \f$k\f$ is the channel.
 *
 * @param input
 * @param output
 * @param threshold
 */
void hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold);

/**
 * @brief Masked hard threshold
 *
 * Given input \f$x\f$, mask \f$m\f$, and threshold \f$\lambda\f$ the output at
 * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
 * \f{equation}{
 *     y_{ij} =
 *     \begin{cases}
 *         \indicator{|x_{ij}| > \lambda} \, x_{ij} & \textrm{if} \,\, m_{ij} \ne 0 \\
 *         |x_{ij}|                                 & \textrm{if} \,\, m_{ij} = 0
 *     \end{cases}
 * \f}
 * where \f$i\f$ is the row, \f$j\f$ is the column and \f$k\f$ is the channel.
 *
 * @param input
 * @param output
 * @param threshold
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are thresholded.  The identity
 *             function is applied to inputs at corresponding zero entries.
 */
void hard_threshold(
    cv::InputArray input,
    cv::OutputArray output,
    cv::Scalar threshold,
    cv::InputArray mask
);

/**
 * @brief Inplace hard threshold
 *
 * This is equivalent to
 * @code{cpp}
 * hard_threshold(array, array, threshold);
 * @endcode
 *
 * @see hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold)
 *
 * @param array
 * @param threshold
 */
void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold);

/**
 * @brief Inplace masked hard threshold
 *
 * This is equivalent to
 * @code{cpp}
 * hard_threshold(array, array, threshold, mask);
 * @endcode
 *
 * @see hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask)
 *
 * @param array
 * @param threshold
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are thresholded.  The identity
 *             function is applied to inputs at corresponding zero entries.
 */
void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask);


//  ----------------------------------------------------------------------------
//  Shrink Coefficients
//  ----------------------------------------------------------------------------
/**
 * @brief Shrink detail coefficients using a global threshold.
 *
 * @param coeffs
 * @param threshold
 * @param threshold_function
 * @param lower_level
 * @param upper_level
 */
void shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    MaskedThresholdFunction threshold_function,
    int lower_level=0,
    int upper_level=-1
);

/**
 * @brief Shrink detail coefficients using soft thresholding and a global threshold.
 *
 * This is a convenience function equivalent to
 * @code{cpp}
 * shrink_details(coeffs, threshold, soft_threshold, lower_level, upper_level);
 * @endcode
 *
 * @param coeffs
 * @param threshold
 * @param lower_level
 * @param upper_level
 */
void soft_shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    int lower_level=0,
    int upper_level=-1
);

/**
 * @brief Shrink detail coefficients using hard thresholding and a global threshold.
 *
 * This is a convenience function equivalent to
 * @code{cpp}
 * shrink_details(coeffs, threshold, hard_threshold, lower_level, upper_level);
 * @endcode
 *
 * @param coeffs
 * @param threshold
 * @param lower_level
 * @param upper_level
 */
void hard_shrink_details(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    int lower_level=0,
    int upper_level=-1
);

//  ----------------------------------------------------------------------------
/**
 * @brief Shrink detail coeffcients inplace using separate thresholds for each level.
 *
 * @param coeffs
 * @param level_thresholds
 * @param threshold_function
 */
void shrink_detail_levels(
    DWT2D::Coeffs& coeffs,
    cv::InputArray level_thresholds,
    ThresholdFunction threshold_function
);
/**
 * @brief Shrink detail coeffcients inplace using soft thresholding and separate thresholds for each level.
 *
 * This is a convenience function equivalent to
 * @code{cpp}
 * shrink_detail_levels(coeffs, thresholds, soft_threshold);
 * @endcode
 *
 * @param coeffs
 * @param thresholds
 */
void soft_shrink_detail_levels(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);

/**
 * @brief Shrink detail coeffcients inplace using hard thresholding and separate thresholds for each level.
 *
 * This is a convenience function equivalent to
 * @code{cpp}
 * shrink_detail_levels(coeffs, thresholds, hard_threshold);
 * @endcode
 *
 * @param coeffs
 * @param thresholds
 */
void hard_shrink_detail_levels(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);

//  ----------------------------------------------------------------------------
/**
 * @brief Shrink detail coeffcients inplace using separate thresholds for each level and subband.
 *
 * @param coeffs
 * @param subband_thresholds
 * @param threshold_function
 */
void shrink_detail_subbands(
    DWT2D::Coeffs& coeffs,
    cv::InputArray subband_thresholds,
    ThresholdFunction threshold_function
);

/**
 * @brief Shrink detail coeffcients inplace using soft thresholding and separate thresholds for each level and subband.
 *
 * This is a convenience function equivalent to
 * @code{cpp}
 * shrink_detail_subbands(coeffs, thresholds, soft_threshold);
 * @endcode
 *
 * @param coeffs
 * @param thresholds
 */
void soft_shrink_detail_subbands(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);

/**
 * @brief Shrink detail coeffcients inplace using hard thresholding and separate thresholds for each level and subband.
 *
 * This is a convenience function equivalent to
 * @code{cpp}
 * shrink_detail_subbands(coeffs, thresholds, hard_threshold);
 * @endcode
 *
 * @param coeffs
 * @param thresholds
 */
void hard_shrink_detail_subbands(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);


//  ----------------------------------------------------------------------------
//  Universal / VisuShrink
//  ----------------------------------------------------------------------------

/**
 * @brief Computes the universal shrinkage threshold.
 *
 * The universal multichannel threshold \f$\lambda\f$ is defined by
 * \f{equation}{
 *     \lambda_k = \sigma_k \, \sqrt{2 \log{N}}
 * \f}
 *
 * @see visu_shrink_threshold()
 * @see https://computing.llnl.gov/sites/default/files/jei2001.pdf

 * @param num_elements The number of detail coefficients.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @return cv::Scalar
 */
cv::Scalar universal_threshold(
    int num_elements,
    const cv::Scalar& stdev = cv::Scalar::all(1.0)
);

/**
 * @brief Computes the universal shrinkage threshold.
 *
 * This is an convenience function equivalent to
 * @code{cpp}
 * universal_threshold(coeffs, coeffs.detail_mask(), stdev);
 * @endcode
 *
 * @see visu_shrink_threshold()
 * @see https://computing.llnl.gov/sites/default/files/jei2001.pdf
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @return cv::Scalar
 */
cv::Scalar universal_threshold(
    const DWT2D::Coeffs& coeffs,
    const cv::Scalar& stdev = cv::Scalar::all(1.0)
);

/**
 * @brief Computes the universal shrinkage threshold.
 *
 * This is an convenience function equivalent to
 * @code{cpp}
 * universal_threshold(details.total(), stdev);
 * @endcode
 *
 * @see visu_shrink_threshold()
 *
 * @param details The discrete wavelet transform detail coefficients.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @return cv::Scalar
 */
cv::Scalar universal_threshold(
    cv::InputArray details,
    const cv::Scalar& stdev = cv::Scalar::all(1.0)
);

/**
 * @brief Computes the universal shrinkage threshold.
 *
 * This is an convenience function equivalent to
 * @code{cpp}
 * universal_threshold(collect_masked(details, mask), stdev);
 * @endcode
 *
 * @see visu_shrink_threshold()
 *
 * @param details The discrete wavelet transform detail coefficients.
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are used in the computation.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @return cv::Scalar
 */
cv::Scalar universal_threshold(
    cv::InputArray details,
    cv::InputArray mask,
    const cv::Scalar& stdev = cv::Scalar::all(1.0)
);


/**
 * @brief Computes the universal shrinkage threshold using a robust estimate of the standard deviation.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @return cv::Scalar
 */
cv::Scalar visu_shrink_threshold(const DWT2D::Coeffs& coeffs);

/**
 * @brief Computes the universal shrinkage threshold using a robust estimate of the standard deviation.
 *
 * @see universal_threshold()
 *
 * @param details The discrete wavelet transform detail coefficients.
 * @return cv::Scalar
 */
cv::Scalar visu_shrink_threshold(cv::InputArray details);

/**
 * @brief Computes the universal shrinkage threshold using a robust estimate of the standard deviation.
 *
 * @see universal_threshold()
 *
 * @param details The discrete wavelet transform detail coefficients.
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are used in the computation.
 * @return cv::Scalar
 */
cv::Scalar visu_shrink_threshold(cv::InputArray details, cv::InputArray mask);

/**
 * @brief Shrink detail coeffcients inplace using soft thresholding and the universal shrinkage threshold
 *
 * @see visu_shrink_threshold()
 *
 * @param coeffs The discrete wavelet transform coefficients.
 */
void visu_soft_shrink(DWT2D::Coeffs& coeffs);
/**
 * @brief Shrink detail coeffcients inplace using hard thresholding and the universal shrinkage threshold
 *
 * @see visu_shrink_threshold()
 *
 * @param coeffs The discrete wavelet transform coefficients.
 */
void visu_hard_shrink(DWT2D::Coeffs& coeffs);


//  ----------------------------------------------------------------------------
//  SureShrink
//  ----------------------------------------------------------------------------
/**
 * @brief The SureShrink Algorithm Variant
 */
enum SureShrinkVariant {
    NORMAL_SURE_SHRINK,
    HYBRID_SURE_SHRINK,
};
const nlopt::algorithm DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM = nlopt::LN_NELDERMEAD;

/**
 * @brief Compute the SureShrink algorithm threshold.
 *
 * For a single channel with \f$\sigma = 1\f$, the SURE threshold is defined by
 * \f{equation}{
 *     \lambda_{SURE} = \argmin_{\lambda \ge 0} \SURE(\lambda, w)
 * \f}
 * where the input \f$w\f$ is a subset of noisy detail coefficients measurements
 * drawn from an unknown normal distribution.
 * \f$\SURE(\lambda, w)\f$ is an unbiased estimator of the MSE between the true
 * detail coefficients and the soft thresholded coefficients using the
 * threshold \f$\lambda\f$.
 *
 * Specifically,
 * \f{equation}{
 *     \SURE(\lambda, w) =
 *         N
 *         - 2 [\# \, \mathrm{of} \, |w_n| < \lambda]
 *         + \sum_{n = 1}^{N} [\min(|w_n|, \lambda)]^2
 * \f}
 * where \f$N\f$ is the number of coefficients and
 * \f$[\# \, \mathrm{of} \, |w_n| < \lambda]\f$ is the number of coefficients lesser
 * than \f$\lambda\f$ in magnitude.
 *
 * When the detail coefficients are sparse it is preferable to employ hybrid
 * algorithm that chooses the universal_threshold() when
 * \f{equation}{
 *     \frac{1}{N} \sum_{n = 1}^{N} \left( \frac{w_n}{\sigma} - 1 \right)^2
 *     \le
 *     \frac{\left(\log_2(N)\right)^{3/2}}{\sqrt{N}}
 * \f}
 * and \f$\lambda_{SURE}\f$ otherwise.
 * Pass `HYBRID_SURE_SHRINK` as the `variant` argument to use
 * this implementation.
 *
 * When the standard deviation of each input channel is not one, each channel
 * standard deviation must be passed in using the `stdev` argument.
 * In which case, the resulting threshold will be suitably scaled to work on the
 * non-standardized coefficients.
 *
 * This function returns a multichannel threshold by applying the single channel
 * algorithm to each channel of the given `detail_coeffs`.
 *
 * @see estimate_stdev()
 * @see universal_threshold()
 *
 * @param detail_coeffs The detail coefficients.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @param variant The variant of the SureShrink algorithm.
 * @param algorithm The optimization algorithm used to compute the SURE threshold.
 * @return cv::Scalar
 */
cv::Scalar compute_sure_threshold(
    cv::InputArray detail_coeffs,
    const cv::Scalar& stdev = cv::Scalar::all(1.0),
    SureShrinkVariant variant = NORMAL_SURE_SHRINK,
    nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
);

/**
 * @brief Compute the SureShrink algorithm threshold.
 *
 * This is a convienence function that is equivalent to
 * @code{cpp}
 * compute_sure_threshold(collect_masked(detail_coeffs, mask), stdev, variant, algorithm);
 * @endcode
 *
 * @param detail_coeffs The detail coefficients.
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which detail coefficients locations are used in the computation.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @param variant The variant of the SureShrink algorithm.
 * @param algorithm The optimization algorithm used to compute the SURE threshold.
 * @return cv::Scalar
 */
cv::Scalar compute_sure_threshold(
    cv::InputArray detail_coeffs,
    cv::InputArray mask,
    const cv::Scalar& stdev = cv::Scalar::all(1.0),
    SureShrinkVariant variant = NORMAL_SURE_SHRINK,
    nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
);

/**
 * @brief Compute the SureShrink algorithm threshold for each level and subband.
 *
 * @see compute_sure_threshold()
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 * @param variant The variant of the SureShrink algorithm.
 * @param algorithm The optimization algorithm used to compute the SURE threshold.
 * @return cv::Mat4d
 */
cv::Mat4d sure_shrink_subband_thresholds(
    const DWT2D::Coeffs& coeffs,
    int levels,
    SureShrinkVariant variant = NORMAL_SURE_SHRINK,
    nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
);

/**
 * @brief Compute the SureShrink algorithm threshold for each level.
 *
 * @see compute_sure_threshold()
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 * @param variant The variant of the SureShrink algorithm.
 * @param algorithm The optimization algorithm used to compute the SURE threshold.
 * @return cv::Mat4d
 */
cv::Mat4d sure_shrink_level_thresholds(
    const DWT2D::Coeffs& coeffs,
    int levels,
    SureShrinkVariant variant = NORMAL_SURE_SHRINK,
    nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
);





/**
 * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 */
void sure_shrink(DWT2D::Coeffs& coeffs);

/**
 * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 */
void sure_shrink(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 * @param algorithm The optimization algorithm used to compute the SURE threshold.
 */
void sure_shrink(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

/**
 * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 */
void sure_shrink_levelwise(DWT2D::Coeffs& coeffs);

/**
 * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 */
void sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 * @param algorithm The optimization algorithm used to compute the SURE threshold.
 */
void sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

/**
 * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 */
void hybrid_sure_shrink(DWT2D::Coeffs& coeffs);

/**
 * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 */
void hybrid_sure_shrink(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 * @param algorithm The optimization algorithm used to compute the SURE threshold.
 */
void hybrid_sure_shrink(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

/**
 * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 */
void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs);

/**
 * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 */
void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param levels The maximum number of levels to shrink.  Shrinking is applied
 *               starting at the lowest level (i.e. smallest scale).
 * @param algorithm The optimization algorithm used to compute the SURE threshold.
 */
void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

//  ----------------------------------------------------------------------------
//  Bayes Shrink
//  ----------------------------------------------------------------------------
/**
 * @brief Shrink detail coeffcients inplace using BayesShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @return cv::Scalar
 */
cv::Scalar bayes_shrink_threshold(const DWT2D::Coeffs& coeffs);

/**
 * @brief Shrink detail coeffcients inplace using BayesShrink algorithm.
 *
 * @param coeffs The discrete wavelet transform coefficients.
 */
void bayes_shrink(DWT2D::Coeffs& coeffs);


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
        cv::Scalar univ_threshold,
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
                result[i] = single_channel_hybrid_using_nlopt(channels[i], stdev[i], univ_threshold[i], algorithm);
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
        cv::Scalar univ_threshold,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);
        cv::Mat masked_input;
        collect_masked<T, N>()(input, masked_input, mask);
        this->operator()(masked_input, stdev, algorithm, variant, univ_threshold, result);
    }

    void operator()(
        cv::InputArray input,
        const cv::Scalar& stdev,
        SureShrinkVariant variant,
        cv::Scalar univ_threshold,
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
                result[i] = single_channel_hybrid_using_brute_force(channels[i], stdev[i], univ_threshold[i]);
                break;
            }
        }
    }

    void operator()(
        cv::InputArray input,
        cv::InputArray mask,
        const cv::Scalar& stdev,
        SureShrinkVariant variant,
        cv::Scalar univ_threshold,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);
        cv::Mat masked_input;
        collect_masked<T, N>()(input, masked_input, mask);
        this->operator()(masked_input, stdev, variant, univ_threshold, result);
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

    T single_channel_hybrid_using_nlopt(const cv::Mat& channel, T stdev, T univ_threshold, nlopt::algorithm algorithm) const
    {
        if (use_universal_threshold(channel, stdev))
            return univ_threshold;

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

    T single_channel_hybrid_using_brute_force(const cv::Mat& channel, T stdev, T univ_threshold) const
    {
        if (use_universal_threshold(channel, stdev))
            return univ_threshold;

        return single_channel_normal_using_brute_force(channel, stdev);
    }

    bool use_universal_threshold(const cv::Mat& channel, T stdev) const
    {
        int n = channel.total();
        auto mse = cv::sum(channel * channel)[0] / (n * stdev * stdev);
        auto universal_test_statistic = 1 + std::pow(std::log2(n), 1.5) / std::sqrt(n);
        auto result = mse < universal_test_statistic;
        CV_LOG_DEBUG(
            NULL,
            (result ? "using universal threshold" : "using SURE threshold")
            << "  mse = " << mse
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
} // namespace cvwt

#endif  // CVWT_SHRINKAGE_HPP

