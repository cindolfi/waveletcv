#ifndef CVWT_SHRINKAGE_HPP
#define CVWT_SHRINKAGE_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
// #include <span>
// #include <ranges>
// #include <memory>
#include <nlopt.hpp>
#include "cvwt/wavelet.hpp"
#include "cvwt/dwt2d.hpp"
#include "cvwt/utils.hpp"

namespace cvwt
{
/**
 * @brief Mean absolute deviation.
 *
 * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
 * \f{equation}{
 *     \mad(x_k) = \median(|x_k| - \median(x_k))
 * \f}
 *
 * @param data The multichannel data.
 * @return cv::Scalar The estimated standard deviation of each channel.
 */
cv::Scalar mad(cv::InputArray data);

/**
 * @brief Masked mean absolute deviation.
 *
 * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
 * \f{equation}{
 *     \mad(x_k) = \median(|x_k| - \median(x_k))
 * \f}
 * where the median is taken over locations where the mask is nonzero.
 *
 * @param data The multichannel data.
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which data locations are used.
 * @return cv::Scalar The estimated standard deviation of each channel.
 */
cv::Scalar mad(cv::InputArray data, cv::InputArray mask);

// /**
//  * @brief Multichannel robust estimation of the standard deviation of normally distributed data.
//  *
//  * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
//  * \f{equation}{
//  *     \hat{\sigma_k} = \frac{\mad(x_k)}{0.675}
//  * \f}
//  *
//  * @param data The multichannel data.
//  * @return cv::Scalar The estimated standard deviation of each channel.
//  */
// cv::Scalar mad_stdev(cv::InputArray data);

// /**
//  * @brief Masked multichannel robust estimation of the standard deviation of normally distributed data.
//  *
//  * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
//  * \f{equation}{
//  *     \hat{\sigma_k} = \frac{\mad(x_k)}{0.675}
//  * \f}
//  * where the median is taken over locations where the mask is nonzero.
//  *
//  * @param data The multichannel data.
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which data locations are used.
//  * @return cv::Scalar The estimated standard deviation of each channel.
//  */
// cv::Scalar mad_stdev(cv::InputArray data, cv::InputArray mask);

/**
 * @brief Multichannel robust estimation of the standard deviation of normally distributed data.
 *
 * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
 * \f{equation}{
 *     \hat{\sigma_k} = \frac{\mad(x_k)}{0.675}
 * \f}
 *
 * @param data The multichannel data.
 * @return cv::Scalar The estimated standard deviation of each channel.
 */
cv::Scalar mad_stdev(cv::InputArray data);

// /**
//  * @brief Masked multichannel robust estimation of the standard deviation of normally distributed data.
//  *
//  * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
//  * \f{equation}{
//  *     \hat{\sigma_k} = \frac{\mad(x_k)}{0.675}
//  * \f}
//  * where the median is taken over locations where the mask is nonzero.
//  *
//  * @param data The multichannel data.
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which data locations are used.
//  * @return cv::Scalar The estimated standard deviation of each channel.
//  */
// cv::Scalar mad_stdev(cv::InputArray data, cv::InputArray mask = cv::noArray());


//  ----------------------------------------------------------------------------
//  Thresholding
//  ----------------------------------------------------------------------------
// using ThresholdFunction = void(cv::InputOutputArray, cv::Scalar);
// using MaskedThresholdFunction = void(cv::InputOutputArray, cv::Scalar, cv::InputArray);

// /**
//  * @brief Multichannel soft threshold
//  *
//  * Given input \f$x\f$ and threshold \f$\lambda\f$ the output at
//  * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
//  * \f{equation}{
//  *     y_{ijk} = \indicator{|x_{ijk}| > \lambda_k} \, \sgn(x_{ijk}) \, (|x_{ijk}| - \lambda_k)
//  * \f}
//  * where \f$i\f$ is the row, \f$j\f$ is the column and \f$k\f$ is the channel.
//  *
//  * @param input
//  * @param output
//  * @param threshold
//  */
// void soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold);

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
    cv::InputArray mask = cv::noArray()
);

// /**
//  * @brief Multichannel inplace soft threshold
//  *
//  * This is equivalent to
//  * @code{cpp}
//  * soft_threshold(array, array, threshold);
//  * @endcode
//  *
//  * @see soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold)
//  *
//  * @param array
//  * @param threshold
//  */
// void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold);

// /**
//  * @brief Multichannel inplace masked soft threshold
//  *
//  * This is equivalent to
//  * @code{cpp}
//  * soft_threshold(array, array, threshold, mask);
//  * @endcode
//  *
//  * @see soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask)
//  *
//  * @param array
//  * @param threshold
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which input locations are thresholded.  The identity
//  *             function is applied to inputs at corresponding zero entries.
//  */
// void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask = cv::noArray());

// /**
//  * @brief Multichannel hard threshold
//  *
//  * Given input \f$x\f$ and threshold \f$\lambda\f$ the output at
//  * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
//  * \f{equation}{
//  *     y_{ijk} = \indicator{|x_{ijk}| > \lambda_{k}} \, x_{ijk}
//  * \f}
//  * where \f$i\f$ is the row, \f$j\f$ is the column and \f$k\f$ is the channel.
//  *
//  * @param input
//  * @param output
//  * @param threshold
//  */
// void hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold);

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
    cv::InputArray mask = cv::noArray()
);

// /**
//  * @brief Inplace hard threshold
//  *
//  * This is equivalent to
//  * @code{cpp}
//  * hard_threshold(array, array, threshold);
//  * @endcode
//  *
//  * @see hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold)
//  *
//  * @param array
//  * @param threshold
//  */
// void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold);

// /**
//  * @brief Inplace masked hard threshold
//  *
//  * This is equivalent to
//  * @code{cpp}
//  * hard_threshold(array, array, threshold, mask);
//  * @endcode
//  *
//  * @see hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask)
//  *
//  * @param array
//  * @param threshold
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which input locations are thresholded.  The identity
//  *             function is applied to inputs at corresponding zero entries.
//  */
// void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask = cv::noArray());

// /**
//  * @brief Multichannel soft threshold
//  *
//  * Given input \f$x\f$ and threshold \f$\lambda\f$ the output at
//  * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
//  * \f{equation}{
//  *     y_{ijk} = \indicator{|x_{ijk}| > \lambda_k} \, \sgn(x_{ijk}) \, (|x_{ijk}| - \lambda_k)
//  * \f}
//  * where \f$i\f$ is the row, \f$j\f$ is the column and \f$k\f$ is the channel.
//  *
//  * @param input
//  * @param output
//  * @param threshold
//  */
// void soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold);

// /**
//  * @brief Multichannel masked soft threshold
//  *
//  * Given input \f$x\f$, mask \f$m\f$, and threshold \f$\lambda\f$ the output at
//  * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
//  * \f{equation}{
//  *     y_{ijk} =
//  *     \begin{cases}
//  *         \indicator{|x_{ijk}| > \lambda_k} \, \sgn(x_{ijk}) \, (|x_{ijk}| - \lambda_k) & \textrm{if} \,\, m_{ij} \ne 0 \\
//  *         |x_{ijk}|                                                                     & \textrm{if} \,\, m_{ij} = 0
//  *     \end{cases}
//  * \f}
//  *
//  * @param input
//  * @param output
//  * @param threshold
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which input locations are thresholded.  The identity
//  *             function is applied to inputs at corresponding zero entries.
//  */
// void soft_threshold(
//     cv::InputArray input,
//     cv::OutputArray output,
//     cv::Scalar threshold,
//     cv::InputArray mask
// );

// /**
//  * @brief Multichannel inplace soft threshold
//  *
//  * This is equivalent to
//  * @code{cpp}
//  * soft_threshold(array, array, threshold);
//  * @endcode
//  *
//  * @see soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold)
//  *
//  * @param array
//  * @param threshold
//  */
// void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold);

// /**
//  * @brief Multichannel inplace masked soft threshold
//  *
//  * This is equivalent to
//  * @code{cpp}
//  * soft_threshold(array, array, threshold, mask);
//  * @endcode
//  *
//  * @see soft_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask)
//  *
//  * @param array
//  * @param threshold
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which input locations are thresholded.  The identity
//  *             function is applied to inputs at corresponding zero entries.
//  */
// void soft_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask);

// /**
//  * @brief Multichannel hard threshold
//  *
//  * Given input \f$x\f$ and threshold \f$\lambda\f$ the output at
//  * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
//  * \f{equation}{
//  *     y_{ijk} = \indicator{|x_{ijk}| > \lambda_{k}} \, x_{ijk}
//  * \f}
//  * where \f$i\f$ is the row, \f$j\f$ is the column and \f$k\f$ is the channel.
//  *
//  * @param input
//  * @param output
//  * @param threshold
//  */
// void hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold);

// /**
//  * @brief Masked hard threshold
//  *
//  * Given input \f$x\f$, mask \f$m\f$, and threshold \f$\lambda\f$ the output at
//  * row \f$i\f$, column \f$j\f$, and channel \f$k\f$ is
//  * \f{equation}{
//  *     y_{ij} =
//  *     \begin{cases}
//  *         \indicator{|x_{ij}| > \lambda} \, x_{ij} & \textrm{if} \,\, m_{ij} \ne 0 \\
//  *         |x_{ij}|                                 & \textrm{if} \,\, m_{ij} = 0
//  *     \end{cases}
//  * \f}
//  * where \f$i\f$ is the row, \f$j\f$ is the column and \f$k\f$ is the channel.
//  *
//  * @param input
//  * @param output
//  * @param threshold
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which input locations are thresholded.  The identity
//  *             function is applied to inputs at corresponding zero entries.
//  */
// void hard_threshold(
//     cv::InputArray input,
//     cv::OutputArray output,
//     cv::Scalar threshold,
//     cv::InputArray mask
// );

// /**
//  * @brief Inplace hard threshold
//  *
//  * This is equivalent to
//  * @code{cpp}
//  * hard_threshold(array, array, threshold);
//  * @endcode
//  *
//  * @see hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold)
//  *
//  * @param array
//  * @param threshold
//  */
// void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold);

// /**
//  * @brief Inplace masked hard threshold
//  *
//  * This is equivalent to
//  * @code{cpp}
//  * hard_threshold(array, array, threshold, mask);
//  * @endcode
//  *
//  * @see hard_threshold(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask)
//  *
//  * @param array
//  * @param threshold
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which input locations are thresholded.  The identity
//  *             function is applied to inputs at corresponding zero entries.
//  */
// void hard_threshold(cv::InputOutputArray array, cv::Scalar threshold, cv::InputArray mask);


//  ----------------------------------------------------------------------------
//  Shrink Coefficients
//  ----------------------------------------------------------------------------
// /**
//  * @brief Shrink detail coefficients using a global threshold.
//  *
//  * @param coeffs
//  * @param threshold
//  * @param threshold_function
//  * @param lower_level
//  * @param upper_level
//  */
// void shrink_details(
//     DWT2D::Coeffs& coeffs,
//     cv::Scalar threshold,
//     MaskedThresholdFunction threshold_function,
//     int lower_level=0,
//     int upper_level=-1
// );

// /**
//  * @brief Shrink detail coefficients using soft thresholding and a global threshold.
//  *
//  * This is a convenience function equivalent to
//  * @code{cpp}
//  * shrink_details(coeffs, threshold, soft_threshold, lower_level, upper_level);
//  * @endcode
//  *
//  * @param coeffs
//  * @param threshold
//  * @param lower_level
//  * @param upper_level
//  */
// void soft_shrink_details(
//     DWT2D::Coeffs& coeffs,
//     cv::Scalar threshold,
//     int lower_level=0,
//     int upper_level=-1
// );

// /**
//  * @brief Shrink detail coefficients using hard thresholding and a global threshold.
//  *
//  * This is a convenience function equivalent to
//  * @code{cpp}
//  * shrink_details(coeffs, threshold, hard_threshold, lower_level, upper_level);
//  * @endcode
//  *
//  * @param coeffs
//  * @param threshold
//  * @param lower_level
//  * @param upper_level
//  */
// void hard_shrink_details(
//     DWT2D::Coeffs& coeffs,
//     cv::Scalar threshold,
//     int lower_level=0,
//     int upper_level=-1
// );

// //  ----------------------------------------------------------------------------
// /**
//  * @brief Shrink detail coeffcients inplace using separate thresholds for each level.
//  *
//  * @param coeffs
//  * @param level_thresholds
//  * @param threshold_function
//  */
// void shrink_detail_levels(
//     DWT2D::Coeffs& coeffs,
//     cv::InputArray level_thresholds,
//     ThresholdFunction threshold_function
// );
// /**
//  * @brief Shrink detail coeffcients inplace using soft thresholding and separate thresholds for each level.
//  *
//  * This is a convenience function equivalent to
//  * @code{cpp}
//  * shrink_detail_levels(coeffs, thresholds, soft_threshold);
//  * @endcode
//  *
//  * @param coeffs
//  * @param thresholds
//  */
// void soft_shrink_detail_levels(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);

// /**
//  * @brief Shrink detail coeffcients inplace using hard thresholding and separate thresholds for each level.
//  *
//  * This is a convenience function equivalent to
//  * @code{cpp}
//  * shrink_detail_levels(coeffs, thresholds, hard_threshold);
//  * @endcode
//  *
//  * @param coeffs
//  * @param thresholds
//  */
// void hard_shrink_detail_levels(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);

// //  ----------------------------------------------------------------------------
// /**
//  * @brief Shrink detail coeffcients inplace using separate thresholds for each level and subband.
//  *
//  * @param coeffs
//  * @param subband_thresholds
//  * @param threshold_function
//  */
// void shrink_detail_subbands(
//     DWT2D::Coeffs& coeffs,
//     cv::InputArray subband_thresholds,
//     ThresholdFunction threshold_function
// );

// /**
//  * @brief Shrink detail coeffcients inplace using soft thresholding and separate thresholds for each level and subband.
//  *
//  * This is a convenience function equivalent to
//  * @code{cpp}
//  * shrink_detail_subbands(coeffs, thresholds, soft_threshold);
//  * @endcode
//  *
//  * @param coeffs
//  * @param thresholds
//  */
// void soft_shrink_detail_subbands(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);

// /**
//  * @brief Shrink detail coeffcients inplace using hard thresholding and separate thresholds for each level and subband.
//  *
//  * This is a convenience function equivalent to
//  * @code{cpp}
//  * shrink_detail_subbands(coeffs, thresholds, hard_threshold);
//  * @endcode
//  *
//  * @param coeffs
//  * @param thresholds
//  */
// void hard_shrink_detail_subbands(DWT2D::Coeffs& coeffs, cv::InputArray thresholds);


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


// /**
//  * @brief Computes the universal shrinkage threshold using a robust estimate of the standard deviation.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @return cv::Scalar
//  */
// cv::Scalar visu_shrink_threshold(const DWT2D::Coeffs& coeffs);

// /**
//  * @brief Computes the universal shrinkage threshold using a robust estimate of the standard deviation.
//  *
//  * @see universal_threshold()
//  *
//  * @param details The discrete wavelet transform detail coefficients.
//  * @return cv::Scalar
//  */
// cv::Scalar visu_shrink_threshold(cv::InputArray details);

// /**
//  * @brief Computes the universal shrinkage threshold using a robust estimate of the standard deviation.
//  *
//  * @see universal_threshold()
//  *
//  * @param details The discrete wavelet transform detail coefficients.
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which input locations are used in the computation.
//  * @return cv::Scalar
//  */
// cv::Scalar visu_shrink_threshold(cv::InputArray details, cv::InputArray mask);

// /**
//  * @brief Shrink detail coeffcients inplace using soft thresholding and the universal shrinkage threshold
//  *
//  * @see visu_shrink_threshold()
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  */
// void visu_soft_shrink(DWT2D::Coeffs& coeffs);
// /**
//  * @brief Shrink detail coeffcients inplace using hard thresholding and the universal shrinkage threshold
//  *
//  * @see visu_shrink_threshold()
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  */
// void visu_hard_shrink(DWT2D::Coeffs& coeffs);


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
enum SureShrinkOptimizer {
    NELDER_MEAD,
    BRUTE_FORCE,
};

enum ShrinkPartition {
    SHRINK_GLOBALLY = 0,
    SHRINK_LEVELS = 1,
    SHRINK_SUBBANDS = 2,
    SHRINK_CUSTOM = 3,
};


const nlopt::algorithm DEFAULT_SURE_SHRINK_NLOPT_ALGORITHM = nlopt::LN_NELDERMEAD;

// /**
//  * @brief Compute the SureShrink algorithm threshold.
//  *
//  * For a single channel with \f$\sigma = 1\f$, the SURE threshold is defined by
//  * \f{equation}{
//  *     \lambda_{SURE} = \argmin_{\lambda \ge 0} \SURE(\lambda, w)
//  * \f}
//  * where the input \f$w\f$ is a subset of noisy detail coefficients measurements
//  * drawn from an unknown normal distribution.
//  * \f$\SURE(\lambda, w)\f$ is an unbiased estimator of the MSE between the true
//  * detail coefficients and the soft thresholded coefficients using the
//  * threshold \f$\lambda\f$.
//  *
//  * Specifically,
//  * \f{equation}{
//  *     \SURE(\lambda, w) =
//  *         N
//  *         - 2 [\# \, \mathrm{of} \, |w_n| < \lambda]
//  *         + \sum_{n = 1}^{N} [\min(|w_n|, \lambda)]^2
//  * \f}
//  * where \f$N\f$ is the number of coefficients and
//  * \f$[\# \, \mathrm{of} \, |w_n| < \lambda]\f$ is the number of coefficients lesser
//  * than \f$\lambda\f$ in magnitude.
//  *
//  * When the detail coefficients are sparse it is preferable to employ hybrid
//  * algorithm that chooses the universal_threshold() when
//  * \f{equation}{
//  *     \frac{1}{N} \sum_{n = 1}^{N} \left( \frac{w_n}{\sigma} - 1 \right)^2
//  *     \le
//  *     \frac{\left(\log_2(N)\right)^{3/2}}{\sqrt{N}}
//  * \f}
//  * and \f$\lambda_{SURE}\f$ otherwise.
//  * Pass `HYBRID_SURE_SHRINK` as the `variant` argument to use
//  * this implementation.
//  *
//  * When the standard deviation of each input channel is not one, each channel
//  * standard deviation must be passed in using the `stdev` argument.
//  * In which case, the resulting threshold will be suitably scaled to work on the
//  * non-standardized coefficients.
//  *
//  * This function returns a multichannel threshold by applying the single channel
//  * algorithm to each channel of the given `detail_coeffs`.
//  *
//  * @see mad_stdev()
//  * @see universal_threshold()
//  *
//  * @param detail_coeffs The detail coefficients.
//  * @param stdev The standard deviations of the detail coefficients channels.
//  * @param variant The variant of the SureShrink algorithm.
//  * @param algorithm The optimization algorithm used to compute the SURE threshold.
//  * @return cv::Scalar
//  */
// cv::Scalar compute_sure_threshold(
//     cv::InputArray detail_coeffs,
//     const cv::Scalar& stdev = cv::Scalar::all(1.0),
//     SureShrinkVariant variant = NORMAL_SURE_SHRINK,
//     nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
// );

// /**
//  * @brief Compute the SureShrink algorithm threshold.
//  *
//  * This is a convienence function that is equivalent to
//  * @code{cpp}
//  * compute_sure_threshold(collect_masked(detail_coeffs, mask), stdev, variant, algorithm);
//  * @endcode
//  *
//  * @param detail_coeffs The detail coefficients.
//  * @param mask A single channel matrix of type CV_8U where nonzero entries
//  *             indicate which detail coefficients locations are used in the computation.
//  * @param stdev The standard deviations of the detail coefficients channels.
//  * @param variant The variant of the SureShrink algorithm.
//  * @param algorithm The optimization algorithm used to compute the SURE threshold.
//  * @return cv::Scalar
//  */
// cv::Scalar compute_sure_threshold(
//     cv::InputArray detail_coeffs,
//     cv::InputArray mask,
//     const cv::Scalar& stdev = cv::Scalar::all(1.0),
//     SureShrinkVariant variant = NORMAL_SURE_SHRINK,
//     nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
// );

// /**
//  * @brief Compute the SureShrink algorithm threshold for each level and subband.
//  *
//  * @see compute_sure_threshold()
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  * @param variant The variant of the SureShrink algorithm.
//  * @param algorithm The optimization algorithm used to compute the SURE threshold.
//  * @return cv::Mat4d
//  */
// cv::Mat4d sure_shrink_subband_thresholds(
//     const DWT2D::Coeffs& coeffs,
//     int levels,
//     SureShrinkVariant variant = NORMAL_SURE_SHRINK,
//     nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
// );

// /**
//  * @brief Compute the SureShrink algorithm threshold for each level.
//  *
//  * @see compute_sure_threshold()
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  * @param variant The variant of the SureShrink algorithm.
//  * @param algorithm The optimization algorithm used to compute the SURE threshold.
//  * @return cv::Mat4d
//  */
// cv::Mat4d sure_shrink_level_thresholds(
//     const DWT2D::Coeffs& coeffs,
//     int levels,
//     SureShrinkVariant variant = NORMAL_SURE_SHRINK,
//     nlopt::algorithm algorithm = nlopt::LN_NELDERMEAD
// );





// /**
//  * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  */
// void sure_shrink(DWT2D::Coeffs& coeffs);

// /**
//  * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  */
// void sure_shrink(DWT2D::Coeffs& coeffs, int levels);

// /**
//  * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  * @param algorithm The optimization algorithm used to compute the SURE threshold.
//  */
// void sure_shrink(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

// /**
//  * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  */
// void sure_shrink_levelwise(DWT2D::Coeffs& coeffs);

// /**
//  * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  */
// void sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels);

// /**
//  * @brief Shrink detail coeffcients inplace using the SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  * @param algorithm The optimization algorithm used to compute the SURE threshold.
//  */
// void sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

// /**
//  * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  */
// void hybrid_sure_shrink(DWT2D::Coeffs& coeffs);

// /**
//  * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  */
// void hybrid_sure_shrink(DWT2D::Coeffs& coeffs, int levels);

// /**
//  * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  * @param algorithm The optimization algorithm used to compute the SURE threshold.
//  */
// void hybrid_sure_shrink(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

// /**
//  * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  */
// void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs);

// /**
//  * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  */
// void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels);

// /**
//  * @brief Shrink detail coeffcients inplace using the Hybrid SureShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @param levels The maximum number of levels to shrink.  Shrinking is applied
//  *               starting at the lowest level (i.e. smallest scale).
//  * @param algorithm The optimization algorithm used to compute the SURE threshold.
//  */
// void hybrid_sure_shrink_levelwise(DWT2D::Coeffs& coeffs, int levels, nlopt::algorithm algorithm);

// //  ----------------------------------------------------------------------------
// //  Bayes Shrink
// //  ----------------------------------------------------------------------------
// /**
//  * @brief Shrink detail coeffcients inplace using BayesShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  * @return cv::Scalar
//  */
// cv::Scalar bayes_shrink_threshold(const DWT2D::Coeffs& coeffs);

// /**
//  * @brief Shrink detail coeffcients inplace using BayesShrink algorithm.
//  *
//  * @param coeffs The discrete wavelet transform coefficients.
//  */
// void bayes_shrink(DWT2D::Coeffs& coeffs);










using StdDevEstimator = cv::Scalar(cv::InputArray);
using MaskedThresholdFunction = void(cv::InputArray, cv::OutputArray, cv::Scalar, cv::InputArray);
template <typename Value, typename Threshold>
using PrimitiveThresholdFunction = Value(Value, Threshold);




namespace internal
{

template <typename T, int N, typename ThresholdFunctor>
struct Threshold
{
    using Pixel = cv::Vec<T, N>;

    ThresholdFunctor threshold_function;

    Threshold() : threshold_function() {}
    Threshold(ThresholdFunctor threshold_function) : threshold_function(threshold_function) {}

    void operator()(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold) const
    {
        assert(input.channels() == N);

        output.create(input.size(), input.type());
        auto result = output.getMat();
        input.getMat().forEach<Pixel>(
            [&](const auto& pixel, auto position) {
                auto& result_pixel = result.at<Pixel>(position);
                for (int i = 0; i < N; ++i)
                    result_pixel[i] = threshold_function(pixel[i], threshold[i]);
            }
        );
    }

    void operator()(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask) const
    {
        assert(input.channels() == N);

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
                        result_pixel[i] = threshold_function(pixel[i], threshold[i]);
                } else {
                    result.at<Pixel>(position) = pixel;
                }
            }
        );
    }
};


struct SoftThresholdFunctor
{
    template <typename T1, typename T2>
    constexpr auto operator()(T1 x, T2 threshold) const
    {
        auto abs_x = std::fabs(x);
        return (abs_x > threshold) * std::copysign(1.0, x) * (abs_x - threshold);
    }
};

template <typename T, int N>
using soft_threshold = Threshold<T, N, SoftThresholdFunctor>;

struct HardThresholdFunctor
{
    template <typename T1, typename T2>
    constexpr auto operator()(T1 x, T2 threshold) const
    {
        auto abs_x = std::fabs(x);
        return (abs_x > threshold) * x;
    }
};

template <typename T, int N>
using hard_threshold = Threshold<T, N, HardThresholdFunctor>;

template <typename T, typename W>
struct WrappedThresholdFunctor
{
    constexpr WrappedThresholdFunctor(
        std::function<PrimitiveThresholdFunction<T, W>> threshold_function
    ) :
        _threshold_function(threshold_function)
    {}

    constexpr T operator()(T value, W threshold) const
    {
        return _threshold_function(value, threshold);
    }

private:
    std::function<PrimitiveThresholdFunction<T, W>> _threshold_function;
};

template <typename T, int N, typename W>
struct WrappedThreshold : public Threshold<T, N, WrappedThresholdFunctor<T, W>>
{
    WrappedThreshold(
        std::function<PrimitiveThresholdFunction<T, W>> threshold_function
    ) :
        Threshold<T, N, WrappedThresholdFunctor<T, W>>(
            WrappedThresholdFunctor(threshold_function)
        )
    {}
};


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

    nlopt::algorithm algorithm(SureShrinkOptimizer optimizer) const
    {
        switch (optimizer) {
            case SureShrinkOptimizer::NELDER_MEAD:
                return nlopt::algorithm::LN_NELDERMEAD;
        }

        assert(false);
        return nlopt::algorithm::LN_NELDERMEAD;
    }

    void operator()(
        cv::InputArray input,
        const cv::Scalar& stdev,
        SureShrinkOptimizer optimizer,
        SureShrinkVariant variant,
        cv::Scalar univ_threshold,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);
        cv::Mat channels[N];
        cv::split(input.getMat(), channels);

        if (optimizer == SureShrinkOptimizer::BRUTE_FORCE) {
            for (int i = 0; i < N; ++i) {
                CV_LOG_INFO(NULL, "computing channel " << i);
                switch (variant) {
                case NORMAL_SURE_SHRINK:
                    result[i] = single_channel_normal_using_brute_force(
                        channels[i],
                        stdev[i]
                    );
                    break;
                case HYBRID_SURE_SHRINK:
                    result[i] = single_channel_hybrid_using_brute_force(
                        channels[i],
                        stdev[i],
                        univ_threshold[i]
                    );
                    break;
                }
            }
        } else {
            for (int i = 0; i < N; ++i) {
                CV_LOG_INFO(NULL, "computing channel " << i);
                switch (variant) {
                case NORMAL_SURE_SHRINK:
                    result[i] = single_channel_normal_using_nlopt(
                        channels[i],
                        stdev[i],
                        algorithm(optimizer)
                    );
                    break;
                case HYBRID_SURE_SHRINK:
                    result[i] = single_channel_hybrid_using_nlopt(
                        channels[i],
                        stdev[i],
                        univ_threshold[i],
                        algorithm(optimizer)
                    );
                    break;
                }
            }
        }
    }

    void operator()(
        cv::InputArray input,
        cv::InputArray mask,
        const cv::Scalar& stdev,
        SureShrinkOptimizer optimizer,
        SureShrinkVariant variant,
        cv::Scalar univ_threshold,
        cv::Scalar& result
    ) const
    {
        assert(input.channels() == N);
        cv::Mat masked_input;
        collect_masked<T, N>()(input, masked_input, mask);
        this->operator()(masked_input, stdev, optimizer, variant, univ_threshold, result);
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


//  ============================================================================
//  Low Level API
//  ============================================================================

/**
 * @brief Create a multichannel threshold function from a primitive functor.
 *
 * @code{cpp}
 * struct MyThresholdFunctor
 * {
 *     template <typename T, typename W>
 *     constexpr T operator(T value, W threshold);
 * };
 *
 * auto my_threshold_function = make_threshold_function<MyThresholdFunctor>();
 *
 * cv::Mat image = ...;
 * cv::Mat thresholded_image;
 * cv::Scalar threshold = ...;
 * my_threshold_function(image, thresholded_image, threshold);
 * @endcode
 *
 * @tparam ThresholdFunctor
 * @return std::function<MaskedThresholdFunction>
 */
template <typename ThresholdFunctor>
std::function<MaskedThresholdFunction> make_threshold_function()
{
    return [](
        cv::InputArray input,
        cv::OutputArray output,
        cv::Scalar threshold,
        cv::InputArray mask = cv::noArray()
    )
    {
        internal::dispatch_on_pixel_depth<internal::Threshold, ThresholdFunctor>(
            input.type(),
            input,
            output,
            threshold,
            mask
        );
    };
}

/**
 * @brief Create a multichannel threshold function from a primitive function.
 *
 * @code{cpp}
 *
 * template <typename T, typename W>
 * T my_primitive_threshold_function(T value, W threshold);
 *
 * auto my_threshold_function = make_threshold_function(my_primitive_threshold_function);
 *
 * cv::Mat image = ...;
 * cv::Mat thresholded_image;
 * cv::Scalar threshold = ...;
 * my_threshold_function(image, thresholded_image, threshold);
 * @endcode
 *
 * @tparam T The primitive value type.
 * @tparam W The primitive threshold type.
 * @param threshold_function The threshold function that acts on primitive types.
 * @return std::function<MaskedThresholdFunction>
 */
template <typename T, typename W>
std::function<MaskedThresholdFunction> make_threshold_function(
    std::function<PrimitiveThresholdFunction<T, W>> threshold_function
)
{
    return [&](
        cv::InputArray input,
        cv::OutputArray output,
        cv::Scalar threshold,
        cv::InputArray mask
    )
    {
        internal::dispatch_on_pixel_depth<internal::WrappedThreshold, W>(
            std::make_tuple(threshold_function),
            input.type(),
            input,
            output,
            threshold,
            mask
        );
    };
}


void shrink_globally(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    std::function<MaskedThresholdFunction> threshold_function,
    const cv::Range& levels = cv::Range::all()
)
{
    auto detail_mask = coeffs.detail_mask(levels);
    threshold_function(coeffs, coeffs, threshold, detail_mask);
}

void shrink_levels(
    DWT2D::Coeffs& coeffs,
    cv::InputArray level_thresholds,
    std::function<MaskedThresholdFunction> threshold_function,
    const cv::Range& levels = cv::Range::all()
)
{
    auto level_thresholds_matrix = level_thresholds.getMat();
    if (level_thresholds_matrix.rows > level_thresholds_matrix.cols)
        level_thresholds_matrix = level_thresholds_matrix.t();

    assert(level_thresholds.channels() == 4);
    assert(level_thresholds_matrix.rows == 1);

    level_thresholds_matrix.forEach<cv::Scalar>(
        [&](const auto& threshold, const auto position) {
            int level = std::max(levels.start, 0) + position[1];
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
    std::function<MaskedThresholdFunction> threshold_function,
    const cv::Range& levels = cv::Range::all()
)
{
    assert(subband_thresholds.channels() == 4);
    assert(levels == cv::Range::all() || levels.size() == subband_thresholds.rows());

    subband_thresholds.getMat().forEach<cv::Scalar>(
        [&](const auto& threshold, auto position) {
            int level = std::max(levels.start, 0) + position[0];
            int subband = position[1];
            auto subband_detail = coeffs.detail(level, subband);
            threshold_function(subband_detail, subband_detail, threshold, cv::noArray());
        }
    );
}


//  ============================================================================
//  High Level API
//  ============================================================================
/**
 * @brief
 * Subclasses must implement
 *
 * compute_thresholds()
 * Algorithms that apply a
 *  - single threshold globally should call compute_global_threshold().
 *  - separate threshold per level should call compute_level_thresholds().
 *  - separate threshold per subband should call compute_subband_thresholds().
 * Algorithms that use a custom partition (i.e. not global,
 * by level, or by subband) should return a cv::Mat of thresholds for each subset,
 * layed out in an implementation defined manner.
 * If the algorithm requires it and it wasn't passed as a parameter, the global
 * standard deviation should be computed in compute_thresholds() and passed to
 * compute_global_threshold(), compute_level_thresholds(),
 * or compute_subband_thresholds().
 * Pass cv::Scalar() as the standard deviation for algorithms that do not use
 * the global standard deviation.
 *
 * shrink_details()
 * Algorithms that apply a
 *  - single threshold globally should call shrink_globally().
 *  - separate threshold per level should call shrink_levels().
 *  - separate threshold per subband should call shrink_subbands().
 * Algorithms that use a custom partition (i.e. not global,
 * by level, or by subband) should call
 * `threshold_coeffs(detail_subset, detail_subset, subset_threshold)` on each subset.
 *
 *
 * compute_level_threshold() if compute_level_thresholds() is used
 * compute_subband_threshold() if compute_subband_threshold() is used
 */
class Shrink
{
public:
    Shrink() = delete;
    Shrink(const Shrink& other) = default;
    Shrink(Shrink&& other) = default;

    Shrink& operator=(const Shrink& other) = default;
    Shrink& operator=(Shrink&& other) = default;

    cv::Range levels() const { return _levels; }
    void set_levels(const cv::Range& levels) { _levels = levels; }
    void set_levels(int levels) { _levels = cv::Range(0, levels); }

    std::function<MaskedThresholdFunction> threshold_function() const { return _threshold_function; }
    void set_threshold_function(std::function<MaskedThresholdFunction> threshold_function) { _threshold_function = threshold_function; }

    std::function<StdDevEstimator> stdev_estimator() const { return _stdev_estimator; }
    void set_stdev_estimator(std::function<StdDevEstimator> stdev_estimator) { _stdev_estimator = stdev_estimator; }

    DWT2D::Coeffs operator()(const DWT2D::Coeffs& coeffs) const { return shrink(coeffs); }
    void operator()(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs) const { shrink(coeffs, shrunk_coeffs); }

    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& global_stdev = cv::Scalar()
    ) const
    {
        auto threshold = compute_thresholds(coeffs, global_stdev);
        auto shrunk_coeffs = coeffs.empty_clone();
        shrink(coeffs, shrunk_coeffs, global_stdev);

        return shrunk_coeffs;
    }

    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        const cv::Scalar& global_stdev = cv::Scalar()
    ) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        start(coeffs, global_stdev);
        if (&coeffs != &shrunk_coeffs)
            shrunk_coeffs = coeffs.clone();

        auto thresholds = compute_partition_thresholds(coeffs, global_stdev);

        switch (partition()) {
        case SHRINK_GLOBALLY:
            shrink_globally(shrunk_coeffs, thresholds.at<cv::Scalar>(0, 0));
            break;
        case SHRINK_LEVELS:
            shrink_levels(shrunk_coeffs, thresholds);
            break;
        case SHRINK_SUBBANDS:
            shrink_subbands(shrunk_coeffs, thresholds);
            break;
        case SHRINK_CUSTOM:
            shrink_custom(shrunk_coeffs, thresholds);
            break;
        }
        finish(coeffs, shrunk_coeffs, global_stdev, thresholds);
    }

    cv::Mat4d compute_thresholds(const DWT2D::Coeffs& coeffs) const
    {
        return compute_thresholds(coeffs, compute_global_stdev(coeffs));
    }

    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& global_stdev
    ) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        start(coeffs, global_stdev);
        auto thresholds = compute_partition_thresholds(coeffs, global_stdev);
        finish(coeffs, DWT2D::Coeffs(), global_stdev, thresholds);

        return thresholds;
    }

    virtual cv::Scalar compute_global_stdev(const DWT2D::Coeffs& coeffs) const
    {
        return cv::Scalar();
    }

    cv::Scalar compute_stdev(const DWT2D::Coeffs& coeffs) const
    {
        cv::Mat detail_coeffs;
        collect_masked(coeffs, detail_coeffs, coeffs.detail_mask());
        return compute_stdev(detail_coeffs);
    }

    cv::Scalar compute_stdev(cv::InputArray detail_coeffs) const
    {
        return _stdev_estimator(detail_coeffs);
    }

    ShrinkPartition partition() const { return _partition; }
    /**
     * @brief Returns true if a separate threshold is used for each level
     */
    bool is_by_level() const { return partition() == SHRINK_LEVELS; }
    /**
     * @brief Returns true if a separate threshold is used for each subband
     */
    bool is_by_subband() const { return partition() == SHRINK_SUBBANDS; }
    /**
     * @brief Returns true if a single threshold is used for all detail coefficients
     */
    bool is_global() const { return partition() == SHRINK_GLOBALLY; }

protected:
    /**
     * @brief Construct a new Shrink object
     *
     * This takes a multichannel threshold function that acts on cv::Mat.
     * E.g.
     * @code{cpp}
     * void my_threshold(cv::InputArray, cv::OutputArray, cv::InputArray threshold, cv::InputArray mask);
     * @endcode
     *
     * @param levels
     * @param threshold_function
     * @param stdev_estimator
     */
    Shrink(
        ShrinkPartition partition,
        const cv::Range& levels,
        std::function<MaskedThresholdFunction> threshold_function,
        std::function<StdDevEstimator> stdev_estimator
    ) :
        _partition(partition),
        _levels(levels),
        _threshold_function(threshold_function),
        _stdev_estimator(stdev_estimator)
    {
    }

    /**
     * @brief Construct a new Shrink object
     *
     * This takes a multichannel threshold function that acts on cv::Mat.
     * E.g.
     * @code{cpp}
     * void my_threshold(cv::InputArray, cv::OutputArray, cv::InputArray threshold, cv::InputArray mask);
     * @endcode
     *
     * @param levels
     * @param threshold_function
     */
    Shrink(
        ShrinkPartition partition,
        const cv::Range& levels,
        std::function<MaskedThresholdFunction> threshold_function
    ) :
        _partition(partition),
        _levels(levels),
        _threshold_function(threshold_function),
        _stdev_estimator(mad_stdev)
    {
    }

    /**
     * @brief Construct a new Shrink object
     *
     * This takes a threshold function that acts on primitive types.
     * E.g.
     * @code{cpp}
     * double my_threshold(double value, double threshold);
     * @endcode
     *
     * @tparam T
     * @tparam W
     * @param levels
     * @param threshold_function
     * @param stdev_estimator
     */
    template <typename T, typename W>
    Shrink(
        ShrinkPartition partition,
        const cv::Range& levels,
        std::function<PrimitiveThresholdFunction<T, W>> threshold_function,
        std::function<StdDevEstimator> stdev_estimator
    ) :
        Shrink(
            partition,
            levels,
            make_threshold_function(threshold_function),
            stdev_estimator
        )
    {
    }

    /**
     * @brief Construct a new Shrink object
     *
     * This takes a threshold function that acts on primitive types.
     * E.g.
     * @code{cpp}
     * double my_threshold(double value, double threshold);
     * @endcode
     *
     * @tparam T
     * @tparam W
     * @param levels
     * @param threshold_function
     */
    template <typename T, typename W>
    Shrink(
        ShrinkPartition partition,
        const cv::Range& levels,
        std::function<PrimitiveThresholdFunction<T, W>> threshold_function
    ) :
        Shrink(
            partition,
            levels,
            make_threshold_function(threshold_function),
            mad_stdev
        )
    {
    }

    /**
     * @brief Construct a new Shrink object
     *
     * This takes a threshold function that acts on primitive types.
     * E.g.
     * @code{cpp}
     * double my_threshold(double value, double threshold);
     * @endcode
     *
     * @tparam T
     * @tparam W
     * @param levels
     * @param threshold_function
     */
    template <typename T, typename W>
    Shrink(
        const cv::Range& levels,
        std::function<PrimitiveThresholdFunction<T, W>> threshold_function
    ) :
        Shrink(levels, threshold_function, mad_stdev)
    {
    }

    /**
     * @brief Prepare for a call to shrink() or compute_thresholds()
     *
     * Most shrinkage algorithms do **not** need to override this function.
     *
     * Subclasses should override this function if temporary calculations
     * must be made and stored as members prior that are used in `shrink_*()` or
     * `compute_*_threshold()`.
     *
     * One potential use case involves computing a set of partition masks that
     * are used by both compute_custom_thresholds() and shrink_custom().
     * Another use case is computing global statistics that are used by
     * algorithms that support multiple partitioning schemes (e.g. by over
     * overriding both compute_level_threshold() and compute_subband_threshold()).
     *
     * Since these are `const`, any data members set in this function must be
     * marked mutable and must be cleaned up in finish().
     *
     * @param coeffs
     * @param global_stdev
     */
    virtual void start(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& global_stdev
    ) const
    {
    }

    /**
     * @brief Finish a call to shrink() or compute_thresholds()
     *
     * @param coeffs
     * @param shrunk_coeffs
     * @param global_stdev
     * @param thresholds
     */
    virtual void finish(
        const DWT2D::Coeffs& coeffs,
        const DWT2D::Coeffs& shrunk_coeffs,
        const cv::Scalar& global_stdev,
        const cv::Mat4d& thresholds
    ) const
    {
    }

    cv::Mat4d compute_partition_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& global_stdev
    ) const
    {
        switch (partition()) {
        case SHRINK_GLOBALLY:
            return cv::Mat4d(
                cv::Size(1, 1),
                compute_global_threshold(coeffs, global_stdev)
            );
        case SHRINK_LEVELS:
            return compute_level_thresholds(coeffs, global_stdev);
        case SHRINK_SUBBANDS:
            return compute_subband_thresholds(coeffs, global_stdev);
        case SHRINK_CUSTOM:
            return compute_custom_thresholds(coeffs, global_stdev);
        }
    }

    virtual void shrink_custom(DWT2D::Coeffs& coeffs, const cv::Mat4d& thresholds) const
    {
        assert(false);
    }

    virtual cv::Scalar compute_global_threshold(
        const DWT2D::Coeffs coeffs,
        const cv::Scalar& global_stdev
    ) const
    {
        assert(false);
        return cv::Scalar();
    }

    virtual cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& global_stdev
    ) const
    {
        assert(false);
        return cv::Scalar();
    }

    virtual cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& global_stdev
    ) const
    {
        assert(false);
        return cv::Scalar();
    }

    virtual cv::Mat4d compute_custom_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& global_stdev
    ) const
    {
        assert(false);
    }

    void threshold_coeffs(
        cv::InputArray coeffs,
        cv::OutputArray thresholded_coeffs,
        const cv::Scalar& threshold,
        cv::InputArray detail_mask = cv::noArray()
    ) const
    {
        _threshold_function(coeffs, thresholded_coeffs, threshold, detail_mask);
    }

    cv::Mat4d compute_level_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& global_stdev
    ) const
    {
        cv::Mat4d thresholds(levels().size(), 1);
        for (int level = levels().start; level < levels().end; ++level) {
            cv::Mat detail_coeffs;
            collect_masked(coeffs, detail_coeffs, coeffs.detail_mask(level));
            auto threshold = compute_level_threshold(detail_coeffs, level, global_stdev);
            thresholds(level - levels().start) = threshold;
        }

        return thresholds;
    }

    cv::Mat4d compute_subband_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& global_stdev
    ) const
    {
        cv::Mat4d thresholds(levels().size(), 3);
        for (int level = levels().start; level < levels().end; ++level) {
            for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL}) {
                auto detail_coeffs = coeffs.detail(level, subband);
                auto threshold = compute_subband_threshold(detail_coeffs, level, subband, global_stdev);
                thresholds(level - levels().start, subband) = threshold;
            }
        }

        return thresholds;
    }

    void shrink_subbands(
        DWT2D::Coeffs& coeffs,
        cv::InputArray subband_thresholds
    ) const
    {
        cvwt::shrink_subbands(
            coeffs,
            subband_thresholds,
            threshold_function(),
            levels()
        );
        // assert(subband_thresholds.channels() == 4);

        // subband_thresholds.getMat().forEach<cv::Scalar>(
        //     [&](const auto& threshold, auto position) {
        //         int level = levels().start + position[0];
        //         int subband = position[1];
        //         auto subband_detail = coeffs.detail(level, subband);
        //         threshold_coeffs(subband_detail, subband_detail, threshold);
        //     }
        // );
    }

    void shrink_globally(DWT2D::Coeffs& coeffs, cv::InputArray threshold) const
    {
        cvwt::shrink_globally(
            coeffs,
            threshold.getMat().at<cv::Scalar>(0, 0),
            threshold_function(),
            levels()
        );
        // auto detail_mask = coeffs.detail_mask(levels());
        // _threshold_function(coeffs, coeffs, threshold.getMat().at<cv::Scalar>(0, 0), detail_mask);
    }

    void shrink_levels(
        DWT2D::Coeffs& coeffs,
        cv::InputArray level_thresholds
    ) const
    {
        auto level_thresholds_matrix = level_thresholds.getMat();
        if (level_thresholds_matrix.rows > level_thresholds_matrix.cols)
            level_thresholds_matrix = level_thresholds_matrix.t();

        assert(level_thresholds.channels() == 4);
        assert(level_thresholds_matrix.rows == 1);

        cvwt::shrink_levels(
            coeffs,
            level_thresholds_matrix,
            threshold_function(),
            levels()
        );

        // level_thresholds_matrix.forEach<cv::Scalar>(
        //     [&](const auto& threshold, const auto position) {
        //         int level = levels().start + position[1];
        //         for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL}) {
        //             auto subband_detail = coeffs.detail(level, subband);
        //             threshold_coeffs(subband_detail, subband_detail, threshold);
        //         }
        //     }
        // );
    }

private:
    ShrinkPartition _partition;
    cv::Range _levels;
    std::function<MaskedThresholdFunction> _threshold_function;
    std::function<StdDevEstimator> _stdev_estimator;
    mutable std::mutex _mutex;
};



class SureShrink : public Shrink
{
public:
    //  all levels
    SureShrink() :
        SureShrink(
            cv::Range::all(),
            SureShrinkVariant::NORMAL_SURE_SHRINK,
            ShrinkPartition::SHRINK_SUBBANDS,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(SureShrinkVariant variant) :
        SureShrink(
            cv::Range::all(),
            variant,
            ShrinkPartition::SHRINK_SUBBANDS,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(ShrinkPartition partition) :
        SureShrink(
            cv::Range::all(),
            SureShrinkVariant::NORMAL_SURE_SHRINK,
            partition,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(SureShrinkVariant variant, ShrinkPartition partition) :
        SureShrink(
            cv::Range::all(),
            variant,
            partition,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    //  range levels
    SureShrink(const cv::Range& levels) :
        SureShrink(
            levels,
            SureShrinkVariant::NORMAL_SURE_SHRINK,
            ShrinkPartition::SHRINK_SUBBANDS,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(const cv::Range& levels, SureShrinkVariant variant) :
        SureShrink(
            levels,
            variant,
            ShrinkPartition::SHRINK_SUBBANDS,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(const cv::Range& levels, ShrinkPartition partition) :
        SureShrink(
            levels,
            SureShrinkVariant::NORMAL_SURE_SHRINK,
            partition,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(const cv::Range& levels, SureShrinkVariant variant, ShrinkPartition partition) :
        SureShrink(
            levels,
            variant,
            partition,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    //  int levels
    SureShrink(int levels) :
        SureShrink(
            cv::Range(0, levels),
            SureShrinkVariant::NORMAL_SURE_SHRINK,
            ShrinkPartition::SHRINK_SUBBANDS,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(int levels, SureShrinkVariant variant) :
        SureShrink(
            cv::Range(0, levels),
            variant,
            ShrinkPartition::SHRINK_SUBBANDS,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(int levels, ShrinkPartition partition) :
        SureShrink(
            cv::Range(0, levels),
            SureShrinkVariant::NORMAL_SURE_SHRINK,
            partition,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    SureShrink(int levels, SureShrinkVariant variant, ShrinkPartition partition) :
        SureShrink(
            cv::Range(0, levels),
            variant,
            partition,
            SureShrinkOptimizer::NELDER_MEAD
        )
    {
    }

    //  everything, range levels
    SureShrink(
        const cv::Range& levels,
        SureShrinkVariant variant,
        ShrinkPartition partition,
        SureShrinkOptimizer optimizer
    ) :
        Shrink(
            partition,
            levels,
            soft_threshold,
            mad_stdev
        ),
        _variant(variant),
        _optimizer(optimizer)
    {
    }

    //  everything, int levels
    SureShrink(
        int levels,
        SureShrinkVariant variant,
        ShrinkPartition partition,
        SureShrinkOptimizer optimizer
    ) :
        SureShrink(
            cv::Range(0, levels),
            variant,
            partition,
            optimizer
        )
    {
    }

    SureShrinkVariant variant() const { return _variant; }
    SureShrinkOptimizer optimizer() const { return _optimizer; }

    cv::Scalar compute_sure_threshold(
        cv::InputArray detail_coeffs,
        const cv::Scalar& global_stdev = cv::Scalar::all(1.0)
    ) const
    {
        if (variant() == SureShrinkVariant::HYBRID_SURE_SHRINK && global_stdev == cv::Scalar())
            internal::throw_bad_arg();

        cv::Scalar univ_threshold;
        if (variant() == HYBRID_SURE_SHRINK)
            univ_threshold = universal_threshold(detail_coeffs, global_stdev);

        cv::Scalar result;
        internal::dispatch_on_pixel_type<internal::compute_sure_threshold>(
            detail_coeffs.type(),
            detail_coeffs,
            global_stdev,
            optimizer(),
            variant(),
            univ_threshold,
            result
        );

        return result;
    }

    cv::Scalar compute_sure_threshold(
        cv::InputArray detail_coeffs,
        cv::InputArray mask,
        const cv::Scalar& global_stdev = cv::Scalar::all(1.0)
    ) const
    {
        if (variant() == SureShrinkVariant::HYBRID_SURE_SHRINK && global_stdev == cv::Scalar())
            internal::throw_bad_arg();

        cv::Scalar univ_threshold;
        if (variant() == HYBRID_SURE_SHRINK)
            univ_threshold = universal_threshold(detail_coeffs, global_stdev);

        cv::Scalar result;
        internal::dispatch_on_pixel_type<internal::compute_sure_threshold>(
            detail_coeffs.type(),
            detail_coeffs,
            mask,
            global_stdev,
            optimizer(),
            variant(),
            univ_threshold,
            result
        );

        return result;
    }
protected:
    cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& global_stdev
    ) const override
    {
        return compute_sure_threshold(detail_coeffs, global_stdev);
    }

    cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& global_stdev
    ) const override
    {
        return compute_sure_threshold(detail_coeffs, global_stdev);
    }
private:
    SureShrinkVariant _variant;
    SureShrinkOptimizer _optimizer;
};






class UniverisalShrink : public Shrink
{
public:
    UniverisalShrink(
        std::function<MaskedThresholdFunction> threshold
    ) :
        UniverisalShrink(cv::Range::all(), threshold, ShrinkPartition::SHRINK_GLOBALLY)
    {
    }

    UniverisalShrink(
        std::function<MaskedThresholdFunction> threshold,
        ShrinkPartition partition
    ) :
        UniverisalShrink(cv::Range::all(), threshold, partition)
    {
    }

    UniverisalShrink(
        int levels,
        std::function<MaskedThresholdFunction> threshold
    ) :
        UniverisalShrink(cv::Range(0, levels), threshold, ShrinkPartition::SHRINK_GLOBALLY)
    {
    }

    UniverisalShrink(
        int levels,
        std::function<MaskedThresholdFunction> threshold,
        ShrinkPartition partition
    ) :
        UniverisalShrink(cv::Range(0, levels), threshold, partition)
    {
    }

    UniverisalShrink(
        const cv::Range& levels,
        std::function<MaskedThresholdFunction> threshold
    ) :
        UniverisalShrink(levels, threshold, ShrinkPartition::SHRINK_GLOBALLY)
    {
    }

    UniverisalShrink(
        const cv::Range& levels,
        std::function<MaskedThresholdFunction> threshold,
        ShrinkPartition partition
    ) :
        Shrink(partition, levels, threshold)
    {
    }


    UniverisalShrink(
        std::function<MaskedThresholdFunction> threshold,
        std::function<StdDevEstimator> stdev_estimator
    ) :
        UniverisalShrink(
            cv::Range::all(),
            threshold,
            ShrinkPartition::SHRINK_GLOBALLY,
            stdev_estimator
        )
    {
    }

    UniverisalShrink(
        std::function<MaskedThresholdFunction> threshold,
        ShrinkPartition partition,
        std::function<StdDevEstimator> stdev_estimator
    ) :
        UniverisalShrink(
            cv::Range::all(),
            threshold,
            partition,
            stdev_estimator
        )
    {
    }

    UniverisalShrink(
        int levels,
        std::function<MaskedThresholdFunction> threshold,
        std::function<StdDevEstimator> stdev_estimator
    ) :
        UniverisalShrink(
            cv::Range(0, levels),
            threshold,
            ShrinkPartition::SHRINK_GLOBALLY,
            stdev_estimator
        )
    {
    }

    UniverisalShrink(
        int levels,
        std::function<MaskedThresholdFunction> threshold,
        ShrinkPartition partition,
        std::function<StdDevEstimator> stdev_estimator
    ) :
        UniverisalShrink(
            cv::Range(0, levels),
            threshold,
            partition,
            stdev_estimator
        )
    {
    }

    UniverisalShrink(
        const cv::Range& levels,
        std::function<MaskedThresholdFunction> threshold,
        std::function<StdDevEstimator> stdev_estimator
    ) :
        UniverisalShrink(
            levels,
            threshold,
            ShrinkPartition::SHRINK_GLOBALLY,
            stdev_estimator
        )
    {
    }

    UniverisalShrink(
        const cv::Range& levels,
        std::function<MaskedThresholdFunction> threshold,
        ShrinkPartition partition,
        std::function<StdDevEstimator> stdev_estimator
    ) :
        Shrink(
            partition,
            levels,
            threshold,
            stdev_estimator
        )
    {
    }

    cv::Scalar compute_global_stdev(const DWT2D::Coeffs& coeffs) const override
    {
        return compute_stdev(coeffs);
    }
protected:
    virtual cv::Scalar compute_global_threshold(
        const DWT2D::Coeffs coeffs,
        const cv::Scalar& global_stdev
    ) const override
    {
        return universal_threshold(coeffs, global_stdev);
    }

    cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& global_stdev
    ) const override
    {
        return universal_threshold(detail_coeffs, global_stdev);
    }

    cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& global_stdev
    ) const override
    {
        return universal_threshold(detail_coeffs, global_stdev);
    }
};



class VisuShrink : public UniverisalShrink
{
public:
    VisuShrink() :
        VisuShrink(cv::Range::all(), ShrinkPartition::SHRINK_GLOBALLY)
    {
    }

    VisuShrink(ShrinkPartition partition) :
        VisuShrink(cv::Range::all(), partition)
    {
    }

    VisuShrink(int levels) :
        VisuShrink(cv::Range(0, levels), ShrinkPartition::SHRINK_GLOBALLY)
    {
    }

    VisuShrink(int levels, ShrinkPartition partition) :
        VisuShrink(cv::Range(0, levels), partition)
    {
    }

    VisuShrink(const cv::Range& levels) :
        VisuShrink(levels, ShrinkPartition::SHRINK_GLOBALLY)
    {
    }

    VisuShrink(const cv::Range& levels, ShrinkPartition partition) :
        UniverisalShrink(levels, soft_threshold, partition)
    {
    }
};


class BayesShrink : public Shrink
{
};

} // namespace cvwt

#endif  // CVWT_SHRINKAGE_HPP

