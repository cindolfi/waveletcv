#ifndef CVWT_STATISTICS_HPP
#define CVWT_STATISTICS_HPP

#include <opencv2/core.hpp>

namespace cvwt
{
/**
 * @name Statistics
 * @{
 */
/**
 * @brief Computes the channel-wise median.
 *
 * @param[in] array
 * @param[in] mask A single channel matrix of type CV_8UC1 or CV_8SC1 where
 *                 nonzero entries indicate which array elements are used.
 */
cv::Scalar median(cv::InputArray array, cv::InputArray mask = cv::noArray());

/**
 * @brief Masked mean absolute deviation.
 *
 * The standard deviation of the \f$k^{th}\f$ channel of \f$x\f$ is estimated by
 * \f{equation}{
 *     \mad(x_k) = \median(|x_k| - \median(x_k))
 * \f}
 * where the median is taken over locations where the mask is nonzero.
 *
 * @param[in] array The multichannel array.
 * @param[in] mask A single channel matrix of type CV_8UC1 or CV_8SC1 where nonzero
 *             entries indicate which array elements are used.
 */
cv::Scalar mad(cv::InputArray array, cv::InputArray mask = cv::noArray());

/**
 * @brief Robust estimation of the standard deviation.
 *
 * This function estimates the standard deviation of each channel separately
 * using the mean absolute deviation (i.e. mad()).  The elements in a each
 * channel are assumed to be i.i.d from a normal distribution.
 *
 * Specifically, the standard deviation of the \f$k^{th}\f$ channel of \f$x\f$
 * is estimated by
 * \f{equation}{
 *     \hat{\sigma_k} = \frac{\mad(x_k)}{0.675}
 * \f}
 *
 * @param[in] array The data samples.  This can a multichannel with up to 4 channels.
 */
cv::Scalar mad_stdev(cv::InputArray array);

/**
 * @brief Returns the maximum absolute value over all channels.
 *
 * @param[in] array
 * @param[in] mask
 */
double maximum_abs_value(cv::InputArray array, cv::InputArray mask = cv::noArray());
/** @}*/

}   // namespace cvwt

#endif  // #define CVWT_STATISTICS_HPP


