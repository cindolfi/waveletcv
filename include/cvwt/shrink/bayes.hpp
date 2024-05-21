#ifndef CVWT_SHRINK_BAYES_HPP
#define CVWT_SHRINK_BAYES_HPP

#include <opencv2/core.hpp>
#include "cvwt/shrink/shrink.hpp"
#include "cvwt/dwt2d.hpp"

namespace cvwt
{
/** @addtogroup shrinkage
 *  @{
 */
/**
 * @brief Implements the BayesShrink algorithm for shrinking DWT coefficients.
 * @headerfile cvwt/shrinkage.hpp
 *
 * The threshold for subband \f$s\f$ is
 * \f{equation}{
 *     \lambda_s = \frac{\hat\sigma^2}{\hat\sigma^2_X}
 * \f}
 * where \f$\hat\sigma^2\f$ is the estimated noise variance and
 * \f$\hat\sigma_X\f$ is the estimated signal variance on subband \f$s\f$.
 *
 * The estimated signal variance is
 * \f{equation}{
 *     \hat\sigma^2_X = \max(\hat\sigma^2_Y - \hat\sigma^2, 0)
 * \f}
 * where
 * \f{equation}{
 *     \hat\sigma^2_Y = \frac{1}{N_s} \sum_{n = 1}^{N_s} w^2_n
 * \f}
 * is the estimate of the variance of the observations.
 *
 * When \f$\hat\sigma^2 \gt \hat\sigma_Y^2\f$, the threshold becomes
 * \f$\lambda_s = \max(|w_n|)\f$ and all subband coefficients are shrunk to zero.
 */
class BayesShrink : public Shrink
{
public:
    BayesShrink() :
        BayesShrink(Shrink::SUBBANDS)
    {}

    BayesShrink(Shrink::Partition partition) :
        BayesShrink(partition, soft_threshold)
    {}

    template <typename T, typename W>
    BayesShrink(
        Shrink::Partition partition,
        PrimitiveShrinkFunction<T, W> threshold_function
    ) :
        BayesShrink(partition, make_shrink_function(threshold_function))
    {}

    BayesShrink(
        Shrink::Partition partition,
        ShrinkFunction threshold_function
    ) :
        Shrink(
            partition,
            threshold_function,
            mad_stdev
        )
    {}

    BayesShrink(const BayesShrink& other) = default;
    BayesShrink(BayesShrink&& other) = default;

    cv::Scalar compute_bayes_threshold(
        cv::InputArray detail_coeffs,
        const cv::Scalar& stdev
    ) const;

    cv::Scalar compute_bayes_threshold(
        cv::InputArray detail_coeffs,
        cv::InputArray mask,
        const cv::Scalar& stdev
    ) const;

protected:
    cv::Scalar compute_global_threshold(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_bayes_threshold(coeffs, coeffs.detail_mask(levels), stdev);
    }

    cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_bayes_threshold(detail_coeffs, stdev);
    }

    cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_bayes_threshold(detail_coeffs, stdev);
    }
};
//  ----------------------------------------------------------------------------
//  BayesShrink Functional API
//  ----------------------------------------------------------------------------
/** @name BayesShrink Functional API
 *  @{
 */

/**
 * @headerfile cvwt/shrink/bayes.hpp <cvwt/shrinkage.hpp>
 * @brief %Shrink DWT coefficients using the BayesShrink algorithm
 *
 * @param[in] coeffs The DWT coefficients.
 */
DWT2D::Coeffs bayes_shrink(const DWT2D::Coeffs& coeffs);

/**
 * @brief %Shrink DWT coefficients using the BayesShrink algorithm
 * @headerfile cvwt/shrinkage.hpp
 *
 * @param[in] coeffs The DWT coefficients.
 * @param[out] shrunk_coeffs The shrunken DWT coefficients.
 */
void bayes_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);
/** @}*/

/** @} shrinkage */
} // namespace cvwt

#endif  // CVWT_SHRINK_BAYES_HPP

