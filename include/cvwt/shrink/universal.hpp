#ifndef CVWT_SHRINK_UNIVERSAL_HPP
#define CVWT_SHRINK_UNIVERSAL_HPP

#include <opencv2/core.hpp>
#include "cvwt/shrink/shrink.hpp"
#include "cvwt/dwt2d.hpp"

namespace cvwt
{
/** @addtogroup shrinkage
 *  @{
 */
//  ----------------------------------------------------------------------------
//  Universal / VisuShrink
//  ----------------------------------------------------------------------------
/**
 * @brief %Shrink using the universal threshold.
 */
class UniversalShrink : public Shrink
{
public:
    UniversalShrink(
        ShrinkFunction threshold_function
    ) :
        UniversalShrink(
            Shrink::GLOBALLY,
            threshold_function
        )
    {}

    template <typename T, typename W>
    UniversalShrink(
        PrimitiveShrinkFunction<T, W> threshold_function
    ) :
        UniversalShrink(
            make_shrink_function(threshold_function)
        )
    {}

    UniversalShrink(
        Shrink::Partition partition,
        ShrinkFunction threshold_function
    ) :
        Shrink(
            partition,
            threshold_function
        )
    {}

    template <typename T, typename W>
    UniversalShrink(
        Shrink::Partition partition,
        PrimitiveShrinkFunction<T, W> threshold_function
    ) :
        UniversalShrink(
            partition,
            make_shrink_function(threshold_function)
        )
    {}

    UniversalShrink(
        ShrinkFunction threshold_function,
        StdDevFunction stdev_function
    ) :
        UniversalShrink(
            Shrink::GLOBALLY,
            threshold_function,
            stdev_function
        )
    {}

    template <typename T, typename W>
    UniversalShrink(
        PrimitiveShrinkFunction<T, W> threshold_function,
        StdDevFunction stdev_function
    ) :
        UniversalShrink(
            make_shrink_function(threshold_function),
            stdev_function
        )
    {}

    UniversalShrink(
        Shrink::Partition partition,
        ShrinkFunction threshold_function,
        StdDevFunction stdev_function
    ) :
        Shrink(
            partition,
            threshold_function,
            stdev_function
        )
    {}

    template <typename T, typename W>
    UniversalShrink(
        Shrink::Partition partition,
        PrimitiveShrinkFunction<T, W> threshold_function,
        StdDevFunction stdev_function
    ) :
        UniversalShrink(
            partition,
            make_shrink_function(threshold_function),
            stdev_function
        )
    {}

    UniversalShrink(const UniversalShrink& other) = default;
    UniversalShrink(UniversalShrink&& other) = default;

    /**
     * @brief Computes the universal shrinkage threshold.
     *
     * The universal multichannel threshold \f$\lambda\f$ is defined by
     * \f{equation}{
     *     \lambda_k = \sigma_k \, \sqrt{2 \log{N}}
     * \f}
     *
     * @see https://computing.llnl.gov/sites/default/files/jei2001.pdf

    * @param num_elements The number of detail coefficients.
    * @param stdev The standard deviations of the detail coefficients channels.
    */
    static cv::Scalar compute_universal_threshold(
        int num_elements,
        const cv::Scalar& stdev = cv::Scalar::all(1.0)
    );

    /**
     * @brief Computes the universal shrinkage threshold.
     *
     * This is an convenience function equivalent to
     * @code{cpp}
     * compute_universal_threshold(coeffs, coeffs.detail_mask(), stdev);
     * @endcode
     *
     * @see https://computing.llnl.gov/sites/default/files/jei2001.pdf
     *
     * @param coeffs The discrete wavelet transform coefficients.
     * @param stdev The standard deviations of the detail coefficients channels.
     */
    static cv::Scalar compute_universal_threshold(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& stdev = cv::Scalar::all(1.0)
    );

    /**
     * @brief Computes the universal shrinkage threshold.
     *
     * This is an convenience function equivalent to
     * @code{cpp}
     * compute_universal_threshold(details.total(), stdev);
     * @endcode
     *
     * @param details The discrete wavelet transform detail coefficients.
     * @param stdev The standard deviations of the detail coefficients channels.
     */
    static cv::Scalar compute_universal_threshold(
        cv::InputArray details,
        const cv::Scalar& stdev = cv::Scalar::all(1.0)
    );

    /**
     * @brief Computes the universal shrinkage threshold.
     *
     * This is an convenience function equivalent to
     * @code{cpp}
     * compute_universal_threshold(collect_masked(details, mask), stdev);
     * @endcode
     *
     * @param details The discrete wavelet transform detail coefficients.
     * @param mask A single channel matrix of type CV_8U where nonzero entries
     *             indicate which input locations are used in the computation.
     * @param stdev The standard deviations of the detail coefficients channels.
     */
    static cv::Scalar compute_universal_threshold(
        cv::InputArray details,
        cv::InputArray mask,
        const cv::Scalar& stdev = cv::Scalar::all(1.0)
    );

protected:
    cv::Scalar compute_global_threshold(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_universal_threshold(coeffs, coeffs.detail_mask(levels), stdev);
    }

    cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_universal_threshold(detail_coeffs, stdev);
    }

    cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_universal_threshold(detail_coeffs, stdev);
    }
};

/**
 * @brief Gobal shrinkage using the universal threshold, the soft threshold
 *        function, and the MAD standard deviation estimator.
 *
 */
class VisuShrink : public UniversalShrink
{
public:
    VisuShrink() :
        VisuShrink(Shrink::GLOBALLY)
    {}

    VisuShrink(Shrink::Partition partition) :
        UniversalShrink(
            partition,
            soft_threshold
        )
    {}

    VisuShrink(const VisuShrink& other) = default;
    VisuShrink(VisuShrink&& other) = default;
};

//  ----------------------------------------------------------------------------
//  VisuShrink Functional API
//  ----------------------------------------------------------------------------
/** @name VisuShrink Functional API
 *  @{
 */
/**
 * @brief %Shrink detail coeffcients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs visu_shrink(const DWT2D::Coeffs& coeffs);

/**
 * @brief %Shrink detail coeffcients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void visu_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief %Shrink detail coeffcients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs visu_shrink(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief %Shrink detail coeffcients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
void visu_shrink(DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels);
/** @}*/

/** @} shrinkage */
} // namespace cvwt

#endif  // CVWT_SHRINK_UNIVERSAL_HPP

