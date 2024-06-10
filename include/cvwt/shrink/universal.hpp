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
 * @brief Shrink using the universal threshold.
 */
class UniversalShrinker : public Shrinker
{
public:
    /**
     * @brief Construct a new Universal Shrinker object.
     *
     * @param[in] partition
     * @param[in] shrink_function
     * @param[in] stdev_function
     */
    UniversalShrinker(
        Shrinker::Partition partition,
        ShrinkFunction shrink_function,
        StdDevFunction stdev_function
    ) :
        Shrinker(
            partition,
            shrink_function,
            stdev_function
        )
    {}

    /**
     * @overload
     *
     * @param[in] shrink_function
     */
    UniversalShrinker(
        ShrinkFunction shrink_function
    ) :
        UniversalShrinker(
            Shrinker::GLOBALLY,
            shrink_function
        )
    {}

    /**
     * @overload
     *
     * @tparam T
     * @tparam W
     * @param[in] shrink_function
     */
    template <typename T, typename W>
    UniversalShrinker(
        PrimitiveShrinkFunction<T, W> shrink_function
    ) :
        UniversalShrinker(
            make_shrink_function(shrink_function)
        )
    {}

    /**
     * @overload
     *
     * @param[in] partition
     * @param[in] shrink_function
     */
    UniversalShrinker(
        Shrinker::Partition partition,
        ShrinkFunction shrink_function
    ) :
        Shrinker(
            partition,
            shrink_function
        )
    {}

    /**
     * @overload
     *
     * @tparam T
     * @tparam W
     * @param[in] partition
     * @param[in] shrink_function
     */
    template <typename T, typename W>
    UniversalShrinker(
        Shrinker::Partition partition,
        PrimitiveShrinkFunction<T, W> shrink_function
    ) :
        UniversalShrinker(
            partition,
            make_shrink_function(shrink_function)
        )
    {}

    /**
     * @overload
     *
     * @param[in] shrink_function
     * @param[in] stdev_function
     */
    UniversalShrinker(
        ShrinkFunction shrink_function,
        StdDevFunction stdev_function
    ) :
        UniversalShrinker(
            Shrinker::GLOBALLY,
            shrink_function,
            stdev_function
        )
    {}

    /**
     * @overload
     *
     * @tparam T
     * @tparam W
     * @param[in] shrink_function
     * @param[in] stdev_function
     */
    template <typename T, typename W>
    UniversalShrinker(
        PrimitiveShrinkFunction<T, W> shrink_function,
        StdDevFunction stdev_function
    ) :
        UniversalShrinker(
            make_shrink_function(shrink_function),
            stdev_function
        )
    {}

    /**
     * @overload
     *
     * @tparam T
     * @tparam W
     * @param[in] partition
     * @param[in] shrink_function
     * @param[in] stdev_function
     */
    template <typename T, typename W>
    UniversalShrinker(
        Shrinker::Partition partition,
        PrimitiveShrinkFunction<T, W> shrink_function,
        StdDevFunction stdev_function
    ) :
        UniversalShrinker(
            partition,
            make_shrink_function(shrink_function),
            stdev_function
        )
    {}

    /**
     * @brief Copy Constructor.
     */
    UniversalShrinker(const UniversalShrinker& other) = default;
    /**
     * @brief Move Constructor.
     */
    UniversalShrinker(UniversalShrinker&& other) = default;

    /**
     * @brief Computes the universal shrinkage threshold.
     *
     * The universal multichannel threshold \f$\lambda\f$ is defined by
     * \f{equation}{
     *     \lambda_k = \sigma_k \, \sqrt{2 \log{N}}
     * \f}
     *
     * @see https://computing.llnl.gov/sites/default/files/jei2001.pdf

    * @param[in] num_elements The number of detail coefficients.
    * @param[in] stdev The standard deviations of the detail coefficients channels.
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
     * @param[in] coeffs The discrete wavelet transform coefficients.
     * @param[in] stdev The standard deviations of the detail coefficients channels.
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
     * @param[in] detail_coeffs The discrete wavelet transform detail coefficients.
     * @param[in] stdev The standard deviations of the detail coefficients channels.
     */
    static cv::Scalar compute_universal_threshold(
        cv::InputArray detail_coeffs,
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
     * @param[in] detail_coeffs The discrete wavelet transform detail coefficients.
     * @param[in] mask A single channel matrix of type CV_8U where nonzero entries
     *             indicate which input locations are used in the computation.
     * @param[in] stdev The standard deviations of the detail coefficients channels.
     */
    static cv::Scalar compute_universal_threshold(
        cv::InputArray detail_coeffs,
        cv::InputArray mask,
        const cv::Scalar& stdev = cv::Scalar::all(1.0)
    );

protected:
    /**
     * @copydoc Shrinker::compute_global_threshold
     */
    cv::Scalar compute_global_threshold(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_universal_threshold(coeffs, coeffs.detail_mask(levels), stdev);
    }

    /**
     * @copydoc Shrinker::compute_level_threshold
     */
    cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_universal_threshold(detail_coeffs, stdev);
    }

    /**
     * @copydoc Shrinker::compute_subband_threshold
     */
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
 * @brief Gobal shrinkage using the universal threshold, the soft threshold.
 *        function, and the MAD standard deviation estimator.
 *
 */
class VisuShrinker : public UniversalShrinker
{
public:
    /**
     * @overload
     *
     */
    VisuShrinker() :
        VisuShrinker(Shrinker::GLOBALLY)
    {}

    /**
     * @brief Construct a new VisuShrinker object.
     *
     * @param[in] partition
     */
    VisuShrinker(Shrinker::Partition partition) :
        UniversalShrinker(
            partition,
            soft_threshold
        )
    {}

    /**
     * @brief Copy Constructor.
     */
    VisuShrinker(const VisuShrinker& other) = default;
    /**
     * @brief Move Constructor.
     */
    VisuShrinker(VisuShrinker&& other) = default;
};

//  ----------------------------------------------------------------------------
//  VisuShrink Functional API
//  ----------------------------------------------------------------------------
/** @name VisuShrink Functional API
 *  @{
 */
/**
 * @brief Shrinks detail coefficients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs visu_shrink(const DWT2D::Coeffs& coeffs);

/**
 * @brief Shrinks detail coefficients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void visu_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief Shrinks detail coefficients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs visu_shrink(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrinks detail coefficients using the VisuShrink algorithm.
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

