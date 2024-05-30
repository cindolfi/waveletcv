#ifndef CVWT_SHRINK_SHRINK_HPP
#define CVWT_SHRINK_SHRINK_HPP

#include <thread>
#include <utility>
#include <opencv2/core.hpp>
#include "cvwt/dwt2d.hpp"
#include "cvwt/utils.hpp"
#include "cvwt/exception.hpp"

namespace cvwt
{
/** @addtogroup shrinkage Shrink DWT Coefficients
 *  @{
 */

/**
 * @brief A function that shrinks matrix elements towards zero.
 */
using ShrinkFunction = std::function<
    void(
        cv::InputArray, // input
        cv::OutputArray, // output
        cv::Scalar, // threshold
        cv::InputArray // mask
    )
>;

/**
 * @brief A function that shrinks primitive values towards zero.
 */
template <typename Value, typename Threshold>
using PrimitiveShrinkFunction = std::function<Value(Value, Threshold)>;

/**
 * @brief A function that estimates the population standard deviation from samples.
 */
using StdDevFunction = std::function<cv::Scalar(cv::InputArray)>;

namespace internal
{
template <typename T, int N, typename ThresholdFunctor>
struct Threshold
{
    using Pixel = cv::Vec<T, N>;
    Threshold() : threshold_function() {}
    Threshold(ThresholdFunctor threshold_function) : threshold_function(threshold_function) {}

    void operator()(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold) const
    {
        assert(input.channels() == N);

        output.create(input.size(), input.type());
        auto result = output.getMat();
        input.getMat().forEach<Pixel>(
            [&](const auto& pixel, const auto position) {
                auto& result_pixel = result.at<Pixel>(position);
                for (int i = 0; i < N; ++i)
                    result_pixel[i] = threshold_function(pixel[i], threshold[i]);
            }
        );
    }

    void operator()(
        cv::InputArray input,
        cv::OutputArray output,
        cv::Scalar threshold,
        cv::InputArray mask
    ) const
    {
        assert(input.channels() == N);
        throw_if_bad_mask_type(mask);

        output.create(input.size(), input.type());
        auto mask_matrix = mask.getMat();
        auto result = output.getMat();
        input.getMat().forEach<Pixel>(
            [&](const auto& pixel, const auto position) {
                if (mask_matrix.at<uchar>(position)) {
                    auto& result_pixel = result.at<Pixel>(position);
                    for (int i = 0; i < N; ++i)
                        result_pixel[i] = threshold_function(pixel[i], threshold[i]);
                } else {
                    result.at<Pixel>(position) = pixel;
                }
            }
        );
    }

public:
    ThresholdFunctor threshold_function;
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
using SoftThreshold = Threshold<T, N, SoftThresholdFunctor>;

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
using HardThreshold = Threshold<T, N, HardThresholdFunctor>;

template <typename T, typename W>
struct WrappedThresholdFunctor
{
    constexpr WrappedThresholdFunctor(
        PrimitiveShrinkFunction<T, W> threshold_function
    ) :
        _threshold_function(threshold_function)
    {}

    constexpr T operator()(T value, W threshold) const
    {
        return _threshold_function(value, threshold);
    }
private:
    PrimitiveShrinkFunction<T, W> _threshold_function;
};

template <typename T, int N, typename W>
struct WrappedThreshold : public Threshold<T, N, WrappedThresholdFunctor<T, W>>
{
    WrappedThreshold(
        PrimitiveShrinkFunction<T, W> threshold_function
    ) :
        Threshold<T, N, WrappedThresholdFunctor<T, W>>(
            WrappedThresholdFunctor(threshold_function)
        )
    {}
};
}   // namespace internal





/** @name Shrink Low Level API
 *  @{
 */
//  ----------------------------------------------------------------------------
//  Thresholding
//  ----------------------------------------------------------------------------
/**
 * @brief Multichannel masked soft threshold.
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
 * @param[in] array
 * @param[out] result
 * @param[in] threshold
 * @param[in] mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are thresholded.  The identity
 *             function is applied to inputs at corresponding zero entries.
 */
void soft_threshold(
    cv::InputArray array,
    cv::OutputArray result,
    cv::Scalar threshold,
    cv::InputArray mask = cv::noArray()
);

/**
 * @brief Multichannel masked hard threshold.
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
 * @param[in] array
 * @param[out] result
 * @param[in] threshold
 * @param[in] mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are thresholded.  The identity
 *             function is applied to inputs at corresponding zero entries.
 */
void hard_threshold(
    cv::InputArray array,
    cv::OutputArray result,
    cv::Scalar threshold,
    cv::InputArray mask = cv::noArray()
);

/**
 * @brief Creates a multichannel shrink function from a primitive shrink functor.
 *
 * @code{cpp}
 * struct MyPrimitiveShrinkFunctor
 * {
 *     template <typename T, typename W>
 *     T operator(T value, W threshold);
 * };
 *
 * auto my_shrink_function = make_shrink_function<MyPrimitiveShrinkFunctor>();
 *
 * cv::Mat image = ...;
 * cv::Mat thresholded_image;
 * cv::Scalar threshold = ...;
 * my_shrink_function(image, thresholded_image, threshold);
 * @endcode
 *
 * @tparam ThresholdFunctor
 */
template <typename PrimitiveShrinkFunctor>
ShrinkFunction make_shrink_function()
{
    return [](
        cv::InputArray array,
        cv::OutputArray result,
        cv::Scalar threshold,
        cv::InputArray mask = cv::noArray()
    )
    {
        internal::dispatch_on_pixel_depth<internal::Threshold, PrimitiveShrinkFunctor>(
            array.type(),
            array,
            result,
            threshold,
            mask
        );
    };
}

/**
 * @brief Creates a multichannel shrink function from a primitive shrink function.
 *
 * @code{cpp}
 * template <typename T, typename W>
 * T my_primitive_shrink_function(T value, W threshold);
 *
 * auto my_shrink_function = make_shrink_function(my_primitive_shrink_function);
 *
 * cv::Mat image = ...;
 * cv::Mat thresholded_image;
 * cv::Scalar threshold = ...;
 * my_shrink_function(image, thresholded_image, threshold);
 * @endcode
 *
 * @tparam T The primitive value type.
 * @tparam W The primitive threshold type.
 * @param[in] shrink_function The threshold function that acts on primitive types.
 */
template <typename T, typename W = T>
ShrinkFunction make_shrink_function(
    PrimitiveShrinkFunction<T, W> shrink_function
)
{
    return [&](
        cv::InputArray array,
        cv::OutputArray result,
        cv::Scalar threshold,
        cv::InputArray mask
    )
    {
        internal::dispatch_on_pixel_depth<internal::WrappedThreshold, W>(
            std::make_tuple(shrink_function),
            array.type(),
            array,
            result,
            threshold,
            mask
        );
    };
}

//  ----------------------------------------------------------------------------
//  Shrink
//  ----------------------------------------------------------------------------
/**
 * @brief Shrinks DWT detail coefficients using a single threshold.
 *
 * The coefficients are shrunk in place.
 *
 * Use the `levels` argument to limit shrinking to a subset of decomposition levels.
 *
 * @param[inout] coeffs
 * @param[in] threshold
 * @param[in] shrink_function
 * @param[in] levels
 */
void shrink_globally(
    DWT2D::Coeffs& coeffs,
    const cv::Scalar& threshold,
    ShrinkFunction shrink_function,
    const cv::Range& levels = cv::Range::all()
);

/**
 * @brief Shrinks DWT coefficients using separate thresholds for each decomposition level.
 *
 * The coefficients are shrunk in place.
 *
 * Use the `levels` argument to limit shrinking to a subset of decomposition levels.
 * If `levels` is `cv::Range::all()`, `level_thresholds.rows` must equal `coeffs.levels()`.
 * Otherwise, `level_thresholds.rows` must equal `levels.size()`.
 *
 * @see
 *  - Shrink
 *  - shrink_globally()
 *  - shrink_subbands()
 *  - make_shrink_function()
 *
 * @param[inout] coeffs The DWT coefficients.
 * @param[in] level_thresholds The threshold paramaters for each level.
 *      This must be an array with N rows, 1 column, and 4 channels (where N is
 *      the number of levels to be shrunk). Each row corresponds to a
 *      decomposition level.
 * @param[in] shrink_function The function that shrinks the coefficients.
 * @param[in] levels The decomposition levels that are shrunk.
 * @sa Shrink, shrink_globally, shrink_subbands, make_shrink_function
 */
void shrink_levels(
    DWT2D::Coeffs& coeffs,
    cv::InputArray level_thresholds,
    ShrinkFunction shrink_function,
    const cv::Range& levels = cv::Range::all()
);

/**
 * @brief Shrinks DWT coefficients using separate thresholds for each decomposition subband.
 *
 * The coefficients are shrunk in place.
 *
 * Use the `levels` argument to limit shrinking to a subset of decomposition levels.
 * If `levels` is `cv::Range::all()`, `subband_thresholds.rows` must equal `coeffs.levels()`.
 * Otherwise, `subband_thresholds.rows` must equal `levels.size()`.
 *
 * @see
 *  - Shrink
 *  - shrink_globally()
 *  - shrink_levels()
 *  - make_shrink_function()
 *
 * @param[inout] coeffs The DWT coefficients.
 * @param[in] subband_thresholds The threshold paramaters for each subband.
 *      This must be an array with N rows, 3 columns, and 4 channels (where N is
 *      the number of levels to be shrunk). Each row corresponds to a
 *      decomposition level.  The first column contains horizontal subband
 *      thresholds, the second column contains vertical subband thresholds, and
 *      the third column contains diagonal subband thresholds.
 * @param[in] shrink_function The function that shrinks the coefficients.
 * @param[in] levels The decomposition levels that are shrunk.
 */
void shrink_subbands(
    DWT2D::Coeffs& coeffs,
    cv::InputArray subband_thresholds,
    ShrinkFunction shrink_function,
    const cv::Range& levels = cv::Range::all()
);
/** @} Shrink Low Level API*/







//  ============================================================================
//  High Level API
//  ============================================================================
/**
 * @brief Base class for DWT coefficient shrinkage algorithms.
 *
 * The DWT coefficients are partitioned into disjoint subsets and each subset
 * is shrunk toward zero by a shrink_function().
 *
 * Algorithms will support one or more of the following partitions:
 *  - All detail coefficients comprise a single set (Shrink::GLOBALLY)
 *  - Coefficients are partitioned by decomposition level (Shrink::LEVELS)
 *  - Coefficients are partitioned by subband (Shrink::SUBBANDS)
 *  - Specialized, implementation defined partition (Shrink::SUBSETS)
 *
 * The format of the thresholds matrix returned by compute_thresholds(), called
 * with N decomposition levels, depends on the partition():
 *  - Shrink::GLOBALLY: 1 row and 1 column.
 *  - Shrink::LEVELS: N rows and 1 column, where each row corresponds to a level.
 *  - Shrink::SUBBANDS: N rows and 3 columns, where each row corresponds to a
 *    level and the columns corresponds to HORIZONTAL, VERTICAL, and DIAGONAL
 *    subbands.
 *  - Shrink::SUBSETS: implementation defined
 *
 * Many shrinkage algorithms assume that the original image pixels are drawn
 * from a normal distribution with a identical variance.  Since this
 * is generally unknown, the noise standard deviation is estimated from the
 * coefficients by compute_noise_stdev().
 *
 * Usage
 * =====
 *
 * Shrinking Coefficients
 * ----------------------
 * To shrink all detail coefficients:
 * @code{cpp}
 * cvwt::DWT2D::Coeffs coeffs = ...;
 * cvwt::Shrink* shrinker = ...;
 * cvwt::DWT2D::Coeffs shrunken_coeffs;
 * shrunken_coeffs = shrinker->shrink(coeffs);
 * @endcode
 *
 * To shrink only the highest resolution detail coefficients:
 * @code{cpp}
 * shrunken_coeffs = shrinker->shrink(coeffs, 1);
 * @endcode
 *
 * To shrink only the first two decomposition levels:
 * @code{cpp}
 * shrunken_coeffs = shrinker->shrink(coeffs, 2);
 * @endcode
 *
 * To shrink all but the highest resolution detail coefficients:
 * @code{cpp}
 * shrunken_coeffs = shrinker->shrink(coeffs, cv::Range(1, coeffs.levels()));
 * @endcode
 *
 * Shrink objects are also functors:
 * @code{cpp}
 * cvwt::BayesShrink bayesshrink;
 * shrunken_coeffs = bayesshrink(coeffs);
 * @endcode
 *
 * Working With Thresholds
 * -----------------------
 * To compute the thresholds:
 * @code{cpp}
 * cv::Mat4d thresholds;
 * thresholds = shrinker->compute_thresholds(coeffs);
 * @endcode
 *
 * To shrink detail coefficients and get the thresholds in a single call:
 * @code{cpp}
 * shrunken_coeffs = shrinker->shrink(coeffs, thresholds);
 * @endcode
 *
 * To get an array containing the threshold for each corresponding coefficient:
 * @code{cpp}
 * cv::Mat coeff_thresholds = shrinker->expand_thresholds(coeffs, thresholds);
 * @endcode
 *
 * To compute a mask that indicates which coefficients were shrunk to zero:
 * @code{cpp}
 * cv::Mat shrunk_coeffs_mask;
 * cvwt::less_than_or_equal(
 *     cv::abs(coeffs),
 *     coeffs_thresholds,
 *     shrunk_coeffs_mask,
 *     coeffs.detail_mask()
 * );
 * @endcode
 *
 * @note In the unlikely situation that the coefficient noise variance is known
 *       (e.g. from a knowlegdge about image acquisition) users should call
 *       the shrink() and compute_thresholds() overloads that take a `stdev`.
 *       In most cases the noise variance must be estimated from the
 *       coefficients and users should call the shrink() and
 *       compute_thresholds() overloads that do not accept a `stdev`, in which
 *       case it is estimated internally using compute_noise_stdev().
 *
 * Subclassing
 * ===========
 *
 * Algorithms that shrink all detail coefficients using a single threshold must
 * implement:
 *  - A constructor that passes or allows the user to pass Shrink::GLOBALLY to
 *    this class's constructor
 *  - compute_global_threshold()
 *
 * Algorithms that shrink detail coefficients using a separate threshold for
 * each level must implement:
 *  - A constructor that passes or allows the user to pass Shrink::LEVELS to
 *    this class's constructor
 *  - compute_level_threshold()
 *
 * Algorithms that shrink detail coefficients using a separate threshold for
 * each subband must implement:
 *  - A constructor that passes or allows the user to pass Shrink::SUBBAND to
 *    this class's constructor
 *  - compute_subband_threshold()
 *
 * Algorithms that shrink detail coefficients using a partitioning scheme other
 * than those listed above must implement:
 *  - A constructor that passes or allows the user to pass Shrink::SUBSETS to
 *    this class's constructor
 *  - compute_subset_thresholds()
 *  - expand_subset_thresholds()
 *  - shrink_subsets()
 *
 * The default implementation of compute_noise_stdev() calls stdev_function() on
 * the diagonal subband at the finest resolution (i.e. coeffs.diagonal_detail(0)).
 * The default stdev_function() is mad_stdev(), which gives a statistically
 * robust estimate of the standard deviation.
 *
 * Subclasses can override compute_noise_stdev() to change which coefficients
 * are used to esitmate the coefficient noise standard deviation.  Subclasses
 * should change the standard deviation estimator by passing a different
 * estimator to this class's constructor.
 *
 * For performance reasons, algorithms that do not require an estimate of the
 * noise variance should override compute_noise_stdev() to do nothing.
 */
class Shrink
{
public:
    /**
     * @brief The scheme used to split the DWT coefficients into disjoint subsets.
     */
    enum Partition {
        /** Shrink all coefficients using a single threshold */
        GLOBALLY = 0,
        /** Shrink each decomposition level using a separate threshold */
        LEVELS = 1,
        /** Shrink each subband using a separate threshold */
        SUBBANDS = 2,
        /** Shrink coefficients according to a custom partition */
        SUBSETS = 3,
    };

    /**
     * @private
     * @brief Uses RAII to acquire a lock and call start_partitioning() and finish_partitioning().
     *
     * The purpose of this class is to ensure that finish_partitioning() is called when
     * shrink() and compute_thresholds() terminates normally as well as when an
     * exception is thrown.
     */
    class PartitioningContext
    {
    public:
        PartitioningContext(
            const Shrink* shrink,
            const DWT2D::Coeffs& coeffs,
            const cv::Range& levels,
            const cv::Scalar& stdev,
            const DWT2D::Coeffs* shrunk_coeffs,
            const cv::Mat4d* subset_thresholds
        ) :
            _shrink(shrink),
            _lock(shrink->_mutex),
            _coeffs(coeffs),
            _levels(levels),
            _stdev(stdev),
            _shrunk_coeffs(shrunk_coeffs),
            _subset_thresholds(subset_thresholds)
        {
            assert(_subset_thresholds);
            _shrink->start_partitioning(_coeffs, _levels, _stdev);
        }

        ~PartitioningContext()
        {
            _shrink->finish_partitioning(
                _coeffs,
                _levels,
                _stdev,
                _shrunk_coeffs ? *_shrunk_coeffs : DWT2D::Coeffs(),
                _subset_thresholds ? *_subset_thresholds : cv::Mat4d()
            );
        }

    private:
        const Shrink* _shrink;
        std::lock_guard<std::mutex> _lock;
        const DWT2D::Coeffs& _coeffs;
        const cv::Range& _levels;
        const cv::Scalar& _stdev;
        const DWT2D::Coeffs* _shrunk_coeffs;
        const cv::Mat4d* _subset_thresholds;
    };

public:
    /**
     * @private
     */
    Shrink() = delete;
    /**
     * @brief Copy Constructor.
     */
    Shrink(const Shrink& other) :
        _partition(other._partition),
        _shrink_function(other._shrink_function),
        _stdev_function(other._stdev_function)
    {}
    /**
     * @brief Move Constructor.
     */
    Shrink(Shrink&& other) = default;

    //  ------------------------------------------------------------------------
    //  Getters & Setters
    /**
     * @brief The scheme used to partition the DWT coefficients.
     */
    Shrink::Partition partition() const { return _partition; }
    /**
     * @brief The function used to shrink a subset of DWT coefficients.
     */
    ShrinkFunction shrink_function() const { return _shrink_function; }
    /**
     * @brief The standard deviation estimator.
     */
    StdDevFunction stdev_function() const { return _stdev_function; }

    //  ------------------------------------------------------------------------
    //  Shrink
    /**
     * @brief Shrinks DWT coefficients.
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       shrink(const DWT2D::Coeffs&, const cv::Scalar&, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, cv::Range::all(), compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[out] thresholds The computed thresholds.
     */
    [[nodiscard]]
    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, cv::Range::all(), compute_noise_stdev(coeffs), thresholds);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, const cv::Scalar&, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, cv::Range::all(), compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[out] shrunk_coeffs The shrunken DWT coefficients.
     * @param[out] thresholds The computed thresholds.
     */
    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, cv::Range::all(), compute_noise_stdev(coeffs), thresholds);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise is *known*.  If the
     *       noise must be *estimated* from the coefficients use
     *       shrink(const DWT2D::Coeffs&, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, cv::Range::all(), stdev, thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] stdev The standard deviation of the coefficient noise.
     * @param[out] thresholds The computed thresholds.
     */
    [[nodiscard]]
    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& stdev,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, cv::Range::all(), stdev, thresholds);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise is *known*.  If the
     *       noise must be *estimated* from the coefficients use
     *       shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, cv::Range::all(), stdev, thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[out] shrunk_coeffs The shrunken DWT coefficients.
     * @param[in] stdev The standard deviation of the coefficient noise.
     * @param[out] thresholds The computed thresholds.
     */
    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        const cv::Scalar& stdev,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, cv::Range::all(), stdev, thresholds);
    }

    //  ------------------------------------------------------------------------
    /**
     * @brief Shrinks DWT coefficients for a subset of levels.
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       shrink(const DWT2D::Coeffs&, int, const cv::Scalar&, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, cv::Range(0, levels), thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to shrink.
     * @param[out] thresholds The computed thresholds.
     */
    [[nodiscard]]
    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        int levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, cv::Range(0, levels), thresholds);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, const cv::Scalar&, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, cv::Range(0, levels), compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[out] shrunk_coeffs The shrunken DWT coefficients.
     * @param[in] levels The subset of levels to shrink.
     * @param[out] thresholds The computed thresholds.
     */
    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        int levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, cv::Range(0, levels), compute_noise_stdev(coeffs), thresholds);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise is *known*.  If the
     *       noise must be *estimated* from the coefficients use
     *       shrink(const DWT2D::Coeffs&, int, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, cv::Range(0, levels), stdev, thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to shrink.
     * @param[in] stdev The standard deviation of the coefficient noise.
     * @param[out] thresholds The computed thresholds.
     */
    [[nodiscard]]
    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& stdev,
        int levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, cv::Range(0, levels), stdev, thresholds);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise is *known*.  If the
     *       noise must be *estimated* from the coefficients use
     *       shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, int, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, cv::Range(0, levels), stdev, thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[out] shrunk_coeffs The shrunken DWT coefficients.
     * @param[in] levels The subset of levels to shrink.
     * @param[in] stdev The standard deviation of the coefficient noise.
     * @param[out] thresholds The computed thresholds.
     */
    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        int levels,
        const cv::Scalar& stdev,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, cv::Range(0, levels), stdev, thresholds);
    }

    //  ------------------------------------------------------------------------
    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       shrink(const DWT2D::Coeffs&, const cv::Range&, const cv::Scalar&, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, levels, compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to shrink.
     * @param[out] thresholds The computed thresholds.
     */
    [[nodiscard]]
    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, levels, compute_noise_stdev(coeffs), thresholds);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       shrink(const DWT2D::Coeffs&, const DWT2D::Coeffs&, const cv::Range&, const cv::Scalar&, cv::OutputArray) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, levels, compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[out] shrunk_coeffs The shrunken DWT coefficients.
     * @param[in] levels The subset of levels to shrink.
     * @param[out] thresholds The computed thresholds.
     */
    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        const cv::Range& levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, levels, compute_noise_stdev(coeffs), thresholds);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       shrink(const DWT2D::Coeffs&, const cv::Range&, cv::OutputArray) const
     *       instead.
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to shrink.
     * @param[in] stdev The standard deviation of the coefficient noise.
     * @param[out] thresholds The computed thresholds.
     */
    [[nodiscard]]
    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        DWT2D::Coeffs shrunk_coeffs;
        shrink(coeffs, shrunk_coeffs, levels, stdev, thresholds);

        return shrunk_coeffs;
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise is *known*.  If the
     *       noise must be *estimated* from the coefficients use
     *       shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, const cv::Range&, cv::OutputArray) const
     *       instead.
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[out] shrunk_coeffs The shrunken DWT coefficients.
     * @param[in] levels The subset of levels to shrink.
     * @param[in] stdev The standard deviation of the coefficient noise.
     * @param[out] thresholds The computed thresholds.
     */
    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev,
        cv::OutputArray thresholds = cv::noArray()
    ) const;

    //  ------------------------------------------------------------------------
    //  Functor Interface
    /**
     * @brief Alias of shrink().
     */
    [[nodiscard]]
    auto operator()(auto&&... args) const
    {
        return shrink(std::forward<decltype(args)>(args)...);
    }

    //  ------------------------------------------------------------------------
    //  Compute Thresholds
    /**
     * @brief Computes the threshold for each of the partitioned subsets.
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       compute_thresholds(const DWT2D::Coeffs&, const cv::Range&, const cv::Scalar&) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * compute_thresholds(coeffs, levels, compute_noise_stdev(coeffs));
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     */
    cv::Mat4d compute_thresholds(const DWT2D::Coeffs& coeffs, const cv::Range& levels) const
    {
        return compute_thresholds(coeffs, levels, compute_noise_stdev(coeffs));
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise is *known*.  If the
     *       noise must be *estimated* from the coefficients use
     *       compute_thresholds(const DWT2D::Coeffs&, const cv::Range&) const
     *       instead.
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       compute_thresholds(const DWT2D::Coeffs&, int, const cv::Scalar&) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * compute_thresholds(coeffs, cv::Range(0, levels), compute_noise_stdev(coeffs));
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     */
    cv::Mat4d compute_thresholds(const DWT2D::Coeffs& coeffs, int levels) const
    {
        return compute_thresholds(coeffs, cv::Range(0, levels), compute_noise_stdev(coeffs));
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise is *known*.  If the
     *       noise must be *estimated* from the coefficients use
     *       compute_thresholds(const DWT2D::Coeffs&, int) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * compute_thresholds(coeffs, cv::Range(0, levels), stdev);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        int levels,
        const cv::Scalar& stdev
    ) const
    {
        return compute_thresholds(coeffs, cv::Range(0, levels), stdev);
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise must be *estimated*
     *       from the coefficients.  If the noise is *known* use
     *       compute_thresholds(const DWT2D::Coeffs&, const cv::Scalar&) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * compute_thresholds(coeffs, cv::Range::all(), compute_noise_stdev(coeffs));
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     */
    cv::Mat4d compute_thresholds(const DWT2D::Coeffs& coeffs) const
    {
        return compute_thresholds(coeffs, cv::Range::all(), compute_noise_stdev(coeffs));
    }

    /**
     * @overload
     *
     * @note Use this overload when the coefficient noise is *known*.  If the
     *       noise must be *estimated* from the coefficients use
     *       compute_thresholds(const DWT2D::Coeffs&) const
     *       instead.
     *
     * This is equivalent to:
     * @code{cpp}
     * compute_thresholds(coeffs, cv::Range::all(), stdev);
     * @endcode
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& stdev
    ) const
    {
        return compute_thresholds(coeffs, cv::Range::all(), stdev);
    }

    //  ------------------------------------------------------------------------
    //  Expand Thresholds
    /**
     * @brief Expands the subset thresholds to a threshold matrix of the same size as the given coefficients.
     *
     * This function expands the matrix of subset thresholds to a matrix that
     * is the same size as `coeffs` where each entry is the threshold for
     * the corresponding coefficient.
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] subset_thresholds The thresholds returned by compute_threshold().
     * @param[in] levels The subset of levels to compute thresholds for.
     */
    cv::Mat expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& subset_thresholds,
        const cv::Range& levels
    ) const
    {
        cv::Mat expanded_thresholds;
        expand_thresholds(coeffs, subset_thresholds, expanded_thresholds, levels);

        return expanded_thresholds;
    }

    /**
     * @overload
     *
     * This function expands the matrix of subset thresholds to a matrix that
     * is the same size as `coeffs` where each entry is the threshold for
     * the corresponding coefficient.
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] subset_thresholds The thresholds returned by compute_threshold().
     * @param[out] expanded_thresholds
     * @param[in] levels The subset of levels to compute thresholds for.
     */
    void expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& subset_thresholds,
        cv::OutputArray expanded_thresholds,
        const cv::Range& levels
    ) const;

    /**
     * @overload
     *
     * This is equivalent to:
     * @code{cpp}
     * expand_thresholds(coeffs, thresholds, cv::Range(0, levels));
     * @endcode
     *
     * @param[in] coeffs
     * @param[in] subset_thresholds
     * @param[in] levels
     */
    cv::Mat expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& subset_thresholds,
        int levels
    ) const
    {
        return expand_thresholds(coeffs, subset_thresholds, cv::Range(0, levels));
    }

    /**
     * @overload
     *
     * This is equivalent to:
     * @code{cpp}
     * expand_thresholds(coeffs, thresholds, cv::Range(0, levels));
     * @endcode
     *
     * @param[in] coeffs
     * @param[in] subset_thresholds
     * @param[out] expanded_thresholds
     * @param[in] levels
     */
    void expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        cv::OutputArray expanded_thresholds,
        const cv::Mat4d& subset_thresholds,
        int levels
    ) const
    {
        expand_thresholds(coeffs, subset_thresholds, expanded_thresholds, cv::Range(0, levels));
    }

    /**
     * @overload
     *
     * This is equivalent to:
     * @code{cpp}
     * expand_thresholds(coeffs, thresholds, cv::Range::all());
     * @endcode
     *
     * @param[in] coeffs
     * @param[in] subset_thresholds
     */
    cv::Mat expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& subset_thresholds
    ) const
    {
        return expand_thresholds(coeffs, subset_thresholds, cv::Range::all());
    }

    /**
     * @overload
     *
     * This is equivalent to:
     * @code{cpp}
     * expand_thresholds(coeffs, thresholds, cv::Range::all());
     * @endcode
     *
     * @param[in] coeffs
     * @param[in] subset_thresholds
     * @param[out] expanded_thresholds
     */
    void expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& subset_thresholds,
        cv::OutputArray expanded_thresholds
    ) const
    {
        expand_thresholds(coeffs, subset_thresholds, expanded_thresholds, cv::Range::all());
    }

    //  ------------------------------------------------------------------------
    //  Compute Standard Deviation
    /**
     * @brief Estimates the coefficient noise standard deviation.
     *
     * The default implementation calls stdev_function() on the diagonal subband
     * at the finest resolution (i.e. coeffs.diagonal_detail(0)).
     *
     * The default stdev_function() is mad_stdev(), which gives a statistically
     * robust estimate of the standard deviation.
     *
     * Subclasses can override compute_noise_stdev() to change which coefficients
     * are used to esitmate the coefficient noise standard deviation.  Subclasses
     * should change the standard deviation estimator by passing a different
     * estimator to this class's constructor.
     *
     * @param[in] coeffs The entire set of DWT coefficients.
     */
    virtual cv::Scalar compute_noise_stdev(const DWT2D::Coeffs& coeffs) const
    {
        return compute_stdev(coeffs.diagonal_detail());
    }

protected:
    /**
     * @brief Construct a new Shrink object.
     *
     * @param[in] partition The scheme used to partition the coefficients.
     * @param[in] shrink_function The function used to shrink coefficients.
     * @param[in] stdev_function The function used to compute an estimate of the
     *                       standard deviation of coefficient noise.
     */
    Shrink(
        Shrink::Partition partition,
        ShrinkFunction shrink_function,
        StdDevFunction stdev_function
    ) :
        _partition(partition),
        _shrink_function(shrink_function),
        _stdev_function(stdev_function)
    {}

    /**
     * @overload
     *
     * This is equivalent to:
     * @code{cpp}
     * Shrink(partition, shrink_function, mad_stdev)
     * @endcode
     *
     * @param[in] partition The scheme used to partition the coefficients.
     * @param[in] shrink_function The function used to shrink coefficients.
     */
    Shrink(
        Shrink::Partition partition,
        ShrinkFunction shrink_function
    ) :
        Shrink(
            partition,
            shrink_function,
            mad_stdev
        )
    {}

    /**
     * @overload
     *
     * This is equivalent to:
     * @code{cpp}
     * Shrink(partition, make_shrink_function(shrink_function), stdev_function)
     * @endcode
     *
     * @tparam T
     * @tparam W
     * @param[in] partition The scheme used to partition the coefficients.
     * @param[in] shrink_function The function used to shrink coefficients.
     * @param[in] stdev_function The function used to compute an estimate of the
     *                       standard deviation of coefficient noise.
     */
    template <typename T, typename W>
    Shrink(
        Shrink::Partition partition,
        PrimitiveShrinkFunction<T, W> shrink_function,
        StdDevFunction stdev_function
    ) :
        Shrink(
            partition,
            make_shrink_function(shrink_function),
            stdev_function
        )
    {}

    /**
     * @overload
     *
     * This is equivalent to:
     * @code{cpp}
     * Shrink(partition, make_shrink_function(shrink_function), mad_stdev)
     * @endcode
     *
     * @tparam T
     * @tparam W
     * @param[in] partition The scheme used to partition the coefficients.
     * @param[in] shrink_function The function used to shrink coefficients.
     */
    template <typename T, typename W>
    Shrink(
        Shrink::Partition partition,
        PrimitiveShrinkFunction<T, W> shrink_function
    ) :
        Shrink(
            partition,
            shrink_function,
            mad_stdev
        )
    {}

    //  ------------------------------------------------------------------------
    //  Subclass API
    /**
     * @brief Prepare for a call to shrink() or compute_thresholds().
     *
     * Most subclasses do **not** need to override this function.
     *
     * A subclass should override this function if it computes *temporary*
     * values that must be accessed by
     *  - `shrink_subsets()` **and** `compute_subset_thresholds()`
     *  - or `compute_subband_threshold()` for **every** subband
     *  - or `compute_level_threshold()` for **every** level
     *
     * The temporary values should be stored as data members.
     *
     * @note Any data members set in this function **must** be declared mutable
     *       and **must** be cleaned up in finish_partitioning().
     *
     * One potential use case involves computing a set of partition masks that
     * are used by both compute_subset_thresholds() and shrink_subsets().
     * Another use case is computing global statistics that are used to compute
     * the threshold for each partitioned subset (i.e. used in
     * compute_level_threshold() or compute_subband_threshold()).
     *
     * @see finish_partitioning()
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    virtual void start_partitioning(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const
    {}

    /**
     * @brief Finish a call to shrink() or compute_thresholds().
     *
     * Cleanup temporary data members created in start_partitioning().  This function
     * is guaranteed to be called by shrink() and compute_thresholds()
     * as long as start_partitioning() did not throw an exception.  If necessary,
     * std::current_exception() can be used to determine if this function was
     * called normally or because an exception occured.  In the latter case,
     * `shrunk_coeffs` or `thresholds` may be empty depending on when the
     * exception was thrown.
     *
     * @code{cpp}
     * void MyShrinker::finish_partitioning(
     *     const DWT2D::Coeffs& coeffs,
     *     const cv::Range& levels,
     *     const cv::Scalar& stdev,
     *     const DWT2D::Coeffs& shrunk_coeffs,
     *     const cv::Mat4d& thresholds
     * ) const
     * {
     *     if (std::current_exception()) {
     *         // finishing up because an exception was thrown
     *     } else {
     *         // finishing up normally
     *     }
     * }
     * @endcode
     *
     * @see start_partitioning()
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     * @param[in] stdev The standard deviation of the coefficient noise.
     * @param[in] shrunk_coeffs The shrunken DWT coefficients.  This is empty when
     *                      called from compute_thresholds().
     * @param[in] thresholds The computed thresholds.
     */
    virtual void finish_partitioning(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev,
        const DWT2D::Coeffs& shrunk_coeffs,
        const cv::Mat4d& thresholds
    ) const
    {}

    /**
     * @brief Computes the thresholds for all subsets.
     *
     * This function:
     *  - Calls compute_global_threshold(), compute_level_thresholds(),
     *    compute_subband_thresholds(), or compute_subset_thresholds() depending
     *    on the value of partition().
     *  - Is called by shrink() and compute_thresholds()
     *  - Does not call start_partitioning() and finish_partitioning()
     *
     * Subclasses do not typically need to call this function.
     *
     * @param[in] coeffs
     * @param[in] levels
     * @param[in] stdev
     */
    cv::Mat4d compute_partition_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    /**
     * @brief Computes the threshold on a single global subset of coefficients.
     *
     * Subclasses that that support a single threshold for all coefficients
     * (i.e. set partition to Shrink::GLOBALLY) **must** override this function.
     *
     * @param[in] coeffs The entire set of DWT coefficients.
     * @param[in] levels The subset of levels over which to compute the threshold.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    virtual cv::Scalar compute_global_threshold(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    /**
     * @brief Computes the threshold for a single level subset of coefficients.
     *
     * @param[in] detail_coeffs The level detail coefficients.
     * @param[in] level The decomposition level of the given coefficients.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    virtual cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& stdev
    ) const;

    /**
     * @brief Computes the threshold for a single subband subset of coefficients.
     *
     * @param[in] detail_coeffs The subband detail coefficients.
     * @param[in] level The decomposition level of the given coefficients.
     * @param[in] subband The subband of the coefficients.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    virtual cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& stdev
    ) const;

    /**
     * @brief Computes the thresholds for an implementation defined partition.
     *
     * @param[in] coeffs The entire set of coefficients.
     * @param[in] levels The subset of levels over which to compute the threshold.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    virtual cv::Mat4d compute_subset_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    /**
     * @brief Expands the thresholds for an implementation defined partition.
     *
     * @param[in] coeffs The entire set of coefficients.
     * @param[in] subset_thresholds The subset thresholds.
     * @param[in] levels The subset of levels over which to compute the threshold.
     * @param[out] expanded_thresholds The expanded thresholds.
     */
    virtual void expand_subset_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& subset_thresholds,
        const cv::Range& levels,
        cv::Mat& expanded_thresholds
    ) const;

    /**
     * @brief Shrinks the coefficients using a implementation defined partition.
     *
     * Subclasses that use a custom partitioning scheme (i.e. set partition to
     * Shrink::SUBSETS) **must** override this function.
     *
     * Implentations should partition the coefficients into one or more subsets
     * and call shrink_coeffs() for each subset in the partition.
     *
     * @param[in] coeffs The entire set of coefficients.
     * @param[in] subset_thresholds The thresholds returned by compute_subbset_thresholds().
     * @param[in] levels The subset of levels to shrink.
     */
    virtual void shrink_subsets(
        DWT2D::Coeffs& coeffs,
        const cv::Mat4d& subset_thresholds,
        const cv::Range& levels
    ) const;



    /**
     * @brief Apply the shrink_function() to the given coefficients.
     *
     * @param[in] coeffs The coefficients.
     * @param[out] thresholded_coeffs The shrunken coefficients.
     * @param[in] threshold The threshold parameter.
     * @param[in] mask Indicates which coefficients to apply the threshold function to.
     */
    void shrink_coeffs(
        cv::InputArray coeffs,
        cv::OutputArray thresholded_coeffs,
        const cv::Scalar& threshold,
        cv::InputArray mask = cv::noArray()
    ) const
    {
        _shrink_function(coeffs, thresholded_coeffs, threshold, mask);
    }

    /**
     * @overload
     *
     * This is an inplace version of shrink_coeffs().
     *
     * @param[inout] coeffs The coefficients.
     * @param[in] threshold The threshold parameter.
     * @param[in] mask Indicates which coefficients to apply the threshold function to.
     */
    void shrink_coeffs(
        cv::InputOutputArray coeffs,
        const cv::Scalar& threshold,
        cv::InputArray mask = cv::noArray()
    ) const
    {
        _shrink_function(coeffs, coeffs, threshold, mask);
    }

    /**
     * @brief Apply the stdev_function() to given array.
     *
     * @param[in] array
     */
    cv::Scalar compute_stdev(cv::InputArray array) const
    {
        return _stdev_function(array);
    }

    /**
     * @brief Sets the shrink function.
     *
     * @param[in] shrink_function
     */
    void set_shrink_function(ShrinkFunction shrink_function) { _shrink_function = shrink_function; }

    /**
     * @brief Set the standard deviation function.
     *
     * @param[in] stdev_function
     */
    void set_stdev_function(StdDevFunction stdev_function) { _stdev_function = stdev_function; }

    //  ------------------------------------------------------------------------
    //  Helpers
    /**
     * @brief Calls compute_level_threshold() for each level subset.
     *
     * @param[in] coeffs The entire set of DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    cv::Mat4d compute_level_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    /**
     * @brief Calls compute_subband_threshold() for each subband subset.
     *
     * @param[in] coeffs The entire set of DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     * @param[in] stdev The standard deviation of the coefficient noise.
     */
    cv::Mat4d compute_subband_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

private:
    Shrink::Partition _partition;
    ShrinkFunction _shrink_function;
    StdDevFunction _stdev_function;
    mutable std::mutex _mutex;
};
/** @} shrinkage*/

} // namespace cvwt

#endif  // CVWT_SHRINK_SHRINK_HPP

