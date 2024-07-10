#ifndef CVWT_SHRINK_SHRINK_HPP
#define CVWT_SHRINK_SHRINK_HPP

#include <thread>
#include <utility>
#include <opencv2/core.hpp>
#include "cvwt/dwt2d.hpp"
#include "cvwt/dispatch.hpp"
#include "cvwt/exception.hpp"
#include "cvwt/array/statistics.hpp"

namespace cvwt
{
/** @addtogroup shrink Shrink DWT Coefficients
 *  @{
 */
/**
 * @brief A function that shrinks matrix elements towards zero.
 *
 * Multichannel, masked
 *
 * @param[in] array The array the shrink
 * @param[out] shrunk_array The shrunk array
 * @param[in] threshold The threshold
 * @param[in] mask The mask
 *
 * @see make_shrink_function(), soft_threshold(), hard_threshold()
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
 * @brief A function that shrinks fundamental types towards zero.
 *
 * @see make_shrink_function()
 */
template <typename Value, typename Threshold>
using PrimitiveShrinkFunction = std::function<Value(Value, Threshold)>;

/**
 * @brief A function that estimates the population standard deviation from samples.
 */
using StdDevFunction = std::function<cv::Scalar(cv::InputArray)>;

namespace internal
{
template <typename T, int CHANNELS, typename ThresholdFunctor>
struct Threshold
{
    using Pixel = cv::Vec<T, CHANNELS>;
    Threshold() : threshold_function() {}
    Threshold(ThresholdFunctor threshold_function) : threshold_function(threshold_function) {}

    void operator()(cv::InputArray input, cv::OutputArray output, const cv::Scalar& threshold) const
    {
        assert(input.channels() == CHANNELS);

        output.create(input.size(), input.type());
        auto result = output.getMat();
        input.getMat().forEach<Pixel>(
            [&](const auto& pixel, const auto position) {
                auto& result_pixel = result.at<Pixel>(position);
                for (int i = 0; i < CHANNELS; ++i)
                    result_pixel[i] = threshold_function(pixel[i], threshold[i]);
            }
        );
    }

    void operator()(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Scalar& threshold,
        cv::InputArray mask
    ) const
    {
        assert(input.channels() == CHANNELS);
        throw_if_bad_mask_for_array(input, mask, AllowedMaskChannels::SINGLE);

        output.create(input.size(), input.type());
        auto mask_matrix = mask.getMat();
        auto result = output.getMat();
        input.getMat().forEach<Pixel>(
            [&](const auto& pixel, const auto position) {
                if (mask_matrix.at<uchar>(position)) {
                    auto& result_pixel = result.at<Pixel>(position);
                    for (int i = 0; i < CHANNELS; ++i)
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

template <typename T, int CHANNELS>
using SoftThreshold = Threshold<T, CHANNELS, SoftThresholdFunctor>;

struct HardThresholdFunctor
{
    template <typename T1, typename T2>
    constexpr auto operator()(T1 x, T2 threshold) const
    {
        auto abs_x = std::fabs(x);
        return (abs_x > threshold) * x;
    }
};

template <typename T, int CHANNELS>
using HardThreshold = Threshold<T, CHANNELS, HardThresholdFunctor>;

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

template <typename T, int CHANNELS, typename W>
struct WrappedThreshold : public Threshold<T, CHANNELS, WrappedThresholdFunctor<T, W>>
{
    WrappedThreshold(
        PrimitiveShrinkFunction<T, W> threshold_function
    ) :
        Threshold<T, CHANNELS, WrappedThresholdFunctor<T, W>>(
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
    const cv::Scalar& threshold,
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
    const cv::Scalar& threshold,
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
        const cv::Scalar& threshold,
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
        const cv::Scalar& threshold,
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

/**
 * @brief Shrinks DWT detail coefficients using a single threshold.
 *
 * The coefficients are shrunk in place.
 *
 * Use the @pref{levels} argument to limit shrinking to a subset of
 * decomposition levels.
 *
 * @param[inout] coeffs
 * @param[in] threshold
 * @param[in] shrink_function
 * @param[in] levels
 *
 * @see Shrinker, shrink_levels, shrink_subbands, make_shrink_function
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
 * Use the @pref{levels} argument to limit shrinking to a subset of
 * decomposition levels.
 * If @pref{levels} is cv::Range::all(),
 * @pref{level_thresholds,rows,cv::Mat::rows} must equal
 * @pref{coeffs,levels(),DWT2D::Coeffs::levels}.
 * Otherwise, @pref{level_thresholds,rows,cv::Mat::rows} must equal
 * @pref{levels,size(),cv::Range::size}.
 *
 * @param[inout] coeffs The DWT coefficients.
 * @param[in] level_thresholds The threshold paramaters for each level.
 *      This must be an array with N rows, 1 column, and 4 channels (where N is
 *      the number of levels to be shrunk). Each row corresponds to a
 *      decomposition level.
 * @param[in] shrink_function The function that shrinks the coefficients.
 * @param[in] levels The decomposition levels that are shrunk.
 *
 * @see Shrinker, shrink_globally, shrink_subbands, make_shrink_function
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
 * Use the @pref{levels} argument to limit shrinking to a subset of
 * decomposition levels.
 * If @pref{levels} is cv::Range::all(),
 * @pref{subband_thresholds,rows,cv::Mat::rows} must equal
 * @pref{coeffs,levels(),DWT2D::Coeffs::levels}.
 * Otherwise, @pref{subband_thresholds,rows,cv::Mat::rows} must equal
 * @pref{levels,size(),cv::Range::size}.
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
 *
 * @see Shrinker, shrink_globally, shrink_levels, make_shrink_function
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
 */
class Shrinker
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
            const Shrinker* shrink,
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
        const Shrinker* _shrink;
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
    Shrinker() = delete;
    /**
     * @brief Copy Constructor.
     */
    Shrinker(const Shrinker& other) :
        _partition(other._partition),
        _shrink_function(other._shrink_function),
        _stdev_function(other._stdev_function)
    {}
    /**
     * @brief Move Constructor.
     */
    Shrinker(Shrinker&& other) = default;

    //  ------------------------------------------------------------------------
    //  Getters & Setters
    /**
     * @brief The scheme used to partition the DWT coefficients.
     */
    Shrinker::Partition partition() const { return _partition; }
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, cv::Range::all(), compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @when_noise_unknown{
     * shrink(const DWT2D::Coeffs&\, const cv::Scalar&\, cv::OutputArray) const,
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, cv::Range::all(), compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @when_noise_unknown{
     * shrink(const DWT2D::Coeffs&\, DWT2D::Coeffs&\, const cv::Scalar&\, cv::OutputArray) const,
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, cv::Range::all(), stdev, thresholds);
     * @endcode
     *
     * @when_noise_known{
     * shrink(const DWT2D::Coeffs&\, cv::OutputArray) const,
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, cv::Range::all(), stdev, thresholds);
     * @endcode
     *
     * @when_noise_known{
     * shrink(const DWT2D::Coeffs&\, DWT2D::Coeffs&\, cv::OutputArray),
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, cv::Range(0, levels), thresholds);
     * @endcode
     *
     * @when_noise_unknown{
     * shrink(const DWT2D::Coeffs&\, int\, const cv::Scalar&\, cv::OutputArray) const,
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, cv::Range(0, levels), compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @when_noise_unknown{
     * shrink(const DWT2D::Coeffs&\, DWT2D::Coeffs&\, int\, const cv::Scalar&\, cv::OutputArray) const,
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, cv::Range(0, levels), stdev, thresholds);
     * @endcode
     *
     * @when_noise_known{
     * shrink(const DWT2D::Coeffs&\, int\, cv::OutputArray) const,
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, cv::Range(0, levels), stdev, thresholds);
     * @endcode
     *
     * @when_noise_known{
     * shrink(const DWT2D::Coeffs&\, DWT2D::Coeffs&\, int\, cv::OutputArray) const,
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, levels, compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @when_noise_unknown{
     * shrink(const DWT2D::Coeffs&\, const cv::Range&\, const cv::Scalar&\, cv::OutputArray) const,
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * shrink(coeffs, shrunk_coeffs, levels, compute_noise_stdev(coeffs), thresholds);
     * @endcode
     *
     * @when_noise_unknown{
     * shrink(const DWT2D::Coeffs&\, DWT2D::Coeffs&\, const cv::Range&\, const cv::Scalar&\, cv::OutputArray) const,
     * shrink()}
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
     * @when_noise_known{
     * shrink(const DWT2D::Coeffs&\, const cv::Range&\, cv::OutputArray) const,
     * shrink()}
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
     * @when_noise_known{
     * shrink(const DWT2D::Coeffs&\, DWT2D::Coeffs&\, const cv::Range&\, cv::OutputArray),
     * shrink()}
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
     * @equivalentto
     * @code{cpp}
     * compute_thresholds(coeffs, levels, compute_noise_stdev(coeffs));
     * @endcode
     *
     * @when_noise_unknown{
     * compute_thresholds(const DWT2D::Coeffs&\, const cv::Range&\, const cv::Scalar&) const,
     * compute_thresholds()}
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     */
    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels
    ) const
    {
        return compute_thresholds(coeffs, levels, compute_noise_stdev(coeffs));
    }

    /**
     * @overload
     *
     * @when_noise_known{
     * compute_thresholds(const DWT2D::Coeffs&\, const cv::Range&) const,
     * compute_thresholds()}
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
     * @equivalentto
     * @code{cpp}
     * compute_thresholds(coeffs, cv::Range(0, levels), compute_noise_stdev(coeffs));
     * @endcode
     *
     * @when_noise_unknown{
     * compute_thresholds(const DWT2D::Coeffs&\, int\, const cv::Scalar&) const,
     * compute_thresholds()}
     *
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     */
    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        int levels
    ) const
    {
        return compute_thresholds(coeffs, cv::Range(0, levels), compute_noise_stdev(coeffs));
    }

    /**
     * @overload
     *
     * @equivalentto
     * @code{cpp}
     * compute_thresholds(coeffs, cv::Range(0, levels), stdev);
     * @endcode
     *
     * @when_noise_known{
     * compute_thresholds(const DWT2D::Coeffs&\, int) const,
     * compute_thresholds()}
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
     * @equivalentto
     * @code{cpp}
     * compute_thresholds(coeffs, cv::Range::all(), compute_noise_stdev(coeffs));
     * @endcode
     *
     * @when_noise_unknown{
     * compute_thresholds(const DWT2D::Coeffs&\, const cv::Scalar&) const,
     * compute_thresholds()}
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
     * @equivalentto
     * @code{cpp}
     * compute_thresholds(coeffs, cv::Range::all(), stdev);
     * @endcode
     *
     * @when_noise_known{
     * compute_thresholds(const DWT2D::Coeffs&) const,
     * compute_thresholds()}
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
     * is the same size as @pref{coeffs} where each entry is the threshold for
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
     * is the same size as @pref{coeffs} where each entry is the threshold for
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
     * @equivalentto
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
     * @equivalentto
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
     * @equivalentto
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
     * @equivalentto
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
     * at the finest resolution
     * (i.e. @pref{coeffs,diagonal_detail(0),DWT2D::Coeffs::diagonal_detail}).
     *
     * The default stdev_function() is mad_stdev(), which gives a statistically
     * robust estimate of the standard deviation.
     *
     * Subclasses can override this function to change which coefficients
     * are used to esitmate the coefficient noise standard deviation.
     * Subclasses should change the standard deviation estimator by passing a
     * different estimator to the constructor.
     *
     * @param[in] coeffs The entire set of DWT coefficients.
     */
    virtual cv::Scalar compute_noise_stdev(const DWT2D::Coeffs& coeffs) const
    {
        return compute_stdev(coeffs.diagonal_detail(0));
    }

protected:
    /**
     * @brief Construct a new Shrinker object.
     *
     * @param[in] partition The scheme used to partition the coefficients.
     * @param[in] shrink_function The function used to shrink coefficients.
     * @param[in] stdev_function The function used to compute an estimate of the
     *                           standard deviation of coefficient noise.
     */
    Shrinker(
        Shrinker::Partition partition,
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
     * @equivalentto
     * @code{cpp}
     * Shrinker(partition, shrink_function, mad_stdev)
     * @endcode
     *
     * @param[in] partition The scheme used to partition the coefficients.
     * @param[in] shrink_function The function used to shrink coefficients.
     */
    Shrinker(
        Shrinker::Partition partition,
        ShrinkFunction shrink_function
    ) :
        Shrinker(
            partition,
            shrink_function,
            mad_stdev
        )
    {}

    /**
     * @overload
     *
     * @equivalentto
     * @code{cpp}
     * Shrinker(partition, make_shrink_function(shrink_function), stdev_function)
     * @endcode
     *
     * @tparam T
     * @tparam W
     * @param[in] partition The scheme used to partition the coefficients.
     * @param[in] shrink_function The function used to shrink coefficients.
     * @param[in] stdev_function The function used to compute an estimate of the
     *                           standard deviation of coefficient noise.
     */
    template <typename T, typename W>
    Shrinker(
        Shrinker::Partition partition,
        PrimitiveShrinkFunction<T, W> shrink_function,
        StdDevFunction stdev_function
    ) :
        Shrinker(
            partition,
            make_shrink_function(shrink_function),
            stdev_function
        )
    {}

    /**
     * @overload
     *
     * @equivalentto
     * @code{cpp}
     * Shrinker(partition, make_shrink_function(shrink_function), mad_stdev)
     * @endcode
     *
     * @tparam T
     * @tparam W
     * @param[in] partition The scheme used to partition the coefficients.
     * @param[in] shrink_function The function used to shrink coefficients.
     */
    template <typename T, typename W>
    Shrinker(
        Shrinker::Partition partition,
        PrimitiveShrinkFunction<T, W> shrink_function
    ) :
        Shrinker(
            partition,
            shrink_function,
            mad_stdev
        )
    {}

    //  ------------------------------------------------------------------------
    //  Subclass API
    /**
     * @brief Prepares for a call to shrink() or compute_thresholds().
     *
     * Most subclasses do **not** need to override this function.
     *
     * A subclass should override this function if it requires *temporary*
     * values that must be accessed by
     *  - shrink_subsets() **and** compute_subset_thresholds()
     *  - Or compute_subband_threshold() for **every** subband
     *  - Or compute_level_threshold() for **every** level
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
     * @brief Finishes a call to shrink() or compute_thresholds().
     *
     * Implementations should cleanup temporary data members created in
     * start_partitioning().  This function is guaranteed to be called by
     * shrink() and compute_thresholds() as long as start_partitioning() did
     * not throw an exception.  If necessary, std::current_exception() can be
     * used to determine if this function was called normally or because an
     * exception occured.  In the latter case, @pref{shrunk_coeffs} or
     * @pref{thresholds} may be empty depending on when the exception was
     * thrown.
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
     * @param[in] coeffs The DWT coefficients.
     * @param[in] levels The subset of levels to compute thresholds for.
     * @param[in] stdev The standard deviation of the coefficient noise.
     * @param[in] shrunk_coeffs The shrunken DWT coefficients.  This is empty when
     *                          called from compute_thresholds().
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
     * @brief Applies the shrink_function() to the given coefficients.
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
     * @param[in] mask Indicates which coefficients to shrink.
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
     * @brief Applies the stdev_function() to given array.
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
     * @brief Sets the standard deviation function.
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
    Shrinker::Partition _partition;
    ShrinkFunction _shrink_function;
    StdDevFunction _stdev_function;
    mutable std::mutex _mutex;
};
/** @} shrink*/

} // namespace cvwt

#endif  // CVWT_SHRINK_SHRINK_HPP

