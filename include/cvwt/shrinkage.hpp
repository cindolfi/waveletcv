#ifndef CVWT_SHRINKAGE_HPP
#define CVWT_SHRINKAGE_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <utility>
#include <nlopt.hpp>
#include "cvwt/wavelet.hpp"
#include "cvwt/dwt2d.hpp"
#include "cvwt/utils.hpp"

namespace cvwt
{
using StdDevFunction = std::function<cv::Scalar(cv::InputArray)>;
using ThresholdFunction = std::function<void(cv::InputArray, cv::OutputArray, cv::Scalar, cv::InputArray)>;
template <typename Value, typename Threshold>
using PrimitiveThresholdFunction = std::function<Value(Value, Threshold)>;

namespace internal
{
template <typename T1, typename T2, int N, cv::CmpTypes compare_type>
struct Compare
{
    using Pixel1 = cv::Vec<T1, N>;
    using Pixel2 = cv::Vec<T2, N>;
    using OutputPixel = cv::Vec<uchar, N>;

    constexpr bool compare(T1 x, T2 y) const
    {
        if constexpr (compare_type == cv::CMP_LT)
            return x < y;
        else if constexpr (compare_type == cv::CMP_LE)
            return x <= y;
        else if constexpr (compare_type == cv::CMP_GT)
            return x > y;
        else if constexpr (compare_type == cv::CMP_GE)
            return x >= y;
    }

    void operator()(
        cv::InputArray input_a,
        cv::InputArray input_b,
        cv::OutputArray output
    ) const
    {
        assert(input_a.channels() == N);
        assert(input_b.channels() == N);
        throw_if_comparing_different_sizes(input_a, input_b);

        output.create(input_a.size(), CV_8UC(N));
        auto a = input_a.getMat();
        auto b = input_b.getMat();
        auto result = output.getMat();
        a.forEach<Pixel1>(
            [&](const auto& x, const auto position) {
                auto y = b.at<Pixel2>(position);
                auto& z = result.at<OutputPixel>(position);
                for (int i = 0; i < N; ++i)
                    z[i] = 255 * compare(x[i], y[i]);
            }
        );
    }

    void operator()(
        cv::InputArray input_a,
        cv::InputArray input_b,
        cv::OutputArray output,
        cv::InputArray mask
    ) const
    {
        assert(input_a.channels() == N);
        assert(input_b.channels() == N);
        throw_if_comparing_different_sizes(input_a, input_b);
        throw_if_bad_mask_type(mask);
        if (mask.size() != input_a.size())
            throw_bad_size(
                "Wrong size mask. Got ", mask.size(), ", must be ", input_a.size(), "."
            );

        output.create(input_a.size(), CV_8UC(N));
        auto a = input_a.getMat();
        auto b = input_b.getMat();
        auto mask_matrix = mask.getMat();
        auto result = output.getMat();
        a.forEach<Pixel1>(
            [&](const auto& x, const auto position) {
                auto y = b.at<Pixel2>(position);
                if (mask_matrix.at<uchar>(position)) {
                    auto& z = result.at<OutputPixel>(position);
                    for (int i = 0; i < N; ++i)
                        z[i] = 255 * compare(x[i], y[i]);
                } else {
                    result.at<OutputPixel>(position) = 0;
                }
            }
        );
    }

    void throw_if_comparing_different_sizes(cv::InputArray a, cv::InputArray b) const
    {
        // if (a.channels() != b.channels())
        //     throw_bad_size(
        //         "Cannot compare matrices with different number of channels. ",
        //         "Got a.channels() = ", get_type_name(a.channels()),
        //         " and b.channels() = ", get_type_name(b.channels()), "."
        //     );

        if (a.size() != b.size())
            throw_bad_size(
                "Cannot compare matrices of different sizes. ",
                "Got a.size() = ", a.size(),
                " and b.size() = ", b.size(), "."
            );
    }
};


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

    void operator()(cv::InputArray input, cv::OutputArray output, cv::Scalar threshold, cv::InputArray mask) const
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
        PrimitiveThresholdFunction<T, W> threshold_function
    ) :
        _threshold_function(threshold_function)
    {}

    constexpr T operator()(T value, W threshold) const
    {
        return _threshold_function(value, threshold);
    }
private:
    PrimitiveThresholdFunction<T, W> _threshold_function;
};

template <typename T, int N, typename W>
struct WrappedThreshold : public Threshold<T, N, WrappedThresholdFunctor<T, W>>
{
    WrappedThreshold(
        PrimitiveThresholdFunction<T, W> threshold_function
    ) :
        Threshold<T, N, WrappedThresholdFunctor<T, W>>(
            WrappedThresholdFunctor(threshold_function)
        )
    {}
};
}   // namespace internal






//  ============================================================================
//  Low Level API
//  ============================================================================
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


//  ----------------------------------------------------------------------------
//  Thresholding
//  ----------------------------------------------------------------------------
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

/**
 * @brief Multichannel masked hard threshold
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
 * @return ThresholdFunction
 */
template <typename ThresholdFunctor>
ThresholdFunction make_threshold_function()
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
 * @return ThresholdFunction
 */
template <typename T, typename W>
ThresholdFunction make_threshold_function(
    PrimitiveThresholdFunction<T, W> threshold_function
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

void less_than(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask = cv::noArray()
);

void less_than_or_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask = cv::noArray()
);

void greater_than(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask = cv::noArray()
);

void greater_than_or_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask = cv::noArray()
);

void compare(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::CmpTypes compare_type,
    cv::InputArray mask = cv::noArray()
);

//  ----------------------------------------------------------------------------
//  Shrink
//  ----------------------------------------------------------------------------
void shrink_globally(
    DWT2D::Coeffs& coeffs,
    cv::Scalar threshold,
    ThresholdFunction threshold_function,
    const cv::Range& levels = cv::Range::all()
);

void shrink_levels(
    DWT2D::Coeffs& coeffs,
    cv::InputArray level_thresholds,
    ThresholdFunction threshold_function,
    const cv::Range& levels = cv::Range::all()
);

void shrink_subbands(
    DWT2D::Coeffs& coeffs,
    cv::InputArray subband_thresholds,
    ThresholdFunction threshold_function,
    const cv::Range& levels = cv::Range::all()
);

//  ----------------------------------------------------------------------------
//  Universal
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
cv::Scalar compute_universal_threshold(
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
 * @see visu_shrink_threshold()
 * @see https://computing.llnl.gov/sites/default/files/jei2001.pdf
 *
 * @param coeffs The discrete wavelet transform coefficients.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @return cv::Scalar
 */
cv::Scalar compute_universal_threshold(
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
 * @see visu_shrink_threshold()
 *
 * @param details The discrete wavelet transform detail coefficients.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @return cv::Scalar
 */
cv::Scalar compute_universal_threshold(
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
 * @see visu_shrink_threshold()
 *
 * @param details The discrete wavelet transform detail coefficients.
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which input locations are used in the computation.
 * @param stdev The standard deviations of the detail coefficients channels.
 * @return cv::Scalar
 */
cv::Scalar compute_universal_threshold(
    cv::InputArray details,
    cv::InputArray mask,
    const cv::Scalar& stdev = cv::Scalar::all(1.0)
);








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
    enum Partition {
        GLOBALLY = 0,
        LEVELS = 1,
        SUBBANDS = 2,
        SUBSETS = 3,
    };

public:
    Shrink() = delete;
    Shrink(const Shrink& other) = default;
    Shrink(Shrink&& other) = default;

    //  ------------------------------------------------------------------------
    //  Getters & Setters
    Shrink::Partition partition() const { return _partition; }
    ThresholdFunction threshold_function() const { return _threshold_function; }
    StdDevFunction stdev_function() const { return _stdev_function; }

    //  ------------------------------------------------------------------------
    //  Shrink
    //  all levels
    // DWT2D::Coeffs shrink(const DWT2D::Coeffs& coeffs) const
    // {
    //     return shrink(coeffs, cv::Range::all(), compute_stdev(coeffs));
    // }

    DWT2D::Coeffs shrink(const DWT2D::Coeffs& coeffs, cv::OutputArray thresholds = cv::noArray()) const
    {
        return shrink(coeffs, cv::Range::all(), compute_stdev(coeffs), thresholds);
    }

    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& stdev,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, cv::Range::all(), stdev, thresholds);
    }

    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, cv::Range::all(), compute_stdev(coeffs), thresholds);
    }

    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        const cv::Scalar& stdev,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, cv::Range::all(), stdev, thresholds);
    }

    //  int levels
    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        int levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, cv::Range(0, levels), thresholds);
    }

    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& stdev,
        int levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, cv::Range(0, levels), stdev, thresholds);
    }

    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        int levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, cv::Range(0, levels), compute_stdev(coeffs), thresholds);
    }

    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        const cv::Scalar& stdev,
        int levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, cv::Range(0, levels), stdev, thresholds);
    }

    //  cv::Range levels
    DWT2D::Coeffs shrink(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        return shrink(coeffs, levels, compute_stdev(coeffs), thresholds);
    }

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

    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        const cv::Range& levels,
        cv::OutputArray thresholds = cv::noArray()
    ) const
    {
        shrink(coeffs, shrunk_coeffs, levels, compute_stdev(coeffs), thresholds);
    }

    void shrink(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev,
        cv::OutputArray thresholds = cv::noArray()
    ) const;

    //  ------------------------------------------------------------------------
    //  Compute Thresholds
    cv::Mat4d compute_thresholds(const DWT2D::Coeffs& coeffs) const
    {
        return compute_thresholds(coeffs, cv::Range::all(), compute_stdev(coeffs));
    }

    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& stdev
    ) const
    {
        return compute_thresholds(coeffs, cv::Range::all(), stdev);
    }

    cv::Mat4d compute_thresholds(const DWT2D::Coeffs& coeffs, int levels) const
    {
        return compute_thresholds(coeffs, cv::Range(0, levels), compute_stdev(coeffs));
    }

    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Scalar& stdev,
        int levels
    ) const
    {
        return compute_thresholds(coeffs, cv::Range(0, levels), stdev);
    }

    cv::Mat4d compute_thresholds(const DWT2D::Coeffs& coeffs, const cv::Range& levels) const
    {
        return compute_thresholds(coeffs, levels, compute_stdev(coeffs));
    }

    cv::Mat4d compute_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    //  ------------------------------------------------------------------------
    //  Expand Thresholds
    cv::Mat expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& thresholds
    ) const
    {
        return expand_thresholds(coeffs, thresholds, cv::Range::all());
    }

    cv::Mat expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& thresholds,
        int levels
    ) const
    {
        return expand_thresholds(coeffs, thresholds, cv::Range(0, levels));
    }

    cv::Mat expand_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& thresholds,
        const cv::Range& levels
    ) const;

    //  ------------------------------------------------------------------------
    //  Compute Standard Deviation
    virtual cv::Scalar compute_stdev(const DWT2D::Coeffs& coeffs) const
    {
        return compute_stdev(coeffs.diagonal_detail());
    }

    cv::Scalar compute_stdev(cv::InputArray detail_coeffs) const
    {
        return _stdev_function(detail_coeffs);
    }

    //  ------------------------------------------------------------------------
    //  Functor Interface
    DWT2D::Coeffs operator()(
        const DWT2D::Coeffs& coeffs,
        auto&&... args
    ) const
    {
        return shrink(coeffs, std::forward<decltype(args)>(args)...);
    }

    void operator()(
        const DWT2D::Coeffs& coeffs,
        DWT2D::Coeffs& shrunk_coeffs,
        auto&&... args
    ) const
    {
        shrink(coeffs, shrunk_coeffs, std::forward<decltype(args)>(args)...);
    }

protected:
    Shrink(
        Shrink::Partition partition,
        ThresholdFunction threshold_function,
        StdDevFunction stdev_function
    ) :
        _partition(partition),
        _threshold_function(threshold_function),
        _stdev_function(stdev_function)
    {}

    Shrink(
        Shrink::Partition partition,
        ThresholdFunction threshold_function
    ) :
        Shrink(
            partition,
            threshold_function,
            mad_stdev
        )
    {}

    template <typename T, typename W>
    Shrink(
        Shrink::Partition partition,
        PrimitiveThresholdFunction<T, W> threshold_function,
        StdDevFunction stdev_function
    ) :
        Shrink(
            partition,
            make_threshold_function(threshold_function),
            stdev_function
        )
    {}

    template <typename T, typename W>
    Shrink(
        Shrink::Partition partition,
        PrimitiveThresholdFunction<T, W> threshold_function
    ) :
        Shrink(
            partition,
            threshold_function,
            mad_stdev
        )
    {}

    //  ------------------------------------------------------------------------
    //  Subclass API
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
     * are used by both compute_subset_thresholds() and shrink_custom().
     * Another use case is computing global statistics that are used by
     * algorithms that support multiple partitioning schemes (e.g. by over
     * overriding both compute_level_threshold() and compute_subband_threshold()).
     *
     * Since these are `const`, any data members set in this function must be
     * marked mutable and must be cleaned up in finish().
     *
     * @param coeffs
     * @param stdev
     */
    virtual void start(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const
    {}

    /**
     * @brief Finish a call to shrink() or compute_thresholds()
     *
     * @param coeffs
     * @param shrunk_coeffs
     * @param stdev
     * @param thresholds
     */
    virtual void finish(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev,
        const DWT2D::Coeffs& shrunk_coeffs,
        const cv::Mat4d& thresholds
    ) const
    {}

    cv::Mat4d compute_partition_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    virtual void shrink_subsets(
        DWT2D::Coeffs& coeffs,
        const cv::Mat4d& thresholds,
        const cv::Range& levels
    ) const;

    virtual cv::Scalar compute_global_threshold(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    virtual cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& stdev
    ) const;

    virtual cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& stdev
    ) const;

    virtual cv::Mat4d compute_subset_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    virtual void expand_subset_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat4d& thresholds,
        const cv::Range& levels,
        cv::Mat& expanded_thresholds
    ) const;

    void threshold(
        cv::InputOutputArray coeffs,
        const cv::Scalar& threshold,
        cv::InputArray mask = cv::noArray()
    ) const
    {
        _threshold_function(coeffs, coeffs, threshold, mask);
    }

    void threshold(
        cv::InputArray coeffs,
        cv::OutputArray thresholded_coeffs,
        const cv::Scalar& threshold,
        cv::InputArray mask = cv::noArray()
    ) const
    {
        _threshold_function(coeffs, thresholded_coeffs, threshold, mask);
    }

    void set_threshold_function(ThresholdFunction threshold_function) { _threshold_function = threshold_function; }
    void set_stdev_function(StdDevFunction stdev_function) { _stdev_function = stdev_function; }

    //  ------------------------------------------------------------------------
    //  Helpers
    cv::Mat4d compute_level_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    cv::Mat4d compute_subband_thresholds(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const;

    static cv::Range resolve_levels(const cv::Range& levels, const DWT2D::Coeffs& coeffs)
    {
        return (levels == cv::Range::all()) ? cv::Range(0, coeffs.levels()) : levels;
    }

private:
    Shrink::Partition _partition;
    ThresholdFunction _threshold_function;
    StdDevFunction _stdev_function;
    mutable std::mutex _mutex;
};


//  ----------------------------------------------------------------------------
//  Universal / VisuShrink
//  ----------------------------------------------------------------------------
/**
 * @brief Shrinkage using the universal threshold.
 *
 */
class UniversalShrink : public Shrink
{
public:
    UniversalShrink(
        ThresholdFunction threshold_function
    ) :
        UniversalShrink(
            Shrink::GLOBALLY,
            threshold_function
        )
    {}

    template <typename T, typename W>
    UniversalShrink(
        PrimitiveThresholdFunction<T, W> threshold_function
    ) :
        UniversalShrink(
            make_threshold_function(threshold_function)
        )
    {}

    UniversalShrink(
        Shrink::Partition partition,
        ThresholdFunction threshold_function
    ) :
        Shrink(
            partition,
            threshold_function
        )
    {}

    template <typename T, typename W>
    UniversalShrink(
        Shrink::Partition partition,
        PrimitiveThresholdFunction<T, W> threshold_function
    ) :
        UniversalShrink(
            partition,
            make_threshold_function(threshold_function)
        )
    {}

    UniversalShrink(
        ThresholdFunction threshold_function,
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
        PrimitiveThresholdFunction<T, W> threshold_function,
        StdDevFunction stdev_function
    ) :
        UniversalShrink(
            make_threshold_function(threshold_function),
            stdev_function
        )
    {}

    UniversalShrink(
        Shrink::Partition partition,
        ThresholdFunction threshold_function,
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
        PrimitiveThresholdFunction<T, W> threshold_function,
        StdDevFunction stdev_function
    ) :
        UniversalShrink(
            partition,
            make_threshold_function(threshold_function),
            stdev_function
        )
    {}

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
};

//  ----------------------------------------------------------------------------
//  VisuShrink Functional API
//  ----------------------------------------------------------------------------
/**
 * @brief Shrink detail coeffcients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs visu_shrink(const DWT2D::Coeffs& coeffs);

/**
 * @brief Shrink detail coeffcients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void visu_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief Shrink detail coeffcients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs visu_shrink(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrink detail coeffcients using the VisuShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
void visu_shrink(DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels);


//  ----------------------------------------------------------------------------
//  SureShrink
//  ----------------------------------------------------------------------------
/**
 * @brief
 *
 * https://computing.llnl.gov/sites/default/files/jei2001.pdf
 */
class SureShrink : public Shrink
{
public:
    enum Variant {
        STRICT,
        HYBRID,
    };

    enum Optimizer {
        NELDER_MEAD,
        BRUTE_FORCE,
    };

    struct OptimizerStopConditions
    {
        double threshold_rel_tol = 1e-8;
        double threshold_abs_tol = 0.0;
        double risk_rel_tol = 1e-8;
        double risk_abs_tol = 0.0;
        double max_time = 10.0;
        int max_evals = 0;
    };

public:
    SureShrink() :
        SureShrink(
            Shrink::SUBBANDS,
            SureShrink::HYBRID,
            SureShrink::NELDER_MEAD
        )
    {}

    SureShrink(SureShrink::Variant variant) :
        SureShrink(
            Shrink::SUBBANDS,
            variant,
            SureShrink::NELDER_MEAD
        )
    {}

    SureShrink(Shrink::Partition partition) :
        SureShrink(
            partition,
            SureShrink::HYBRID,
            SureShrink::NELDER_MEAD
        )
    {}

    SureShrink(Shrink::Partition partition, SureShrink::Variant variant) :
        SureShrink(
            partition,
            variant,
            SureShrink::NELDER_MEAD
        )
    {}

    SureShrink(
        Shrink::Partition partition,
        SureShrink::Variant variant,
        SureShrink::Optimizer optimizer
    ) :
        Shrink(
            partition,
            soft_threshold,
            mad_stdev
        ),
        _variant(variant),
        _optimizer(optimizer)
    {}

    Variant variant() const { return _variant; }
    Optimizer optimizer() const { return _optimizer; }
    OptimizerStopConditions optimizer_stop_conditions() const { return _optimizer_stop_conditions; }
    void set_optimizer_stop_conditions(const OptimizerStopConditions& stop_conditions)
    {
        _optimizer_stop_conditions = stop_conditions;
    }

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
     * algorithm that chooses the compute_universal_threshold() when
     * \f{equation}{
     *     \frac{1}{N} \sum_{n = 1}^{N} \left( \frac{w_n}{\sigma} - 1 \right)^2
     *     \le
     *     \frac{\left(\log_2(N)\right)^{3/2}}{\sqrt{N}}
     * \f}
     * and \f$\lambda_{SURE}\f$ otherwise.
     * Pass `SureShrink::HYBRID` as the `variant` argument to use
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
     * @see mad_stdev()
     * @see compute_universal_threshold()
     *
     * @param detail_coeffs The detail coefficients.
     * @param stdev The standard deviations of the detail coefficients channels.
     * @param variant The variant of the SureShrink algorithm.
     * @param algorithm The optimization algorithm used to compute the SURE threshold.
     * @return cv::Scalar
     */
    cv::Scalar compute_sure_threshold(
        cv::InputArray detail_coeffs,
        const cv::Scalar& stdev
    ) const;

    cv::Scalar compute_sure_threshold(
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
        return compute_sure_threshold(coeffs, coeffs.detail_mask(levels), stdev);
    }

    cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_sure_threshold(detail_coeffs, stdev);
    }

    cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_sure_threshold(detail_coeffs, stdev);
    }

private:
    SureShrink::Variant _variant;
    SureShrink::Optimizer _optimizer;
    OptimizerStopConditions _optimizer_stop_conditions;
};

namespace internal
{
template <typename T>
double nlopt_sure_threshold_objective(
    const std::vector<double>& x,
    std::vector<double>& grad,
    void* f_data
);


template <typename T>
struct SingleChannelComputeSureThreshold
{
    double operator()(
        cv::InputArray input,
        double stdev,
        SureShrink::Optimizer optimizer,
        SureShrink::Variant variant,
        const SureShrink::OptimizerStopConditions& stop_conditions
    ) const
    {
        auto input_matrix = input.getMat();
        if (stdev != 1.0)
            input_matrix = input_matrix / stdev;

        double threshold;
        if (variant == SureShrink::HYBRID && use_universal_threshold(input_matrix)) {
            threshold = compute_universal_threshold(input_matrix.total(), 1.0)[0];
        } else {
            if (optimizer == SureShrink::BRUTE_FORCE) {
                threshold = compute_sure_threshold_using_brute_force(input_matrix);
            } else {
                threshold = compute_sure_threshold_using_nlopt(
                    input_matrix,
                    to_nlopt_algorithm(optimizer),
                    stop_conditions
                );
            }
        }

        return stdev * threshold;
    }

    double sure_risk(cv::InputArray input, double threshold, double stdev) const
    {
        auto input_matrix = input.getMat();
        if (stdev != 1.0) {
            input_matrix = input_matrix / stdev;
            threshold = threshold / stdev;
        }

        return compute_sure_risk(input_matrix, threshold);
    }

    friend double nlopt_sure_threshold_objective<T>(
        const std::vector<double>& x,
        std::vector<double>& grad,
        void* f_data
    );

private:
    nlopt::algorithm to_nlopt_algorithm(SureShrink::Optimizer optimizer) const
    {
        switch (optimizer) {
            case SureShrink::NELDER_MEAD:
                return nlopt::algorithm::LN_NELDERMEAD;
        }

        assert(false);
        return nlopt::algorithm::LN_NELDERMEAD;
    }

    struct NLObjectiveData
    {
        const SingleChannelComputeSureThreshold<T>* self;
        cv::Mat channel;
    };

    double compute_sure_threshold_using_nlopt(
        const cv::Mat& channel,
        nlopt::algorithm algorithm,
        const SureShrink::OptimizerStopConditions& stop_conditions
    ) const
    {
        NLObjectiveData data(this, channel);

        nlopt::opt optimizer(algorithm, 1);
        optimizer.set_min_objective(nlopt_sure_threshold_objective<T>, &data);

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

    double compute_sure_threshold_using_brute_force(const cv::Mat& channel) const
    {
        cv::Mat flat_channel;
        flatten(channel, flat_channel);
        flat_channel = flat_channel;

        std::vector<T> risks(flat_channel.total());
        flat_channel.forEach<T>(
            [&](const auto& pixel, auto index) {
                risks[index[1]] = compute_sure_risk(flat_channel, pixel);
            }
        );

        auto threshold_index = std::ranges::distance(
            risks.begin(),
            std::ranges::min_element(risks)
        );

        return std::fabs(flat_channel.at<T>(threshold_index));
    }

    bool use_universal_threshold(const cv::Mat& channel) const
    {
        int n = channel.total();
        auto universal_test_statistic = std::pow(std::log2(n), 1.5) / std::sqrt(n);
        auto mse = (cv::sum(channel * channel - 1) / n)[0];

        auto result = mse < universal_test_statistic;

        CV_LOG_DEBUG(
            NULL,
            (result ? "using universal threshold" : "using SURE threshold")
            << "  mse = " << mse
            << "  universal_test_statistic = " << universal_test_statistic
        );

        return result;
    }

    double compute_sure_risk(const cv::Mat& x, double threshold) const
    {
        assert(x.channels() == 1);
        auto abs_x = cv::abs(x);
        auto clamped_abs_x = cv::min(abs_x, threshold);
        return x.total()
            + cv::sum(clamped_abs_x.mul(clamped_abs_x))[0]
            - 2 * cv::countNonZero(abs_x <= threshold);
    }

    double compute_sure_risk(const cv::Mat& x, double threshold, double stdev) const
    {
        return compute_sure_risk(x / stdev, threshold / stdev);
    }
};

template <typename T>
double nlopt_sure_threshold_objective(
    const std::vector<double>& x,
    std::vector<double>& grad,
    void* f_data
)
{
    auto data = static_cast<SingleChannelComputeSureThreshold<T>::NLObjectiveData*>(f_data);
    return data->self->compute_sure_risk(data->channel, x[0]);
};



template <typename T, int N>
struct ComputeSureThreshold
{
    cv::Scalar operator()(
        cv::InputArray input,
        const cv::Scalar& stdev,
        SureShrink::Optimizer optimizer,
        SureShrink::Variant variant,
        const SureShrink::OptimizerStopConditions& stop_conditions
    ) const
    {
        SingleChannelComputeSureThreshold<T> compute_single_channel_sure_threshold;

        assert(input.channels() == N);
        cv::Mat channels[N];
        cv::split(input.getMat(), channels);
        cv::Scalar result;
        for (int i = 0; i < N; ++i) {
            CV_LOG_INFO(NULL, "computing channel " << i);
            result[i] = compute_single_channel_sure_threshold(
                channels[i],
                stdev[i],
                optimizer,
                variant,
                stop_conditions
            );
        }

        return result;
    }

    cv::Scalar operator()(
        cv::InputArray input,
        cv::InputArray mask,
        const cv::Scalar& stdev,
        SureShrink::Optimizer optimizer,
        SureShrink::Variant variant,
        const SureShrink::OptimizerStopConditions& stop_conditions
    ) const
    {
        assert(input.channels() == N);
        cv::Mat masked_input;
        CollectMasked<T, N>()(input, masked_input, mask);
        return this->operator()(masked_input, stdev, optimizer, variant, stop_conditions);
    }

    void sure_risk(
        cv::InputArray input,
        const cv::Scalar& threshold,
        const cv::Scalar& stdev,
        cv::Scalar& result
    ) const
    {
        SingleChannelComputeSureThreshold<T> compute_single_channel_sure_threshold;

        assert(input.channels() == N);

        cv::Mat channels[N];
        cv::split(input.getMat(), channels);
        for (int i = 0; i < N; ++i)
            result[i] = compute_single_channel_sure_threshold.sure_risk(
                channels[i],
                threshold[i],
                stdev[i]
            );
    }
};
}   // namespace internal


//  ----------------------------------------------------------------------------
//  SureShrink Functional API
//  ----------------------------------------------------------------------------
/**
 * @brief Shrink detail coeffcients using the Hybrid SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs sure_shrink(const DWT2D::Coeffs& coeffs);

/**
 * @brief Shrink detail coeffcients using the Hybrid SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void sure_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief Shrink detail coeffcients using the Hybrid SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs sure_shrink(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrink detail coeffcients using the Hybrid SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
void sure_shrink(DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels);

/**
 * @brief Shrink detail coeffcients using the Hybrid SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs);

/**
 * @brief Shrink detail coeffcients using the Hybrid SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief Shrink detail coeffcients using the Hybrid SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrink detail coeffcients using the Hybrid SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
void sure_shrink_levelwise(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& shrunk_coeffs,
    int levels
);


//  ----------------------------------------------------------------------------
//  Bayes Shrink
//  ----------------------------------------------------------------------------
/**
 * @brief
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
        PrimitiveThresholdFunction<T, W> threshold_function
    ) :
        BayesShrink(partition, make_threshold_function(threshold_function))
    {}

    BayesShrink(
        Shrink::Partition partition,
        ThresholdFunction threshold_function
    ) :
        Shrink(
            partition,
            threshold_function,
            mad_stdev
        )
    {}

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
        cv::Range d = (levels == cv::Range::all()) ? levels : cv::Range(levels.start, levels.end - 1);
        return compute_bayes_threshold(coeffs, coeffs.detail_mask(d), stdev);
        // return compute_bayes_threshold(coeffs, coeffs.detail_mask(levels), stdev);
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

namespace internal
{
template <typename T, int CHANNELS>
struct ComputeBayesThreshold
{
    cv::Scalar operator()(cv::InputArray input, const cv::Scalar& stdev) const
    {
        throw_if_empty(input);
        assert(input.channels() == CHANNELS);
        if constexpr (CHANNELS == 1) {
            return compute_single_channel_threshold(input.getMat(), stdev[0]);
        } else {
            cv::Mat channels[CHANNELS];
            cv::split(input.getMat(), channels);
            cv::Scalar result;
            for (int i = 0; i < CHANNELS; ++i)
                result[i] = compute_single_channel_threshold(channels[i], stdev[i]);

            return result;
        }
    }

    cv::Scalar operator()(
        cv::InputArray input,
        cv::InputArray mask,
        const cv::Scalar& stdev
    ) const
    {
        cv::Mat masked_input;
        CollectMasked<T, CHANNELS>()(input, masked_input, mask);
        return this->operator()(masked_input, stdev);
    }

private:
    double compute_single_channel_threshold(const cv::Mat& array, double stdev) const
    {
        assert(array.channels() == 1);
        double noise_variance = stdev * stdev;
        double observation_variance = cv::sum(array.mul(array))[0] / array.total();

        if (noise_variance >= observation_variance) {
            return maximum_abs_value(array);
        } else {
            auto signal_stdev = std::sqrt(observation_variance - noise_variance);
            return noise_variance / signal_stdev;
        }
    }
};
}   // namespace internal

//  ----------------------------------------------------------------------------
//  BayesShrink Functional API
//  ----------------------------------------------------------------------------
DWT2D::Coeffs bayes_shrink(const DWT2D::Coeffs& coeffs);
void bayes_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

} // namespace cvwt

#endif  // CVWT_SHRINKAGE_HPP

