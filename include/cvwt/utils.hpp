#ifndef CVWT_UTILS_HPP
#define CVWT_UTILS_HPP

#include <span>
#include <ranges>
#include <memory>
#include <string>
#include <atomic>
#include <type_traits>
#include <iostream>
#include <opencv2/core.hpp>
#include "cvwt/exception.hpp"

namespace cvwt
{
namespace internal
{
template <std::floating_point T>
inline constexpr T sqrt_epsilon = std::sqrt(std::numeric_limits<T>::epsilon());

template <typename... T>
using promote_types = decltype((std::declval<T>() + ...));
}

/**
 * @brief Returns true if the floating point value is approximately zero.
 *
 * @param[in] x
 * @param[in] absolute_tolerance
 */
template <typename T>
requires std::floating_point<std::remove_cvref_t<T>>
bool is_approx_zero(T x, double absolute_tolerance)
{
    return std::abs(x) < static_cast<T>(absolute_tolerance);
}

/**
 * @overload
 */
template <typename T>
requires std::floating_point<std::remove_cvref_t<T>>
bool is_approx_zero(T x)
{
    return is_approx_zero(x, internal::sqrt_epsilon<T>);
}

/**
 * @brief Returns true if the two floating point values are approximately equal.
 *
 * @param[in] x
 * @param[in] y
 * @param[in] relative_tolerance
 * @param[in] zero_absolute_tolerance
 */
template <typename T1, typename T2>
requires std::floating_point<std::remove_cvref_t<T1>>
    && std::floating_point<std::remove_cvref_t<T2>>
constexpr bool is_approx_equal(
    T1 x,
    T2 y,
    double relative_tolerance,
    double zero_absolute_tolerance
)
{
    using T = internal::promote_types<T1, T2>;

    //  https://www.reidatcheson.com/floating%20point/comparison/2019/03/20/floating-point-comparison.html
    if (x == 0.0)
        return is_approx_zero(y, zero_absolute_tolerance);

    if (y == 0.0)
        return is_approx_zero(x, zero_absolute_tolerance);

    T min = std::max(
        std::min(std::abs(x), std::abs(y)),
        std::numeric_limits<T>::min()
    );
    return std::abs(x - y) / min < static_cast<T>(relative_tolerance);
}

/**
 * @overload
 */
template <typename T1, typename T2>
requires std::floating_point<std::remove_cvref_t<T1>>
    && std::floating_point<std::remove_cvref_t<T2>>
constexpr bool is_approx_equal(T1 x, T2 y, double relative_tolerance)
{
    using T = internal::promote_types<T1, T2>;

    return is_approx_equal(
        x, y,
        static_cast<T>(relative_tolerance),
        internal::sqrt_epsilon<T>
    );
}

/**
 * @overload
 */
template <typename T1, typename T2>
requires std::floating_point<std::remove_cvref_t<T1>>
    && std::floating_point<std::remove_cvref_t<T2>>
constexpr bool is_approx_equal(T1 x, T2 y)
{
    using T = internal::promote_types<T1, T2>;

    return is_approx_equal(
        x, y,
        internal::sqrt_epsilon<T>,
        internal::sqrt_epsilon<T>
    );
}


namespace internal
{
std::string get_type_name(int type);
cv::Scalar set_unused_channels(const cv::Scalar& scalar, int channels, double value = 0.0);
struct Index {
    int row;
    int col;
};
inline Index unravel_index(const cv::Mat& array, int flat_index)
{
    return {
        .row = flat_index / array.cols,
        .col = flat_index % array.cols
    };
}

template <template <typename T, int CHANNELS, auto ...> typename Functor, auto ...TemplateArgs>
auto dispatch_on_pixel_type(int type, auto&&... args)
{
    switch (type) {
        //  32 bit floating point
        case CV_32FC1: return Functor<float, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_32FC2: return Functor<float, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_32FC3: return Functor<float, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_32FC4: return Functor<float, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64FC1: return Functor<double, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_64FC2: return Functor<double, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_64FC3: return Functor<double, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_64FC4: return Functor<double, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32SC1: return Functor<int, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_32SC2: return Functor<int, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_32SC3: return Functor<int, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_32SC4: return Functor<int, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16SC1: return Functor<short, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_16SC2: return Functor<short, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_16SC3: return Functor<short, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_16SC4: return Functor<short, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16UC1: return Functor<ushort, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_16UC2: return Functor<ushort, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_16UC3: return Functor<ushort, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_16UC4: return Functor<ushort, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8SC1: return Functor<char, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_8SC2: return Functor<char, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_8SC3: return Functor<char, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_8SC4: return Functor<char, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8UC1: return Functor<uchar, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_8UC2: return Functor<uchar, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_8UC3: return Functor<uchar, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        case CV_8UC4: return Functor<uchar, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
    }

    throw_not_implemented(
        "Dispatch on pixel type ", get_type_name(type), " is not implemented."
    );
}

template <template <typename T, auto ...> class Functor, auto ...TemplateArgs>
auto dispatch_on_pixel_depth(int type, auto&&... args)
{
    switch (CV_MAT_DEPTH(type)) {
        //  32 bit floating point
        case CV_32F: return Functor<float, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return Functor<double, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return Functor<int, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return Functor<short, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return Functor<ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return Functor<char, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return Functor<uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
    }

    throw_not_implemented(
        "Dispatch on pixel ", get_type_name(type), " is not implemented."
    );
}

template <template <typename T, auto ...> class Functor, auto ...TemplateArgs, typename ...ConstructorArgs>
auto dispatch_on_pixel_depth(std::tuple<ConstructorArgs...>&& constructor_args, int type, auto&&... args)
{
    auto create_functor = []<typename T>(auto... cargs) {
        return Functor<T, TemplateArgs...>(cargs...);
    };

    switch (CV_MAT_DEPTH(type)) {
        //  32 bit floating point
        case CV_32F: {
            auto functor = std::apply(
                create_functor.template operator()<float>,
                constructor_args
            );
            return functor(std::forward<decltype(args)>(args)...);
        }
        //  64 bit floating point
        case CV_64F: {
            auto functor = std::apply(
                create_functor.template operator()<double>,
                constructor_args
            );
            return functor(std::forward<decltype(args)>(args)...);
        }
        //  32 bit signed integer
        case CV_32S: {
            auto functor = std::apply(
                create_functor.template operator()<int>,
                constructor_args
            );
            return functor(std::forward<decltype(args)>(args)...);
        }
        //  16 bit signed integer
        case CV_16S: {
            auto functor = std::apply(
                create_functor.template operator()<short>,
                constructor_args
            );
            return functor(std::forward<decltype(args)>(args)...);
        }
        //  16 bit unsigned integer
        case CV_16U: {
            auto functor = std::apply(
                create_functor.template operator()<ushort>,
                constructor_args
            );
            return functor(std::forward<decltype(args)>(args)...);
        }
        //  8 bit signed integer
        case CV_8S: {
            auto functor = std::apply(
                create_functor.template operator()<char>,
                constructor_args
            );
            return functor(std::forward<decltype(args)>(args)...);
        }
        //  8 bit unsigned integer
        case CV_8U: {
            auto functor = std::apply(
                create_functor.template operator()<uchar>,
                constructor_args
            );
            return functor(std::forward<decltype(args)>(args)...);
        }
    }

    throw_not_implemented(
        "Dispatch on pixel depth ", get_type_name(type), " is not implemented."
    );
}

template <template <typename T1, typename T2, auto ...> class Functor, auto ...TemplateArgs>
auto dispatch_on_pixel_depths(int type1, int type2, auto&&... args)
{
    int depth1 = CV_MAT_DEPTH(type1);
    int depth2 = CV_MAT_DEPTH(type2);
    switch (depth1) {
    //  32 bit floating point
    case CV_32F:
        switch (depth2) {
        //  32 bit floating point
        case CV_32F: return Functor<float, float, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return Functor<float, double, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return Functor<float, int, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return Functor<float, short, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return Functor<float, ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return Functor<float, char, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return Functor<float, uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        }
        break;
    //  64 bit floating point
    case CV_64F:
        switch (depth2) {
        //  32 bit floating point
        case CV_32F: return Functor<double, float, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return Functor<double, double, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return Functor<double, int, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return Functor<double, short, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return Functor<double, ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return Functor<double, char, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return Functor<double, uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        }
        break;
    //  32 bit signed integer
    case CV_32S:
        switch (depth2) {
        //  32 bit floating point
        case CV_32F: return Functor<int, float, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return Functor<int, double, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return Functor<int, int, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return Functor<int, short, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return Functor<int, ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return Functor<int, char, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return Functor<int, uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        }
        break;
    //  16 bit signed integer
    case CV_16S:
        switch (depth2) {
        //  32 bit floating point
        case CV_32F: return Functor<short, float, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return Functor<short, double, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return Functor<short, int, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return Functor<short, short, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return Functor<short, ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return Functor<short, char, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return Functor<short, uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        }
        break;
    //  16 bit unsigned integer
    case CV_16U:
        switch (depth2) {
        //  32 bit floating point
        case CV_32F: return Functor<ushort, float, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return Functor<ushort, double, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return Functor<ushort, int, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return Functor<ushort, short, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return Functor<ushort, ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return Functor<ushort, char, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return Functor<ushort, uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        }
        break;
    //  8 bit signed integer
    case CV_8S:
        switch (depth2) {
        //  32 bit floating point
        case CV_32F: return Functor<char, float, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return Functor<char, double, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return Functor<char, int, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return Functor<char, short, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return Functor<char, ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return Functor<char, char, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return Functor<char, uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        }
        break;
    //  8 bit unsigned integer
    case CV_8U:
        switch (depth2) {
        //  32 bit floating point
        case CV_32F: return Functor<uchar, float, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return Functor<uchar, double, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return Functor<uchar, int, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return Functor<uchar, short, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return Functor<uchar, ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return Functor<uchar, char, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return Functor<uchar, uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...);
        }
        break;
    }

    throw_not_implemented(
        "Dispatch on pixel depths ",
        get_type_name(type1),
        " and ",
        get_type_name(type2),
        " is not implemented."
    );
}
}   // namespace internal

/**
 * @name Utilities
 * @{
 */
/**
 * @brief Collect values indicated by the given mask.
 *
 * @param[in] array
 * @param[out] collected
 * @param[in] mask
 */
void collect_masked(cv::InputArray array, cv::OutputArray collected, cv::InputArray mask);

/**
 * @brief Returns true if all values two matrices are equal.
 *
 * @param[in] a
 * @param[in] b
 */
bool matrix_equals(cv::InputArray a, cv::InputArray b);


/**
 * @brief Returns true if all corresponding values in two matrices are approximately equal.
 *
 * @param[in] a
 * @param[in] b
 * @param[in] relative_tolerance
 * @param[in] zero_absolute_tolerance
 */
bool approx_equals(
    cv::InputArray a,
    cv::InputArray b,
    double relative_tolerance,
    double zero_absolute_tolerance
);
/**
 * @overload
 */
bool approx_equals(cv::InputArray a, cv::InputArray b, double relative_tolerance);
/**
 * @overload
 */
bool approx_equals(cv::InputArray a, cv::InputArray b);

/**
 * @brief Returns true if all corresponding values in two matrices are approximately equal.
 *
 * @param[in] a
 * @param[in] absolute_tolerance
 */
bool approx_zeros(cv::InputArray a, double absolute_tolerance);
/**
 * @overload
 */
bool approx_zeros(cv::InputArray a);


/**
 * @brief Returns true if two matrices refer to the same data and are equal.
 *
 * @param[in] a
 * @param[in] b
 */
bool identical(const cv::Mat& a, const cv::Mat& b);

/**
 * @brief Returns true if the two matrices refer to the same data.
 *
 * @param[in] a
 * @param[in] b
 */
bool shares_data(const cv::Mat& a, const cv::Mat& b);

/**
 * @brief Negates all even indexed values.
 *
 * @param[in] vector A row or column vector.
 * @param[out] result The input vector with the even indexed values negated.
 */
void negate_even_indices(cv::InputArray vector, cv::OutputArray result);

/**
 * @brief Negates all odd indexed values.
 *
 * @param[in] vector A row or column vector.
 * @param[out] result The input vector with the odd indexed values negated.
 */
void negate_odd_indices(cv::InputArray vector, cv::OutputArray result);

/**
 * @brief Returns true if array is cv::noArray().
 *
 * @param[in] array
 */
bool is_no_array(cv::InputArray array);

/**
 * @brief Returns the maximum absolute value over all channels.
 *
 * @param[in] array
 * @param[in] mask
 */
double maximum_abs_value(cv::InputArray array, cv::InputArray mask = cv::noArray());

/**
 * @brief Replace all NaN values.
 *
 * This is a version of cv::patch_nans() that accepts arrays of any depth, not
 * just CV_32F.
 *
 * @param[inout] array The array containing NaN values.
 * @param[in] value The value used to replace NaN.
 */
void patch_nans(cv::InputOutputArray array, double value = 0.0);

/**
 * @brief Returns true if the given value can be used as a scalar for the given array.
 *
 * Scalars can be added to or subtracted from the array, be assigned to all or
 * some array elements, or be used with comparison functions (e.g. compare(),
 * less_than(), etc.).
 *
 * A scalar is defined to be one of the following:
 *  - A fundamental type (e.g. float, double, etc.)
 *  - A vector containing @pref{array}.channels() elements (e.g. cv::Vec,
 *    std::vector, array, etc.)
 *  - A cv::Scalar if @pref{array}.channels() is less than or equal to 4
 *
 * @param[in] scalar
 * @param[in] array
 */
bool is_scalar_for_array(cv::InputArray scalar, cv::InputArray array);

/**
 * @brief Returns true if the array is a row vector or column vector.
 *
 * The @pref{array} is a vector if it has a single row or column and is
 * continuous.
 *
 * @param array The potential vector.
 */
bool is_vector(cv::InputArray array);

/**
 * @brief Returns true if the array is a row vector or column vector.
 *
 * The @pref{array} is a vector if it has a single row or column, is continuous,
 * and has @pref{channels} number of channels.
 *
 * @param array The potential vector.
 * @param channels The required number of channels.
 */
bool is_vector(cv::InputArray array, int channels);

/**
 * @brief Returns true if the array is a row vector or column vector.
 *
 * The @pref{array} is a vector if it has a single column and is continuous.
 *
 * @param array The potential vector.
 */
bool is_column_vector(cv::InputArray array);

/**
 * @brief Returns true if the array is a column vector.
 *
 * The @pref{array} is a vector if it has a single column, is continuous, and
 * has @pref{channels} number of channels.
 *
 * @param array The potential vector.
 * @param channels The required number of channels.
 */
bool is_column_vector(cv::InputArray array, int channels);

/**
 * @brief Returns true if the array is a row vector or column vector.
 *
 * The @pref{array} is a vector if it has a single row and is continuous.
 *
 * @param array The potential vector.
 */
bool is_row_vector(cv::InputArray array);

/**
 * @brief Returns true if the array is a row vector.
 *
 * The @pref{array} is a vector if it has a single row, is continuous, and
 * has @pref{channels} number of channels.
 *
 * @param array The potential vector.
 * @param channels The required number of channels.
 */
bool is_row_vector(cv::InputArray array, int channels);
/** @}*/

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
/** @}*/


/**
 * @name Multichannel Comparison Functions
 * @{
 */
/**
 * @brief Returns a multichannel mask indicating which elements of a matrix are
 *        less than the elements of another matrix.
 *
 * @param[in] a
 * @param[in] b
 * @param[out] result
 * @param[in] mask
 */
void less_than(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask = cv::noArray()
);
/**
 * @brief Returns a multichannel mask indicating which elements of a matrix are
 *        less than or equal to the elements of another matrix.
 *
 * @param[in] a
 * @param[in] b
 * @param[out] result
 * @param[in] mask
 */
void less_than_or_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask = cv::noArray()
);
/**
 * @brief Returns a multichannel mask indicating which elements of a matrix are
 *        greater than the elements of another matrix.
 *
 * @param[in] a
 * @param[in] b
 * @param[out] result
 * @param[in] mask
 */
void greater_than(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask = cv::noArray()
);
/**
 * @brief Returns a multichannel mask indicating which elements of a matrix are
 *        greater than or equal to the elements of another matrix.
 *
 * @param[in] a
 * @param[in] b
 * @param[out] result
 * @param[in] mask
 */
void greater_than_or_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask = cv::noArray()
);
/**
 * @brief Returns a multichannel mask indicating how the elements of a matrix
 *        compare to the elements of another matrix.
 *
 * @param[in] a
 * @param[in] b
 * @param[out] result
 * @param[in] compare_type
 * @param[in] mask
 */
void compare(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::CmpTypes compare_type,
    cv::InputArray mask = cv::noArray()
);
/** @}*/
}   // namespace cvwt

#endif  // CVWT_UTILS_HPP

