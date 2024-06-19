#ifndef CVWT_ARRAY_COMPARE_HPP
#define CVWT_ARRAY_COMPARE_HPP

#include <type_traits>
#include <utility>
#include <opencv2/core.hpp>

namespace cvwt
{
namespace internal
{
template <std::floating_point T>
inline constexpr T sqrt_epsilon = std::sqrt(std::numeric_limits<T>::epsilon());

template <typename... T>
using promote_types = decltype((std::declval<T>() + ...));
}   // namespace internal


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
 * @brief Returns true if all values are approximately zero.
 *
 * @param[in] a
 * @param[in] absolute_tolerance
 */
bool is_approx_zero(cv::InputArray a, double absolute_tolerance);

/**
 * @overload
 */
bool is_approx_zero(cv::InputArray a);

/**
 * @brief Returns true if two floating point values are approximately equal.
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

/**
 * @brief Returns true if all corresponding array elements are approximately equal.
 *
 * @param[in] a
 * @param[in] b
 * @param[in] relative_tolerance
 * @param[in] zero_absolute_tolerance
 */
bool is_approx_equal(
    cv::InputArray a,
    cv::InputArray b,
    double relative_tolerance,
    double zero_absolute_tolerance
);

/**
 * @overload
 */
bool is_approx_equal(cv::InputArray a, cv::InputArray b, double relative_tolerance);

/**
 * @overload
 */
bool is_approx_equal(cv::InputArray a, cv::InputArray b);

/**
 * @brief Returns true if all values two matrices are equal.
 *
 * @param[in] a
 * @param[in] b
 */
bool is_equal(cv::InputArray a, cv::InputArray b);

/**
 * @brief Returns true if two matrices refer to the same data and are equal.
 *
 * @param[in] a
 * @param[in] b
 */
bool is_identical(cv::InputArray a, cv::InputArray b);



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

#endif  // CVWT_ARRAY_COMPARE_HPP

