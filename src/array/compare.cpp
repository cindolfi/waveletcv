#include "cvwt/array/compare.hpp"

#include <atomic>
#include "cvwt/dispatch.hpp"
#include "cvwt/exception.hpp"
#include "cvwt/utils.hpp"
#include "cvwt/array/array.hpp"


namespace cvwt
{
namespace internal
{
template <typename T1, typename T2, cv::CmpTypes COMPARE_TYPE>
struct Compare
{
    constexpr bool compare(T1 x, T2 y) const
    {
        if constexpr (COMPARE_TYPE == cv::CMP_LT)
            return x < y;
        else if constexpr (COMPARE_TYPE == cv::CMP_LE)
            return x <= y;
        else if constexpr (COMPARE_TYPE == cv::CMP_GT)
            return x > y;
        else if constexpr (COMPARE_TYPE == cv::CMP_GE)
            return x >= y;
        else if constexpr (COMPARE_TYPE == cv::CMP_EQ)
            return x == y;
        else if constexpr (COMPARE_TYPE == cv::CMP_NE)
            return x != y;
    }

    void operator()(
        cv::InputArray input_a,
        cv::InputArray input_b,
        cv::OutputArray output
    ) const
    {
        auto a = input_a.getMat();
        auto b = input_b.getMat();

        if (is_scalar_for_array(input_b, input_a)) {
            output.create(a.size(), CV_8UC(a.channels()));
            auto result = output.getMat();
            if (b.total() == 1)
                compare_matrix_to_value(a, b, result);
            else
                compare_matrix_to_scalar(a, b, result);
        } else if (is_scalar_for_array(input_a, input_b)) {
            output.create(b.size(), CV_8UC(b.channels()));
            auto result = output.getMat();
            if (a.total() == 1)
                compare_value_to_matrix(a, b, result);
            else
                compare_scalar_to_matrix(a, b, result);
        } else {
            throw_if_comparing_different_sizes(input_a, input_b);
            output.create(a.size(), CV_8UC(a.channels()));
            auto result = output.getMat();
            compare_matrix_to_matrix(a, b, result);
        }
    }

    void compare_matrix_to_scalar(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result
    ) const
    {
        int channels = result.channels();
        auto y = b.ptr<T2>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto x = a.ptr<T1>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = 255 * compare(x[i], y[i]);
                }
            }
        );
    }

    void compare_matrix_to_value(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result
    ) const
    {
        int channels = result.channels();
        auto y = b.at<T2>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto x = a.ptr<T1>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = 255 * compare(x[i], y);
                }
            }
        );
    }

    void compare_scalar_to_matrix(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result
    ) const
    {
        int channels = result.channels();
        auto x = a.ptr<T1>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto y = b.ptr<T2>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = 255 * compare(x[i], y[i]);
                }
            }
        );
    }

    void compare_value_to_matrix(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result
    ) const
    {
        int channels = result.channels();
        auto x = a.at<T1>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto y = b.ptr<T2>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = 255 * compare(x, y[i]);
                }
            }
        );
    }

    void compare_matrix_to_matrix(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result
    ) const
    {
        int channels = result.channels();
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto x = a.ptr<T1>(row, col);
                    auto y = b.ptr<T2>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = 255 * compare(x[i], y[i]);
                }
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
        throw_if_bad_mask_depth(mask);

        auto a = input_a.getMat();
        auto b = input_b.getMat();
        auto mask_matrix = mask.getMat();
        if (is_scalar_for_array(input_b, input_a)) {
            throw_if_bad_mask_for_array(input_a, mask, AllowedMaskChannels::SINGLE_OR_SAME);
            output.create(a.size(), CV_8UC(a.channels()));
            auto result = output.getMat();
            if (mask.channels() == 1)
                if (b.total() == 1)
                    compare_matrix_to_value_using_single_channel_mask(a, b, result, mask_matrix);
                else
                    compare_matrix_to_scalar_using_single_channel_mask(a, b, result, mask_matrix);
            else
                if (b.total() == 1)
                    compare_matrix_to_value_using_multi_channel_mask(a, b, result, mask_matrix);
                else
                    compare_matrix_to_scalar_using_multi_channel_mask(a, b, result, mask_matrix);
        } else if (is_scalar_for_array(input_a, input_b)) {
            throw_if_bad_mask_for_array(input_b, mask, AllowedMaskChannels::SINGLE_OR_SAME);
            output.create(b.size(), CV_8UC(b.channels()));
            auto result = output.getMat();
            if (mask.channels() == 1)
                if (a.total() == 1)
                    compare_value_to_matrix_using_single_channel_mask(a, b, result, mask_matrix);
                else
                    compare_scalar_to_matrix_using_single_channel_mask(a, b, result, mask_matrix);
            else
                if (a.total() == 1)
                    compare_value_to_matrix_using_multi_channel_mask(a, b, result, mask_matrix);
                else
                    compare_scalar_to_matrix_using_multi_channel_mask(a, b, result, mask_matrix);
        } else {
            throw_if_comparing_different_sizes(input_a, input_b);
            throw_if_bad_mask_for_array(input_a, mask, AllowedMaskChannels::SINGLE_OR_SAME);
            output.create(a.size(), CV_8UC(a.channels()));
            auto result = output.getMat();
            if (mask.channels() == 1)
                compare_matrix_to_matrix_using_single_channel_mask(a, b, result, mask_matrix);
            else
                compare_matrix_to_matrix_using_multi_channel_mask(a, b, result, mask_matrix);
        }
    }

    void compare_matrix_to_scalar_using_single_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        auto y = b.ptr<T2>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto z = result.ptr<uchar>(row, col);
                    if (mask_matrix.at<uchar>(row, col)) {
                        auto x = a.ptr<T1>(row, col);
                        for (int i = 0; i < channels; ++i)
                            z[i] = 255 * compare(x[i], y[i]);
                    } else {
                        for (int i = 0; i < channels; ++i)
                            z[i] = 0;
                    }
                }
            }
        );
    }

    void compare_matrix_to_value_using_single_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        auto y = b.at<T2>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto z = result.ptr<uchar>(row, col);
                    if (mask_matrix.at<uchar>(row, col)) {
                        auto x = a.ptr<T1>(row, col);
                        for (int i = 0; i < channels; ++i)
                            z[i] = 255 * compare(x[i], y);
                    } else {
                        for (int i = 0; i < channels; ++i)
                            z[i] = 0;
                    }
                }
            }
        );
    }

    void compare_scalar_to_matrix_using_single_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        auto x = a.ptr<T1>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto z = result.ptr<uchar>(row, col);
                    if (mask_matrix.at<uchar>(row, col)) {
                        auto y = b.ptr<T2>(row, col);
                        for (int i = 0; i < channels; ++i)
                            z[i] = 255 * compare(x[i], y[i]);
                    } else {
                        for (int i = 0; i < channels; ++i)
                            z[i] = 0;
                    }
                }
            }
        );
    }

    void compare_value_to_matrix_using_single_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        auto x = a.at<T1>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto z = result.ptr<uchar>(row, col);
                    if (mask_matrix.at<uchar>(row, col)) {
                        auto y = b.ptr<T2>(row, col);
                        for (int i = 0; i < channels; ++i)
                            z[i] = 255 * compare(x, y[i]);
                    } else {
                        for (int i = 0; i < channels; ++i)
                            z[i] = 0;
                    }
                }
            }
        );
    }

    void compare_matrix_to_matrix_using_single_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto z = result.ptr<uchar>(row, col);
                    if (mask_matrix.at<uchar>(row, col)) {
                        auto x = a.ptr<T1>(row, col);
                        auto y = b.ptr<T2>(row, col);
                        for (int i = 0; i < channels; ++i)
                            z[i] = 255 * compare(x[i], y[i]);
                    } else {
                        for (int i = 0; i < channels; ++i)
                            z[i] = 0;
                    }
                }
            }
        );
    }

    void compare_matrix_to_scalar_using_multi_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        auto y = b.ptr<T2>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto x = a.ptr<T1>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    auto m = mask_matrix.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = m[i] ? 255 * compare(x[i], y[i]) : 0;
                }
            }
        );
    }

    void compare_matrix_to_value_using_multi_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        auto y = b.at<T2>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto x = a.ptr<T1>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    auto m = mask_matrix.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = m[i] ? 255 * compare(x[i], y) : 0;
                }
            }
        );
    }

    void compare_scalar_to_matrix_using_multi_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        auto x = a.ptr<T1>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto y = b.ptr<T2>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    auto m = mask_matrix.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = m[i] ? 255 * compare(x[i], y[i]) : 0;
                }
            }
        );
    }

    void compare_value_to_matrix_using_multi_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        auto x = a.at<T1>(0);
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto y = b.ptr<T2>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    auto m = mask_matrix.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = m[i] ? 255 * compare(x, y[i]) : 0;
                }
            }
        );
    }

    void compare_matrix_to_matrix_using_multi_channel_mask(
        const cv::Mat& a,
        const cv::Mat& b,
        cv::Mat& result,
        const cv::Mat& mask_matrix
    ) const
    {
        int channels = result.channels();
        cv::parallel_for_(
            cv::Range(0, result.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(result, index);
                    auto x = a.ptr<T1>(row, col);
                    auto y = b.ptr<T2>(row, col);
                    auto z = result.ptr<uchar>(row, col);
                    auto m = mask_matrix.ptr<uchar>(row, col);
                    for (int i = 0; i < channels; ++i)
                        z[i] = m[i] ? 255 * compare(x[i], y[i]) : 0;
                }
            }
        );
    }

    void throw_if_comparing_different_sizes(
        cv::InputArray a,
        cv::InputArray b,
        const std::source_location& location = std::source_location::current()
    ) const
    {
        if (a.size() != b.size())
            throw_bad_size(
                "Cannot compare matrices of different sizes. ",
                "Got a.size() = ", a.size(),
                " and b.size() = ", b.size(), ".",
                location
            );

        if (a.channels() != b.channels())
            throw_bad_size(
                "Cannot compare matrices of with different number of channels. ",
                "Got a.channels() = ", a.channels(),
                " and b.channels() = ", b.channels(), ".",
                location
            );
    }
};


template <typename T1, typename T2>
using LessThan = Compare<T1, T2, cv::CMP_LT>;

template <typename T1, typename T2>
using LessThanOrEqual = Compare<T1, T2, cv::CMP_LE>;

template <typename T1, typename T2>
using GreaterThan = Compare<T1, T2, cv::CMP_GT>;

template <typename T1, typename T2>
using GreaterThanOrEqual = Compare<T1, T2, cv::CMP_GE>;

template <typename T1, typename T2>
using Equal = Compare<T1, T2, cv::CMP_EQ>;

template <typename T1, typename T2>
using NotEqual = Compare<T1, T2, cv::CMP_NE>;


template <typename T1, typename T2>
struct IsApproxEqual
{
    using Float = promote_types<T1, T2, float>;

    bool operator()(
        cv::InputArray a,
        cv::InputArray b,
        double relative_tolerance,
        double zero_absolute_tolerance
    ) const
    {
        auto _is_approx_equal = [&](T1 x, T2 y) {
            return is_approx_equal<Float, Float>(
                x,
                y,
                relative_tolerance,
                zero_absolute_tolerance
            );
        };

        return compare_approx_equal(a, b, _is_approx_equal);
    }

    bool operator()(
        cv::InputArray a,
        cv::InputArray b,
        double relative_tolerance
    ) const
    {
        auto _is_approx_equal = [&](T1 x, T2 y) {
            return is_approx_equal<Float, Float>(
                x,
                y,
                relative_tolerance
            );
        };

        return compare_approx_equal(a, b, _is_approx_equal);
    }

    bool operator()(
        cv::InputArray a,
        cv::InputArray b
    ) const
    {
        auto _is_approx_equal = [&](T1 x, T2 y) {
            return is_approx_equal<Float, Float>(x, y);
        };

        return compare_approx_equal(a, b, _is_approx_equal);
    }

protected:
    bool compare_approx_equal(cv::InputArray a, cv::InputArray b, auto is_approx_equal) const
    {
        if (a.dims() != b.dims() || a.size() != b.size() || a.channels() != b.channels())
            return false;

        cv::Mat matrix_a = a.getMat();
        cv::Mat matrix_b = b.getMat();
        const int channels = matrix_a.channels();

        std::atomic<bool> result = true;
        cv::parallel_for_(
            cv::Range(0, matrix_a.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(matrix_a, index);
                    auto x = matrix_a.ptr<T1>(row, col);
                    auto y = matrix_b.ptr<T2>(row, col);
                    for (int k = 0; k < channels; ++k) {
                        if (!is_approx_equal(x[k], y[k])) {
                            result = false;
                            break;
                        }

                        if (!result)
                            break;
                    }
                }
            }
        );

        return result;
    }
};

template <typename T>
struct IsApproxZero
{
    using Float = promote_types<T, float>;

    bool operator()(cv::InputArray a, double zero_absolute_tolerance) const
    {
        auto _is_approx_zero = [&](T x) {
            return is_approx_zero<Float>(x, zero_absolute_tolerance);
        };

        return compare_approx_zero(a, _is_approx_zero);
    }

    bool operator()(cv::InputArray a) const
    {
        auto _is_approx_zero = [&](T x) {
            return is_approx_zero<Float>(x);
        };

        return compare_approx_zero(a, _is_approx_zero);
    }

protected:
bool compare_approx_zero(cv::InputArray a, auto is_approx_zero) const
    {
        cv::Mat matrix_a = a.getMat();
        const int channels = matrix_a.channels();

        std::atomic<bool> result = true;
        cv::parallel_for_(
            cv::Range(0, matrix_a.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    auto [row, col] = unravel_index(matrix_a, index);
                    auto x = matrix_a.ptr<T>(row, col);
                    for (int k = 0; k < channels; ++k)
                        if (!is_approx_zero(x[k])) {
                            result = false;
                            goto done;
                        }
                }
                done:;
            }
        );

        return result;
    }
};
}   // namespace internal

bool is_equal(cv::InputArray a, cv::InputArray b)
{
    if (a.dims() != b.dims() || a.size() != b.size() || a.channels() != b.channels())
        return false;

    cv::Mat matrix_a = a.getMat();
    cv::Mat matrix_b = b.getMat();
    const cv::Mat* matrices[2] = {&matrix_a, &matrix_b};
    cv::Mat planes[2];
    cv::NAryMatIterator it(matrices, planes, 2);
    for (int p = 0; p < it.nplanes; ++p, ++it)
        if (cv::countNonZero(it.planes[0].reshape(1) != it.planes[1].reshape(1)) != 0)
            return false;

    return true;
}

bool is_approx_equal(
    cv::InputArray a,
    cv::InputArray b,
    double relative_tolerance,
    double zero_absolute_tolerance
)
{
    return internal::dispatch_on_pixel_depths<internal::IsApproxEqual>(
        a.depth(), b.depth(), a, b, relative_tolerance, zero_absolute_tolerance
    );
}

bool is_approx_equal(cv::InputArray a, cv::InputArray b, double relative_tolerance)
{
    return internal::dispatch_on_pixel_depths<internal::IsApproxEqual>(
        a.depth(), b.depth(), a, b, relative_tolerance
    );
}

bool is_approx_equal(cv::InputArray a, cv::InputArray b)
{
    return internal::dispatch_on_pixel_depths<internal::IsApproxEqual>(
        a.depth(), b.depth(), a, b
    );
}

bool is_approx_zero(cv::InputArray a, double absolute_tolerance)
{
    return internal::dispatch_on_pixel_depth<internal::IsApproxZero>(
        a.depth(), a, absolute_tolerance
    );
}

bool is_approx_zero(cv::InputArray a)
{
    return internal::dispatch_on_pixel_depth<internal::IsApproxZero>(
        a.depth(), a
    );
}

bool is_identical(cv::InputArray a, cv::InputArray b)
{
    cv::Mat matrix_a = a.getMat();
    cv::Mat matrix_b = b.getMat();
    return matrix_a.data == matrix_b.data
        && std::ranges::equal(
            std::ranges::subrange(matrix_a.step.p, matrix_a.step.p + matrix_a.dims),
            std::ranges::subrange(matrix_b.step.p, matrix_b.step.p + matrix_b.dims)
        );
}

void less_than(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_depths<internal::LessThan>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::LessThan>(
            a.depth(), b.depth(), a, b, result, mask
        );
}

void less_than_or_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_depths<internal::LessThanOrEqual>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::LessThanOrEqual>(
            a.depth(), b.depth(), a, b, result, mask
        );
}

void greater_than(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_depths<internal::GreaterThan>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::GreaterThan>(
            a.depth(), b.depth(), a, b, result, mask
        );
}

void greater_than_or_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_depths<internal::GreaterThanOrEqual>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::GreaterThanOrEqual>(
            a.depth(), b.depth(), a, b, result, mask
        );
}

void equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_depths<internal::Equal>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::Equal>(
            a.depth(), b.depth(), a, b, result, mask
        );
}

void not_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_depths<internal::NotEqual>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::NotEqual>(
            a.depth(), b.depth(), a, b, result, mask
        );
}

void compare(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::CmpTypes compare_type,
    cv::InputArray mask
)
{
    switch (compare_type) {
    case cv::CMP_LT:
        less_than(a, b, result, mask);
        break;
    case cv::CMP_LE:
        less_than_or_equal(a, b, result, mask);
        break;
    case cv::CMP_GT:
        greater_than(a, b, result, mask);
        break;
    case cv::CMP_GE:
        greater_than_or_equal(a, b, result, mask);
        break;
    case cv::CMP_EQ:
        equal(a, b, result, mask);
        break;
    case cv::CMP_NE:
        not_equal(a, b, result, mask);
        break;
    }
}
}   // namespace cvwt
