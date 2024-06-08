#include "cvwt/utils.hpp"

#include <iostream>
#include <sstream>
#include <experimental/iterator>
#include <atomic>
#include <limits>

namespace cvwt
{
namespace internal
{
std::string get_type_name(int type)
{
    std::string channels = std::to_string(CV_MAT_CN(type));
    switch (CV_MAT_DEPTH(type)){
        case CV_64F: return "CV_64FC" + channels;
        case CV_32F: return "CV_32FC" + channels;
        case CV_32S: return "CV_32SC" + channels;
        case CV_16S: return "CV_16SC" + channels;
        case CV_16U: return "CV_16UC" + channels;
        case CV_8S: return "CV_8SC" + channels;
        case CV_8U: return "CV_8UC" + channels;
    }

    return std::to_string(type);
}

cv::Scalar set_unused_channels(const cv::Scalar& scalar, int channels, double value)
{
    cv::Scalar result(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < channels; ++i)
        result[i] = scalar[i];

    return result;
}

template <typename T>
struct CollectMasked
{
    void operator()(cv::InputArray input, cv::OutputArray output, cv::InputArray mask) const
    {
        throw_if_bad_mask_type(mask);
        throw_if_bad_mask_for_array(input, mask, AllowedMaskChannels::SINGLE);
        if (input.empty())
            output.create(cv::Size(), input.type());
        else
            output.create(cv::countNonZero(mask), 1, input.type());

        auto collected = output.getMat();
        if (!collected.empty()) {
            std::atomic<int> insert_index = 0;
            auto input_matrix = input.getMat();
            auto mask_matrix = mask.getMat();
            int channels = input_matrix.channels();
            cv::parallel_for_(
                cv::Range(0, input_matrix.total()),
                [&](const cv::Range& range) {
                    for (int index = range.start; index < range.end; ++index) {
                        int row = index / input_matrix.cols;
                        int col = index % input_matrix.cols;
                        if (mask_matrix.at<uchar>(row, col)) {
                            auto pixel = input_matrix.ptr<T>(row, col);
                            auto collected_pixel = collected.ptr<T>(insert_index++);
                            std::copy(pixel, pixel + channels, collected_pixel);
                        }
                    }
                }
            );
        }
    }
};

template <typename T, int N>
struct Median
{
    using Pixel = cv::Vec<T, N>;

    cv::Scalar operator()(cv::InputArray array) const
    {
        throw_if_empty(array);
        //  Noncontinuous arrays must be cloned because compute_median()
        //  requires continuous memory layout.  We also need to clone continuous
        //  single channel arrays because the partial sort done by
        //  std::nth_element() is done inplace.  Multichannel arrays do not need
        //  to be cloned because they are split using cv::split(), which copies
        //  each channel.  One and two element arrays are special cases that do
        //  not use std::nth_element().
        cv::Mat matrix = array.getMat();
        if (array.total() > 2 && (!matrix.isContinuous() || matrix.channels() == 1))
            matrix = matrix.clone();

        return compute_median(matrix);
    }

    cv::Scalar operator()(cv::InputArray array, cv::InputArray mask) const
    {
        throw_if_empty(array);
        cv::Mat masked_array;
        collect_masked(array, masked_array, mask);

        return compute_median(masked_array);
    }

private:
    Pixel compute_median(cv::Mat& array) const
    {
        assert(array.channels() == N);
        assert(!array.empty());
        assert(array.isContinuous());

        if (array.total() == 1) {
            return array.at<Pixel>(0);
        } else if (array.total() == 2) {
            return 0.5 * (array.at<Pixel>(0) + array.at<Pixel>(1));
        } else {
            Pixel result;
            if constexpr (N == 1) {
                result[0] = single_channel_median(array);
            } else {
                cv::Mat array_channels[N];
                cv::split(array, array_channels);
                cv::parallel_for_(
                    cv::Range(0, N),
                    [&](const cv::Range& range) {
                        for (int i = range.start; i < range.end; ++i)
                            result[i] = single_channel_median(array_channels[i]);
                    }
                );
            }

            return result;
        }
    }

    T single_channel_median(cv::Mat& array) const
    {
        assert(array.channels() == 1);

        std::span<T> values(array.ptr<T>(), array.total());
        int mid_index = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid_index, values.end());

        auto result = values[mid_index];
        if (values.size() % 2 == 0) {
            std::nth_element(values.begin(), values.begin() + mid_index - 1, values.end());
            result = 0.5 * (result + values[mid_index - 1]);
        }

        return result;
    }
};

template <typename T, int EVEN_OR_ODD>
requires(EVEN_OR_ODD == 0 || EVEN_OR_ODD == 1)
struct NegateEveryOther
{
    void operator()(cv::InputArray array, cv::OutputArray output) const
    {
        output.create(array.size(), array.type());
        if (array.empty())
            return;

        throw_if_not_vector(array, -1);

        const int channels = array.channels();
        cv::Mat array_matrix = array.getMat();
        cv::Mat output_matrix = output.getMat();
        cv::parallel_for_(
            cv::Range(0, array.total()),
            [&](const cv::Range& range) {
                for (int i = range.start; i < range.end; ++i) {
                    T sign = (i % 2 == EVEN_OR_ODD) ? -1 : 1;
                    auto y = output_matrix.ptr<T>(i);
                    auto x = array_matrix.ptr<T>(i);
                    for (int k = 0; k < channels; ++k)
                        y[k] = sign * x[k];
                }
            }
        );

        // array.getMat().forEach<T>(
        //     [&](const auto& coeff, const auto index) {
        //         int i = index[0];
        //         T sign = i % 2 == EVEN_OR_ODD ? -1 : 1;
        //         auto y = output_matrix.ptr<T>(i);
        //         for (int k = 0; k < channels; ++k)
        //             y[k] = sign * x[k]
        //     }
        // );
    }
};

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

    void throw_if_comparing_different_sizes(cv::InputArray a, cv::InputArray b) const
    {
        if (a.size() != b.size())
            throw_bad_size(
                "Cannot compare matrices of different sizes. ",
                "Got a.size() = ", a.size(),
                " and b.size() = ", b.size(), "."
            );

        if (a.channels() != b.channels())
            throw_bad_size(
                "Cannot compare matrices of with different number of channels. ",
                "Got a.channels() = ", a.channels(),
                " and b.channels() = ", b.channels(), "."
            );
    }
};

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

void collect_masked(cv::InputArray array, cv::OutputArray collected, cv::InputArray mask)
{
    internal::dispatch_on_pixel_depth<internal::CollectMasked>(
        array.depth(),
        array,
        collected,
        mask
    );
}

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

bool is_data_shared(cv::InputArray a, cv::InputArray b)
{
    return a.getMat().datastart == b.getMat().datastart;
}

cv::Scalar median(cv::InputArray array, cv::InputArray mask)
{
    if (is_not_array(mask)) {
        return internal::dispatch_on_pixel_type<internal::Median>(
            array.type(),
            array
        );
    } else {
        return internal::dispatch_on_pixel_type<internal::Median>(
            array.type(),
            array,
            mask
        );
    }
}

cv::Scalar mad(cv::InputArray array, cv::InputArray mask)
{
    auto x = array.getMat();
    return median(cv::abs(x - median(x, mask)), mask);
}

cv::Scalar mad_stdev(cv::InputArray array)
{
    return mad(array) / 0.675;
}

void negate_even_indices(cv::InputArray vector, cv::OutputArray result)
{
    internal::dispatch_on_pixel_depth<internal::NegateEveryOther, 0>(
        vector.depth(),
        vector,
        result
    );
}

void negate_odd_indices(cv::InputArray vector, cv::OutputArray result)
{
    internal::dispatch_on_pixel_depth<internal::NegateEveryOther, 1>(
        vector.depth(),
        vector,
        result
    );
}

bool is_not_array(cv::InputArray array)
{
    return array.kind() == cv::_InputArray::NONE;
}

void patch_nans(cv::InputOutputArray array, double value)
{
    if (array.depth() == CV_64F || array.depth() == CV_32F) {
        cv::Mat nan_mask;
        compare(array, array, nan_mask, cv::CMP_NE);
        array.setTo(value, nan_mask);
    }
}

bool is_scalar_for_array(cv::InputArray scalar, cv::InputArray array)
{
    //  This is adapted from checkScalar in OpenCV source code.
    if (scalar.dims() > 2 || !scalar.isContinuous())
        return false;

    int channels = array.channels();

    return scalar.isVector()
        && (
            (scalar.total() == channels || scalar.total() == 1)
            || (scalar.size() == cv::Size(1, 4) && scalar.type() == CV_64F && channels <= 4)
        );
}

bool is_vector(cv::InputArray array)
{
    return (array.rows() == 1 || array.cols() == 1)
        && array.isContinuous();
}

bool is_vector(cv::InputArray array, int channels)
{
    return (array.rows() == 1 || array.cols() == 1)
        && array.channels() == channels
        && array.isContinuous();
}

bool is_column_vector(cv::InputArray array)
{
    return array.cols() == 1
        && array.isContinuous();
}

bool is_column_vector(cv::InputArray array, int channels)
{
    return array.cols() == 1
        && array.channels() == channels
        && array.isContinuous();
}

bool is_row_vector(cv::InputArray array)
{
    return array.rows() == 1
        && array.isContinuous();
}

bool is_row_vector(cv::InputArray array, int channels)
{
    return array.rows() == 1
        && array.channels() == channels
        && array.isContinuous();
}

double maximum_abs_value(cv::InputArray array, cv::InputArray mask)
{
    internal::throw_if_empty(array, "Input is empty.");

    auto abs_max = [](cv::InputArray channel_matrix, cv::InputArray channel_mask) {
        double min, max;
        cv::minMaxIdx(channel_matrix, &min, &max, nullptr, nullptr, channel_mask);
        return std::max(std::abs(min), std::abs(max));
    };

    double result = 0.0;
    if (is_not_array(mask)) {
        if (array.channels() == 1) {
            result = abs_max(array, cv::noArray());
        } else {
            auto array_matrix = array.isContinuous() ? array.getMat()
                                                     : array.getMat().clone();
            result = abs_max(array_matrix.reshape(1), cv::noArray());
        }
    } else {
        internal::throw_if_bad_mask_type(mask);
        internal::throw_if_empty(
            mask,
            "Mask is empty. Use cv::noArray() to indicate no mask."
        );
        if (mask.size() != array.size())
            internal::throw_bad_size(
                "The array and mask must be the same size, ",
                "got array.size() = ", array.size(),
                " and mask.size() = ", mask.size(), "."
            );

        if (array.channels() == 1) {
            if (mask.channels() > 1) {
                internal::throw_bad_size(
                    "Wrong number of mask channels for single channel array. ",
                    "Must be 1, got mask.channels() = ", mask.channels(), "."
                );
            }
            result = abs_max(array, mask);
        } else {
            std::vector<cv::Mat> array_channels;
            cv::split(array, array_channels);
            if (mask.channels() == 1) {
                for (const auto& array_channel : array_channels)
                    result = std::max(result, abs_max(array_channel, mask));
            } else if (array.channels() == mask.channels()) {
                std::vector<cv::Mat> mask_channels;
                cv::split(mask, mask_channels);
                for (int i = 0; i < array_channels.size(); ++i) {
                    result = std::max(
                        result,
                        abs_max(array_channels.at(i), mask_channels.at(i))
                    );
                }
            } else {
                internal::throw_bad_size(
                    "Wrong number of mask channels for ",
                    "array.channels() = ", array.channels(), ". ",
                    "Must be 1 or ", array.channels(),
                    ", got mask.channels() = ", mask.channels(), "."
                );
            }
        }
    }

    return result;
}

void less_than(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    cv::InputArray mask
)
{
    if (is_not_array(mask))
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_LT>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_LT>(
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
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_LE>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_LE>(
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
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_GT>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_GT>(
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
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_GE>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_GE>(
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
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_EQ>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_EQ>(
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
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_NE>(
            a.depth(), b.depth(), a, b, result
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_NE>(
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
