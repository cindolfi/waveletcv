#include "cvwt/utils.hpp"

#include <iostream>
#include <sstream>
#include <experimental/iterator>

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
    assert(false);
    return "";
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

        cv::Mat output_matrix = output.getMat();
        array.getMat().forEach<T>(
            [&](const auto& coeff, const auto index) {
                int i = index[0];
                output_matrix.at<T>(i) = (i % 2 == EVEN_OR_ODD ? -1 : 1) * coeff;
            }
        );
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
        throw_if_comparing_different_sizes(input_a, input_b);

        int channels = input_a.channels();
        output.create(input_a.size(), CV_8UC(channels));
        auto a = input_a.getMat();
        auto b = input_b.getMat();
        auto result = output.getMat();

        cv::parallel_for_(
            cv::Range(0, a.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    int row = index / a.cols;
                    int col = index % a.cols;
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
        throw_if_comparing_different_sizes(input_a, input_b);
        throw_if_bad_mask_type(mask);
        throw_if_bad_mask_for_array(input_a, mask, AllowedMaskChannels::SINGLE);

        if (mask.size() != input_a.size())
            throw_bad_size(
                "Wrong size mask. Got ", mask.size(), ", must be ", input_a.size(), "."
            );

        int channels = input_a.channels();
        output.create(input_a.size(), CV_8UC(channels));
        auto a = input_a.getMat();
        auto b = input_b.getMat();
        auto mask_matrix = mask.getMat();
        auto result = output.getMat();

        cv::parallel_for_(
            cv::Range(0, a.total()),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    int row = index / a.cols;
                    int col = index % a.cols;
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
}   // namespace internal


void flatten(cv::InputArray array, cv::OutputArray result)
{
    array.copyTo(result);
    result.assign(result.getMat().reshape(0, array.total()));
}

void collect_masked(cv::InputArray array, cv::OutputArray collected, cv::InputArray mask)
{
    internal::dispatch_on_pixel_depth<internal::CollectMasked>(
        array.depth(),
        array,
        collected,
        mask
    );
}

bool matrix_equals(const cv::Mat& a, const cv::Mat& b)
{
    if (a.dims != b.dims || a.size != b.size || a.channels() != b.channels())
        return false;

    const cv::Mat* matrices[2] = {&a, &b};
    cv::Mat planes[2];
    cv::NAryMatIterator it(matrices, planes, 2);
    for (int p = 0; p < it.nplanes; ++p, ++it)
        if (cv::countNonZero(it.planes[0] != it.planes[1]) != 0)
            return false;

    return true;
}

bool identical(const cv::Mat& a, const cv::Mat& b)
{
    return a.data == b.data
        && std::ranges::equal(
            std::ranges::subrange(a.step.p, a.step.p + a.dims),
            std::ranges::subrange(b.step.p, b.step.p + b.dims)
        );
}

bool shares_data(const cv::Mat& a, const cv::Mat& b)
{
    return a.datastart == b.datastart;
}

cv::Scalar median(cv::InputArray array, cv::InputArray mask)
{
    if (is_no_array(mask)) {
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

bool is_no_array(cv::InputArray array)
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

double maximum_abs_value(cv::InputArray array, cv::InputArray mask)
{
    internal::throw_if_empty(array, "Input is empty.");

    auto abs_max = [](cv::InputArray channel_matrix, cv::InputArray channel_mask) {
        double min, max;
        cv::minMaxIdx(channel_matrix, &min, &max, nullptr, nullptr, channel_mask);
        return std::max(std::abs(min), std::abs(max));
    };

    double result = 0.0;
    if (is_no_array(mask)) {
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
    cv::OutputArray output,
    cv::InputArray mask
)
{
    if (is_no_array(mask))
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_LT>(
            a.depth(), b.depth(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_LT>(
            a.depth(), b.depth(), a, b, output, mask
        );
}

void less_than_or_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask
)
{
    if (is_no_array(mask))
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_LE>(
            a.depth(), b.depth(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_LE>(
            a.depth(), b.depth(), a, b, output, mask
        );
}

void greater_than(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask
)
{
    if (is_no_array(mask))
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_GT>(
            a.depth(), b.depth(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_GT>(
            a.depth(), b.depth(), a, b, output, mask
        );
}

void greater_than_or_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask
)
{
    if (is_no_array(mask))
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_GE>(
            a.depth(), b.depth(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_GE>(
            a.depth(), b.depth(), a, b, output, mask
        );
}

void equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask
)
{
    if (is_no_array(mask))
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_EQ>(
            a.depth(), b.depth(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_EQ>(
            a.depth(), b.depth(), a, b, output, mask
        );
}

void not_equal(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::InputArray mask
)
{
    if (is_no_array(mask))
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_NE>(
            a.depth(), b.depth(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths<internal::Compare, cv::CMP_NE>(
            a.depth(), b.depth(), a, b, output, mask
        );
}

void compare(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray output,
    cv::CmpTypes compare_type,
    cv::InputArray mask
)
{
    switch (compare_type) {
    case cv::CMP_LT:
        less_than(a, b, output, mask);
        break;
    case cv::CMP_LE:
        less_than_or_equal(a, b, output, mask);
        break;
    case cv::CMP_GT:
        greater_than(a, b, output, mask);
        break;
    case cv::CMP_GE:
        greater_than_or_equal(a, b, output, mask);
        break;
    case cv::CMP_EQ:
        equal(a, b, output, mask);
        break;
    case cv::CMP_NE:
        not_equal(a, b, output, mask);
        break;
    }
}
}   // namespace cvwt
