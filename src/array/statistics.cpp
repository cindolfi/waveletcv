#include "wtcv/array/statistics.hpp"

#include <span>
#include "wtcv/dispatch.hpp"
#include "wtcv/exception.hpp"
#include "wtcv/array/array.hpp"

namespace wtcv
{
namespace internal
{
template <typename T, int CHANNELS>
struct Median
{
    using Pixel = cv::Vec<T, CHANNELS>;

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
        assert(array.channels() == CHANNELS);
        assert(!array.empty());
        assert(array.isContinuous());

        if (array.total() == 1) {
            return array.at<Pixel>(0);
        } else if (array.total() == 2) {
            return 0.5 * (array.at<Pixel>(0) + array.at<Pixel>(1));
        } else {
            Pixel result;
            if constexpr (CHANNELS == 1) {
                result[0] = single_channel_median(array);
            } else {
                cv::Mat array_channels[CHANNELS];
                cv::split(array, array_channels);
                cv::parallel_for_(
                    cv::Range(0, CHANNELS),
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
}   // namespace internal


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

double maximum_abs_value(cv::InputArray array, cv::InputArray mask)
{
    throw_if_empty(array, "Input is empty.");

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
        throw_if_bad_mask_for_array(
            array,
            mask,
            AllowedMaskChannels::SINGLE_OR_SAME
        );

        if (array.channels() == 1) {
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
            }
        }
    }

    return result;
}
}   // namespace wtcv
