#ifndef WAVELET_UTILS_HPP
#define WAVELET_UTILS_HPP

#include <opencv2/core.hpp>

namespace wavelet
{
void flatten(cv::InputArray input, cv::OutputArray output);
void collect_masked(cv::InputArray input, cv::OutputArray output, cv::InputArray mask);
cv::Scalar median(cv::InputArray input);
cv::Scalar median(cv::InputArray input, cv::InputArray mask);

namespace internal
{
template<template<typename T, int N, auto ...> class Functor, auto ...TemplateArgs>
void dispatch_on_pixel_type(int type, auto&&... args)
{
    switch (type) {
        //  32 bit floating point
        case CV_32FC1: Functor<float, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_32FC2: Functor<float, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_32FC3: Functor<float, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_32FC4: Functor<float, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  64 bit floating point
        case CV_64FC1: Functor<double, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_64FC2: Functor<double, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_64FC3: Functor<double, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_64FC4: Functor<double, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  32 bit signed integer
        case CV_32SC1: Functor<int, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_32SC2: Functor<int, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_32SC3: Functor<int, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_32SC4: Functor<int, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  16 bit signed integer
        case CV_16SC1: Functor<short, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_16SC2: Functor<short, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_16SC3: Functor<short, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_16SC4: Functor<short, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  16 bit unsigned integer
        case CV_16UC1: Functor<ushort, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_16UC2: Functor<ushort, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_16UC3: Functor<ushort, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_16UC4: Functor<ushort, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  8 bit signed integer
        case CV_8SC1: Functor<char, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_8SC2: Functor<char, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_8SC3: Functor<char, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_8SC4: Functor<char, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  8 bit unsigned integer
        case CV_8UC1: Functor<uchar, 1, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_8UC2: Functor<uchar, 2, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_8UC3: Functor<uchar, 3, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        case CV_8UC4: Functor<uchar, 4, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
    }
}

template <typename T, int N>
struct collect_masked
{
    using Pixel = cv::Vec<T, N>;

    void operator()(cv::InputArray input, cv::OutputArray output, cv::InputArray mask) const
    {
        assert(input.channels() == N);
        assert(mask.type() == CV_8U);

        std::vector<Pixel> result;
        result.reserve(cv::countNonZero(mask));
        auto mask_mat = mask.getMat();
        std::mutex push_back_mutex;
        input.getMat().forEach<Pixel>(
            [&](const auto& pixel, const auto position) {
                if (mask_mat.at<uchar>(position)) {
                    std::lock_guard<std::mutex> lock(push_back_mutex);
                    result.push_back(pixel);
                }
            }
        );

        cv::Mat(result).copyTo(output);
    }
};

template <typename T, int N>
struct median
{
    using Pixel = cv::Vec<T, N>;

    void operator()(cv::InputArray input, cv::Scalar& result) const
    {
        auto matrix = input.getMat();
        assert(matrix.channels() == N);
        if (matrix.total() == 1) {
            result = matrix.at<Pixel>(0, 0);
        } else {
            cv::Mat channels[N];
            cv::split(matrix, channels);
            for (int i = 0; i < N; ++i)
                result[i] = single_channel_median(channels[i]);
        }
    }

    void operator()(cv::InputArray input, cv::InputArray mask, cv::Scalar& result) const
    {
        cv::Mat masked_input;
        collect_masked<T, N>()(input, masked_input, mask);
        this->operator()(masked_input, result);
    }

private:
    T single_channel_median(const cv::Mat& channel) const
    {
        std::vector<T> values(channel.total());
        flatten(channel, values);
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
}   // namespace wavelet

#endif  // WAVELET_UTILS_HPP

