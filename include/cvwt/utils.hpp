#ifndef CVWT_UTILS_HPP
#define CVWT_UTILS_HPP

#include <span>
#include <ranges>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <iostream>
#include "cvwt/exception.hpp"

namespace cvwt
{
/**
 * @name Utilities
 * @{
 */
/**
 * @brief Flatten an array.
 *
 * @param[in] array
 * @param[out] result
 */
void flatten(cv::InputArray array, cv::OutputArray result);
/**
 * @brief Collect values indicated by the given mask.
 *
 * @param[in] array
 * @param[out] collected
 * @param mask
 */
void collect_masked(cv::InputArray array, cv::OutputArray collected, cv::InputArray mask);
/**
 * @brief Returns true if all values two matrices are equal.
 *
 * @param[in] a
 * @param[in] b
 */
bool matrix_equals(const cv::Mat& a, const cv::Mat& b);
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
 * @param matrix
 */
bool shares_data(const cv::Mat& a, const cv::Mat& b);
/**
 * @brief Negates all even indexed values.
 *
 * @param[in] vector A row or column vector.
 * @param[out] result The input vector with the even indexed values negated.
 */
void negate_evens(cv::InputArray vector, cv::OutputArray result);
/**
 * @brief Negates all odd indexed values.
 *
 * @param[in] vector A row or column vector.
 * @param[out] result The input vector with the odd indexed values negated.
 */
void negate_odds(cv::InputArray vector, cv::OutputArray result);
/**
 * @brief Returns true if array is cv::noArray()
 */
bool is_no_array(cv::InputArray array);
/**
 * @brief Joins a range of items into a deliminated string
 *
 * @param items
 * @param delim
 */
std::string join_string(const std::ranges::range auto& items, const std::string& delim = ", ");



double maximum_abs_value(cv::InputArray array, cv::InputArray mask = cv::noArray());

void patch_nans(cv::InputOutputArray array, double value = 0.0);
/** @}*/

/**
 * @name Statistics
 * @{
 */
/**
 * @brief Computes the multichannel median.
 *
 * @param[in] array
 * @param[in] mask
 * @return cv::Scalar
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
 * @param data The multichannel data.
 * @param mask A single channel matrix of type CV_8U where nonzero entries
 *             indicate which data locations are used.
 * @return cv::Scalar The estimated standard deviation of each channel.
 */
cv::Scalar mad(cv::InputArray data, cv::InputArray mask = cv::noArray());

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
/**
 * @}
 */


/**
 * @name Multichannel Comparison Functions
 * @{
 */
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
/** @}*/



namespace internal
{
std::string get_type_name(int type);
cv::Scalar set_unused_channels(const cv::Scalar& scalar, int channels, double value = 0.0);

template <template <typename T, int N, auto ...> typename Functor, auto ...TemplateArgs>
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
        "Dispatch for pixel type = ", get_type_name(type), " is not implemented."
    );
}

template <template <typename, typename, int, auto...> class Functor, typename T1>
struct BindFirst
{
    template <typename T2, int N, auto... Args>
    using type = Functor<T1, T2, N, Args...>;
};

template <template <typename, typename, int, auto...> class Functor, auto ...TemplateArgs>
auto dispatch_on_pixel_depths_and_same_channels(int type1, int type2, auto&&... args)
{
    if (CV_MAT_CN(type1) != CV_MAT_CN(type2)) {
        throw_bad_size(
            "Types must have the same number of channels. ",
            "Got type1 = ", get_type_name(type1),
            " and type2 = ", get_type_name(type2), "."
        );
    }

    switch (CV_MAT_DEPTH(type1)) {
        //  32 bit floating point
        case CV_32F: return dispatch_on_pixel_type<BindFirst<Functor, float>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  64 bit floating point
        case CV_64F: return dispatch_on_pixel_type<BindFirst<Functor, double>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  32 bit signed integer
        case CV_32S: return dispatch_on_pixel_type<BindFirst<Functor, int>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  16 bit signed integer
        case CV_16S: return dispatch_on_pixel_type<BindFirst<Functor, short>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  16 bit unsigned integer
        case CV_16U: return dispatch_on_pixel_type<BindFirst<Functor, ushort>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  8 bit signed integer
        case CV_8S: return dispatch_on_pixel_type<BindFirst<Functor, char>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  8 bit unsigned integer
        case CV_8U: return dispatch_on_pixel_type<BindFirst<Functor, uchar>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
    }

    throw_not_implemented(
        "Dispatch for pixel type = ", get_type_name(type1), " is not implemented."
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
        "Dispatch for pixel depth = ", get_type_name(CV_MAT_DEPTH(type)), " is not implemented."
    );
}

template <template <typename, auto ...> class Functor, auto ...TemplateArgs, typename ...ConstructorArgs>
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
        "Dispatch for pixel depth = ", get_type_name(CV_MAT_DEPTH(type)), " is not implemented."
    );
}

template <template <typename, typename, auto...> class Functor, typename T1>
struct BindFirstDepth
{
    template <typename T2, auto... Args>
    using type = Functor<T1, T2, Args...>;
};

template <template <typename, typename, auto...> class Functor, auto ...TemplateArgs>
auto dispatch_on_pixel_depths(int type1, int type2, auto&&... args)
{
    switch (CV_MAT_DEPTH(type1)) {
        //  32 bit floating point
        case CV_32F: return dispatch_on_pixel_depth<BindFirstDepth<Functor, float>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  64 bit floating point
        case CV_64F: return dispatch_on_pixel_depth<BindFirstDepth<Functor, double>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  32 bit signed integer
        case CV_32S: return dispatch_on_pixel_depth<BindFirstDepth<Functor, int>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  16 bit signed integer
        case CV_16S: return dispatch_on_pixel_depth<BindFirstDepth<Functor, short>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  16 bit unsigned integer
        case CV_16U: return dispatch_on_pixel_depth<BindFirstDepth<Functor, ushort>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  8 bit signed integer
        case CV_8S: return dispatch_on_pixel_depth<BindFirstDepth<Functor, char>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
        //  8 bit unsigned integer
        case CV_8U: return dispatch_on_pixel_depth<BindFirstDepth<Functor, uchar>::template type, TemplateArgs...>(
            type2,
            std::forward<decltype(args)>(args)...
        );
    }

    throw_not_implemented(
        "Dispatch for pixel type = ", get_type_name(type1), " is not implemented."
    );
}

template <typename T, int N>
struct CollectMasked
{
    using Pixel = cv::Vec<T, N>;

    void operator()(cv::InputArray input, cv::OutputArray output, cv::InputArray mask) const
    {
        assert(input.channels() == N);
        throw_if_bad_mask_type(mask);
        throw_if_bad_mask_for_array(input, mask, AllowedMaskChannels::SINGLE);
        if (input.empty()) {
            output.create(cv::Size(), input.type());
        } else {
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
        CollectMasked<T, N>()(array, masked_array, mask);

        return compute_median(masked_array);
    }

private:
    cv::Scalar compute_median(cv::Mat& array) const
    {
        assert(array.channels() == N);
        assert(!array.empty());
        assert(array.isContinuous());

        if (array.total() == 1) {
            return array.at<Pixel>(0);
        } else if (array.total() == 2) {
            return 0.5 * (array.at<Pixel>(0) + array.at<Pixel>(1));
        } else {
            cv::Scalar result;
            if constexpr (N == 1) {
                result[0] = single_channel_median(array);
            } else {
                cv::Mat channels[N];
                cv::split(array, channels);
                for (int i = 0; i < N; ++i)
                    result[i] = single_channel_median(channels[i]);
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

template <typename T, int CHANNELS, int EVEN_OR_ODD>
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
        else if constexpr (compare_type == cv::CMP_EQ)
            return x == y;
        else if constexpr (compare_type == cv::CMP_NE)
            return x != y;
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
        if (a.size() != b.size())
            throw_bad_size(
                "Cannot compare matrices of different sizes. ",
                "Got a.size() = ", a.size(),
                " and b.size() = ", b.size(), "."
            );
    }
};
}   // namespace internal
}   // namespace cvwt


#endif  // CVWT_UTILS_HPP

