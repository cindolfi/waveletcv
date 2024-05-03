#ifndef CVWT_UTILS_HPP
#define CVWT_UTILS_HPP

#include <span>
#include <ranges>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <iostream>

namespace cvwt
{
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
 * @param[out] result
 * @param mask
 */
void collect_masked(cv::InputArray array, cv::OutputArray result, cv::InputArray mask);
/**
 * @brief Returns true if all values two matrices are equal.
 *
 * @param[in] a
 * @param[in] b
 */
bool equals(const cv::Mat& a, const cv::Mat& b);
/**
 * @brief Returns true if two matrices refer to the same data.
 *
 * @param[in] a
 * @param[in] b
 */
bool identical(const cv::Mat& a, const cv::Mat& b);
/**
 * @brief Computes the multichannel median.
 *
 * @param[in] array
 * @param[in] mask
 * @return cv::Scalar
 */
cv::Scalar median(cv::InputArray array, cv::InputArray mask = cv::noArray());
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

namespace internal
{
std::string get_type_name(int type);


#define throw_error(code, ...) _throw_error(code, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_error(
    cv::Error::Code code,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    std::stringstream message;
    (message << ... << message_parts);

    cv::error(code, message.str(), function, file, line);
}

#define throw_bad_size(...) _throw_bad_size(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_bad_size(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(cv::Error::StsBadSize, function, file, line, message_parts...);
}

#define throw_if_empty(array, ...) _throw_if_empty(array, #array, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

void _throw_if_empty(
    cv::InputArray array,
    const char* array_name,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
)
{
    if (array.empty()) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_size(
                function, file, line,
                "The array `", array_name, "` cannot be empty."
            );
        else
            _throw_bad_size(function, file, line, message_parts...);
    }
}

#define throw_bad_arg(...) _throw_bad_arg(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_bad_arg(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(
        cv::Error::StsBadArg,
        function, file, line,
        message_parts...
    );
}

#define throw_out_of_range(...) _throw_out_of_range(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_out_of_range(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(
        cv::Error::StsOutOfRange,
        function, file, line,
        message_parts...
    );
}

#define throw_bad_mask(...) _throw_bad_mask(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_bad_mask(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(
        cv::Error::StsBadMask,
        function, file, line,
        message_parts...
    );
}

#define throw_if_bad_mask_type(mask, ...) _throw_if_bad_mask_type(mask, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

void _throw_if_bad_mask_type(
    cv::InputArray mask,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
)
{
    if (mask.type() != CV_8UC1 && mask.type() != CV_8SC1) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_mask(
                function, file, line,
                "Mask type must be CV_8UC1 or CV_8SC1, got ",
                get_type_name(mask.type()), ". "
            );
        else
            _throw_bad_mask(
                function, file, line,
                message_parts...,
                " [Mask type must be CV_8UC1 or CV_8SC1, got ",
                get_type_name(mask.type()), "]"
            );
    }
}

enum AllowedMaskChannels
{
    SINGLE,
    SAME,
    SINGLE_OR_SAME,
};

#define throw_if_bad_mask_for_array(array, mask, allowed_channels, ...) _throw_if_bad_mask_for_array(array, mask, #array, #mask, allowed_channels, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

void _throw_if_bad_mask_for_array(
    cv::InputArray array,
    cv::InputArray mask,
    const std::string& array_name,
    const std::string& mask_name,
    AllowedMaskChannels allowed_channels,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
)
{
    if (!array.sameSize(mask)) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_mask(
                function, file, line,
                "The `", mask_name, "` must be the same size as `", array_name, "`. "
                "Got ", array_name, ".size() = ", array.size(),
                " and ", mask_name, ".size() = ", mask.size(), ". "
            );
        else
            _throw_bad_mask(function, file, line, message_parts...);
    }

    switch (allowed_channels) {
    case AllowedMaskChannels::SINGLE:
        if (mask.channels() != 1) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    function, file, line,
                    "Wrong number of `", mask_name, "` channels for `", array_name, "`. "
                    "The number of `", mask_name, "` channels must be 1. "
                    "Got ", mask_name, ".channels() = ", mask.channels(), ". "
                );
            else
                _throw_bad_mask(function, file, line, message_parts...);
        }
        break;
    case AllowedMaskChannels::SAME:
        if (mask.channels() != array.channels()) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    function, file, line,
                    "Wrong number of `", mask_name, "` channels for `", array_name, "`. ",
                    "The number of `", mask_name, "` channels must be the same as `", array_name, "`. ",
                    "Got ", array_name, ".channels() = ", array.channels(),
                    " and ", mask_name, ".channels() = ", mask.channels(), "."
                );
            else
                _throw_bad_mask(function, file, line, message_parts...);
        }
        break;
    case AllowedMaskChannels::SINGLE_OR_SAME:
        if (mask.channels() != 1 && mask.channels() != array.channels()) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    function, file, line,
                    "The number of `", mask_name, "` channels must be 1 or the same as `", array_name, "`. ",
                    "Got ", array_name, ".channels() = ", array.channels(),
                    " and ", mask_name, ".channels() = ", mask.channels(), "."
                );
            else
                _throw_bad_mask(function, file, line, message_parts...);
        }
        break;
    }
}

#define throw_not_implemented(...) _throw_not_implemented(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_not_implemented(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(cv::Error::StsNotImplemented, function, file, line, message_parts...);
}

#define throw_member_not_implemented(class_name, function_name, ...) _throw_member_not_implemented(class_name, function_name, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_member_not_implemented(
    const std::string& class_name,
    const std::string& function_name,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
)  noexcept(false)
{
    _throw_not_implemented(
        function, file, line,
        class_name, "::", function_name, " is not implemented. ",
        message_parts...
    );
}

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
auto dispatch_on_pixel_depths(int type1, int type2, auto&&... args)
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
        case CV_32F: return dispatch_on_pixel_type<BindFirst<Functor, float>::template type, TemplateArgs...>(type2, std::forward<decltype(args)>(args)...);
        //  64 bit floating point
        case CV_64F: return dispatch_on_pixel_type<BindFirst<Functor, double>::template type, TemplateArgs...>(type2, std::forward<decltype(args)>(args)...);
        //  32 bit signed integer
        case CV_32S: return dispatch_on_pixel_type<BindFirst<Functor, int>::template type, TemplateArgs...>(type2, std::forward<decltype(args)>(args)...);
        //  16 bit signed integer
        case CV_16S: return dispatch_on_pixel_type<BindFirst<Functor, short>::template type, TemplateArgs...>(type2, std::forward<decltype(args)>(args)...);
        //  16 bit unsigned integer
        case CV_16U: return dispatch_on_pixel_type<BindFirst<Functor, ushort>::template type, TemplateArgs...>(type2, std::forward<decltype(args)>(args)...);
        //  8 bit signed integer
        case CV_8S: return dispatch_on_pixel_type<BindFirst<Functor, char>::template type, TemplateArgs...>(type2, std::forward<decltype(args)>(args)...);
        //  8 bit unsigned integer
        case CV_8U: return dispatch_on_pixel_type<BindFirst<Functor, uchar>::template type, TemplateArgs...>(type2, std::forward<decltype(args)>(args)...);
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
}   // namespace internal
}   // namespace cvwt




// class PlaneIterator
// {
// public:
//     using value_type = std::vector<cv::Mat>;
//     using reference_type = value_type&;
//     using difference_type = int;

// private:
//     PlaneIterator(std::shared_ptr<std::vector<cv::Mat>>&& arrays);

// public:
//     explicit PlaneIterator(std::initializer_list<cv::Mat> arrays) :
//         PlaneIterator(std::make_shared<std::vector<cv::Mat>>(arrays))
//     {}

//     explicit PlaneIterator(const std::vector<cv::Mat>& arrays) :
//         PlaneIterator(std::make_shared<std::vector<cv::Mat>>(arrays))
//     {}

//     explicit PlaneIterator(std::vector<cv::Mat>&& arrays) :
//         PlaneIterator(std::make_shared<std::vector<cv::Mat>>(arrays))
//     {}

//     explicit PlaneIterator(std::same_as<cv::Mat> auto... arrays) :
//         PlaneIterator({arrays...})
//     {}

//     PlaneIterator(const PlaneIterator& other) = default;
//     PlaneIterator(PlaneIterator&& other) = default;

//     PlaneIterator& operator=(const PlaneIterator& other) = default;
//     PlaneIterator& operator=(PlaneIterator&& other) = default;

//     const reference_type operator*() { return _planes; }
//     const reference_type operator->() { return _planes; }

//     PlaneIterator& operator++()
//     {
//         ++_channel;
//         gather_planes();
//         return *this;
//     }
//     PlaneIterator operator++(int) { auto copy = *this; ++*this; return copy; }

//     PlaneIterator operator--()
//     {
//         --_channel;
//         gather_planes();
//         return *this;
//     }
//     PlaneIterator operator--(int) { auto copy = *this; --*this; return copy; }

//     PlaneIterator operator+(difference_type offset) const
//     {
//         auto copy = *this;
//         copy._channel += offset;
//         return copy;
//     }
//     difference_type operator-(const PlaneIterator& rhs) const { return channel() - rhs.channel(); }

//     bool operator==(const PlaneIterator& other) const
//     {
//         return channel() == other.channel() && _arrays == other._arrays;
//     }

//     int channels() const { return _arrays->empty() ? 0 : _arrays->front().channels(); }
//     int channel() const { return _channel; }
//     int dims() const { return _arrays->empty() ? 2 : _arrays->front().dims; }

// protected:
//     void gather_planes();

// private:
//     std::shared_ptr<std::vector<cv::Mat>> _arrays;
//     std::vector<cv::Mat> _planes;
//     int _channel;
// };


// std::ranges::subrange<PlaneIterator> planes_range(const std::vector<cv::Mat>& arrays);
// // auto planes_range(std::vector<cv::Mat>&& arrays);
// std::ranges::subrange<PlaneIterator> planes_range(std::same_as<cv::Mat> auto... arrays)
// {
//     PlaneIterator planes_begin(arrays...);
//     return std::ranges::subrange(
//         planes_begin,
//         planes_begin + planes_begin.channels()
//     );
// }

// // auto planes_range(const std::vector<cv::Mat>& arrays)
// // {
// //     PlaneIterator planes_begin(arrays);
// //     return std::views::counted(
// //         planes_begin,
// //         planes_begin.channels()
// //     );
// // }

// // auto planes_range(std::vector<cv::Mat>&& arrays)
// // {
// //     PlaneIterator planes_begin(arrays);
// //     return std::views::counted(
// //         planes_begin,
// //         planes_begin.channels()
// //     );
// // }

// // auto planes_range(std::same_as<cv::Mat> auto... arrays)
// // {
// //     PlaneIterator planes_begin(arrays...);
// //     return std::views::counted(
// //         planes_begin,
// //         planes_begin.channels()
// //     );
// // }

#endif  // CVWT_UTILS_HPP

