#ifndef CVWT_UTILS_HPP
#define CVWT_UTILS_HPP

#include <span>
#include <ranges>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

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
 * @return cv::Scalar
 */
cv::Scalar median(cv::InputArray array);
/**
 * @brief Computes the multichannel median.
 *
 * @param[in] array
 * @param[in] mask
 * @return cv::Scalar
 */
cv::Scalar median(cv::InputArray array, cv::InputArray mask);
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



class PlaneIterator
{
public:
    using value_type = std::vector<cv::Mat>;
    using reference_type = value_type&;
    using difference_type = int;

private:
    PlaneIterator(std::shared_ptr<std::vector<cv::Mat>>&& arrays);

public:
    explicit PlaneIterator(std::initializer_list<cv::Mat> arrays) :
        PlaneIterator(std::make_shared<std::vector<cv::Mat>>(arrays))
    {}

    explicit PlaneIterator(const std::vector<cv::Mat>& arrays) :
        PlaneIterator(std::make_shared<std::vector<cv::Mat>>(arrays))
    {}

    explicit PlaneIterator(std::vector<cv::Mat>&& arrays) :
        PlaneIterator(std::make_shared<std::vector<cv::Mat>>(arrays))
    {}

    explicit PlaneIterator(std::same_as<cv::Mat> auto... arrays) :
        PlaneIterator({arrays...})
    {}

    PlaneIterator(const PlaneIterator& other) = default;
    PlaneIterator(PlaneIterator&& other) = default;

    PlaneIterator& operator=(const PlaneIterator& other) = default;
    PlaneIterator& operator=(PlaneIterator&& other) = default;

    const reference_type operator*() { return _planes; }
    const reference_type operator->() { return _planes; }

    PlaneIterator& operator++()
    {
        ++_channel;
        gather_planes();
        return *this;
    }
    PlaneIterator operator++(int) { auto copy = *this; ++*this; return copy; }

    PlaneIterator operator--()
    {
        --_channel;
        gather_planes();
        return *this;
    }
    PlaneIterator operator--(int) { auto copy = *this; --*this; return copy; }

    PlaneIterator operator+(difference_type offset) const
    {
        auto copy = *this;
        copy._channel += offset;
        return copy;
    }
    difference_type operator-(const PlaneIterator& rhs) const { return channel() - rhs.channel(); }

    bool operator==(const PlaneIterator& other) const
    {
        return channel() == other.channel() && _arrays == other._arrays;
    }

    int channels() const { return _arrays->empty() ? 0 : _arrays->front().channels(); }
    int channel() const { return _channel; }
    int dims() const { return _arrays->empty() ? 2 : _arrays->front().dims; }

protected:
    void gather_planes();

private:
    std::shared_ptr<std::vector<cv::Mat>> _arrays;
    std::vector<cv::Mat> _planes;
    int _channel;
};


std::ranges::subrange<PlaneIterator> planes_range(const std::vector<cv::Mat>& arrays);
// auto planes_range(std::vector<cv::Mat>&& arrays);
std::ranges::subrange<PlaneIterator> planes_range(std::same_as<cv::Mat> auto... arrays)
{
    PlaneIterator planes_begin(arrays...);
    return std::ranges::subrange(
        planes_begin,
        planes_begin + planes_begin.channels()
    );
}

// auto planes_range(const std::vector<cv::Mat>& arrays)
// {
//     PlaneIterator planes_begin(arrays);
//     return std::views::counted(
//         planes_begin,
//         planes_begin.channels()
//     );
// }

// auto planes_range(std::vector<cv::Mat>&& arrays)
// {
//     PlaneIterator planes_begin(arrays);
//     return std::views::counted(
//         planes_begin,
//         planes_begin.channels()
//     );
// }

// auto planes_range(std::same_as<cv::Mat> auto... arrays)
// {
//     PlaneIterator planes_begin(arrays...);
//     return std::views::counted(
//         planes_begin,
//         planes_begin.channels()
//     );
// }

namespace internal
{
std::string get_type_name(int type);

void throw_error(cv::Error::Code code, auto... message_parts)
{
    std::stringstream message;
    (message << ... << message_parts);

    CV_Error(code, message.str());
}

void throw_bad_size(auto... message_parts)
{
    throw_error(cv::Error::StsBadSize, message_parts...);
}

void throw_bad_arg(auto... message_parts)
{
    throw_error(cv::Error::StsBadArg, message_parts...);
}

void throw_out_of_range(auto... message_parts)
{
    throw_error(cv::Error::StsOutOfRange, message_parts...);
}

void throw_bad_mask(auto... message_parts)
{
    throw_error(cv::Error::StsBadMask, message_parts...);
}

void throw_if_bad_mask_type(cv::InputArray mask, auto... message_parts)
{
    if (mask.type() != CV_8UC1 || mask.type() != CV_8SC1)
        throw_bad_mask(
            "Mask type must be CV_8UC1 or CV_8SC1, got ",
            get_type_name(mask.type()), ". ",
            message_parts...
        );
}

void throw_not_implemented(auto... message_parts)
{
    throw_error(cv::Error::StsNotImplemented, message_parts...);
}

template <template <typename T, int N, auto ...> class Functor, auto ...TemplateArgs>
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

template <template <typename T, auto ...> class Functor, auto ...TemplateArgs>
void dispatch_on_pixel_depth(int type, auto&&... args)
{
    switch (CV_MAT_DEPTH(type)) {
        //  32 bit floating point
        case CV_32F: Functor<float, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  64 bit floating point
        case CV_64F: Functor<double, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  32 bit signed integer
        case CV_32S: Functor<int, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  16 bit signed integer
        case CV_16S: Functor<short, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  16 bit unsigned integer
        case CV_16U: Functor<ushort, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  8 bit signed integer
        case CV_8S: Functor<char, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
        //  8 bit unsigned integer
        case CV_8U: Functor<uchar, TemplateArgs...>()(std::forward<decltype(args)>(args)...); return;
    }
}

template <template <typename T, auto ...> class Functor, auto ...TemplateArgs, typename ...ConstructorArgs>
void dispatch_on_pixel_depth(std::tuple<ConstructorArgs...> constructor_args, int type, auto&&... args)
{
    switch (CV_MAT_DEPTH(type)) {
        //  32 bit floating point
        case CV_32F:
            auto create_float_functor = [](auto... cargs) {
                return Functor<float, TemplateArgs...>(cargs...);
            };
            auto float_functor = std::apply(create_float_functor, constructor_args);
            float_functor(std::forward<decltype(args)>(args)...);
            return;
        //  64 bit floating point
        case CV_64F:
            // auto functor = std::apply(Functor<double, TemplateArgs...>, constructor_args);
            auto create_double_functor = [](auto... cargs) {
                return Functor<double, TemplateArgs...>(cargs...);
            };
            auto double_functor = std::apply(create_double_functor, constructor_args);
            double_functor(std::forward<decltype(args)>(args)...);
            return;
        //  32 bit signed integer
        case CV_32S:
            // auto functor = std::apply(Functor<int, TemplateArgs...>, constructor_args);
            auto create_int_functor = [](auto... cargs) {
                return Functor<int, TemplateArgs...>(cargs...);
            };
            auto int_functor = std::apply(create_int_functor, constructor_args);
            int_functor(std::forward<decltype(args)>(args)...);
            return;
        //  16 bit signed integer
        case CV_16S:
            // auto functor = std::apply(Functor<short, TemplateArgs...>, constructor_args);
            auto create_short_functor = [](auto... cargs) {
                return Functor<short, TemplateArgs...>(cargs...);
            };
            auto short_functor = std::apply(create_short_functor, constructor_args);
            short_functor(std::forward<decltype(args)>(args)...);
            return;
        //  16 bit unsigned integer
        case CV_16U:
            // auto functor = std::apply(Functor<ushort, TemplateArgs...>, constructor_args);
            auto create_ushort_functor = [](auto... cargs) {
                return Functor<ushort, TemplateArgs...>(cargs...);
            };
            auto ushort_functor = std::apply(create_ushort_functor, constructor_args);
            ushort_functor(std::forward<decltype(args)>(args)...);
            return;
        //  8 bit signed integer
        case CV_8S:
            // auto functor = std::apply(Functor<char, TemplateArgs...>, constructor_args);
            auto create_char_functor = [](auto... cargs) {
                return Functor<char, TemplateArgs...>(cargs...);
            };
            auto char_functor = std::apply(create_char_functor, constructor_args);
            char_functor(std::forward<decltype(args)>(args)...);
            return;
        //  8 bit unsigned integer
        case CV_8U:
            // auto functor = std::apply(Functor<uchar, TemplateArgs...>, constructor_args);
            auto create_uchar_functor = [](auto... cargs) {
                return Functor<uchar, TemplateArgs...>(cargs...);
            };
            auto uchar_functor = std::apply(create_uchar_functor, constructor_args);
            uchar_functor(std::forward<decltype(args)>(args)...);
            return;
    }
}

template <typename T, int N>
struct collect_masked
{
    using Pixel = cv::Vec<T, N>;

    void operator()(cv::InputArray input, cv::OutputArray output, cv::InputArray mask) const
    {
        assert(input.channels() == N);
        throw_if_bad_mask_type(mask);

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

    void operator()(cv::InputArray array, cv::Scalar& result) const
    {
        auto matrix = array.getMat();
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

    void operator()(cv::InputArray array, cv::InputArray mask, cv::Scalar& result) const
    {
        cv::Mat masked_array;
        collect_masked<T, N>()(array, masked_array, mask);
        this->operator()(masked_array, result);
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

#endif  // CVWT_UTILS_HPP

