#include "cvwt/utils.hpp"

#include <iostream>
#include <sstream>
#include <experimental/iterator>

namespace cvwt
{
void flatten(cv::InputArray array, cv::OutputArray result)
{
    array.copyTo(result);
    result.assign(result.getMat().reshape(0, array.total()));
}

void collect_masked(cv::InputArray array, cv::OutputArray collected, cv::InputArray mask)
{
    internal::dispatch_on_pixel_type<internal::CollectMasked>(
        array.type(),
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

cv::Scalar mad(cv::InputArray data, cv::InputArray mask)
{
    auto x = data.getMat();
    return median(cv::abs(x - median(x, mask)), mask);
}

cv::Scalar mad_stdev(cv::InputArray data)
{
    return mad(data) / 0.675;
}

void negate_evens(cv::InputArray vector, cv::OutputArray result)
{
    internal::dispatch_on_pixel_type<internal::NegateEveryOther, 0>(
        vector.type(),
        vector,
        result
    );
}

void negate_odds(cv::InputArray vector, cv::OutputArray result)
{
    internal::dispatch_on_pixel_type<internal::NegateEveryOther, 1>(
        vector.type(),
        vector,
        result
    );
}

bool is_no_array(cv::InputArray array)
{
    return array.kind() == cv::_InputArray::NONE;
}

std::string join_string(const std::ranges::range auto& items, const std::string& delim)
{
    std::stringstream stream;
    auto joiner = std::experimental::make_ostream_joiner(stream, delim);
    std::copy(items.begin(), items.end(), joiner);

    return stream.str();
}


void patch_nans(cv::InputOutputArray array, double value)
{
    cv::Mat nan_mask;
    compare(array, array, nan_mask, cv::CMP_NE);
    array.setTo(value, nan_mask);
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
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_LT>(
            a.type(), b.type(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_LT>(
            a.type(), b.type(), a, b, output, mask
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
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_LE>(
            a.type(), b.type(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_LE>(
            a.type(), b.type(), a, b, output, mask
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
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_GT>(
            a.type(), b.type(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_GT>(
            a.type(), b.type(), a, b, output, mask
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
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_GE>(
            a.type(), b.type(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_GE>(
            a.type(), b.type(), a, b, output, mask
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
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_EQ>(
            a.type(), b.type(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_EQ>(
            a.type(), b.type(), a, b, output, mask
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
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_NE>(
            a.type(), b.type(), a, b, output
        );
    else
        internal::dispatch_on_pixel_depths_and_same_channels<internal::Compare, cv::CMP_NE>(
            a.type(), b.type(), a, b, output, mask
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
}   // namespace internal
}   // namespace cvwt
