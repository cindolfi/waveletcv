#include "cvwt/utils.hpp"

#include <iostream>
#include <sstream>
#include <experimental/iterator>

namespace cvwt
{
void flatten(cv::InputArray array, cv::OutputArray result)
{
    // cv::Mat matrix;
    // if (array.isSubmatrix())
    //     array.copyTo(matrix);
    // else
    //     matrix = array.getMat();

    // matrix.reshape(0, 1).copyTo(result);

    array.copyTo(result);
    result.assign(result.getMat().reshape(0, array.total()));
}

void collect_masked(cv::InputArray array, cv::OutputArray result, cv::InputArray mask)
{
    internal::dispatch_on_pixel_type<internal::CollectMasked>(
        array.type(),
        array,
        result,
        mask
    );
}

bool equals(const cv::Mat& a, const cv::Mat& b)
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

// cv::Scalar median(cv::InputArray array)
// {
//     cv::Scalar result;
//     internal::dispatch_on_pixel_type<internal::Median>(
//         array.type(),
//         array,
//         result
//     );

//     return result;
// }

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
}   // namespace internal
}   // namespace cvwt


// PlaneIterator::PlaneIterator(std::shared_ptr<std::vector<cv::Mat>>&& arrays) :
//     _arrays(arrays),
//     _planes(_arrays->size()),
//     _channel(0)
// {
//     if (!_arrays->empty()) {
//         int dims = _arrays->front().dims;
//         auto equal_dims = [&](const auto& array) { return array.dims == dims; };
//         if ((dims != 2 && dims != 3) || !std::ranges::all_of(*_arrays, equal_dims)) {
//             auto get_dims = [](const auto& array) { return array.dims; };
//             std::string all_dims = join_string(
//                 *_arrays | std::views::transform(get_dims)
//             );
//             internal::throw_bad_arg(
//                 "All arrays must be 2 or 3-dimensional, got dimensions ",
//                 all_dims, "."
//             );
//         }

//         auto equal_channels = [&](const auto& array) { return array.channels() == channels(); };
//         if (!std::ranges::all_of(*_arrays, equal_channels)) {
//             auto get_channels = [](const auto& array) { return array.channels(); };
//             std::string all_channels = join_string(
//                 *_arrays | std::views::transform(get_channels)
//             );
//             internal::throw_bad_arg(
//                 "All arrays must have the same number of channels, got ",
//                 all_channels, "."
//             );
//         }
//     }

//     gather_planes();
// }

// void PlaneIterator::gather_planes()
// {
//     if (channel() >= channels() || channel() < 0) {
//         std::cout << channel() << "  " << channels() << "\n";
//         // assert(false);
//         for (int i = 0; i < _arrays->size(); ++i) {
//             _planes[i] = cv::Mat(0, 0, (*_arrays)[i].depth());
//         }
//     } else {
//         if (dims() == 5) {
//             for (int i = 0; i < _arrays->size(); ++i)
//                 _planes[i] = (*_arrays)[i];
//         } else {
//             for (int i = 0; i < _arrays->size(); ++i) {
//                 auto& array = (*_arrays)[i];
//                 _planes[i] = cv::Mat(
//                     std::vector<int>({array.rows, array.cols}),
//                     array.depth(),
//                     array.data + channel() * array.elemSize1(),
//                     array.step.p
//                 );
//                 _planes[i].step.p[1] = array.step.p[1];
//                 // _planes[i] = cv::Mat(
//                 //     array.size(),
//                 //     array.depth(),
//                 //     array.data + channel() * array.step[2],
//                 //     array.step[0]
//                 // );
//             }
//         }
//     }
// }

// std::ranges::subrange<PlaneIterator> planes_range(const std::vector<cv::Mat>& arrays)
// {
//     PlaneIterator planes_begin(arrays);
//     return std::ranges::subrange(
//         planes_begin,
//         planes_begin + planes_begin.channels()
//     );
// }

// // auto planes_range(std::vector<cv::Mat>&& arrays)
// // {
// //     PlaneIterator planes_begin(arrays);
// //     return std::views::counted(
// //         planes_begin,
// //         planes_begin.channels()
// //     );
// // }

// // std::ranges::subrange<PlaneIterator> planes_range(std::same_as<cv::Mat> auto... arrays)
// // {
// //     PlaneIterator planes_begin(arrays...);
// //     return std::ranges::subrange(
// //         planes_begin,
// //         planes_begin + planes_begin.channels()
// //     );
// // }