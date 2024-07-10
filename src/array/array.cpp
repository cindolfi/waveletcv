#include "wtcv/array/array.hpp"

#include <atomic>
#include <limits>
#include "wtcv/dispatch.hpp"
#include "wtcv/exception.hpp"
#include "wtcv/array/compare.hpp"

namespace wtcv
{
namespace internal
{
template <typename T>
struct CollectMasked
{
    void operator()(cv::InputArray input, cv::OutputArray output, cv::InputArray mask) const
    {
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
    }
};

template <typename T>
using NegateEvenIndices = NegateEveryOther<T, 0>;

template <typename T>
using NegateOddIndices = NegateEveryOther<T, 1>;
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

bool is_data_shared(cv::InputArray a, cv::InputArray b)
{
    return a.getMat().datastart == b.getMat().datastart;
}

void negate_even_indices(cv::InputArray vector, cv::OutputArray result)
{
    internal::dispatch_on_pixel_depth<internal::NegateEvenIndices>(
        vector.depth(),
        vector,
        result
    );
}

void negate_odd_indices(cv::InputArray vector, cv::OutputArray result)
{
    internal::dispatch_on_pixel_depth<internal::NegateOddIndices>(
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
            // (scalar.total() == channels || scalar.total() == 1)
            //  single channel vector
            (scalar.total() == channels && scalar.channels() == 1)
            //  single element, multichannel vector
            || (scalar.total() == 1 && scalar.channels() == channels)
            //  single element, single channel vector - i.e. fundamental type
            || (scalar.total() == 1 && scalar.channels() == 1)
            //  cv::Scalar
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
}   // namespace wtcv
