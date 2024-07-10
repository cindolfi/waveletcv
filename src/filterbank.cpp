#include "wtcv/filterbank.hpp"

#include <opencv2/imgproc.hpp>
#include "wtcv/utils.hpp"
#include "wtcv/array/array.hpp"
#include "wtcv/dispatch.hpp"
#include "wtcv/array/compare.hpp"
#include "wtcv/exception.hpp"

namespace wtcv
{
namespace internal
{
int convolve_output_type(int input_type, int kernel_type)
{
    return CV_MAKE_TYPE(
        std::max(CV_MAT_DEPTH(input_type), CV_MAT_DEPTH(kernel_type)),
        std::max(CV_MAT_CN(input_type), CV_MAT_CN(kernel_type))
    );
}

template<typename T1, typename T2>
using MultiplyType = decltype(std::declval<T1>() * std::declval<T2>());

template <
    typename T1,
    typename T2,
    typename OutputType
>
inline void accum_dot_single_with_multi_channel(
    const cv::Mat& single_channel_array,
    const cv::Mat& multi_channel_array,
    OutputType* output_pixel
)
{
    assert(single_channel_array.size() == single_channel_array.size());
    assert(single_channel_array.channels() == 1);

    for (int i = 0; i < single_channel_array.rows; ++i) {
        for (int j = 0; j < single_channel_array.cols; ++j) {
            auto a = single_channel_array.at<T1>(i, j);
            auto b = multi_channel_array.ptr<T2>(i, j);
            for (int k = 0; k < multi_channel_array.channels(); ++k)
                output_pixel[k] += a * b[k];
        }
    }
}

template <typename InputType, typename KernelType>
struct ConvolveRowsAndDownsampleCols
{
    using OutputType = MultiplyType<InputType, KernelType>;

    /**
     * Input is assumed to be padded, kernel is assumed to be flipped and transposed (if necessary)
    */
    void operator()(cv::InputArray input, cv::OutputArray output, const cv::Mat& kernel)
    {
        auto output_size = input.size() - kernel.size() + cv::Size(1, 1);
        output_size.width = output_size.width / 2;
        output.assign(
            cv::Mat::zeros(
                output_size,
                convolve_output_type(input.type(), kernel.type())
            )
        );
        const auto input_matrix = input.getMat();
        auto output_matrix = output.getMat();
        cv::parallel_for_(
            cv::Range(0, output_matrix.total()),
            [&](const cv::Range& range) {
                for (int k = range.start; k < range.end; ++k) {
                    const int index[2] = {k / output_matrix.cols, k % output_matrix.cols};
                    accum_dot_single_with_multi_channel<KernelType, InputType>(
                        kernel,
                        input_matrix(cv::Rect(
                            cv::Point(2 * index[1], index[0]),
                            kernel.size()
                        )),
                        output_matrix.ptr<OutputType>(index)
                    );
                }
            }
        );
    }
};

template <typename InputType, typename KernelType>
struct ConvolveColsAndDownsampleRows
{
    using OutputType = MultiplyType<InputType, KernelType>;

    /**
     * Input is assumed to be padded, kernel is assumed to be flipped and transposed (if necessary)
    */
    void operator()(cv::InputArray input, cv::OutputArray output, const cv::Mat& kernel)
    {
        auto output_size = input.size() - kernel.size() + cv::Size(1, 1);
        output_size.height = output_size.height / 2;
        output.assign(
            cv::Mat::zeros(
                output_size,
                convolve_output_type(input.type(), kernel.type())
            )
        );
        const auto input_matrix = input.getMat();
        auto output_matrix = output.getMat();
        cv::parallel_for_(
            cv::Range(0, output_matrix.total()),
            [&](const cv::Range& range) {
                for (int k = range.start; k < range.end; ++k) {
                    const int index[2] = {k / output_matrix.cols, k % output_matrix.cols};
                    accum_dot_single_with_multi_channel<KernelType, InputType>(
                        kernel,
                        input_matrix(cv::Rect(
                            cv::Point(index[1], 2 * index[0]),
                            kernel.size()
                        )),
                        output_matrix.ptr<OutputType>(index)
                    );
                }
            }
        );
    }
};

template <typename InputType, typename KernelType>
struct UpsampleRowsAndConvolveCols
{
    using OutputType = MultiplyType<InputType, KernelType>;

    /**
     * Kernels are assumed to be flipped and transposed (if necessary)
    */
    void operator()(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Mat& even_index_kernel,
        const cv::Mat& odd_index_kernel,
        const cv::Size& image_size
    ) const
    {
        if (output.empty())
            output.assign(cv::Mat::zeros(
                cv::Size(input.size().width, image_size.height),
                convolve_output_type(input.type(), even_index_kernel.type())
            ));

        const auto input_matrix = input.getMat();
        auto output_matrix = output.getMat();
        cv::parallel_for_(
            cv::Range(0, output_matrix.total()),
            [&](const cv::Range& range) {
                for (int k = range.start; k < range.end; ++k) {
                    const int index[2] = {k / output_matrix.cols, k % output_matrix.cols};
                    const cv::Mat& kernel = (index[0] % 2 == 1)
                                          ? odd_index_kernel
                                          : even_index_kernel;
                    accum_dot_single_with_multi_channel<KernelType, InputType>(
                        kernel,
                        input_matrix(cv::Rect(
                            cv::Point(index[1], index[0] / 2),
                            kernel.size()
                        )),
                        output_matrix.ptr<OutputType>(index)
                    );
                }
            }
        );
    }
};

template <typename InputType, typename KernelType>
struct UpsampleColsAndConvolveRows
{
    using OutputType = MultiplyType<InputType, KernelType>;

    /**
     * Kernels are assumed to be flipped and transposed (if necessary)
    */
    void operator()(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Mat& even_index_kernel,
        const cv::Mat& odd_index_kernel,
        const cv::Size& image_size
    ) const
    {
        if (output.empty())
            output.assign(cv::Mat::zeros(
                cv::Size(image_size.width, input.size().height),
                convolve_output_type(input.type(), even_index_kernel.type())
            ));

        const auto input_matrix = input.getMat();
        auto output_matrix = output.getMat();
        cv::parallel_for_(
            cv::Range(0, output_matrix.total()),
            [&](const cv::Range& range) {
                for (int k = range.start; k < range.end; ++k) {
                    const int index[2] = {k / output_matrix.cols, k % output_matrix.cols};
                    const cv::Mat& kernel = (index[1] % 2 == 1)
                                          ? odd_index_kernel
                                          : even_index_kernel;
                    accum_dot_single_with_multi_channel<KernelType, InputType>(
                        kernel,
                        input_matrix(cv::Rect(
                            cv::Point(index[1] / 2, index[0]),
                            kernel.size()
                        )),
                        output_matrix.ptr<OutputType>(index)
                    );
                }
            }
        );
    }
};

template <typename T>
struct SplitKernelIntoOddAndEvenParts
{
    void operator()(cv::InputArray full_kernel, cv::OutputArray even_output, cv::OutputArray odd_output) const
    {
        //  Build the kernels to use for even output rows/columns and odd output
        //  rows/columns.
        //  We are simulating upsampling by virtually interlacing zeros between
        //  each input. All even rows/columns in this virtually upsampled input
        //  are all zeros.  As such, only odd indexed kernel values are used
        //  for even output rows/columns. Likewise, only even indexed kernel
        //  values are used for odd output rows/columns.
        auto kernel = full_kernel.getMat();
        assert(kernel.rows == 1 || kernel.cols == 1);
        int kernel_length = std::max(kernel.rows, kernel.cols);

        cv::Size even_size(kernel_length / 2 + kernel_length % 2, 1);
        cv::Size odd_size(kernel_length / 2, 1);
        if (kernel.cols == 1) {
            std::swap(even_size.width, even_size.height);
            std::swap(odd_size.width, odd_size.height);
        }

        even_output.create(even_size, kernel.type());
        auto even_kernel = even_output.getMat();
        for (int i = 0; i < even_kernel.total(); ++i)
            even_kernel.at<T>(i) = kernel.at<T>(2 * i + 1);

        odd_output.create(odd_size, kernel.type());
        auto odd_kernel = odd_output.getMat();
        for (int i = 0; i < odd_kernel.total(); ++i)
            odd_kernel.at<T>(i) = kernel.at<T>(2 * i);
    }
};

template <typename T>
struct MergeEvenAndOddKernels
{
    void operator()(cv::InputArray even_kernel_input, cv::InputArray odd_kernel_input, cv::OutputArray full_kernel) const
    {
        cv::Mat even_kernel = even_kernel_input.getMat();
        cv::Mat odd_kernel = odd_kernel_input.getMat();

        int kernel_length = even_kernel.total() + odd_kernel.total();
        cv::Size kernel_size(kernel_length, 1);
        if (even_kernel.cols == 1)
            std::swap(kernel_size.width, kernel_size.height);

        full_kernel.create(kernel_size, even_kernel.type());
        auto kernel = full_kernel.getMat();
        for (int i = 0; i < kernel.total(); ++i) {
            //  This is correct.  The even/odd naming convention applies to the
            //  even/odd-ness of the image rows & columns, which is opposite
            //  to the even/odd-ness of the kernel indices.
            cv::Mat& source_kernel = (i % 2 == 0) ? odd_kernel : even_kernel;
            kernel.at<T>(i) = source_kernel.at<T>(i / 2);
        }
    }
};

template <typename T>
struct StripZeros
{
    void operator()(cv::InputArray kernel, cv::OutputArray stripped_kernel) const
    {
        const int channels = kernel.channels();
        auto array = kernel.getMat();
        int left = 0;
        while (is_zero(array, left) && left < array.total())
            ++left;

        int right = array.total() - 1;
        while (is_zero(array, right) && right >= 0)
            --right;

        array.rowRange(left, right + 1).copyTo(stripped_kernel);
    }

    bool is_zero(const cv::Mat& array, int index) const
    {
        auto element = array.ptr<T>(index);
        return std::all_of(
            element,
            element + array.channels(),
            [](auto x) { return x == 0.0; }
        );
    }
};

void strip_zeros(cv::InputArray kernel, cv::OutputArray stripped_kernel)
{
    internal::dispatch_on_pixel_depth<internal::StripZeros>(
        kernel.depth(), kernel, stripped_kernel
    );
}

void strided_correlate(
    cv::InputArray a,
    cv::InputArray b,
    cv::OutputArray result,
    int stride
)
{
    assert(is_vector(a, 1));
    assert(is_vector(b, 1));
    assert(a.size() == b.size());
    assert(stride >= 1);

    cv::filter2D(a, result, -1, b, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    if (stride > 1) {
        cv::resize(
            result,
            result,
            cv::Size(1, a.total() / stride),
            0, 0,
            cv::INTER_NEAREST
        );
    }
}

cv::Mat convolve(cv::InputArray x, cv::InputArray y)
{
    cv::Mat result;
    cv::Mat flipped_y;
    cv::flip(y, flipped_y, -1);
    cv::filter2D(x, result, -1, flipped_y, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);

    return result;
};

cv::Mat as_column_vector(const cv::Mat& vector)
{
    assert(vector.rows == 1 || vector.cols == 1);
    return (vector.cols == 1) ? vector : vector.t();
}

FilterBankImpl::FilterBankImpl() :
    filter_length(0),
    decompose(),
    reconstruct()
{
}

FilterBankImpl::FilterBankImpl(
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass,
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass
) :
    filter_length(std::max(decompose_lowpass.total(), reconstruct_lowpass.total()))
{
    throw_if_wrong_size(
        reconstruct_lowpass,
        reconstruct_highpass,
        decompose_lowpass,
        decompose_highpass
    );
    throw_if_wrong_type(
        reconstruct_lowpass,
        reconstruct_highpass,
        decompose_lowpass,
        decompose_highpass
    );

    cv::flip(as_column_vector(decompose_lowpass), decompose.lowpass, -1);
    cv::flip(as_column_vector(decompose_highpass), decompose.highpass, -1);

    cv::Mat flipped_reconstruct_lowpass;
    cv::flip(as_column_vector(reconstruct_lowpass), flipped_reconstruct_lowpass, -1);
    split_kernel_into_odd_and_even_parts(
        flipped_reconstruct_lowpass,
        reconstruct.even_lowpass,
        reconstruct.odd_lowpass
    );

    cv::Mat flipped_reconstruct_highpass;
    cv::flip(as_column_vector(reconstruct_highpass), flipped_reconstruct_highpass, -1);
    split_kernel_into_odd_and_even_parts(
        flipped_reconstruct_highpass,
        reconstruct.even_highpass,
        reconstruct.odd_highpass
    );
}

void FilterBankImpl::split_kernel_into_odd_and_even_parts(
    cv::InputArray kernel,
    cv::OutputArray even_kernel,
    cv::OutputArray odd_kernel
) const
{
    dispatch_on_pixel_depth<SplitKernelIntoOddAndEvenParts>(
        kernel.depth(),
        kernel,
        even_kernel,
        odd_kernel
    );
}

void FilterBankImpl::merge_even_and_odd_kernels(
    cv::InputArray even_kernel,
    cv::InputArray odd_kernel,
    cv::OutputArray kernel
) const
{
    dispatch_on_pixel_depth<MergeEvenAndOddKernels>(
        even_kernel.depth(),
        even_kernel,
        odd_kernel,
        kernel
    );
}

KernelPair FilterBankImpl::decompose_kernels() const
{
    cv::Mat lowpass;
    cv::flip(decompose.lowpass, lowpass, -1);
    cv::Mat highpass;
    cv::flip(decompose.highpass, highpass, -1);

    return KernelPair(lowpass, highpass);
}

KernelPair FilterBankImpl::reconstruct_kernels() const
{
    cv::Mat lowpass;
    merge_even_and_odd_kernels(reconstruct.even_lowpass, reconstruct.odd_lowpass, lowpass);
    cv::flip(lowpass, lowpass, -1);

    cv::Mat highpass;
    merge_even_and_odd_kernels(reconstruct.even_highpass, reconstruct.odd_highpass, highpass);
    cv::flip(highpass, highpass, -1);

    return KernelPair(lowpass, highpass);
}

inline
void FilterBankImpl::throw_if_wrong_size(
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass,
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass,
    const std::source_location& location
) const CVWT_FILTER_BANK_NOEXCEPT
{
#if CVWT_FILTER_BANK_EXCEPTIONS_ENABLED
    bool all_empty = decompose_lowpass.empty()
        && decompose_highpass.empty()
        && reconstruct_lowpass.empty()
        && reconstruct_highpass.empty();

    if (!all_empty) {
        bool all_nonempty = !decompose_lowpass.empty()
            && !decompose_highpass.empty()
            && !reconstruct_lowpass.empty()
            && !reconstruct_highpass.empty();

        if (!all_nonempty) {
            throw_bad_arg(
                "FilterBank: Kernels must all be empty or all nonempty. Got ",
                (reconstruct_lowpass.empty() ? "empty" : "nonempty"), " reconstruct_lowpass, ",
                (reconstruct_highpass.empty() ? "empty" : "nonempty"), " reconstruct_highpass, ",
                (decompose_lowpass.empty() ? "empty" : "nonempty"), " decompose_lowpass, and ",
                (decompose_highpass.empty() ? "empty" : "nonempty"), " decompose_highpass.",
                location
            );
        }

        if (decompose_lowpass.size() != decompose_highpass.size()
            || !is_vector(decompose_lowpass, 1)) {
            throw_bad_size(
                "FilterBank: decompose_lowpass and decompose_highpass kernels "
                "must be single channel vectors of the same size. ",
                "Got decompose_lowpass.size() = ", decompose_lowpass.size(),
                " and decompose_highpass.size() = ", decompose_highpass.size(), ". ",
                "Got decompose_lowpass.channels() = ", decompose_lowpass.channels(),
                " and decompose_highpass.channels() = ", decompose_highpass.channels(), ".",
                location
            );
        }

        if (reconstruct_lowpass.size() != reconstruct_highpass.size()
            || !is_vector(reconstruct_lowpass, 1)) {
            throw_bad_size(
                "FilterBank: reconstruct_lowpass and reconstruct_highpass kernels "
                "must be single channel vectors of the same size. ",
                "Got reconstruct_lowpass.size() = ", reconstruct_lowpass.size(),
                " and reconstruct_highpass.size() = ", reconstruct_highpass.size(), ". ",
                "Got reconstruct_lowpass.channels() = ", reconstruct_lowpass.channels(),
                " and reconstruct_highpass.channels() = ", reconstruct_highpass.channels(), ".",
                location
            );
        }
    }
#endif
}

inline
void FilterBankImpl::throw_if_wrong_type(
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass,
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass,
    const std::source_location& location
) const CVWT_FILTER_BANK_NOEXCEPT
{
#if CVWT_FILTER_BANK_EXCEPTIONS_ENABLED
    bool all_same_type = decompose_lowpass.type() == decompose_highpass.type()
        && decompose_lowpass.type() == reconstruct_lowpass.type()
        && decompose_lowpass.type() == reconstruct_highpass.type();

    if (!all_same_type) {
        throw_bad_arg(
            "FilterBank: Kernels must all be the same type. Got ",
            "reconstruct_lowpass.type() = ", get_type_name(reconstruct_lowpass.type()), ", ",
            "reconstruct_highpass.type() = ", get_type_name(reconstruct_highpass.type()), ",  ",
            "decompose_lowpass.type() = ", get_type_name(decompose_lowpass.type()), ", and ",
            "decompose_highpass.type() = ", get_type_name(decompose_highpass.type()), ".",
            location
        );
    }
#endif
}
} // namespace internal


//  ============================================================================
//  Public API
//  ============================================================================
//  ----------------------------------------------------------------------------
//  KernelPair
//  ----------------------------------------------------------------------------
KernelPair KernelPair::as_type(int type) const
{
    cv::Mat lowpass;
    _lowpass.convertTo(lowpass, type);

    cv::Mat highpass;
    _highpass.convertTo(highpass, type);

    return KernelPair(lowpass, highpass);
}

KernelPair make_kernel_pair(cv::InputArray lowpass, cv::InputArray highpass)
{
    if (lowpass.size() != highpass.size()) {
        throw_bad_size(
            "Kernels must be the same size. ",
            "Got lowpass.size() = ", lowpass.size(),
            " and highpass.size() = ", highpass.size(), "."
        );
    }

    if (lowpass.type() != highpass.type()) {
        throw_bad_arg(
            "Kernels must be the same type. ",
            "Got lowpass.type() = ", internal::get_type_name(lowpass.type()),
            " and highpass.type() = ", internal::get_type_name(highpass.type()), "."
        );
    }

    return KernelPair(lowpass.getMat(), highpass.getMat());
}

bool KernelPair::operator==(const KernelPair& other) const
{
    return this == &other || (
        is_equal(_lowpass, other._lowpass)
        && is_equal(_highpass, other._highpass)
    );
}

//  ----------------------------------------------------------------------------
//  FilterBank
//  ----------------------------------------------------------------------------
FilterBank::FilterBank() :
    _p(std::make_shared<internal::FilterBankImpl>())
{
}

FilterBank::FilterBank(
    cv::InputArray decompose_lowpass,
    cv::InputArray decompose_highpass,
    cv::InputArray reconstruct_lowpass,
    cv::InputArray reconstruct_highpass
) :
    _p(
        std::make_shared<internal::FilterBankImpl>(
            decompose_lowpass.getMat(),
            decompose_highpass.getMat(),
            reconstruct_lowpass.getMat(),
            reconstruct_highpass.getMat()
        )
    )
{
}

FilterBank FilterBank::as_type(int type) const
{
    if (type != CV_64FC1 && type != CV_32FC1)
        throw_bad_arg(
            "Filter bank type must be CV_64FC1 or CV_32FC1. Got ",
            internal::get_type_name(type), "."
        );

    if (type == this->type())
        return *this;

    auto decompose_kernels = this->decompose_kernels().as_type(type);
    auto reconstruct_kernels = this->reconstruct_kernels().as_type(type);

    return FilterBank(
        decompose_kernels.lowpass(),
        decompose_kernels.highpass(),
        reconstruct_kernels.lowpass(),
        reconstruct_kernels.highpass()
    );
}

void FilterBank::decompose(
    cv::InputArray image,
    cv::OutputArray approx,
    cv::OutputArray horizontal_detail,
    cv::OutputArray vertical_detail,
    cv::OutputArray diagonal_detail,
    int border_type,
    const cv::Scalar& border_value
) const
{
    throw_if_decompose_image_is_wrong_size(image);

    cv::Mat exptrapolated_image;
    extrapolate_border(image, exptrapolated_image, border_type, border_value);

    //  Stage 1
    cv::Mat stage1_lowpass_output;
    convolve_rows_and_downsample_cols(
        exptrapolated_image,
        stage1_lowpass_output,
        _p->decompose.lowpass
    );
    cv::Mat stage1_highpass_output;
    convolve_rows_and_downsample_cols(
        exptrapolated_image,
        stage1_highpass_output,
        _p->decompose.highpass
    );

    //  Stage 2
    convolve_cols_and_downsample_rows(
        stage1_lowpass_output,
        approx,
        _p->decompose.lowpass
    );
    convolve_cols_and_downsample_rows(
        stage1_lowpass_output,
        horizontal_detail,
        _p->decompose.highpass
    );
    convolve_cols_and_downsample_rows(
        stage1_highpass_output,
        vertical_detail,
        _p->decompose.lowpass
    );
    convolve_cols_and_downsample_rows(
        stage1_highpass_output,
        diagonal_detail,
        _p->decompose.highpass
    );
}


void FilterBank::reconstruct(
    cv::InputArray approx,
    cv::InputArray horizontal_detail,
    cv::InputArray vertical_detail,
    cv::InputArray diagonal_detail,
    cv::OutputArray image,
    const cv::Size& image_size
) const
{
    throw_if_reconstruct_coeffs_are_wrong_size(
        approx,
        horizontal_detail,
        vertical_detail,
        diagonal_detail
    );

    //  Stage 1a
    cv::Mat stage1a_output;
    upsample_rows_and_convolve_cols(
        approx.getMat(),
        stage1a_output,
        _p->reconstruct.even_lowpass,
        _p->reconstruct.odd_lowpass,
        image_size
    );
    upsample_rows_and_convolve_cols(
        horizontal_detail.getMat(),
        stage1a_output,
        _p->reconstruct.even_highpass,
        _p->reconstruct.odd_highpass,
        image_size
    );

    //  Stage 1b
    cv::Mat stage1b_output;
    upsample_rows_and_convolve_cols(
        vertical_detail.getMat(),
        stage1b_output,
        _p->reconstruct.even_lowpass,
        _p->reconstruct.odd_lowpass,
        image_size
    );
    upsample_rows_and_convolve_cols(
        diagonal_detail.getMat(),
        stage1b_output,
        _p->reconstruct.even_highpass,
        _p->reconstruct.odd_highpass,
        image_size
    );

    //  Stage 2
    cv::Mat stage2_output;
    upsample_cols_and_convolve_rows(
        stage1a_output,
        image,
        _p->reconstruct.even_lowpass,
        _p->reconstruct.odd_lowpass,
        image_size
    );
    upsample_cols_and_convolve_rows(
        stage1b_output,
        image,
        _p->reconstruct.even_highpass,
        _p->reconstruct.odd_highpass,
        image_size
    );
}

cv::Size FilterBank::subband_size(const cv::Size& image_size) const
{
    return cv::Size(
        (image_size.width + filter_length() - 1) / 2,
        (image_size.height + filter_length() - 1) / 2
    );
}

FilterBank FilterBank::reverse() const
{
    auto reconstruct_kernels = this->reconstruct_kernels();
    cv::Mat decompose_lowpass, decompose_highpass;
    cv::flip(reconstruct_kernels.lowpass(), decompose_lowpass, -1);
    cv::flip(reconstruct_kernels.highpass(), decompose_highpass, -1);

    auto decompose_kernels = this->decompose_kernels();
    cv::Mat reconstruct_lowpass, reconstruct_highpass;
    cv::flip(decompose_kernels.lowpass(), reconstruct_lowpass, -1);
    cv::flip(decompose_kernels.highpass(), reconstruct_highpass, -1);

    return FilterBank(
        decompose_lowpass,
        decompose_highpass,
        reconstruct_lowpass,
        reconstruct_highpass
    );
}

bool FilterBank::is_orthogonal() const
{
    auto decompose_kernels = this->decompose_kernels();
    auto delta = cv::Mat::eye(filter_length() / 2, 1, depth());

    //  "Wavelets and Filter Banks"" Nguyen & Strang - Equation 5.14
    cv::Mat lowpass_correlation;
    internal::strided_correlate(
        decompose_kernels.lowpass(),
        decompose_kernels.lowpass(),
        lowpass_correlation,
        2
    );
    if (!is_approx_equal(lowpass_correlation, delta))
        return false;

    //  "Wavelets and Filter Banks" Nguyen & Strang - Equation 5.16
    cv::Mat highpass_correlation;
    internal::strided_correlate(
        decompose_kernels.highpass(),
        decompose_kernels.highpass(),
        highpass_correlation,
        2
    );
    if (!is_approx_equal(highpass_correlation, delta))
        return false;

    //  "Wavelets and Filter Banks" Nguyen & Strang - Equation 5.15
    cv::Mat cross_correlation;
    internal::strided_correlate(
        decompose_kernels.lowpass(),
        decompose_kernels.highpass(),
        cross_correlation,
        2
    );
    if (!is_approx_zero(cross_correlation))
        return false;

    return is_biorthogonal();
}

bool FilterBank::is_biorthogonal() const
{
    return satisfies_perfect_reconstruction(decompose_kernels(), reconstruct_kernels());
}

bool FilterBank::satisfies_perfect_reconstruction(
    const KernelPair& decompose_kernels,
    const KernelPair& reconstruct_kernels
)
{
    return satisfies_alias_cancellation(decompose_kernels, reconstruct_kernels)
        && satisfies_no_distortion(decompose_kernels, reconstruct_kernels);
}

bool FilterBank::satisfies_alias_cancellation() const
{
    return satisfies_alias_cancellation(decompose_kernels(), reconstruct_kernels());
}

bool FilterBank::satisfies_alias_cancellation(
    const KernelPair& decompose_kernels,
    const KernelPair& reconstruct_kernels
)
{
    //  "Wavelets and Filter Banks" Nguyen & Strang - Equation 4.5
    cv::Mat decompose_lowpass_alternated_signs;
    negate_even_indices(decompose_kernels.lowpass(), decompose_lowpass_alternated_signs);

    cv::Mat decompose_highpass_alternated_signs;
    negate_even_indices(decompose_kernels.highpass(), decompose_highpass_alternated_signs);

    return is_approx_zero(
        internal::convolve(
            reconstruct_kernels.lowpass(),
            decompose_lowpass_alternated_signs
        )
        + internal::convolve(
            reconstruct_kernels.highpass(),
            decompose_highpass_alternated_signs
        )
    );
}

bool FilterBank::satisfies_no_distortion() const
{
    return satisfies_no_distortion(decompose_kernels(), reconstruct_kernels());
}

bool FilterBank::satisfies_no_distortion(
    const KernelPair& decompose_kernels,
    const KernelPair& reconstruct_kernels
)
{
    //  "Wavelets and Filter Banks" Nguyen & Strang - Equation 4.4
    int depth = std::max(decompose_kernels.depth(), reconstruct_kernels.depth());
    auto delta = cv::Mat::eye(reconstruct_kernels.lowpass().total(), 1, depth);
    return is_approx_equal(
        2 * delta,
        internal::convolve(
            reconstruct_kernels.lowpass(),
            decompose_kernels.lowpass()
        )
        + internal::convolve(
            reconstruct_kernels.highpass(),
            decompose_kernels.highpass()
        )
    );
}

bool FilterBank::is_symmetric() const
{
    auto decompose_kernels = this->decompose_kernels();
    auto reconstruct_kernels = this->reconstruct_kernels();
    return is_symmetric(decompose_kernels.lowpass())
        && is_symmetric(decompose_kernels.highpass())
        && is_symmetric(reconstruct_kernels.lowpass())
        && is_symmetric(reconstruct_kernels.highpass());
}

bool FilterBank::is_symmetric(cv::InputArray kernel)
{
    cv::Mat stripped_kernel;
    internal::strip_zeros(kernel, stripped_kernel);

    cv::Mat flipped_kernel;
    cv::flip(stripped_kernel, flipped_kernel, -1);

    return is_equal(stripped_kernel, flipped_kernel);
}

bool FilterBank::is_antisymmetric() const
{
    auto decompose_kernels = this->decompose_kernels();
    auto reconstruct_kernels = this->reconstruct_kernels();
    return is_antisymmetric(decompose_kernels.lowpass())
        && is_antisymmetric(decompose_kernels.highpass())
        && is_antisymmetric(reconstruct_kernels.lowpass())
        && is_antisymmetric(reconstruct_kernels.highpass());
}

bool FilterBank::is_antisymmetric(cv::InputArray kernel)
{
    cv::Mat stripped_kernel;
    internal::strip_zeros(kernel, stripped_kernel);

    cv::Mat flipped_kernel;
    cv::flip(stripped_kernel, flipped_kernel, -1);

    return is_equal(stripped_kernel, -flipped_kernel);
}

bool FilterBank::is_linear_phase() const
{
    auto decompose_kernels = this->decompose_kernels();
    auto reconstruct_kernels = this->reconstruct_kernels();
    return is_linear_phase(decompose_kernels.lowpass())
        && is_linear_phase(decompose_kernels.highpass())
        && is_linear_phase(reconstruct_kernels.lowpass())
        && is_linear_phase(reconstruct_kernels.highpass());
}

bool FilterBank::is_linear_phase(cv::InputArray kernel)
{
    cv::Mat stripped_kernel;
    internal::strip_zeros(kernel, stripped_kernel);

    cv::Mat flipped_kernel;
    cv::flip(stripped_kernel, flipped_kernel, -1);

    return is_equal(stripped_kernel, flipped_kernel)
        || is_equal(stripped_kernel, -flipped_kernel);
}

FilterBank FilterBank::create_orthogonal(cv::InputArray reconstruct_lowpass_coeffs)
{
    cv::Mat reconstruct_lowpass;
    cv::normalize(reconstruct_lowpass_coeffs, reconstruct_lowpass);

    auto filter_bank = create_conjugate_mirror(reconstruct_lowpass);

    filter_bank.throw_if_not_orthogonal();

    return filter_bank;
}

FilterBank FilterBank::create_conjugate_mirror(cv::InputArray reconstruct_lowpass_coeffs)
{
    cv::Mat decompose_lowpass_coeffs;
    cv::flip(reconstruct_lowpass_coeffs, decompose_lowpass_coeffs, -1);

    return create_quadrature_mirror(
        reconstruct_lowpass_coeffs,
        decompose_lowpass_coeffs
    );
}

FilterBank FilterBank::create_biorthogonal(
    cv::InputArray reconstruct_lowpass_coeffs,
    cv::InputArray decompose_lowpass_coeffs
)
{
    cv::Mat reconstruct_lowpass;
    cv::normalize(reconstruct_lowpass_coeffs, reconstruct_lowpass);

    cv::Mat decompose_lowpass;
    cv::normalize(decompose_lowpass_coeffs, decompose_lowpass);

    auto filter_bank = create_quadrature_mirror(reconstruct_lowpass, decompose_lowpass);

    filter_bank.throw_if_not_orthogonal();

    return filter_bank;
}

FilterBank FilterBank::create_quadrature_mirror(
    cv::InputArray reconstruct_lowpass_coeffs,
    cv::InputArray decompose_lowpass_coeffs
)
{
    cv::Mat decompose_lowpass = decompose_lowpass_coeffs.getMat().clone();
    cv::Mat reconstruct_lowpass = reconstruct_lowpass_coeffs.getMat().clone();

    cv::Mat decompose_highpass;
    negate_even_indices(reconstruct_lowpass, decompose_highpass);
    cv::Mat reconstruct_highpass;
    negate_odd_indices(decompose_lowpass, reconstruct_highpass);

    return FilterBank(
        decompose_lowpass,
        decompose_highpass,
        reconstruct_lowpass,
        reconstruct_highpass
    );
}

int FilterBank::promote_type(int type) const
{
    return CV_MAKE_TYPE(
        std::max(depth(), CV_MAT_DEPTH(type)),
        CV_MAT_CN(type)
    );
}

void FilterBank::extrapolate_border(
    cv::InputArray image,
    cv::OutputArray output,
    int border_type,
    const cv::Scalar& border_value
) const
{
    cv::copyMakeBorder(
        image,
        output,
        filter_length() - 2,
        filter_length(),
        filter_length() - 2,
        filter_length(),
        border_type,
        border_value
    );
}

void FilterBank::convolve_rows_and_downsample_cols(
    cv::InputArray input,
    cv::OutputArray output,
    const cv::Mat& kernel
) const
{
    internal::dispatch_on_pixel_depths<internal::ConvolveRowsAndDownsampleCols>(
        input.depth(),
        kernel.depth(),
        input,
        output,
        kernel.t()
    );
}

void FilterBank::convolve_cols_and_downsample_rows(
    cv::InputArray input,
    cv::OutputArray output,
    const cv::Mat& kernel
) const
{
    internal::dispatch_on_pixel_depths<internal::ConvolveColsAndDownsampleRows>(
        input.depth(),
        kernel.depth(),
        input,
        output,
        kernel
    );
}

void FilterBank::upsample_cols_and_convolve_rows(
    cv::InputArray input,
    cv::OutputArray output,
    const cv::Mat& even_kernel,
    const cv::Mat& odd_kernel,
    const cv::Size& output_size
) const
{
    internal::dispatch_on_pixel_depths<internal::UpsampleColsAndConvolveRows>(
        input.depth(),
        even_kernel.depth(),
        input,
        output,
        even_kernel.t(),
        odd_kernel.t(),
        output_size
    );
}

void FilterBank::upsample_rows_and_convolve_cols(
    cv::InputArray input,
    cv::OutputArray output,
    const cv::Mat& even_kernel,
    const cv::Mat& odd_kernel,
    const cv::Size& output_size
) const
{
    internal::dispatch_on_pixel_depths<internal::UpsampleRowsAndConvolveCols>(
        input.depth(),
        even_kernel.depth(),
        input,
        output,
        even_kernel,
        odd_kernel,
        output_size
    );
}

bool FilterBank::operator==(const FilterBank& other) const
{
    return this == &other || (
        is_equal(_p->decompose.lowpass, other._p->decompose.lowpass)
        && is_equal(_p->decompose.highpass, other._p->decompose.highpass)
        && is_equal(_p->reconstruct.even_lowpass, other._p->reconstruct.even_lowpass)
        && is_equal(_p->reconstruct.odd_lowpass, other._p->reconstruct.odd_lowpass)
        && is_equal(_p->reconstruct.even_highpass, other._p->reconstruct.even_highpass)
        && is_equal(_p->reconstruct.odd_highpass, other._p->reconstruct.odd_highpass)
    );
}

inline
void FilterBank::throw_if_decompose_image_is_wrong_size(
    cv::InputArray image,
    const std::source_location& location
) const CVWT_FILTER_BANK_NOEXCEPT
{
#if CVWT_FILTER_BANK_EXCEPTIONS_ENABLED
    if (image.rows() <= 1 || image.cols() <= 1) {
        throw_bad_size(
            "FilterBank: Input size must be greater [1 x 1], got ", image.size(),
            location
        );
    }
#endif
}

inline
void FilterBank::throw_if_reconstruct_coeffs_are_wrong_size(
    cv::InputArray approx,
    cv::InputArray horizontal_detail,
    cv::InputArray vertical_detail,
    cv::InputArray diagonal_detail,
    const std::source_location& location
) const CVWT_FILTER_BANK_NOEXCEPT
{
#if CVWT_FILTER_BANK_EXCEPTIONS_ENABLED
    if (horizontal_detail.size() != approx.size()
        || vertical_detail.size() != approx.size()
        || diagonal_detail.size() != approx.size()) {
        throw_bad_size(
            "FilterBank: Inputs must all be the same size, got ",
            "approx.size() = ", approx.size(), ", ",
            "horizontal_detail.size() = ", horizontal_detail.size(), ", ",
            "vertical_detail.size() = ", vertical_detail.size(), ", and ",
            "diagonal_detail.size() = ", diagonal_detail.size(), ".",
            location
        );
    }

    if (horizontal_detail.channels() != approx.channels()
        || vertical_detail.channels() != approx.channels()
        || diagonal_detail.channels() != approx.channels()) {
        throw_bad_size(
            "FilterBank: Inputs must all be the same number of channels, got ",
            "approx.channels() = ", approx.channels(), ", ",
            "horizontal_detail.channels() = ", horizontal_detail.channels(), ", ",
            "vertical_detail.channels() = ", vertical_detail.channels(), ", and ",
            "diagonal_detail.channels() = ", diagonal_detail.channels(), ".",
            location
        );
    }
#endif
}

inline
void FilterBank::throw_if_not_orthogonal(
    const std::source_location& location
) const CVWT_FILTER_BANK_NOEXCEPT
{
#if CVWT_FILTER_BANK_EXCEPTIONS_ENABLED
    if (!is_orthogonal())
        throw_bad_arg("FilterBank is not orthogonal", location);
#endif
}

inline
void FilterBank::throw_if_not_biorthogonal(
    const std::source_location& location
) const CVWT_FILTER_BANK_NOEXCEPT
{
#if CVWT_FILTER_BANK_EXCEPTIONS_ENABLED
    if (!is_biorthogonal())
        throw_bad_arg("FilterBank is not biorthogonal", location);
#endif
}

std::ostream& operator<<(std::ostream& stream, const FilterBank& filter_bank)
{
    stream
        << "decompose:\n"
        << "    lowpass: " << filter_bank.decompose_kernels().lowpass().reshape(0, 1) << "\n"
        << "    highpass: " << filter_bank.decompose_kernels().highpass().reshape(0, 1) << "\n"
        << "reconstruct:\n"
        << "    lowpass: " << filter_bank.reconstruct_kernels().lowpass().reshape(0, 1) << "\n"
        << "    highpass: " << filter_bank.reconstruct_kernels().highpass().reshape(0, 1) << "\n";

    return stream;
}
} // namespace wtcv

namespace std
{
string to_string(wtcv::DetailSubband subband)
{
    switch (subband) {
        case wtcv::HORIZONTAL: return "HORIZONTAL";
        case wtcv::VERTICAL: return "VERTICAL";
        case wtcv::DIAGONAL: return "DIAGONAL";
    }

    return "";
}
} // namespace std

