#include "cvwt/filterbank.hpp"
#include <functional>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include "cvwt/utils.hpp"

namespace cvwt
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

cv::Mat as_column_vector(const cv::Mat& vector)
{
    assert(vector.rows == 1 || vector.cols == 1);
    return (vector.cols == 1) ? vector : vector.t();
}

FilterBankImpl::FilterBankImpl() :
    filter_length(0),
    decompose(),
    reconstruct(),
    promoted_decompose(),
    promoted_reconstruct()
{
}

FilterBankImpl::FilterBankImpl(
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass,
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass
) :
    filter_length(std::max(decompose_lowpass.total(), reconstruct_lowpass.total())),
    promoted_decompose(),
    promoted_reconstruct()
{
    throw_if_wrong_size(
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

void FilterBankImpl::split_kernel_into_odd_and_even_parts(cv::InputArray kernel, cv::OutputArray even_kernel, cv::OutputArray odd_kernel) const
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

#if CVWT_ARGUMENT_CHECKING_ENABLED
void FilterBankImpl::throw_if_wrong_size(
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass,
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass
) const
{
    auto all_empty = decompose_lowpass.empty()
        && decompose_highpass.empty()
        && reconstruct_lowpass.empty()
        && reconstruct_highpass.empty();

    if (!all_empty) {
        if (decompose_lowpass.empty()
            || decompose_highpass.empty()
            || reconstruct_lowpass.empty()
            || reconstruct_highpass.empty()
        ) {
            internal::throw_bad_arg(
                "FilterBank: Kernels must all be empty or all nonempty, got ",
                (reconstruct_lowpass.empty() ? "empty" : "nonempty"), " reconstruct_lowpass, ",
                (reconstruct_highpass.empty() ? "empty" : "nonempty"), " reconstruct_highpass, ",
                (decompose_lowpass.empty() ? "empty" : "nonempty"), " decompose_lowpass, and ",
                (decompose_highpass.empty() ? "empty" : "nonempty"), " decompose_highpass."
            );
        }

        if (decompose_lowpass.size() != decompose_highpass.size()
            || (decompose_lowpass.rows != 1 && decompose_lowpass.cols != 1)
        ) {
            internal::throw_bad_size(
                "FilterBank: decompose_lowpass and decompose_highpass must be row or column vectors of the same size, ",
                "got decompose_lowpass.size() = ", decompose_lowpass.size(), ", ",
                "decompose_highpass.size() = ", decompose_highpass.size(), "."
            );
        }

        if (reconstruct_lowpass.size() != reconstruct_highpass.size()
            || (reconstruct_lowpass.rows != 1 && reconstruct_lowpass.cols != 1)
        ) {
            internal::throw_bad_size(
                "FilterBank: reconstruct_lowpass and reconstruct_highpass must be row or column vectors of the same size, ",
                "got reconstruct_lowpass.size() = ", reconstruct_lowpass.size(), ", ",
                "reconstruct_highpass.size() = ", reconstruct_highpass.size(), "."
            );
        }
    }
}
#endif  // CVWT_ARGUMENT_CHECKING_ENABLED
} // namespace internal


/**
 * =============================================================================
 * Public API
 * =============================================================================
*/
/**
 * -----------------------------------------------------------------------------
 * KernelPair
 * -----------------------------------------------------------------------------
*/
bool KernelPair::operator==(const KernelPair& other) const
{
    return this == &other || (
        matrix_equals(_lowpass, other._lowpass) && matrix_equals(_highpass, other._highpass)
    );
}

/**
 * -----------------------------------------------------------------------------
 * FilterBank
 * -----------------------------------------------------------------------------
*/
FilterBank::FilterBank() :
    _p(std::make_shared<internal::FilterBankImpl>())
{
}

FilterBank::FilterBank(
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass,
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass
) :
    _p(
        std::make_shared<internal::FilterBankImpl>(
            decompose_lowpass,
            decompose_highpass,
            reconstruct_lowpass,
            reconstruct_highpass
        )
    )
{
}

FilterBank::FilterBank(
    const KernelPair& decompose_kernels,
    const KernelPair& reconstruct_kernels
) :
    FilterBank(
        decompose_kernels.lowpass(),
        decompose_kernels.highpass(),
        reconstruct_kernels.lowpass(),
        reconstruct_kernels.highpass()
    )
{
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
    cv::OutputArray output,
    const cv::Size& output_size
) const
{
    throw_if_reconstruct_coeffs_are_wrong_size(approx, horizontal_detail, vertical_detail, diagonal_detail);

    //  Stage 1a
    cv::Mat stage1a_output;
    upsample_rows_and_convolve_cols(
        approx.getMat(),
        stage1a_output,
        _p->reconstruct.even_lowpass,
        _p->reconstruct.odd_lowpass,
        output_size
    );
    upsample_rows_and_convolve_cols(
        horizontal_detail.getMat(),
        stage1a_output,
        _p->reconstruct.even_highpass,
        _p->reconstruct.odd_highpass,
        output_size
    );

    //  Stage 1b
    cv::Mat stage1b_output;
    upsample_rows_and_convolve_cols(
        vertical_detail.getMat(),
        stage1b_output,
        _p->reconstruct.even_lowpass,
        _p->reconstruct.odd_lowpass,
        output_size
    );
    upsample_rows_and_convolve_cols(
        diagonal_detail.getMat(),
        stage1b_output,
        _p->reconstruct.even_highpass,
        _p->reconstruct.odd_highpass,
        output_size
    );

    //  Stage 2
    cv::Mat stage2_output;
    upsample_cols_and_convolve_rows(
        stage1a_output,
        output,
        _p->reconstruct.even_lowpass,
        _p->reconstruct.odd_lowpass,
        output_size
    );
    upsample_cols_and_convolve_rows(
        stage1b_output,
        output,
        _p->reconstruct.even_highpass,
        _p->reconstruct.odd_highpass,
        output_size
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

FilterBank FilterBank::create_orthogonal_filter_bank(cv::InputArray reconstruct_lowpass_coeffs)
{
    cv::Mat decompose_lowpass_coeffs;
    cv::flip(reconstruct_lowpass_coeffs, decompose_lowpass_coeffs, -1);

    return create_biorthogonal_filter_bank(
        reconstruct_lowpass_coeffs,
        decompose_lowpass_coeffs
    );
}

FilterBank FilterBank::create_biorthogonal_filter_bank(
    cv::InputArray reconstruct_lowpass_coeffs,
    cv::InputArray decompose_lowpass_coeffs
)
{
    cv::Mat decompose_lowpass = decompose_lowpass_coeffs.getMat().clone();
    cv::Mat reconstruct_lowpass = reconstruct_lowpass_coeffs.getMat().clone();

    cv::Mat decompose_highpass;
    negate_evens(reconstruct_lowpass, decompose_highpass);
    cv::Mat reconstruct_highpass;
    negate_odds(decompose_lowpass, reconstruct_highpass);

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
        matrix_equals(_p->decompose.lowpass, other._p->decompose.lowpass)
        && matrix_equals(_p->decompose.highpass, other._p->decompose.highpass)
        && matrix_equals(_p->reconstruct.even_lowpass, other._p->reconstruct.even_lowpass)
        && matrix_equals(_p->reconstruct.odd_lowpass, other._p->reconstruct.odd_lowpass)
        && matrix_equals(_p->reconstruct.even_highpass, other._p->reconstruct.even_highpass)
        && matrix_equals(_p->reconstruct.odd_highpass, other._p->reconstruct.odd_highpass)
    );
}

#if CVWT_ARGUMENT_CHECKING_ENABLED
void FilterBank::throw_if_decompose_image_is_wrong_size(cv::InputArray image) const
{
    if (image.rows() <= 1 || image.cols() <= 1) {
        internal::throw_bad_size(
            "FilterBank: Input size must be greater [1 x 1], got ", image.size()
        );
    }
}

void FilterBank::throw_if_reconstruct_coeffs_are_wrong_size(
    cv::InputArray approx,
    cv::InputArray horizontal_detail,
    cv::InputArray vertical_detail,
    cv::InputArray diagonal_detail
) const
{
    if (horizontal_detail.size() != approx.size()
        || vertical_detail.size() != approx.size()
        || diagonal_detail.size() != approx.size()) {
        internal::throw_bad_size(
            "FilterBank: Inputs must all be the same size, got ",
            "approx.size() = ", approx.size(), ", ",
            "horizontal_detail.size() = ", horizontal_detail.size(), ", ",
            "vertical_detail.size() = ", vertical_detail.size(), ", and ",
            "diagonal_detail.size() = ", diagonal_detail.size(), "."
        );
    }

    if (horizontal_detail.channels() != approx.channels()
        || vertical_detail.channels() != approx.channels()
        || diagonal_detail.channels() != approx.channels()) {
        internal::throw_bad_size(
            "FilterBank: Inputs must all be the same number of channels, got ",
            "approx.channels() = ", approx.channels(), ", ",
            "horizontal_detail.channels() = ", horizontal_detail.channels(), ", ",
            "vertical_detail.channels() = ", vertical_detail.channels(), ", and ",
            "diagonal_detail.channels() = ", diagonal_detail.channels(), "."
        );
    }
}
#endif  // CVWT_ARGUMENT_CHECKING_ENABLED

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
} // namespace cvwt

