#include "wavelet/wavelet.hpp"
#include "wavelet/daubechies.hpp"
#include "wavelet/symlets.hpp"
#include "wavelet/coiflets.hpp"
#include "wavelet/biorthogonal.hpp"
#include "wavelet/utils.hpp"
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <functional>
#include <experimental/iterator>
#include <iostream>

namespace wavelet
{
namespace internal
{
template <typename Pixel>
inline void dot(const cv::Mat& a, const cv::Mat& b, Pixel& output_pixel)
{
    auto c = cv::sum(a.mul(b));
    for (int i = 0; i < Pixel::channels; ++i)
        output_pixel[i] = c[i];
}

template <typename T, int CHANNELS>
struct ConvolveRowsAndDownsampleCols
{
    using Pixel = cv::Vec<T, CHANNELS>;

    /**
     * Input is assumed to be padded, kernel is assumed to be flipped and transposed (if necessary)
    */
    void operator()(cv::InputArray input, cv::OutputArray output, const cv::Mat& kernel)
    {
        auto output_size = input.size() - kernel.size() + cv::Size(1, 1);
        output_size.width = output_size.width / 2;
        output.create(output_size, input.type());

        const auto input_image = input.getMat();
        output.getMat().forEach<Pixel>(
            [&](auto& output_pixel, const auto index) {
                dot(
                    kernel,
                    input_image(cv::Rect(
                        cv::Point(2 * index[1], index[0]),
                        kernel.size()
                    )),
                    output_pixel
                );
            }
        );
    }
};

template <typename T, int CHANNELS>
struct ConvolveColsAndDownsampleRows
{
    using Pixel = cv::Vec<T, CHANNELS>;

    /**
     * Input is assumed to be padded, kernel is assumed to be flipped and transposed (if necessary)
    */
    void operator()(cv::InputArray input, cv::OutputArray output, const cv::Mat& kernel)
    {
        auto output_size = input.size() - kernel.size() + cv::Size(1, 1);
        output_size.height = output_size.height / 2;
        output.create(output_size, input.type());

        const auto input_image = input.getMat();
        output.getMat().forEach<Pixel>(
            [&](auto& output_pixel, const auto index) {
                dot(
                    kernel,
                    input_image(cv::Rect(
                        cv::Point(index[1], 2 * index[0]),
                        kernel.size()
                    )),
                    output_pixel
                );
            }
        );
    }
};

template <typename T, int CHANNELS>
struct UpsampleRowsAndConvolveCols
{
    using Pixel = cv::Vec<T, CHANNELS>;

    /**
     * Kernels are assumed to be flipped and transposed (if necessary)
    */
    void operator()(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Mat& even_index_kernel,
        const cv::Mat& odd_index_kernel,
        const cv::Size& output_size
    ) const
    {
        output.create(
            cv::Size(input.size().width, output_size.height),
            input.type()
        );
        const auto input_image = input.getMat();
        output.getMat().forEach<Pixel>(
            [&](auto& output_pixel, const auto index) {
                const cv::Mat& kernel = (index[0] % 2 == 1)
                                      ? odd_index_kernel
                                      : even_index_kernel;
                dot(
                    kernel,
                    input_image(cv::Rect(
                        cv::Point(index[1], index[0] / 2),
                        kernel.size()
                    )),
                    output_pixel
                );
            }
        );
    }
};

template <typename T, int CHANNELS>
struct UpsampleColsAndConvolveRows
{
    using Pixel = cv::Vec<T, CHANNELS>;

    /**
     * Kernels are assumed to be flipped and transposed (if necessary)
    */
    void operator()(
        cv::InputArray input,
        cv::OutputArray output,
        const cv::Mat& even_index_kernel,
        const cv::Mat& odd_index_kernel,
        const cv::Size& output_size
    ) const
    {
        output.create(
            cv::Size(output_size.width, input.size().height),
            input.type()
        );
        const auto input_image = input.getMat();
        output.getMat().forEach<Pixel>(
            [&](auto& output_pixel, const auto index) {
                const cv::Mat& kernel = (index[1] % 2 == 1)
                                      ? odd_index_kernel
                                      : even_index_kernel;
                dot(
                    kernel,
                    input_image(cv::Rect(
                        cv::Point(index[1] / 2, index[0]),
                        kernel.size()
                    )),
                    output_pixel
                );
            }
        );
    }
};

template <typename T, int CHANNELS>
struct SplitKernelIntoOddAndEvenParts
{
    using Pixel = cv::Vec<T, CHANNELS>;

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
            even_kernel.at<Pixel>(i) = kernel.at<Pixel>(2 * i + 1);

        odd_output.create(odd_size, kernel.type());
        auto odd_kernel = odd_output.getMat();
        for (int i = 0; i < odd_kernel.total(); ++i)
            odd_kernel.at<Pixel>(i) = kernel.at<Pixel>(2 * i);
    }
};

template <typename T, int CHANNELS>
struct MergeEvenAndOddKernels
{
    using Pixel = cv::Vec<T, CHANNELS>;

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
            kernel.at<Pixel>(i) = source_kernel.at<Pixel>(i / 2);
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
    check_kernels(
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
    dispatch_on_pixel_type<SplitKernelIntoOddAndEvenParts>(
        kernel.type(),
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
    dispatch_on_pixel_type<MergeEvenAndOddKernels>(
        even_kernel.type(),
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

#ifndef DISABLE_ARG_CHECKS
void FilterBankImpl::check_kernels(
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
            throw_bad_arg(
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
            throw_bad_size(
                "FilterBank: decompose_lowpass and decompose_highpass must be row or column vectors of the same size, ",
                "got decompose_lowpass.size() = ", decompose_lowpass.size(), ", ",
                "decompose_highpass.size() = ", decompose_highpass.size(), "."
            );
        }

        if (reconstruct_lowpass.size() != reconstruct_highpass.size()
            || (reconstruct_lowpass.rows != 1 && reconstruct_lowpass.cols != 1)
        ) {
            throw_bad_size(
                "FilterBank: reconstruct_lowpass and reconstruct_highpass must be row or column vectors of the same size, ",
                "got reconstruct_lowpass.size() = ", reconstruct_lowpass.size(), ", ",
                "reconstruct_highpass.size() = ", reconstruct_highpass.size(), "."
            );
        }
    }
}
#else
inline void FilterBankImpl::check_kernels(
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass,
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass
) const
{
}
#endif
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
        equals(_lowpass, other._lowpass) && equals(_highpass, other._highpass)
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

void FilterBank::forward(
    cv::InputArray input,
    cv::OutputArray approx,
    cv::OutputArray horizontal_detail,
    cv::OutputArray vertical_detail,
    cv::OutputArray diagonal_detail,
    int border_type,
    const cv::Scalar& border_value
) const
{
    check_forward_input(input);

    bool was_prepared = is_prepared_forward(input.type());
    if (!was_prepared)
        prepare_forward(input.type());

    cv::Mat promoted_input;
    promote_input(input, promoted_input);

    cv::Mat padded_input;
    pad(promoted_input, padded_input, border_type, border_value);

    //  Stage 1
    cv::Mat stage1_lowpass_output;
    convolve_rows_and_downsample_cols(
        padded_input,
        stage1_lowpass_output,
        _p->promoted_decompose.lowpass
    );
    cv::Mat stage1_highpass_output;
    convolve_rows_and_downsample_cols(
        padded_input,
        stage1_highpass_output,
        _p->promoted_decompose.highpass
    );

    //  Stage 2
    convolve_cols_and_downsample_rows(
        stage1_lowpass_output,
        approx,
        _p->promoted_decompose.lowpass
    );
    convolve_cols_and_downsample_rows(
        stage1_lowpass_output,
        horizontal_detail,
        _p->promoted_decompose.highpass
    );
    convolve_cols_and_downsample_rows(
        stage1_highpass_output,
        vertical_detail,
        _p->promoted_decompose.lowpass
    );
    convolve_cols_and_downsample_rows(
        stage1_highpass_output,
        diagonal_detail,
        _p->promoted_decompose.highpass
    );

    if (!was_prepared)
        finish_forward();
}

void FilterBank::prepare_forward(int type) const
{
    if (is_prepared_forward(type))
        return;

    promote_kernel(_p->decompose.lowpass, _p->promoted_decompose.lowpass, type);
    promote_kernel(_p->decompose.highpass, _p->promoted_decompose.highpass, type);
}

void FilterBank::finish_forward() const
{
    _p->promoted_decompose.release();
}

void FilterBank::inverse(
    cv::InputArray approx,
    cv::InputArray horizontal_detail,
    cv::InputArray vertical_detail,
    cv::InputArray diagonal_detail,
    cv::OutputArray output,
    const cv::Size& output_size
) const
{
    check_inverse_inputs(approx, horizontal_detail, vertical_detail, diagonal_detail);

    bool was_prepared = is_prepared_inverse(approx.type());
    if (!was_prepared)
        prepare_inverse(approx.type());

    cv::Mat promoted_approx;
    promote_input(approx, promoted_approx);
    cv::Mat promoted_horizontal_detail;
    promote_input(horizontal_detail, promoted_horizontal_detail);
    cv::Mat promoted_vertical_detail;
    promote_input(vertical_detail, promoted_vertical_detail);
    cv::Mat promoted_diagonal_detail;
    promote_input(diagonal_detail, promoted_diagonal_detail);

    //  Stage 1a
    cv::Mat stage1a_lowpass_output;
    upsample_rows_and_convolve_cols(
        promoted_approx,
        stage1a_lowpass_output,
        _p->promoted_reconstruct.even_lowpass,
        _p->promoted_reconstruct.odd_lowpass,
        output_size
    );
    cv::Mat stage1a_highpass_output;
    upsample_rows_and_convolve_cols(
        promoted_horizontal_detail,
        stage1a_highpass_output,
        _p->promoted_reconstruct.even_highpass,
        _p->promoted_reconstruct.odd_highpass,
        output_size
    );

    //  Stage 1b
    cv::Mat stage1b_lowpass_output;
    upsample_rows_and_convolve_cols(
        promoted_vertical_detail,
        stage1b_lowpass_output,
        _p->promoted_reconstruct.even_lowpass,
        _p->promoted_reconstruct.odd_lowpass,
        output_size
    );
    cv::Mat stage1b_highpass_output;
    upsample_rows_and_convolve_cols(
        promoted_diagonal_detail,
        stage1b_highpass_output,
        _p->promoted_reconstruct.even_highpass,
        _p->promoted_reconstruct.odd_highpass,
        output_size
    );

    //  Stage 2
    cv::Mat stage2_lowpass_output;
    upsample_cols_and_convolve_rows(
        stage1a_lowpass_output + stage1a_highpass_output,
        stage2_lowpass_output,
        _p->promoted_reconstruct.even_lowpass,
        _p->promoted_reconstruct.odd_lowpass,
        output_size
    );
    cv::Mat stage2_highpass_output;
    upsample_cols_and_convolve_rows(
        stage1b_lowpass_output + stage1b_highpass_output,
        stage2_highpass_output,
        _p->promoted_reconstruct.even_highpass,
        _p->promoted_reconstruct.odd_highpass,
        output_size
    );

    output.assign(stage2_lowpass_output + stage2_highpass_output);

    if (!was_prepared)
        finish_inverse();
}

void FilterBank::prepare_inverse(int type) const
{
    if (is_prepared_inverse(type))
        return;

    promote_kernel(
        _p->reconstruct.even_lowpass,
        _p->promoted_reconstruct.even_lowpass,
        type
    );
    promote_kernel(
        _p->reconstruct.odd_lowpass,
        _p->promoted_reconstruct.odd_lowpass,
        type
    );
    promote_kernel(
        _p->reconstruct.even_highpass,
        _p->promoted_reconstruct.even_highpass,
        type
    );
    promote_kernel(
        _p->reconstruct.odd_highpass,
        _p->promoted_reconstruct.odd_highpass,
        type
    );
}

void FilterBank::finish_inverse() const
{
    _p->promoted_reconstruct.release();
}

cv::Size FilterBank::output_size(const cv::Size& input_size) const
{
    return cv::Size(output_size(input_size.width), output_size(input_size.height));
}

int FilterBank::output_size(int input_size) const
{
    return 2 * subband_size(input_size);
}

cv::Size FilterBank::subband_size(const cv::Size& input_size) const
{
    return cv::Size(subband_size(input_size.width), subband_size(input_size.height));
}

int FilterBank::subband_size(int input_size) const
{
    return (input_size + filter_length() - 1) / 2;
}

KernelPair FilterBank::create_orthogonal_decompose_kernels(cv::InputArray reconstruct_lowpass_coeffs)
{
    return create_biorthogonal_decompose_kernels(
        reconstruct_lowpass_coeffs,
        reconstruct_lowpass_coeffs
    );
}

KernelPair FilterBank::create_orthogonal_reconstruct_kernels(cv::InputArray reconstruct_lowpass_coeffs)
{
    return create_biorthogonal_reconstruct_kernels(
        reconstruct_lowpass_coeffs,
        reconstruct_lowpass_coeffs
    );
}

KernelPair FilterBank::create_biorthogonal_decompose_kernels(
    cv::InputArray reconstruct_lowpass_coeffs,
    cv::InputArray decompose_lowpass_coeffs
)
{
    cv::Mat lowpass;
    cv::flip(decompose_lowpass_coeffs, lowpass, -1);

    cv::Mat highpass;
    negate_evens(reconstruct_lowpass_coeffs, highpass);

    return KernelPair(lowpass, highpass);
}

KernelPair FilterBank::create_biorthogonal_reconstruct_kernels(
    cv::InputArray reconstruct_lowpass_coeffs,
    cv::InputArray decompose_lowpass_coeffs
)
{
    cv::Mat lowpass = reconstruct_lowpass_coeffs.getMat().clone();

    cv::Mat highpass;
    cv::flip(decompose_lowpass_coeffs, highpass, -1);
    negate_odds(highpass, highpass);

    return KernelPair(lowpass, highpass);
}

int FilterBank::promote_type(int type) const
{
    return CV_MAKE_TYPE(
        std::max(depth(), CV_MAT_DEPTH(type)),
        CV_MAT_CN(type)
    );
}

void FilterBank::promote_input(
    cv::InputArray input,
    cv::OutputArray promoted_input
) const
{
    int type = promote_type(input.type());
    input.getMat().convertTo(promoted_input, type);
}

void FilterBank::promote_kernel(
    cv::InputArray kernel,
    cv::OutputArray promoted_kernel,
    int type
) const
{
    type = promote_type(type);

    cv::Mat converted;
    kernel.getMat().convertTo(converted, type);

    int channels = CV_MAT_CN(type);
    if (channels == 1) {
        promoted_kernel.assign(converted);
    } else {
        std::vector<cv::Mat> kernels(channels);
        std::ranges::fill(kernels, converted);
        cv::merge(kernels, promoted_kernel);
    }
}

void FilterBank::pad(
    cv::InputArray input,
    cv::OutputArray output,
    int border_type,
    const cv::Scalar& border_value
) const
{
    cv::copyMakeBorder(
        input,
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
    cv::InputArray data,
    cv::OutputArray output,
    const cv::Mat& kernel
) const
{
    internal::dispatch_on_pixel_type<internal::ConvolveRowsAndDownsampleCols>(
        data.type(),
        data,
        output,
        kernel.t()
    );
}

void FilterBank::convolve_cols_and_downsample_rows(
    cv::InputArray data,
    cv::OutputArray output,
    const cv::Mat& kernel
) const
{
    internal::dispatch_on_pixel_type<internal::ConvolveColsAndDownsampleRows>(
        data.type(),
        data,
        output,
        kernel
    );
}

void FilterBank::upsample_cols_and_convolve_rows(
    cv::InputArray data,
    cv::OutputArray output,
    const cv::Mat& even_kernel,
    const cv::Mat& odd_kernel,
    const cv::Size& output_size
) const
{
    internal::dispatch_on_pixel_type<internal::UpsampleColsAndConvolveRows>(
        data.type(),
        data,
        output,
        even_kernel.t(),
        odd_kernel.t(),
        output_size
    );
}

void FilterBank::upsample_rows_and_convolve_cols(
    cv::InputArray data,
    cv::OutputArray output,
    const cv::Mat& even_kernel,
    const cv::Mat& odd_kernel,
    const cv::Size& output_size
) const
{
    internal::dispatch_on_pixel_type<internal::UpsampleRowsAndConvolveCols>(
        data.type(),
        data,
        output,
        even_kernel,
        odd_kernel,
        output_size
    );
}

bool FilterBank::operator==(const FilterBank& other) const
{
    return this == &other || (
        equals(_p->decompose.lowpass, other._p->decompose.lowpass)
        && equals(_p->decompose.highpass, other._p->decompose.highpass)
        && equals(_p->reconstruct.even_lowpass, other._p->reconstruct.even_lowpass)
        && equals(_p->reconstruct.odd_lowpass, other._p->reconstruct.odd_lowpass)
        && equals(_p->reconstruct.even_highpass, other._p->reconstruct.even_highpass)
        && equals(_p->reconstruct.odd_highpass, other._p->reconstruct.odd_highpass)
    );
}

#ifndef DISABLE_ARG_CHECKS
void FilterBank::check_forward_input(cv::InputArray input) const
{
    if (input.rows() <= 1 || input.cols() <= 1) {
        std::stringstream message;
        message << "FilterBank: Input size must be greater [1 x 1], got " << input.size();
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}

void FilterBank::check_inverse_inputs(
    cv::InputArray approx,
    cv::InputArray horizontal_detail,
    cv::InputArray vertical_detail,
    cv::InputArray diagonal_detail
) const
{
    if (horizontal_detail.size() != approx.size()
        || vertical_detail.size() != approx.size()
        || diagonal_detail.size() != approx.size()) {
        std::stringstream message;
        message << "FilterBank: Inputs must all be the same size, got "
            << "approx.size() = " << approx.size() << ", "
            << "horizontal_detail.size() = " << horizontal_detail.size() << ", "
            << "vertical_detail.size() = " << vertical_detail.size() << ", and "
            << "diagonal_detail.size() = " << diagonal_detail.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }

    if (horizontal_detail.channels() != approx.channels()
        || vertical_detail.channels() != approx.channels()
        || diagonal_detail.channels() != approx.channels()) {
        std::stringstream message;
        message << "FilterBank: Inputs must all be the same number of channels, got "
            << "approx.channels() = " << approx.channels() << ", "
            << "horizontal_detail.channels() = " << horizontal_detail.channels() << ", "
            << "vertical_detail.channels() = " << vertical_detail.channels() << ", and "
            << "diagonal_detail.channels() = " << diagonal_detail.channels() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}
#else
inline void WaveletFilterBank::check_forward_input(cv::InputArray input) const {}
inline void WaveletFilterBank::check_inverse_inputs(
    cv::InputArray approx,
    cv::InputArray horizontal_detail,
    cv::InputArray vertical_detail,
    cv::InputArray diagonal_detail
) const {}
#endif // DISABLE_ARG_CHECKS

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

/**
 * -----------------------------------------------------------------------------
 * Wavelet
 * -----------------------------------------------------------------------------
*/
Wavelet::Wavelet() : _p(std::make_shared<WaveletImpl>())
{
}

Wavelet::Wavelet(
    int vanishing_moments_psi,
    int vanishing_moments_phi,
    bool orthogonal,
    bool biorthogonal,
    Symmetry symmetry,
    const std::string& family,
    const std::string& name,
    const FilterBank& filter_bank
) :
    _p(
        std::make_shared<WaveletImpl>(
            vanishing_moments_psi,
            vanishing_moments_phi,
            orthogonal,
            biorthogonal,
            symmetry,
            family,
            name,
            filter_bank
        )
    )
{
}

bool Wavelet::operator==(const Wavelet& other) const
{
    return _p == other._p
        || (name() == other.name() && filter_bank() == other.filter_bank());
}

std::ostream& operator<<(std::ostream& stream, const Wavelet& wavelet)
{
    stream << "Wavelet(" << wavelet.name() << ")";
    return stream;
}

Wavelet Wavelet::create(const std::string& name)
{
    return _wavelet_factories.at(name)();
}

std::vector<std::string> Wavelet::registered_wavelets()
{
    std::vector<std::string> keys;
    std::transform(
        _wavelet_factories.begin(),
        _wavelet_factories.end(),
        std::back_inserter(keys),
        [](auto const& pair) { return pair.first; }
    );

    return keys;
}


std::map<std::string, std::function<Wavelet()>> Wavelet::_wavelet_factories{
    {"haar", create_haar},
    //  daubechies
    {"db1", std::bind(create_daubechies, 1)},
    {"db2", std::bind(create_daubechies, 2)},
    {"db3", std::bind(create_daubechies, 3)},
    {"db4", std::bind(create_daubechies, 4)},
    {"db5", std::bind(create_daubechies, 5)},
    {"db6", std::bind(create_daubechies, 6)},
    {"db7", std::bind(create_daubechies, 7)},
    {"db8", std::bind(create_daubechies, 8)},
    {"db9", std::bind(create_daubechies, 9)},
    {"db10", std::bind(create_daubechies, 10)},
    {"db11", std::bind(create_daubechies, 11)},
    {"db12", std::bind(create_daubechies, 12)},
    {"db13", std::bind(create_daubechies, 13)},
    {"db14", std::bind(create_daubechies, 14)},
    {"db15", std::bind(create_daubechies, 15)},
    {"db16", std::bind(create_daubechies, 16)},
    {"db17", std::bind(create_daubechies, 17)},
    {"db18", std::bind(create_daubechies, 18)},
    {"db19", std::bind(create_daubechies, 19)},
    {"db20", std::bind(create_daubechies, 20)},
    {"db21", std::bind(create_daubechies, 21)},
    {"db22", std::bind(create_daubechies, 22)},
    {"db23", std::bind(create_daubechies, 23)},
    {"db24", std::bind(create_daubechies, 24)},
    {"db25", std::bind(create_daubechies, 25)},
    {"db26", std::bind(create_daubechies, 26)},
    {"db27", std::bind(create_daubechies, 27)},
    {"db28", std::bind(create_daubechies, 28)},
    {"db29", std::bind(create_daubechies, 29)},
    {"db30", std::bind(create_daubechies, 30)},
    {"db31", std::bind(create_daubechies, 31)},
    {"db32", std::bind(create_daubechies, 32)},
    {"db33", std::bind(create_daubechies, 33)},
    {"db34", std::bind(create_daubechies, 34)},
    {"db35", std::bind(create_daubechies, 35)},
    {"db36", std::bind(create_daubechies, 36)},
    {"db37", std::bind(create_daubechies, 37)},
    {"db38", std::bind(create_daubechies, 38)},
    //  symlets
    {"sym2", std::bind(create_symlets, 2)},
    {"sym3", std::bind(create_symlets, 3)},
    {"sym4", std::bind(create_symlets, 4)},
    {"sym5", std::bind(create_symlets, 5)},
    {"sym6", std::bind(create_symlets, 6)},
    {"sym7", std::bind(create_symlets, 7)},
    {"sym8", std::bind(create_symlets, 8)},
    {"sym9", std::bind(create_symlets, 9)},
    {"sym10", std::bind(create_symlets, 10)},
    {"sym11", std::bind(create_symlets, 11)},
    {"sym12", std::bind(create_symlets, 12)},
    {"sym13", std::bind(create_symlets, 13)},
    {"sym14", std::bind(create_symlets, 14)},
    {"sym15", std::bind(create_symlets, 15)},
    {"sym16", std::bind(create_symlets, 16)},
    {"sym17", std::bind(create_symlets, 17)},
    {"sym18", std::bind(create_symlets, 18)},
    {"sym19", std::bind(create_symlets, 19)},
    {"sym20", std::bind(create_symlets, 20)},
    //  coiflets
    {"coif1", std::bind(create_coiflets, 1)},
    {"coif2", std::bind(create_coiflets, 2)},
    {"coif3", std::bind(create_coiflets, 3)},
    {"coif4", std::bind(create_coiflets, 4)},
    {"coif5", std::bind(create_coiflets, 5)},
    {"coif6", std::bind(create_coiflets, 6)},
    {"coif7", std::bind(create_coiflets, 7)},
    {"coif8", std::bind(create_coiflets, 8)},
    {"coif9", std::bind(create_coiflets, 9)},
    {"coif10", std::bind(create_coiflets, 10)},
    {"coif11", std::bind(create_coiflets, 11)},
    {"coif12", std::bind(create_coiflets, 12)},
    {"coif13", std::bind(create_coiflets, 13)},
    {"coif14", std::bind(create_coiflets, 14)},
    {"coif15", std::bind(create_coiflets, 15)},
    {"coif16", std::bind(create_coiflets, 16)},
    {"coif17", std::bind(create_coiflets, 17)},
    //  biorthongonal
    {"bior1.1", std::bind(create_biorthogonal, 1, 1)},
    {"bior1.3", std::bind(create_biorthogonal, 1, 3)},
    {"bior1.5", std::bind(create_biorthogonal, 1, 5)},
    {"bior2.2", std::bind(create_biorthogonal, 2, 2)},
    {"bior2.4", std::bind(create_biorthogonal, 2, 4)},
    {"bior2.6", std::bind(create_biorthogonal, 2, 6)},
    {"bior2.8", std::bind(create_biorthogonal, 2, 8)},
    {"bior3.1", std::bind(create_biorthogonal, 3, 1)},
    {"bior3.3", std::bind(create_biorthogonal, 3, 3)},
    {"bior3.5", std::bind(create_biorthogonal, 3, 5)},
    {"bior3.7", std::bind(create_biorthogonal, 3, 7)},
    {"bior3.9", std::bind(create_biorthogonal, 3, 9)},
    {"bior4.4", std::bind(create_biorthogonal, 4, 4)},
    {"bior5.5", std::bind(create_biorthogonal, 5, 5)},
    {"bior6.8", std::bind(create_biorthogonal, 6, 8)},
    //  reverse biorthongonal
    {"rbior1.1", std::bind(create_reverse_biorthogonal, 1, 1)},
    {"rbior1.3", std::bind(create_reverse_biorthogonal, 1, 3)},
    {"rbior1.5", std::bind(create_reverse_biorthogonal, 1, 5)},
    {"rbior2.2", std::bind(create_reverse_biorthogonal, 2, 2)},
    {"rbior2.4", std::bind(create_reverse_biorthogonal, 2, 4)},
    {"rbior2.6", std::bind(create_reverse_biorthogonal, 2, 6)},
    {"rbior2.8", std::bind(create_reverse_biorthogonal, 2, 8)},
    {"rbior3.1", std::bind(create_reverse_biorthogonal, 3, 1)},
    {"rbior3.3", std::bind(create_reverse_biorthogonal, 3, 3)},
    {"rbior3.5", std::bind(create_reverse_biorthogonal, 3, 5)},
    {"rbior3.7", std::bind(create_reverse_biorthogonal, 3, 7)},
    {"rbior3.9", std::bind(create_reverse_biorthogonal, 3, 9)},
    {"rbior4.4", std::bind(create_reverse_biorthogonal, 4, 4)},
    {"rbior5.5", std::bind(create_reverse_biorthogonal, 5, 5)},
    {"rbior6.8", std::bind(create_reverse_biorthogonal, 6, 8)},
};


/**
 * -----------------------------------------------------------------------------
 * Wavelet Factories
 * -----------------------------------------------------------------------------
*/
Wavelet create_haar()
{
    cv::Mat reconstruct_lowpass_coeffs(DAUBECHIES_FILTER_COEFFS["db1"]);

    return Wavelet(
        1, // vanishing_moments_psi
        0, // vanishing_moments_phi
        true, // orthogonal
        true, // biorthogonal
        Symmetry::ASYMMETRIC, // symmetry
        "Haar", // family
        "haar", // name
        FilterBank(
            FilterBank::create_orthogonal_decompose_kernels(reconstruct_lowpass_coeffs),
            FilterBank::create_orthogonal_reconstruct_kernels(reconstruct_lowpass_coeffs)
        )
    );
}

Wavelet create_daubechies(int order)
{
    auto name = internal::get_orthogonal_name(DAUBECHIES_PREFIX, order);
    internal::check_wavelet_name(name, DAUBECHIES_FAMILY, DAUBECHIES_FILTER_COEFFS);
    cv::Mat reconstruct_lowpass_coeffs(DAUBECHIES_FILTER_COEFFS[name]);

    return Wavelet(
        order, // vanishing_moments_psi
        0, // vanishing_moments_phi
        true, // orthogonal
        true, // biorthogonal
        Symmetry::ASYMMETRIC, // symmetry
        DAUBECHIES_FAMILY, // family
        name, // name
        FilterBank(
            FilterBank::create_orthogonal_decompose_kernels(reconstruct_lowpass_coeffs),
            FilterBank::create_orthogonal_reconstruct_kernels(reconstruct_lowpass_coeffs)
        )
    );
}

Wavelet create_symlets(int order)
{
    auto name = internal::get_orthogonal_name(SYMLETS_PREFIX, order);
    internal::check_wavelet_name(name, SYMLETS_FAMILY, SYMLETS_FILTER_COEFFS);
    cv::Mat reconstruct_lowpass_coeffs(SYMLETS_FILTER_COEFFS[name]);

    return Wavelet(
        order, // vanishing_moments_psi
        0, // vanishing_moments_phi
        true, // orthogonal
        true, // biorthogonal
        Symmetry::NEAR_SYMMETRIC, // symmetry
        SYMLETS_FAMILY, // family
        name, // name
        FilterBank(
            FilterBank::create_orthogonal_decompose_kernels(reconstruct_lowpass_coeffs),
            FilterBank::create_orthogonal_reconstruct_kernels(reconstruct_lowpass_coeffs)
        )
    );
}

Wavelet create_coiflets(int order)
{
    auto name = internal::get_orthogonal_name(COIFLETS_PREFIX, order);
    internal::check_wavelet_name(name, COIFLETS_FAMILY, COIFLETS_FILTER_COEFFS);
    cv::Mat reconstruct_lowpass_coeffs(COIFLETS_FILTER_COEFFS[name]);

    return Wavelet(
        2 * order, // vanishing_moments_psi
        2 * order - 1, // vanishing_moments_phi
        true, // orthogonal
        true, // biorthogonal
        Symmetry::NEAR_SYMMETRIC, // symmetry
        COIFLETS_FAMILY, // family
        name, // name
        FilterBank(
            FilterBank::create_orthogonal_decompose_kernels(reconstruct_lowpass_coeffs),
            FilterBank::create_orthogonal_reconstruct_kernels(reconstruct_lowpass_coeffs)
        )
    );
}

Wavelet create_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi)
{
    auto name = internal::get_biorthogonal_name(
        BIORTHOGONAL_PREFIX,
        vanishing_moments_psi,
        vanishing_moments_phi
    );
    internal::check_wavelet_name(name, BIORTHOGONAL_FAMILY, BIORTHOGONAL_FILTER_COEFFS);
    cv::Mat reconstruct_lowpass_coeffs(BIORTHOGONAL_FILTER_COEFFS.at(name).first);
    cv::Mat decompose_lowpass_coeffs(BIORTHOGONAL_FILTER_COEFFS.at(name).second);

    return Wavelet(
        vanishing_moments_psi, // vanishing_moments_psi
        vanishing_moments_phi, // vanishing_moments_phi
        false, // orthogonal
        true, // biorthogonal
        Symmetry::SYMMETRIC, // symmetry
        BIORTHOGONAL_FAMILY, // family
        name, // name
        FilterBank(
            FilterBank::create_biorthogonal_decompose_kernels(
                reconstruct_lowpass_coeffs,
                decompose_lowpass_coeffs
            ),
            FilterBank::create_biorthogonal_reconstruct_kernels(
                reconstruct_lowpass_coeffs,
                decompose_lowpass_coeffs
            )
        )
    );
}

Wavelet create_reverse_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi)
{
    auto name = internal::get_biorthogonal_name(
        BIORTHOGONAL_PREFIX,
        vanishing_moments_psi,
        vanishing_moments_phi
    );
    auto family = "Reverse " + BIORTHOGONAL_FAMILY;
    internal::check_wavelet_name(name, family, BIORTHOGONAL_FILTER_COEFFS, "r");
    cv::Mat reconstruct_lowpass_coeffs(BIORTHOGONAL_FILTER_COEFFS.at(name).first);
    cv::Mat decompose_lowpass_coeffs(BIORTHOGONAL_FILTER_COEFFS.at(name).second);

    return Wavelet(
        vanishing_moments_psi, // vanishing_moments_psi
        vanishing_moments_phi, // vanishing_moments_phi
        false, // orthogonal
        true, // biorthogonal
        Symmetry::SYMMETRIC, // symmetry
        family, // family
        "r" + name, // name
        //  Normally, the FilterBank::create_biorthogonal_*_kernels functions
        //  take reconstruct_lowpass_coeffs as the first argument and
        //  decompose_lowpass_coeffs as the second.  But here we reverse the
        //  order per the definition of the reverse biorthogonal wavelet (i.e.
        //  this is not a mistake).
        FilterBank(
            FilterBank::create_biorthogonal_decompose_kernels(
                decompose_lowpass_coeffs,
                reconstruct_lowpass_coeffs
            ),
            FilterBank::create_biorthogonal_reconstruct_kernels(
                decompose_lowpass_coeffs,
                reconstruct_lowpass_coeffs
            )
        )
    );
}

namespace internal
{
std::string get_orthogonal_name(const std::string& prefix, int order)
{
    return prefix + std::to_string(order);
}

std::string get_biorthogonal_name(
    const std::string& prefix,
    int vanishing_moments_psi,
    int vanishing_moments_phi
)
{
    return prefix + std::to_string(vanishing_moments_psi) + "." + std::to_string(vanishing_moments_phi);
}

#ifndef DISABLE_ARG_CHECK
template <typename V>
void check_wavelet_name(
    const std::string& name,
    const std::string& family,
    const std::map<std::string, V>& filter_coeffs,
    const std::string& name_prefix
)
{
    if (!filter_coeffs.contains(name)) {
        std::stringstream available_names;
        for (auto name : std::views::keys(filter_coeffs))
            available_names << name_prefix << name << ", ";

        available_names.seekp(available_names.tellp() - 2);

        throw_bad_arg(
            "Invalid ", family, " wavelet order. ",
            "Must be one of: ", available_names.str(), ". ",
            "Got ", name_prefix + name, "."
        );
    }
}
#else
template <typename V>
inline void check_wavelet_name(
    const std::string& name,
    const std::string& family,
    const std::map<std::string, V>& filter_coeffs,
    const std::string& name_prefix
)
{
}
#endif  // DISABLE_ARG_CHECK

} // namespace internal
} // namespace wavelet

