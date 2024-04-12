#include "cvwt/filterbank.hpp"
#include "cvwt/utils.hpp"
#include <opencv2/imgproc.hpp>
#include <ranges>

namespace cvwt
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

    bool was_prepared = is_decompose_prepared(image.type());
    if (!was_prepared)
        prepare_decompose(image.type());

    cv::Mat promoted_image;
    promote_image(image, promoted_image);

    cv::Mat padded_image;
    pad(promoted_image, padded_image, border_type, border_value);

    //  Stage 1
    cv::Mat stage1_lowpass_output;
    convolve_rows_and_downsample_cols(
        padded_image,
        stage1_lowpass_output,
        _p->promoted_decompose.lowpass
    );
    cv::Mat stage1_highpass_output;
    convolve_rows_and_downsample_cols(
        padded_image,
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
        finish_decompose();
}

void FilterBank::prepare_decompose(int type) const
{
    if (is_decompose_prepared(type))
        return;

    promote_kernel(_p->decompose.lowpass, _p->promoted_decompose.lowpass, type);
    promote_kernel(_p->decompose.highpass, _p->promoted_decompose.highpass, type);
}

void FilterBank::finish_decompose() const
{
    _p->promoted_decompose.release();
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

    bool was_prepared = is_reconstruct_prepared(approx.type());
    if (!was_prepared)
        prepare_reconstruct(approx.type());

    cv::Mat promoted_approx;
    promote_image(approx, promoted_approx);
    cv::Mat promoted_horizontal_detail;
    promote_image(horizontal_detail, promoted_horizontal_detail);
    cv::Mat promoted_vertical_detail;
    promote_image(vertical_detail, promoted_vertical_detail);
    cv::Mat promoted_diagonal_detail;
    promote_image(diagonal_detail, promoted_diagonal_detail);

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
        finish_reconstruct();
}

void FilterBank::prepare_reconstruct(int type) const
{
    if (is_reconstruct_prepared(type))
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

void FilterBank::finish_reconstruct() const
{
    _p->promoted_reconstruct.release();
}

cv::Size FilterBank::subband_size(const cv::Size& image_size) const
{
    return cv::Size(
        (image_size.width + filter_length() - 1) / 2,
        (image_size.height + filter_length() - 1) / 2
    );
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

void FilterBank::promote_image(
    cv::InputArray image,
    cv::OutputArray promoted_image
) const
{
    int type = promote_type(image.type());
    image.getMat().convertTo(promoted_image, type);
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
    internal::dispatch_on_pixel_type<internal::ConvolveRowsAndDownsampleCols>(
        input.type(),
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
    internal::dispatch_on_pixel_type<internal::ConvolveColsAndDownsampleRows>(
        input.type(),
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
    internal::dispatch_on_pixel_type<internal::UpsampleColsAndConvolveRows>(
        input.type(),
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
    internal::dispatch_on_pixel_type<internal::UpsampleRowsAndConvolveCols>(
        input.type(),
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
        equals(_p->decompose.lowpass, other._p->decompose.lowpass)
        && equals(_p->decompose.highpass, other._p->decompose.highpass)
        && equals(_p->reconstruct.even_lowpass, other._p->reconstruct.even_lowpass)
        && equals(_p->reconstruct.odd_lowpass, other._p->reconstruct.odd_lowpass)
        && equals(_p->reconstruct.even_highpass, other._p->reconstruct.even_highpass)
        && equals(_p->reconstruct.odd_highpass, other._p->reconstruct.odd_highpass)
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

