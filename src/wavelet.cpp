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

/**
 * -----------------------------------------------------------------------------
 * WaveletFilterBank
 * -----------------------------------------------------------------------------
*/
bool KernelPair::operator==(const KernelPair& other) const
{
    return this == &other || (
        equals(_lowpass, other._lowpass) && equals(_highpass, other._highpass)
    );
}

WaveletFilterBankImpl::WaveletFilterBankImpl() :
    decompose_lowpass(0, 0, CV_64F),
    decompose_highpass(0, 0, CV_64F),
    reconstruct_even_lowpass(0, 0, CV_64F),
    reconstruct_odd_lowpass(0, 0, CV_64F),
    reconstruct_even_highpass(0, 0, CV_64F),
    reconstruct_odd_highpass(0, 0, CV_64F),
    filter_length(0)
{
}

WaveletFilterBankImpl::WaveletFilterBankImpl(
    const cv::Mat& reconstruct_lowpass_,
    const cv::Mat& reconstruct_highpass_,
    const cv::Mat& decompose_lowpass_,
    const cv::Mat& decompose_highpass_
) :
    filter_length(std::max(decompose_lowpass_.total(), decompose_highpass_.total()))
{
    check_kernels(
        reconstruct_lowpass_,
        reconstruct_highpass_,
        decompose_lowpass_,
        decompose_highpass_
    );

    cv::flip(decompose_lowpass_, decompose_lowpass, -1);
    cv::flip(decompose_highpass_, decompose_highpass, -1);

    cv::Mat flipped_reconstruct_lowpass;
    cv::flip(reconstruct_lowpass_, flipped_reconstruct_lowpass, -1);
    split_kernel_into_odd_and_even_parts(
        flipped_reconstruct_lowpass,
        reconstruct_even_lowpass,
        reconstruct_odd_lowpass
    );

    cv::Mat flipped_reconstruct_highpass;
    cv::flip(reconstruct_highpass_, flipped_reconstruct_highpass, -1);
    split_kernel_into_odd_and_even_parts(
        flipped_reconstruct_highpass,
        reconstruct_even_highpass,
        reconstruct_odd_highpass
    );
}

void WaveletFilterBankImpl::split_kernel_into_odd_and_even_parts(cv::InputArray kernel, cv::OutputArray even_kernel, cv::OutputArray odd_kernel) const
{
    dispatch_on_pixel_type<SplitKernelIntoOddAndEvenParts>(
        kernel.type(),
        kernel,
        even_kernel,
        odd_kernel
    );
}

void WaveletFilterBankImpl::merge_even_and_odd_kernels(
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

KernelPair WaveletFilterBankImpl::reconstruct_kernels() const
{
    cv::Mat lowpass;
    merge_even_and_odd_kernels(reconstruct_even_lowpass, reconstruct_odd_lowpass, lowpass);
    cv::flip(lowpass, lowpass, -1);

    cv::Mat highpass;
    merge_even_and_odd_kernels(reconstruct_even_highpass, reconstruct_odd_highpass, highpass);
    cv::flip(highpass, highpass, -1);

    return KernelPair(lowpass, highpass);
}

KernelPair WaveletFilterBankImpl::decompose_kernels() const
{
    cv::Mat lowpass;
    cv::flip(decompose_lowpass, lowpass, -1);
    cv::Mat highpass;
    cv::flip(decompose_highpass, highpass, -1);

    return KernelPair(lowpass, highpass);
}

#ifndef DISABLE_ARG_CHECKS
void WaveletFilterBankImpl::check_kernels(
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass,
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass
) const
{
    std::vector<cv::Mat> kernels = {
        reconstruct_lowpass,
        reconstruct_highpass,
        decompose_lowpass,
        decompose_highpass
    };
    if (!std::ranges::all_of(kernels, [](auto kernel) { return kernel.empty(); })
        && !std::ranges::none_of(kernels, [](auto kernel) { return kernel.empty(); })) {
        std::stringstream message;
        message
            << "FilterBank: Kernels must all be empty or all nonempty, got "
            << (reconstruct_lowpass.empty() ? "empty" : "nonempty") << " reconstruct_lowpass, "
            << (reconstruct_highpass.empty() ? "empty" : "nonempty") << " reconstruct_highpass, "
            << (decompose_lowpass.empty() ? "empty" : "nonempty") << " decompose_lowpass, and "
            << (decompose_highpass.empty() ? "empty" : "nonempty") << " decompose_highpass.";
        CV_Error(cv::Error::StsBadArg, message.str());
    }

    if (reconstruct_lowpass.size() != reconstruct_highpass.size()) {
        std::stringstream message;
        message
            << "FilterBank: reconstruct_lowpass must have same size as reconstruct_highpass, "
            << "got reconstruct_lowpass.size() = " << reconstruct_lowpass.size() << ", "
            << "reconstruct_highpass.size() = " << reconstruct_highpass.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }

    if (decompose_lowpass.size() != decompose_highpass.size()) {
        std::stringstream message;
        message
            << "FilterBank: decompose_lowpass must have same size as decompose_highpass, "
            << "got decompose_lowpass.size() = " << decompose_lowpass.size() << ", "
            << "decompose_highpass.size() = " << decompose_highpass.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}
#else
inline void WaveletFilterBankImpl::check_kernels(
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass,
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass
) const
{
}
#endif

WaveletFilterBank::WaveletFilterBank() :
    _p(std::make_shared<WaveletFilterBankImpl>())
{
}

WaveletFilterBank::WaveletFilterBank(
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass,
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass
) :
    _p(
        std::make_shared<WaveletFilterBankImpl>(
            reconstruct_lowpass,
            reconstruct_highpass,
            decompose_lowpass,
            decompose_highpass
        )
    )
{
}

WaveletFilterBank::WaveletFilterBank(
    const KernelPair& reconstruct_kernels,
    const KernelPair& decompose_kernels
) :
    WaveletFilterBank(
        reconstruct_kernels.lowpass(),
        reconstruct_kernels.highpass(),
        decompose_kernels.lowpass(),
        decompose_kernels.highpass()
    )
{
}

void WaveletFilterBank::forward(
    cv::InputArray input,
    cv::OutputArray approx,
    cv::OutputArray horizontal_detail,
    cv::OutputArray vertical_detail,
    cv::OutputArray diagonal_detail,
    int border_type,
    const cv::Scalar& border_value
) const
{
    if (input.rows() <= 1 || input.cols() <= 1) {
        std::stringstream message;
        message << "Input size must be greater [1 x 1], got " << input.size();
        CV_Error(cv::Error::StsBadSize, message.str());
    }

    cv::Mat padded_input;
    pad(input, padded_input, border_type, border_value);

    //  stage 1
    cv::Mat stage1_lowpass_output;
    convolve_rows_and_downsample_cols(padded_input, stage1_lowpass_output, _p->decompose_lowpass);
    cv::Mat stage1_highpass_output;
    convolve_rows_and_downsample_cols(padded_input, stage1_highpass_output, _p->decompose_highpass);

    //  stage 2
    convolve_cols_and_downsample_rows(stage1_lowpass_output, approx, _p->decompose_lowpass);
    convolve_cols_and_downsample_rows(stage1_lowpass_output, horizontal_detail, _p->decompose_highpass);
    convolve_cols_and_downsample_rows(stage1_highpass_output, vertical_detail, _p->decompose_lowpass);
    convolve_cols_and_downsample_rows(stage1_highpass_output, diagonal_detail, _p->decompose_highpass);
}

void WaveletFilterBank::inverse(
    cv::InputArray approx,
    cv::InputArray horizontal_detail,
    cv::InputArray vertical_detail,
    cv::InputArray diagonal_detail,
    cv::OutputArray output,
    const cv::Size& output_size
) const
{
    //  stage 1a
    cv::Mat stage1a_lowpass_output;
    upsample_rows_and_convolve_cols(
        approx,
        stage1a_lowpass_output,
        _p->reconstruct_even_lowpass,
        _p->reconstruct_odd_lowpass,
        output_size
    );
    cv::Mat stage1a_highpass_output;
    upsample_rows_and_convolve_cols(
        horizontal_detail,
        stage1a_highpass_output,
        _p->reconstruct_even_highpass,
        _p->reconstruct_odd_highpass,
        output_size
    );

    //  stage 1b
    cv::Mat stage1b_lowpass_output;
    upsample_rows_and_convolve_cols(
        vertical_detail,
        stage1b_lowpass_output,
        _p->reconstruct_even_lowpass,
        _p->reconstruct_odd_lowpass,
        output_size
    );
    cv::Mat stage1b_highpass_output;
    upsample_rows_and_convolve_cols(
        diagonal_detail,
        stage1b_highpass_output,
        _p->reconstruct_even_highpass,
        _p->reconstruct_odd_highpass,
        output_size
    );

    //  stage 2
    cv::Mat stage2_lowpass_output;
    upsample_cols_and_convolve_rows(
        stage1a_lowpass_output + stage1a_highpass_output,
        stage2_lowpass_output,
        _p->reconstruct_even_lowpass,
        _p->reconstruct_odd_lowpass,
        output_size
    );
    cv::Mat stage2_highpass_output;
    upsample_cols_and_convolve_rows(
        stage1b_lowpass_output + stage1b_highpass_output,
        stage2_highpass_output,
        _p->reconstruct_even_highpass,
        _p->reconstruct_odd_highpass,
        output_size
    );

    output.assign(stage2_lowpass_output + stage2_highpass_output);
}

cv::Size WaveletFilterBank::output_size(const cv::Size& input_size) const
{
    return cv::Size(output_size(input_size.width), output_size(input_size.height));
}

int WaveletFilterBank::output_size(int input_size) const
{
    return 2 * subband_size(input_size);
}

cv::Size WaveletFilterBank::subband_size(const cv::Size& input_size) const
{
    return cv::Size(subband_size(input_size.width), subband_size(input_size.height));
}

int WaveletFilterBank::subband_size(int input_size) const
{
    return (input_size + filter_length() - 1) / 2;
}

void WaveletFilterBank::pad(
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

void WaveletFilterBank::convolve_rows_and_downsample_cols(
    cv::InputArray data,
    cv::OutputArray output,
    const cv::Mat& kernel
) const
{
    dispatch_on_pixel_type<ConvolveRowsAndDownsampleCols>(
        data.type(),
        data,
        output,
        kernel.t()
    );
}

void WaveletFilterBank::convolve_cols_and_downsample_rows(
    cv::InputArray data,
    cv::OutputArray output,
    const cv::Mat& kernel
) const
{
    dispatch_on_pixel_type<ConvolveColsAndDownsampleRows>(
        data.type(),
        data,
        output,
        kernel
    );
}

void WaveletFilterBank::upsample_cols_and_convolve_rows(
    cv::InputArray data,
    cv::OutputArray output,
    const cv::Mat& even_kernel,
    const cv::Mat& odd_kernel,
    const cv::Size& valid_size
) const
{
    dispatch_on_pixel_type<UpsampleColsAndConvolveRows>(
        data.type(),
        data,
        output,
        even_kernel.t(),
        odd_kernel.t(),
        valid_size
    );
}

void WaveletFilterBank::upsample_rows_and_convolve_cols(
    cv::InputArray data,
    cv::OutputArray output,
    const cv::Mat& even_kernel,
    const cv::Mat& odd_kernel,
    const cv::Size& valid_size
) const
{
    dispatch_on_pixel_type<UpsampleRowsAndConvolveCols>(
        data.type(),
        data,
        output,
        even_kernel,
        odd_kernel,
        valid_size
    );
}

bool WaveletFilterBank::operator==(const WaveletFilterBank& other) const
{
    return this == &other || (
        equals(_p->decompose_lowpass, other._p->decompose_lowpass)
        && equals(_p->decompose_highpass, other._p->decompose_highpass)
        && equals(_p->reconstruct_even_lowpass, other._p->reconstruct_even_lowpass)
        && equals(_p->reconstruct_odd_lowpass, other._p->reconstruct_odd_lowpass)
        && equals(_p->reconstruct_even_highpass, other._p->reconstruct_even_highpass)
        && equals(_p->reconstruct_odd_highpass, other._p->reconstruct_odd_highpass)
    );
}

std::ostream& operator<<(std::ostream& stream, const WaveletFilterBank& filter_bank)
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

} // namespace internal




/**
 * =============================================================================
 * Public API
 * =============================================================================
*/
/**
 * -----------------------------------------------------------------------------
 * Wavelet
 * -----------------------------------------------------------------------------
*/
Wavelet::Wavelet(
        int vanishing_moments_psi,
        int vanishing_moments_phi,
        int support_width,
        bool orthogonal,
        bool biorthogonal,
        Wavelet::Symmetry symmetry,
        bool compact_support,
        const std::string& family,
        const std::string& name,
        const FilterBank& filter_bank
    ) :
    _p(
        std::make_shared<WaveletImpl>(
            vanishing_moments_psi,
            vanishing_moments_phi,
            support_width,
            orthogonal,
            biorthogonal,
            symmetry,
            compact_support,
            family,
            name,
            filter_bank
        )
    )
{
}

Wavelet::Wavelet() : _p(std::make_shared<WaveletImpl>())
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
    {"haar", haar},
    //  daubechies
    {"db1", std::bind(daubechies, 1)},
    {"db2", std::bind(daubechies, 2)},
    {"db3", std::bind(daubechies, 3)},
    {"db4", std::bind(daubechies, 4)},
    {"db5", std::bind(daubechies, 5)},
    {"db6", std::bind(daubechies, 6)},
    {"db7", std::bind(daubechies, 7)},
    {"db8", std::bind(daubechies, 8)},
    {"db9", std::bind(daubechies, 9)},
    {"db10", std::bind(daubechies, 10)},
    {"db11", std::bind(daubechies, 11)},
    {"db12", std::bind(daubechies, 12)},
    {"db13", std::bind(daubechies, 13)},
    {"db14", std::bind(daubechies, 14)},
    {"db15", std::bind(daubechies, 15)},
    {"db16", std::bind(daubechies, 16)},
    {"db17", std::bind(daubechies, 17)},
    {"db18", std::bind(daubechies, 18)},
    {"db19", std::bind(daubechies, 19)},
    {"db20", std::bind(daubechies, 20)},
    {"db21", std::bind(daubechies, 21)},
    {"db22", std::bind(daubechies, 22)},
    {"db23", std::bind(daubechies, 23)},
    {"db24", std::bind(daubechies, 24)},
    {"db25", std::bind(daubechies, 25)},
    {"db26", std::bind(daubechies, 26)},
    {"db27", std::bind(daubechies, 27)},
    {"db28", std::bind(daubechies, 28)},
    {"db29", std::bind(daubechies, 29)},
    {"db30", std::bind(daubechies, 30)},
    {"db31", std::bind(daubechies, 31)},
    {"db32", std::bind(daubechies, 32)},
    {"db33", std::bind(daubechies, 33)},
    {"db34", std::bind(daubechies, 34)},
    {"db35", std::bind(daubechies, 35)},
    {"db36", std::bind(daubechies, 36)},
    {"db37", std::bind(daubechies, 37)},
    {"db38", std::bind(daubechies, 38)},
    //  symlets
    {"sym2", std::bind(symlets, 2)},
    {"sym3", std::bind(symlets, 3)},
    {"sym4", std::bind(symlets, 4)},
    {"sym5", std::bind(symlets, 5)},
    {"sym6", std::bind(symlets, 6)},
    {"sym7", std::bind(symlets, 7)},
    {"sym8", std::bind(symlets, 8)},
    {"sym9", std::bind(symlets, 9)},
    {"sym10", std::bind(symlets, 10)},
    {"sym11", std::bind(symlets, 11)},
    {"sym12", std::bind(symlets, 12)},
    {"sym13", std::bind(symlets, 13)},
    {"sym14", std::bind(symlets, 14)},
    {"sym15", std::bind(symlets, 15)},
    {"sym16", std::bind(symlets, 16)},
    {"sym17", std::bind(symlets, 17)},
    {"sym18", std::bind(symlets, 18)},
    {"sym19", std::bind(symlets, 19)},
    {"sym20", std::bind(symlets, 20)},
    //  coiflets
    {"coif1", std::bind(coiflets, 1)},
    {"coif2", std::bind(coiflets, 2)},
    {"coif3", std::bind(coiflets, 3)},
    {"coif4", std::bind(coiflets, 4)},
    {"coif5", std::bind(coiflets, 5)},
    {"coif6", std::bind(coiflets, 6)},
    {"coif7", std::bind(coiflets, 7)},
    {"coif8", std::bind(coiflets, 8)},
    {"coif9", std::bind(coiflets, 9)},
    {"coif10", std::bind(coiflets, 10)},
    {"coif11", std::bind(coiflets, 11)},
    {"coif12", std::bind(coiflets, 12)},
    {"coif13", std::bind(coiflets, 13)},
    {"coif14", std::bind(coiflets, 14)},
    {"coif15", std::bind(coiflets, 15)},
    {"coif16", std::bind(coiflets, 16)},
    {"coif17", std::bind(coiflets, 17)},
    //  biorthongonal
    {"bior1.1", std::bind(wavelet::biorthogonal, 1, 1)},
    {"bior1.3", std::bind(wavelet::biorthogonal, 1, 3)},
    {"bior1.5", std::bind(wavelet::biorthogonal, 1, 5)},
    {"bior2.2", std::bind(wavelet::biorthogonal, 2, 2)},
    {"bior2.4", std::bind(wavelet::biorthogonal, 2, 4)},
    {"bior2.6", std::bind(wavelet::biorthogonal, 2, 6)},
    {"bior2.8", std::bind(wavelet::biorthogonal, 2, 8)},
    {"bior3.1", std::bind(wavelet::biorthogonal, 3, 1)},
    {"bior3.3", std::bind(wavelet::biorthogonal, 3, 3)},
    {"bior3.5", std::bind(wavelet::biorthogonal, 3, 5)},
    {"bior3.7", std::bind(wavelet::biorthogonal, 3, 7)},
    {"bior3.9", std::bind(wavelet::biorthogonal, 3, 9)},
    {"bior4.4", std::bind(wavelet::biorthogonal, 4, 4)},
    {"bior5.5", std::bind(wavelet::biorthogonal, 5, 5)},
    {"bior6.8", std::bind(wavelet::biorthogonal, 6, 8)},
};


void check_wavelet_order(int order, int min_order, int max_order, const std::string family)
{
    if (order < min_order || order > max_order) {
        std::stringstream message;
        message
            << "Invalid " << family << " wavelet order.  "
            << "Must be " << min_order << " <= order <= " << max_order << " - "
            << "got order = " << order << ".";
        CV_Error(cv::Error::StsBadArg, message.str());
    }
}

std::string get_and_check_biorthogonal_name(int vanishing_moments_psi, int vanishing_moments_phi)
{
    auto name = (std::stringstream()
        << "bior" << vanishing_moments_psi << "." << vanishing_moments_phi
    ).str();

    if (!BIORTHOGONAL_FILTER_COEFFS.contains(name)) {
        std::stringstream message;
        message
            << "Invalid Biorthogonal wavelet order.  "
            << "Must be one of: ";
        for (auto x : std::views::keys(BIORTHOGONAL_FILTER_COEFFS))
            message << x << ", ";

        message
            << " - got biorN.M with " << "N = " << vanishing_moments_psi << " and M = " << vanishing_moments_phi <<  ".";
        CV_Error(cv::Error::StsBadArg, message.str());
    }

    return name;
}

Wavelet haar()
{
    return Wavelet(
        1, // vanishing_moments_psi
        0, // vanishing_moments_phi
        1, // support_width
        true, // orthogonal
        true, // biorthogonal
        Wavelet::Symmetry::ASYMMETRIC, // symmetry
        true, // compact_support
        "Haar", // family
        "haar", // name
        Wavelet::FilterBank(
            Wavelet::FilterBank::build_orthogonal_reconstruct_kernels(DAUBECHIES_FILTER_COEFFS[1]),
            Wavelet::FilterBank::build_orthogonal_decompose_kernels(DAUBECHIES_FILTER_COEFFS[1])
        )
    );
}

Wavelet daubechies(int order)
{
    check_wavelet_order(order, DAUBECHIES_MIN_ORDER, DAUBECHIES_MAX_ORDER, "Daubechies");

    return Wavelet(
        // 2 * order, // vanishing_moments_psi
        order, // vanishing_moments_psi
        0, // vanishing_moments_phi
        2 * order - 1, // support_width
        true, // orthogonal
        true, // biorthogonal
        Wavelet::Symmetry::ASYMMETRIC, // symmetry
        true, // compact_support
        "Daubechies", // family
        "db" + std::to_string(order), // name
        Wavelet::FilterBank(
            Wavelet::FilterBank::build_orthogonal_reconstruct_kernels(DAUBECHIES_FILTER_COEFFS[order]),
            Wavelet::FilterBank::build_orthogonal_decompose_kernels(DAUBECHIES_FILTER_COEFFS[order])
        )
    );
}

Wavelet symlets(int order)
{
    check_wavelet_order(order, SYMLETS_MIN_ORDER, SYMLETS_MAX_ORDER, "Symlets");

    return Wavelet(
        order, // vanishing_moments_psi
        0, // vanishing_moments_phi
        2 * order - 1, // support_width
        true, // orthogonal
        true, // biorthogonal
        Wavelet::Symmetry::NEAR_SYMMETRIC, // symmetry
        true, // compact_support
        "Symlets", // family
        "sym" + std::to_string(order), // name
        Wavelet::FilterBank(
            Wavelet::FilterBank::build_orthogonal_reconstruct_kernels(SYMLET_FILTER_COEFFS[order]),
            Wavelet::FilterBank::build_orthogonal_decompose_kernels(SYMLET_FILTER_COEFFS[order])
        )
    );
}

Wavelet coiflets(int order)
{
    check_wavelet_order(order, COIFLETS_MIN_ORDER, COIFLETS_MAX_ORDER, "Coiflets");

    return Wavelet(
        2 * order, // vanishing_moments_psi
        2 * order - 1, // vanishing_moments_phi
        6 * order - 1, // support_width
        true, // orthogonal
        true, // biorthogonal
        Wavelet::Symmetry::NEAR_SYMMETRIC, // symmetry
        true, // compact_support
        "Coiflets", // family
        "coif" + std::to_string(order), // name
        Wavelet::FilterBank(
            Wavelet::FilterBank::build_orthogonal_reconstruct_kernels(COIFLETS_FILTER_COEFFS[order]),
            Wavelet::FilterBank::build_orthogonal_decompose_kernels(COIFLETS_FILTER_COEFFS[order])
        )
    );
}

Wavelet biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi)
{
    auto name = get_and_check_biorthogonal_name(vanishing_moments_psi, vanishing_moments_phi);
    auto reconstruct_lowpass_coeffs = BIORTHOGONAL_FILTER_COEFFS.at(name).first;
    auto decompose_lowpass_coeffs = BIORTHOGONAL_FILTER_COEFFS.at(name).second;

    return Wavelet(
        vanishing_moments_psi, // vanishing_moments_psi
        vanishing_moments_phi, // vanishing_moments_phi
        -1, // support_width
        false, // orthogonal
        true, // biorthogonal
        Wavelet::Symmetry::SYMMETRIC, // symmetry
        true, // compact_support
        "Biorthogonal", // family
        name, // name
        Wavelet::FilterBank(
            Wavelet::FilterBank::build_biorthogonal_reconstruct_kernels(
                reconstruct_lowpass_coeffs,
                decompose_lowpass_coeffs
            ),
            Wavelet::FilterBank::build_biorthogonal_decompose_kernels(
                reconstruct_lowpass_coeffs,
                decompose_lowpass_coeffs
            )
        )
    );
}

} // namespace wavelet

