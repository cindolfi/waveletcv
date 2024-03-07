#include "wavelet/wavelet.hpp"
#include "wavelet/daubechies.hpp"
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <functional>

namespace wavelet
{

namespace internal
{
/**
 * -----------------------------------------------------------------------------
 * WaveletFilterBank
 * -----------------------------------------------------------------------------
*/
WaveletFilterBank::WaveletFilterBank(
    const WaveletFilterBank::KernelPair& synthesis_kernels,
    const WaveletFilterBank::KernelPair& analysis_kernels
) :
    _synthesis_kernels(synthesis_kernels.flipped()),
    _analysis_kernels(analysis_kernels.flipped())
{
}

WaveletFilterBank::WaveletFilterBank(
    const cv::Mat& synthesis_lowpass,
    const cv::Mat& synthesis_highpass,
    const cv::Mat& analysis_lowpass,
    const cv::Mat& analysis_highpass
) :
    WaveletFilterBank(
        WaveletFilterBank::KernelPair(synthesis_lowpass, synthesis_highpass),
        WaveletFilterBank::KernelPair(analysis_lowpass, analysis_highpass)
    )
{
}

WaveletFilterBank::KernelPair WaveletFilterBank::KernelPair::flipped() const
{
    cv::Mat lowpass;
    cv::flip(_lowpass, lowpass, 0);

    cv::Mat highpass;
    cv::flip(_highpass, highpass, 0);

    return KernelPair(lowpass, highpass);
}

void WaveletFilterBank::forward(
    cv::InputArray x,
    cv::OutputArray approx,
    cv::OutputArray horizontal_detail,
    cv::OutputArray vertical_detail,
    cv::OutputArray diagonal_detail,
    int border_type
) const
{
    // int depth = max_possible_depth(x);
    // if (depth <= 0)
    //     return;

    auto data = x.getMat();

    auto stage1_lowpass_output = forward_stage1_lowpass(data, border_type);
    auto stage1_highpass_output = forward_stage1_highpass(data, border_type);

    forward_stage2_lowpass(stage1_lowpass_output, approx, border_type);
    forward_stage2_highpass(stage1_lowpass_output, horizontal_detail, border_type);
    forward_stage2_lowpass(stage1_highpass_output, vertical_detail, border_type);
    forward_stage2_highpass(stage1_highpass_output, diagonal_detail, border_type);
}

cv::Mat WaveletFilterBank::forward_stage1_lowpass(cv::InputArray data, int border_type) const
{
    cv::Mat result;
    forward_stage1_lowpass(data, result, border_type);
    return result;
}

cv::Mat WaveletFilterBank::forward_stage1_highpass(cv::InputArray data, int border_type) const
{
    cv::Mat result;
    forward_stage1_highpass(data, result, border_type);
    return result;
}

cv::Mat WaveletFilterBank::forward_stage2_lowpass(cv::InputArray data, int border_type) const
{
    cv::Mat result;
    forward_stage2_lowpass(data, result, border_type);
    return result;
}

cv::Mat WaveletFilterBank::forward_stage2_highpass(cv::InputArray data, int border_type) const
{
    cv::Mat result;
    forward_stage2_highpass(data, result, border_type);
    return result;
}

void WaveletFilterBank::forward_stage1_lowpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat filtered;
    filter_rows(data, filtered, _synthesis_kernels.lowpass(), border_type);
    downsample_cols(filtered, output);
}

void WaveletFilterBank::forward_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat filtered;
    filter_rows(data, filtered, _synthesis_kernels.highpass(), border_type);
    downsample_cols(filtered, output);
}

void WaveletFilterBank::forward_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat filtered;
    filter_cols(data, filtered, _synthesis_kernels.lowpass(), border_type);
    downsample_rows(filtered, output);
}

void WaveletFilterBank::forward_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat filtered;
    filter_cols(data, filtered, _synthesis_kernels.highpass(), border_type);
    downsample_rows(filtered, output);
}


void WaveletFilterBank::inverse(
    cv::InputArray approx,
    cv::InputArray horizontal_detail,
    cv::InputArray vertical_detail,
    cv::InputArray diagonal_detail,
    cv::OutputArray output,
    int border_type
) const
{
    auto x1 = inverse_stage1_lowpass(approx, border_type)
        + inverse_stage1_highpass(horizontal_detail, border_type);

    auto x2 = inverse_stage1_lowpass(vertical_detail, border_type)
        + inverse_stage1_highpass(diagonal_detail, border_type);

    output.assign(
        inverse_stage2_lowpass(x1, border_type)
        + inverse_stage2_highpass(x2, border_type)
    );
}

cv::Mat WaveletFilterBank::inverse_stage1_lowpass(cv::InputArray data, int border_type) const
{
    cv::Mat result;
    inverse_stage1_lowpass(data, result, border_type);
    return result;
}

cv::Mat WaveletFilterBank::inverse_stage1_highpass(cv::InputArray data, int border_type) const
{
    cv::Mat result;
    inverse_stage1_highpass(data, result, border_type);
    return result;
}

cv::Mat WaveletFilterBank::inverse_stage2_lowpass(cv::InputArray data, int border_type) const
{
    cv::Mat result;
    inverse_stage2_lowpass(data, result, border_type);
    return result;
}

cv::Mat WaveletFilterBank::inverse_stage2_highpass(cv::InputArray data, int border_type) const
{
    cv::Mat result;
    inverse_stage2_highpass(data, result, border_type);
    return result;
}

void WaveletFilterBank::inverse_stage1_lowpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat upsampled;
    upsample_rows(data, upsampled);
    filter_cols(upsampled, output, _analysis_kernels.lowpass(), border_type);
}

void WaveletFilterBank::inverse_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat upsampled;
    upsample_rows(data, upsampled);
    filter_cols(upsampled, output, _analysis_kernels.highpass(), border_type);
}

void WaveletFilterBank::inverse_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat upsampled;
    upsample_cols(data, upsampled);
    filter_rows(upsampled, output, _analysis_kernels.lowpass(), border_type);
}

void WaveletFilterBank::inverse_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat upsampled;
    upsample_cols(data, upsampled);
    filter_rows(upsampled, output, _analysis_kernels.highpass(), border_type);
}

void WaveletFilterBank::downsample_rows(cv::InputArray data, cv::OutputArray output) const
{
    //  use resize without any smoothing to implement downsampling by 2
    cv::resize(data, output, cv::Size(data.cols(), data.rows() / 2), 0.0, 0.0, cv::INTER_NEAREST);
}

void WaveletFilterBank::downsample_cols(cv::InputArray data, cv::OutputArray output) const
{
    //  use resize without any smoothing to implement downsampling by 2
    cv::resize(data, output, cv::Size(data.cols() / 2, data.rows()), 0.0, 0.0, cv::INTER_NEAREST);
}

void WaveletFilterBank::upsample_rows(cv::InputArray data, cv::OutputArray output) const
{
    //  use resize without any smoothing to implement upsampling by 2
    cv::resize(data, output, cv::Size(data.cols(), 2 * data.rows()), 0.0, 0.0, cv::INTER_NEAREST);
    auto output_matrix = output.getMat();
    for (int i = 0; i < output_matrix.rows; i += 2)
        output_matrix.row(i).setTo(0);
}

void WaveletFilterBank::upsample_cols(cv::InputArray data, cv::OutputArray output) const
{
    //  use resize without any smoothing to implement upsampling by 2
    cv::resize(data, output, cv::Size(2 * data.cols(), data.rows()), 0.0, 0.0, cv::INTER_NEAREST);
    auto output_matrix = output.getMat();
    for (int j = 0; j < output_matrix.cols; j += 2)
        output_matrix.col(j).setTo(0);
}

void WaveletFilterBank::filter_rows(cv::InputArray data, cv::OutputArray output, const cv::Mat& kernel, int border_type) const
{
    cv::filter2D(data, output, -1, kernel.t(), cv::Point(-1, -1), 0.0, border_type);
}

void WaveletFilterBank::filter_cols(cv::InputArray data, cv::OutputArray output, const cv::Mat& kernel, int border_type) const
{
    cv::filter2D(data, output, -1, kernel, cv::Point(-1, -1), 0.0, border_type);
}

template <typename T>
WaveletFilterBank::KernelPair WaveletFilterBank::build_analysis_kernels(const std::vector<T>& coeffs)
{
    std::vector<T> lowpass(coeffs.size());
    std::vector<T> highpass = coeffs;

    negate_evens(highpass);
    std::ranges::reverse_copy(coeffs, lowpass.begin());

    return KernelPair(lowpass, highpass);
}

template <typename T>
WaveletFilterBank::KernelPair WaveletFilterBank::build_synthesis_kernels(const std::vector<T>& coeffs)
{
    std::vector<T> lowpass = coeffs;
    std::vector<T> highpass(coeffs.size());

    std::reverse_copy(lowpass.begin(), lowpass.end(), highpass.begin());
    negate_odds(highpass);

    return KernelPair(lowpass, highpass);
}

template <typename T>
void WaveletFilterBank::negate_odds(std::vector<T>& coeffs)
{
    std::transform(
        coeffs.cbegin(),
        coeffs.cend(),
        coeffs.begin(),
        [&, i = 0] (auto coeff) mutable { return (i++ % 2 ? -1 : 1) * coeff; }
    );
}

template <typename T>
void WaveletFilterBank::negate_evens(std::vector<T>& coeffs)
{
    std::transform(
        coeffs.cbegin(),
        coeffs.cend(),
        coeffs.begin(),
        [&, i = 0] (auto coeff) mutable { return (i++ % 2 ? 1 : -1) * coeff; }
    );
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
        int order,
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
            order,
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

Wavelet Wavelet::create(const std::string& name)
{
    return _wavelet_factories.at(name)();
}

template<class... Args>
void Wavelet::register_factory(const std::string& name, Wavelet factory(Args...), const Args&... args)
{
    _wavelet_factories[name] = std::bind(factory, args...);
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
};


Wavelet haar()
{
    return Wavelet(
        1, // order
        2, // vanishing_moments_psi
        0, // vanishing_moments_phi
        1, // support_width
        true, // orthogonal
        true, // biorthogonal
        Wavelet::Symmetry::ASYMMETRIC, // symmetry
        true, // compact_support
        "Haar", // family
        "haar", // name
        Wavelet::FilterBank(
            Wavelet::FilterBank::build_synthesis_kernels(DAUBECHIES_FILTER_COEFFS[0]),
            Wavelet::FilterBank::build_analysis_kernels(DAUBECHIES_FILTER_COEFFS[0])
        )
    );
}

Wavelet daubechies(int order)
{
    return Wavelet(
        order, // order
        2 * order, // vanishing_moments_psi
        0, // vanishing_moments_phi
        2 * order - 1, // support_width
        true, // orthogonal
        true, // biorthogonal
        Wavelet::Symmetry::ASYMMETRIC, // symmetry
        true, // compact_support
        "Daubechies", // family
        "db" + std::to_string(order), // name
        Wavelet::FilterBank(
            Wavelet::FilterBank::build_synthesis_kernels(DAUBECHIES_FILTER_COEFFS[order - 1]),
            Wavelet::FilterBank::build_analysis_kernels(DAUBECHIES_FILTER_COEFFS[order - 1])
        )
    );
}

} // namespace wavelet






