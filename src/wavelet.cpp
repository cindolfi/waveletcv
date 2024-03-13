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
/**
 * -----------------------------------------------------------------------------
 * WaveletFilterBank
 * -----------------------------------------------------------------------------
*/
WaveletFilterBank::WaveletFilterBank(
    const WaveletFilterBank::KernelPair& reconstruct_kernels,
    const WaveletFilterBank::KernelPair& decompose_kernels
) :
    _reconstruct_kernels(reconstruct_kernels),
    _decompose_kernels(decompose_kernels)
{
}

WaveletFilterBank::WaveletFilterBank(
    const cv::Mat& reconstruct_lowpass,
    const cv::Mat& reconstruct_highpass,
    const cv::Mat& decompose_lowpass,
    const cv::Mat& decompose_highpass
) :
    WaveletFilterBank(
        WaveletFilterBank::KernelPair(reconstruct_lowpass, reconstruct_highpass),
        WaveletFilterBank::KernelPair(decompose_lowpass, decompose_highpass)
    )
{
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
    filter_rows(data, filtered, _decompose_kernels.lowpass(), border_type);
    downsample_cols(filtered, output);
}

void WaveletFilterBank::forward_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat filtered;
    filter_rows(data, filtered, _decompose_kernels.highpass(), border_type);
    downsample_cols(filtered, output);
}

void WaveletFilterBank::forward_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat filtered;
    filter_cols(data, filtered, _decompose_kernels.lowpass(), border_type);
    downsample_rows(filtered, output);
}

void WaveletFilterBank::forward_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type) const
{
    cv::Mat filtered;
    filter_cols(data, filtered, _decompose_kernels.highpass(), border_type);
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

void WaveletFilterBank::inverse_stage1_lowpass(cv::InputArray data, cv::OutputArray output, int border_type, int offset) const
{
    cv::Mat upsampled;
    upsample_rows(data, upsampled);
    filter_cols(upsampled, output, _reconstruct_kernels.lowpass(), border_type);
}

void WaveletFilterBank::inverse_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type, int offset) const
{
    cv::Mat upsampled;
    upsample_rows(data, upsampled);
    filter_cols(upsampled, output, _reconstruct_kernels.highpass(), border_type);
}

void WaveletFilterBank::inverse_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type, int offset) const
{
    cv::Mat upsampled;
    upsample_cols(data, upsampled);
    filter_rows(upsampled, output, _reconstruct_kernels.lowpass(), border_type);
}

void WaveletFilterBank::inverse_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type, int offset) const
{
    cv::Mat upsampled;
    upsample_cols(data, upsampled);
    filter_rows(upsampled, output, _reconstruct_kernels.highpass(), border_type);
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

std::ostream& operator<<(std::ostream& stream, const WaveletFilterBank& filter_bank)
{
    stream << "decompose:\n"
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
        // int order,
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
            // order,
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
        // auto valid_names = std::views::keys(BIORTHOGONAL_FILTER_COEFFS);
        // auto h = join(valid_names);
        std::stringstream message;
        // message
        //     << "Invalid Biorthogonal wavelet order.  "
        //     << "Must be one of: " << h << " - "
        //     << "got biorN.M with " << "N = " << vanishing_moments_psi << " and M = " << vanishing_moments_phi <<  ".";

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
        // 1, // order
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
        // order, // order
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
        // order, // order
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
        // order, // order
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
        // 10 * vanishing_moments_psi + vanishing_moments_phi, // order
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



// for(i = 0; i < w->rec_len; ++i){
    //     // w->rec_lo_float[i] = bior_float[N - 1][0][i + n];
    //     // w->dec_lo_float[i] = bior_float[N - 1][M_idx + 1][w->dec_len - 1 - i];

    //     // w->rec_hi_float[i] = ((i % 2) ? -1 : 1)
    //     //     * bior_float[N - 1][M_idx + 1][w->dec_len - 1 - i];
    //     // w->dec_hi_float[i] = (((w->dec_len-1-i) % 2) ? -1 : 1)
    //     //     * bior_float[N - 1][0][i + n];

    //     decompose_lowpass[i] = kernel_a[i + n];
    //     reconstruct_lowpass[i] = kernel_b[filters_length - 1 - i];

    //     decompose_highpass[i] = kernel_b[filters_length - 1 - i];
    //     reconstruct_highpass[i] = kernel_a[i + n];
    // }

} // namespace wavelet

