#include "cvwt/wavelet.hpp"
#include "cvwt/daubechies.hpp"
#include "cvwt/symlets.hpp"
#include "cvwt/coiflets.hpp"
#include "cvwt/biorthogonal.hpp"
#include "cvwt/utils.hpp"
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <functional>

namespace cvwt
{
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


//  ----------------------------------------------------------------------------
//  Wavelet Factories
//  ----------------------------------------------------------------------------
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
    auto name = internal::get_orthogonal_name(DAUBECHIES_NAME, order);
    internal::throw_if_invalid_wavelet_name(name, DAUBECHIES_FAMILY, DAUBECHIES_FILTER_COEFFS);
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
    auto name = internal::get_orthogonal_name(SYMLETS_NAME, order);
    internal::throw_if_invalid_wavelet_name(name, SYMLETS_FAMILY, SYMLETS_FILTER_COEFFS);
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
    auto name = internal::get_orthogonal_name(COIFLETS_NAME, order);
    internal::throw_if_invalid_wavelet_name(name, COIFLETS_FAMILY, COIFLETS_FILTER_COEFFS);
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
        BIORTHOGONAL_NAME,
        vanishing_moments_psi,
        vanishing_moments_phi
    );
    internal::throw_if_invalid_wavelet_name(name, BIORTHOGONAL_FAMILY, BIORTHOGONAL_FILTER_COEFFS);
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
        BIORTHOGONAL_NAME,
        vanishing_moments_psi,
        vanishing_moments_phi
    );
    auto family = "Reverse " + BIORTHOGONAL_FAMILY;
    internal::throw_if_invalid_wavelet_name(name, family, BIORTHOGONAL_FILTER_COEFFS, "r");
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

#if CVWT_ARGUMENT_CHECKING_ENABLED
template <typename V>
void throw_if_invalid_wavelet_name(
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

        internal::throw_bad_arg(
            "Invalid ", family, " wavelet order. ",
            "Must be one of: ", available_names.str(), ". ",
            "Got ", name_prefix + name, "."
        );
    }
}
#endif  // CVWT_ARGUMENT_CHECKING_ENABLED
} // namespace internal
} // namespace cvwt

