#include "cvwt/wavelet.hpp"
#include "cvwt/daubechies.hpp"
#include "cvwt/symlets.hpp"
#include "cvwt/coiflets.hpp"
#include "cvwt/biorthogonal.hpp"
#include "cvwt/utils.hpp"
#include "cvwt/exception.hpp"
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <functional>
#include <sstream>

namespace cvwt
{
Wavelet::Wavelet() : _p(std::make_shared<WaveletImpl>())
{
}

Wavelet::Wavelet(
    const FilterBank& filter_bank,
    Orthogonality orthogonality,
    Symmetry symmetry,
    const std::string& name,
    const std::string& family,
    int vanishing_moments_psi,
    int vanishing_moments_phi
) :
    _p(
        std::make_shared<WaveletImpl>(
            filter_bank,
            orthogonality,
            symmetry,
            family,
            name,
            vanishing_moments_psi,
            vanishing_moments_phi
        )
    )
{
}

Wavelet::Wavelet(
    const FilterBank& filter_bank,
    Orthogonality orthogonality,
    const std::string& name,
    const std::string& family,
    int vanishing_moments_psi,
    int vanishing_moments_phi
) :
    _p(
        std::make_shared<WaveletImpl>(
            filter_bank,
            orthogonality,
            infer_symmetry(filter_bank),
            family,
            name,
            vanishing_moments_psi,
            vanishing_moments_phi
        )
    )
{
}

Wavelet::Wavelet(
    const FilterBank& filter_bank,
    Symmetry symmetry,
    const std::string& name,
    const std::string& family,
    int vanishing_moments_psi,
    int vanishing_moments_phi
) :
    _p(
        std::make_shared<WaveletImpl>(
            filter_bank,
            infer_orthogonality(filter_bank),
            symmetry,
            name,
            family,
            vanishing_moments_psi,
            vanishing_moments_phi
        )
    )
{
}

Wavelet::Wavelet(
    const FilterBank& filter_bank,
    const std::string& name,
    const std::string& family,
    int vanishing_moments_psi,
    int vanishing_moments_phi
) :
    _p(
        std::make_shared<WaveletImpl>(
            filter_bank,
            infer_orthogonality(filter_bank),
            infer_symmetry(filter_bank),
            family,
            name,
            vanishing_moments_psi,
            vanishing_moments_phi
        )
    )
{
}


Orthogonality Wavelet::infer_orthogonality(const FilterBank& filter_bank) const
{
    Orthogonality orthogonality;
    if (filter_bank.is_biorthogonal())
        orthogonality = Orthogonality::BIORTHOGONAL;
    else if (filter_bank.is_orthogonal())
        orthogonality = Orthogonality::ORTHOGONAL;
    else
        orthogonality = Orthogonality::NONE;

    return orthogonality;
}

Symmetry Wavelet::infer_symmetry(const FilterBank& filter_bank) const
{
    Symmetry symmetry;
    if (filter_bank.is_symmetric())
        symmetry = Symmetry::SYMMETRIC;
    else
        symmetry = Symmetry::ASYMMETRIC;

    return symmetry;
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

std::set<std::string> Wavelet::available_wavelets()
{
    return _available_wavelets;
}

std::set<std::string> Wavelet::_available_wavelets{
    "haar",
    //  daubechies
    "db1",
    "db2",
    "db3",
    "db4",
    "db5",
    "db6",
    "db7",
    "db8",
    "db9",
    "db10",
    "db11",
    "db12",
    "db13",
    "db14",
    "db15",
    "db16",
    "db17",
    "db18",
    "db19",
    "db20",
    "db21",
    "db22",
    "db23",
    "db24",
    "db25",
    "db26",
    "db27",
    "db28",
    "db29",
    "db30",
    "db31",
    "db32",
    "db33",
    "db34",
    "db35",
    "db36",
    "db37",
    "db38",
    //  symlets
    "sym2",
    "sym3",
    "sym4",
    "sym5",
    "sym6",
    "sym7",
    "sym8",
    "sym9",
    "sym10",
    "sym11",
    "sym12",
    "sym13",
    "sym14",
    "sym15",
    "sym16",
    "sym17",
    "sym18",
    "sym19",
    "sym20",
    //  coiflets
    "coif1",
    "coif2",
    "coif3",
    "coif4",
    "coif5",
    "coif6",
    "coif7",
    "coif8",
    "coif9",
    "coif10",
    "coif11",
    "coif12",
    "coif13",
    "coif14",
    "coif15",
    "coif16",
    "coif17",
    //  biorthongonal
    "bior1.1",
    "bior1.3",
    "bior1.5",
    "bior2.2",
    "bior2.4",
    "bior2.6",
    "bior2.8",
    "bior3.1",
    "bior3.3",
    "bior3.5",
    "bior3.7",
    "bior3.9",
    "bior4.4",
    "bior5.5",
    "bior6.8",
    //  reverse biorthongonal
    "rbior1.1",
    "rbior1.3",
    "rbior1.5",
    "rbior2.2",
    "rbior2.4",
    "rbior2.6",
    "rbior2.8",
    "rbior3.1",
    "rbior3.3",
    "rbior3.5",
    "rbior3.7",
    "rbior3.9",
    "rbior4.4",
    "rbior5.5",
    "rbior6.8",
};


//  ----------------------------------------------------------------------------
//  Wavelet Factories
//  ----------------------------------------------------------------------------
Wavelet create_haar()
{
    return Wavelet(
        FilterBank::create_orthogonal_filter_bank(
            cv::Mat(internal::DAUBECHIES_FILTER_COEFFS["db1"])
        ),
        Orthogonality::ORTHOGONAL,
        Symmetry::ASYMMETRIC,
        "haar",
        "Haar",
        1, // vanishing_moments_psi
        0  // vanishing_moments_phi
    );
}

Wavelet create_daubechies(int order)
{
    auto name = internal::make_orthogonal_name(
        internal::DAUBECHIES_NAME,
        order
    );
    internal::throw_if_invalid_wavelet_name(
        name,
        internal::DAUBECHIES_FAMILY,
        internal::DAUBECHIES_FILTER_COEFFS
    );

    return Wavelet(
        FilterBank::create_orthogonal_filter_bank(
            cv::Mat(internal::DAUBECHIES_FILTER_COEFFS[name])
        ),
        Orthogonality::ORTHOGONAL,
        // (order > 1) ? Symmetry::ASYMMETRIC : Symmetry::SYMMETRIC,
        Symmetry::ASYMMETRIC,
        name,
        internal::DAUBECHIES_FAMILY,
        order, // vanishing_moments_psi
        0      // vanishing_moments_phi
    );
}

Wavelet create_symlets(int order)
{
    auto name = internal::make_orthogonal_name(
        internal::SYMLETS_NAME,
        order
    );
    internal::throw_if_invalid_wavelet_name(
        name,
        internal::SYMLETS_FAMILY,
        internal::SYMLETS_FILTER_COEFFS
    );

    return Wavelet(
        FilterBank::create_orthogonal_filter_bank(
            cv::Mat(internal::SYMLETS_FILTER_COEFFS[name])
        ),
        Orthogonality::ORTHOGONAL,
        Symmetry::NEARLY_SYMMETRIC,
        name,
        internal::SYMLETS_FAMILY,
        order, // vanishing_moments_psi
        0      // vanishing_moments_phi
    );
}

Wavelet create_coiflets(int order)
{
    auto name = internal::make_orthogonal_name(
        internal::COIFLETS_NAME,
        order
    );
    internal::throw_if_invalid_wavelet_name(
        name,
        internal::COIFLETS_FAMILY,
        internal::COIFLETS_FILTER_COEFFS
    );

    return Wavelet(
        FilterBank::create_orthogonal_filter_bank(
            cv::Mat(internal::COIFLETS_FILTER_COEFFS[name])
        ),
        Orthogonality::ORTHOGONAL,
        Symmetry::NEARLY_SYMMETRIC,
        name,
        internal::COIFLETS_FAMILY,
        2 * order,     // vanishing_moments_psi
        2 * order - 1  // vanishing_moments_phi
    );
}

Wavelet create_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi)
{
    auto name = internal::make_biorthogonal_name(
        internal::BIORTHOGONAL_NAME,
        vanishing_moments_psi,
        vanishing_moments_phi
    );
    internal::throw_if_invalid_wavelet_name(
        name,
        internal::BIORTHOGONAL_FAMILY,
        internal::BIORTHOGONAL_FILTER_COEFFS
    );

    return Wavelet(
        FilterBank::create_biorthogonal_filter_bank(
            cv::Mat(internal::BIORTHOGONAL_FILTER_COEFFS.at(name).first),
            cv::Mat(internal::BIORTHOGONAL_FILTER_COEFFS.at(name).second)
        ),
        Orthogonality::BIORTHOGONAL,
        Symmetry::SYMMETRIC,
        name,
        internal::BIORTHOGONAL_FAMILY,
        vanishing_moments_psi,
        vanishing_moments_phi
    );
}

Wavelet create_reverse_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi)
{
    auto name = internal::make_biorthogonal_name(
        internal::BIORTHOGONAL_NAME,
        vanishing_moments_psi,
        vanishing_moments_phi
    );
    auto family = "Reverse " + internal::BIORTHOGONAL_FAMILY;
    internal::throw_if_invalid_wavelet_name(
        name,
        family,
        internal::BIORTHOGONAL_FILTER_COEFFS,
        "r"
    );
    auto biorthogonal_filter_bank = FilterBank::create_biorthogonal_filter_bank(
        cv::Mat(internal::BIORTHOGONAL_FILTER_COEFFS.at(name).first),
        cv::Mat(internal::BIORTHOGONAL_FILTER_COEFFS.at(name).second)
    );

    return Wavelet(
        biorthogonal_filter_bank.reverse(),
        Orthogonality::BIORTHOGONAL,
        Symmetry::SYMMETRIC,
        "r" + name,
        family,
        vanishing_moments_psi,
        vanishing_moments_phi
    );
}

namespace internal
{
std::string make_orthogonal_name(const std::string& prefix, int order)
{
    std::stringstream stream;
    stream << prefix << order;
    return stream.str();
    // return prefix + std::to_string(order);
}

std::string make_biorthogonal_name(
    const std::string& prefix,
    int vanishing_moments_psi,
    int vanishing_moments_phi
)
{
    std::stringstream stream;
    stream << prefix << vanishing_moments_psi << "." << vanishing_moments_phi;
    return stream.str();
    // return prefix + std::to_string(vanishing_moments_psi) + "." + std::to_string(vanishing_moments_phi);
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

        throw_bad_arg(
            "Invalid ", family, " wavelet order. ",
            "Must be one of: ", available_names.str(), ". ",
            "Got ", name_prefix + name, "."
        );
    }
}
#endif  // CVWT_ARGUMENT_CHECKING_ENABLED
} // namespace internal
} // namespace cvwt

