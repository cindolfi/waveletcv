#ifndef CVWT_WAVELET_HPP
#define CVWT_WAVELET_HPP

#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <map>
#include "cvwt/filterbank.hpp"

namespace cvwt
{
enum class Symmetry {
    SYMMETRIC,
    NEAR_SYMMETRIC,
    ASYMMETRIC,
};

class Wavelet
{
protected:
    struct WaveletImpl
    {
        int vanishing_moments_psi = 0;
        int vanishing_moments_phi = 0;
        bool orthogonal = false;
        bool biorthogonal = false;
        Symmetry symmetry = Symmetry::ASYMMETRIC;
        std::string family = "";
        std::string name = "";
        FilterBank filter_bank;
    };

public:
    Wavelet();
    Wavelet(
        int vanishing_moments_psi,
        int vanishing_moments_phi,
        bool orthogonal,
        bool biorthogonal,
        Symmetry symmetry,
        const std::string& family,
        const std::string& name,
        const FilterBank& filter_bank
    );

    int vanishing_moments_psi() const { return _p->vanishing_moments_psi; }
    int vanishing_moments_phi() const { return _p->vanishing_moments_phi; }
    bool orthogonal() const { return _p->orthogonal; }
    bool biorthogonal() const { return _p->biorthogonal; }
    Symmetry symmetry() const { return _p->symmetry; }
    std::string family() const { return _p->family; }
    std::string name() const { return _p->name; }
    FilterBank filter_bank() const { return _p->filter_bank; }
    int filter_length() const { return _p->filter_bank.filter_length(); }
    bool is_valid() const { return _p->filter_bank.filter_length() > 0; }

    static Wavelet create(const std::string& name);
    static std::vector<std::string> registered_wavelets();
    template<class... Args>
    static void register_factory(
        const std::string& name,
        Wavelet factory(Args...),
        const Args&... args
    )
    {
        _wavelet_factories[name] = std::bind(factory, args...);
    }

    bool operator==(const Wavelet& other) const;

    friend std::ostream& operator<<(std::ostream& stream, const Wavelet& wavelet);
private:
    std::shared_ptr<WaveletImpl> _p;
    static std::map<std::string, std::function<Wavelet()>> _wavelet_factories;
};

std::ostream& operator<<(std::ostream& stream, const Wavelet& wavelet);

/**
 * Factories
*/
Wavelet create_daubechies(int order);
Wavelet create_haar();
Wavelet create_symlets(int order);
Wavelet create_coiflets(int order);
Wavelet create_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi);
Wavelet create_reverse_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi);

namespace internal
{
    std::string get_orthogonal_name(const std::string& prefix, int order);
    std::string get_biorthogonal_name(
        const std::string& prefix,
        int vanishing_moments_psi,
        int vanishing_moments_phi
    );
    #if CVWT_ARGUMENT_CHECKING_ENABLED
    template <typename V>
    void throw_if_invalid_wavelet_name(
        const std::string& name,
        const std::string& family,
        const std::map<std::string, V>& filter_coeffs,
        const std::string& name_prefix = ""
    );
    #else
    template <typename V>
    void throw_if_invalid_wavelet_name(
        const std::string& name,
        const std::string& family,
        const std::map<std::string, V>& filter_coeffs,
        const std::string& name_prefix = ""
    ) noexcept
    {}
    #endif  // CVWT_ARGUMENT_CHECKING_ENABLED
} // namespace internal
} // namespace cvwt

#endif  // CVWT_WAVELET_HPP

