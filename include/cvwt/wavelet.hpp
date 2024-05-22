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
/**
 * @brief
 *
 */
enum class Symmetry {
    SYMMETRIC,
    NEAR_SYMMETRIC,
    ASYMMETRIC,
};

/**
 * @brief A named wavelet
 *
 */
class Wavelet
{
public:
    /**
     * @brief Construct a new Wavelet object
     */
    Wavelet();
    /**
     * @brief Construct a new Wavelet object
     *
     * @param vanishing_moments_psi
     * @param vanishing_moments_phi
     * @param orthogonal
     * @param biorthogonal
     * @param symmetry
     * @param family
     * @param name
     * @param filter_bank
     */
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
    /**
     * @brief Copy Constructor
     */
    Wavelet(const Wavelet& other) = default;
    /**
     * @brief Move Constructor
     */
    Wavelet(Wavelet&& other) = default;

    /**
     * @brief Copy Assignment
     */
    Wavelet& operator=(const Wavelet& other) = default;
    /**
     * @brief Move Assignment
     */
    Wavelet& operator=(Wavelet&& other) = default;

    /**
     * @brief The number of vanishing moments of the wavelet function
     */
    int vanishing_moments_psi() const { return _p->vanishing_moments_psi; }

    /**
     * @brief The number of vanishing moments of the scaling function
     */
    int vanishing_moments_phi() const { return _p->vanishing_moments_phi; }

    /**
     * @brief Returns true if the wavelet is orthogonal
     */
    bool orthogonal() const { return _p->orthogonal; }

    /**
     * @brief Returns true if the wavelet is biorthogonal
     */
    bool biorthogonal() const { return _p->biorthogonal; }

    /**
     * @brief Returns how symmetric the wavelet is
     */
    Symmetry symmetry() const { return _p->symmetry; }

    /**
     * @brief The family name the wavelet belongs to
     */
    std::string family() const { return _p->family; }

    /**
     * @brief The name of the wavelet
     */
    std::string name() const { return _p->name; }

    /**
     * @brief The wavelet filter bank
     */
    FilterBank filter_bank() const { return _p->filter_bank; }

    /**
     * @brief The length of the wavelet's filter kernel
     */
    int filter_length() const { return _p->filter_bank.filter_length(); }

    /**
     * @brief Returns true if the wavelet is valid
     *
     * A valid wavelet has a non-empty Wavelet::name() and a non-empty
     * Wavelet::filter_bank(). A default constructed wavelet is always invalid.
     *
     */
    bool is_valid() const { return _p->filter_bank.filter_length() > 0; }

    /**
     * @brief Creates a Wavelet object by name
     *
     * ```cpp
     * Wavelet wavelet = Wavelet::create("db2");
     * ```
     * Use Wavelet::available_wavelets() to get available wavelet names.
     * Use Wavelet::register_factory() to register a factory for custom wavelets.
     *
     * @param name
     */
    static Wavelet create(const std::string& name);

    /**
     * @brief Returns a collection of wavelet names that can be used with Wavelet::create()
     *
     */
    static std::vector<std::string> available_wavelets();

    /**
     * @brief Register a Wavelet factory used by Wavelet::create()
     *
     * This function is used to add support for creating custom wavelets with
     * Wavelet::create().  The given arguments are bound to the factory function
     * using std::bind(factory, args...)).  The Wavelet::name() of the wavelet
     * created by the factory is mapped to the bound factory function.
     * Wavelet::create() takes a name and returns a wavelet object created by
     * the mapped factory.
     *
     * For example:
     * ```cpp
     * //  Returns a custom Wavelet object whose name depends on some_param
     * Wavelet create_my_custom_wavelet(int some_param);
     *
     * //  Register a factory for each possible set of arguments to the wavelet
     * //  factory function.
     * Wavelet::register_factory(create_my_custom_wavelet, 2);
     * Wavelet::register_factory(create_my_custom_wavelet, 3);
     * Wavelet::register_factory(create_my_custom_wavelet, 4);
     * ```
     *
     * @param factory A callable that creates and returns a Wavelet object.
     * @param args The arguments passed to wavelet factory.
     */
    template <typename... Args>
    static void register_factory(
        Wavelet factory(Args...),
        const Args&... args
    )
    {
        auto bound_factory = std::bind(factory, args...);
        Wavelet wavelet = bound_factory();
        _wavelet_factories[wavelet.name()] = bound_factory;
    }

    /**
     * @brief Two Wavelets are equal if they have the same name and have equal filter banks
     */
    bool operator==(const Wavelet& other) const;

    friend std::ostream& operator<<(std::ostream& stream, const Wavelet& wavelet);
private:
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

    std::shared_ptr<WaveletImpl> _p;
    static std::map<std::string, std::function<Wavelet()>> _wavelet_factories;
};

/**
 * @brief Writes a string representation a Wavelet object to the output stream
 *
 * @param stream
 * @param wavelet
 */
std::ostream& operator<<(std::ostream& stream, const Wavelet& wavelet);


//  ----------------------------------------------------------------------------
//  Factories
//  ----------------------------------------------------------------------------
/**
 * @name Wavelet Factories
 * @{
 */
/**
 * @brief Create a Haar Wavelet object
 */
Wavelet create_haar();

/**
 * @brief Create a Daubechies Wavelet object
 *
 * @param order The order of the wavelet.  Must be 2 <= order <= 20.
 */
Wavelet create_daubechies(int order);

/**
 * @brief Create a Symlets Wavelet object
 *
 * @param order The order of the wavelet.  Must be 2 <= order <= 20.
 */
Wavelet create_symlets(int order);

/**
 * @brief Create a Coiflets Wavelet object
 *
 * @param order The order of the wavelet.  Must be 1 <= order <= 17.
 */
Wavelet create_coiflets(int order);

/**
 * @brief Create a Biorthogonal Wavelet object
 *
 * @param vanishing_moments_psi The number of vanishing moments of the wavelet function.
 * @param vanishing_moments_phi The number of vanishing moments of the scaling function.
 */
Wavelet create_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi);

/**
 * @brief Create a Reverse Biorthogonal Wavelet object
 *
 * @param vanishing_moments_psi The number of vanishing moments of the wavelet function.
 * @param vanishing_moments_phi The number of vanishing moments of the scaling function.
 */
Wavelet create_reverse_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi);
/** @}*/

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

