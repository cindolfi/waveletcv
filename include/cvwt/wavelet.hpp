#ifndef CVWT_WAVELET_HPP
#define CVWT_WAVELET_HPP

#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <array>
#include <map>
#include "cvwt/filterbank.hpp"

namespace cvwt
{
/**
 * @brief The degree of Wavelet symmetry.
 */
enum class Symmetry {
    SYMMETRIC,
    NEARLY_SYMMETRIC,
    ASYMMETRIC,
};

enum class Orthogonality {
    ORTHOGONAL,
    BIORTHOGONAL,
    NEARLY_ORTHOGONAL,
    SEMIORTHOGONAL,
    NONE,
};

/**
 * @brief A Wavelet.
 *
 * Predefined Wavelets
 * ===================
 * The following predefined wavelets can be constructed using the indicated
 * factory or by Wavelet::create() using one of the indicated names.
 *  - Haar
 *      - Factory: create_haar()
 *      - Names: haar
 *  - Daubechies
 *      - Factory: create_daubechies()
 *      - Names: db1, db2, ..., db38
 *  - Symlets
 *      - Factory: create_symlet()
 *      - Names: sym2, sym3, ..., sym20
 *  - Coiflets
 *      - Factory: create_coiflet()
 *      - Names: coif1, coif2, ..., coif17
 *  - Biorthogonal
 *      - Factory: create_biorthogonal()
 *      - Names: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior.2,8,
 *               bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5,
 *               and bior6.8
 *  - Reverse Biorthognal
 *      - Factory: create_reverse_biorthogonal()
 *      - Names: rbior1.1, rbior1.3, rbior1.5, rbior2.2, rbior2.4, rbior2.6,
 *               rbior.2,8, rbior3.1, rbior3.3, rbior3.5, rbior3.7, rbior3.9,
 *               rbior4.4, rbior5.5, and rbior6.8
 *
 * Custom Wavelets
 * ===============
 * Custom wavelets are constructed by providing a defining FilterBank.
 * Additional properties such as orthogonality, symmetry, and the number
 * vanishing moments can be optionally given.
 *
 * When the orthogonality property is not explicitly provided it is inferred
 * using FilterBank::is_orthogonal() and FilterBank::is_biorthogonal().  Only
 * Orthogonality::ORTHOGONAL and Orthogonality::BIORTHOGONAL can be inferred.
 * Orthogonality::NEARLY_ORTHOGONAL and Orthogonality::SEMIORTHOGONAL cannot be
 * inferred and must be set explicitly.
 *
 * Likewise, when the symmetry property is not explicity provided it is inferred
 * using FilterBank::is_symmetric().  Only Symmetry::SYMMETRIC can be inferred.
 * Symmetry::NEARLY_SYMMETRIC cannot be inferred and must be set explicity.
 *
 * Support For Constructing Custom Wavelets By Name
 * ------------------------------------------------
 * Providing support for creating custom wavelets by name is straight forward -
 * simply implement a wavelet factory and register each possible parameter set
 * with register_factory().
 * For example, the predefined Daubechies wavelets are registered at startup
 * using code equivalent to the following.
 * @code{cpp}
 * for (int order = 1; order < 39; ++order)
 *     Wavelet::register_factory(create_daubechies, order);
 * @endcode
 * Under the hood, std::bind_front() binds `order` to create_daubechies().
 * The name() of the Wavelet returned by the bound factory is then mapped to the
 * bound factory for use by Wavelet::create().
 * @code{cpp}
 * // Create a 4th order Daubechies wavelet
 * Wavelet db4_wavelet = Wavelet::create("db4");
 * @endcode
 * Use this approach whenever the wavelet name() **uniquely** identifies an
 * **entire** set of factory parameters.
 *
 * Use the register_factory() overload that takes a factory %name whenever the
 * wavelet name() does not uniquely determine the filter_bank() or the factory
 * depends on unbound parameters provided at creation (i.e. the factory name is
 * associated to some, but not all, factory parameters).
 *
 * Consider a case where the filter bank is determined by an `order` and some
 * `extra_param`, but the factory %name only depends on the `order`.
 * @code{cpp}
 * FilterBank create_my_wavelet_filter_bank(int order, float extra_param)
 * {
 *     return FilterBank(...);
 * }
 *
 * // Define a wavelet factory
 * Wavelet create_my_wavelet(int order, float extra_param)
 * {
 *     return Wavelet(
 *         create_my_wavelet_filter_bank(order, extra_param),
 *         "my" + std::to_string(order), // name
 *         "MyFamily" // family
 *     );
 * }
 *
 * // Register factories for orders 1, 2, 3, and 4
 * Wavelet::register_factory("my1", create_my_wavelet, 1);
 * Wavelet::register_factory("my2", create_my_wavelet, 2);
 * Wavelet::register_factory("my3", create_my_wavelet, 3);
 * Wavelet::register_factory("my4", create_my_wavelet, 4);
 *
 * // Create a 4th order MyFamily wavelet using extra_param = 6.0
 * Wavelet my4_wavelet = Wavelet::create("my4", 6.0);
 *
 * // Create a 2nd order MyFamily wavelet using extra_param = -4.0
 * Wavelet my2_wavelet = Wavelet::create("my2", -4.0);
 * @endcode
 * Note that in this case the name() of the wavelet returned by
 * Wavelet::create() may or may not be equal to the factory %name.
 */
class Wavelet
{
public:
    /**
     * @brief Construct an invalid Wavelet.
     */
    Wavelet();

    /**
     * @brief Construct a new Wavelet.
     *
     * @param[in] filter_bank
     * @param[in] orthogonality
     * @param[in] symmetry
     * @param[in] name
     * @param[in] family
     * @param[in] vanishing_moments_psi
     * @param[in] vanishing_moments_phi
     */
    Wavelet(
        const FilterBank& filter_bank,
        Orthogonality orthogonality,
        Symmetry symmetry,
        const std::string& name = "",
        const std::string& family = "",
        int vanishing_moments_psi = -1,
        int vanishing_moments_phi = -1
    );

    /**
     * @brief Construct a new Wavelet object
     *
     * @param filter_bank
     * @param orthogonality
     * @param name
     * @param family
     * @param vanishing_moments_psi
     * @param vanishing_moments_phi
     */
    Wavelet(
        const FilterBank& filter_bank,
        Orthogonality orthogonality,
        const std::string& name = "",
        const std::string& family = "",
        int vanishing_moments_psi = -1,
        int vanishing_moments_phi = -1
    );

    /**
     * @brief Construct a new Wavelet object
     *
     * @param filter_bank
     * @param symmetry
     * @param name
     * @param family
     * @param vanishing_moments_psi
     * @param vanishing_moments_phi
     */
    Wavelet(
        const FilterBank& filter_bank,
        Symmetry symmetry,
        const std::string& name = "",
        const std::string& family = "",
        int vanishing_moments_psi = -1,
        int vanishing_moments_phi = -1
    );

    /**
     * @brief Construct a new Wavelet object
     *
     * @param filter_bank
     * @param name
     * @param family
     * @param vanishing_moments_psi
     * @param vanishing_moments_phi
     */
    Wavelet(
        const FilterBank& filter_bank,
        const std::string& name = "",
        const std::string& family = "",
        int vanishing_moments_psi = -1,
        int vanishing_moments_phi = -1
    );

    /**
     * @brief Copy Constructor.
     */
    Wavelet(const Wavelet& other) = default;

    /**
     * @brief Move Constructor.
     */
    Wavelet(Wavelet&& other) = default;

    /**
     * @brief Copy Assignment.
     */
    Wavelet& operator=(const Wavelet& other) = default;

    /**
     * @brief Move Assignment.
     */
    Wavelet& operator=(Wavelet&& other) = default;

    /**
     * @brief The number of vanishing moments of the wavelet function.
     */
    int vanishing_moments_psi() const { return _p->vanishing_moments_psi; }

    /**
     * @brief The number of vanishing moments of the scaling function.
     */
    int vanishing_moments_phi() const { return _p->vanishing_moments_phi; }

    /**
     * @brief Returns true if the wavelet is orthogonal.
     */
    Orthogonality orthogonality() const { return _p->orthogonality; }

    /**
     * @brief Returns true if the wavelet is orthogonal.
     */
    bool is_orthogonal() const
    {
        return _p->orthogonality == Orthogonality::ORTHOGONAL
            || _p->filter_bank.is_orthogonal();
    }

    /**
     * @brief Returns true if the wavelet is biorthogonal.
     */
    bool is_biorthogonal() const
    {
        return _p->orthogonality == Orthogonality::BIORTHOGONAL
            || _p->orthogonality == Orthogonality::ORTHOGONAL
            || _p->filter_bank.is_biorthogonal();
    }

    /**
     * @brief Returns the degree of symmetry of the filter.
     */
    Symmetry symmetry() const { return _p->symmetry; }

    /**
     * @brief Returns the degree of symmetry of the filter.
     */
    bool is_symmetric() const { return _p->symmetry == Symmetry::SYMMETRIC; }

    /**
     * @brief The name of the wavelet family.
     */
    std::string family() const { return _p->family; }

    /**
     * @brief The name of the wavelet.
     */
    std::string name() const { return _p->name; }

    /**
     * @brief The filter bank that defines the wavelet.
     */
    FilterBank filter_bank() const { return _p->filter_bank; }

    /**
     * @brief The length of the wavelet's filter kernel.
     */
    int filter_length() const { return _p->filter_bank.filter_length(); }

    /**
     * @brief Returns true if the wavelet is valid.
     *
     * A valid wavelet has a nonempty filter_bank().
     * A default constructed wavelet is always invalid.
     */
    bool is_valid() const { return _p->filter_bank.filter_length() > 0; }

    /**
     * @brief Creates a Wavelet by name.
     *
     * @code{cpp}
     * Wavelet wavelet = Wavelet::create("db2");
     * @endcode
     *
     * Use available_wavelets() to get available wavelet names.
     * Use register_factory() to register a factory for custom wavelets.
     *
     * @param[in] name The name of the wavelet.
     * @param[in] args The unbound arguments of the wavelet factory
     *                 registered with register_factory().
     */
    // static Wavelet create(const std::string& name);
    static Wavelet create(const std::string& name, auto&&... args)
    {
        auto factory = _wavelet_factories<decltype(args)...>.at(name);
        return factory(std::forward<decltype(args)>(args)...);
        // return _wavelet_factories<decltype(args)...>.at(name)(std::forward<decltype(args)>(args)...);
    }

    /**
     * @brief Returns a collection of wavelet names that can be used with create().
     */
    static std::set<std::string> available_wavelets();

    /**
     * @brief Register a Wavelet factory for use by create().
     *
     * This function is used to support creating custom wavelets with create().
     * The given @pref{args} are bound to the @pref{factory} function
     * using `std::bind_front(factory, args...))`.  The name() of the wavelet
     * created by the factory is used as the %name of the factory.
     * The create() function looks up the bound factory function and uses it to
     * create a new Wavelet object.
     *
     * For example:
     * @code{cpp}
     * //  Returns a custom Wavelet object whose name depends on some_param.
     * Wavelet create_my_custom_wavelet(int some_param);
     *
     * //  Register a factory for each possible set of arguments to the wavelet
     * //  factory function.
     * Wavelet::register_factory(create_my_custom_wavelet, 2);
     * Wavelet::register_factory(create_my_custom_wavelet, 3);
     * Wavelet::register_factory(create_my_custom_wavelet, 4);
     * @endcode
     *
     * @param[in] factory A callable that creates a Wavelet object.
     * @param[in] args All of the arguments passed to @pref{factory}.
     */
    template <typename... BoundArgs>
    static void register_factory(
        Wavelet factory(BoundArgs...),
        const BoundArgs&... args
    )
    {
        auto bound_factory = std::bind_front(factory, args...);
        Wavelet wavelet = bound_factory();
        insert_factory(wavelet.name(), bound_factory);
    }

    /**
     * @brief Register a Wavelet factory for use by create().
     *
     * This function is used to support creating custom wavelets with create().
     * The given @pref{args} are bound to the @pref{factory} function
     * using `std::bind_front(factory, args...))`.
     *
     * For example:
     * @code{cpp}
     * // Returns a custom Wavelet object whose name depends on some_param
     * // but not on another_param.
     * Wavelet create_my_custom_wavelet(int some_param, float another_param);
     *
     * // Register a factory for each possible set of bound arguments to the
     * // wavelet factory function.
     * Wavelet::register_factory("my2", create_my_custom_wavelet, 2);
     * Wavelet::register_factory("my3", create_my_custom_wavelet, 3);
     * Wavelet::register_factory("my4", create_my_custom_wavelet, 4);
     *
     * // Provide remaining unbound parameter at creation.
     * Wavelet my2_wavelet = Wavelet::create("my2", 4.0);
     * @endcode
     *
     * @param[in] name The name of the wavelet factory.
     * @param[in] factory A callable that creates a Wavelet object.
     * @param[in] args The front arguments passed to @pref{factory}.
     */
    template <typename... BoundArgs, typename... CallArgs>
    static void register_factory(
        const std::string& name,
        Wavelet factory(BoundArgs..., CallArgs...),
        const BoundArgs&... args
    )
    {
        insert_factory(name, std::bind_front(factory, args...));
    }

    /**
     * @brief Wavelets are equal if they have the same name and have equal filter banks.
     */
    bool operator==(const Wavelet& other) const;

    /**
     * @private
     */
    friend std::ostream& operator<<(std::ostream& stream, const Wavelet& wavelet);
private:
    Orthogonality infer_orthogonality(const FilterBank& filter_bank) const;
    Symmetry infer_symmetry(const FilterBank& filter_bank) const;

    template <typename... CallArgs>
    static void throw_if_already_registered(const std::string& name)
    {
        if (_wavelet_factories<CallArgs...>.contains(name)) {
            throw_bad_arg(
                "A wavelet factory has already been registered to `", name, "`."
            );
        }
    }

    template <typename... CallArgs>
    static void insert_factory(
        const std::string& name,
        Wavelet factory(CallArgs...)
    )
    {
        throw_if_already_registered<CallArgs...>(name);
        _available_wavelets.insert(name);
        _wavelet_factories<CallArgs...>[name] = factory;
    }

    struct WaveletImpl
    {
        FilterBank filter_bank;
        Orthogonality orthogonality = Orthogonality::NONE;
        Symmetry symmetry = Symmetry::ASYMMETRIC;
        std::string family = "";
        std::string name = "";
        int vanishing_moments_psi = -1;
        int vanishing_moments_phi = -1;
    };

    std::shared_ptr<WaveletImpl> _p;

    template <typename... CallArgs>
    static std::map<std::string, std::function<Wavelet(CallArgs...)>> _wavelet_factories;
    static std::set<std::string> _available_wavelets;
};

/**
 * @brief Writes a string representation of a Wavelet to the output stream.
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
 * @brief Create a Haar Wavelet.
 */
Wavelet create_haar();

/**
 * @brief Create a Daubechies Wavelet.
 *
 * @param[in] order The order of the wavelet.  Must be 2 <= order <= 38.
 */
Wavelet create_daubechies(int order);

/**
 * @brief Create a Symlets Wavelet.
 *
 * @param[in] order The order of the wavelet.  Must be 2 <= order <= 20.
 */
Wavelet create_symlets(int order);

/**
 * @brief Create a Coiflets Wavelet.
 *
 * @param[in] order The order of the wavelet.  Must be 1 <= order <= 17.
 */
Wavelet create_coiflets(int order);

/**
 * @brief Create a Biorthogonal Wavelet.
 *
 * @param[in] vanishing_moments_psi The number of vanishing moments of the wavelet function.
 * @param[in] vanishing_moments_phi The number of vanishing moments of the scaling function.
 */
Wavelet create_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi);

/**
 * @brief Create a Reverse Biorthogonal Wavelet.
 *
 * @param[in] vanishing_moments_psi The number of vanishing moments of the wavelet function.
 * @param[in] vanishing_moments_phi The number of vanishing moments of the scaling function.
 */
Wavelet create_reverse_biorthogonal(int vanishing_moments_psi, int vanishing_moments_phi);
/** @}*/


template <typename... CallArgs>
std::map<std::string, std::function<Wavelet(CallArgs...)>> Wavelet::_wavelet_factories{
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

namespace internal
{
    std::string make_orthogonal_name(const std::string& prefix, int order);
    std::string make_biorthogonal_name(
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

