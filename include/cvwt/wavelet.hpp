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

/**
 * @brief The degree of Wavelet orthogonality.
 */
enum class Orthogonality {
    ORTHOGONAL,
    BIORTHOGONAL,
    NEARLY_ORTHOGONAL,
    SEMIORTHOGONAL,
    NONE,
};


/**
 * @brief A %Wavelet.
 *
 * Predefined Wavelets
 * ===================
 * The following predefined wavelets can be constructed by the indicated
 * factory or by create() using one of the indicated names.
 *  - Haar
 *      - Factory: create_haar()
 *      - Names: haar
 *  - Daubechies
 *      - Factory: create_daubechies()
 *      - Names: db1, db2, ..., db38
 *  - Symlets
 *      - Factory: create_symlets()
 *      - Names: sym2, sym3, ..., sym20
 *  - Coiflets
 *      - Factory: create_coiflets()
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
 * Constructing Custom Wavelets By Name
 * ------------------------------------
 * Providing support for creating custom wavelets by name is straight forward -
 * simply implement a wavelet factory and register each possible parameter set
 * with register_factory().
 * For example, the predefined Daubechies wavelets are registered at startup
 * using code equivalent to the following.
 * @code{cpp}
 * for (int order = 1; order < 39; ++order)
 *     Wavelet::register_factory(create_daubechies, order);
 * @endcode
 * And creating a 4th order Daubechies wavelet by name is done with
 * @code{cpp}
 * Wavelet db4_wavelet = Wavelet::create("db4");
 * @endcode
 * Under the hood, std::bind() binds `order` to create_daubechies()
 * and uses the name() of the wavelet (e.g. "db1") as the factory name.
 *
 * Use this approach whenever the wavelet name() *uniquely* identifies an
 * *entire* set of factory parameters.
 * If, however, it is impossible or inpractical to enumerate all possible sets
 * of factory parameters, or the wavelet name() does not uniquely identify the
 * filter_bank(), use the version of
 * @ref register_factory(const std::string& name, Wavelet factory(BoundArgs..., CallArgs...), const BoundArgs&... args) "register_factory()"
 * that takes a factory name and registers a factory that accepts unbound
 * parameters at creation.
 *
 * @note Wavelet objects are designed to be allocated on the stack and should
 *       **not** be created with `new`.  They contain a single std::shared_ptr,
 *       making copying and moving an inexpensive operation.
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
     * @param[in] wavelet_vanishing_moments
     * @param[in] scaling_vanishing_moments
     */
    Wavelet(
        const FilterBank& filter_bank,
        Orthogonality orthogonality,
        Symmetry symmetry,
        const std::string& name = "",
        const std::string& family = "",
        int wavelet_vanishing_moments = -1,
        int scaling_vanishing_moments = -1
    );

    /**
     * @brief Construct a new Wavelet object
     *
     * @param filter_bank
     * @param orthogonality
     * @param name
     * @param family
     * @param wavelet_vanishing_moments
     * @param scaling_vanishing_moments
     */
    Wavelet(
        const FilterBank& filter_bank,
        Orthogonality orthogonality,
        const std::string& name = "",
        const std::string& family = "",
        int wavelet_vanishing_moments = -1,
        int scaling_vanishing_moments = -1
    );

    /**
     * @brief Construct a new Wavelet object
     *
     * @param filter_bank
     * @param symmetry
     * @param name
     * @param family
     * @param wavelet_vanishing_moments
     * @param scaling_vanishing_moments
     */
    Wavelet(
        const FilterBank& filter_bank,
        Symmetry symmetry,
        const std::string& name = "",
        const std::string& family = "",
        int wavelet_vanishing_moments = -1,
        int scaling_vanishing_moments = -1
    );

    /**
     * @brief Construct a new Wavelet object
     *
     * @param filter_bank
     * @param name
     * @param family
     * @param wavelet_vanishing_moments
     * @param scaling_vanishing_moments
     */
    Wavelet(
        const FilterBank& filter_bank,
        const std::string& name = "",
        const std::string& family = "",
        int wavelet_vanishing_moments = -1,
        int scaling_vanishing_moments = -1
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
    int wavelet_vanishing_moments() const { return _p->wavelet_vanishing_moments; }

    /**
     * @brief The number of vanishing moments of the scaling function.
     */
    int scaling_vanishing_moments() const { return _p->scaling_vanishing_moments; }

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
     * @brief The degree of wavelet symmetry.
     */
    Symmetry symmetry() const { return _p->symmetry; }

    /**
     * @brief Returns true if wavelet is symmetric.
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
     * @brief The length of the filter bank kernels.
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
     * @param[in] name The name of the wavelet factory.
     * @param[in] args The unbound arguments of the wavelet factory
     *                 registered with register_factory().
     */
    static Wavelet create(const std::string& name, auto&&... args)
    {
        auto factory = _wavelet_factories<decltype(args)...>.at(name);
        return factory(std::forward<decltype(args)>(args)...);
    }

    /**
     * @brief The set of names that are accepted by create().
     */
    static std::set<std::string> available_wavelets();

    /**
     * @brief Register a Wavelet factory for use by create().
     *
     * @note Use this overload when all sets of factory parameters can be
     *       enumerated and the wavelet name() uniquely determines the
     *       filter_bank().  Otherwise, use
     *       @ref register_factory(const std::string& name, Wavelet factory(BoundArgs..., CallArgs...), const BoundArgs&... args) "register_factory()"
     *       instead.
     *
     * The registered factory function is
     * <code>std::bind(@pref{factory}, @pref{args}...))</code> and the
     * registered factory name is the name() of the wavelet created by the bound
     * factory.
     *
     * Consider an example where the filter bank is determined by an `order`
     * that can be 2, 3, or 4.
     * @code{cpp}
     * // Define a factory function that returns a Wavelet whose name depends on order.
     * Wavelet create_my_custom_wavelet(int order)
     * {
     *     // Compute or lookup filter bank based on order.
     *     FilterBank filter_bank = ...;
     *     std::string name = "my" + std::to_string(order);
     *     std::string family = "My Wavelet";
     *
     *     return Wavelet(filter_bank, name, family);
     * }
     *
     * // Register a factory for each possible set of factory arguments.
     * Wavelet::register_factory(create_my_custom_wavelet, 2);
     * Wavelet::register_factory(create_my_custom_wavelet, 3);
     * Wavelet::register_factory(create_my_custom_wavelet, 4);
     *
     * // Create a 4th order wavelet.
     * Wavelet my4_wavelet = Wavelet::create("my4");
     * @endcode
     *
     * @param[in] factory A callable that creates a Wavelet object.
     * @param[in] args All of the arguments passed to @pref{factory}.
     * @throws cv::Exception If a factory with the same name is already registered.
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
     *
     * @overload
     *
     * @note Use this overload when it is impossible or inpractical to enumerate
     *       all sets of factory parameters, or when the wavelet name() does not
     *       uniquely determine the filter_bank().  Otherwise, use
     *       @ref register_factory(Wavelet factory(BoundArgs...), const BoundArgs&... args) "register_factory()"
     *       instead.
     *
     * The registered factory function is
     * <code>std::bind_front(@pref{factory}, @pref{args}...))</code>.
     *
     * Consider an example where the filter bank is determined by an `order`
     * that can be 2, 3, or 4 and a floating point `extra_param`.  Since
     * `extra_param` cannot be enumerated, the the factory name only depends on
     * the `order`.
     * @code{cpp}
     * // Define a wavelet factory.
     * Wavelet create_my_wavelet(int order, float extra_param)
     * {
     *     // Compute or lookup filter bank based on order and extra_param.
     *     FilterBank filter_bank = ...;
     *     std::string name = "my" + std::to_string(order) + "_" + std::to_string(extra_param);
     *     std::string family = "MyFamily"
     *
     *     return Wavelet(filter_bank, name, family);
     * }
     *
     * // Register factories for orders 2, 3, and 4.
     * Wavelet::register_factory("my2", create_my_wavelet, 2);
     * Wavelet::register_factory("my3", create_my_wavelet, 3);
     * Wavelet::register_factory("my4", create_my_wavelet, 4);
     *
     * // Create two 4th order wavelets.
     * // The first uses extra_param = 6.0 and the second uses extra_param = 4.0.
     * Wavelet my4_wavelet1 = Wavelet::create("my4", 6.0);
     * Wavelet my4_wavelet2 = Wavelet::create("my4", 4.0);
     * @endcode
     * Note that in this case the wavelet name() and the factory name are different.
     *
     * @param[in] name The name of the wavelet factory.
     * @param[in] factory A callable that creates a Wavelet object.
     * @param[in] args The front arguments passed to @pref{factory}.
     * @throws cv::Exception If a factory with the same name is already registered.
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
    static void throw_if_already_registered(const std::string& name) CVWT_WAVELET_NOEXCEPT
    {
    #if CVWT_WAVELET_EXCEPTIONS_ENABLED
        if (_wavelet_factories<CallArgs...>.contains(name)) {
            throw_bad_arg(
                "A wavelet factory has already been registered to `", name, "`."
            );
        }
    #endif
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
        int wavelet_vanishing_moments = -1;
        int scaling_vanishing_moments = -1;
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
 * @brief Creates a Haar Wavelet.
 */
Wavelet create_haar();

/**
 * @brief Creates a Daubechies Wavelet.
 *
 * @param[in] order The order of the wavelet.  Must be 2 <= order <= 38.
 * @throws cv::Exception If @pref{order} is invalid.
 */
Wavelet create_daubechies(int order);

/**
 * @brief Creates a Symlets Wavelet.
 *
 * @param[in] order The order of the wavelet.  Must be 2 <= order <= 20.
 * @throws cv::Exception If @pref{order} is invalid.
 */
Wavelet create_symlets(int order);

/**
 * @brief Creates a Coiflets Wavelet.
 *
 * @param[in] order The order of the wavelet.  Must be 1 <= order <= 17.
 * @throws cv::Exception If @pref{order} is invalid.
 */
Wavelet create_coiflets(int order);

/**
 * @brief Creates a Biorthogonal Wavelet.
 *
 * @param[in] wavelet_vanishing_moments The number of vanishing moments of the wavelet function.
 * @param[in] scaling_vanishing_moments The number of vanishing moments of the scaling function.
 * @throws cv::Exception If @pref{wavelet_vanishing_moments} and
 *                       @pref{scaling_vanishing_moments} are an invalid pair.
 */
Wavelet create_biorthogonal(int wavelet_vanishing_moments, int scaling_vanishing_moments);

/**
 * @brief Creates a Reverse Biorthogonal Wavelet.
 *
 * @param[in] wavelet_vanishing_moments The number of vanishing moments of the wavelet function.
 * @param[in] scaling_vanishing_moments The number of vanishing moments of the scaling function.
 * @throws cv::Exception If @pref{wavelet_vanishing_moments} and
 *                       @pref{scaling_vanishing_moments} are an invalid pair.
 */
Wavelet create_reverse_biorthogonal(int wavelet_vanishing_moments, int scaling_vanishing_moments);
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
        int wavelet_vanishing_moments,
        int scaling_vanishing_moments
    );

    template <typename V>
    void throw_if_invalid_wavelet_name(
        const std::string& name,
        const std::string& family,
        const std::map<std::string, V>& filter_coeffs,
        const std::string& name_prefix = ""
    ) CVWT_WAVELET_NOEXCEPT;

} // namespace internal
} // namespace cvwt

#endif  // CVWT_WAVELET_HPP

