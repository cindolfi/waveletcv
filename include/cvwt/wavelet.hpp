#ifndef CVWT_WAVELET_HPP
#define CVWT_WAVELET_HPP

#include <string>
#include <memory>
#include <set>
#include <map>
#include <source_location>
#include <opencv2/core.hpp>
#include "cvwt/filterbank.hpp"
#include "cvwt/exception.hpp"

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
 * @note Wavelet objects are designed to be allocated on the stack and passed
 *       by reference.  They manage their memory internally using a single
 *       std::shared_ptr. Allocation using new incurs an two heap allocations.
 *       Passing by value incurs the cost of copying the shared pointer.
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
     * @brief Copies the wavelet and converts the filter bank data type.
     *
     * @param[in] type The filter bank data type.
     */
    [[nodiscard]]
    Wavelet as_type(int type) const;

    /**
     * @brief The filter bank data type.
     *
     * @param[in] type The filter bank data type.
     */
    int type() const { return _p->filter_bank.type(); }

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
     *       uniquely determine the filter_bank().  Otherwise, use this
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
    static void throw_if_already_registered(
        const std::string& name,
        const std::source_location& location = std::source_location::current()
    ) CVWT_WAVELET_NOEXCEPT
    {
    #if CVWT_WAVELET_EXCEPTIONS_ENABLED
        if (_wavelet_factories<CallArgs...>.contains(name)) {
            throw_bad_arg(
                "A wavelet factory has already been registered to `", name, "`.",
                location
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
        std::string name = "";
        std::string family = "";
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
 *
 * @param[in] type The wavelet data type.
 */
Wavelet create_haar(int type = CV_64FC1);

/**
 * @brief Creates a Daubechies Wavelet.
 *
 * @param[in] order The order of the wavelet.
 *                  Must be 2 \f$\le\f$ @pref{order} \f$\le\f$ 38.
 * @param[in] type The wavelet data type.
 * @throws cv::Exception If @pref{order} is invalid.
 */
Wavelet create_daubechies(int order, int type = CV_64FC1);

/**
 * @brief Creates a Symlets Wavelet.
 *
 * @param[in] order The order of the wavelet.
 *                  Must be 2 \f$\le\f$ @pref{order} \f$\le\f$ 20.
 * @param[in] type The wavelet data type.
 * @throws cv::Exception If @pref{order} is invalid.
 */
Wavelet create_symlets(int order, int type = CV_64FC1);

/**
 * @brief Creates a Coiflets Wavelet.
 *
 * @param[in] order The order of the wavelet.
 *                  Must be 1 \f$\le\f$ @pref{order} \f$\le\f$ 17.
 * @param[in] type The wavelet data type.
 * @throws cv::Exception If @pref{order} is invalid.
 */
Wavelet create_coiflets(int order, int type = CV_64FC1);

/**
 * @brief Creates a Biorthogonal Wavelet.
 *
 * @param[in] wavelet_vanishing_moments The number of vanishing moments of the wavelet function.
 * @param[in] scaling_vanishing_moments The number of vanishing moments of the scaling function.
 * @param[in] type The wavelet data type.
 * @throws cv::Exception If @pref{wavelet_vanishing_moments} and
 *                       @pref{scaling_vanishing_moments} are an invalid pair.
 */
Wavelet create_biorthogonal(
    int wavelet_vanishing_moments,
    int scaling_vanishing_moments,
    int type = CV_64FC1
);

/**
 * @brief Creates a Reverse Biorthogonal Wavelet.
 *
 * @param[in] wavelet_vanishing_moments The number of vanishing moments of the wavelet function.
 * @param[in] scaling_vanishing_moments The number of vanishing moments of the scaling function.
 * @param[in] type The wavelet data type.
 * @throws cv::Exception If @pref{wavelet_vanishing_moments} and
 *                       @pref{scaling_vanishing_moments} are an invalid pair.
 */
Wavelet create_reverse_biorthogonal(
    int wavelet_vanishing_moments,
    int scaling_vanishing_moments,
    int type = CV_64FC1
);
/** @}*/


template <typename... CallArgs>
std::map<std::string, std::function<Wavelet(CallArgs...)>> Wavelet::_wavelet_factories{
    {"haar", std::bind(create_haar, CV_64FC1)},
    //  daubechies
    {"db1", std::bind(create_daubechies, 1, CV_64FC1)},
    {"db2", std::bind(create_daubechies, 2, CV_64FC1)},
    {"db3", std::bind(create_daubechies, 3, CV_64FC1)},
    {"db4", std::bind(create_daubechies, 4, CV_64FC1)},
    {"db5", std::bind(create_daubechies, 5, CV_64FC1)},
    {"db6", std::bind(create_daubechies, 6, CV_64FC1)},
    {"db7", std::bind(create_daubechies, 7, CV_64FC1)},
    {"db8", std::bind(create_daubechies, 8, CV_64FC1)},
    {"db9", std::bind(create_daubechies, 9, CV_64FC1)},
    {"db10", std::bind(create_daubechies, 10, CV_64FC1)},
    {"db11", std::bind(create_daubechies, 11, CV_64FC1)},
    {"db12", std::bind(create_daubechies, 12, CV_64FC1)},
    {"db13", std::bind(create_daubechies, 13, CV_64FC1)},
    {"db14", std::bind(create_daubechies, 14, CV_64FC1)},
    {"db15", std::bind(create_daubechies, 15, CV_64FC1)},
    {"db16", std::bind(create_daubechies, 16, CV_64FC1)},
    {"db17", std::bind(create_daubechies, 17, CV_64FC1)},
    {"db18", std::bind(create_daubechies, 18, CV_64FC1)},
    {"db19", std::bind(create_daubechies, 19, CV_64FC1)},
    {"db20", std::bind(create_daubechies, 20, CV_64FC1)},
    {"db21", std::bind(create_daubechies, 21, CV_64FC1)},
    {"db22", std::bind(create_daubechies, 22, CV_64FC1)},
    {"db23", std::bind(create_daubechies, 23, CV_64FC1)},
    {"db24", std::bind(create_daubechies, 24, CV_64FC1)},
    {"db25", std::bind(create_daubechies, 25, CV_64FC1)},
    {"db26", std::bind(create_daubechies, 26, CV_64FC1)},
    {"db27", std::bind(create_daubechies, 27, CV_64FC1)},
    {"db28", std::bind(create_daubechies, 28, CV_64FC1)},
    {"db29", std::bind(create_daubechies, 29, CV_64FC1)},
    {"db30", std::bind(create_daubechies, 30, CV_64FC1)},
    {"db31", std::bind(create_daubechies, 31, CV_64FC1)},
    {"db32", std::bind(create_daubechies, 32, CV_64FC1)},
    {"db33", std::bind(create_daubechies, 33, CV_64FC1)},
    {"db34", std::bind(create_daubechies, 34, CV_64FC1)},
    {"db35", std::bind(create_daubechies, 35, CV_64FC1)},
    {"db36", std::bind(create_daubechies, 36, CV_64FC1)},
    {"db37", std::bind(create_daubechies, 37, CV_64FC1)},
    {"db38", std::bind(create_daubechies, 38, CV_64FC1)},
    //  symlets
    {"sym2", std::bind(create_symlets, 2, CV_64FC1)},
    {"sym3", std::bind(create_symlets, 3, CV_64FC1)},
    {"sym4", std::bind(create_symlets, 4, CV_64FC1)},
    {"sym5", std::bind(create_symlets, 5, CV_64FC1)},
    {"sym6", std::bind(create_symlets, 6, CV_64FC1)},
    {"sym7", std::bind(create_symlets, 7, CV_64FC1)},
    {"sym8", std::bind(create_symlets, 8, CV_64FC1)},
    {"sym9", std::bind(create_symlets, 9, CV_64FC1)},
    {"sym10", std::bind(create_symlets, 10, CV_64FC1)},
    {"sym11", std::bind(create_symlets, 11, CV_64FC1)},
    {"sym12", std::bind(create_symlets, 12, CV_64FC1)},
    {"sym13", std::bind(create_symlets, 13, CV_64FC1)},
    {"sym14", std::bind(create_symlets, 14, CV_64FC1)},
    {"sym15", std::bind(create_symlets, 15, CV_64FC1)},
    {"sym16", std::bind(create_symlets, 16, CV_64FC1)},
    {"sym17", std::bind(create_symlets, 17, CV_64FC1)},
    {"sym18", std::bind(create_symlets, 18, CV_64FC1)},
    {"sym19", std::bind(create_symlets, 19, CV_64FC1)},
    {"sym20", std::bind(create_symlets, 20, CV_64FC1)},
    //  coiflets
    {"coif1", std::bind(create_coiflets, 1, CV_64FC1)},
    {"coif2", std::bind(create_coiflets, 2, CV_64FC1)},
    {"coif3", std::bind(create_coiflets, 3, CV_64FC1)},
    {"coif4", std::bind(create_coiflets, 4, CV_64FC1)},
    {"coif5", std::bind(create_coiflets, 5, CV_64FC1)},
    {"coif6", std::bind(create_coiflets, 6, CV_64FC1)},
    {"coif7", std::bind(create_coiflets, 7, CV_64FC1)},
    {"coif8", std::bind(create_coiflets, 8, CV_64FC1)},
    {"coif9", std::bind(create_coiflets, 9, CV_64FC1)},
    {"coif10", std::bind(create_coiflets, 10, CV_64FC1)},
    {"coif11", std::bind(create_coiflets, 11, CV_64FC1)},
    {"coif12", std::bind(create_coiflets, 12, CV_64FC1)},
    {"coif13", std::bind(create_coiflets, 13, CV_64FC1)},
    {"coif14", std::bind(create_coiflets, 14, CV_64FC1)},
    {"coif15", std::bind(create_coiflets, 15, CV_64FC1)},
    {"coif16", std::bind(create_coiflets, 16, CV_64FC1)},
    {"coif17", std::bind(create_coiflets, 17, CV_64FC1)},
    //  biorthongonal
    {"bior1.1", std::bind(create_biorthogonal, 1, 1, CV_64FC1)},
    {"bior1.3", std::bind(create_biorthogonal, 1, 3, CV_64FC1)},
    {"bior1.5", std::bind(create_biorthogonal, 1, 5, CV_64FC1)},
    {"bior2.2", std::bind(create_biorthogonal, 2, 2, CV_64FC1)},
    {"bior2.4", std::bind(create_biorthogonal, 2, 4, CV_64FC1)},
    {"bior2.6", std::bind(create_biorthogonal, 2, 6, CV_64FC1)},
    {"bior2.8", std::bind(create_biorthogonal, 2, 8, CV_64FC1)},
    {"bior3.1", std::bind(create_biorthogonal, 3, 1, CV_64FC1)},
    {"bior3.3", std::bind(create_biorthogonal, 3, 3, CV_64FC1)},
    {"bior3.5", std::bind(create_biorthogonal, 3, 5, CV_64FC1)},
    {"bior3.7", std::bind(create_biorthogonal, 3, 7, CV_64FC1)},
    {"bior3.9", std::bind(create_biorthogonal, 3, 9, CV_64FC1)},
    {"bior4.4", std::bind(create_biorthogonal, 4, 4, CV_64FC1)},
    {"bior5.5", std::bind(create_biorthogonal, 5, 5, CV_64FC1)},
    {"bior6.8", std::bind(create_biorthogonal, 6, 8, CV_64FC1)},
    //  reverse biorthongonal
    {"rbior1.1", std::bind(create_reverse_biorthogonal, 1, 1, CV_64FC1)},
    {"rbior1.3", std::bind(create_reverse_biorthogonal, 1, 3, CV_64FC1)},
    {"rbior1.5", std::bind(create_reverse_biorthogonal, 1, 5, CV_64FC1)},
    {"rbior2.2", std::bind(create_reverse_biorthogonal, 2, 2, CV_64FC1)},
    {"rbior2.4", std::bind(create_reverse_biorthogonal, 2, 4, CV_64FC1)},
    {"rbior2.6", std::bind(create_reverse_biorthogonal, 2, 6, CV_64FC1)},
    {"rbior2.8", std::bind(create_reverse_biorthogonal, 2, 8, CV_64FC1)},
    {"rbior3.1", std::bind(create_reverse_biorthogonal, 3, 1, CV_64FC1)},
    {"rbior3.3", std::bind(create_reverse_biorthogonal, 3, 3, CV_64FC1)},
    {"rbior3.5", std::bind(create_reverse_biorthogonal, 3, 5, CV_64FC1)},
    {"rbior3.7", std::bind(create_reverse_biorthogonal, 3, 7, CV_64FC1)},
    {"rbior3.9", std::bind(create_reverse_biorthogonal, 3, 9, CV_64FC1)},
    {"rbior4.4", std::bind(create_reverse_biorthogonal, 4, 4, CV_64FC1)},
    {"rbior5.5", std::bind(create_reverse_biorthogonal, 5, 5, CV_64FC1)},
    {"rbior6.8", std::bind(create_reverse_biorthogonal, 6, 8, CV_64FC1)},
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
        const std::string& name_prefix = "",
        const std::source_location& location = std::source_location::current()
    ) CVWT_WAVELET_NOEXCEPT;

} // namespace internal
} // namespace cvwt

#endif  // CVWT_WAVELET_HPP

