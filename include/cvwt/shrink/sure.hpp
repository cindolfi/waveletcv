#ifndef CVWT_SHRINK_SURE_HPP
#define CVWT_SHRINK_SURE_HPP

#include <opencv2/core.hpp>
#include "cvwt/shrink/shrink.hpp"
#include "cvwt/dwt2d.hpp"

namespace cvwt
{
/** @addtogroup shrinkage
 *  @{
 */
//  ----------------------------------------------------------------------------
//  SureShrink
//  ----------------------------------------------------------------------------
/**
 * @brief Implements the SureShrink algorithm for shrinking DWT coefficients.
 * @headerfile <cvwt/shrinkage.hpp>
 *
 * The coefficients can be partitioned Shrink::GLOBALLY into a single subset,
 * by Shrink::LEVELS into level subsets, or by Shrink::SUBBANDS into subband
 * subsets.  The default is Shrink::SUBBANDS.
 *
 * This algorithm uses soft_threshold().  There are two possible methods used
 * to compute the thresholds.  The SureShrink::STRICT variant always computes
 * each subset threshold by minimizing the SURE risk.  The SureShrink::HYBRID
 * variant chooses between the SURE risk minimizer or the universal_threshold()
 * depending on the coefficients L2 norm.  See
 * https://computing.llnl.gov/sites/default/files/jei2001.pdf for the details.
 *
 * The thresholds are determined by minimizing the SURE risk.  The optimization
 * algorithm is set by passing a SureShrink::Optimizer to the constructor.  When
 * the optimization algorithm is SureShrink::BRUTE_FORCE each subset's threshold
 * is computed by evaluating the risk using the absolute value of each
 * coefficient in that subset (i.e.\f$O(rows \cdot cols \cdot channels)\f$ runtime).  Although
 * this is the most accurate method, it can be prohibitively slow for larger
 * subsets of coefficients.  SureShrink::AUTO is the default optimization
 * algorithm.  It uses the algorithm set by SureShrink::AUTO_OPTIMIZER for
 * larger subsets and SureShrink::BRUTE_FORCE for smaller subsets.  The cutoff
 * size between these two modes is set with SureShrink::AUTO_BRUTE_FORCE_SIZE_LIMIT.
 *
 */
class SureShrink : public Shrink
{
public:
    enum Variant {
        /** Always use the threshold that minimizes the SURE risk */
        STRICT,
        /** Use the threshold that minimizes the SURE risk unless the L2 norm of the coefficient subset is less than a certain limit.  In which case, use the universal_threshold(). */
        HYBRID,
    };

    enum Optimizer {
        /** Use BRUTE_FORCE or AUTO_OPTIMIZER depending on the size of the coefficient subset */
        AUTO,
        /** [Nelder-Mead Simplex](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#nelder-mead-simplex) */
        NELDER_MEAD,
        /** [Subplex](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#sbplx-based-on-subplex) */
        SBPLX,
        /** [COBYLA (Constrained Optimization By Linear Approximations)](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#cobyla-constrained-optimization-by-linear-approximations) */
        COBYLA,
        /** [BOBYQA (Bound Optimization BY Quadratic Approximation)](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#bobyqa) */
        BOBYQA,
        /** [Dividing Rectangles](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#direct-and-direct-l) */
        DIRECT,
        /** [Dividing Rectangles (locally biased variant)](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#direct-and-direct-l) */
        DIRECT_L,
        /** Compute the SURE risk for all possible thresholds */
        BRUTE_FORCE,
    };

    struct OptimizerStopConditions
    {
        /** The relative change of the threshold. */
        double threshold_rel_tol = 1e-4;
        /** The absolute change of the threshold. */
        double threshold_abs_tol = 1e-6;
        double risk_rel_tol = 1e-6;
        double risk_abs_tol = 0.0;
        double max_time = 10.0;
        bool timeout_is_error = false;
        int max_evals = 0;
        bool max_evals_is_error = false;
    };

    /**
     * @brief Base for exceptions that occur when optimization is stopped before converging.
     */
    class StoppedEarly : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    /**
     * @brief Exception thrown when optimization reaches the maximum allowed time.
     *
     * @see
     *  - optimizer_stop_conditions()
     *  - set_optimizer_stop_conditions()
     *  - fail_on_timeout()
     *  - warn_on_timeout()
     */
    class TimeoutOccured : public StoppedEarly
    {
    public:
        TimeoutOccured() : StoppedEarly("The maximum allowed duration was reached.") {}
    };

    /**
     * @brief Exception thrown when optimization reaches the maximum allowed number of evaluations.
     *
     * @see
     *  - optimizer_stop_conditions()
     *  - set_optimizer_stop_conditions()
     *  - fail_on_max_evaluations()
     *  - warn_on_max_evaluations()
     */
    class MaxEvaluationsReached : public StoppedEarly
    {
    public:
        MaxEvaluationsReached() : StoppedEarly("The maximum number of evaluations was reached.") {}
    };

    /**
     * @brief The number of primitive array elements at which the AUTO
     *        optimization algorithm switches from BRUTE_FORCE to AUTO_OPTIMIZER.
     *
     * The defaults is 32 * 32 * 3 (i.e. a 3 channel, 32x32 image).
     */
    static int AUTO_BRUTE_FORCE_SIZE_LIMIT;

    /**
     * @brief The optimization algorithm used by the AUTO optimization algorithm.
     *
     * The default is SBPLX.
     */
    static Optimizer AUTO_OPTIMIZER;

public:
    /**
     * @brief Default constructor.
     *
     * Equiqualent to:
     * @code{cpp}
     * SureShrink(Shrink::SUBBANDS, SureShrink::HYBRID, SureShrink::AUTO)
     * @endcode
     */
    SureShrink() :
        SureShrink(
            Shrink::SUBBANDS,
            SureShrink::HYBRID,
            SureShrink::AUTO
        )
    {}

    /**
     * @brief This is an overloaded constructor provided for convenience.
     *
     * Equiqualent to:
     * @code{cpp}
     * SureShrink(Shrink::SUBBANDS, variant, SureShrink::AUTO)
     * @endcode
     *
     * @param variant The variant of the algorithm.
     */
    SureShrink(SureShrink::Variant variant) :
        SureShrink(
            Shrink::SUBBANDS,
            variant,
            SureShrink::AUTO
        )
    {}

    /**
     * @brief This is an overloaded constructor provided for convenience.
     *
     * Equiqualent to:
     * @code{cpp}
     * SureShrink(partition, SureShrink::HYBRID, SureShrink::AUTO)
     * @endcode
     *
     * @param partition
     */
    SureShrink(Shrink::Partition partition) :
        SureShrink(
            partition,
            SureShrink::HYBRID,
            SureShrink::AUTO
        )
    {}

    /**
     * @brief This is an overloaded constructor provided for convenience.
     *
     * Equiqualent to:
     * @code{cpp}
     * SureShrink(partition, variant, SureShrink::AUTO)
     * @endcode
     *
     * @param partition
     * @param variant
     */
    SureShrink(Shrink::Partition partition, SureShrink::Variant variant) :
        SureShrink(
            partition,
            variant,
            SureShrink::AUTO
        )
    {}

    /**
     * @brief Construct a SureShrink object.
     *
     * @param partition How the coeffcients are partitioned.
     * @param variant The variant of the algorithm to use.
     * @param optimizer The optimization algorithm used to compute the thresholds.
     */
    SureShrink(
        Shrink::Partition partition,
        SureShrink::Variant variant,
        SureShrink::Optimizer optimizer
    ) :
        Shrink(
            partition,
            soft_threshold,
            mad_stdev
        ),
        _variant(variant),
        _optimizer(optimizer)
    {}

    /**
     * @brief Construct a SureShrink object with specified optimizer stopping conditions.
     *
     * @param partition How the coeffcients are partitioned.
     * @param variant The variant of the algorithm to use.
     * @param optimizer The optimization algorithm used to compute the thresholds.
     * @param stop_conditions The conditions used to determine optimizer convergence.
     */
    SureShrink(
        Shrink::Partition partition,
        SureShrink::Variant variant,
        SureShrink::Optimizer optimizer,
        const OptimizerStopConditions& stop_conditions
    ) :
        Shrink(
            partition,
            soft_threshold,
            mad_stdev
        ),
        _variant(variant),
        _optimizer(optimizer),
        _stop_conditions(stop_conditions)
    {}

    /**
     * @brief Copy constructor.
     */
    SureShrink(const SureShrink& other) = default;
    /**
     * @brief Move constructor.
     */
    SureShrink(SureShrink&& other) = default;

    Variant variant() const { return _variant; }
    Optimizer optimizer() const { return _optimizer; }
    OptimizerStopConditions optimizer_stop_conditions() const { return _stop_conditions; }
    void set_optimizer_stop_conditions(const OptimizerStopConditions& stop_conditions)
    {
        _stop_conditions = stop_conditions;
    }
    /**
     * @brief Throw an exception when the optimizer runs for the maximum allowed time.
     */
    void fail_on_timeout() { _stop_conditions.timeout_is_error = true; }
    /**
     * @brief Issue a warning when the optimizer runs for the maximum allowed time.
     */
    void warn_on_timeout() { _stop_conditions.timeout_is_error = false; }
    /**
     * @brief Return true if an exception is thrown when the optimizer runs for the maximum allowed time.
     */
    bool is_timeout_an_error() { return _stop_conditions.timeout_is_error; }

    /**
     *
     * @brief Throw an exception when the optimizer performs the maximum number of allowed evaluations.
     */
    void fail_on_max_evaluations() { _stop_conditions.max_evals_is_error = true; }
    /**
     * @brief Issue a warning when the optimizer performs the maximum number of allowed evaluations.
     */
    void warn_on_max_evaluations() { _stop_conditions.max_evals_is_error = false; }
    /**
     * @brief Return true if an exception is thrown when the optimizer performs the maximum number of allowed evaluations.
     */
    bool is_max_evaluations_reached_an_error() const { return _stop_conditions.max_evals_is_error; }

    /**
     * @brief Compute the SureShrink algorithm threshold.
     *
     * For a single channel with \f$\sigma = 1\f$, the SURE threshold is defined by
     * \f{equation}{
     *     \lambda_{SURE} = \argmin_{\lambda \ge 0} \SURE(\lambda, w)
     * \f}
     * where the input \f$w\f$ is a subset of noisy detail coefficients measurements
     * drawn from an unknown normal distribution.
     * \f$\SURE(\lambda, w)\f$ is an unbiased estimator of the MSE between the true
     * detail coefficients and the soft thresholded coefficients using the
     * threshold \f$\lambda\f$.
     *
     * Specifically,
     * \f{equation}{
     *     \SURE(\lambda, w) =
     *         N
     *         - 2 [\# \, \mathrm{of} \, |w_n| < \lambda]
     *         + \sum_{n = 1}^{N} [\min(|w_n|, \lambda)]^2
     * \f}
     * where \f$N\f$ is the number of coefficients and
     * \f$[\# \, \mathrm{of} \, |w_n| < \lambda]\f$ is the number of coefficients lesser
     * than \f$\lambda\f$ in magnitude.
     *
     * When the detail coefficients are sparse it is preferable to employ hybrid
     * algorithm that chooses the compute_universal_threshold() when
     * \f{equation}{
     *     \frac{1}{N} \sum_{n = 1}^{N} \left( \frac{w_n}{\sigma} - 1 \right)^2
     *     \le
     *     \frac{\left(\log_2(N)\right)^{3/2}}{\sqrt{N}}
     * \f}
     * and \f$\lambda_{SURE}\f$ otherwise.
     * Pass `SureShrink::HYBRID` as the `variant` argument to use
     * this implementation.
     *
     * When the standard deviation of each input channel is not one, each channel
     * standard deviation must be passed in using the `stdev` argument.
     * In which case, the resulting threshold will be suitably scaled to work on the
     * non-standardized coefficients.
     *
     * This function returns a multichannel threshold by applying the single channel
     * algorithm to each channel of the given `detail_coeffs`.
     *
     * @see compute_universal_threshold()
     *
     * @param detail_coeffs The detail coefficients.
     * @param stdev The standard deviations of each of the detail coefficients channels.
     * @param mask Indicates which coefficients are used to compute the
     *             threshold.  This must be a single channel with type CV_8U1 or
     *             CV_8S1.
     */
    cv::Scalar compute_sure_threshold(
        cv::InputArray detail_coeffs,
        const cv::Scalar& stdev,
        cv::InputArray mask = cv::noArray()
    ) const;

    cv::Scalar compute_sure_risk(
        cv::InputArray detail_coeffs,
        const cv::Scalar& threshold,
        const cv::Scalar& stdev,
        cv::InputArray mask = cv::noArray()
    );
protected:
    /**
     * @brief Computes the threshold on a single global subset of coefficients.
     *
     * @param coeffs The entire set of DWT coefficients.
     * @param levels The subset of levels over which to compute the threshold.
     * @param stdev The standard deviation of the coefficient noise.
     */
    cv::Scalar compute_global_threshold(
        const DWT2D::Coeffs& coeffs,
        const cv::Range& levels,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_sure_threshold(coeffs, stdev, coeffs.detail_mask(levels));
    }

    /**
     * @brief Computes the threshold for a single level subset of coefficients.
     *
     * @param detail_coeffs The level detail coefficients.
     * @param level The decomposition level of the given coefficients.
     * @param stdev The standard deviation of the coefficient noise.
     */
    cv::Scalar compute_level_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_sure_threshold(detail_coeffs, stdev);
    }

    /**
     * @brief Computes the threshold for a single subband subset of coefficients.
     *
     * @param detail_coeffs The subband detail coefficients.
     * @param level The decomposition level of the given coefficients.
     * @param subband The subband of the coefficients.
     * @param stdev The standard deviation of the coefficient noise.
     */
    cv::Scalar compute_subband_threshold(
        const cv::Mat& detail_coeffs,
        int level,
        int subband,
        const cv::Scalar& stdev
    ) const override
    {
        return compute_sure_threshold(detail_coeffs, stdev);
    }

    /**
     * @brief Resolves SureShrink::Optimizer::AUTO to an actual
     *        SureShrink::Optimizer depending on the size of the coefficients.
     *
     * This is equalivalent to optimizer() if the optimizer is not
     * SureShrink::Optimizer::AUTO.
     *
     * @param detail_coeffs A subset of detail coefficients.
     */
    Optimizer resolve_optimizer(cv::InputArray detail_coeffs) const;

private:
    SureShrink::Variant _variant;
    SureShrink::Optimizer _optimizer;
    OptimizerStopConditions _stop_conditions;
};




//  ----------------------------------------------------------------------------
//  SureShrink Functional API
//  ----------------------------------------------------------------------------
/** @name SureShrink Functional API
 *  @{
 */
/**
 * @brief %Shrink detail coeffcients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs sure_shrink(const DWT2D::Coeffs& coeffs);

/**
 * @brief %Shrink detail coeffcients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void sure_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief %Shrink detail coeffcients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs sure_shrink(DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief %Shrink detail coeffcients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
void sure_shrink(DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels);

/**
 * @brief %Shrink detail coeffcients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs);

/**
 * @brief %Shrink detail coeffcients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief %Shrink detail coeffcients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief %Shrink detail coeffcients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
void sure_shrink_levelwise(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& shrunk_coeffs,
    int levels
);
/** @}*/

/** @} shrinkage */
} // namespace cvwt

#endif  // CVWT_SHRINK_SURE_HPP

