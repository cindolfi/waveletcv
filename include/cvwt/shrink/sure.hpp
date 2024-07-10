#ifndef CVWT_SHRINK_SURE_HPP
#define CVWT_SHRINK_SURE_HPP

#include <opencv2/core.hpp>
#include "cvwt/shrink/shrink.hpp"
#include "cvwt/dwt2d.hpp"

namespace cvwt
{
/** @addtogroup shrink
 *  @{
 */
//  ----------------------------------------------------------------------------
//  SureShrink
//  ----------------------------------------------------------------------------
/**
 * @brief Implements the SureShrink algorithm for shrinking DWT coefficients.
 * @headerfile <cvwt/shrink.hpp>
 *
 * The coefficients can be partitioned Shrinker::GLOBALLY into a single subset,
 * by Shrinker::LEVELS into level subsets, or by Shrinker::SUBBANDS into subband
 * subsets.  The default is Shrinker::SUBBANDS.
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
 * coefficient in that subset (i.e.\f$O(rows \cdot cols \cdot channels)\f$
 * runtime).  Although this is the most accurate method, it can be prohibitively
 * slow for larger subsets of coefficients.  SureShrink::AUTO is the default
 * optimization algorithm.  It uses the algorithm set by
 * SureShrink::AUTO_OPTIMIZER for larger subsets and SureShrink::BRUTE_FORCE for
 * smaller subsets.  The cutoff size between these two modes is set with
 * SureShrink::AUTO_BRUTE_FORCE_SIZE_LIMIT.
 */
class SureShrinker : public Shrinker
{
public:
    /**
     * @brief The variant of the SureShrink algorithm.
     */
    enum Variant {
        /** Always use the threshold that minimizes the SURE risk */
        STRICT,
        /** Use the threshold that minimizes the SURE risk unless the L2 norm of the coefficient subset is less than a certain limit.  In which case, use the universal_threshold(). */
        HYBRID,
    };

    /**
     * @brief The optimizer used to compute thresholds.
     */
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

    /**
     * @brief The conditions that define optimizer convergence.
     */
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
     * @see optimizer_stop_conditions, set_optimizer_stop_conditions,
     *      fail_on_timeout, warn_on_timeout
     */
    class TimeoutOccured : public StoppedEarly
    {
    public:
        TimeoutOccured() : StoppedEarly("The maximum allowed duration was reached.") {}
    };

    /**
     * @brief Exception thrown when optimization reaches the maximum allowed number of evaluations.
     *
     * @see optimizer_stop_conditions, set_optimizer_stop_conditions,
     *      fail_on_max_evaluations, warn_on_max_evaluations
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
     * SureShrink(Shrinker::SUBBANDS, SureShrink::HYBRID, SureShrink::AUTO)
     * @endcode
     */
    SureShrinker() :
        SureShrinker(
            Shrinker::SUBBANDS,
            SureShrinker::HYBRID,
            SureShrinker::AUTO
        )
    {}

    /**
     * @overload
     *
     * Equiqualent to:
     * @code{cpp}
     * SureShrink(Shrinker::SUBBANDS, variant, SureShrink::AUTO)
     * @endcode
     *
     * @param[in] variant The variant of the algorithm.
     */
    SureShrinker(SureShrinker::Variant variant) :
        SureShrinker(
            Shrinker::SUBBANDS,
            variant,
            SureShrinker::AUTO
        )
    {}

    /**
     * @overload
     *
     * Equiqualent to:
     * @code{cpp}
     * SureShrink(partition, SureShrink::HYBRID, SureShrink::AUTO)
     * @endcode
     *
     * @param[in] partition
     */
    SureShrinker(Shrinker::Partition partition) :
        SureShrinker(
            partition,
            SureShrinker::HYBRID,
            SureShrinker::AUTO
        )
    {}

    /**
     * @overload
     *
     * Equiqualent to:
     * @code{cpp}
     * SureShrink(partition, variant, SureShrink::AUTO)
     * @endcode
     *
     * @param[in] partition
     * @param[in] variant
     */
    SureShrinker(Shrinker::Partition partition, SureShrinker::Variant variant) :
        SureShrinker(
            partition,
            variant,
            SureShrinker::AUTO
        )
    {}

    /**
     * @brief Construct a SureShrinker object.
     *
     * @param[in] partition How the coefficients are partitioned.
     * @param[in] variant The variant of the algorithm to use.
     * @param[in] optimizer The optimization algorithm used to compute the thresholds.
     */
    SureShrinker(
        Shrinker::Partition partition,
        SureShrinker::Variant variant,
        SureShrinker::Optimizer optimizer
    ) :
        Shrinker(
            partition,
            soft_threshold,
            mad_stdev
        ),
        _variant(variant),
        _optimizer(optimizer)
    {}

    /**
     * @brief Construct a SureShrinker object with specified optimizer stopping conditions.
     *
     * @param[in] partition How the coefficients are partitioned.
     * @param[in] variant The variant of the algorithm to use.
     * @param[in] optimizer The optimization algorithm used to compute the thresholds.
     * @param[in] stop_conditions The conditions used to determine optimizer convergence.
     */
    SureShrinker(
        Shrinker::Partition partition,
        SureShrinker::Variant variant,
        SureShrinker::Optimizer optimizer,
        const OptimizerStopConditions& stop_conditions
    ) :
        Shrinker(
            partition,
            soft_threshold,
            mad_stdev
        ),
        _variant(variant),
        _optimizer(optimizer),
        _stop_conditions(stop_conditions)
    {}

    /**
     * @brief Copy Constructor.
     */
    SureShrinker(const SureShrinker& other) = default;
    /**
     * @brief Move Constructor.
     */
    SureShrinker(SureShrinker&& other) = default;

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
     * @brief Returns true if an exception is thrown when the optimizer runs for the maximum allowed time.
     */
    bool is_timeout_an_error() { return _stop_conditions.timeout_is_error; }

    /**
     * @brief Throw an exception when the optimizer performs the maximum number of allowed evaluations.
     */
    void fail_on_max_evaluations() { _stop_conditions.max_evals_is_error = true; }
    /**
     * @brief Issue a warning when the optimizer performs the maximum number of allowed evaluations.
     */
    void warn_on_max_evaluations() { _stop_conditions.max_evals_is_error = false; }
    /**
     * @brief Returns true if an exception is thrown when the optimizer performs the maximum number of allowed evaluations.
     */
    bool is_max_evaluations_reached_an_error() const { return _stop_conditions.max_evals_is_error; }

    /**
     * @brief Computes the SureShrink algorithm threshold.
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
     * Pass SureShrinker::HYBRID as the @pref{variant} argument to use
     * this implementation.
     *
     * When the standard deviation of each input channel is not one, each channel
     * standard deviation must be passed in using the @pref{stdev} argument.
     * In which case, the resulting threshold will be suitably scaled to work on the
     * non-standardized coefficients.
     *
     * This function returns a multichannel threshold by applying the single channel
     * algorithm to each channel of the given @pref{detail_coeffs}.
     *
     * @param[in] detail_coeffs The detail coefficients.
     * @param[in] stdev The standard deviations of each of the detail coefficients channels.
     * @param[in] mask Indicates which coefficients are used to compute the
     *             threshold.  This must be a single channel with type CV_8U1 or
     *             CV_8S1.
     */
    cv::Scalar compute_sure_threshold(
        cv::InputArray detail_coeffs,
        const cv::Scalar& stdev,
        cv::InputArray mask = cv::noArray()
    ) const;

    /**
     * @brief Computes the SURE risk estimate.
     *
     * @param[in] detail_coeffs
     * @param[in] threshold
     * @param[in] stdev
     * @param[in] mask
     */
    cv::Scalar compute_sure_risk(
        cv::InputArray detail_coeffs,
        const cv::Scalar& threshold,
        const cv::Scalar& stdev,
        cv::InputArray mask = cv::noArray()
    );
protected:
    /**
     * @copydoc Shrinker::compute_global_threshold
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
     * @copydoc Shrinker::compute_level_threshold
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
     * @copydoc Shrinker::compute_subband_threshold
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
     * @param[in] detail_coeffs A subset of detail coefficients.
     */
    Optimizer resolve_optimizer(cv::InputArray detail_coeffs) const;

private:
    SureShrinker::Variant _variant;
    SureShrinker::Optimizer _optimizer;
    OptimizerStopConditions _stop_conditions;
};




//  ----------------------------------------------------------------------------
//  SureShrink Functional API
//  ----------------------------------------------------------------------------
/** @name SureShrink Functional API
 *  @{
 */
/**
 * @brief Shrinks detail coefficients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs sure_shrink(const DWT2D::Coeffs& coeffs);

/**
 * @brief Shrinks detail coefficients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void sure_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief Shrinks detail coefficients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs sure_shrink(const DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrinks detail coefficients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
void sure_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels);

/**
 * @brief Shrinks detail coefficients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 */
DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs);

/**
 * @brief Shrinks detail coefficients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[out] shrunk_coeffs The shrunk discrete wavelet transform coefficients.
 */
void sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs);

/**
 * @brief Shrinks detail coefficients using the SureShrink algorithm.
 *
 * @param[in] coeffs The discrete wavelet transform coefficients.
 * @param[in] levels The maximum number of levels to shrink.  Shrinking is applied
 *                   starting at the lowest level (i.e. smallest scale).
 */
DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, int levels);

/**
 * @brief Shrinks detail coefficients using the SureShrink algorithm.
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

/** @} shrink */
} // namespace cvwt

#endif  // CVWT_SHRINK_SURE_HPP

