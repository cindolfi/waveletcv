
#include "cvwt/shrink/sure.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <nlopt.hpp>
#include "cvwt/dispatch.hpp"
#include "cvwt/exception.hpp"
#include "cvwt/array/array.hpp"
#include "cvwt/shrink/universal.hpp"

namespace cvwt
{
namespace internal
{
template <typename T>
struct SingleChannelComputeSureRisk
{
    /**
     * @brief Returns the threshold that minimizes the SURE risk estimate of
     *        coeffs shrunk by soft thresholding.
     *
     * @param coeffs A single channel of coefficients.
     * @param threshold The threshold.
     * @param stdev The standard deviation of the coefficients.
     */
    double operator()(cv::InputArray coeffs, double threshold, double stdev) const
    {
        assert(coeffs.channels() == 1);
        auto abs_coeffs = cv::abs(coeffs.getMat());
        if (stdev != 1.0) {
            abs_coeffs = abs_coeffs / stdev;
            threshold = threshold / stdev;
        }

        auto clamped_abs_coeffs = cv::min(abs_coeffs, threshold);
        return coeffs.total()
            + cv::sum(clamped_abs_coeffs.mul(clamped_abs_coeffs))[0]
            - 2 * cv::countNonZero(abs_coeffs <= threshold);
    }
};

template <typename T, int CHANNELS>
struct ComputeSureRisk
{
    cv::Scalar operator()(
        cv::InputArray coeffs,
        const cv::Scalar& threshold,
        const cv::Scalar& stdev
    ) const
    {
        assert(coeffs.channels() == CHANNELS);
        SingleChannelComputeSureRisk<T> compute_single_channel_sure_risk;
        if constexpr (CHANNELS == 1) {
            return compute_single_channel_sure_risk(coeffs, threshold[0], stdev[0]);
        } else {
            cv::Scalar result;
            cv::Mat coeffs_channels[CHANNELS];
            cv::split(coeffs.getMat(), coeffs_channels);
            for (int i = 0; i < CHANNELS; ++i)
                result[i] = compute_single_channel_sure_risk(
                    coeffs_channels[i],
                    threshold[i],
                    stdev[i]
                );

            return result;
        }
    }

    cv::Scalar operator()(
        cv::InputArray coeffs,
        const cv::Scalar& threshold,
        const cv::Scalar& stdev,
        cv::InputArray mask
    ) const
    {
        assert(coeffs.channels() == CHANNELS);
        cv::Mat masked_coeffs;
        collect_masked(coeffs, masked_coeffs, mask);
        return this->operator()(masked_coeffs, threshold, stdev);
    }
};

//  Forward declaration
template <typename T>
double nlopt_sure_threshold_objective(
    const std::vector<double>& x,
    std::vector<double>& grad,
    void* f_data
);

template <typename T>
struct SingleChannelComputeSureThreshold
{
    /** The smallest total size of coefficients that do not use brute force */
    static constexpr int MINIMUM_TOTAL_FOR_NLOPT = 16;
    /** The optimizers initial value on a scale of 0.0 to 1.0
     * (i.e. min. possible threshold to max. possible threshold) */
    static constexpr double INITIAL_RELATIVE_THRESHOLD = 0.2;

    /**
     * @brief Returns the threshold that minimizes the SURE risk estimate of
     *        coeffs shrunk by soft thresholding.
     *
     * @param coeffs A single channel of coefficients.
     * @param stdev The standard deviation of the coefficients.
     * @param optimizer The optimization algorithm used to compute the threshold.
     * @param variant The variant of the algorithm.
     * @param stop_conditions The parameters that determine optimizer convergence.
     */
    double operator()(
        cv::InputArray coeffs,
        double stdev,
        SureShrinker::Optimizer optimizer,
        SureShrinker::Variant variant,
        const SureShrinker::OptimizerStopConditions& stop_conditions
    ) const
    {
        assert(coeffs.channels() == 1);
        auto coeffs_matrix = coeffs.getMat();
        if (stdev != 1.0)
            coeffs_matrix = coeffs_matrix / stdev;

        double threshold;
        if (variant == SureShrinker::HYBRID && use_universal_threshold(coeffs_matrix)) {
            threshold = UniversalShrinker::compute_universal_threshold(
                coeffs_matrix.total(),
                1.0
            )[0];
        } else {
            if (coeffs_matrix.total() == 1) {
                threshold = cv::abs(coeffs_matrix.at<T>(0, 0));
            } else if (optimizer == SureShrinker::BRUTE_FORCE
                       || coeffs_matrix.total() <= MINIMUM_TOTAL_FOR_NLOPT) {
                threshold = compute_sure_threshold_using_brute_force(coeffs_matrix);
            } else {
                threshold = compute_sure_threshold_using_nlopt(
                    coeffs_matrix,
                    to_nlopt_algorithm(optimizer),
                    stop_conditions
                );
            }
        }

        return stdev * threshold;
    }

    friend double nlopt_sure_threshold_objective<T>(
        const std::vector<double>& x,
        std::vector<double>& grad,
        void* f_data
    );

private:
    nlopt::algorithm to_nlopt_algorithm(SureShrinker::Optimizer optimizer) const
    {
        switch (optimizer) {
        case SureShrinker::NELDER_MEAD: return nlopt::algorithm::LN_NELDERMEAD;
        case SureShrinker::SBPLX: return nlopt::algorithm::LN_SBPLX;
        case SureShrinker::COBYLA: return nlopt::algorithm::LN_COBYLA;
        case SureShrinker::BOBYQA: return nlopt::algorithm::LN_BOBYQA;
        case SureShrinker::DIRECT: return nlopt::algorithm::GN_DIRECT;
        case SureShrinker::DIRECT_L: return nlopt::algorithm::GN_DIRECT_L;
        }

        assert(false);
        return nlopt::algorithm::LN_NELDERMEAD;
    }

    struct NLObjectiveData
    {
        cv::Mat coeffs;
    };

    /**
     * @brief Returns the threshold that minimizes the SURE risk estimate of
     *        coeffs shrunk by soft thresholding.
     *
     * @param coeffs This is assumed to be a single channel and have unit
     *               standard deviation.
     */
    double compute_sure_threshold_using_nlopt(
        const cv::Mat& coeffs,
        nlopt::algorithm algorithm,
        const SureShrinker::OptimizerStopConditions& stop_conditions
    ) const
    {
        NLObjectiveData data(coeffs);

        nlopt::opt optimizer(algorithm, 1);
        optimizer.set_min_objective(nlopt_sure_threshold_objective<T>, &data);

        optimizer.set_maxtime(stop_conditions.max_time);
        optimizer.set_maxeval(stop_conditions.max_evals);
        optimizer.set_xtol_abs(stop_conditions.threshold_abs_tol);
        optimizer.set_xtol_rel(stop_conditions.threshold_rel_tol);
        optimizer.set_ftol_abs(stop_conditions.risk_abs_tol);
        optimizer.set_ftol_rel(stop_conditions.risk_rel_tol);

        double min_threshold;
        double max_threshold;
        cv::minMaxIdx(cv::abs(coeffs), &min_threshold, &max_threshold);
        optimizer.set_lower_bounds({min_threshold});
        optimizer.set_upper_bounds({max_threshold});
        std::vector<double> threshold = {
            (1.0 - INITIAL_RELATIVE_THRESHOLD) * min_threshold
            + INITIAL_RELATIVE_THRESHOLD * max_threshold
        };

        double optimal_risk;
        try {
            auto result = optimizer.optimize(threshold, optimal_risk);
            switch (result) {
                case nlopt::SUCCESS:
                    CV_LOG_DEBUG(NULL, "success");
                    break;
                case nlopt::STOPVAL_REACHED:
                    CV_LOG_WARNING(NULL, "stop value reached");
                    break;
                case nlopt::FTOL_REACHED:
                    CV_LOG_DEBUG(NULL, "risk tolerance reached");
                    break;
                case nlopt::XTOL_REACHED:
                    CV_LOG_DEBUG(NULL, "threshold tolerance reached");
                    break;
                case nlopt::MAXEVAL_REACHED:
                    if (stop_conditions.max_evals_is_error)
                        throw SureShrinker::MaxEvaluationsReached();

                    CV_LOG_WARNING(NULL, "maximum evaluations was reached");
                    break;
                case nlopt::MAXTIME_REACHED:
                    if (stop_conditions.timeout_is_error)
                        throw SureShrinker::TimeoutOccured();

                    CV_LOG_WARNING(NULL, "maximum time was reached");
                    break;
            }

            CV_LOG_IF_DEBUG(NULL, result > 0,
                "optimal threshold = " << threshold[0]
                << ", optimal risk = " << optimal_risk
            );
        } catch (const std::exception& error) {
            CV_LOG_ERROR(NULL, "nlopt " << error.what());
            throw error;
        }

        return threshold[0];
    }

    /**
     * @brief Returns the threshold that minimizes the SURE risk estimate of
     *        coeffs shrunk by soft thresholding.
     *
     * @param coeffs This is assumed to be a single channel and have unit
     *               standard deviation.
     */
    double compute_sure_threshold_using_brute_force(const cv::Mat& coeffs) const
    {
        SingleChannelComputeSureRisk<T> compute_sure_risk;

        cv::Mat_<T> risks(coeffs.size());
        coeffs.forEach<T>(
            [&](const auto& threshold, const auto index) {
                risks(index) = compute_sure_risk(coeffs, cv::abs(threshold), 1.0);
            }
        );

        int min_index[2];
        cv::minMaxIdx(risks, nullptr, nullptr, min_index, nullptr);
        return std::fabs(coeffs.at<T>(min_index));
    }

    /**
     * @brief Returns true if the universal threshold should be used for the hybrid variant.
     *
     * @param coeffs This is assumed to be a single channel and have unit
     *               standard deviation.
     */
    bool use_universal_threshold(const cv::Mat& coeffs) const
    {
        int n = coeffs.total();
        auto universal_l2_norm_limit = 1 + std::pow(std::log2(n), 1.5) / std::sqrt(n);
        auto l2_norm = cv::sum(coeffs.mul(coeffs))[0] / n;

        auto result = l2_norm < universal_l2_norm_limit;
        CV_LOG_DEBUG(
            NULL,
            (result ? "using universal threshold" : "using SURE threshold")
            << "  l2_norm = " << l2_norm
            << "  universal_l2_norm_limit = " << universal_l2_norm_limit
        );

        return result;
    }
};

template <typename T>
double nlopt_sure_threshold_objective(
    const std::vector<double>& x,
    std::vector<double>& grad,
    void* f_data
)
{
    auto data = static_cast<SingleChannelComputeSureThreshold<T>::NLObjectiveData*>(f_data);
    return SingleChannelComputeSureRisk<T>()(data->coeffs, x[0], 1.0);
};


template <typename T, int CHANNELS>
struct ComputeSureThreshold
{
    cv::Scalar operator()(
        cv::InputArray coeffs,
        const cv::Scalar& stdev,
        SureShrinker::Optimizer optimizer,
        SureShrinker::Variant variant,
        const SureShrinker::OptimizerStopConditions& stop_conditions
    ) const
    {
        assert(coeffs.channels() == CHANNELS);
        SingleChannelComputeSureThreshold<T> compute_single_channel_sure_threshold;
        cv::Mat coeffs_channels[CHANNELS];
        cv::split(coeffs.getMat(), coeffs_channels);
        cv::Scalar result;
        for (int i = 0; i < CHANNELS; ++i) {
            CV_LOG_DEBUG(NULL, "computing channel " << i);
            result[i] = compute_single_channel_sure_threshold(
                coeffs_channels[i],
                stdev[i],
                optimizer,
                variant,
                stop_conditions
            );
        }

        return result;
    }

    cv::Scalar operator()(
        cv::InputArray coeffs,
        cv::InputArray mask,
        const cv::Scalar& stdev,
        SureShrinker::Optimizer optimizer,
        SureShrinker::Variant variant,
        const SureShrinker::OptimizerStopConditions& stop_conditions
    ) const
    {
        assert(coeffs.channels() == CHANNELS);
        cv::Mat masked_coeffs;
        collect_masked(coeffs, masked_coeffs, mask);

        return this->operator()(masked_coeffs, stdev, optimizer, variant, stop_conditions);
    }
};
}   // namespace internal




//  ----------------------------------------------------------------------------
//  SureShrink
//  ----------------------------------------------------------------------------
int SureShrinker::AUTO_BRUTE_FORCE_SIZE_LIMIT = 32 * 32 * 3;
SureShrinker::Optimizer SureShrinker::AUTO_OPTIMIZER = SureShrinker::Optimizer::SBPLX;

cv::Scalar SureShrinker::compute_sure_threshold(
    cv::InputArray detail_coeffs,
    const cv::Scalar& stdev,
    cv::InputArray mask
) const
{
    if (is_not_array(mask)) {
        return internal::dispatch_on_pixel_type<internal::ComputeSureThreshold>(
            detail_coeffs.type(),
            detail_coeffs,
            stdev,
            resolve_optimizer(detail_coeffs),
            variant(),
            _stop_conditions
        );
    } else {
        return internal::dispatch_on_pixel_type<internal::ComputeSureThreshold>(
            detail_coeffs.type(),
            detail_coeffs,
            mask,
            stdev,
            resolve_optimizer(detail_coeffs),
            variant(),
            _stop_conditions
        );
    }
}

cv::Scalar SureShrinker::compute_sure_risk(
    cv::InputArray coeffs,
    const cv::Scalar& threshold,
    const cv::Scalar& stdev,
    cv::InputArray mask
)
{

    if (is_not_array(mask)) {
        return internal::dispatch_on_pixel_type<internal::ComputeSureRisk>(
            coeffs.type(), coeffs, threshold, stdev
        );
    } else {
        return internal::dispatch_on_pixel_type<internal::ComputeSureRisk>(
            coeffs.type(), coeffs, threshold, stdev, mask
        );
    }
}

SureShrinker::Optimizer SureShrinker::resolve_optimizer(cv::InputArray detail_coeffs) const
{
    SureShrinker::Optimizer resolved_optimizer = optimizer();
    if (optimizer() == SureShrinker::AUTO) {
        if (detail_coeffs.total() * detail_coeffs.channels() <= AUTO_BRUTE_FORCE_SIZE_LIMIT)
            resolved_optimizer = SureShrinker::BRUTE_FORCE;
        else
            resolved_optimizer = AUTO_OPTIMIZER;
    }

    return resolved_optimizer;
}

//  ----------------------------------------------------------------------------
//  SureShrink Functional API
//  ----------------------------------------------------------------------------
DWT2D::Coeffs sure_shrink(const DWT2D::Coeffs& coeffs)
{
    SureShrinker shrink;
    return shrink(coeffs);
}

void sure_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
{
    SureShrinker shrink;
    shrink(coeffs, shrunk_coeffs);
}

DWT2D::Coeffs sure_shrink(const DWT2D::Coeffs& coeffs, int levels)
{
    SureShrinker shrink;
    return shrink(coeffs, levels);
}

void sure_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels)
{
    SureShrinker shrink;
    shrink(coeffs, shrunk_coeffs, levels);
}

DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs)
{
    SureShrinker shrink(Shrinker::LEVELS);
    return shrink(coeffs);
}

void sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
{
    SureShrinker shrink(Shrinker::LEVELS);
    shrink(coeffs, shrunk_coeffs);
}

DWT2D::Coeffs sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, int levels)
{
    SureShrinker shrink(Shrinker::LEVELS);
    return shrink(coeffs, levels);
}

void sure_shrink_levelwise(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs, int levels)
{
    SureShrinker shrink(Shrinker::LEVELS);
    shrink(coeffs, shrunk_coeffs, levels);
}
}   // namespace cvwt

