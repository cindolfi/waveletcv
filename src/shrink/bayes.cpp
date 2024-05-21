
#include "cvwt/shrink/bayes.hpp"

#include "cvwt/utils.hpp"
#include "cvwt/exception.hpp"

namespace cvwt
{
namespace internal
{
template <typename T, int CHANNELS>
struct ComputeBayesThreshold
{
    cv::Scalar operator()(cv::InputArray input, const cv::Scalar& stdev) const
    {
        throw_if_empty(input);
        assert(input.channels() == CHANNELS);
        if constexpr (CHANNELS == 1) {
            return compute_single_channel_threshold(input.getMat(), stdev[0]);
        } else {
            cv::Mat channels[CHANNELS];
            cv::split(input.getMat(), channels);
            cv::Scalar result;
            for (int i = 0; i < CHANNELS; ++i)
                result[i] = compute_single_channel_threshold(channels[i], stdev[i]);

            return result;
        }
    }

    cv::Scalar operator()(
        cv::InputArray input,
        cv::InputArray mask,
        const cv::Scalar& stdev
    ) const
    {
        cv::Mat masked_input;
        CollectMasked<T, CHANNELS>()(input, masked_input, mask);
        return this->operator()(masked_input, stdev);
    }

private:
    double compute_single_channel_threshold(const cv::Mat& array, double stdev) const
    {
        assert(array.channels() == 1);
        double noise_variance = stdev * stdev;
        double observation_variance = cv::sum(array.mul(array))[0] / array.total();

        if (noise_variance >= observation_variance) {
            return maximum_abs_value(array);
        } else {
            auto signal_stdev = std::sqrt(observation_variance - noise_variance);
            return noise_variance / signal_stdev;
        }
    }
};
}   // namespace internal


//  ----------------------------------------------------------------------------
//  BayesShrink
//  ----------------------------------------------------------------------------
cv::Scalar BayesShrink::compute_bayes_threshold(
    cv::InputArray detail_coeffs,
    const cv::Scalar& stdev
) const
{
    return internal::dispatch_on_pixel_type<internal::ComputeBayesThreshold>(
        detail_coeffs.type(),
        detail_coeffs,
        stdev
    );
}

cv::Scalar BayesShrink::compute_bayes_threshold(
    cv::InputArray detail_coeffs,
    cv::InputArray mask,
    const cv::Scalar& stdev
) const
{
    return internal::dispatch_on_pixel_type<internal::ComputeBayesThreshold>(
        detail_coeffs.type(),
        detail_coeffs,
        mask,
        stdev
    );
}

DWT2D::Coeffs bayes_shrink(const DWT2D::Coeffs& coeffs)
{
    BayesShrink shrink;
    return shrink(coeffs);
}

void bayes_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
{
    BayesShrink shrink;
    shrink(coeffs, shrunk_coeffs);
}

}   // namespace cvwt

