/**
*/
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <wavelet/dwt2d.hpp>
#include <wavelet/utils.hpp>
#include "common.hpp"

using namespace testing;

namespace wavelet::internal
{
void PrintTo(const DWT2D::Coeffs& coeffs, std::ostream* stream)
{
    PrintTo(cv::Mat(coeffs), stream);
    *stream << "(" << coeffs.levels() << " levels)\n";
}
}   // namespace wavelet::internal

void clamp_near_zero(cv::InputArray input, cv::OutputArray output, double tolerance)
{
    std::vector<cv::Mat> channels(input.channels());
    cv::split(input.getMat(), channels);
    for (auto& channel : channels)
        channel.setTo(0, cv::abs(channel) <= tolerance);

    cv::merge(channels, output);
}

void clamp_near_zero(cv::InputOutputArray array, double tolerance)
{
    clamp_near_zero(array, array, tolerance);
}

namespace cv
{
void PrintTo(const cv::Mat& matrix, std::ostream* stream)
{
    wavelet::internal::dispatch_on_pixel_type<print_matrix_to>(
        matrix.type(),
        matrix,
        stream
    );
}
}   // namespace cv

std::string get_subband_name(int subband)
{
    switch (subband){
        case wavelet::HORIZONTAL: return "horizontal";
        case wavelet::VERTICAL: return "vertical";
        case wavelet::DIAGONAL: return "diagonal";
    }
    assert(false);
    return "";
}

cv::Mat create_matrix(int rows, int cols, int type, double initial_value)
{
    std::vector<double> elements(rows * cols);
    std::iota(elements.begin(), elements.end(), initial_value);
    auto result = cv::Mat(elements, true).reshape(0, rows);
    result.convertTo(result, type);

    return result;
}

bool multichannel_compare(const cv::Mat& a, const cv::Mat& b, int cmp_type)
{
    if (a.size() != b.size() || a.channels() != b.channels())
        return false;

    if (a.empty() && b.empty())
        return true;

    std::vector<cv::Mat> a_channels(a.channels());
    cv::split(a, a_channels);

    std::vector<cv::Mat> b_channels(b.channels());
    cv::split(b, b_channels);

    for (int i = 0; i < a.channels(); ++i) {
        cv::Mat channel_result;
        cv::compare(a_channels[i], b_channels[i], channel_result, cmp_type);
        if (cv::countNonZero(channel_result) != a.total())
            return false;
    }

    return true;
}

bool multichannel_compare(const cv::Mat& a, const cv::Scalar& b, int cmp_type)
{
    std::vector<cv::Mat> a_channels;
    cv::split(a, a_channels);
    for (int i = 0; i < a.channels(); ++i) {
        cv::Mat channel_result;
        cv::compare(a_channels[i], b[i], channel_result, cmp_type);
        if (cv::countNonZero(channel_result) != a.total())
            return false;
    }

    return true;
}

bool multichannel_compare(const cv::Mat& a, double b, int cmp_type)
{
    std::vector<cv::Mat> a_channels(a.channels());
    cv::split(a, a_channels);
    for (int i = 0; i < a.channels(); ++i) {
        cv::Mat channel_result;
        cv::compare(a_channels[i], b, channel_result, cmp_type);
        if (cv::countNonZero(channel_result) != a.total())
            return false;
    }

    return true;
}

bool multichannel_float_equals(const cv::Mat& a, const cv::Mat& b, std::size_t num_ulps)
{
    if (a.size() != b.size() || a.channels() != b.channels())
        return false;

    if (a.empty() && b.empty())
        return true;

    cv::Mat double_a;
    a.convertTo(double_a, CV_64F);
    std::vector<cv::Mat> a_channels(a.channels());
    cv::split(double_a, a_channels);

    cv::Mat double_b;
    b.convertTo(double_b, CV_64F);
    std::vector<cv::Mat> b_channels(b.channels());
    cv::split(double_b, b_channels);

    for (int i = 0; i < a.channels(); ++i) {
        std::vector<double> a_values = a_channels[i].reshape(0, 1);
        std::vector<double> b_values = b_channels[i].reshape(0, 1);
        for (int j = 0; j < a_values.size(); ++j)
            if (!float_equals(a_values[j], b_values[j], num_ulps))
                return false;
    }

    return true;
}

bool multichannel_float_equals(const cv::Mat& a, const cv::Scalar& b, std::size_t num_ulps)
{
    if (a.empty())
        return false;

    cv::Mat double_a;
    a.convertTo(double_a, CV_64F);

    std::vector<cv::Mat> a_channels(a.channels());
    cv::split(double_a, a_channels);

    for (int i = 0; i < a.channels(); ++i) {
        std::vector<double> a_values = a_channels[i];
        for (int j = 0; j < a_values.size(); ++j)
            if (!float_equals(a_values[j], b[i], num_ulps))
                return false;
    }

    return true;
}

void print_matrix_difference_statistics(const cv::Mat& a, const cv::Mat& b, double tolerance, testing::MatchResultListener* result_listener)
{
    cv::Mat diff;
    cv::absdiff(a, b, diff);

    bool is_multichannel = (diff.channels() > 1);
    auto endline = is_multichannel ? "\n" : " ";
    *result_listener << "statistics are: " << endline;

    std::vector<cv::Mat> diff_channels(diff.channels());
    cv::split(diff, diff_channels);
    for (int i = 0; i < diff.channels(); ++i) {
        if (is_multichannel)
            *result_listener << "    channel " << i << ": ";

        auto diff_channel = diff_channels[i];
        cv::Mat zero_mask = diff_channel > 0;
        try {
            double min_above_tolerance, min_above_zero, max;
            cv::Point min_above_tolerance_point, min_above_zero_point, max_point;
            cv::minMaxLoc(
                diff_channel,
                &min_above_zero,
                &max,
                &min_above_zero_point,
                &max_point,
                diff_channel > 0
            );
            *result_listener << "max(|diff|) = " << max << " at " << max_point << ",  "
                << "min(|diff|) = " << min_above_zero << " at " << min_above_zero_point;

            if (0 < tolerance && tolerance <= max) {
                try {
                    cv::minMaxLoc(
                        diff_channel,
                        &min_above_tolerance,
                        &max,
                        &min_above_tolerance_point,
                        &max_point,
                        diff_channel > tolerance
                    );
                    *result_listener << ",  min(|diff > " << tolerance << "|) = "
                        << min_above_tolerance << " at " << min_above_tolerance_point;
                } catch (cv::Exception error) {
                }
            }
        } catch (cv::Exception error) {
            *result_listener << "max abs diff is " << 0;
        }

        *result_listener << endline;
    }
}

bool matrix_equals(const cv::Mat& a, const cv::Mat& b, testing::MatchResultListener* result_listener)
{
    if (a.empty() && b.empty())
        return true;

    if (result_listener) {
        *result_listener << "where ";
        print_matrix_difference_statistics(a, b, 0.0, result_listener);

        std::stringstream stream;
        cv::PrintTo(a - b, &stream);
        *result_listener << "the difference is\n" << stream.str();
    }

    return multichannel_compare(a, b, cv::CMP_EQ);
}

bool matrix_equals(const cv::Mat& a, const cv::Scalar& b, testing::MatchResultListener* result_listener)
{
    if (a.empty())
        return false;

    if (result_listener) {
        std::stringstream stream;
        cv::PrintTo(a - b, &stream);
        *result_listener << "where the difference is" << stream.str();
    }

    return multichannel_compare(a, b, cv::CMP_EQ);
}

bool matrix_float_equals(const cv::Mat& a, const cv::Mat& b, std::size_t num_ulps, testing::MatchResultListener* result_listener)
{
    if (a.empty() && b.empty())
        return true;

    if (result_listener) {
        *result_listener << "where ";
        print_matrix_difference_statistics(a, b, 0.0, result_listener);
        std::stringstream stream;
        cv::PrintTo(a - b, &stream);
        *result_listener << "the difference is" << stream.str();
    }

    return multichannel_float_equals(a, b, num_ulps);
}

bool matrix_float_equals(const cv::Mat& a, const cv::Scalar& b, std::size_t num_ulps, testing::MatchResultListener* result_listener)
{
    if (a.empty())
        return false;

    if (result_listener) {
        std::stringstream stream;
        cv::PrintTo(a - b, &stream);
        *result_listener << "where the difference is" << stream.str();
    }

    return multichannel_float_equals(a, b, num_ulps);
}

bool matrix_less_than(const cv::Mat& a, const cv::Scalar& b)
{
    return multichannel_compare(a, b, cv::CMP_LT);
}

bool matrix_less_than_or_equal(const cv::Mat& a, const cv::Scalar& b)
{
    return multichannel_compare(a, b, cv::CMP_LE);
}

bool matrix_greater_than(const cv::Mat& a, const cv::Scalar& b)
{
    return multichannel_compare(a, b, cv::CMP_GT);
}

bool matrix_greater_than_or_equal(const cv::Mat& a, const cv::Scalar& b)
{
    return multichannel_compare(a, b, cv::CMP_GE);
}

bool matrix_is_all_zeros(const cv::Mat& a)
{
    return cv::countNonZero(a == 0.0) == a.total();
}

bool matrix_near(const cv::Mat& a, const cv::Mat& b, float tolerance, testing::MatchResultListener* result_listener)
{
    if (a.size() != b.size() || a.channels() != b.channels())
        return false;

    if (tolerance <= 0)
        tolerance = std::numeric_limits<float>::epsilon();

    cv::Mat diff;
    cv::absdiff(a, b, diff);

    if (result_listener) {
        *result_listener << "where ";
        print_matrix_difference_statistics(a, b, tolerance, result_listener);
        std::stringstream stream;
        cv::PrintTo(a - b, &stream);
        *result_listener << "the difference is" << stream.str();
    }
    return multichannel_compare(diff, tolerance, cv::CMP_LE);
}

bool scalar_equals(const cv::Scalar& a, const cv::Scalar& b)
{
    return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] == b[3]);
}

bool scalar_double_equals(const cv::Scalar& a, const cv::Scalar& b, int num_ulps)
{
    return float_equals(a[0], b[0], num_ulps)
        && float_equals(a[1], b[1], num_ulps)
        && float_equals(a[2], b[2], num_ulps)
        && float_equals(a[3], b[3], num_ulps);
}

bool scalar_near(const cv::Scalar& a, const cv::Scalar& b, double tolerance)
{
    return std::fabs(a[0] - b[0]) <= tolerance
        && std::fabs(a[1] - b[1]) <= tolerance
        && std::fabs(a[2] - b[2]) <= tolerance
        && std::fabs(a[3] - b[3]) <= tolerance;
}

std::string classification_str(int classification)
{
    switch (classification)
    {
        case FP_INFINITE:
            return "Inf";
        case FP_NAN:
            return "NaN";
        case FP_NORMAL:
            return "normal";
        case FP_SUBNORMAL:
            return "subnormal";
        case FP_ZERO:
            return "zero";
        default:
            return "unknown";
    }
}

int ulps_between(float a, float b)
{
    static_assert(sizeof(float) == sizeof(int));
    if (!std::isnormal(a) || !std::isnormal(b))
        return -1;

    if (a == b)
        return 0;

    return abs(*(int*)&a - *(int*)&b);
}

int ulps_between(double a, double b)
{
    static_assert(sizeof(double) == sizeof(long));
    if (!std::isnormal(a) || !std::isnormal(b))
        return -1;

    if (a == b)
        return 0;

    return abs(*(long*)&a - *(long*)&b);
}



// bool equal_within_ulps(float a, float b, int num_ulps)
// {
//     static_assert(sizeof(float) == sizeof(int));

//     if (!std::isnormal(a))
//         a = 0.0;

//     if (!std::isnormal(b))
//         b = 0.0;

//     bool result = false;
//     if (std::isnan(a))
//         result = std::isnan(b);
//     else if (std::isnan(b))
//         result = std::isnan(a);
//     else if (std::isinf(a))
//         result = std::isinf(b) && std::signbit(a) == std::signbit(b);
//     else if (std::isinf(b))
//         result = std::isinf(a) && std::signbit(a) == std::signbit(b);
//     else
//         result = ulps(a, b) <= num_ulps;

//     if (!result) {
//         auto a_classification = std::fpclassify(a);
//         auto b_classification = std::fpclassify(b);
//         int ulps = abs(*(long*)&a - *(long*)&b);

//         std::cout << "a = " << a << " (" << classification_str(a_classification) << ")  "
//             << "b = " << b << " (" << classification_str(b_classification) << ")  "
//             << "min = " << std::numeric_limits<double>::min() << "  "
//             << "ulps = " << ulps << "  " << (result ? "true" : "false") << "\n";
//     }

//     return result;
// }


// bool equal_within_ulps(double a, double b, int num_ulps)
// {
//     static_assert(sizeof(double) == sizeof(long));

//     if (!std::isnormal(a))
//         a = 0.0;

//     if (!std::isnormal(b))
//         b = 0.0;

//     bool result = false;
//     if (std::isnan(a))
//         result = std::isnan(b);
//     else if (std::isnan(b))
//         result = std::isnan(a);
//     else if (std::isinf(a))
//         result = std::isinf(b) && std::signbit(a) == std::signbit(b);
//     else if (std::isinf(b))
//         result = std::isinf(a) && std::signbit(a) == std::signbit(b);
//     else
//         result = ulps(a, b) <= num_ulps;

//     if (!result) {
//         auto a_classification = std::fpclassify(a);
//         auto b_classification = std::fpclassify(b);
//         int ulps = abs(*(long*)&a - *(long*)&b);

//         std::cout << "a = " << a << " (" << classification_str(a_classification) << ")  "
//             << "b = " << b << " (" << classification_str(b_classification) << ")  "
//             << "min = " << std::numeric_limits<double>::min() << "  "
//             << "ulps = " << ulps << "  " << (result ? "true" : "false") << "\n";
//     }

//     return result;
// }

