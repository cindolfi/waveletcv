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

bool multichannel_equal_within_ulps(const cv::Mat& a, const cv::Mat& b, std::size_t num_ulps)
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
        std::vector<double> a_values = a_channels[i];
        std::vector<double> b_values = b_channels[i];
        for (int j = 0; j < a_values.size(); ++j)
            if (!equal_within_ulps(a_values[j], b_values[j], num_ulps))
                return false;
    }

    return true;
}

bool multichannel_equal_within_ulps(const cv::Mat& a, const cv::Scalar& b, std::size_t num_ulps)
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
            if (!equal_within_ulps(a_values[j], b[i], num_ulps))
                return false;
    }

    return true;
}

bool matrix_equals(const cv::Mat& a, const cv::Mat& b, testing::MatchResultListener* result_listener)
{
    if (a.empty() && b.empty())
        return true;

    if (result_listener)
        *result_listener << "where the difference is\n" << cv::Mat(a - b);

    return multichannel_compare(a, b, cv::CMP_EQ);
}

bool matrix_equals(const cv::Mat& a, const cv::Scalar& b, testing::MatchResultListener* result_listener)
{
    if (a.empty())
        return false;

    if (result_listener)
        *result_listener << "where the difference is\n" << cv::Mat(a - b);

    return multichannel_compare(a, b, cv::CMP_EQ);
}

bool matrix_float_equals(const cv::Mat& a, const cv::Mat& b, std::size_t num_ulps, testing::MatchResultListener* result_listener)
{
    if (a.empty() && b.empty())
        return true;

    if (result_listener)
        *result_listener << "where the difference is\n" << cv::Mat(a - b);

    return multichannel_equal_within_ulps(a, b, num_ulps);
}

bool matrix_float_equals(const cv::Mat& a, const cv::Scalar& b, std::size_t num_ulps, testing::MatchResultListener* result_listener)
{
    if (a.empty())
        return false;

    if (result_listener)
        *result_listener << "where the difference is\n" << cv::Mat(a - b);

    return multichannel_equal_within_ulps(a, b, num_ulps);
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

bool matrix_near(const cv::Mat& a, const cv::Mat& b, float tolerance)
{
    if (a.size() != b.size() || a.channels() != b.channels())
        return false;

    if (tolerance <= 0)
        tolerance = std::numeric_limits<float>::epsilon();

    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return multichannel_compare(diff, tolerance, cv::CMP_LE);
}

bool scalar_equals(const cv::Scalar& a, const cv::Scalar& b)
{
    return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] == b[3]);
}

bool scalar_double_equals(const cv::Scalar& a, const cv::Scalar& b, int num_ulps)
{
    return equal_within_ulps(a[0], b[0], num_ulps)
        && equal_within_ulps(a[1], b[1], num_ulps)
        && equal_within_ulps(a[2], b[2], num_ulps)
        && equal_within_ulps(a[3], b[3], num_ulps);
}


bool scalar_near(const cv::Scalar& a, const cv::Scalar& b, double tolerance)
{
    return a[0] - b[0] <= tolerance
        && a[1] - b[1] <= tolerance
        && a[2] - b[2] <= tolerance
        && a[3] - b[3] <= tolerance;
}

bool equal_within_ulps(float a, float b, int num_ulps)
{
    assert(sizeof(float) == sizeof(int));
    if (a == b)
        return true;

    return abs(*(int*)&a - *(int*)&b) <= num_ulps;
}

bool equal_within_ulps(double a, double b, int num_ulps)
{
    assert(sizeof(double) == sizeof(long));
    if (a == b)
        return true;

    return abs(*(long*)&a - *(long*)&b) <= num_ulps;
}

