/**
*/
#include <numeric>
#include <wavelet/dwt2d.hpp>
#include <wavelet/utils.hpp>
#include "common.hpp"

using namespace testing;

namespace wavelet::internal
{
void PrintTo(const DWT2D::Coeffs& coeffs, std::ostream* stream)
{
    wavelet::internal::dispatch_on_pixel_type<print_matrix_to>(
        coeffs.type(),
        coeffs,
        stream
    );
}
}   // namespace wavelet::internal

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

bool matrix_equals(const cv::Mat& a, const cv::Mat& b)
{
    if (a.empty() && b.empty())
        return true;

    return multichannel_compare(a, b, cv::CMP_EQ);
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

