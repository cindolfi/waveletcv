#ifndef WAVELET_TEST_COMMON_HPP
#define WAVELET_TEST_COMMON_HPP
/**
 * Common Helpers & Utilitites
*/
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <experimental/iterator>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <opencv2/core.hpp>
#include <wavelet/wavelet.hpp>
#include <wavelet/dwt2d.hpp>

namespace wavelet::internal
{
void PrintTo(const DWT2D::Coeffs& coeffs, std::ostream* stream);
}

namespace cv
{
void PrintTo(const cv::Mat& matrix, std::ostream* stream);
}

void clamp_near_zero(cv::InputArray input, cv::OutputArray output, double tolerance=1e-6);
void clamp_near_zero(cv::InputOutputArray array, double tolerance=1e-6);

template<typename T>
std::string join(std::vector<T> items, std::string delim)
{
    std::stringstream stream;
    std::copy(
        items.begin(),
        items.end(),
        std::experimental::make_ostream_joiner(stream, delim)
    );

    return stream.str();
}

template<typename T, int CHANNELS>
struct print_matrix_to
{
    using Pixel = cv::Vec<T, CHANNELS>;
    const int precision = 3;
    const int row_start_items = 7;
    const int row_end_items = 7;
    const int column_start_items = 7;
    const int column_end_items = 7;
    const int truncate_rows_limit = 16;
    const int truncate_cols_limit = 16;

    bool is_row_hidden(const cv::Mat& matrix, int row) const
    {
        return matrix.rows > truncate_rows_limit
            && row > row_start_items
            && row < matrix.rows - row_end_items;
    }

    bool is_row_ellipsis(const cv::Mat& matrix, int row) const
    {
        return matrix.rows > truncate_rows_limit && row == row_start_items;
    }

    bool is_column_hidden(const cv::Mat& matrix, int col) const
    {
        return matrix.cols > truncate_cols_limit
            && col > column_start_items
            && col < matrix.cols - column_end_items;
    }

    bool is_column_ellipsis(const cv::Mat& matrix, int col) const
    {
        return matrix.cols > truncate_cols_limit && col == column_start_items;
    }

    void operator()(const cv::Mat& matrix, std::ostream* stream) const
    {
        cv::Mat widths = calculate_widths(matrix);

        *stream << std::endl;
        for (int row = 0; row < matrix.rows; ++row) {
            if (is_row_ellipsis(matrix, row)) {
                *stream << ".\n.\n.";
            } else if (!is_row_hidden(matrix, row)) {
                for (int col = 0; col < matrix.cols; ++col) {
                    if (is_column_ellipsis(matrix, col)) {
                        *stream << " ... ";
                    } else if (!is_column_hidden(matrix, col)) {
                        if (matrix.channels() > 1)
                            *stream << "|";

                        auto pixel = matrix.at<Pixel>(row, col);
                        for (int k = 0 ; k < pixel.channels; ++k) {
                            *stream << std::setfill(' ')
                                    << std::setw((col > 0) + widths.at<int>(col, k))
                                    << std::right
                                    << std::setprecision(precision)
                                    << +pixel[0];
                        }
                    }
                }

                if (matrix.channels() > 1)
                    *stream << "|";

                *stream << "\n";
            }
        }
    }

    cv::Mat calculate_widths(const cv::Mat& matrix) const
    {
        auto calc_string_length = [&](auto value) -> int {
            return (
                std::stringstream()
                    << std::setprecision(precision)
                    << value
                ).str().size();
        };

        cv::Mat widths(matrix.cols, matrix.channels(), CV_MAKE_TYPE(CV_32S, matrix.channels()));

        int single_channel_min_width = 1;
        for (int i = 0; i < matrix.cols; ++i) {
            auto column = matrix.col(i);
            std::vector<cv::Mat> channels;
            cv::split(column, channels);
            for (int j = 0; j < channels.size(); ++j) {
                double abs_min, abs_max;
                cv::minMaxIdx(cv::abs(channels[j]), &abs_min, &abs_max);
                double min, max;
                cv::minMaxIdx(channels[j], &min, &max);

                int width = 0;
                width = std::max(width, calc_string_length(abs_min));
                width = std::max(width, calc_string_length(abs_max));
                width = std::max(width, calc_string_length(min));
                width = std::max(width, calc_string_length(max));
                //  this gives columns of all zeros, ones, etc some extra padding
                //  to make them stand out in matrices with wide columns
                if (width >= 2)
                    single_channel_min_width = 2;

                widths.at<int>(i, j) = width;
            }
        }

        if (matrix.channels() == 1)
            cv::max(widths, single_channel_min_width, widths);

        return widths;
    }
};

cv::Mat create_matrix(int rows, int cols, int type, double initial_value = 0.0);

bool matrix_equals(const cv::Mat& a, const cv::Mat& b, testing::MatchResultListener* result_listener = nullptr);
bool matrix_equals(const cv::Mat& a, const cv::Scalar& b, testing::MatchResultListener* result_listener = nullptr);
MATCHER_P(MatrixEq, other, "") { return matrix_equals(arg, other, result_listener); }

bool matrix_float_equals(const cv::Mat& a, const cv::Mat& b, std::size_t nulps = 4, testing::MatchResultListener* result_listener = nullptr);
bool matrix_float_equals(const cv::Mat& a, const cv::Scalar& b, std::size_t nulps = 4, testing::MatchResultListener* result_listener = nullptr);
MATCHER_P(MatrixFloatEq, other, "") { return matrix_float_equals(arg, other, 4, result_listener); }
MATCHER_P2(MatrixFloatEq, other, nulps, "") { return matrix_float_equals(arg, other, nulps, result_listener); }

bool matrix_less_than(const cv::Mat& a, const cv::Scalar& b);
bool matrix_less_than_or_equal(const cv::Mat& a, const cv::Scalar& b);
MATCHER_P(MatrixLessOrEqual, other, "") { return matrix_less_than_or_equal(arg, other); }

bool matrix_greater_than(const cv::Mat& a, const cv::Scalar& b);
bool matrix_greater_than_or_equal(const cv::Mat& a, const cv::Scalar& b);
MATCHER_P(MatrixGreaterOrEqual, other, "") { return matrix_greater_than_or_equal(arg, other); }

bool matrix_is_all_zeros(const cv::Mat& a);
bool matrix_near(const cv::Mat& a, const cv::Mat& b, float tolerance=0.0);
MATCHER_P2(MatrixNear, other, error, "") { return matrix_near(arg, other, error); }

template <typename ResultListener>
bool is_matrix_min(
    const cv::Mat& matrix,
    double value,
    ResultListener* result_listener,
    const cv::Mat& mask=cv::noArray()
)
{
    auto min = std::numeric_limits<double>::max();
    auto max = std::numeric_limits<double>::min();
    cv::minMaxIdx(matrix, &min, &max, nullptr, nullptr, mask);

    *result_listener << "where the actual min is " << min;

    double tolerance = std::numeric_limits<double>::epsilon();
    if (matrix.type() == CV_32F)
        tolerance = std::numeric_limits<float>::epsilon();

    return std::fabs(min - value) <= tolerance;
}
MATCHER_P(IsMatrixMin, value, "") { return is_matrix_min(arg, value, result_listener); }
MATCHER_P2(IsMaskedMatrixMin, value, mask, "") { return is_matrix_min(arg, value, result_listener, mask); }

template <typename ResultListener>
bool is_matrix_max(
    const cv::Mat& matrix,
    double value,
    ResultListener* result_listener,
    const cv::Mat& mask=cv::noArray()
)
{
    auto min = std::numeric_limits<double>::max();
    auto max = std::numeric_limits<double>::min();
    cv::minMaxIdx(matrix, &min, &max, nullptr, nullptr, mask);

    *result_listener << "where the actual max is " << max;

    double tolerance = std::numeric_limits<double>::epsilon();
    if (matrix.type() == CV_32F)
        tolerance = std::numeric_limits<float>::epsilon();

    return std::fabs(max - value) <= tolerance;
}


bool equal_within_ulps(float a, float b, int nulps);
bool equal_within_ulps(double a, double b, int nulps);

MATCHER_P(IsMatrixMax, value, "") { return is_matrix_max(arg, value, result_listener); }
MATCHER_P2(IsMaskedMatrixMax, value, mask, "") { return is_matrix_max(arg, value, result_listener, mask); }

bool scalar_equals(const cv::Scalar& a, const cv::Scalar& b);
MATCHER_P(ScalarEq, other, "") { return scalar_equals(arg, other); }
MATCHER(ScalarEq, "") { return scalar_equals(std::get<0>(arg), std::get<1>(arg)); }

bool scalar_double_equals(const cv::Scalar& a, const cv::Scalar& b, int num_ulps=4);
MATCHER_P(ScalarDoubleEq, other, "") { return scalar_double_equals(arg, other); }

bool scalar_near(const cv::Scalar& a, const cv::Scalar& b, double tolerance=1e-10);
MATCHER_P2(ScalarNear, other, tolerance, "") { return scalar_near(arg, other, tolerance); }

#endif  // WAVELET_TEST_COMMON_HPP

