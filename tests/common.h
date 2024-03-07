/**
 * Common Helpers & Utilitites
*/
#include <experimental/iterator>
#include <iomanip>
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


template<template<typename T, int N> class Functor>
void dispatch_on_pixel_type(int type, auto&&... args)
{
    switch (type) {
        case CV_32FC1: Functor<float, 1>()(std::forward<decltype(args)>(args)...); return;
        case CV_32FC2: Functor<float, 2>()(std::forward<decltype(args)>(args)...); return;
        case CV_32FC3: Functor<float, 3>()(std::forward<decltype(args)>(args)...); return;
        case CV_32FC4: Functor<float, 4>()(std::forward<decltype(args)>(args)...); return;

        case CV_64FC1: Functor<double, 1>()(std::forward<decltype(args)>(args)...); return;
        case CV_64FC2: Functor<double, 2>()(std::forward<decltype(args)>(args)...); return;
        case CV_64FC3: Functor<double, 3>()(std::forward<decltype(args)>(args)...); return;
        case CV_64FC4: Functor<double, 4>()(std::forward<decltype(args)>(args)...); return;

        case CV_32SC1: Functor<int, 1>()(std::forward<decltype(args)>(args)...); return;
        case CV_32SC2: Functor<int, 2>()(std::forward<decltype(args)>(args)...); return;
        case CV_32SC3: Functor<int, 3>()(std::forward<decltype(args)>(args)...); return;
        case CV_32SC4: Functor<int, 4>()(std::forward<decltype(args)>(args)...); return;

        case CV_16SC1: Functor<short, 1>()(std::forward<decltype(args)>(args)...); return;
        case CV_16SC2: Functor<short, 2>()(std::forward<decltype(args)>(args)...); return;
        case CV_16SC3: Functor<short, 3>()(std::forward<decltype(args)>(args)...); return;
        case CV_16SC4: Functor<short, 4>()(std::forward<decltype(args)>(args)...); return;

        case CV_16UC1: Functor<ushort, 1>()(std::forward<decltype(args)>(args)...); return;
        case CV_16UC2: Functor<ushort, 2>()(std::forward<decltype(args)>(args)...); return;
        case CV_16UC3: Functor<ushort, 3>()(std::forward<decltype(args)>(args)...); return;
        case CV_16UC4: Functor<ushort, 4>()(std::forward<decltype(args)>(args)...); return;

        case CV_8SC1: Functor<int8_t, 1>()(std::forward<decltype(args)>(args)...); return;
        case CV_8SC2: Functor<int8_t, 2>()(std::forward<decltype(args)>(args)...); return;
        case CV_8SC3: Functor<int8_t, 3>()(std::forward<decltype(args)>(args)...); return;
        case CV_8SC4: Functor<int8_t, 4>()(std::forward<decltype(args)>(args)...); return;

        case CV_8UC1: Functor<uint8_t, 1>()(std::forward<decltype(args)>(args)...); return;
        case CV_8UC2: Functor<uint8_t, 2>()(std::forward<decltype(args)>(args)...); return;
        case CV_8UC3: Functor<uint8_t, 3>()(std::forward<decltype(args)>(args)...); return;
        case CV_8UC4: Functor<uint8_t, 4>()(std::forward<decltype(args)>(args)...); return;
    }
}



template<typename T, int CHANNELS>
struct print_matrix_to
{
    void operator()(const cv::Mat& matrix, std::ostream* stream) const
    {
        using Pixel = cv::Vec<T, CHANNELS>;
        *stream << std::endl;

        double min;
        double max;
        cv::minMaxIdx(matrix, &min, &max);
        auto max_abs_value = std::max(std::fabs(min), std::fabs(max));
        int width = 1;
        if (max_abs_value > 0)
            width = 1 + std::floor(std::log10(max_abs_value));

        ++width;
        //  make room for negative sign
        if (min < 0)
            ++width;

        //  a little breathing room for single channel matrices
        if (matrix.channels() == 1)
            ++width;

        for (int row = 0; row < matrix.rows; ++row) {
            for (int col = 0; col < matrix.cols; ++col) {
                auto pixel = matrix.at<Pixel>(row, col);

                if (matrix.channels() == 1) {
                    *stream << std::setfill(' ')
                            << std::setw(width)
                            << std::right
                            << std::setprecision(3)
                            << +pixel[0];
                } else {
                    *stream << "|";
                    for (int i = 0 ; i < pixel.channels; ++i) {
                        *stream << std::setfill(' ')
                                << std::setw(width)
                                << std::right
                                << std::setprecision(3)
                                << +pixel[i];
                    }
                }
            }
            if (matrix.channels() != 1)
                *stream << "|";
            *stream << std::endl;
        }
    }
};


cv::Mat create_matrix(int rows, int cols, int type, double initial_value = 0.0);
void print_matrix(const cv::Mat& matrix, float zero_clamp=1e-7);

bool matrix_equals(const cv::Mat& a, const cv::Mat& b);
bool matrix_equals(const cv::Mat& a, const cv::Scalar& b);
MATCHER_P(MatrixEq, other, "") { return matrix_equals(arg, other); }

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

template <class T>
std::enable_if_t<not std::numeric_limits<T>::is_integer, bool>
equal_within_ulps(T x, T y, std::size_t n)
{
    // Since `epsilon()` is the gap size (ULP, unit in the last place)
    // of floating-point numbers in interval [1, 2), we can scale it to
    // the gap size in interval [2^e, 2^{e+1}), where `e` is the exponent
    // of `x` and `y`.

    // If `x` and `y` have different gap sizes (which means they have
    // different exponents), we take the smaller one. Taking the bigger
    // one is also reasonable, I guess.
    const T m = std::min(std::fabs(x), std::fabs(y));

    // Subnormal numbers have fixed exponent, which is `min_exponent - 1`.
    const int exp = m < std::numeric_limits<T>::min()
                  ? std::numeric_limits<T>::min_exponent - 1
                  : std::ilogb(m);

    // We consider `x` and `y` equal if the difference between them is
    // within `n` ULPs.
    return std::fabs(x - y) <= n * std::ldexp(std::numeric_limits<T>::epsilon(), exp);
}

MATCHER_P(IsMatrixMax, value, "") { return is_matrix_max(arg, value, result_listener); }
MATCHER_P2(IsMaskedMatrixMax, value, mask, "") { return is_matrix_max(arg, value, result_listener, mask); }

bool scalar_equals(const cv::Scalar& a, const cv::Scalar& b);
MATCHER_P(ScalarEq, other, "") { return scalar_equals(arg, other); }
MATCHER(ScalarEq, "") { return scalar_equals(std::get<0>(arg), std::get<1>(arg)); }

bool scalar_double_equals(const cv::Scalar& a, const cv::Scalar& b, int num_ulps=4);
MATCHER_P(ScalarDoubleEq, other, "") { return scalar_double_equals(arg, other); }

bool scalar_near(const cv::Scalar& a, const cv::Scalar& b, double tolerance=1e-10);
MATCHER_P2(ScalarNear, other, tolerance, "") { return scalar_near(arg, other, tolerance); }





