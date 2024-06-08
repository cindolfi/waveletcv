#include "cvwt/dwt2d.hpp"
#include "cvwt/exception.hpp"
#include "cvwt/utils.hpp"

namespace cvwt
{
void CoeffsExpr::throw_if_incompatible(
    const DWT2D::Coeffs& coeffs_a,
    const DWT2D::Coeffs& coeffs_b
)
{
    if (coeffs_a.levels() != coeffs_b.levels())
        throw_bad_arg(
            "Incompatible DWT2D coefficients. "
            "Must have the same levels(), "
            "got lhs levels() = ", coeffs_a.levels(),
            " and rhs levels() = ", coeffs_b.levels(), "."
        );

    if (coeffs_a.image_size() != coeffs_b.image_size())
        throw_bad_arg(
            "Incompatible DWT2D coefficients. "
            "Must have the same image_size(), "
            "got lhs image_size() = ", coeffs_a.image_size(),
            " and rhs image_size() = ", coeffs_b.image_size(), "."
        );

    if (coeffs_a.wavelet() != coeffs_b.wavelet())
        throw_bad_arg(
            "Incompatible DWT2D coefficients. "
            "Must have the same wavelet(), "
            "got lhs wavelet() = ", coeffs_a.wavelet(),
            " and rhs wavelet() = ", coeffs_b.wavelet(), "."
        );

    if (coeffs_a.border_type() != coeffs_b.border_type())
        throw_bad_arg(
            "Incompatible DWT2D coefficients. "
            "Must have the same border_type(), "
            "got lhs border_type() = ", coeffs_a.border_type(),
            " and rhs border_type() = ", coeffs_b.border_type(), "."
        );
}

//  ============================================================================
//  Arithmetic
//  ============================================================================
//  ----------------------------------------------------------------------------
//  negation
CoeffsExpr operator-(const DWT2D::Coeffs& coeffs)
{
    return CoeffsExpr(
        coeffs,
        -static_cast<const cv::Mat&>(coeffs)
    );
}

CoeffsExpr operator-(const CoeffsExpr& expression)
{
    return CoeffsExpr(
        expression.coeffs,
        -static_cast<const cv::MatExpr&>(expression)
    );
}

//  ----------------------------------------------------------------------------
//  addition
CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        lhs,
        rhs,
        static_cast<const cv::Mat&>(lhs) + static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator+(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        rhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) + static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator+(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        rhs,
        static_cast<const cv::MatExpr&>(lhs) + static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator+(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) + rhs
    );
}

CoeffsExpr operator+(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) + rhs
    );
}

CoeffsExpr operator+(const CoeffsExpr& lhs, const cv::Scalar& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) + rhs
    );
}

CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) + rhs
    );
}

CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) + rhs
    );
}

CoeffsExpr operator+(const DWT2D::Coeffs& lhs, const cv::Scalar& rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) + rhs
    );
}

//  ----------------------------------------------------------------------------
//  subtraction
CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        lhs,
        rhs,
        static_cast<const cv::Mat&>(lhs) - static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator-(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        rhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) - static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator-(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        rhs,
        static_cast<const cv::MatExpr&>(lhs) - static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        lhs,
        rhs.coeffs,
        static_cast<const cv::Mat&>(lhs) - static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator-(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) - rhs
    );
}

CoeffsExpr operator-(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        rhs.coeffs,
        lhs - static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator-(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) - rhs
    );
}

CoeffsExpr operator-(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        rhs.coeffs,
        lhs - static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator-(const CoeffsExpr& lhs, const cv::Scalar& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) - rhs
    );
}

CoeffsExpr operator-(const cv::Scalar& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        rhs.coeffs,
        lhs - static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) - rhs
    );
}

CoeffsExpr operator-(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        rhs,
        lhs - static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) - rhs
    );
}

CoeffsExpr operator-(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        rhs,
        lhs - static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator-(const DWT2D::Coeffs& lhs, const cv::Scalar& rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) - rhs
    );
}

CoeffsExpr operator-(const cv::Scalar& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        rhs,
        lhs - static_cast<const cv::Mat&>(rhs)
    );
}

//  ----------------------------------------------------------------------------
//  multiplication
CoeffsExpr operator*(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    return lhs.mul(rhs);
}

CoeffsExpr operator*(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    return lhs.mul(rhs);
}

CoeffsExpr operator*(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return lhs.mul(rhs);
}

CoeffsExpr operator*(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    return lhs.mul(rhs);
}

CoeffsExpr operator*(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    return lhs.mul(rhs);
}

CoeffsExpr operator*(const CoeffsExpr& lhs, double rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) * rhs
    );
}

CoeffsExpr operator*(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    return lhs.mul(rhs);
}

CoeffsExpr operator*(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    return lhs.mul(rhs);
}

CoeffsExpr operator*(const DWT2D::Coeffs& lhs, double rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) * rhs
    );
}

//  ----------------------------------------------------------------------------
//  division
CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        lhs,
        rhs,
        static_cast<const cv::Mat&>(lhs) / static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator/(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        rhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) / static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator/(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        rhs,
        static_cast<const cv::MatExpr&>(lhs) / static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        lhs,
        rhs.coeffs,
        static_cast<const cv::Mat&>(lhs) / static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator/(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) / rhs
    );
}

CoeffsExpr operator/(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        rhs.coeffs,
        lhs / static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator/(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) / rhs
    );
}

CoeffsExpr operator/(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        rhs.coeffs,
        lhs / static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator/(const CoeffsExpr& lhs, double rhs)
{
    return CoeffsExpr(
        lhs.coeffs,
        static_cast<const cv::MatExpr&>(lhs) / rhs
    );
}

CoeffsExpr operator/(double lhs, const CoeffsExpr& rhs)
{
    return CoeffsExpr(
        rhs.coeffs,
        lhs / static_cast<const cv::MatExpr&>(rhs)
    );
}

CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) / rhs
    );
}

CoeffsExpr operator/(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        rhs,
        lhs / static_cast<const cv::Mat&>(rhs)
    );
}

CoeffsExpr operator/(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) / rhs
    );
}

CoeffsExpr operator/(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        rhs,
        lhs / static_cast<const cv::Mat&>(rhs)
    );
}


CoeffsExpr operator/(const DWT2D::Coeffs& lhs, double rhs)
{
    return CoeffsExpr(
        lhs,
        static_cast<const cv::Mat&>(lhs) / rhs
    );
}

CoeffsExpr operator/(double lhs, const DWT2D::Coeffs& rhs)
{
    return CoeffsExpr(
        rhs,
        lhs / static_cast<const cv::Mat&>(rhs)
    );
}


//  ============================================================================
//  Compare
//  ============================================================================
namespace internal
{
CompareOp global_compare_op;

void CompareOp::assign(const cv::MatExpr& expression, cv::Mat& destination, int _type) const
{
    cv::Mat temp;
    cv::Mat& result = (_type == -1 || _type == CV_8U) ? destination : temp;

    auto cmp_type = static_cast<cv::CmpTypes>(expression.flags & 0xFF);

    if (expression.flags & A_IS_SCALAR) {
        std::vector<double> scalar = expression.a;
        cvwt::compare(
            scalar,
            expression.b,
            result,
            cmp_type
        );
    } else if (expression.flags & B_IS_SCALAR) {
        std::vector<double> scalar = expression.b;
        cvwt::compare(
            expression.a,
            scalar,
            result,
            cmp_type
        );
    } else {
        cvwt::compare(expression.a, expression.b, result, cmp_type);
    }

    if (result.data != destination.data)
        result.convertTo(destination, _type);
}

void CompareOp::make_expression(
    cv::MatExpr& expression,
    cv::CmpTypes cmp_type,
    const cv::Mat& a,
    const cv::Mat& b,
    int flags
)
{
    expression = cv::MatExpr(&global_compare_op, int(cmp_type) | flags, a, b);
}

void CompareOp::make_expression(
    cv::MatExpr& expression,
    cv::CmpTypes cmp_type,
    const cv::Mat& a,
    double b
)
{
    expression = cv::MatExpr(
        &global_compare_op,
        int(cmp_type) | B_IS_SCALAR,
        a,
        (cv::Mat_<double>(1, 1) << b)
    );
}

void CompareOp::make_expression(
    cv::MatExpr& expression,
    cv::CmpTypes cmp_type,
    double a,
    const cv::Mat& b
)
{
    expression = cv::MatExpr(
        &global_compare_op,
        int(cmp_type) | A_IS_SCALAR,
        (cv::Mat_<double>(1, 1) << a),
        b
    );
}
}   // namespace internal

//  ----------------------------------------------------------------------------
//  equal
cv::MatExpr operator==(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs, rhs);
    return result;
}

cv::MatExpr operator==(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs.coeffs, rhs.coeffs);
    return result;
}

cv::MatExpr operator==(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs.coeffs, rhs);
    return result;
}

cv::MatExpr operator==(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs.coeffs, rhs);
    return result;
}

cv::MatExpr operator==(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs.coeffs, rhs);
    return result;
}

cv::MatExpr operator==(const CoeffsExpr& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs.coeffs, rhs);
    return result;
}

cv::MatExpr operator==(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs, rhs);
    return result;
}

cv::MatExpr operator==(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs, rhs);
    return result;
}

cv::MatExpr operator==(const DWT2D::Coeffs& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_EQ, lhs, rhs);
    return result;
}

//  ----------------------------------------------------------------------------
//  not equal
cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs, rhs);
    return result;
}

cv::MatExpr operator!=(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs.coeffs, rhs.coeffs);
    return result;
}

cv::MatExpr operator!=(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs.coeffs, rhs);
    return result;
}

cv::MatExpr operator!=(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs.coeffs, rhs);
    return result;
}

cv::MatExpr operator!=(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs.coeffs, rhs);
    return result;
}

cv::MatExpr operator!=(const CoeffsExpr& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs.coeffs, rhs);
    return result;
}

cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs, rhs);
    return result;
}

cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs, rhs);
    return result;
}

cv::MatExpr operator!=(const DWT2D::Coeffs& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_NE, lhs, rhs);
    return result;
}

//  ----------------------------------------------------------------------------
//  less than
cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs, rhs);
    return result;
}

cv::MatExpr operator<(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs.coeffs, rhs.coeffs);
    return result;
}

cv::MatExpr operator<(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, rhs, lhs);
    return result;
}

cv::MatExpr operator<(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator<(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, rhs, lhs);
    return result;
}

cv::MatExpr operator<(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator<(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, rhs, lhs);
    return result;
}

cv::MatExpr operator<(const CoeffsExpr& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator<(double lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, rhs, lhs);
    return result;
}

cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs, rhs);
    return result;
}
cv::MatExpr operator<(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, rhs, lhs);
    return result;
}

cv::MatExpr operator<(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs, rhs);
    return result;
}
cv::MatExpr operator<(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, rhs, lhs);
    return result;
}

cv::MatExpr operator<(const DWT2D::Coeffs& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, lhs, rhs);
    return result;
}
cv::MatExpr operator<(double lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, rhs, lhs);
    return result;
}

//  ----------------------------------------------------------------------------
//  less than or equal
cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs, rhs);
    return result;
}

cv::MatExpr operator<=(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs.coeffs, rhs.coeffs);
    return result;
}

cv::MatExpr operator<=(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, rhs, lhs);
    return result;
}

cv::MatExpr operator<=(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator<=(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, rhs, lhs);
    return result;
}

cv::MatExpr operator<=(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator<=(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, rhs, lhs);
    return result;
}

cv::MatExpr operator<=(const CoeffsExpr& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator<=(double lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, rhs, lhs);
    return result;
}

cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs, rhs);
    return result;
}
cv::MatExpr operator<=(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, rhs, lhs);
    return result;
}

cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs, rhs);
    return result;
}
cv::MatExpr operator<=(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, rhs, lhs);
    return result;
}

cv::MatExpr operator<=(const DWT2D::Coeffs& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, lhs, rhs);
    return result;
}
cv::MatExpr operator<=(double lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, rhs, lhs);
    return result;
}

//  ----------------------------------------------------------------------------
//  greater than
cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs, rhs);
    return result;
}

cv::MatExpr operator>(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs.coeffs, rhs.coeffs);
    return result;
}

cv::MatExpr operator>(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, rhs, lhs);
    return result;
}

cv::MatExpr operator>(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator>(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, rhs, lhs);
    return result;
}

cv::MatExpr operator>(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator>(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, rhs, lhs);
    return result;
}

cv::MatExpr operator>(const CoeffsExpr& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator>(double lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, rhs, lhs);
    return result;
}

cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs, rhs);
    return result;
}
cv::MatExpr operator>(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, rhs, lhs);
    return result;
}

cv::MatExpr operator>(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs, rhs);
    return result;
}
cv::MatExpr operator>(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, rhs, lhs);
    return result;
}

cv::MatExpr operator>(const DWT2D::Coeffs& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GT, lhs, rhs);
    return result;
}
cv::MatExpr operator>(double lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LT, rhs, lhs);
    return result;
}

//  ----------------------------------------------------------------------------
//  greater than or equal
cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs, rhs);
    return result;
}

cv::MatExpr operator>=(const CoeffsExpr& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs.coeffs, rhs.coeffs);
    return result;
}

cv::MatExpr operator>=(const CoeffsExpr& lhs, const DWT2D::Coeffs& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs.coeffs, rhs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const CoeffsExpr& rhs)
{
    CoeffsExpr::throw_if_incompatible(lhs, rhs.coeffs);
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, rhs, lhs);
    return result;
}

cv::MatExpr operator>=(const CoeffsExpr& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator>=(const cv::MatExpr& lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, rhs, lhs);
    return result;
}

cv::MatExpr operator>=(const CoeffsExpr& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator>=(const cv::Mat& lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, rhs, lhs);
    return result;
}

cv::MatExpr operator>=(const CoeffsExpr& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs.coeffs, rhs);
    return result;
}
cv::MatExpr operator>=(double lhs, const CoeffsExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, rhs, lhs);
    return result;
}

cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const cv::MatExpr& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs, rhs);
    return result;
}
cv::MatExpr operator>=(const cv::MatExpr& lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, rhs, lhs);
    return result;
}

cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, const cv::Mat& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs, rhs);
    return result;
}
cv::MatExpr operator>=(const cv::Mat& lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, rhs, lhs);
    return result;
}

cv::MatExpr operator>=(const DWT2D::Coeffs& lhs, double rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_GE, lhs, rhs);
    return result;
}
cv::MatExpr operator>=(double lhs, const DWT2D::Coeffs& rhs)
{
    cv::MatExpr result;
    internal::CompareOp::make_expression(result, cv::CMP_LE, rhs, lhs);
    return result;
}
} // namespace cvwt


namespace cv
{
using namespace cvwt;

//  ----------------------------------------------------------------------------
//  abs
CoeffsExpr abs(const CoeffsExpr& expression)
{
    return CoeffsExpr(
        expression.coeffs,
        abs(static_cast<const MatExpr&>(expression))
    );
}

CoeffsExpr abs(const DWT2D::Coeffs& coeffs)
{
    return CoeffsExpr(
        coeffs,
        abs(static_cast<const Mat&>(coeffs))
    );
}

//  ----------------------------------------------------------------------------
//  max
CoeffsExpr max(const CoeffsExpr& a, const CoeffsExpr& b)
{
    return CoeffsExpr(
        a.coeffs,
        b.coeffs,
        max(static_cast<const Mat&>(a), static_cast<const Mat&>(b))
    );
}

CoeffsExpr max(const DWT2D::Coeffs& a, const DWT2D::Coeffs& b)
{
    return CoeffsExpr(
        a,
        b,
        max(static_cast<const Mat&>(a), static_cast<const Mat&>(b))
    );
}

CoeffsExpr max(const CoeffsExpr& a, const DWT2D::Coeffs& b)
{
    return CoeffsExpr(
        a.coeffs,
        b,
        max(static_cast<const MatExpr&>(a), static_cast<const Mat&>(b))
    );
}

CoeffsExpr max(const cvwt::CoeffsExpr& a, const cv::Mat& b)
{
    return CoeffsExpr(
        a.coeffs,
        max(static_cast<const MatExpr&>(a), b)
    );
}

CoeffsExpr max(const cvwt::CoeffsExpr& a, const cv::MatExpr& b)
{
    return CoeffsExpr(
        a.coeffs,
        max(static_cast<const MatExpr&>(a), static_cast<const cv::Mat&>(b))
    );
}

CoeffsExpr max(const cvwt::CoeffsExpr& a, double b)
{
    return CoeffsExpr(
        a.coeffs,
        max(static_cast<const MatExpr&>(a), b)
    );
}

CoeffsExpr max(const DWT2D::Coeffs& a, const Mat& b)
{
    return CoeffsExpr(
        a,
        max(static_cast<const Mat&>(a), b)
    );
}

CoeffsExpr max(const DWT2D::Coeffs& a, const MatExpr& b)
{
    return CoeffsExpr(
        a,
        max(static_cast<const Mat&>(a), b)
    );
}

CoeffsExpr max(const DWT2D::Coeffs& a, double b)
{
    return CoeffsExpr(
        a,
        max(static_cast<const Mat&>(a), b)
    );
}

//  ----------------------------------------------------------------------------
//  min
CoeffsExpr min(const CoeffsExpr& a, const CoeffsExpr& b)
{
    return CoeffsExpr(
        a.coeffs,
        b.coeffs,
        min(static_cast<const Mat&>(a), static_cast<const Mat&>(b))
    );
}

CoeffsExpr min(const DWT2D::Coeffs& a, const DWT2D::Coeffs& b)
{
    return CoeffsExpr(
        a,
        b,
        min(static_cast<const Mat&>(a), static_cast<const Mat&>(b))
    );
}

CoeffsExpr min(const CoeffsExpr& a, const DWT2D::Coeffs& b)
{
    return CoeffsExpr(
        a.coeffs,
        b,
        min(static_cast<const MatExpr&>(a), static_cast<const Mat&>(b))
    );
}

CoeffsExpr min(const cvwt::CoeffsExpr& a, const cv::Mat& b)
{
    return CoeffsExpr(
        a.coeffs,
        min(static_cast<const MatExpr&>(a), b)
    );
}

CoeffsExpr min(const cvwt::CoeffsExpr& a, const cv::MatExpr& b)
{
    return CoeffsExpr(
        a.coeffs,
        min(static_cast<const MatExpr&>(a), static_cast<const cv::Mat&>(b))
    );
}

CoeffsExpr min(const cvwt::CoeffsExpr& a, double b)
{
    return CoeffsExpr(
        a.coeffs,
        min(static_cast<const MatExpr&>(a), b)
    );
}

CoeffsExpr min(const DWT2D::Coeffs& a, const Mat& b)
{
    return CoeffsExpr(
        a,
        min(static_cast<const Mat&>(a), b)
    );
}

CoeffsExpr min(const DWT2D::Coeffs& a, const MatExpr& b)
{
    return CoeffsExpr(
        a,
        min(static_cast<const Mat&>(a), b)
    );
}

CoeffsExpr min(const DWT2D::Coeffs& a, double b)
{
    return CoeffsExpr(
        a,
        min(static_cast<const Mat&>(a), b)
    );
}
} // namespace cv

