/**
 * DWT2D Coefficient Expressions Unit Tests
*/
// #include <vector>
// #include <sstream>
// #include <algorithm>
#include <cvwt/dwt2d.hpp>
// #include <cvwt/compare.hpp>
// #include <cvwt/utils.hpp>
#include "common.hpp"
// #include "base_dwt2d.hpp"

using namespace cvwt;
using namespace testing;

enum Operation
{
    ADD,
    SUBTRACT,
    MULTIPLY,
    MULTIPLY_METHOD,
    DIVIDE,
    EQUAL,
    NOT_EQUAL,
    LESS_THAN,
    LESS_THAN_OR_EQUAL,
    GREATER_THAN,
    GREATER_THAN_OR_EQUAL,
    ABS,
    MIN,
    MAX,
    NEGATE,
};

enum CoeffCompatibility
{
    COMPATIBLE,
    INCOMPATIBLE,
};

std::string to_string(Operation op)
{
    switch (op)
    {
    case Operation::ADD: return "ADD";
    case Operation::SUBTRACT: return "SUBTRACT";
    case Operation::MULTIPLY: return "MULTIPLY";
    case Operation::MULTIPLY_METHOD: return "MULTIPLY_METHOD";
    case Operation::DIVIDE: return "DIVIDE";
    case Operation::EQUAL: return "EQUAL";
    case Operation::NOT_EQUAL: return "NOT_EQUAL";
    case Operation::LESS_THAN: return "LESS_THAN";
    case Operation::LESS_THAN_OR_EQUAL: return "LESS_THAN_OR_EQUAL";
    case Operation::GREATER_THAN: return "GREATER_THAN";
    case Operation::GREATER_THAN_OR_EQUAL: return "GREATER_THAN_OR_EQUAL";
    case Operation::ABS: return "ABS";
    case Operation::MIN: return "MIN";
    case Operation::MAX: return "MAX";
    case Operation::NEGATE: return "NEGATE";
    }

    return "";
}

std::string to_string(CoeffCompatibility compatibility)
{
    switch (compatibility)
    {
    case CoeffCompatibility::COMPATIBLE: return "COMPATIBLE";
    case CoeffCompatibility::INCOMPATIBLE: return "INCOMPATIBLE";
    }

    return "";
}

auto print_expression_test_label = [](const auto& info)
{
    return to_string(std::get<0>(info.param)) + "__" + to_string(std::get<1>(info.param));
};

template <class DerivedTest>
class CoeffsExpressionTest : public testing::TestWithParam<std::tuple<Operation, CoeffCompatibility>>
{
public:
    using Pixel = cv::Scalar;
    using FixedMatrix = cv::Matx<Pixel, 8, 4>;

protected:
    void SetUp() override
    {
        auto param = GetParam();
        op = std::get<0>(param);
        compatibility = std::get<1>(param);
        int levels = (compatibility == COMPATIBLE) ? 1 : 2;

        matrix_a.copyTo(fixed_matrix_a);
        coeffs_a = as_coeffs(matrix_a, 1);
        if (!matrix_b.empty()) {
            matrix_b.copyTo(fixed_matrix_b);
            coeffs_b = as_coeffs(matrix_b, levels);
        }
    }

    DWT2D::Coeffs as_coeffs(const cv::Mat& matrix, int levels) const
    {
        DWT2D dwt(create_haar());
        return dwt.create_coeffs(matrix, matrix.size(), levels);
    }

    void assert_same_metadata(
        const DWT2D::Coeffs& actual_coeffs,
        const DWT2D::Coeffs& expected_coeffs
    ) const
    {
        EXPECT_EQ(actual_coeffs.levels(), expected_coeffs.levels());
        EXPECT_EQ(actual_coeffs.image_size(), expected_coeffs.image_size());
        EXPECT_EQ(actual_coeffs.wavelet(), expected_coeffs.wavelet());
        for (int level = 0; level < expected_coeffs.levels(); ++level)
            EXPECT_EQ(actual_coeffs.detail_size(level), expected_coeffs.detail_size(level));
    }

    template <typename A, typename B>
    void test_operation(A&& a, B&& b) const
    {
        const DerivedTest* derived = static_cast<const DerivedTest*>(this);
        if (compatibility == COMPATIBLE) {
            if constexpr (std::is_floating_point_v<std::remove_cvref_t<A>>)
                derived->execute_test_operation(a, b, scalar_a, matrix_b);
            else if constexpr (std::is_floating_point_v<std::remove_cvref_t<B>>)
                derived->execute_test_operation(a, b, matrix_a, scalar_b);
            else
                derived->execute_test_operation(a, b, matrix_a, matrix_b);
        } else {
            constexpr bool both_coeffs =
                (std::is_same_v<std::remove_cvref_t<A>, CoeffsExpr>
                || std::is_same_v<std::remove_cvref_t<A>, DWT2D::Coeffs>)
                &&
                (std::is_same_v<std::remove_cvref_t<B>, CoeffsExpr>
                || std::is_same_v<std::remove_cvref_t<B>, DWT2D::Coeffs>);

            if constexpr (both_coeffs) {
                EXPECT_THROW(
                    {
                        derived->execute_test_operation(a, b, matrix_a, matrix_b);
                    },
                    cv::Exception
                );
            } else {
                //  Don't need to repeat tests that do not involve 2 coefficients objects
                GTEST_SKIP();
            }
        }
    }

    template <typename A>
    void test_operation(A&& a) const
    {
        const DerivedTest* derived = static_cast<const DerivedTest*>(this);
        if constexpr (std::is_floating_point_v<A>)
            derived->execute_test_operation(a, scalar_a);
        else
            derived->execute_test_operation(a, matrix_a);
    }

    DWT2D::Coeffs coeffs_a;
    DWT2D::Coeffs coeffs_b;
    cv::Mat matrix_a;
    cv::Mat matrix_b;
    FixedMatrix fixed_matrix_a;
    FixedMatrix fixed_matrix_b;
    double scalar_a;
    double scalar_b;
    bool compatible_coeffs;
    Operation op;
    CoeffCompatibility compatibility;
};


//  ----------------------------------------------------------------------------
//  Arithmetic Tests
//  ----------------------------------------------------------------------------
class CoeffsArithmeticTest : public CoeffsExpressionTest<CoeffsArithmeticTest>
{
protected:
    void SetUp() override
    {
        matrix_a = (cv::Mat_<Pixel>(8, 4) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7), cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7),
            cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9), cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9),
            cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7), cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7),
            cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9), cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9),
            cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11)
        );
        matrix_b = (cv::Mat_<Pixel>(8, 4) <<
            cv::Scalar(-0.1,  0.2, -0.3,  0.4), cv::Scalar( 0.2, -0.3,  0.4, -0.5), cv::Scalar(-0.1,  0.2, -0.3,  0.4), cv::Scalar( 0.2, -0.3,  0.4, -0.5),
            cv::Scalar( 0.3, -0.4,  0.5, -0.6), cv::Scalar(-0.4,  0.5, -0.6,  0.7), cv::Scalar( 0.3, -0.4,  0.5, -0.6), cv::Scalar(-0.4,  0.5, -0.6,  0.7),
            cv::Scalar(-0.5,  0.6, -0.7,  0.8), cv::Scalar( 0.6, -0.7,  0.8, -0.9), cv::Scalar(-0.5,  0.6, -0.7,  0.8), cv::Scalar( 0.6, -0.7,  0.8, -0.9),
            cv::Scalar( 0.7, -0.8,  0.9, -1.0), cv::Scalar(-0.8,  0.9, -1.0,  1.1), cv::Scalar( 0.7, -0.8,  0.9, -1.0), cv::Scalar(-0.8,  0.9, -1.0,  1.1),
            cv::Scalar(-0.1,  0.2, -0.3,  0.4), cv::Scalar( 0.2, -0.3,  0.4, -0.5), cv::Scalar(-0.1,  0.2, -0.3,  0.4), cv::Scalar( 0.2, -0.3,  0.4, -0.5),
            cv::Scalar( 0.3, -0.4,  0.5, -0.6), cv::Scalar(-0.4,  0.5, -0.6,  0.7), cv::Scalar( 0.3, -0.4,  0.5, -0.6), cv::Scalar(-0.4,  0.5, -0.6,  0.7),
            cv::Scalar(-0.5,  0.6, -0.7,  0.8), cv::Scalar( 0.6, -0.7,  0.8, -0.9), cv::Scalar(-0.5,  0.6, -0.7,  0.8), cv::Scalar( 0.6, -0.7,  0.8, -0.9),
            cv::Scalar( 0.7, -0.8,  0.9, -1.0), cv::Scalar(-0.8,  0.9, -1.0,  1.1), cv::Scalar( 0.7, -0.8,  0.9, -1.0), cv::Scalar(-0.8,  0.9, -1.0,  1.1)
        );
        scalar_a = -0.5;
        scalar_b = 4;
        CoeffsExpressionTest<CoeffsArithmeticTest>::SetUp();
    }

public:
    template <typename A, typename B, typename C, typename D>
    void execute_test_operation(A&& a, B&& b, C&& c, D&& d) const
    {
        CoeffsExpr expr;
        cv::Mat result;
        switch (op)
        {
        case Operation::ADD:
            expr = a + b;
            result = c + d;
            break;
        case Operation::SUBTRACT:
            expr = a - b;
            result = c - d;
            break;
        case Operation::MULTIPLY:
            if constexpr (std::is_floating_point_v<std::remove_cvref_t<A>>
                          || std::is_floating_point_v<std::remove_cvref_t<B>>) {
                expr = a * b;
                result = c * d;
            } else if constexpr (std::is_same_v<std::remove_cvref_t<A>, CoeffsExpr>
                                 || std::is_same_v<std::remove_cvref_t<A>, DWT2D::Coeffs>) {
                // expr = a.mul(b);
                expr = a * b;
                result = c.mul(d);
            } else {
                expr = b * a;
                result = d.mul(a);
                // GTEST_SKIP();
            }
            break;
        case Operation::MULTIPLY_METHOD:
            if constexpr (std::is_floating_point_v<std::remove_cvref_t<A>>
                          || std::is_floating_point_v<std::remove_cvref_t<B>>) {
                GTEST_SKIP();
            } else if constexpr (std::is_same_v<std::remove_cvref_t<A>, CoeffsExpr>
                                 || std::is_same_v<std::remove_cvref_t<A>, DWT2D::Coeffs>) {
                expr = a.mul(b);
                result = c.mul(d);
            } else {
                expr = b.mul(a);
                result = d.mul(a);
            }
            break;
        case Operation::DIVIDE:
            expr = a / b;
            result = c / d;
            break;
        default:
            assert(false);
        }

        auto expected_coeffs = coeffs_a.clone_and_assign(result);
        auto actual_coeffs = static_cast<DWT2D::Coeffs>(expr);
        EXPECT_THAT(actual_coeffs, MatrixEq(expected_coeffs));
        assert_same_metadata(actual_coeffs, expected_coeffs);
    }
};

TEST_P(CoeffsArithmeticTest, CoeffsExprOpCoeffsExpr)
{
    test_operation(CoeffsExpr(coeffs_a), CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsArithmeticTest, CoeffsOpCoeffs)
{
    test_operation(coeffs_a, coeffs_b);
}

TEST_P(CoeffsArithmeticTest, CoeffsExprOpCoeffs)
{
    test_operation(CoeffsExpr(coeffs_a), coeffs_b);
}

TEST_P(CoeffsArithmeticTest, CoeffsOpCoeffsExpr)
{
    test_operation(coeffs_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsArithmeticTest, CoeffsExprOpMatrixExpr)
{
    test_operation(CoeffsExpr(coeffs_a), cv::MatExpr(matrix_b));
}

TEST_P(CoeffsArithmeticTest, MatrixExprOpCoeffsExpr)
{
    test_operation(cv::MatExpr(matrix_a), CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsArithmeticTest, CoeffsExprOpMatrix)
{
    test_operation(CoeffsExpr(coeffs_a), matrix_b);
}

TEST_P(CoeffsArithmeticTest, MatrixOpCoeffsExpr)
{
    test_operation(matrix_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsArithmeticTest, CoeffsExprOpFixedMatrix)
{
    test_operation(CoeffsExpr(coeffs_a), fixed_matrix_b);
}

TEST_P(CoeffsArithmeticTest, FixedMatrixOpCoeffsExpr)
{
    test_operation(fixed_matrix_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsArithmeticTest, CoeffsExprOpScalar)
{
    test_operation(CoeffsExpr(coeffs_a), scalar_b);
}

TEST_P(CoeffsArithmeticTest, ScalarOpCoeffsExpr)
{
    test_operation(scalar_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsArithmeticTest, CoeffsOpMatrixExpr)
{
    test_operation(coeffs_a, cv::MatExpr(matrix_b));
}

TEST_P(CoeffsArithmeticTest, MatrixExprOpCoeffs)
{
    test_operation(cv::MatExpr(matrix_a), coeffs_b);
}

TEST_P(CoeffsArithmeticTest, CoeffsOpMatrix)
{
    test_operation(coeffs_a, matrix_b);
}

TEST_P(CoeffsArithmeticTest, MatrixOpCoeffs)
{
    test_operation(matrix_a, coeffs_b);
}

TEST_P(CoeffsArithmeticTest, CoeffsOpFixedMatrix)
{
    test_operation(coeffs_a, fixed_matrix_b);
}

TEST_P(CoeffsArithmeticTest, FixedMatrixOpCoeffs)
{
    test_operation(fixed_matrix_a, coeffs_b);
}

TEST_P(CoeffsArithmeticTest, CoeffsOpScalar)
{
    test_operation(coeffs_a, scalar_b);
}

TEST_P(CoeffsArithmeticTest, ScalarOpCoeffs)
{
    test_operation(scalar_a, coeffs_b);
}


INSTANTIATE_TEST_CASE_P(
    CoeffsExpressionGroup,
    CoeffsArithmeticTest,
    testing::Combine(
        testing::Values(
            Operation::ADD,
            Operation::SUBTRACT,
            Operation::MULTIPLY,
            Operation::MULTIPLY_METHOD,
            Operation::DIVIDE
        ),
        testing::Values(
            COMPATIBLE,
            INCOMPATIBLE
        )
    ),
    print_expression_test_label
);


//  ----------------------------------------------------------------------------
//  Comparison Tests
//  ----------------------------------------------------------------------------
class CoeffsCompareTest : public CoeffsExpressionTest<CoeffsCompareTest>
{
protected:
    void SetUp() override
    {
        matrix_a = (cv::Mat_<Pixel>(8, 4) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7), cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7),
            cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9), cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9),
            cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7), cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7),
            cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9), cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9),
            cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11)
        );
        matrix_b = (cv::Mat_<Pixel>(8, 4) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(-3, -4, -5, -6), cv::Scalar(4, 5, 6, 7), cv::Scalar(-3, -4, -5, -6), cv::Scalar(4, 5, 6, 7),
            cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9), cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9),
            cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(-3, -4, -5, -6), cv::Scalar(4, 5, 6, 7), cv::Scalar(-3, -4, -5, -6), cv::Scalar(4, 5, 6, 7),
            cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9), cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9),
            cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10)
        );
        scalar_a = -4;
        scalar_b = 4;
        CoeffsExpressionTest<CoeffsCompareTest>::SetUp();
    }

public:
    template <typename A, typename B, typename C, typename D>
    void execute_test_operation(A&& a, B&& b, C&& c, D&& d) const
    {
        cv::Mat actual_result;
        cv::Mat expected_result;
        switch (op)
        {
        case Operation::EQUAL:
            actual_result = a == b;
            compare(c, d, expected_result, cv::CMP_EQ);
            break;
        case Operation::NOT_EQUAL:
            actual_result = a != b;
            compare(c, d, expected_result, cv::CMP_NE);
            break;
        case Operation::LESS_THAN:
            actual_result = a < b;
            compare(c, d, expected_result, cv::CMP_LT);
            break;
        case Operation::LESS_THAN_OR_EQUAL:
            actual_result = a <= b;
            compare(c, d, expected_result, cv::CMP_LE);
            break;
        case Operation::GREATER_THAN:
            actual_result = a > b;
            compare(c, d, expected_result, cv::CMP_GT);
            break;
        case Operation::GREATER_THAN_OR_EQUAL:
            actual_result = a >= b;
            compare(c, d, expected_result, cv::CMP_GE);
            break;
        default:
            assert(false);
        }

        EXPECT_EQ(actual_result.type(), expected_result.type());
        EXPECT_THAT(actual_result, MatrixEq(expected_result));
    }
};

TEST_P(CoeffsCompareTest, CoeffsExprOpCoeffsExpr)
{
    test_operation(CoeffsExpr(coeffs_a), CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsCompareTest, CoeffsOpCoeffs)
{
    test_operation(coeffs_a, coeffs_b);
}

TEST_P(CoeffsCompareTest, CoeffsExprOpCoeffs)
{
    test_operation(CoeffsExpr(coeffs_a), coeffs_b);
}

TEST_P(CoeffsCompareTest, CoeffsOpCoeffsExpr)
{
    test_operation(coeffs_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsCompareTest, CoeffsExprOpMatrixExpr)
{
    test_operation(CoeffsExpr(coeffs_a), cv::MatExpr(matrix_b));
}

TEST_P(CoeffsCompareTest, MatrixExprOpCoeffsExpr)
{
    test_operation(cv::MatExpr(matrix_a), CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsCompareTest, CoeffsExprOpMatrix)
{
    test_operation(CoeffsExpr(coeffs_a), matrix_b);
}

TEST_P(CoeffsCompareTest, MatrixOpCoeffsExpr)
{
    test_operation(matrix_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsCompareTest, CoeffsExprOpFixedMatrix)
{
    test_operation(CoeffsExpr(coeffs_a), fixed_matrix_b);
}

TEST_P(CoeffsCompareTest, FixedMatrixOpCoeffsExpr)
{
    test_operation(fixed_matrix_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsCompareTest, CoeffsExprOpScalar)
{
    test_operation(CoeffsExpr(coeffs_a), scalar_b);
}

TEST_P(CoeffsCompareTest, ScalarOpCoeffsExpr)
{
    test_operation(scalar_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsCompareTest, CoeffsOpMatrixExpr)
{
    test_operation(coeffs_a, cv::MatExpr(matrix_b));
}

TEST_P(CoeffsCompareTest, MatrixExprOpCoeffs)
{
    test_operation(cv::MatExpr(matrix_a), coeffs_b);
}

TEST_P(CoeffsCompareTest, CoeffsOpMatrix)
{
    test_operation(coeffs_a, matrix_b);
}

TEST_P(CoeffsCompareTest, MatrixOpCoeffs)
{
    test_operation(matrix_a, coeffs_b);
}

TEST_P(CoeffsCompareTest, CoeffsOpFixedMatrix)
{
    test_operation(coeffs_a, fixed_matrix_b);
}

TEST_P(CoeffsCompareTest, FixedMatrixOpCoeffs)
{
    test_operation(fixed_matrix_a, coeffs_b);
}

TEST_P(CoeffsCompareTest, CoeffsOpScalar)
{
    test_operation(coeffs_a, scalar_b);
}

TEST_P(CoeffsCompareTest, ScalarOpCoeffs)
{
    test_operation(scalar_a, coeffs_b);
}


INSTANTIATE_TEST_CASE_P(
    CoeffsExpressionGroup,
    CoeffsCompareTest,
    testing::Combine(
        testing::Values(
            Operation::EQUAL,
            Operation::NOT_EQUAL,
            Operation::LESS_THAN,
            Operation::LESS_THAN_OR_EQUAL,
            Operation::GREATER_THAN,
            Operation::GREATER_THAN_OR_EQUAL
        ),
        testing::Values(
            COMPATIBLE,
            INCOMPATIBLE
        )
    ),
    print_expression_test_label
);


//  ----------------------------------------------------------------------------
//  Unary Operator Tests
//  ----------------------------------------------------------------------------
class CoeffsUnaryTest : public CoeffsExpressionTest<CoeffsUnaryTest>
{
protected:
    void SetUp() override
    {
        matrix_a = (cv::Mat_<Pixel>(8, 4) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7), cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7),
            cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9), cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9),
            cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7), cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7),
            cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9), cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9),
            cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11)
        );
        CoeffsExpressionTest<CoeffsUnaryTest>::SetUp();
    }

public:
    template <typename A, typename B>
    void execute_test_operation(A&& a, B&& b) const
    {
        CoeffsExpr actual_expr;
        cv::MatExpr expected_expr;
        switch (op)
        {
        case Operation::ABS:
            actual_expr = cv::abs(a);
            expected_expr = cv::abs(b);
            break;
        case Operation::NEGATE:
            actual_expr = -a;
            expected_expr = -b;
            break;
        default:
            assert(false);
        }

        auto expected_coeffs = coeffs_a.clone_and_assign(expected_expr);
        auto actual_coeffs = static_cast<DWT2D::Coeffs>(actual_expr);
        EXPECT_THAT(actual_coeffs, MatrixEq(expected_coeffs));
        assert_same_metadata(actual_coeffs, expected_coeffs);
    }
};

TEST_P(CoeffsUnaryTest, CoeffsExpr)
{
    test_operation(CoeffsExpr(coeffs_a));
}

TEST_P(CoeffsUnaryTest, Coeffs)
{
    test_operation(coeffs_a);
}

INSTANTIATE_TEST_CASE_P(
    CoeffsExpressionGroup,
    CoeffsUnaryTest,
    testing::Combine(
        testing::Values(
            Operation::ABS,
            Operation::NEGATE
        ),
        testing::Values(
            COMPATIBLE,
            INCOMPATIBLE
        )
    ),
    print_expression_test_label
);


//  ----------------------------------------------------------------------------
//  Min Max Tests
//  ----------------------------------------------------------------------------
class CoeffsMinMaxTest : public CoeffsExpressionTest<CoeffsMinMaxTest>
{
protected:
    void SetUp() override
    {
        matrix_a = (cv::Mat_<Pixel>(8, 4) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7), cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7),
            cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9), cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9),
            cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7), cv::Scalar(3, 4, 5, 6), cv::Scalar(-4, -5, -6, -7),
            cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9), cv::Scalar(5, 6, -7, -8), cv::Scalar(-6, -7, 8, 9),
            cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11)
        );
        matrix_b = (cv::Mat_<Pixel>(8, 4) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(-3, -4, -5, -6), cv::Scalar(4, 5, 6, 7), cv::Scalar(-3, -4, -5, -6), cv::Scalar(4, 5, 6, 7),
            cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9), cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9),
            cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
            cv::Scalar(-3, -4, -5, -6), cv::Scalar(4, 5, 6, 7), cv::Scalar(-3, -4, -5, -6), cv::Scalar(4, 5, 6, 7),
            cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9), cv::Scalar(5, 6, 7, 8), cv::Scalar(6, 7, 8, 9),
            cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10), cv::Scalar(8, 8, 10, 10)
        );
        scalar_a = -4;
        scalar_b = 4;
        CoeffsExpressionTest<CoeffsMinMaxTest>::SetUp();
    }

public:
    template <typename A, typename B, typename C, typename D>
    void execute_test_operation(A&& a, B&& b, C&& c, D&& d) const
    {
        CoeffsExpr actual_result;
        cv::MatExpr expected_result;
        switch (op)
        {
        case Operation::MIN:
            actual_result = cv::min(a, b);
            expected_result = cv::min(c, d);
            break;
        case Operation::MAX:
            actual_result = cv::max(a, b);
            expected_result = cv::max(c, d);
            break;
        default:
            assert(false);
        }

        auto expected_coeffs = coeffs_a.clone_and_assign(expected_result);
        auto actual_coeffs = static_cast<DWT2D::Coeffs>(actual_result);
        EXPECT_THAT(actual_coeffs, MatrixEq(expected_coeffs));
        assert_same_metadata(actual_coeffs, expected_coeffs);
    }
};

TEST_P(CoeffsMinMaxTest, CoeffsExprOpCoeffsExpr)
{
    test_operation(CoeffsExpr(coeffs_a), CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsMinMaxTest, CoeffsOpCoeffs)
{
    test_operation(coeffs_a, coeffs_b);
}

TEST_P(CoeffsMinMaxTest, CoeffsExprOpCoeffs)
{
    test_operation(CoeffsExpr(coeffs_a), coeffs_b);
}

TEST_P(CoeffsMinMaxTest, CoeffsOpCoeffsExpr)
{
    test_operation(coeffs_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsMinMaxTest, CoeffsExprOpMatrixExpr)
{
    test_operation(CoeffsExpr(coeffs_a), cv::MatExpr(matrix_b));
}

TEST_P(CoeffsMinMaxTest, MatrixExprOpCoeffsExpr)
{
    test_operation(cv::MatExpr(matrix_a), CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsMinMaxTest, CoeffsExprOpMatrix)
{
    test_operation(CoeffsExpr(coeffs_a), matrix_b);
}

TEST_P(CoeffsMinMaxTest, MatrixOpCoeffsExpr)
{
    test_operation(matrix_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsMinMaxTest, CoeffsExprOpFixedMatrix)
{
    test_operation(CoeffsExpr(coeffs_a), fixed_matrix_b);
}

TEST_P(CoeffsMinMaxTest, FixedMatrixOpCoeffsExpr)
{
    test_operation(fixed_matrix_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsMinMaxTest, CoeffsExprOpScalar)
{
    test_operation(CoeffsExpr(coeffs_a), scalar_b);
}

TEST_P(CoeffsMinMaxTest, ScalarOpCoeffsExpr)
{
    test_operation(scalar_a, CoeffsExpr(coeffs_b));
}

TEST_P(CoeffsMinMaxTest, CoeffsOpMatrixExpr)
{
    test_operation(coeffs_a, cv::MatExpr(matrix_b));
}

TEST_P(CoeffsMinMaxTest, MatrixExprOpCoeffs)
{
    test_operation(cv::MatExpr(matrix_a), coeffs_b);
}

TEST_P(CoeffsMinMaxTest, CoeffsOpMatrix)
{
    test_operation(coeffs_a, matrix_b);
}

TEST_P(CoeffsMinMaxTest, MatrixOpCoeffs)
{
    test_operation(matrix_a, coeffs_b);
}

TEST_P(CoeffsMinMaxTest, CoeffsOpFixedMatrix)
{
    test_operation(coeffs_a, fixed_matrix_b);
}

TEST_P(CoeffsMinMaxTest, FixedMatrixOpCoeffs)
{
    test_operation(fixed_matrix_a, coeffs_b);
}

TEST_P(CoeffsMinMaxTest, CoeffsOpScalar)
{
    test_operation(coeffs_a, scalar_b);
}

TEST_P(CoeffsMinMaxTest, ScalarOpCoeffs)
{
    test_operation(scalar_a, coeffs_b);
}


INSTANTIATE_TEST_CASE_P(
    CoeffsExpressionGroup,
    CoeffsMinMaxTest,
    testing::Combine(
        testing::Values(
            Operation::MIN,
            Operation::MAX
        ),
        testing::Values(
            COMPATIBLE,
            INCOMPATIBLE
        )
    ),
    print_expression_test_label
);

