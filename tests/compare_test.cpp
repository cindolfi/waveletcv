/**
 * Compare Unit Test
*/
#include <numeric>
#include <cvwt/array/compare.hpp>
#include "common.hpp"

using namespace cvwt;
using namespace testing;

//  ----------------------------------------------------------------------------
//  Compare Tests
//  ----------------------------------------------------------------------------
struct CompareTestParam
{
    cv::Mat a;
    cv::Mat b;
    cv::CmpTypes cmp_type;
    cv::Mat expected_a_op_b_result;
    cv::Mat expected_b_op_a_result;
};

void PrintTo(const CompareTestParam& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "a = \n";
    PrintTo(param.a, stream);
    *stream << "b = \n";
    PrintTo(param.b, stream);
    *stream << "expected_a_op_b_result = \n";
    PrintTo(param.expected_a_op_b_result, stream);
    *stream << "expected_b_op_a_result = \n";
    PrintTo(param.expected_b_op_a_result, stream);
}

class CompareTest : public testing::TestWithParam<CompareTestParam>
{
public:
    static std::vector<ParamType> create_test_params()
    {
        cv::Mat a = (cv::Mat4d(3, 3) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2)
        );
        cv::Mat b = (cv::Mat4d(3, 3) <<
            cv::Scalar(1, 1, 1, 1), cv::Scalar(-1, -1, -1, -1), cv::Scalar(0, 0, 0, 0),
            cv::Scalar(2, 2, 2, 2), cv::Scalar(-2, -2, -2, -2), cv::Scalar(-3, -1, 1, 3),
            cv::Scalar(3, 3, 3, 3), cv::Scalar(-3, -3, -3, -3), cv::Scalar(-2, 0, 0, 2)
        );

        cv::Mat a_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 0, 0, 0), cv::Scalar(255, 0, 0, 0), cv::Scalar(0, 0, 0, 0),
            cv::Scalar(0, 255, 0, 0), cv::Scalar(0, 255, 0, 0), cv::Scalar(0, 255, 255, 0),
            cv::Scalar(0, 0, 255, 0), cv::Scalar(0, 0, 255, 0), cv::Scalar(255, 0, 0, 255)
        );
        cv::Mat a_not_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 255, 255, 255), cv::Scalar(0, 255, 255, 255), cv::Scalar(255, 255, 255, 255),
            cv::Scalar(255, 0, 255, 255), cv::Scalar(255, 0, 255, 255), cv::Scalar(255, 0, 0, 255),
            cv::Scalar(255, 255, 0, 255), cv::Scalar(255, 255, 0, 255), cv::Scalar(0, 255, 255, 0)
        );
        cv::Mat a_greater_than_or_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 0, 0, 0), cv::Scalar(0, 0, 255, 255),
            cv::Scalar(0, 255, 255, 255), cv::Scalar(255, 255, 0, 0), cv::Scalar(255, 255, 255, 0),
            cv::Scalar(0, 0, 255, 255), cv::Scalar(255, 255, 255, 0), cv::Scalar(255, 0, 255, 255)
        );
        cv::Mat a_greater_than_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 255, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 255, 255),
            cv::Scalar(0, 0, 255, 255), cv::Scalar(255, 0, 0, 0), cv::Scalar(255, 0, 0, 0),
            cv::Scalar(0, 0, 0, 255), cv::Scalar(255, 255, 0, 0), cv::Scalar(0, 0, 255, 0)
        );
        cv::Mat a_less_than_or_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 0, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 0, 0),
            cv::Scalar(255, 255, 0, 0), cv::Scalar(0, 255, 255, 255), cv::Scalar(0, 255, 255, 255),
            cv::Scalar(255, 255, 255, 0), cv::Scalar(0, 0, 255, 255), cv::Scalar(255, 255, 0, 255)
        );
        cv::Mat a_less_than_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 255, 255, 255), cv::Scalar(255, 255, 0, 0),
            cv::Scalar(255, 0, 0, 0), cv::Scalar(0, 0, 255, 255), cv::Scalar(0, 0, 0, 255),
            cv::Scalar(255, 255, 0, 0), cv::Scalar(0, 0, 0, 255), cv::Scalar(0, 255, 0, 0)
        );

        return {
            //  0
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_EQ,
                .expected_a_op_b_result = a_equal_b_result,
                .expected_b_op_a_result = a_equal_b_result,
            },
            //  1
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_NE,
                .expected_a_op_b_result = a_not_equal_b_result,
                .expected_b_op_a_result = a_not_equal_b_result,
            },
            //  2
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_GE,
                .expected_a_op_b_result = a_greater_than_or_equal_b_result,
                .expected_b_op_a_result = a_less_than_or_equal_b_result,
            },
            //  3
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_GT,
                .expected_a_op_b_result = a_greater_than_b_result,
                .expected_b_op_a_result = a_less_than_b_result,
            },
            //  4
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_LE,
                .expected_a_op_b_result = a_less_than_or_equal_b_result,
                .expected_b_op_a_result = a_greater_than_or_equal_b_result,
            },
            //  5
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_LT,
                .expected_a_op_b_result = a_less_than_b_result,
                .expected_b_op_a_result = a_greater_than_b_result,
            },
        };
    }
};

TEST_P(CompareTest, CorrectComparison)
{
    auto param = GetParam();

    cv::Mat actual_result;
    compare(param.a, param.b, actual_result, param.cmp_type);

    EXPECT_THAT(actual_result, MatrixEq(param.expected_a_op_b_result));
}

TEST_P(CompareTest, CorrectReverseComparison)
{
    auto param = GetParam();

    cv::Mat actual_result;
    compare(param.b, param.a, actual_result, param.cmp_type);

    EXPECT_THAT(actual_result, MatrixEq(param.expected_b_op_a_result));
}

INSTANTIATE_TEST_CASE_P(
    CompareGroup,
    CompareTest,
    testing::ValuesIn(CompareTest::create_test_params())
);

//  ----------------------------------------------------------------------------
struct CompareScalarTestParam
{
    cv::Mat a;
    cv::Scalar b;
    cv::CmpTypes cmp_type;
    cv::Mat expected_a_op_b_result;
    cv::Mat expected_b_op_a_result;
};

void PrintTo(const CompareScalarTestParam& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "a = \n";
    PrintTo(param.a, stream);
    *stream << "b = " << param.b << "\n";
    *stream << "expected_a_op_b_result = \n";
    PrintTo(param.expected_a_op_b_result, stream);
    *stream << "expected_b_op_a_result = \n";
    PrintTo(param.expected_b_op_a_result, stream);
}

class CompareScalarTest : public testing::TestWithParam<CompareScalarTestParam>
{
public:
    static std::vector<ParamType> create_test_params()
    {
        cv::Mat a = (cv::Mat4d(3, 3) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2)
        );
        cv::Scalar b(1, 2, -2, 2);

        cv::Mat a_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 255, 0, 0), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255),
            cv::Scalar(255, 255, 0, 0), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255),
            cv::Scalar(255, 255, 0, 0), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255)
        );
        cv::Mat a_not_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 0, 255, 255), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0),
            cv::Scalar(0, 0, 255, 255), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0),
            cv::Scalar(0, 0, 255, 255), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0)
        );
        cv::Mat a_greater_than_or_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 255, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 255, 255),
            cv::Scalar(255, 255, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 255, 255),
            cv::Scalar(255, 255, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 255, 255)
        );
        cv::Mat a_greater_than_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 0, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 255, 0),
            cv::Scalar(0, 0, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 255, 0),
            cv::Scalar(0, 0, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 255, 0)
        );
        cv::Mat a_less_than_or_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 255, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 0, 255),
            cv::Scalar(255, 255, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 0, 255),
            cv::Scalar(255, 255, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 0, 255)
        );
        cv::Mat a_less_than_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 0, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 0, 0),
            cv::Scalar(0, 0, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 0, 0),
            cv::Scalar(0, 0, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 0, 0)
        );

        return {
            //  0
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_EQ,
                .expected_a_op_b_result = a_equal_b_result,
                .expected_b_op_a_result = a_equal_b_result,
            },
            //  1
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_NE,
                .expected_a_op_b_result = a_not_equal_b_result,
                .expected_b_op_a_result = a_not_equal_b_result,
            },
            //  2
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_GE,
                .expected_a_op_b_result = a_greater_than_or_equal_b_result,
                .expected_b_op_a_result = a_less_than_or_equal_b_result,
            },
            //  3
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_GT,
                .expected_a_op_b_result = a_greater_than_b_result,
                .expected_b_op_a_result = a_less_than_b_result,
            },
            //  4
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_LE,
                .expected_a_op_b_result = a_less_than_or_equal_b_result,
                .expected_b_op_a_result = a_greater_than_or_equal_b_result,
            },
            //  5
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_LT,
                .expected_a_op_b_result = a_less_than_b_result,
                .expected_b_op_a_result = a_greater_than_b_result,
            },
        };
    }
};

TEST_P(CompareScalarTest, CorrectComparison)
{
    auto param = GetParam();

    cv::Mat actual_result;
    compare(param.a, param.b, actual_result, param.cmp_type);

    EXPECT_THAT(actual_result, MatrixEq(param.expected_a_op_b_result));
}

TEST_P(CompareScalarTest, CorrectReverseComparison)
{
    auto param = GetParam();

    cv::Mat actual_result;
    compare(param.b, param.a, actual_result, param.cmp_type);

    EXPECT_THAT(actual_result, MatrixEq(param.expected_b_op_a_result));
}

INSTANTIATE_TEST_CASE_P(
    CompareGroup,
    CompareScalarTest,
    testing::ValuesIn(CompareScalarTest::create_test_params())
);

//  ----------------------------------------------------------------------------
struct ComparePrimitiveScalarTestParam
{
    cv::Mat a;
    double b;
    cv::CmpTypes cmp_type;
    cv::Mat expected_a_op_b_result;
    cv::Mat expected_b_op_a_result;
};

void PrintTo(const ComparePrimitiveScalarTestParam& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "a = \n";
    PrintTo(param.a, stream);
    *stream << "b = " << param.b << "\n";
    *stream << "expected_a_op_b_result = \n";
    PrintTo(param.expected_a_op_b_result, stream);
    *stream << "expected_b_op_a_result = \n";
    PrintTo(param.expected_b_op_a_result, stream);
}

class ComparePrimitiveScalarTest : public testing::TestWithParam<ComparePrimitiveScalarTestParam>
{
public:
    static std::vector<ParamType> create_test_params()
    {
        cv::Mat a = (cv::Mat4d(3, 3) <<
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2),
            cv::Scalar(1, 2, 3, 4), cv::Scalar(-1, -2, -3, -4), cv::Scalar(-2, -1, 1, 2)
        );
        double b = 2;

        cv::Mat a_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 255, 0, 0), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255),
            cv::Scalar(0, 255, 0, 0), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255),
            cv::Scalar(0, 255, 0, 0), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255)
        );
        cv::Mat a_not_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 0, 255, 255), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0),
            cv::Scalar(255, 0, 255, 255), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0),
            cv::Scalar(255, 0, 255, 255), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0)
        );
        cv::Mat a_greater_than_or_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 255, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255),
            cv::Scalar(0, 255, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255),
            cv::Scalar(0, 255, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255)
        );
        cv::Mat a_greater_than_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(0, 0, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 0),
            cv::Scalar(0, 0, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 0),
            cv::Scalar(0, 0, 255, 255), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 0)
        );
        cv::Mat a_less_than_or_equal_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 255, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 255),
            cv::Scalar(255, 255, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 255),
            cv::Scalar(255, 255, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 255)
        );
        cv::Mat a_less_than_b_result = (cv::Mat_<cv::Scalar_<uchar>>(3, 3) <<
            cv::Scalar(255, 0, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0),
            cv::Scalar(255, 0, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0),
            cv::Scalar(255, 0, 0, 0), cv::Scalar(255, 255, 255, 255), cv::Scalar(255, 255, 255, 0)
        );

        return {
            //  0
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_EQ,
                .expected_a_op_b_result = a_equal_b_result,
                .expected_b_op_a_result = a_equal_b_result,
            },
            //  1
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_NE,
                .expected_a_op_b_result = a_not_equal_b_result,
                .expected_b_op_a_result = a_not_equal_b_result,
            },
            //  2
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_GE,
                .expected_a_op_b_result = a_greater_than_or_equal_b_result,
                .expected_b_op_a_result = a_less_than_or_equal_b_result,
            },
            //  3
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_GT,
                .expected_a_op_b_result = a_greater_than_b_result,
                .expected_b_op_a_result = a_less_than_b_result,
            },
            //  4
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_LE,
                .expected_a_op_b_result = a_less_than_or_equal_b_result,
                .expected_b_op_a_result = a_greater_than_or_equal_b_result,
            },
            //  5
            {
                .a = a,
                .b = b,
                .cmp_type = cv::CMP_LT,
                .expected_a_op_b_result = a_less_than_b_result,
                .expected_b_op_a_result = a_greater_than_b_result,
            },
        };
    }
};

TEST_P(ComparePrimitiveScalarTest, CorrectComparison)
{
    auto param = GetParam();

    cv::Mat actual_result;
    compare(param.a, param.b, actual_result, param.cmp_type);

    EXPECT_THAT(actual_result, MatrixEq(param.expected_a_op_b_result));
}

TEST_P(ComparePrimitiveScalarTest, CorrectReverseComparison)
{
    auto param = GetParam();

    cv::Mat actual_result;
    compare(param.b, param.a, actual_result, param.cmp_type);

    EXPECT_THAT(actual_result, MatrixEq(param.expected_b_op_a_result));
}

INSTANTIATE_TEST_CASE_P(
    CompareGroup,
    ComparePrimitiveScalarTest,
    testing::ValuesIn(ComparePrimitiveScalarTest::create_test_params())
);


//  ----------------------------------------------------------------------------
//  Is Approx Equal Tests
//  ----------------------------------------------------------------------------
class IsApproxEqualTest : public testing::Test
{
public:
    template <std::floating_point T>
    static constexpr T default_tolerance = std::sqrt(std::numeric_limits<T>::epsilon());

    static constexpr int ROWS = 4;
    static constexpr int COLUMNS = 4;
    static constexpr int TYPE = CV_64FC3;
protected:
    void SetUp() override
    {
        cv::theRNG().state = 1;
    }

    template <typename T>
    cv::Mat create_epsilon(T tolerance) const
    {
        cv::Mat epsilon = cv::Mat(ROWS, COLUMNS, TYPE);
        cv::randu(epsilon, -tolerance, tolerance);
        // cv::randn(epsilon, cv::Scalar::all(0), cv::Scalar::all(tolerance));

        return epsilon;
    }

    cv::Mat create_ones() const
    {
        return cv::Mat::ones(ROWS, COLUMNS, TYPE);
    }

    cv::Mat create_zeros() const
    {
        return cv::Mat::ones(ROWS, COLUMNS, TYPE);
    }
};

TEST_F(IsApproxEqualTest, ZerosWithDifferenceLessThanAbsToleranceAreApproxEqual)
{
    double relative_tolerance = 1e-16;
    double absolute_tolerance = 1e-4;

    cv::Mat epsilon = create_epsilon(absolute_tolerance);
    cv::Mat a = cv::Mat::zeros(4, 4, CV_64FC3);
    cv::Mat b = a + epsilon;
    // std::cout << epsilon << "\n";
    // std::cout << epsilon.at<double>(0, 0) << "\n";

    EXPECT_TRUE(is_approx_equal(a, b, relative_tolerance, absolute_tolerance));
}

TEST_F(IsApproxEqualTest, ZerosWithDifferenceGreaterThanAbsToleranceAreNotApproxEqual)
{
    double relative_tolerance = 1e-16;
    double absolute_tolerance = 1e-4;

    cv::Mat epsilon = create_epsilon(4 * absolute_tolerance);
    cv::Mat a = create_zeros();
    cv::Mat b = a + epsilon;

    // std::cout << b << "\n";

    EXPECT_FALSE(is_approx_equal(a, b, relative_tolerance, absolute_tolerance));
}

TEST_F(IsApproxEqualTest, ZerosWithDifferenceLessThanDefaultAbsToleranceAreApproxEqual)
{
    cv::Mat epsilon = create_epsilon(default_tolerance<double>);
    cv::Mat a = create_zeros();
    cv::Mat b = a + epsilon;

    EXPECT_TRUE(is_approx_equal(a, b));
}

TEST_F(IsApproxEqualTest, ZerosWithDifferenceGreaterThanDefaultAbsToleranceAreNotApproxEqual)
{
    cv::Mat epsilon = create_epsilon(4 * default_tolerance<double>);
    cv::Mat a = create_zeros();
    cv::Mat b = a + epsilon;

    EXPECT_FALSE(is_approx_equal(a, b));
}

TEST_F(IsApproxEqualTest, OnesWithDifferenceLessThanRelToleranceAreApproxEqual)
{
    double relative_tolerance = 1e-4;

    cv::Mat epsilon = create_epsilon(relative_tolerance);
    cv::Mat a = create_ones();
    cv::Mat b = a.mul(1 + epsilon);

    EXPECT_TRUE(is_approx_equal(a, b, relative_tolerance));
}

TEST_F(IsApproxEqualTest, OnesWithDifferenceGreaterThanRelToleranceAreNotApproxEqual)
{
    double relative_tolerance = 1e-4;

    cv::Mat epsilon = create_epsilon(4 * relative_tolerance);
    cv::Mat a = create_ones();
    cv::Mat b = a.mul(1 + epsilon);

    EXPECT_FALSE(is_approx_equal(a, b, relative_tolerance));
}

TEST_F(IsApproxEqualTest, OnesWithDifferenceLessThanDefaultRelToleranceAreApproxEqual)
{
    cv::Mat epsilon = create_epsilon(default_tolerance<double>);
    cv::Mat a = create_ones();
    cv::Mat b = a.mul(1 + epsilon);

    EXPECT_TRUE(is_approx_equal(a, b));
}

TEST_F(IsApproxEqualTest, OnesWithDifferenceGreaterThanDefaultRelToleranceAreNotApproxEqual)
{
    cv::Mat epsilon = create_epsilon(4 * default_tolerance<double>);
    cv::Mat a = create_ones();
    cv::Mat b = a.mul(1 + epsilon);

    EXPECT_FALSE(is_approx_equal(a, b));
}


//  ----------------------------------------------------------------------------
//  Is Approx Zero Tests
//  ----------------------------------------------------------------------------
class IsApproxZeroTest : public IsApproxEqualTest
{};

TEST_F(IsApproxZeroTest, LessThanAbsToleranceIsApproxZero)
{
    double absolute_tolerance = 1e-4;
    cv::Mat a = create_epsilon(absolute_tolerance);

    EXPECT_TRUE(is_approx_zero(a, absolute_tolerance));
}

TEST_F(IsApproxZeroTest, GreaterThanAbsToleranceIsNotApproxZero)
{
    double absolute_tolerance = 1e-4;
    cv::Mat a = create_epsilon(4 * absolute_tolerance);

    EXPECT_FALSE(is_approx_zero(a, absolute_tolerance));
}

TEST_F(IsApproxZeroTest, LessThanDefaultAbsToleranceIsApproxZero)
{
    cv::Mat a = create_epsilon(default_tolerance<double>);

    EXPECT_TRUE(is_approx_zero(a));
}

TEST_F(IsApproxZeroTest, GreaterThanDefaultAbsToleranceIsNotApproxZero)
{
    cv::Mat a = create_epsilon(4 * default_tolerance<double>);

    EXPECT_FALSE(is_approx_zero(a));
}

