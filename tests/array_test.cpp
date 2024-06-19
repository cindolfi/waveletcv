/**
 * Array Unit Test
*/
#include <cvwt/array.hpp>
#include "common.hpp"

using namespace cvwt;
using namespace testing;



//  ----------------------------------------------------------------------------
//  Collect Masked Test
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct MultiChannelCollectMaskedTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    cv::Mat matrix;
    cv::Mat mask;
    cv::Mat expected_values;
};

template<typename T, int CHANNELS>
void PrintTo(const MultiChannelCollectMaskedTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
    *stream << "mask =";
    PrintTo(param.mask, stream);
}

class MultiChannelCollectMaskedTest : public testing::TestWithParam<MultiChannelCollectMaskedTestParam<double, 4>>
{
public:
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;
    using Mask = cv::Mat_<uchar>;

    static std::vector<ParamType> create_params()
    {
        std::vector<ParamType> params = {
            //  0
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(0),
                    cv::Scalar::all(0), cv::Scalar::all(0)
                ),
                .mask = (Mask(2, 2) <<
                    0, 0,
                    0, 0
                ),
                .expected_values = Matrix(),
            },
            //  1
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    1, 0,
                    0, 0
                ),
                .expected_values = (Matrix(1, 1) <<
                    cv::Scalar::all(0)
                ),
            },
            //  2
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    0, 1,
                    0, 0
                ),
                .expected_values = (Matrix(1, 1) <<
                    cv::Scalar::all(1)
                ),
            },
            //  3
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    0, 0,
                    1, 0
                ),
                .expected_values = (Matrix(1, 1) <<
                    cv::Scalar::all(2)
                ),
            },
            //  4
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    0, 0,
                    0, 1
                ),
                .expected_values = (Matrix(1, 1) <<
                    cv::Scalar::all(3)
                ),
            },
            //  5
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    1, 1,
                    0, 0
                ),
                .expected_values = (Matrix(2, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(1)
                ),
            },
            //  6
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    1, 0,
                    1, 0
                ),
                .expected_values = (Matrix(2, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(2)
                ),
            },
            //  7
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    0, 1,
                    0, 1
                ),
                .expected_values = (Matrix(2, 1) <<
                    cv::Scalar::all(1),
                    cv::Scalar::all(3)
                ),
            },
            //  8
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    0, 0,
                    1, 1
                ),
                .expected_values = (Matrix(2, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(3)
                ),
            },
            //  9
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    1, 0,
                    0, 1
                ),
                .expected_values = (Matrix(2, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(3)
                ),
            },
            //  10
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    0, 1,
                    1, 0
                ),
                .expected_values = (Matrix(2, 1) <<
                    cv::Scalar::all(1),
                    cv::Scalar::all(2)
                ),
            },
            //  11
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    1, 1,
                    1, 0
                ),
                .expected_values = (Matrix(3, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2)
                ),
            },
            //  12
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    1, 1,
                    0, 1
                ),
                .expected_values = (Matrix(3, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(1),
                    cv::Scalar::all(3)
                ),
            },
            //  13
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    1, 0,
                    1, 1
                ),
                .expected_values = (Matrix(3, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(2),
                    cv::Scalar::all(3)
                ),
            },
            //  14
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    0, 1,
                    1, 1
                ),
                .expected_values = (Matrix(3, 1) <<
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(3)
                ),
            },
            //  15
            {
                .matrix = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(3)
                ),
                .mask = (Mask(2, 2) <<
                    1, 1,
                    1, 1
                ),
                .expected_values = (Matrix(4, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(3)
                ),
            },
        };

        return params;
    }
};


TEST_P(MultiChannelCollectMaskedTest, CollectedCorrectly)
{
    auto param = GetParam();
    cv::Mat actual_values;
    collect_masked(param.matrix, actual_values, param.mask);

    std::vector<MultiChannelCollectMaskedTest::Pixel> values;
    if (!actual_values.empty())
        values = actual_values;

    std::vector<MultiChannelCollectMaskedTest::Pixel> expected_values;
    if (!param.expected_values.empty())
        expected_values = param.expected_values;

    EXPECT_THAT(values, UnorderedPointwise(ScalarEq(), expected_values));
}

INSTANTIATE_TEST_CASE_P(
    CollectMasked,
    MultiChannelCollectMaskedTest,
    testing::ValuesIn(MultiChannelCollectMaskedTest::create_params())
);




//  ----------------------------------------------------------------------------
//  Negate Every Other Tests
//  ----------------------------------------------------------------------------
struct NegateEveryOtherTestParam
{
    cv::Mat input;
    cv::Mat negated_at_evens_indices;
};

void PrintTo(const NegateEveryOtherTestParam& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "input =";
    PrintTo(param.input, stream);
    // *stream << "negated_at_evens_indices =";
    // PrintTo(param.negated_at_evens_indices, stream);
}

class NegateEveryOtherTest : public testing::TestWithParam<NegateEveryOtherTestParam>
{
public:
    static cv::Mat create_vector(const std::initializer_list<double>& values)
    {
        return cv::Mat(std::vector<double>(values), true);
    }

    static std::vector<ParamType> create_test_params()
    {
        return {
            //  0
            {
                .input = cv::Mat(),
                .negated_at_evens_indices = cv::Mat(),
            },
            //  1
            {
                .input = create_vector({1}),
                .negated_at_evens_indices = create_vector({-1}),
            },
            //  2
            {
                .input = create_vector({1, 2}),
                .negated_at_evens_indices = create_vector({-1, 2}),
            },
            //  3
            {
                .input = create_vector({1, 2, 3}),
                .negated_at_evens_indices = create_vector({-1, 2, -3}),
            },
            //  4
            {
                .input = create_vector({1, 2, 3, 4}),
                .negated_at_evens_indices = create_vector({-1, 2, -3, 4}),
            },
            //  5
            {
                .input = create_vector({1, 2, 3, 4, 5}),
                .negated_at_evens_indices = create_vector({-1, 2, -3, 4, -5}),
            },
            //  6
            {
                .input = create_vector({1, 2, 3, 4, 5, 6}),
                .negated_at_evens_indices = create_vector({-1, 2, -3, 4, -5, 6}),
            },
        };
    }
};

TEST_P(NegateEveryOtherTest, NegateEvens)
{
    auto param = GetParam();
    auto expected_output = param.negated_at_evens_indices;

    cv::Mat actual_output;
    negate_even_indices(param.input, actual_output);

    EXPECT_THAT(actual_output, MatrixEq(expected_output));
}

TEST_P(NegateEveryOtherTest, NegateOdds)
{
    auto param = GetParam();
    auto expected_output = param.negated_at_evens_indices.empty()
                         ? cv::Mat()
                         : -param.negated_at_evens_indices;

    cv::Mat actual_output;
    negate_odd_indices(param.input, actual_output);

    EXPECT_THAT(actual_output, MatrixEq(expected_output));
}

INSTANTIATE_TEST_CASE_P(
    NegateGroup,
    NegateEveryOtherTest,
    testing::ValuesIn(NegateEveryOtherTest::create_test_params())
);


//  ----------------------------------------------------------------------------
//  Is No Array
//  ----------------------------------------------------------------------------
TEST(IsNotArrayTest, IsArray)
{
    EXPECT_FALSE(is_not_array(cv::Mat()));
}

TEST(IsNotArrayTest, IsNotArray)
{
    EXPECT_TRUE(is_not_array(cv::noArray()));
}


//  ----------------------------------------------------------------------------
//  Is Scalar For Array
//  ----------------------------------------------------------------------------
//  ----------------------------------------------------------------------------
//  Fundamental Type
TEST(IsScalarForArrayTest, FundamentalSameTypeIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    double scalar = 0;
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, FundamentalDifferentTypeIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    int scalar = 0;
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

//  ----------------------------------------------------------------------------
//  std::vector
TEST(IsScalarForArrayTest, EmptyStdVectorIsNotScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = std::vector<double>();
    EXPECT_FALSE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, StdVectorIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = std::vector<double>(3);
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, SingleElementStdVectorIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = std::vector<double>(1);
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, WrongSizeStdVectorIsNotScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = std::vector<double>(4);
    EXPECT_FALSE(is_scalar_for_array(scalar, array));
}

//  ----------------------------------------------------------------------------
//  cv::Vec
TEST(IsScalarForArrayTest, EmptyVecIsNotScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Vec<double, 0>();
    EXPECT_FALSE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, VecIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Vec<double, 3>();
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, SingleElementVecIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Vec<double, 1>();
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, WrongSizeVecIsNotScalar)
{
    //  Use float instead of double because cv::Scalar is really just cv::Vec<double, 4>
    auto array = cv::Mat(4, 4, CV_32FC3);
    auto scalar = cv::Vec<float, 4>();
    EXPECT_FALSE(is_scalar_for_array(scalar, array));
}

//  ----------------------------------------------------------------------------
//  Fixed Matrix
TEST(IsScalarForArrayTest, EmptyFixedMatrixIsNotScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Matx<double, 0, 0>();
    EXPECT_FALSE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, SingleRowFixedMatrixIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Matx<double, 1, 3>();
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, WrongSizeSingleRowFixedMatrixIsNotScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Matx<double, 1, 4>();
    EXPECT_FALSE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, SingleColumnFixedMatrixIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Matx<double, 3, 1>();
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, WrongSizeSingleColumnFixedMatrixIsNotScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Matx<double, 4, 1>();
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, SingleElementFixedMatrixIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Matx<double, 1, 1>();
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

//  ----------------------------------------------------------------------------
//  cv::Scalar
TEST(IsScalarForArrayTest, ScalarIsScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Scalar();
    EXPECT_TRUE(is_scalar_for_array(scalar, array));
}

TEST(IsScalarForArrayTest, ScalarIsNotScalarForChannelsGreaterThanFour)
{
    auto array = cv::Mat(4, 4, CV_MAKE_TYPE(CV_64F, 6));
    auto scalar = cv::Scalar();
    EXPECT_FALSE(is_scalar_for_array(scalar, array));
}

//  ----------------------------------------------------------------------------
//  cv::Mat
TEST(IsScalarForArrayTest, MatrixIsNotScalar)
{
    auto array = cv::Mat(4, 4, CV_64FC3);
    auto scalar = cv::Mat(1, 3, CV_64F);
    EXPECT_FALSE(is_scalar_for_array(scalar, array));
}


//  ----------------------------------------------------------------------------
//  Is Vector
//  ----------------------------------------------------------------------------
TEST(IsVectorTest, EmptyMatrixIsNotVector)
{
    EXPECT_FALSE(is_vector(cv::Mat()));
}

TEST(IsVectorTest, SingleChannelRowMatrixIsRowVector)
{
    auto vector = cv::Mat(1, 4, CV_64FC1);
    EXPECT_TRUE(is_vector(vector));
    EXPECT_TRUE(is_vector(vector, 1));
    EXPECT_FALSE(is_vector(vector, 4));

    EXPECT_TRUE(is_row_vector(vector));
    EXPECT_TRUE(is_row_vector(vector, 1));
    EXPECT_FALSE(is_row_vector(vector, 4));

    EXPECT_FALSE(is_column_vector(vector));
    EXPECT_FALSE(is_column_vector(vector, 1));
    EXPECT_FALSE(is_column_vector(vector, 4));
}

TEST(IsVectorTest, MultiChannelRowMatrixIsRowVector)
{
    auto vector = cv::Mat(1, 4, CV_32FC3);
    EXPECT_TRUE(is_vector(vector));
    EXPECT_TRUE(is_vector(vector, 3));
    EXPECT_FALSE(is_vector(vector, 4));

    EXPECT_TRUE(is_row_vector(vector));
    EXPECT_TRUE(is_row_vector(vector, 3));
    EXPECT_FALSE(is_row_vector(vector, 4));

    EXPECT_FALSE(is_column_vector(vector));
    EXPECT_FALSE(is_column_vector(vector, 3));
    EXPECT_FALSE(is_column_vector(vector, 4));
}

TEST(IsVectorTest, SingleChannelColumnMatrixIsColumnVector)
{
    auto vector = cv::Mat(4, 1, CV_64FC1);
    EXPECT_TRUE(is_vector(vector));
    EXPECT_TRUE(is_vector(vector, 1));
    EXPECT_FALSE(is_vector(vector, 4));

    EXPECT_FALSE(is_row_vector(vector));
    EXPECT_FALSE(is_row_vector(vector, 1));
    EXPECT_FALSE(is_row_vector(vector, 4));

    EXPECT_TRUE(is_column_vector(vector));
    EXPECT_TRUE(is_column_vector(vector, 1));
    EXPECT_FALSE(is_column_vector(vector, 4));
}

TEST(IsVectorTest, MultiChannelColumnMatrixIsColumnVector)
{
    auto vector = cv::Mat(4, 1, CV_32FC3);
    EXPECT_TRUE(is_vector(vector));
    EXPECT_TRUE(is_vector(vector, 3));
    EXPECT_FALSE(is_vector(vector, 4));

    EXPECT_FALSE(is_row_vector(vector));
    EXPECT_FALSE(is_row_vector(vector, 3));
    EXPECT_FALSE(is_row_vector(vector, 4));

    EXPECT_TRUE(is_column_vector(vector));
    EXPECT_TRUE(is_column_vector(vector, 3));
    EXPECT_FALSE(is_column_vector(vector, 4));
}

TEST(IsVectorTest, SingleChannelMatrixIsNotVector)
{
    auto matrix = cv::Mat(4, 4, CV_32FC1);
    EXPECT_FALSE(is_vector(matrix));
    EXPECT_FALSE(is_vector(matrix, 1));
    EXPECT_FALSE(is_vector(matrix, 4));

    EXPECT_FALSE(is_row_vector(matrix));
    EXPECT_FALSE(is_row_vector(matrix, 1));
    EXPECT_FALSE(is_row_vector(matrix, 4));

    EXPECT_FALSE(is_column_vector(matrix));
    EXPECT_FALSE(is_column_vector(matrix, 1));
    EXPECT_FALSE(is_column_vector(matrix, 4));
}

TEST(IsVectorTest, MultiChannelMatrixIsNotVector)
{
    auto matrix = cv::Mat(4, 4, CV_32FC3);
    EXPECT_FALSE(is_vector(matrix));
    EXPECT_FALSE(is_vector(matrix, 3));
    EXPECT_FALSE(is_vector(matrix, 4));

    EXPECT_FALSE(is_row_vector(matrix));
    EXPECT_FALSE(is_row_vector(matrix, 3));
    EXPECT_FALSE(is_row_vector(matrix, 4));

    EXPECT_FALSE(is_column_vector(matrix));
    EXPECT_FALSE(is_column_vector(matrix, 3));
    EXPECT_FALSE(is_column_vector(matrix, 4));
}

TEST(IsVectorTest, StdVectorIsRowVector)
{
    auto vector = std::vector<double>(4);
    EXPECT_TRUE(is_vector(vector));
    EXPECT_TRUE(is_vector(vector, 1));
    EXPECT_FALSE(is_vector(vector, 4));

    EXPECT_TRUE(is_row_vector(vector));
    EXPECT_TRUE(is_row_vector(vector, 1));
    EXPECT_FALSE(is_row_vector(vector, 4));

    EXPECT_FALSE(is_column_vector(vector));
    EXPECT_FALSE(is_column_vector(vector, 1));
    EXPECT_FALSE(is_column_vector(vector, 4));
}

TEST(IsVectorTest, SingleElementStdVectorIsRowAndColumnVector)
{
    auto vector = std::vector<double>(1);
    EXPECT_TRUE(is_vector(vector));
    EXPECT_TRUE(is_vector(vector, 1));
    EXPECT_FALSE(is_vector(vector, 4));

    EXPECT_TRUE(is_row_vector(vector));
    EXPECT_TRUE(is_row_vector(vector, 1));
    EXPECT_FALSE(is_row_vector(vector, 4));

    EXPECT_TRUE(is_column_vector(vector));
    EXPECT_TRUE(is_column_vector(vector, 1));
    EXPECT_FALSE(is_column_vector(vector, 4));
}

TEST(IsVectorTest, SingleChannelVecIsRowAndColumnVector)
{
    auto vector = cv::Vec<double, 1>();
    EXPECT_TRUE(is_vector(vector));
    EXPECT_TRUE(is_vector(vector, 1));
    EXPECT_FALSE(is_vector(vector, 4));

    EXPECT_TRUE(is_row_vector(vector));
    EXPECT_TRUE(is_row_vector(vector, 1));
    EXPECT_FALSE(is_row_vector(vector, 4));

    EXPECT_TRUE(is_column_vector(vector));
    EXPECT_TRUE(is_column_vector(vector, 1));
    EXPECT_FALSE(is_column_vector(vector, 4));
}

TEST(IsVectorTest, MultiChannelVecIsColumnVector)
{
    auto vector = cv::Vec<float, 3>();

    EXPECT_TRUE(is_vector(vector));
    EXPECT_TRUE(is_vector(vector, 1));
    EXPECT_FALSE(is_vector(vector, 3));

    EXPECT_FALSE(is_row_vector(vector));
    EXPECT_FALSE(is_row_vector(vector, 1));
    EXPECT_FALSE(is_row_vector(vector, 3));

    EXPECT_TRUE(is_column_vector(vector));
    EXPECT_TRUE(is_column_vector(vector, 1));
    EXPECT_FALSE(is_column_vector(vector, 3));
}

