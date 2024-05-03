/**
 * Shrinkage Unit Test
*/
#include <cvwt/shrinkage.hpp>
#include "common.hpp"

using namespace cvwt;
using namespace testing;


const std::vector<int> PERMUTATION1 = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
const std::vector<int> PERMUTATION2 = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};

//  ============================================================================
//  MAD
//  ============================================================================
template<typename T, int CHANNELS>
struct MadTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    cv::Mat matrix;
    cv::Scalar expected_mad;
    cv::Scalar expected_mad_stdev;
};

template<typename T, int CHANNELS>
void PrintTo(const MadTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "expected_mad = " << param.expected_mad << "\n";
    *stream << "expected_mad_stdev = " << param.expected_mad_stdev << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
}



class MadTest : public testing::TestWithParam<MadTestParam<double, 4>>
{
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;

public:
    static std::vector<ParamType> create_params()
    {
        return {
            //  0
            {
                .matrix = (Matrix(16, 1) <<
                    cv::Scalar::all(-7),
                    cv::Scalar::all(-6),
                    cv::Scalar::all(-5),
                    cv::Scalar::all(-4),
                    cv::Scalar::all(-3),
                    cv::Scalar::all(-2),
                    cv::Scalar::all(-1),
                    cv::Scalar::all( 0),
                    cv::Scalar::all( 1),
                    cv::Scalar::all( 2),
                    cv::Scalar::all( 3),
                    cv::Scalar::all( 4),
                    cv::Scalar::all( 5),
                    cv::Scalar::all( 6),
                    cv::Scalar::all( 7),
                    cv::Scalar::all( 8)
                ),
                .expected_mad = cv::Scalar::all(4.0),
                .expected_mad_stdev = cv::Scalar::all(5.925925925925926),
            },
            //  1
            {
                .matrix = (Matrix(16, 1) <<
                    cv::Scalar::all(-1.7236),
                    cv::Scalar::all(-0.3368),
                    cv::Scalar::all(-1.4274),
                    cv::Scalar::all( 1.3655),
                    cv::Scalar::all(-0.0454),
                    cv::Scalar::all( 0.4042),
                    cv::Scalar::all( 0.3494),
                    cv::Scalar::all(-2.0478),
                    cv::Scalar::all(-0.9477),
                    cv::Scalar::all( 0.1772),
                    cv::Scalar::all( 0.1628),
                    cv::Scalar::all(-1.6662),
                    cv::Scalar::all(-0.2807),
                    cv::Scalar::all( 0.1733),
                    cv::Scalar::all( 0.1010),
                    cv::Scalar::all( 0.2946)
                ),
                .expected_mad = cv::Scalar::all(0.3431),
                .expected_mad_stdev = cv::Scalar::all(0.5082962962962962),
            },
            //  2
            {
                .matrix = (Matrix(16, 1) <<
                    cv::Scalar(-7, -14, -1.7236, -0.3164),
                    cv::Scalar(-6, -12, -0.3368,  2.5333),
                    cv::Scalar(-5, -10, -1.4274,  0.9211),
                    cv::Scalar(-4,  -8,  1.3655, -1.0288),
                    cv::Scalar(-3,  -6, -0.0454,  0.0947),
                    cv::Scalar(-2,  -4,  0.4042,  1.1946),
                    cv::Scalar(-1,  -2,  0.3494,  0.2992),
                    cv::Scalar( 0,   0, -2.0478,  0.5138),
                    cv::Scalar( 1,   2, -0.9477,  1.0836),
                    cv::Scalar( 2,   4,  0.1772, -0.0993),
                    cv::Scalar( 3,   6,  0.1628,  0.5039),
                    cv::Scalar( 4,   8, -1.6662, -0.5849),
                    cv::Scalar( 5,  10, -0.2807, -0.6364),
                    cv::Scalar( 6,  12,  0.1733, -1.4400),
                    cv::Scalar( 7,  14,  0.1010,  0.7369),
                    cv::Scalar( 8,  16,  0.2946, -0.9443)
                ),
                .expected_mad = cv::Scalar(
                    4.0,
                    8.0,
                    0.3431,
                    0.7530
                ),
                .expected_mad_stdev = cv::Scalar(
                    5.925925925925926,
                    11.851851851851851,
                    0.5082962962962962,
                    1.1155555555555554
                ),
            },
        };
    }
};

TEST_P(MadTest, MadComputedCorrectly)
{
    auto param = GetParam();

    auto actual_mad = mad(param.matrix);

    EXPECT_THAT(actual_mad, ScalarDoubleEq(param.expected_mad));
}

TEST_P(MadTest, MadStdevComputedCorrectly)
{
    auto param = GetParam();

    auto actual_mad_stdev = mad_stdev(param.matrix);

    EXPECT_THAT(actual_mad_stdev, ScalarDoubleEq(param.expected_mad_stdev));
}

// TEST_P(MadTest, InvariantUnderShift1)
// {
//     auto param = GetParam();
//     auto matrix = param.matrix + cv::Scalar(0.5, 1.0, 1.25, 1.333333);

//     auto actual_mad = mad(matrix);

//     EXPECT_THAT(actual_mad, ScalarDoubleEq(param.expected_mad));
// }

// TEST_P(MadTest, InvariantUnderShift2)
// {
//     auto param = GetParam();
//     auto matrix = param.matrix + cv::Scalar(-1.2, -1.6, -2.8, -3.4);

//     auto actual_mad = mad(matrix);

//     EXPECT_THAT(actual_mad, ScalarDoubleEq(param.expected_mad));
//     ASSERT_FALSE(true);
// }

TEST_P(MadTest, InvariantUnderNegation)
{
    auto param = GetParam();
    auto matrix = -param.matrix;

    auto actual_mad = mad(matrix);

    EXPECT_THAT(actual_mad, ScalarDoubleEq(param.expected_mad));
}

TEST_P(MadTest, CovariantUnderScale1)
{
    auto param = GetParam();
    double scale = 2.0;
    auto matrix = scale * param.matrix;

    auto actual_mad = mad(matrix);

    EXPECT_THAT(actual_mad, ScalarDoubleEq(scale * param.expected_mad));
}

TEST_P(MadTest, CovariantUnderScale2)
{
    auto param = GetParam();
    double scale = 0.5;
    auto matrix = scale * param.matrix;

    auto actual_mad = mad(matrix);

    EXPECT_THAT(actual_mad, ScalarDoubleEq(scale* param.expected_mad));
}

TEST_P(MadTest, InvariantUnderReversal)
{
    auto param = GetParam();
    cv::Mat matrix;
    cv::flip(param.matrix, matrix, -1);

    auto actual_mad = mad(matrix);

    EXPECT_THAT(actual_mad, ScalarDoubleEq(param.expected_mad));
}

TEST_P(MadTest, InvariantUnderPermutation1)
{
    auto param = GetParam();
    auto matrix = permute_matrix(param.matrix, PERMUTATION1);

    auto actual_mad = mad(matrix);

    EXPECT_THAT(
        actual_mad,
        ScalarDoubleEq(param.expected_mad)
    ) << "where permuation = [" << join(PERMUTATION1, ", ") << "]";
}

TEST_P(MadTest, InvariantUnderPermutation2)
{
    auto param = GetParam();
    auto matrix = permute_matrix(param.matrix, PERMUTATION2);

    auto actual_mad = mad(matrix);

    EXPECT_THAT(
        actual_mad,
        ScalarDoubleEq(param.expected_mad)
    ) << "where permuation = [" << join(PERMUTATION2, ", ") << "]";
}

TEST_P(MadTest, InvariantUnderReshape1)
{
    auto param = GetParam();
    auto matrix = param.matrix;
    matrix = matrix.reshape(0, 4);

    auto actual_mad = mad(matrix);

    EXPECT_THAT(
        actual_mad,
        ScalarDoubleEq(param.expected_mad)
    ) << "where size = " << matrix.size();
}

TEST_P(MadTest, InvariantUnderReshape2)
{
    auto param = GetParam();
    auto matrix = param.matrix;
    matrix = matrix.reshape(0, 1);

    auto actual_mad = mad(matrix);

    EXPECT_THAT(
        actual_mad,
        ScalarDoubleEq(param.expected_mad)
    ) << "where size = " << matrix.size();
}

INSTANTIATE_TEST_CASE_P(
    MadGroup,
    MadTest,
    testing::ValuesIn(MadTest::create_params())
);

//  ============================================================================
//  Threshold
//  ============================================================================
//  ----------------------------------------------------------------------------

//  ----------------------------------------------------------------------------
//  Soft Threshold Test
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct SoftThresholdTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    cv::Mat matrix;
    cv::Scalar threshold;
    cv::Mat expected;
};

template<typename T, int CHANNELS>
void PrintTo(const SoftThresholdTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "threshold = " << param.threshold << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
}

/**
 * TODO: negative thresholds
*/
class SoftThresholdTest : public testing::TestWithParam<SoftThresholdTestParam<double, 4>>
{
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;

public:
    static std::vector<ParamType> create_params()
    {
        Matrix matrix(16, 1);
        matrix <<
            cv::Scalar::all(-7),
            cv::Scalar::all(-6),
            cv::Scalar::all(-5),
            cv::Scalar::all(-4),
            cv::Scalar::all(-3),
            cv::Scalar::all(-2),
            cv::Scalar::all(-1),
            cv::Scalar::all( 0),
            cv::Scalar::all( 1),
            cv::Scalar::all( 2),
            cv::Scalar::all( 3),
            cv::Scalar::all( 4),
            cv::Scalar::all( 5),
            cv::Scalar::all( 6),
            cv::Scalar::all( 7),
            cv::Scalar::all( 8);

        return {
            //  0
            {
                .matrix = matrix.clone(),
                .threshold = cv::Scalar(0, 0, 0, 0),
                .expected = matrix.clone(),
            },
            //  1
            {
                .matrix = matrix.clone(),
                .threshold = cv::Scalar(2, 2, 2, 2),
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-4, -4, -4, -4),
                    cv::Scalar(-3, -3, -3, -3),
                    cv::Scalar(-2, -2, -2, -2),
                    cv::Scalar(-1, -1, -1, -1),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 1,  1,  1,  1),
                    cv::Scalar( 2,  2,  2,  2),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 5,  5,  5,  5),
                    cv::Scalar( 6,  6,  6,  6)
                ),
            },
            //  2
            {
                .matrix = matrix.clone(),
                .threshold = cv::Scalar(2, 3, 4, 5),
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-5, -4, -3, -2),
                    cv::Scalar(-4, -3, -2, -1),
                    cv::Scalar(-3, -2, -1,  0),
                    cv::Scalar(-2, -1,  0,  0),
                    cv::Scalar(-1,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 1,  0,  0,  0),
                    cv::Scalar( 2,  1,  0,  0),
                    cv::Scalar( 3,  2,  1,  0),
                    cv::Scalar( 4,  3,  2,  1),
                    cv::Scalar( 5,  4,  3,  2),
                    cv::Scalar( 6,  5,  4,  3)
                ),
            },
        };
    }
};


TEST_P(SoftThresholdTest, RowVectorThresholdedCorrectly)
{
    std::cout << "\n";
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 1);
    auto expected = param.expected.reshape(0, 1);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(SoftThresholdTest, RowVectorThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 1);
    auto expected = param.expected.reshape(0, 1);

    soft_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}


TEST_P(SoftThresholdTest, ColumnVectorThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 16);
    auto expected = param.expected.reshape(0, 16);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(SoftThresholdTest, ColumnVectorThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 16);
    auto expected = param.expected.reshape(0, 16);

    soft_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(SoftThresholdTest, SquareMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 4);
    auto expected = param.expected.reshape(0, 4);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(SoftThresholdTest, SquareMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 4);
    auto expected = param.expected.reshape(0, 4);

    soft_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(SoftThresholdTest, TallMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 8);
    auto expected = param.expected.reshape(0, 8);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(SoftThresholdTest, TallMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 8);
    auto expected = param.expected.reshape(0, 8);

    soft_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(SoftThresholdTest, WideMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 2);
    auto expected = param.expected.reshape(0, 2);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(SoftThresholdTest, WideMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 2);
    auto expected = param.expected.reshape(0, 2);

    soft_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}


INSTANTIATE_TEST_CASE_P(
    SoftThresholdGroup,
    SoftThresholdTest,
    testing::ValuesIn(SoftThresholdTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Masked Soft Threshold
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct MaskedSoftThresholdTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    cv::Mat matrix;
    cv::Mat mask;
    cv::Scalar threshold;
    cv::Mat expected;
};

template<typename T, int CHANNELS>
void PrintTo(const MaskedSoftThresholdTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
    *stream << "mask =";
    PrintTo(param.mask, stream);
    *stream << "threshold = " << param.threshold << "\n";
}

/**
 * TODO: negative thresholds
*/
class MaskedSoftThresholdTest : public testing::TestWithParam<MaskedSoftThresholdTestParam<double, 4>>
{
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;
    using Mask = cv::Mat_<uchar>;

public:
    static std::vector<ParamType> create_params()
    {
        Matrix matrix(16, 1);
        matrix <<
            cv::Scalar::all(-7),
            cv::Scalar::all(-6),
            cv::Scalar::all(-5),
            cv::Scalar::all(-4),
            cv::Scalar::all(-3),
            cv::Scalar::all(-2),
            cv::Scalar::all(-1),
            cv::Scalar::all( 0),
            cv::Scalar::all( 1),
            cv::Scalar::all( 2),
            cv::Scalar::all( 3),
            cv::Scalar::all( 4),
            cv::Scalar::all( 5),
            cv::Scalar::all( 6),
            cv::Scalar::all( 7),
            cv::Scalar::all( 8);

        Mask all_ones_mask(16, 1);
        all_ones_mask << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

        Mask all_zeros_mask(16, 1);
        all_zeros_mask << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        Mask general_mask(16, 1);
        general_mask << 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1;

        cv::Scalar constant_threshold_of_zeros(0, 0, 0, 0);
        cv::Scalar constant_threshold_of_twos(2, 2, 2, 2);
        cv::Scalar independent_channel_thresholds(2, 3, 4, 5);

        return {
            //  0
            {
                .matrix = matrix.clone(),
                .mask = all_ones_mask.clone(),
                .threshold = constant_threshold_of_zeros,
                .expected = matrix.clone(),
            },
            //  1
            {
                .matrix = matrix.clone(),
                .mask = all_zeros_mask.clone(),
                .threshold = constant_threshold_of_zeros,
                .expected = matrix.clone(),
            },
            //  2
            {
                .matrix = matrix.clone(),
                .mask = general_mask.clone(),
                .threshold = constant_threshold_of_zeros,
                .expected = matrix.clone(),
            },
            //  ----------------------------------------------------------------
            //  3
            {
                .matrix = matrix.clone(),
                .mask = all_ones_mask.clone(),
                .threshold = constant_threshold_of_twos,
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-4, -4, -4, -4),
                    cv::Scalar(-3, -3, -3, -3),
                    cv::Scalar(-2, -2, -2, -2),
                    cv::Scalar(-1, -1, -1, -1),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 1,  1,  1,  1),
                    cv::Scalar( 2,  2,  2,  2),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 5,  5,  5,  5),
                    cv::Scalar( 6,  6,  6,  6)
                ),
            },
            //  4
            {
                .matrix = matrix.clone(),
                .mask = all_zeros_mask.clone(),
                .threshold = constant_threshold_of_twos,
                .expected = matrix.clone(),
            },
            //  5
            {
                .matrix = matrix.clone(),
                .mask = general_mask.clone(),
                .threshold = constant_threshold_of_twos,
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-2, -2, -2, -2),
                    cv::Scalar(-1, -1, -1, -1),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 5,  5,  5,  5),
                    cv::Scalar( 6,  6,  6,  6)
                ),
            },
            //  ----------------------------------------------------------------
            //  6
            {
                .matrix = matrix.clone(),
                .mask = all_ones_mask.clone(),
                .threshold = independent_channel_thresholds,
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-5, -4, -3, -2),
                    cv::Scalar(-4, -3, -2, -1),
                    cv::Scalar(-3, -2, -1,  0),
                    cv::Scalar(-2, -1,  0,  0),
                    cv::Scalar(-1,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 1,  0,  0,  0),
                    cv::Scalar( 2,  1,  0,  0),
                    cv::Scalar( 3,  2,  1,  0),
                    cv::Scalar( 4,  3,  2,  1),
                    cv::Scalar( 5,  4,  3,  2),
                    cv::Scalar( 6,  5,  4,  3)
                ),
            },
            //  7
            {
                .matrix = matrix.clone(),
                .mask = all_zeros_mask.clone(),
                .threshold = independent_channel_thresholds,
                .expected = matrix.clone(),
            },
            //  8
            {
                .matrix = matrix.clone(),
                .mask = general_mask.clone(),
                .threshold = independent_channel_thresholds,
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-5, -4, -3, -2),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-2, -1,  0,  0),
                    cv::Scalar(-1,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 3,  2,  1,  0),
                    cv::Scalar( 4,  3,  2,  1),
                    cv::Scalar( 5,  4,  3,  2),
                    cv::Scalar( 6,  5,  4,  3)
                ),
            },
        };
    }
};

TEST_P(MaskedSoftThresholdTest, ColumnVectorThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 16);
    auto expected = param.expected.reshape(0, 16);
    auto mask = param.mask.reshape(0, 16);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, ColumnVectorThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 16);
    auto expected = param.expected.reshape(0, 16);
    auto mask = param.mask.reshape(0, 16);

    soft_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, RowVectorThresholdedCorrectly)
{
    auto param = GetParam();
    cv::Mat actual;
    auto matrix = param.matrix.reshape(0, 1);
    auto expected = param.expected.reshape(0, 1);
    auto mask = param.mask.reshape(0, 1);

    soft_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, RowVectorThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 1);
    auto expected = param.expected.reshape(0, 1);
    auto mask = param.mask.reshape(0, 1);

    soft_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, SquareMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 4);
    auto expected = param.expected.reshape(0, 4);
    auto mask = param.mask.reshape(0, 4);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, SquareMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 4);
    auto expected = param.expected.reshape(0, 4);
    auto mask = param.mask.reshape(0, 4);

    soft_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, TallMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 8);
    auto expected = param.expected.reshape(0, 8);
    auto mask = param.mask.reshape(0, 8);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, TallMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 8);
    auto expected = param.expected.reshape(0, 8);
    auto mask = param.mask.reshape(0, 8);

    soft_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, WideMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 2);
    auto expected = param.expected.reshape(0, 2);
    auto mask = param.mask.reshape(0, 2);

    cv::Mat actual;
    soft_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedSoftThresholdTest, WideMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 2);
    auto expected = param.expected.reshape(0, 2);
    auto mask = param.mask.reshape(0, 2);

    soft_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}


INSTANTIATE_TEST_CASE_P(
    SoftThresholdGroup,
    MaskedSoftThresholdTest,
    testing::ValuesIn(MaskedSoftThresholdTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Hard Threshold Test
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct HardThresholdTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    cv::Mat matrix;
    cv::Scalar threshold;
    cv::Mat expected;
};

template<typename T, int CHANNELS>
void PrintTo(const HardThresholdTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
    *stream << "threshold = " << param.threshold << "\n";
}

/**
 * TODO: negative thresholds
*/
class HardThresholdTest : public testing::TestWithParam<HardThresholdTestParam<double, 4>>
{
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;

public:
    static std::vector<ParamType> create_params()
    {
        Matrix matrix(16, 1);
        matrix <<
            cv::Scalar::all(-7),
            cv::Scalar::all(-6),
            cv::Scalar::all(-5),
            cv::Scalar::all(-4),
            cv::Scalar::all(-3),
            cv::Scalar::all(-2),
            cv::Scalar::all(-1),
            cv::Scalar::all( 0),
            cv::Scalar::all( 1),
            cv::Scalar::all( 2),
            cv::Scalar::all( 3),
            cv::Scalar::all( 4),
            cv::Scalar::all( 5),
            cv::Scalar::all( 6),
            cv::Scalar::all( 7),
            cv::Scalar::all( 8);

        return {
            //  0
            {
                .matrix = matrix.clone(),
                .threshold = cv::Scalar(0, 0, 0, 0),
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-7, -7, -7, -7),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-4, -4, -4, -4),
                    cv::Scalar(-3, -3, -3, -3),
                    cv::Scalar(-2, -2, -2, -2),
                    cv::Scalar(-1, -1, -1, -1),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 1,  1,  1,  1),
                    cv::Scalar( 2,  2,  2,  2),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 5,  5,  5,  5),
                    cv::Scalar( 6,  6,  6,  6),
                    cv::Scalar( 7,  7,  7,  7),
                    cv::Scalar( 8,  8,  8,  8)
                ),
            },
            //  1
            {
                .matrix = matrix.clone(),
                .threshold = cv::Scalar(2, 2, 2, 2),
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-7, -7, -7, -7),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-4, -4, -4, -4),
                    cv::Scalar(-3, -3, -3, -3),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 5,  5,  5,  5),
                    cv::Scalar( 6,  6,  6,  6),
                    cv::Scalar( 7,  7,  7,  7),
                    cv::Scalar( 8,  8,  8,  8)
                ),
            },
            //  2
            {
                .matrix = matrix.clone(),
                .threshold = cv::Scalar(2, 3, 4, 5),
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-7, -7, -7, -7),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5,  0),
                    cv::Scalar(-4, -4,  0,  0),
                    cv::Scalar(-3,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 3,  0,  0,  0),
                    cv::Scalar( 4,  4,  0,  0),
                    cv::Scalar( 5,  5,  5,  0),
                    cv::Scalar( 6,  6,  6,  6),
                    cv::Scalar( 7,  7,  7,  7),
                    cv::Scalar( 8,  8,  8,  8)
                ),
            },
        };
    }
};


TEST_P(HardThresholdTest, RowVectorThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 1);
    auto expected = param.expected.reshape(0, 1);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(HardThresholdTest, RowVectorThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 1);
    auto expected = param.expected.reshape(0, 1);

    hard_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(HardThresholdTest, ColumnVectorThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 16);
    auto expected = param.expected.reshape(0, 16);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(HardThresholdTest, ColumnVectorThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 16);
    auto expected = param.expected.reshape(0, 16);

    hard_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(HardThresholdTest, SquareMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 4);
    auto expected = param.expected.reshape(0, 4);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(HardThresholdTest, SquareMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 4);
    auto expected = param.expected.reshape(0, 4);

    hard_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(HardThresholdTest, TallMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 8);
    auto expected = param.expected.reshape(0, 8);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(HardThresholdTest, TallMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 8);
    auto expected = param.expected.reshape(0, 8);

    hard_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(HardThresholdTest, WideMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 2);
    auto expected = param.expected.reshape(0, 2);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(HardThresholdTest, WideMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 2);
    auto expected = param.expected.reshape(0, 2);

    hard_threshold(matrix, matrix, param.threshold);

    EXPECT_THAT(matrix, MatrixEq(expected));
}


INSTANTIATE_TEST_CASE_P(
    HardThresholdGroup,
    HardThresholdTest,
    testing::ValuesIn(HardThresholdTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Masked Hard Threshold
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct MaskedHardThresholdTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    cv::Mat matrix;
    cv::Mat mask;
    cv::Scalar threshold;
    cv::Mat expected;
};

template<typename T, int CHANNELS>
void PrintTo(const MaskedHardThresholdTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
    *stream << "mask =";
    PrintTo(param.mask, stream);
    *stream << "threshold = " << param.threshold << "\n";
}

/**
 * TODO: negative thresholds
*/
class MaskedHardThresholdTest : public testing::TestWithParam<MaskedHardThresholdTestParam<double, 4>>
{
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;
    using Mask = cv::Mat_<uchar>;

public:
    static std::vector<ParamType> create_params()
    {
        Matrix matrix(16, 1);
        matrix <<
            cv::Scalar::all(-7),
            cv::Scalar::all(-6),
            cv::Scalar::all(-5),
            cv::Scalar::all(-4),
            cv::Scalar::all(-3),
            cv::Scalar::all(-2),
            cv::Scalar::all(-1),
            cv::Scalar::all( 0),
            cv::Scalar::all( 1),
            cv::Scalar::all( 2),
            cv::Scalar::all( 3),
            cv::Scalar::all( 4),
            cv::Scalar::all( 5),
            cv::Scalar::all( 6),
            cv::Scalar::all( 7),
            cv::Scalar::all( 8);

        Mask all_ones_mask(16, 1);
        all_ones_mask << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

        Mask all_zeros_mask(16, 1);
        all_zeros_mask << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        Mask general_mask(16, 1);
        general_mask << 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1;

        cv::Scalar constant_threshold_of_zeros(0, 0, 0, 0);
        cv::Scalar constant_threshold_of_threes(3, 3, 3, 3);
        cv::Scalar independent_channel_thresholds(2, 3, 4, 5);

        return {
            //  0
            {
                .matrix = matrix.clone(),
                .mask = all_zeros_mask,
                .threshold = constant_threshold_of_zeros,
                .expected = matrix.clone(),
            },
            //  1
            {
                .matrix = matrix.clone(),
                .mask = all_ones_mask,
                .threshold = constant_threshold_of_zeros,
                .expected = matrix.clone(),
            },
            //  2
            {
                .matrix = matrix.clone(),
                .mask = general_mask.clone(),
                .threshold = constant_threshold_of_zeros,
                .expected = matrix.clone(),
            },
            //  ----------------------------------------------------------------
            //  3
            {
                .matrix = matrix.clone(),
                .mask = all_zeros_mask,
                .threshold = constant_threshold_of_threes,
                .expected = matrix.clone(),
            },
            //  4
            {
                .matrix = matrix.clone(),
                .mask = all_ones_mask,
                .threshold = constant_threshold_of_threes,
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-7, -7, -7, -7),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-4, -4, -4, -4),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 5,  5,  5,  5),
                    cv::Scalar( 6,  6,  6,  6),
                    cv::Scalar( 7,  7,  7,  7),
                    cv::Scalar( 8,  8,  8,  8)
                ),
            },
            //  5
            {
                .matrix = matrix.clone(),
                .mask = general_mask.clone(),
                .threshold = constant_threshold_of_threes,
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-7, -7, -7, -7),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5, -5),
                    cv::Scalar(-4, -4, -4, -4),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar(-2, -2, -2, -2),
                    cv::Scalar(-1, -1, -1, -1),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 1,  1,  1,  1),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  4,  4),
                    cv::Scalar( 5,  5,  5,  5),
                    cv::Scalar( 6,  6,  6,  6),
                    cv::Scalar( 7,  7,  7,  7),
                    cv::Scalar( 8,  8,  8,  8)
                ),
            },
            //  ----------------------------------------------------------------
            //  6
            {
                .matrix = matrix.clone(),
                .mask = all_zeros_mask,
                .threshold = independent_channel_thresholds,
                .expected = matrix.clone(),
            },
            //  7
            {
                .matrix = matrix.clone(),
                .mask = all_ones_mask,
                .threshold = independent_channel_thresholds,
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-7, -7, -7, -7),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5,  0),
                    cv::Scalar(-4, -4,  0,  0),
                    cv::Scalar(-3,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 3,  0,  0,  0),
                    cv::Scalar( 4,  4,  0,  0),
                    cv::Scalar( 5,  5,  5,  0),
                    cv::Scalar( 6,  6,  6,  6),
                    cv::Scalar( 7,  7,  7,  7),
                    cv::Scalar( 8,  8,  8,  8)
                ),
            },
            //  8
            {
                .matrix = matrix.clone(),
                .mask = general_mask.clone(),
                .threshold = independent_channel_thresholds,
                .expected = (Matrix(16, 1) <<
                    cv::Scalar(-7, -7, -7, -7),
                    cv::Scalar(-6, -6, -6, -6),
                    cv::Scalar(-5, -5, -5,  0),
                    cv::Scalar(-4, -4,  0,  0),
                    cv::Scalar(-3,  0,  0,  0),
                    cv::Scalar(-2, -2, -2, -2),
                    cv::Scalar(-1, -1, -1, -1),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 1,  1,  1,  1),
                    cv::Scalar( 0,  0,  0,  0),
                    cv::Scalar( 3,  3,  3,  3),
                    cv::Scalar( 4,  4,  0,  0),
                    cv::Scalar( 5,  5,  5,  0),
                    cv::Scalar( 6,  6,  6,  6),
                    cv::Scalar( 7,  7,  7,  7),
                    cv::Scalar( 8,  8,  8,  8)
                ),
            },
        };
    }
};

TEST_P(MaskedHardThresholdTest, ColumnVectorThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 16);
    auto expected = param.expected.reshape(0, 16);
    auto mask = param.mask.reshape(0, 16);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, ColumnVectorThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 16);
    auto expected = param.expected.reshape(0, 16);
    auto mask = param.mask.reshape(0, 16);

    hard_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, RowVectorThresholdedCorrectly)
{
    auto param = GetParam();
    cv::Mat actual;
    auto matrix = param.matrix.reshape(0, 1);
    auto expected = param.expected.reshape(0, 1);
    auto mask = param.mask.reshape(0, 1);

    hard_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, RowVectorThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 1);
    auto expected = param.expected.reshape(0, 1);
    auto mask = param.mask.reshape(0, 1);

    hard_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, SquareMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 4);
    auto expected = param.expected.reshape(0, 4);
    auto mask = param.mask.reshape(0, 4);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, SquareMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 4);
    auto expected = param.expected.reshape(0, 4);
    auto mask = param.mask.reshape(0, 4);

    hard_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, TallMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 8);
    auto expected = param.expected.reshape(0, 8);
    auto mask = param.mask.reshape(0, 8);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, TallMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 8);
    auto expected = param.expected.reshape(0, 8);
    auto mask = param.mask.reshape(0, 8);

    hard_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, WideMatrixThresholdedCorrectly)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 2);
    auto expected = param.expected.reshape(0, 2);
    auto mask = param.mask.reshape(0, 2);

    cv::Mat actual;
    hard_threshold(matrix, actual, param.threshold, mask);

    EXPECT_THAT(actual, MatrixEq(expected));
}

TEST_P(MaskedHardThresholdTest, WideMatrixThresholdedCorrectlyInplace)
{
    auto param = GetParam();
    auto matrix = param.matrix.reshape(0, 2);
    auto expected = param.expected.reshape(0, 2);
    auto mask = param.mask.reshape(0, 2);

    hard_threshold(matrix, matrix, param.threshold, mask);

    EXPECT_THAT(matrix, MatrixEq(expected));
}


INSTANTIATE_TEST_CASE_P(
    HardThresholdGroup,
    MaskedHardThresholdTest,
    testing::ValuesIn(MaskedHardThresholdTest::create_params())
);


//  ============================================================================
//  Shrink
//  ============================================================================
template<typename T, int CHANNELS>
struct ShrinkTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    DWT2D::Coeffs coeffs;
    cv::Scalar threshold;
    int lower_level;
    int upper_level;
    DWT2D::Coeffs expected;
};

template<typename T, int CHANNELS>
void PrintTo(const ShrinkTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "coeffs =";
    PrintTo(param.coeffs, stream);
    *stream << "threshold = " << param.threshold << "\n";
    *stream << "lower_level = " << param.lower_level << "\n";
    *stream << "upper_level = " << param.upper_level << "\n";
}

template<typename ParamType>
class ShrinkTestBase : public testing::TestWithParam<ParamType>
{
public:
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;
    using Mask = cv::Mat_<uchar>;
    using LevelDetailValues = std::vector<std::vector<typename Pixel::value_type>>;
    using LevelDetailScalars = std::vector<std::vector<cv::Scalar>>;
    static const int LEVELS = 4;
    static const int ROWS = 16;
    static const int COLS = 16;

    template <typename T>
    static DWT2D::Coeffs create_level_coeffs(
        std::initializer_list<T> values,
        cv::Scalar approx_value = cv::Scalar(-2, -1, 1, 2)
    )
    {
        assert(values.size() == 3 * LEVELS);

        auto type = cv::traits::Type<Pixel>::value;
        DWT2D dwt(create_haar());
        auto coeffs = dwt.create_coeffs(ROWS, COLS, type, LEVELS);
        int i = 0;
        for (const auto& value : values) {
            int level = i / 3;
            int subband = i % 3;

            coeffs.set_detail(level, subband, cv::Scalar::all(value));
            ++i;
        }

        coeffs.set_approx(approx_value);

        return coeffs;
    }

    template<typename T>
    static DWT2D::Coeffs create_level_coeffs(
        std::initializer_list<std::initializer_list<T>> values,
        cv::Scalar approx_value = cv::Scalar(-2, -1, 1, 2)
    )
    {
        assert(values.size() == 3 * LEVELS);

        auto type = cv::traits::Type<Pixel>::value;
        DWT2D dwt(create_haar());
        auto coeffs = dwt.create_coeffs(ROWS, COLS, type, LEVELS);
        int i = 0;
        for (const auto& value : values) {
            assert(value.size() == Pixel::channels);

            cv::Scalar scalar_value;
            auto iter = value.begin();
            for (int j = 0; j < value.size(); ++j) {
                scalar_value[j] = *iter++;
                // ++iter;
            }

            int level = i / 3;
            int subband = i % 3;
            if (level != 3)
                coeffs.set_detail(level, subband, scalar_value);
            ++i;
        }

        coeffs.set_approx(approx_value);

        return coeffs;
    }
};

//  ----------------------------------------------------------------------------
//  Soft Shrink Details
//  ----------------------------------------------------------------------------
/**
 * TODO: negative thresholds
*/
class SoftShrinkTest : public ShrinkTestBase<ShrinkTestParam<double, 4>>
{
public:
    static std::vector<ParamType> create_params()
    {
        auto coeffs = create_level_coeffs({
            -5, -4, -3,
            -2, -1,  0,
             1,  2,  3,
             4,  5,  6,
        });
        auto constant_coeffs = create_level_coeffs({
            2, 2, 2,
            2, 2, 2,
            2, 2, 2,
            2, 2, 2,
        });

        return {
            //  0
            {
                .coeffs = coeffs.clone(),
                .threshold = cv::Scalar::all(0),
                .lower_level = 0,
                .upper_level = -1,
                .expected = coeffs.clone(),
            },
            //  1
            {
                .coeffs = coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = -1,
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  2
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = 0,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  3
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 1,
                .upper_level = 1,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  4
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 2,
                .upper_level = 2,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    2, 2, 2,
                    0, 0, 0,
                    2, 2, 2,
                }),
            },
            //  5
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 3,
                .upper_level = 3,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                    0, 0, 0,
                }),
            },
            //  6
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = 1,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  7
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 1,
                .upper_level = 2,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                }),
            },
            //  8
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 2,
                .upper_level = 3,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    2, 2, 2,
                    0, 0, 0,
                    0, 0, 0,
                }),
            },
            //  9
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = -2,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                }),
            },
            //  10
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = -3,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  11
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = -4,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  12
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = -3,
                .upper_level = -2,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                }),
            },
        };
    }
};

TEST_P(SoftShrinkTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs.clone();

    shrink_globally(
        coeffs,
        param.threshold,
        soft_threshold,
        cv::Range(param.lower_level, param.upper_level)
    );
    // soft_shrink_details(coeffs, param.threshold, param.lower_level, param.upper_level);

    EXPECT_THAT(coeffs, MatrixFloatEq(param.expected));
}

INSTANTIATE_TEST_CASE_P(
    ShrinkDetailsGroup,
    SoftShrinkTest,
    testing::ValuesIn(SoftShrinkTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Hard Shrink Details
//  ----------------------------------------------------------------------------
/**
 * TODO: negative thresholds
*/
class HardShrinkTest : public ShrinkTestBase<ShrinkTestParam<double, 4>>
{
public:
    static std::vector<ParamType> create_params()
    {
        auto coeffs = create_level_coeffs({
            -5, -4, -3,
            -2, -1,  0,
             1,  2,  3,
             4,  5,  6
        });
        auto constant_coeffs = create_level_coeffs({
            2, 2, 2,
            2, 2, 2,
            2, 2, 2,
            2, 2, 2,
        });

        return {
            //  0
            {
                .coeffs = coeffs.clone(),
                .threshold = cv::Scalar::all(0),
                .lower_level = 0,
                .upper_level = -1,
                .expected = coeffs.clone(),
            },
            //  1
            {
                .coeffs = coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = -1,
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  2
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = 0,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  3
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 1,
                .upper_level = 1,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  4
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 2,
                .upper_level = 2,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    2, 2, 2,
                    0, 0, 0,
                    2, 2, 2,
                }),
            },
            //  5
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 3,
                .upper_level = 3,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                    0, 0, 0,
                }),
            },
            //  6
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = 1,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  7
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 1,
                .upper_level = 2,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                }),
            },
            //  8
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 2,
                .upper_level = 3,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    2, 2, 2,
                    0, 0, 0,
                    0, 0, 0,
                }),
            },
            //  9
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = -2,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                }),
            },
            //  10
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = -3,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  11
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = 0,
                .upper_level = -4,
                .expected = create_level_coeffs({
                    0, 0, 0,
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                }),
            },
            //  12
            {
                .coeffs = constant_coeffs.clone(),
                .threshold = cv::Scalar::all(2),
                .lower_level = -3,
                .upper_level = -2,
                .expected = create_level_coeffs({
                    2, 2, 2,
                    0, 0, 0,
                    0, 0, 0,
                    2, 2, 2,
                }),
            },
        };
    }
};

TEST_P(HardShrinkTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs.clone();

    shrink_globally(
        coeffs,
        param.threshold,
        hard_threshold,
        cv::Range(param.lower_level, param.upper_level)
    );
    // hard_shrink_details(coeffs, param.threshold, param.lower_level, param.upper_level);

    EXPECT_THAT(coeffs, MatrixFloatEq(param.expected));
}

INSTANTIATE_TEST_CASE_P(
    ShrinkDetailsGroup,
    HardShrinkTest,
    testing::ValuesIn(HardShrinkTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Soft Shrink Levels
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct ShrinkLevelsTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    DWT2D::Coeffs coeffs;
    std::vector<cv::Scalar> thresholds;
    DWT2D::Coeffs expected;
};

template<typename T, int CHANNELS>
void PrintTo(const ShrinkLevelsTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "coeffs =";
    PrintTo(param.coeffs, stream);
    *stream << "thresholds = " << join(param.thresholds, ", ") << "\n";
}

/**
 * TODO: negative thresholds
*/
class SoftShrinkLevelsTest : public ShrinkTestBase<ShrinkLevelsTestParam<double, 4>>
{
public:
    static std::vector<ParamType> create_params()
    {
        auto coeffs = create_level_coeffs({
            -5, -4, -3,
            -2, -1,  0,
             1,  2,  3,
             4,  5,  6
        });
        auto indenpendent_channel_coeffs = create_level_coeffs({
            {-8, -7, -6, -5}, {-7, -6, -5, -4}, {-6, -5, -4, -3},
            {-4, -3, -2, -1}, {-3, -2, -1,  0}, {-2, -1,  0,  1},
            { 0,  1,  2,  3}, { 1,  2,  3,  4}, { 2,  3,  4,  5},
            { 4,  5,  6,  7}, { 5,  6,  7,  8}, { 6,  7,  8,  9}
        });

        return {
            //  0
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                },
                .expected = coeffs.clone(),
            },
            //  1
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  2
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(4),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -1,  0,  0,
                     0,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  3
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -3, -2, -1,
                    -1,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  4
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  1,  2,
                     2,  3,  4,
                }),
            },
            //  5
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(4),
                },
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  0,  1,
                     0,  1,  2,
                }),
            },
            //  6
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(4),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -1,  0,  0,
                    -1,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  7
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(3),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -3, -2, -1,
                    -1,  0,  0,
                     0,  0,  0,
                     2,  3,  4,
                }),
            },
            //  8
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(4),
                },
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  1,  2,
                     0,  1,  2,
                }),
            },
            //  9
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(4),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -1,  0,  0,
                     0,  0,  0,
                     0,  1,  2,
                     2,  3,  4,
                }),
            },
            //  10
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(4),
                },
                .expected = create_level_coeffs({
                    -3, -2, -1,
                    -1,  0,  0,
                     0,  0,  1,
                     0,  1,  2,
                }),
            },
            //  11
            {
                .coeffs = indenpendent_channel_coeffs.clone(),
                .thresholds = {
                    cv::Scalar(6, 6, 5, 5),
                    cv::Scalar(4, 2, 1, 0),
                    cv::Scalar(2, 2, 2, 2),
                    cv::Scalar(0, 0, 0, 0),
                },
                .expected = create_level_coeffs({
                    {-2, -1, -1,  0}, {-1,  0,  0, 0}, {0, 0, 0, 0},
                    { 0, -1, -1, -1}, { 0,  0,  0, 0}, {0, 0, 0, 1},
                    { 0,  0,  0,  1}, { 0,  0,  1, 2}, {0, 1, 2, 3},
                    { 4,  5,  6,  7}, { 5,  6,  7, 8}, {6, 7, 8, 9},
                }),
            },
        };
    }
};

TEST_P(SoftShrinkLevelsTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs.clone();

    shrink_levels(
        coeffs,
        param.thresholds,
        soft_threshold
        // cv::Range(param.lower_level, param.upper_level)
    );
    // soft_shrink_detail_levels(coeffs, param.thresholds);

    EXPECT_THAT(coeffs, MatrixFloatEq(param.expected));
}

INSTANTIATE_TEST_CASE_P(
    ShrinkDetailsGroup,
    SoftShrinkLevelsTest,
    testing::ValuesIn(SoftShrinkLevelsTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Hard Shrink Levels
//  ----------------------------------------------------------------------------
/**
 * TODO: negative thresholds
*/
class HardShrinkLevelsTest : public ShrinkTestBase<ShrinkLevelsTestParam<double, 4>>
{
public:
    static std::vector<ParamType> create_params()
    {
        auto coeffs = create_level_coeffs({
            -5, -4, -3,
            -2, -1,  0,
             1,  2,  3,
             4,  5,  6
        });
        auto indenpendent_channel_coeffs = create_level_coeffs({
            {-8, -7, -6, -5}, {-7, -6, -5, -4}, {-6, -5, -4, -3},
            {-4, -3, -2, -1}, {-3, -2, -1,  0}, {-2, -1,  0,  1},
            { 0,  1,  2,  3}, { 1,  2,  3,  4}, { 2,  3,  4,  5},
            { 4,  5,  6,  7}, { 5,  6,  7,  8}, { 6,  7,  8,  9}
        });

        return {
            //  0
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                },
                .expected = coeffs.clone(),
            },
            //  1
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  2
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(4),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -5,  0,  0,
                     0,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  3
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -5, -4, -3,
                    -2,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  4
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  2,  3,
                     4,  5,  6,
                }),
            },
            //  5
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(4),
                },
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  0,  3,
                     0,  5,  6,
                }),
            },
            //  6
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(4),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -5,  0,  0,
                    -2,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  7
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(3),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -5, -4, -3,
                    -2,  0,  0,
                     0,  0,  0,
                     4,  5,  6,
                }),
            },
            //  8
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(4),
                },
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  2,  3,
                     0,  5,  6,
                }),
            },
            //  9
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(4),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                },
                .expected = create_level_coeffs({
                    -5,  0,  0,
                     0,  0,  0,
                     0,  2,  3,
                     4,  5,  6,
                }),
            },
            //  10
            {
                .coeffs = coeffs.clone(),
                .thresholds = {
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(4),
                },
                .expected = create_level_coeffs({
                    -5, -4, -3,
                    -2,  0,  0,
                     0,  0,  3,
                     0,  5,  6,
                }),
            },
            //  11
            {
                .coeffs = indenpendent_channel_coeffs.clone(),
                .thresholds = {
                    cv::Scalar(6, 6, 5, 5),
                    cv::Scalar(4, 2, 1, 0),
                    cv::Scalar(2, 2, 2, 2),
                    cv::Scalar(0, 0, 0, 0),
                },
                .expected = create_level_coeffs({
                    {-8, -7, -6,  0}, {-7, 0, 0, 0}, {0, 0, 0, 0},
                    { 0, -3, -2, -1}, { 0, 0, 0, 0}, {0, 0, 0, 1},
                    { 0,  0,  0,  3}, { 0, 0, 3, 4}, {0, 3, 4, 5},
                    { 4,  5,  6,  7}, { 5, 6, 7, 8}, {6, 7, 8, 9},
                }),
            },
        };
    }
};

TEST_P(HardShrinkLevelsTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs;

    shrink_levels(
        coeffs,
        param.thresholds,
        hard_threshold
        // cv::Range(param.lower_level, param.upper_level)
    );
    // hard_shrink_detail_levels(coeffs, param.thresholds);

    EXPECT_THAT(coeffs, MatrixFloatEq(param.expected));
}

INSTANTIATE_TEST_CASE_P(
    ShrinkDetailsGroup,
    HardShrinkLevelsTest,
    testing::ValuesIn(HardShrinkLevelsTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Soft Shrink Subbands
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct ShrinkSubbandsTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    DWT2D::Coeffs coeffs;
    cv::Mat thresholds;
    DWT2D::Coeffs expected;
};

template<typename T, int CHANNELS>
void PrintTo(const ShrinkSubbandsTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "coeffs =";
    PrintTo(param.coeffs, stream);
    *stream << "thresholds =";
    PrintTo(param.thresholds, stream);
}

/**
 * TODO: negative thresholds
*/
class SoftShrinkSubbandsTest : public ShrinkTestBase<ShrinkSubbandsTestParam<double, 4>>
{
public:
    static std::vector<ParamType> create_params()
    {
        auto coeffs = create_level_coeffs({
            -5, -4, -3,
            -2, -1,  0,
             1,  2,  3,
             4,  5,  6
        });
        auto indenpendent_channel_coeffs = create_level_coeffs({
            {-8, -7, -6, -5}, {-7, -6, -5, -4}, {-6, -5, -4, -3},
            {-4, -3, -2, -1}, {-3, -2, -1,  0}, {-2, -1,  0,  1},
            { 0,  1,  2,  3}, { 1,  2,  3,  4}, { 2,  3,  4,  5},
            { 4,  5,  6,  7}, { 5,  6,  7,  8}, { 6,  7,  8,  9}
        });

        return {
            //  0
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(0), cv::Scalar::all(0), cv::Scalar::all(0),
                    cv::Scalar::all(0), cv::Scalar::all(0), cv::Scalar::all(0),
                    cv::Scalar::all(0), cv::Scalar::all(0), cv::Scalar::all(0),
                    cv::Scalar::all(0), cv::Scalar::all(0), cv::Scalar::all(0)
                ),
                .expected = coeffs.clone(),
            },
            //  1
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  2
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -1,  0,  0,
                     0,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  3
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -3, -2, -1,
                    -1,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  4
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  1,  2,
                     2,  3,  4,
                }),
            },
            //  5
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4)
                ),
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  0,  1,
                     0,  1,  2,
                }),
            },
            //  6
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -1,  0,  0,
                    -1,  0,  0,
                     0,  0,  1,
                     2,  3,  4,
                }),
            },
            //  7
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(3), cv::Scalar::all(3), cv::Scalar::all(3),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -3, -2, -1,
                    -1,  0,  0,
                     0,  0,  0,
                     2,  3,  4,
                }),
            },
            //  8
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4)
                ),
                .expected = create_level_coeffs({
                    -3, -2, -1,
                     0,  0,  0,
                     0,  1,  2,
                     0,  1,  2,
                }),
            },
            //  9
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -1,  0,  0,
                     0,  0,  0,
                     0,  1,  2,
                     2,  3,  4,
                }),
            },
            //  10
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4)
                ),
                .expected = create_level_coeffs({
                    -3, -2, -1,
                    -1,  0,  0,
                     0,  0,  1,
                     0,  1,  2,
                }),
            },
            //  11
            {
                .coeffs = indenpendent_channel_coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar(6, 6, 5, 5), cv::Scalar(6, 6, 5, 5), cv::Scalar(6, 6, 5, 5),
                    cv::Scalar(4, 2, 1, 0), cv::Scalar(4, 2, 1, 0), cv::Scalar(4, 2, 1, 0),
                    cv::Scalar(2, 2, 2, 2), cv::Scalar(2, 2, 2, 2), cv::Scalar(2, 2, 2, 2),
                    cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 0)
                ),
                .expected = create_level_coeffs({
                    {-2, -1, -1,  0}, {-1,  0,  0, 0}, {0, 0, 0, 0},
                    { 0, -1, -1, -1}, { 0,  0,  0, 0}, {0, 0, 0, 1},
                    { 0,  0,  0,  1}, { 0,  0,  1, 2}, {0, 1, 2, 3},
                    { 4,  5,  6,  7}, { 5,  6,  7, 8}, {6, 7, 8, 9},
                }),
            },
        };
    }
};

TEST_P(SoftShrinkSubbandsTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs;

    shrink_subbands(
        coeffs,
        param.thresholds,
        soft_threshold
        // cv::Range(param.lower_level, param.upper_level)
    );
    // soft_shrink_detail_subbands(coeffs, param.thresholds);

    EXPECT_THAT(coeffs, MatrixFloatEq(param.expected));
}

INSTANTIATE_TEST_CASE_P(
    ShrinkDetailsGroup,
    SoftShrinkSubbandsTest,
    testing::ValuesIn(SoftShrinkSubbandsTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Hard Shrink Subbands
//  ----------------------------------------------------------------------------
/**
 * TODO: negative thresholds
*/
class HardShrinkSubbandsTest : public ShrinkTestBase<ShrinkSubbandsTestParam<double, 4>>
{
public:
    static std::vector<ParamType> create_params()
    {
        auto coeffs = create_level_coeffs({
            -5, -4, -3,
            -2, -1,  0,
             1,  2,  3,
             4,  5,  6
        });
        auto indenpendent_channel_coeffs = create_level_coeffs({
            {-8, -7, -6, -5}, {-7, -6, -5, -4}, {-6, -5, -4, -3},
            {-4, -3, -2, -1}, {-3, -2, -1,  0}, {-2, -1,  0,  1},
            { 0,  1,  2,  3}, { 1,  2,  3,  4}, { 2,  3,  4,  5},
            { 4,  5,  6,  7}, { 5,  6,  7,  8}, { 6,  7,  8,  9}
        });

        return {
            //  0
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(0), cv::Scalar::all(0), cv::Scalar::all(0),
                    cv::Scalar::all(0), cv::Scalar::all(0), cv::Scalar::all(0),
                    cv::Scalar::all(0), cv::Scalar::all(0), cv::Scalar::all(0),
                    cv::Scalar::all(0), cv::Scalar::all(0), cv::Scalar::all(0)
                ),
                .expected = coeffs.clone(),
            },
            //  1
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  2
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -5,  0,  0,
                     0,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  3
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -5, -4, -3,
                    -2,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  4
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  2,  3,
                     4,  5,  6,
                }),
            },
            //  5
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4)
                ),
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  0,  3,
                     0,  5,  6,
                }),
            },
            //  6
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -5,  0,  0,
                    -2,  0,  0,
                     0,  0,  3,
                     4,  5,  6,
                }),
            },
            //  7
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(3), cv::Scalar::all(3), cv::Scalar::all(3),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -5, -4, -3,
                    -2,  0,  0,
                     0,  0,  0,
                     4,  5,  6,
                }),
            },
            //  8
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4)
                ),
                .expected = create_level_coeffs({
                    -5, -4, -3,
                     0,  0,  0,
                     0,  2,  3,
                     0,  5,  6,
                }),
            },
            //  9
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2)
                ),
                .expected = create_level_coeffs({
                    -5,  0,  0,
                     0,  0,  0,
                     0,  2,  3,
                     4,  5,  6,
                }),
            },
            //  10
            {
                .coeffs = coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(1), cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(2), cv::Scalar::all(2), cv::Scalar::all(2),
                    cv::Scalar::all(4), cv::Scalar::all(4), cv::Scalar::all(4)
                ),
                .expected = create_level_coeffs({
                    -5, -4, -3,
                    -2,  0,  0,
                     0,  0,  3,
                     0,  5,  6,
                }),
            },
            //  11
            {
                .coeffs = indenpendent_channel_coeffs.clone(),
                .thresholds = (Matrix(4, 3) <<
                    cv::Scalar(6, 6, 5, 5), cv::Scalar(6, 6, 5, 5), cv::Scalar(6, 6, 5, 5),
                    cv::Scalar(4, 2, 1, 0), cv::Scalar(4, 2, 1, 0), cv::Scalar(4, 2, 1, 0),
                    cv::Scalar(2, 2, 2, 2), cv::Scalar(2, 2, 2, 2), cv::Scalar(2, 2, 2, 2),
                    cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 0)
                ),
                .expected = create_level_coeffs({
                    {-8, -7, -6,  0}, {-7, 0, 0, 0}, {0, 0, 0, 0},
                    { 0, -3, -2, -1}, { 0, 0, 0, 0}, {0, 0, 0, 1},
                    { 0,  0,  0,  3}, { 0, 0, 3, 4}, {0, 3, 4, 5},
                    { 4,  5,  6,  7}, { 5, 6, 7, 8}, {6, 7, 8, 9},
                }),
            },
        };
    }
};

TEST_P(HardShrinkSubbandsTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs;

    shrink_subbands(
        coeffs,
        param.thresholds,
        hard_threshold
        // cv::Range(param.lower_level, param.upper_level)
    );
    // hard_shrink_detail_subbands(coeffs, param.thresholds);

    EXPECT_THAT(coeffs, MatrixFloatEq(param.expected));
}

INSTANTIATE_TEST_CASE_P(
    ShrinkDetailsGroup,
    HardShrinkSubbandsTest,
    testing::ValuesIn(HardShrinkSubbandsTest::create_params())
);




//  ============================================================================
//  Sure Test
//  ============================================================================
template<typename T, int CHANNELS>
struct SureTestBaseParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;
    cv::Mat matrix;
    cv::Scalar stdev;
};

template<typename ParamType>
class SureTestBase : public testing::TestWithParam<ParamType>
{
public:
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;
    using BaseParamsMap = std::map< std::string, SureTestBaseParam<typename Pixel::value_type, Pixel::channels>>;

    static BaseParamsMap create_base_params()
    {
        return {
            {
                "zero_constant",
                {
                    .matrix = (Matrix(16, 1) <<
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0),
                        cv::Scalar::all(0)),
                    .stdev = cv::Scalar::all(1),
                },
            },
            {
                "nonzero_constant",
                {
                    .matrix = (Matrix(16, 1) <<
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1),
                        cv::Scalar::all(1)),
                    .stdev = cv::Scalar::all(1),
                },
            },
            {
                "not_zero_crossing_increasing",
                {
                    .matrix = (Matrix(16, 1) <<
                        cv::Scalar::all(1),
                        cv::Scalar::all(2),
                        cv::Scalar::all(3),
                        cv::Scalar::all(4),
                        cv::Scalar::all(5),
                        cv::Scalar::all(6),
                        cv::Scalar::all(7),
                        cv::Scalar::all(8),
                        cv::Scalar::all(9),
                        cv::Scalar::all(10),
                        cv::Scalar::all(11),
                        cv::Scalar::all(12),
                        cv::Scalar::all(13),
                        cv::Scalar::all(14),
                        cv::Scalar::all(15),
                        cv::Scalar::all(16)),
                    .stdev = cv::Scalar::all(4.6097722286464435),
                },
            },
            {
                "zero_crossing_increasing",
                {
                    .matrix = (Matrix(16, 1) <<
                        cv::Scalar::all(-7),
                        cv::Scalar::all(-6),
                        cv::Scalar::all(-5),
                        cv::Scalar::all(-4),
                        cv::Scalar::all(-3),
                        cv::Scalar::all(-2),
                        cv::Scalar::all(-1),
                        cv::Scalar::all(0),
                        cv::Scalar::all(1),
                        cv::Scalar::all(2),
                        cv::Scalar::all(3),
                        cv::Scalar::all(4),
                        cv::Scalar::all(5),
                        cv::Scalar::all(6),
                        cv::Scalar::all(7),
                        cv::Scalar::all(8)),
                    .stdev = cv::Scalar::all(4.6097722286464435),
                },
            },
            {
                "independent_channels",
                {
                    .matrix = (Matrix(16, 1) <<
                        cv::Scalar(-7,  1, 0, 1),
                        cv::Scalar(-6,  2, 0, 1),
                        cv::Scalar(-5,  3, 0, 1),
                        cv::Scalar(-4,  4, 0, 1),
                        cv::Scalar(-3,  5, 0, 1),
                        cv::Scalar(-2,  6, 0, 1),
                        cv::Scalar(-1,  7, 0, 1),
                        cv::Scalar( 0,  8, 0, 1),
                        cv::Scalar( 1,  9, 0, 1),
                        cv::Scalar( 2, 10, 0, 1),
                        cv::Scalar( 3, 11, 0, 1),
                        cv::Scalar( 4, 12, 0, 1),
                        cv::Scalar( 5, 13, 0, 1),
                        cv::Scalar( 6, 14, 0, 1),
                        cv::Scalar( 7, 15, 0, 1),
                        cv::Scalar( 8, 16, 0, 1)),
                    .stdev = cv::Scalar(4.6097722286464435, 4.6097722286464435, 1, 1),
                },
            },
        };
    }
};


//  ----------------------------------------------------------------------------
//  Sure Risk
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct SureRiskTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;
    cv::Mat matrix;
    cv::Scalar stdev;
    cv::Scalar threshold;
    cv::Scalar expected_risk;
};

template<typename T, int CHANNELS>
void PrintTo(const SureRiskTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
    *stream << "stdev = " << param.stdev << "\n";
    *stream << "threshold = " << param.threshold << "\n";
}

class SureRiskTest : public SureTestBase<SureRiskTestParam<double, 4>>
{
public:
    static constexpr double tolerance = 1e-12;

    auto compute_sure_risk(const cv::Mat& matrix, const cv::Scalar& threshold, const cv::Scalar& stdev)
    {
        cvwt::internal::ComputeSureThreshold<
            SureRiskTest::Pixel::value_type,
            SureRiskTest::Pixel::channels
        > compute;

        cv::Scalar risk;
        compute.sure_risk(matrix, threshold, stdev, risk);

        return risk;
    }

    static std::vector<ParamType> create_params()
    {
        auto base_params = create_base_params();
        return {
            //  ----------------------------------------------------------------
            //  zero constant
            //  0
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(-1.0),
                .expected_risk = cv::Scalar::all(32),
            },
            //  1
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(-0.75),
                .expected_risk = cv::Scalar::all(25),
            },
            //  2
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(-0.5),
                .expected_risk = cv::Scalar::all(20),
            },
            //  3
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(-0.25),
                .expected_risk = cv::Scalar::all(17),
            },
            //  4
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(0.0),
                .expected_risk = cv::Scalar::all(-16),
            },
            //  5
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(0.25),
                .expected_risk = cv::Scalar::all(-16),
            },
            //  6
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(0.5),
                .expected_risk = cv::Scalar::all(-16),
            },
            //  7
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(0.75),
                .expected_risk = cv::Scalar::all(-16),
            },
            //  8
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .threshold = cv::Scalar::all(1.0),
                .expected_risk = cv::Scalar::all(-16),
            },
            //  ----------------------------------------------------------------
            //  nonzero constant
            //  9
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(0),
                .expected_risk = cv::Scalar::all(16),
            },
            //  10
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(0.25),
                .expected_risk = cv::Scalar::all(17),
            },
            //  11
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(0.5),
                .expected_risk = cv::Scalar::all(20),
            },
            //  12
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(0.75),
                .expected_risk = cv::Scalar::all(25),
            },
            //  13
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(1.0),
                .expected_risk = cv::Scalar::all(0),
            },
            //  14
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(1.25),
                .expected_risk = cv::Scalar::all(0),
            },
            //  15
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(1.5),
                .expected_risk = cv::Scalar::all(0),
            },
            //  16
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(1.75),
                .expected_risk = cv::Scalar::all(0),
            },
            //  17
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .threshold = cv::Scalar::all(2.0),
                .expected_risk = cv::Scalar::all(0),
            },
            //  ----------------------------------------------------------------
            //  not zero crossing, increasing
            //  18
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(0),
                .expected_risk = cv::Scalar::all(16),
            },
            //  19
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(0.5),
                .expected_risk = cv::Scalar::all(16.188235294117646),
            },
            //  20
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(1),
                .expected_risk = cv::Scalar::all(14.75294117647059),
            },
            //  21
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(1.5),
                .expected_risk = cv::Scalar::all(15.63529411764706),
            },
            //  22
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(7.5),
                .expected_risk = cv::Scalar::all(32.411764705882355),
            },
            //  23
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(8),
                .expected_risk = cv::Scalar::all(33.69411764705882),
            },
            //  24
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(8.5),
                .expected_risk = cv::Scalar::all(36.8),
            },
            //  25
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(15.5),
                .expected_risk = cv::Scalar::all(55.65882352941176),
            },
            //  26
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(16),
                .expected_risk = cv::Scalar::all(54.4),
            },
            //  27
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(16.5),
                .expected_risk = cv::Scalar::all(54.4),
            },
            //  ----------------------------------------------------------------
            //  zero crossing, increasing
            //  28
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(-7.5),
                .expected_risk = cv::Scalar::all(58.352941176470594),
            },
            //  29
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(-7),
                .expected_risk = cv::Scalar::all(52.89411764705882),
            },
            //  30
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(-6.5),
                .expected_risk = cv::Scalar::all(47.811764705882354),
            },
            //  31
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(-0.5),
                .expected_risk = cv::Scalar::all(16.188235294117646),
            },
            //  32
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(0),
                .expected_risk = cv::Scalar::all(14.0),
            },
            //  33
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(0.5),
                .expected_risk = cv::Scalar::all(14.176470588235293),
            },
            //  34
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(7.5),
                .expected_risk = cv::Scalar::all(1.823529411764703),
            },
            //  35
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(8),
                .expected_risk = cv::Scalar::all(0.1882352941176464),
            },
            //  36
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .threshold = cv::Scalar::all(8.5),
                .expected_risk = cv::Scalar::all(0.1882352941176464),
            },
            //  ----------------------------------------------------------------
            //  independent channels
            //  37
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(-7.5, 0.5, -1.0, 0.0),
                .expected_risk = cv::Scalar(58.352941176470594, 16.188235294117646, 32, 16),
            },
            //  38
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(-7.0, 1.0, -0.75, 0.25),
                .expected_risk = cv::Scalar(52.89411764705882, 14.75294117647059, 25, 17),
            },
            //  39
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(-6.5, 1.5, -0.5, 0.5),
                .expected_risk = cv::Scalar(47.811764705882354, 15.63529411764706, 20, 20),
            },
            //  40
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(-0.5, 7.5, -0.25, 0.75),
                .expected_risk = cv::Scalar(16.188235294117646, 32.411764705882355, 17, 25),
            },
            //  41
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(0.0, 8.0, 0.0, 1.0),
                .expected_risk = cv::Scalar(14.0, 33.69411764705882, -16, 0),
            },
            //  42
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(0.5, 8.5, 0.25, 1.25),
                .expected_risk = cv::Scalar(14.176470588235293, 36.8, -16, 0),
            },
            //  43
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(7.5, 15.5, 0.5, 1.5),
                .expected_risk = cv::Scalar(1.823529411764703, 55.65882352941176, -16, 0),
            },
            //  44
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(8.0, 16.0, 0.75, 1.75),
                .expected_risk = cv::Scalar(0.1882352941176464, 54.4, -16, 0),
            },
            //  45
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .threshold = cv::Scalar(8.5, 16.5, 1.0, 2.0),
                .expected_risk = cv::Scalar(0.1882352941176464, 54.4, -16, 0),
            },
        };
    }
};

TEST_P(SureRiskTest, CorrectRisk)
{
    auto param = GetParam();

    auto actual_risk = compute_sure_risk(param.matrix, param.threshold, param.stdev);

    EXPECT_THAT(actual_risk, ScalarNear(param.expected_risk, SureRiskTest::tolerance));
}

TEST_P(SureRiskTest, InvariantUnderNegation)
{
    auto param = GetParam();
    auto matrix = -param.matrix;

    auto actual_risk = compute_sure_risk(matrix, param.threshold, param.stdev);

    EXPECT_THAT(actual_risk, ScalarNear(param.expected_risk, SureRiskTest::tolerance));
}

TEST_P(SureRiskTest, InvariantUnderReversal)
{
    auto param = GetParam();
    cv::Mat matrix;
    cv::flip(param.matrix, matrix, -1);

    auto actual_risk = compute_sure_risk(matrix, param.threshold, param.stdev);

    EXPECT_THAT(
        actual_risk,
        ScalarNear(param.expected_risk, SureRiskTest::tolerance)
    );
}

TEST_P(SureRiskTest, InvariantUnderPermutation1)
{
    auto param = GetParam();
    auto matrix = permute_matrix(param.matrix, PERMUTATION1);

    auto actual_risk = compute_sure_risk(matrix, param.threshold, param.stdev);

    EXPECT_THAT(
        actual_risk,
        ScalarNear(param.expected_risk, SureRiskTest::tolerance)
    ) << "where permuation = [" << join(PERMUTATION1, ", ") << "]";
}

TEST_P(SureRiskTest, InvariantUnderPermutation2)
{
    auto param = GetParam();
    auto matrix = permute_matrix(param.matrix, PERMUTATION2);

    auto actual_risk = compute_sure_risk(matrix, param.threshold, param.stdev);

    EXPECT_THAT(
        actual_risk,
        ScalarNear(param.expected_risk, SureRiskTest::tolerance)
    ) << "where permuation = [" << join(PERMUTATION2, ", ") << "]";
}

TEST_P(SureRiskTest, ConsistentWithPrescalingByStdDev)
{
    auto param = GetParam();
    cv::Mat matrix;
    cv::divide(param.matrix, param.stdev, matrix);
    cv::Scalar threshold;
    cv::divide(param.threshold, param.stdev, threshold);

    auto actual_risk = compute_sure_risk(matrix, threshold, cv::Scalar::all(1.0));

    EXPECT_THAT(actual_risk, ScalarNear(param.expected_risk, SureRiskTest::tolerance));
}

INSTANTIATE_TEST_CASE_P(
    SureShrinkGroup,
    SureRiskTest,
    testing::ValuesIn(SureRiskTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Sure Threshold
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct SureThresholdTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;
    cv::Mat matrix;
    cv::Scalar stdev;
    SureShrink::Optimizer optimizer;
    cv::Scalar expected_threshold;
};

template<typename T, int CHANNELS>
void PrintTo(const SureThresholdTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
    *stream << "optimizer =" << param.optimizer << "\n";
    *stream << "stdev = " << param.stdev << "\n";
}

class SureThresholdTest : public SureTestBase<SureThresholdTestParam<double, 4>>
{
protected:
    SureThresholdTest() :
        SureTestBase<SureThresholdTestParam<double, 4>>(),
        shrinker(SureShrink::STRICT)
    {
    }

    SureShrink shrinker;

public:
    static std::vector<ParamType> create_params()
    {
        auto base_params = create_base_params();
        std::vector<ParamType> brute_force_params = {
            //  0, 1
            {
                .matrix = base_params["zero_constant"].matrix,
                .stdev = base_params["zero_constant"].stdev,
                .optimizer = SureShrink::BRUTE_FORCE,
                .expected_threshold = cv::Scalar::all(0.0),
            },
            //  2, 3
            {
                .matrix = base_params["nonzero_constant"].matrix,
                .stdev = base_params["nonzero_constant"].stdev,
                .optimizer = SureShrink::BRUTE_FORCE,
                .expected_threshold = cv::Scalar::all(1.0),
            },
            //  4, 5
            {
                .matrix = base_params["not_zero_crossing_increasing"].matrix,
                .stdev = base_params["not_zero_crossing_increasing"].stdev,
                .optimizer = SureShrink::BRUTE_FORCE,
                .expected_threshold = cv::Scalar::all(1.0),
            },
            //  6, 7
            {
                .matrix = base_params["zero_crossing_increasing"].matrix,
                .stdev = base_params["zero_crossing_increasing"].stdev,
                .optimizer = SureShrink::BRUTE_FORCE,
                .expected_threshold = cv::Scalar::all(8.0),
            },
            //  8, 9
            {
                .matrix = base_params["independent_channels"].matrix,
                .stdev = base_params["independent_channels"].stdev,
                .optimizer = SureShrink::BRUTE_FORCE,
                .expected_threshold = cv::Scalar(8.0, 1.0, 0.0, 1.0),
            },
        };

        auto replace_optimizer = [](ParamType param, SureShrink::Optimizer optimizer) {
            param.optimizer = optimizer;
            return param;
        };

        std::vector<ParamType> params;
        for (auto& param : brute_force_params) {
            params.push_back(param);
            params.push_back(replace_optimizer(param, SureShrink::NELDER_MEAD));
        }

        return params;
    }
};

TEST_P(SureThresholdTest, CorrectThreshold)
{
    auto param = GetParam();

    auto actual_threshold = shrinker.compute_sure_threshold(param.matrix, param.stdev);

    EXPECT_THAT(actual_threshold, ScalarDoubleEq(param.expected_threshold));
}

TEST_P(SureThresholdTest, InvariantUnderReversal)
{
    auto param = GetParam();
    cv::Mat matrix;
    cv::flip(param.matrix, matrix, -1);

    auto actual_threshold = shrinker.compute_sure_threshold(matrix, param.stdev);

    EXPECT_THAT(
        actual_threshold,
        ScalarDoubleEq(param.expected_threshold)
    );
}

TEST_P(SureThresholdTest, InvariantUnderPermutation1)
{
    auto param = GetParam();
    auto matrix = permute_matrix(param.matrix, PERMUTATION1);

    auto actual_threshold = shrinker.compute_sure_threshold(matrix, param.stdev);

    EXPECT_THAT(
        actual_threshold,
        ScalarDoubleEq(param.expected_threshold)
    ) << "where permuation = [" << join(PERMUTATION1, ", ") << "]";
}

TEST_P(SureThresholdTest, InvariantUnderPermutation2)
{
    auto param = GetParam();
    auto matrix = permute_matrix(param.matrix, PERMUTATION2);

    auto actual_threshold = shrinker.compute_sure_threshold(matrix, param.stdev);

    EXPECT_THAT(
        actual_threshold,
        ScalarDoubleEq(param.expected_threshold)
    ) << "where permuation = [" << join(PERMUTATION2, ", ") << "]";
}

TEST_P(SureThresholdTest, ConsistentWithPrescalingByStdDev)
{
    auto param = GetParam();
    cv::Mat matrix;
    cv::divide(param.matrix, param.stdev, matrix);
    cv::Scalar expected_threshold;
    cv::divide(param.expected_threshold, param.stdev, expected_threshold);

    auto actual_threshold = shrinker.compute_sure_threshold(matrix, cv::Scalar::all(1.0));

    EXPECT_THAT(actual_threshold, ScalarDoubleEq(expected_threshold));
}


INSTANTIATE_TEST_CASE_P(
    SureShrinkGroup,
    SureThresholdTest,
    testing::ValuesIn(SureThresholdTest::create_params())
);





//  ----------------------------------------------------------------------------
//  Bayes Threshold
//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct BayesThresholdTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;
    cv::Mat matrix;
    Shrink::Partition partition;
    cv::Scalar expected_stdev;
    cv::Mat4d expected_thresholds;
};

template<typename T, int CHANNELS>
void PrintTo(const BayesThresholdTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "matrix =";
    PrintTo(param.matrix, stream);
    *stream << "partition = " << param.partition << "\n";
    *stream << "expected_stdev = " << param.expected_stdev << "\n";
    *stream << "expected_thresholds = " << param.expected_thresholds << "\n";
}


class BayesThresholdTest : public testing::TestWithParam<BayesThresholdTestParam<double, 4>>
{
public:
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;

    const int NEARNESS_TOLERANCE = 1e-14;
    const int LEVELS = 3;

protected:
    BayesThresholdTest() :
        testing::TestWithParam<ParamType>(),
        shrinker(GetParam().partition),
        dwt(Wavelet::create("haar"))
    {
    }

    void SetUp() override
    {
        DWT2D dwt(Wavelet::create("haar"));
        auto param = GetParam();
        coeffs = dwt.create_coeffs(param.matrix, param.matrix.size(), LEVELS);
    }

    BayesShrink shrinker;
    DWT2D dwt;
    DWT2D::Coeffs coeffs;

public:
    static std::string param_name(int index)
    {
        switch (index) {
        case 0: return "linear_subbands";
        case 1: return "linear_levels";
        case 2: return "linear_globally";
        // case 3: return "random_subbands";
        // case 4: return "random_levels";
        // case 5: return "random_globally";
        }

        return "";
    }

    static std::vector<ParamType> create_params()
    {
        cv::Mat linear_space_matrix = (Matrix(16, 16) <<
            //  row 0
            cv::Scalar(0.0000, 0.2502, 0.5005, 0.7507),
            cv::Scalar(0.0010, 0.2512, 0.5015, 0.7517),
            cv::Scalar(0.0020, 0.2522, 0.5024, 0.7527),
            cv::Scalar(0.0029, 0.2532, 0.5034, 0.7537),
            cv::Scalar(0.0039, 0.2542, 0.5044, 0.7546),
            cv::Scalar(0.0049, 0.2551, 0.5054, 0.7556),
            cv::Scalar(0.0059, 0.2561, 0.5064, 0.7566),
            cv::Scalar(0.0068, 0.2571, 0.5073, 0.7576),
            cv::Scalar(0.0078, 0.2581, 0.5083, 0.7586),
            cv::Scalar(0.0088, 0.2590, 0.5093, 0.7595),
            cv::Scalar(0.0098, 0.2600, 0.5103, 0.7605),
            cv::Scalar(0.0108, 0.2610, 0.5112, 0.7615),
            cv::Scalar(0.0117, 0.2620, 0.5122, 0.7625),
            cv::Scalar(0.0127, 0.2630, 0.5132, 0.7634),
            cv::Scalar(0.0137, 0.2639, 0.5142, 0.7644),
            cv::Scalar(0.0147, 0.2649, 0.5152, 0.7654),
            //  row 1
            cv::Scalar(0.0156, 0.2659, 0.5161, 0.7664),
            cv::Scalar(0.0166, 0.2669, 0.5171, 0.7674),
            cv::Scalar(0.0176, 0.2678, 0.5181, 0.7683),
            cv::Scalar(0.0186, 0.2688, 0.5191, 0.7693),
            cv::Scalar(0.0196, 0.2698, 0.5200, 0.7703),
            cv::Scalar(0.0205, 0.2708, 0.5210, 0.7713),
            cv::Scalar(0.0215, 0.2717, 0.5220, 0.7722),
            cv::Scalar(0.0225, 0.2727, 0.5230, 0.7732),
            cv::Scalar(0.0235, 0.2737, 0.5239, 0.7742),
            cv::Scalar(0.0244, 0.2747, 0.5249, 0.7752),
            cv::Scalar(0.0254, 0.2757, 0.5259, 0.7761),
            cv::Scalar(0.0264, 0.2766, 0.5269, 0.7771),
            cv::Scalar(0.0274, 0.2776, 0.5279, 0.7781),
            cv::Scalar(0.0283, 0.2786, 0.5288, 0.7791),
            cv::Scalar(0.0293, 0.2796, 0.5298, 0.7801),
            cv::Scalar(0.0303, 0.2805, 0.5308, 0.7810),
            //  row 2
            cv::Scalar(0.0313, 0.2815, 0.5318, 0.7820),
            cv::Scalar(0.0323, 0.2825, 0.5327, 0.7830),
            cv::Scalar(0.0332, 0.2835, 0.5337, 0.7840),
            cv::Scalar(0.0342, 0.2845, 0.5347, 0.7849),
            cv::Scalar(0.0352, 0.2854, 0.5357, 0.7859),
            cv::Scalar(0.0362, 0.2864, 0.5367, 0.7869),
            cv::Scalar(0.0371, 0.2874, 0.5376, 0.7879),
            cv::Scalar(0.0381, 0.2884, 0.5386, 0.7889),
            cv::Scalar(0.0391, 0.2893, 0.5396, 0.7898),
            cv::Scalar(0.0401, 0.2903, 0.5406, 0.7908),
            cv::Scalar(0.0411, 0.2913, 0.5415, 0.7918),
            cv::Scalar(0.0420, 0.2923, 0.5425, 0.7928),
            cv::Scalar(0.0430, 0.2933, 0.5435, 0.7937),
            cv::Scalar(0.0440, 0.2942, 0.5445, 0.7947),
            cv::Scalar(0.0450, 0.2952, 0.5455, 0.7957),
            cv::Scalar(0.0459, 0.2962, 0.5464, 0.7967),
            //  row 3
            cv::Scalar(0.0469, 0.2972, 0.5474, 0.7977),
            cv::Scalar(0.0479, 0.2981, 0.5484, 0.7986),
            cv::Scalar(0.0489, 0.2991, 0.5494, 0.7996),
            cv::Scalar(0.0499, 0.3001, 0.5503, 0.8006),
            cv::Scalar(0.0508, 0.3011, 0.5513, 0.8016),
            cv::Scalar(0.0518, 0.3021, 0.5523, 0.8025),
            cv::Scalar(0.0528, 0.3030, 0.5533, 0.8035),
            cv::Scalar(0.0538, 0.3040, 0.5543, 0.8045),
            cv::Scalar(0.0547, 0.3050, 0.5552, 0.8055),
            cv::Scalar(0.0557, 0.3060, 0.5562, 0.8065),
            cv::Scalar(0.0567, 0.3069, 0.5572, 0.8074),
            cv::Scalar(0.0577, 0.3079, 0.5582, 0.8084),
            cv::Scalar(0.0587, 0.3089, 0.5591, 0.8094),
            cv::Scalar(0.0596, 0.3099, 0.5601, 0.8104),
            cv::Scalar(0.0606, 0.3109, 0.5611, 0.8113),
            cv::Scalar(0.0616, 0.3118, 0.5621, 0.8123),
            //  row 4
            cv::Scalar(0.0626, 0.3128, 0.5630, 0.8133),
            cv::Scalar(0.0635, 0.3138, 0.5640, 0.8143),
            cv::Scalar(0.0645, 0.3148, 0.5650, 0.8152),
            cv::Scalar(0.0655, 0.3157, 0.5660, 0.8162),
            cv::Scalar(0.0665, 0.3167, 0.5670, 0.8172),
            cv::Scalar(0.0674, 0.3177, 0.5679, 0.8182),
            cv::Scalar(0.0684, 0.3187, 0.5689, 0.8192),
            cv::Scalar(0.0694, 0.3196, 0.5699, 0.8201),
            cv::Scalar(0.0704, 0.3206, 0.5709, 0.8211),
            cv::Scalar(0.0714, 0.3216, 0.5718, 0.8221),
            cv::Scalar(0.0723, 0.3226, 0.5728, 0.8231),
            cv::Scalar(0.0733, 0.3236, 0.5738, 0.8240),
            cv::Scalar(0.0743, 0.3245, 0.5748, 0.8250),
            cv::Scalar(0.0753, 0.3255, 0.5758, 0.8260),
            cv::Scalar(0.0762, 0.3265, 0.5767, 0.8270),
            cv::Scalar(0.0772, 0.3275, 0.5777, 0.8280),
            //  row 5
            cv::Scalar(0.0782, 0.3284, 0.5787, 0.8289),
            cv::Scalar(0.0792, 0.3294, 0.5797, 0.8299),
            cv::Scalar(0.0802, 0.3304, 0.5806, 0.8309),
            cv::Scalar(0.0811, 0.3314, 0.5816, 0.8319),
            cv::Scalar(0.0821, 0.3324, 0.5826, 0.8328),
            cv::Scalar(0.0831, 0.3333, 0.5836, 0.8338),
            cv::Scalar(0.0841, 0.3343, 0.5846, 0.8348),
            cv::Scalar(0.0850, 0.3353, 0.5855, 0.8358),
            cv::Scalar(0.0860, 0.3363, 0.5865, 0.8368),
            cv::Scalar(0.0870, 0.3372, 0.5875, 0.8377),
            cv::Scalar(0.0880, 0.3382, 0.5885, 0.8387),
            cv::Scalar(0.0890, 0.3392, 0.5894, 0.8397),
            cv::Scalar(0.0899, 0.3402, 0.5904, 0.8407),
            cv::Scalar(0.0909, 0.3412, 0.5914, 0.8416),
            cv::Scalar(0.0919, 0.3421, 0.5924, 0.8426),
            cv::Scalar(0.0929, 0.3431, 0.5934, 0.8436),
            //  row 6
            cv::Scalar(0.0938, 0.3441, 0.5943, 0.8446),
            cv::Scalar(0.0948, 0.3451, 0.5953, 0.8456),
            cv::Scalar(0.0958, 0.3460, 0.5963, 0.8465),
            cv::Scalar(0.0968, 0.3470, 0.5973, 0.8475),
            cv::Scalar(0.0978, 0.3480, 0.5982, 0.8485),
            cv::Scalar(0.0987, 0.3490, 0.5992, 0.8495),
            cv::Scalar(0.0997, 0.3500, 0.6002, 0.8504),
            cv::Scalar(0.1007, 0.3509, 0.6012, 0.8514),
            cv::Scalar(0.1017, 0.3519, 0.6022, 0.8524),
            cv::Scalar(0.1026, 0.3529, 0.6031, 0.8534),
            cv::Scalar(0.1036, 0.3539, 0.6041, 0.8543),
            cv::Scalar(0.1046, 0.3548, 0.6051, 0.8553),
            cv::Scalar(0.1056, 0.3558, 0.6061, 0.8563),
            cv::Scalar(0.1065, 0.3568, 0.6070, 0.8573),
            cv::Scalar(0.1075, 0.3578, 0.6080, 0.8583),
            cv::Scalar(0.1085, 0.3587, 0.6090, 0.8592),
            //  row 7
            cv::Scalar(0.1095, 0.3597, 0.6100, 0.8602),
            cv::Scalar(0.1105, 0.3607, 0.6109, 0.8612),
            cv::Scalar(0.1114, 0.3617, 0.6119, 0.8622),
            cv::Scalar(0.1124, 0.3627, 0.6129, 0.8631),
            cv::Scalar(0.1134, 0.3636, 0.6139, 0.8641),
            cv::Scalar(0.1144, 0.3646, 0.6149, 0.8651),
            cv::Scalar(0.1153, 0.3656, 0.6158, 0.8661),
            cv::Scalar(0.1163, 0.3666, 0.6168, 0.8671),
            cv::Scalar(0.1173, 0.3675, 0.6178, 0.8680),
            cv::Scalar(0.1183, 0.3685, 0.6188, 0.8690),
            cv::Scalar(0.1193, 0.3695, 0.6197, 0.8700),
            cv::Scalar(0.1202, 0.3705, 0.6207, 0.8710),
            cv::Scalar(0.1212, 0.3715, 0.6217, 0.8719),
            cv::Scalar(0.1222, 0.3724, 0.6227, 0.8729),
            cv::Scalar(0.1232, 0.3734, 0.6237, 0.8739),
            cv::Scalar(0.1241, 0.3744, 0.6246, 0.8749),
            //  row 8
            cv::Scalar(0.1251, 0.3754, 0.6256, 0.8759),
            cv::Scalar(0.1261, 0.3763, 0.6266, 0.8768),
            cv::Scalar(0.1271, 0.3773, 0.6276, 0.8778),
            cv::Scalar(0.1281, 0.3783, 0.6285, 0.8788),
            cv::Scalar(0.1290, 0.3793, 0.6295, 0.8798),
            cv::Scalar(0.1300, 0.3803, 0.6305, 0.8807),
            cv::Scalar(0.1310, 0.3812, 0.6315, 0.8817),
            cv::Scalar(0.1320, 0.3822, 0.6325, 0.8827),
            cv::Scalar(0.1329, 0.3832, 0.6334, 0.8837),
            cv::Scalar(0.1339, 0.3842, 0.6344, 0.8847),
            cv::Scalar(0.1349, 0.3851, 0.6354, 0.8856),
            cv::Scalar(0.1359, 0.3861, 0.6364, 0.8866),
            cv::Scalar(0.1369, 0.3871, 0.6373, 0.8876),
            cv::Scalar(0.1378, 0.3881, 0.6383, 0.8886),
            cv::Scalar(0.1388, 0.3891, 0.6393, 0.8895),
            cv::Scalar(0.1398, 0.3900, 0.6403, 0.8905),
            //  row 9
            cv::Scalar(0.1408, 0.3910, 0.6413, 0.8915),
            cv::Scalar(0.1417, 0.3920, 0.6422, 0.8925),
            cv::Scalar(0.1427, 0.3930, 0.6432, 0.8935),
            cv::Scalar(0.1437, 0.3939, 0.6442, 0.8944),
            cv::Scalar(0.1447, 0.3949, 0.6452, 0.8954),
            cv::Scalar(0.1457, 0.3959, 0.6461, 0.8964),
            cv::Scalar(0.1466, 0.3969, 0.6471, 0.8974),
            cv::Scalar(0.1476, 0.3978, 0.6481, 0.8983),
            cv::Scalar(0.1486, 0.3988, 0.6491, 0.8993),
            cv::Scalar(0.1496, 0.3998, 0.6500, 0.9003),
            cv::Scalar(0.1505, 0.4008, 0.6510, 0.9013),
            cv::Scalar(0.1515, 0.4018, 0.6520, 0.9022),
            cv::Scalar(0.1525, 0.4027, 0.6530, 0.9032),
            cv::Scalar(0.1535, 0.4037, 0.6540, 0.9042),
            cv::Scalar(0.1544, 0.4047, 0.6549, 0.9052),
            cv::Scalar(0.1554, 0.4057, 0.6559, 0.9062),
            //  row 10
            cv::Scalar(0.1564, 0.4066, 0.6569, 0.9071),
            cv::Scalar(0.1574, 0.4076, 0.6579, 0.9081),
            cv::Scalar(0.1584, 0.4086, 0.6588, 0.9091),
            cv::Scalar(0.1593, 0.4096, 0.6598, 0.9101),
            cv::Scalar(0.1603, 0.4106, 0.6608, 0.9110),
            cv::Scalar(0.1613, 0.4115, 0.6618, 0.9120),
            cv::Scalar(0.1623, 0.4125, 0.6628, 0.9130),
            cv::Scalar(0.1632, 0.4135, 0.6637, 0.9140),
            cv::Scalar(0.1642, 0.4145, 0.6647, 0.9150),
            cv::Scalar(0.1652, 0.4154, 0.6657, 0.9159),
            cv::Scalar(0.1662, 0.4164, 0.6667, 0.9169),
            cv::Scalar(0.1672, 0.4174, 0.6676, 0.9179),
            cv::Scalar(0.1681, 0.4184, 0.6686, 0.9189),
            cv::Scalar(0.1691, 0.4194, 0.6696, 0.9198),
            cv::Scalar(0.1701, 0.4203, 0.6706, 0.9208),
            cv::Scalar(0.1711, 0.4213, 0.6716, 0.9218),
            //  row 11
            cv::Scalar(0.1720, 0.4223, 0.6725, 0.9228),
            cv::Scalar(0.1730, 0.4233, 0.6735, 0.9238),
            cv::Scalar(0.1740, 0.4242, 0.6745, 0.9247),
            cv::Scalar(0.1750, 0.4252, 0.6755, 0.9257),
            cv::Scalar(0.1760, 0.4262, 0.6764, 0.9267),
            cv::Scalar(0.1769, 0.4272, 0.6774, 0.9277),
            cv::Scalar(0.1779, 0.4282, 0.6784, 0.9286),
            cv::Scalar(0.1789, 0.4291, 0.6794, 0.9296),
            cv::Scalar(0.1799, 0.4301, 0.6804, 0.9306),
            cv::Scalar(0.1808, 0.4311, 0.6813, 0.9316),
            cv::Scalar(0.1818, 0.4321, 0.6823, 0.9326),
            cv::Scalar(0.1828, 0.4330, 0.6833, 0.9335),
            cv::Scalar(0.1838, 0.4340, 0.6843, 0.9345),
            cv::Scalar(0.1848, 0.4350, 0.6852, 0.9355),
            cv::Scalar(0.1857, 0.4360, 0.6862, 0.9365),
            cv::Scalar(0.1867, 0.4370, 0.6872, 0.9374),
            //  row 12
            cv::Scalar(0.1877, 0.4379, 0.6882, 0.9384),
            cv::Scalar(0.1887, 0.4389, 0.6891, 0.9394),
            cv::Scalar(0.1896, 0.4399, 0.6901, 0.9404),
            cv::Scalar(0.1906, 0.4409, 0.6911, 0.9413),
            cv::Scalar(0.1916, 0.4418, 0.6921, 0.9423),
            cv::Scalar(0.1926, 0.4428, 0.6931, 0.9433),
            cv::Scalar(0.1935, 0.4438, 0.6940, 0.9443),
            cv::Scalar(0.1945, 0.4448, 0.6950, 0.9453),
            cv::Scalar(0.1955, 0.4457, 0.6960, 0.9462),
            cv::Scalar(0.1965, 0.4467, 0.6970, 0.9472),
            cv::Scalar(0.1975, 0.4477, 0.6979, 0.9482),
            cv::Scalar(0.1984, 0.4487, 0.6989, 0.9492),
            cv::Scalar(0.1994, 0.4497, 0.6999, 0.9501),
            cv::Scalar(0.2004, 0.4506, 0.7009, 0.9511),
            cv::Scalar(0.2014, 0.4516, 0.7019, 0.9521),
            cv::Scalar(0.2023, 0.4526, 0.7028, 0.9531),
            //  row 13
            cv::Scalar(0.2033, 0.4536, 0.7038, 0.9541),
            cv::Scalar(0.2043, 0.4545, 0.7048, 0.9550),
            cv::Scalar(0.2053, 0.4555, 0.7058, 0.9560),
            cv::Scalar(0.2063, 0.4565, 0.7067, 0.9570),
            cv::Scalar(0.2072, 0.4575, 0.7077, 0.9580),
            cv::Scalar(0.2082, 0.4585, 0.7087, 0.9589),
            cv::Scalar(0.2092, 0.4594, 0.7097, 0.9599),
            cv::Scalar(0.2102, 0.4604, 0.7107, 0.9609),
            cv::Scalar(0.2111, 0.4614, 0.7116, 0.9619),
            cv::Scalar(0.2121, 0.4624, 0.7126, 0.9629),
            cv::Scalar(0.2131, 0.4633, 0.7136, 0.9638),
            cv::Scalar(0.2141, 0.4643, 0.7146, 0.9648),
            cv::Scalar(0.2151, 0.4653, 0.7155, 0.9658),
            cv::Scalar(0.2160, 0.4663, 0.7165, 0.9668),
            cv::Scalar(0.2170, 0.4673, 0.7175, 0.9677),
            cv::Scalar(0.2180, 0.4682, 0.7185, 0.9687),
            //  row 14
            cv::Scalar(0.2190, 0.4692, 0.7195, 0.9697),
            cv::Scalar(0.2199, 0.4702, 0.7204, 0.9707),
            cv::Scalar(0.2209, 0.4712, 0.7214, 0.9717),
            cv::Scalar(0.2219, 0.4721, 0.7224, 0.9726),
            cv::Scalar(0.2229, 0.4731, 0.7234, 0.9736),
            cv::Scalar(0.2239, 0.4741, 0.7243, 0.9746),
            cv::Scalar(0.2248, 0.4751, 0.7253, 0.9756),
            cv::Scalar(0.2258, 0.4761, 0.7263, 0.9765),
            cv::Scalar(0.2268, 0.4770, 0.7273, 0.9775),
            cv::Scalar(0.2278, 0.4780, 0.7283, 0.9785),
            cv::Scalar(0.2287, 0.4790, 0.7292, 0.9795),
            cv::Scalar(0.2297, 0.4800, 0.7302, 0.9804),
            cv::Scalar(0.2307, 0.4809, 0.7312, 0.9814),
            cv::Scalar(0.2317, 0.4819, 0.7322, 0.9824),
            cv::Scalar(0.2326, 0.4829, 0.7331, 0.9834),
            cv::Scalar(0.2336, 0.4839, 0.7341, 0.9844),
            //  row 15
            cv::Scalar(0.2346, 0.4848, 0.7351, 0.9853),
            cv::Scalar(0.2356, 0.4858, 0.7361, 0.9863),
            cv::Scalar(0.2366, 0.4868, 0.7370, 0.9873),
            cv::Scalar(0.2375, 0.4878, 0.7380, 0.9883),
            cv::Scalar(0.2385, 0.4888, 0.7390, 0.9892),
            cv::Scalar(0.2395, 0.4897, 0.7400, 0.9902),
            cv::Scalar(0.2405, 0.4907, 0.7410, 0.9912),
            cv::Scalar(0.2414, 0.4917, 0.7419, 0.9922),
            cv::Scalar(0.2424, 0.4927, 0.7429, 0.9932),
            cv::Scalar(0.2434, 0.4936, 0.7439, 0.9941),
            cv::Scalar(0.2444, 0.4946, 0.7449, 0.9951),
            cv::Scalar(0.2454, 0.4956, 0.7458, 0.9961),
            cv::Scalar(0.2463, 0.4966, 0.7468, 0.9971),
            cv::Scalar(0.2473, 0.4976, 0.7478, 0.9980),
            cv::Scalar(0.2483, 0.4985, 0.7488, 0.9990),
            cv::Scalar(0.2493, 0.4995, 0.7498, 1.0000)
        );

        cv::Scalar linear_space_expected_stdev = cv::Scalar(
            0.0463703703703704,
            0.0462962962962963,
            0.0463703703703704,
            0.0462962962962963
        );



        std::vector<ParamType> params = {
            //  Subbands Partition, Linear space from zero to one
            {
                .matrix = linear_space_matrix,
                .partition = BayesShrink::SUBBANDS,
                .expected_stdev = linear_space_expected_stdev,
                .expected_thresholds = (cv::Mat4d(3, 3) <<
                    //  level 0
                    cv::Scalar(0.0118847554733989, 0.0049552342666022, 0.0031475248665646, 0.0022958861605492),
                    cv::Scalar(0.0363832595314128, 0.0068070185132008, 0.0038008972709787, 0.0026260362524452),
                    cv::Scalar(0.0113864955897503, 0.0048670261362589, 0.0031118684248553, 0.0022768044248884),
                    //  level 1
                    cv::Scalar(0.0282074009641329, 0.0063980190481687, 0.0036668040852824, 0.0025603739111730),
                    cv::Scalar(0.0538000000000000, 0.0077721786523217, 0.0040755548546290, 0.0027535641011884),
                    cv::Scalar(0.0266503396172219, 0.0063235713253610, 0.0036423988004784, 0.0025484475440286),
                    // level 2
                    cv::Scalar(0.0479000000000000, 0.0074886999717402, 0.0039956520407572, 0.0027165128064440),
                    cv::Scalar(0.0186000000000000, 0.0083570343783328, 0.0042268623475755, 0.0028215646858654),
                    cv::Scalar(0.0499000000000000, 0.0074367555727893, 0.0039811740837613, 0.0027098044672065)
                ),
            },
            //  Levels Partition, Linear space from zero to one
            {
                .matrix = linear_space_matrix,
                .partition = BayesShrink::LEVELS,
                .expected_stdev = linear_space_expected_stdev,
                .expected_thresholds = (cv::Mat4d(3, 1) <<
                    //  level 0
                    cv::Scalar(0.0138906092115623, 0.0053574004927013, 0.0033123900965170, 0.0023844888917240),
                    //  level 1
                    cv::Scalar(0.0350214293302716, 0.0067424378016843, 0.0037800646048746, 0.0026159042987668),
                    // level 2
                    cv::Scalar(0.0499000000000000, 0.0077281086804240, 0.0040633435993217, 0.0027478896839230)
                ),
            },
            //  Global Partition, Linear space from zero to one
            {
                .matrix = linear_space_matrix,
                .partition = BayesShrink::GLOBALLY,
                .expected_stdev = linear_space_expected_stdev,
                .expected_thresholds = (cv::Mat4d(5, 1) <<
                    //  All Levels
                    cv::Scalar(0.0156284404900803, 0.0056314262484730, 0.0034168166405375, 0.0024387095375987),
                    //  First Level
                    cv::Scalar(0.0138906092115623, 0.0053574004927013, 0.0033123900965170, 0.0023844888917240),
                    //  First Two Levels
                    cv::Scalar(0.0152335015158094, 0.0055665361223893, 0.0033920678746390, 0.0024258639545910),
                    //  Last Level
                    cv::Scalar(0.0499000000000000, 0.0077281086804240, 0.0040633435993217, 0.0027478896839230),
                    //  Last Two Levels
                    cv::Scalar(0.0404578141327428, 0.0069094685394569, 0.0038319837909622, 0.0026407803810164)
                ),
            },
        };

        return params;
    }
};

TEST_P(BayesThresholdTest, CorrectStandardDeviation)
{
    auto param = GetParam();
    auto actual_stdev = shrinker.compute_stdev(coeffs);

    EXPECT_THAT(actual_stdev, ScalarNear(param.expected_stdev, 1e-12));
}

TEST_P(BayesThresholdTest, CorrectThresholds)
{
    auto param = GetParam();
    auto expected_thresholds = param.partition == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(0)
                             : param.expected_thresholds;

    auto actual_thresholds = shrinker.compute_thresholds(coeffs);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
}

TEST_P(BayesThresholdTest, ThresholdsConsistentWithPrescalingByStdDev)
{
    auto param = GetParam();
    cv::divide(coeffs, param.expected_stdev, coeffs);
    auto expected_thresholds = param.partition == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(0)
                             : param.expected_thresholds;
    cv::divide(expected_thresholds, param.expected_stdev, expected_thresholds);

    auto actual_thresholds = shrinker.compute_thresholds(coeffs, cv::Scalar::all(1.0));

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
}

TEST_P(BayesThresholdTest, AllLevels)
{
    auto param = GetParam();
    cv::Range levels = cv::Range::all();
    auto expected_thresholds = param.partition == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(0)
                             : param.expected_thresholds;

    auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
}

TEST_P(BayesThresholdTest, FirstLevel)
{
    auto param = GetParam();
    int levels = 1;
    auto expected_thresholds = param.partition == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(1)
                             : param.expected_thresholds.rowRange(0, levels);

    auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
}

TEST_P(BayesThresholdTest, FirstTwoLevels)
{
    auto param = GetParam();
    int levels = 2;
    auto expected_thresholds = param.partition == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(2)
                             : param.expected_thresholds.rowRange(0, levels);

    auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
}

TEST_P(BayesThresholdTest, LastLevel)
{
    auto param = GetParam();
    cv::Range levels(LEVELS - 1, LEVELS);
    auto expected_thresholds = param.partition == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(3)
                             : param.expected_thresholds.rowRange(levels);

    auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
}

TEST_P(BayesThresholdTest, LastTwoLevels)
{
    auto param = GetParam();
    cv::Range levels(LEVELS - 2, LEVELS);
    auto expected_thresholds = param.partition == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(4)
                             : param.expected_thresholds.rowRange(levels);

    auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
}


INSTANTIATE_TEST_CASE_P(
    BayesShrinkGroup,
    BayesThresholdTest,
    testing::ValuesIn(BayesThresholdTest::create_params()),
    [](const auto& info) { return BayesThresholdTest::param_name(info.index); }
);



// class BayesLevelwiseThresholdTest : public BaseBayesThresholdTest
// {
// protected:
//     BayesLevelwiseThresholdTest() :
//         BaseBayesThresholdTest()
//     {
//     }

// public:
//     static std::vector<ParamType> create_params()
//     {
//         std::vector<ParamType> params = BaseBayesThresholdTest::create_params();
//         params[0].expected_thresholds = (cv::Mat4d(3, 3) <<
//             //  level 0
//             cv::Scalar(0.0118847554733989, 0.0049552342666022, 0.0031475248665646, 0.0022958861605492),
//             cv::Scalar(0.0363832595314128, 0.0068070185132008, 0.0038008972709787, 0.0026260362524452),
//             cv::Scalar(0.0113864955897503, 0.0048670261362589, 0.0031118684248553, 0.0022768044248884),
//             //  level 1
//             cv::Scalar(0.0282074009641329, 0.0063980190481687, 0.0036668040852824, 0.0025603739111730),
//             cv::Scalar(0.0538000000000000, 0.0077721786523217, 0.0040755548546290, 0.0027535641011884),
//             cv::Scalar(0.0266503396172219, 0.0063235713253610, 0.0036423988004784, 0.0025484475440286),
//             // level 2
//             cv::Scalar(0.0479000000000000, 0.0074886999717402, 0.0039956520407572, 0.0027165128064440),
//             cv::Scalar(0.0186000000000000, 0.0083570343783328, 0.0042268623475755, 0.0028215646858654),
//             cv::Scalar(0.0499000000000000, 0.0074367555727893, 0.0039811740837613, 0.0027098044672065)
//         );
//         params[1].expected_thresholds = (cv::Mat4d(3, 3) <<
//             //  level 0
//             cv::Scalar(0.7598366255744056, 2.7464921615122035, 2.5918999999999999, 1.5957953440524637),
//             cv::Scalar(1.4484663323243601, 1.9811133866740702, 2.8978999999999999, 1.8326563812374326),
//             cv::Scalar(1.3711234675595814, 2.4820000000000002, 2.5905000000000000, 1.7710688736143609),
//             //  level 1
//             cv::Scalar(0.7102561558068973, 1.2974000000000001, 1.8768000000000000, 2.2578000000000000),
//             cv::Scalar(0.8811380309437045, 2.9189296662583955, 1.9648000000000001, 1.3546710943053131),
//             cv::Scalar(1.1663556381945410, 1.6355999999999999, 1.8886000000000001, 1.5276268497358454),
//             // level 2
//             cv::Scalar(0.5221923166217919, 1.2262000000000000, 1.1895291132468244, 0.9825505960055069),
//             cv::Scalar(0.6234351165213589, 1.2084999999999999, 1.1565000000000001, 1.7526999999999999),
//             cv::Scalar(0.5708646746248460, 2.3906193880144553, 1.2541000000000000, 0.6429998985959891)
//         );

//         return params;
//     }
// };

// TEST_P(BayesLevelwiseThresholdTest, CorrectStandardDeviation)
// {
//     auto param = GetParam();

//     auto actual_stdev = shrinker.compute_stdev(coeffs);

//     EXPECT_THAT(actual_stdev, ScalarNear(param.expected_stdev, 1e-12));
// }

// TEST_P(BayesLevelwiseThresholdTest, CorrectThresholds)
// {
//     auto param = GetParam();

//     auto actual_thresholds = shrinker.compute_thresholds(coeffs);

//     EXPECT_THAT(actual_thresholds, MatrixNear(param.expected_thresholds, NEARNESS_TOLERANCE));
// }

// TEST_P(BayesLevelwiseThresholdTest, ThresholdsConsistentWithPrescalingByStdDev)
// {
//     auto param = GetParam();
//     cv::divide(coeffs, param.expected_stdev, coeffs);
//     cv::Mat expected_thresholds;
//     cv::divide(param.expected_thresholds, param.expected_stdev, expected_thresholds);

//     auto actual_thresholds = shrinker.compute_thresholds(coeffs, cv::Scalar::all(1.0));

//     EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
// }

// TEST_P(BayesLevelwiseThresholdTest, FirstLevel)
// {
//     auto param = GetParam();
//     int levels = 1;
//     auto expected_thresholds = param.expected_thresholds.rowRange(0, levels);
//     auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

//     EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
// }

// TEST_P(BayesLevelwiseThresholdTest, FirstTwoLevels)
// {
//     auto param = GetParam();
//     int levels = 2;
//     auto expected_thresholds = param.expected_thresholds.rowRange(0, levels);
//     auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

//     EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
// }

// TEST_P(BayesLevelwiseThresholdTest, LastLevel)
// {
//     auto param = GetParam();
//     cv::Range levels(2, 3);
//     auto expected_thresholds = param.expected_thresholds.rowRange(levels);
//     auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

//     EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
// }

// TEST_P(BayesLevelwiseThresholdTest, LastTwoLevels)
// {
//     auto param = GetParam();
//     cv::Range levels(1, 3);
//     auto expected_thresholds = param.expected_thresholds.rowRange(levels);
//     auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

//     EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, NEARNESS_TOLERANCE));
// }

// TEST_P(BayesLevelwiseThresholdTest, AllLevels)
// {
//     auto param = GetParam();
//     cv::Range levels = cv::Range::all();
//     auto actual_thresholds = shrinker.compute_thresholds(coeffs, levels);

//     EXPECT_THAT(actual_thresholds, MatrixNear(param.expected_thresholds, NEARNESS_TOLERANCE));
// }


// INSTANTIATE_TEST_CASE_P(
//     BayesShrinkGroup,
//     BayesThresholdTest,
//     testing::ValuesIn(BayesThresholdTest::create_params()),
//     [](const auto& info) { return BayesThresholdTest::param_name(info.index); }
// );