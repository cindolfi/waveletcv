/**
 * Shrinkage Unit Test
*/
#include <iterator>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlopt.hpp>
#include <cvwt/shrinkage.hpp>
#include "common.hpp"
#include "json.hpp"

using namespace cvwt;
using namespace testing;

const std::vector<int> PERMUTATION1 = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
const std::vector<int> PERMUTATION2 = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};

//  ============================================================================
//  Threshold
//  ============================================================================
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
struct ShrinkGloballyTestParam
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
void PrintTo(const ShrinkGloballyTestParam<T, CHANNELS>& param, std::ostream* stream)
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
//  Soft Shrink Globally
//  ----------------------------------------------------------------------------
class SoftShrinkGloballyTest : public ShrinkTestBase<ShrinkGloballyTestParam<double, 4>>
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
                .upper_level = 4,
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
                .upper_level = 1,
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
                .upper_level = 2,
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
                .upper_level = 3,
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
                .upper_level = 4,
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
                .upper_level = 2,
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
                .upper_level = 3,
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
                .upper_level = 4,
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
                .upper_level = -1,
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
                .upper_level = -2,
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
                .upper_level = -3,
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
                .upper_level = -1,
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

TEST_P(SoftShrinkGloballyTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs.clone();

    shrink_globally(
        coeffs,
        param.threshold,
        soft_threshold,
        cv::Range(param.lower_level, param.upper_level)
    );

    EXPECT_THAT(coeffs, MatrixFloatEq(param.expected));
}

INSTANTIATE_TEST_CASE_P(
    ShrinkDetailsGroup,
    SoftShrinkGloballyTest,
    testing::ValuesIn(SoftShrinkGloballyTest::create_params())
);


//  ----------------------------------------------------------------------------
//  Hard Shrink Globally
//  ----------------------------------------------------------------------------
class HardShrinkGloballyTest : public ShrinkTestBase<ShrinkGloballyTestParam<double, 4>>
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
                .upper_level = 4,
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
                .upper_level = 1,
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
                .upper_level = 2,
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
                .upper_level = 3,
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
                .upper_level = 4,
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
                .upper_level = 2,
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
                .upper_level = 3,
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
                .upper_level = 4,
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
                .upper_level = -1,
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
                .upper_level = -2,
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
                .upper_level = -3,
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
                .upper_level = -1,
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

TEST_P(HardShrinkGloballyTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs.clone();

    shrink_globally(
        coeffs,
        param.threshold,
        hard_threshold,
        cv::Range(param.lower_level, param.upper_level)
    );

    EXPECT_THAT(coeffs, MatrixFloatEq(param.expected));
}

INSTANTIATE_TEST_CASE_P(
    ShrinkDetailsGroup,
    HardShrinkGloballyTest,
    testing::ValuesIn(HardShrinkGloballyTest::create_params())
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
    cv::Mat4d thresholds;
    // std::vector<cv::Scalar> thresholds;
    DWT2D::Coeffs expected;
};

template<typename T, int CHANNELS>
void PrintTo(const ShrinkLevelsTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "coeffs =";
    PrintTo(param.coeffs, stream);
    *stream << "thresholds = " << param.thresholds << "\n";
}

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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0)
                ),
                .expected = coeffs.clone(),
            },
            //  1
            {
                .coeffs = coeffs.clone(),
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(4),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(4)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(4),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(3),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(4)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(4),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(4)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar(6, 6, 5, 5),
                    cv::Scalar(4, 2, 1, 0),
                    cv::Scalar(2, 2, 2, 2),
                    cv::Scalar(0, 0, 0, 0)
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

TEST_P(SoftShrinkLevelsTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs.clone();

    shrink_levels(coeffs, param.thresholds, soft_threshold);

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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0)
                ),
                .expected = coeffs.clone(),
            },
            //  1
            {
                .coeffs = coeffs.clone(),
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(4),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(4)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(4),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(3),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(4)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(4),
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar::all(2),
                    cv::Scalar::all(1),
                    cv::Scalar::all(2),
                    cv::Scalar::all(4)
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
                .thresholds = (cv::Mat4d(4, 1) <<
                    cv::Scalar(6, 6, 5, 5),
                    cv::Scalar(4, 2, 1, 0),
                    cv::Scalar(2, 2, 2, 2),
                    cv::Scalar(0, 0, 0, 0)
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

TEST_P(HardShrinkLevelsTest, DetailsShrunkCorrectly)
{
    auto param = GetParam();
    auto coeffs = param.coeffs;

    shrink_levels(coeffs, param.thresholds, hard_threshold);

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

    shrink_subbands(coeffs, param.thresholds, soft_threshold);

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

    shrink_subbands(coeffs, param.thresholds, hard_threshold);

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

class SureRiskTest : public testing::TestWithParam<SureRiskTestParam<double, 4>>
{
public:
    static constexpr double tolerance = 1e-12;
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;
    using BaseParamsMap = std::map< std::string, ParamType>;

    cv::Scalar compute_sure_risk(
        const cv::Mat& matrix,
        const cv::Scalar& threshold,
        const cv::Scalar& stdev
    )
    {
        return SureShrink().compute_sure_risk(matrix, threshold, stdev);
    }

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


//  ============================================================================
//  Shrink Threshold Test
//  ============================================================================
struct ShrinkThresholdTestParam
{
    std::string label;
    std::shared_ptr<Shrink> shrinker;
    DWT2D::Coeffs coeffs;
    double tolerance;
    cv::Scalar expected_stdev;
    cv::Mat4d expected_thresholds;
};

void PrintTo(const ShrinkThresholdTestParam& param, std::ostream* stream)
{
    *stream << "\n";
    *stream << "label =" << param.label << "\n";
    *stream << "coeffs =";
    PrintTo(param.coeffs, stream);
    *stream << "tolerance = " << param.tolerance << "\n";
    *stream << "expected_stdev = " << param.expected_stdev << "\n";
    *stream << "expected_thresholds = " << param.expected_thresholds << "\n";
}

class ShrinkThresholdTest : public testing::TestWithParam<ShrinkThresholdTestParam>
{
    const unsigned long NLOPT_SEED = 42;

protected:
    void SetUp() override
    {
        nlopt::srand(NLOPT_SEED);
    }

public:
    static constexpr double TOLERANCE = 1e-15;
    static constexpr double SURE_SHRINK_DEFAULT_OPTIMIZER_TOLERANCE = 0.5 / 256.0;

    static std::vector<ParamType> create_params()
    {
        //  SHRINKAGE_THRESHOLD_TEST_DATA_PATH is defined in CMakeLists.txt
        std::ifstream test_case_data_file(SHRINKAGE_THRESHOLD_TEST_DATA_PATH);
        auto test_cases = json::parse(test_case_data_file);

        auto coeffs_map = test_cases["coeffs"].get<std::map<std::string, DWT2D::Coeffs>>();

        std::vector<ParamType> params;
        for (auto& json_param : test_cases["params"]) {
            auto coeffs_name = json_param["coeffs"].get<std::string>();
            auto label = json_param["shrinker"]["label"].get<std::string>() + "__" + coeffs_name;
            params.emplace_back(
                label,
                create_shrinker(json_param["shrinker"]),
                coeffs_map.at(coeffs_name),
                get_threshold_tolerance(json_param["shrinker"]),
                json_param["expected_stdev"].get<cv::Scalar>(),
                json_param["expected_thresholds"].get<cv::Mat4d>()
            );
        }

        return params;
    }

private:
    static std::shared_ptr<Shrink> create_shrinker(const json& json_shrinker)
    {
        auto type = json_shrinker["type"].get<std::string>();
        auto args = json_shrinker["args"];
        if (type == "SureShrink")
            return create_sure_shrink(args);
        else if (type == "BayesShrink")
            return create_bayes_shrink(args);
        else if (type == "VisuShrink")
            return create_visu_shrink(args);

        throw std::invalid_argument("Invalid shrinker type");
    }

    static std::shared_ptr<Shrink> create_bayes_shrink(const json& args)
    {
        return std::make_shared<BayesShrink>(
            partition_map.at(args["partition"])
        );
    }

    static std::shared_ptr<Shrink> create_sure_shrink(const json& args)
    {
        auto shrinker = std::make_shared<SureShrink>(
            partition_map.at(args["partition"].get<std::string>()),
            sure_variant_map.at(args["variant"].get<std::string>()),
            sure_optimizer_map.at(args["optimizer"].get<std::string>())
        );
        shrinker->fail_on_timeout();

        return shrinker;
    }

    static std::shared_ptr<Shrink> create_visu_shrink(const json& args)
    {
        return std::make_shared<VisuShrink>(
            partition_map.at(args["partition"].get<std::string>())
        );
    }

    static double get_threshold_tolerance(const json& json_shrinker)
    {
        double tolerance = TOLERANCE;
        auto type = json_shrinker["type"].get<std::string>();
        if (type == "SureShrink") {
            tolerance = sure_optimizer_tolerance_map.at(
                json_shrinker["args"]["optimizer"].get<std::string>()
            );
        }

        return tolerance;
    }

private:
    static std::map<std::string, Shrink::Partition> partition_map;
    static std::map<std::string, SureShrink::Variant> sure_variant_map;
    static std::map<std::string, SureShrink::Optimizer> sure_optimizer_map;
    static std::map<std::string, double> sure_optimizer_tolerance_map;
};

std::map<std::string, Shrink::Partition> ShrinkThresholdTest::partition_map(
    {
        {"global", Shrink::GLOBALLY},
        {"levels", Shrink::LEVELS},
        {"subbands", Shrink::SUBBANDS},
    }
);
std::map<std::string, SureShrink::Variant> ShrinkThresholdTest::sure_variant_map(
    {
        {"strict", SureShrink::STRICT},
        {"hybrid", SureShrink::HYBRID},
    }
);
std::map<std::string, SureShrink::Optimizer> ShrinkThresholdTest::sure_optimizer_map(
    {
        {"auto", SureShrink::AUTO},
        {"nelder_mead", SureShrink::NELDER_MEAD},
        {"sbplx", SureShrink::SBPLX},
        {"cobyla", SureShrink::COBYLA},
        {"bobyqa", SureShrink::BOBYQA},
        {"direct", SureShrink::DIRECT},
        {"direct_l", SureShrink::DIRECT_L},
        {"brute_force", SureShrink::BRUTE_FORCE},
    }
);

std::map<std::string, double> ShrinkThresholdTest::sure_optimizer_tolerance_map(
    {
        {"auto", 3 * ShrinkThresholdTest::SURE_SHRINK_DEFAULT_OPTIMIZER_TOLERANCE},
        {"nelder_mead", 8 * ShrinkThresholdTest::SURE_SHRINK_DEFAULT_OPTIMIZER_TOLERANCE},
        {"sbplx", 3 * ShrinkThresholdTest::SURE_SHRINK_DEFAULT_OPTIMIZER_TOLERANCE},
        {"cobyla", 3 * ShrinkThresholdTest::SURE_SHRINK_DEFAULT_OPTIMIZER_TOLERANCE},
        {"bobyqa", 3 * ShrinkThresholdTest::SURE_SHRINK_DEFAULT_OPTIMIZER_TOLERANCE},
        {"direct", ShrinkThresholdTest::SURE_SHRINK_DEFAULT_OPTIMIZER_TOLERANCE},
        {"direct_l", ShrinkThresholdTest::SURE_SHRINK_DEFAULT_OPTIMIZER_TOLERANCE},
        {"brute_force", ShrinkThresholdTest::TOLERANCE},
    }
);

TEST_P(ShrinkThresholdTest, CorrectNoiseStandardDeviation)
{
    auto param = GetParam();
    auto actual_stdev = param.shrinker->compute_noise_stdev(param.coeffs);

    EXPECT_THAT(actual_stdev, ScalarNear(param.expected_stdev, 1e-12));
}

TEST_P(ShrinkThresholdTest, CorrectThresholds)
{
    auto param = GetParam();
    auto expected_thresholds = param.shrinker->partition() == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(0)
                             : param.expected_thresholds;

    auto actual_thresholds = param.shrinker->compute_thresholds(param.coeffs);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, param.tolerance));
}

TEST_P(ShrinkThresholdTest, ConsistentWithPrescalingByStdDev)
{
    auto param = GetParam();
    auto sure_shrinker = std::dynamic_pointer_cast<SureShrink>(param.shrinker);
    if (sure_shrinker && sure_shrinker->optimizer() != SureShrink::BRUTE_FORCE)
        GTEST_SKIP();

    auto coeffs = param.coeffs.empty_clone();
    cv::divide(param.coeffs, param.expected_stdev, coeffs);
    auto expected_thresholds = param.shrinker->partition() == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(0)
                             : param.expected_thresholds;
    cv::divide(expected_thresholds, param.expected_stdev, expected_thresholds);
    patch_nans(expected_thresholds);

    auto actual_thresholds = param.shrinker->compute_thresholds(coeffs, cv::Scalar::all(1.0));

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, param.tolerance));
}

TEST_P(ShrinkThresholdTest, AllLevels)
{
    auto param = GetParam();
    cv::Range levels = cv::Range::all();
    auto expected_thresholds = param.shrinker->partition() == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(0)
                             : param.expected_thresholds;

    auto actual_thresholds = param.shrinker->compute_thresholds(param.coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, param.tolerance));
}

TEST_P(ShrinkThresholdTest, FirstLevel)
{
    auto param = GetParam();
    int levels = 1;
    auto expected_thresholds = param.shrinker->partition() == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(1)
                             : param.expected_thresholds.rowRange(0, levels);

    auto actual_thresholds = param.shrinker->compute_thresholds(param.coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, param.tolerance));
}

TEST_P(ShrinkThresholdTest, LastLevel)
{
    auto param = GetParam();
    cv::Range levels(param.coeffs.levels() - 1, param.coeffs.levels());
    auto expected_thresholds = param.shrinker->partition() == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(2)
                             : param.expected_thresholds.rowRange(levels);

    auto actual_thresholds = param.shrinker->compute_thresholds(param.coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, param.tolerance));
}

TEST_P(ShrinkThresholdTest, FirstTwoLevels)
{
    auto param = GetParam();
    if (param.coeffs.levels() < 2)
        GTEST_SKIP();

    int levels = 2;
    auto expected_thresholds = param.shrinker->partition() == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(3)
                             : param.expected_thresholds.rowRange(0, levels);

    auto actual_thresholds = param.shrinker->compute_thresholds(param.coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, param.tolerance));
}

TEST_P(ShrinkThresholdTest, LastTwoLevels)
{
    auto param = GetParam();
    if (param.coeffs.levels() < 3)
        GTEST_SKIP();

    cv::Range levels(param.coeffs.levels() - 2, param.coeffs.levels());
    auto expected_thresholds = param.shrinker->partition() == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(4)
                             : param.expected_thresholds.rowRange(levels);

    auto actual_thresholds = param.shrinker->compute_thresholds(param.coeffs, levels);

    EXPECT_THAT(actual_thresholds, MatrixNear(expected_thresholds, param.tolerance));
}

TEST_P(ShrinkThresholdTest, ExpandThresholds)
{
    auto param = GetParam();
    auto expected_thresholds = param.shrinker->partition() == Shrink::GLOBALLY
                             ? param.expected_thresholds.row(0)
                             : param.expected_thresholds;

    auto expanded_thresholds = param.shrinker->expand_thresholds(
        param.coeffs,
        expected_thresholds,
        param.coeffs.levels()
    );

    cv::Mat collected;
    switch (param.shrinker->partition()) {
    case Shrink::GLOBALLY:
        collect_masked(expanded_thresholds, collected, param.coeffs.detail_mask());
        EXPECT_THAT(
            collected,
            MatrixAllEq(expected_thresholds.at<cv::Scalar>(0))
        );
        break;
    case Shrink::LEVELS:
        for (int level = 0; level < param.coeffs.levels(); ++level) {
            collect_masked(expanded_thresholds, collected, param.coeffs.detail_mask(level));
            EXPECT_THAT(
                collected,
                MatrixAllEq(expected_thresholds.at<cv::Scalar>(level))
            );
        }
        break;
    case Shrink::SUBBANDS:
        for (int level = 0; level < param.coeffs.levels(); ++level) {
            for (auto subband : {HORIZONTAL, VERTICAL, DIAGONAL}) {
                collect_masked(
                    expanded_thresholds,
                    collected,
                    param.coeffs.detail_mask(level, subband)
                );
                EXPECT_THAT(
                    collected,
                    MatrixAllEq(expected_thresholds.at<cv::Scalar>(level, subband))
                );
            }
        }
        break;
    }
}

INSTANTIATE_TEST_CASE_P(
    ShrinkGroup,
    ShrinkThresholdTest,
    testing::ValuesIn(ShrinkThresholdTest::create_params()),
    [](const auto& info) { return info.param.label; }
);

