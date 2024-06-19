/**
 * Statistics Unit Test
*/
#include <numeric>
#include <cvwt/statistics.hpp>
#include "common.hpp"

using namespace cvwt;
using namespace testing;

//  ----------------------------------------------------------------------------
//  Median Test
//  ----------------------------------------------------------------------------
struct MedianTestParam
{
    cv::Mat matrix;
    double expected_median;
};

void PrintTo(const MedianTestParam& param, std::ostream* stream)
{
    *stream << "{" << join(std::vector<double>(param.matrix), ", ") << "} => " << param.expected_median;
}

class MedianTest : public testing::TestWithParam<MedianTestParam>
{
    static const int MAX_SIZE = 6;

public:
    static std::vector<std::string> create_labels()
    {
        std::vector<std::string> labels = {
            "zeros_size_1",
            "zeros_size_2",
            "zeros_size_3",
            "zeros_size_4",
            "constant_nonzero_size_1",
            "constant_nonzero_size_2",
            "constant_nonzero_size_3",
            "constant_nonzero_size_4",
        };
        for (int n = 1; n < MAX_SIZE; ++n)
        {
            std::vector<int> values(n);
            std::iota(values.begin(), values.end(), 0);
            do {
                std::stringstream stream;
                stream << "seq_size_" << n << "_perm_";
                for (auto x : values)
                    stream << x;
                labels.push_back(stream.str());
            } while (std::next_permutation(values.begin(), values.end()));
        }

        return labels;
    }

    static std::vector<MedianTestParam> create_params()
    {
        std::vector<MedianTestParam> params = {
            //  zeros
            {
                .matrix = cv::Mat(1, 1, CV_64F, cv::Scalar::all(0.0)),
                .expected_median = 0,
            },
            {
                .matrix = cv::Mat(2, 1, CV_64F, cv::Scalar::all(0.0)),
                .expected_median = 0,
            },
            {
                .matrix = cv::Mat(3, 1, CV_64F, cv::Scalar::all(0.0)),
                .expected_median = 0,
            },
            {
                .matrix = cv::Mat(4, 1, CV_64F, cv::Scalar::all(0.0)),
                .expected_median = 0,
            },
            //  nonzero constants
            {
                .matrix = cv::Mat(1, 1, CV_64F, cv::Scalar::all(1.0)),
                .expected_median = 1,
            },
            {
                .matrix = cv::Mat(2, 1, CV_64F, cv::Scalar::all(1.0)),
                .expected_median = 1,
            },
            {
                .matrix = cv::Mat(3, 1, CV_64F, cv::Scalar::all(1.0)),
                .expected_median = 1,
            },
            {
                .matrix = cv::Mat(4, 1, CV_64F, cv::Scalar::all(1.0)),
                .expected_median = 1,
            },
        };
        for (int n = 1; n < MAX_SIZE; ++n)
        {
            std::vector<int> values(n);
            std::iota(values.begin(), values.end(), 0);
            do {
                std::vector<double> x;
                std::ranges::copy(values, std::back_inserter(x));
                params.push_back({
                    .matrix = cv::Mat(x, true),
                    .expected_median = 0.5 * (n - 1),
                });
            } while (std::next_permutation(values.begin(), values.end()));
        }

        return params;
    }
};

TEST_P(MedianTest, MedianIsCorrect)
{
    auto param = GetParam();
    auto actual_median = median(param.matrix)[0];

    EXPECT_DOUBLE_EQ(actual_median, param.expected_median);
}

TEST_P(MedianTest, MatrixIsUnaffected)
{
    auto param = GetParam();
    auto matrix_clone = param.matrix.clone();
    auto actual_median = median(param.matrix)[0];

    EXPECT_THAT(param.matrix, MatrixEq(matrix_clone));
}

TEST_P(MedianTest, CovariantUnderNegation)
{
    auto param = GetParam();
    cv::Mat matrix = -param.matrix;
    auto actual_median = median(matrix)[0];

    EXPECT_DOUBLE_EQ(actual_median, -param.expected_median);
}

auto median_test_labels = MedianTest::create_labels();

INSTANTIATE_TEST_CASE_P(
    StatisticsGroup,
    MedianTest,
    testing::ValuesIn(MedianTest::create_params()),
    [&](const auto& info) { return median_test_labels[info.index]; }
);


//  ----------------------------------------------------------------------------
template<typename T, int CHANNELS>
struct MultiChannelMedianTestParam
{
    using Pixel = cv::Vec<T, CHANNELS>;
    using Matrix = cv::Mat_<Pixel>;

    cv::Mat values;
    cv::Scalar expected_median;
};

template<typename T, int CHANNELS>
void PrintTo(const MultiChannelMedianTestParam<T, CHANNELS>& param, std::ostream* stream)
{
    *stream << "\n";
    PrintTo(param.values, stream);
    *stream << "=> " << param.expected_median;
}

class MultiChannelMedianTest : public testing::TestWithParam<MultiChannelMedianTestParam<double, 4>>
{
    using Matrix = typename ParamType::Matrix;
    using Pixel = typename ParamType::Pixel;

public:
    static std::vector<std::string> create_labels()
    {
        std::vector<std::string> labels = {
            "zeros_1x1",
            "zeros_1x2",
            "zeros_3x1",
            "zeros_2x2",
            "constant_nonzero_1x1",
            "constant_nonzero_1x2",
            "constant_nonzero_3x1",
            "constant_nonzero_2x2",
            "constant_per_channel_2x2",
            "constant_per_channel_4x1",
            "constant_per_channel_1x4",
            "constant_per_channel_3x3",
            "constant_per_channel_9x1",
            "constant_per_channel_1x9",
            "permutation_per_channel_2x2",
            "permutation_per_channel_4x1",
            "permutation_per_channel_1x4",
            "permutation_per_channel_3x3",
            "permutation_per_channel_9x1",
            "permutation_per_channel_1x9",
            "different_per_channel_2x2",
            "different_per_channel_4x1",
            "different_per_channel_1x4",
            "different_per_channel_3x3",
            "different_per_channel_9x1",
            "different_per_channel_1x9",
        };

        return labels;
    }

    static std::vector<ParamType> create_params()
    {
        std::vector<ParamType> params = {
            //  zeros
            {
                .values = (Matrix(1, 1) << cv::Scalar::all(0)),
                .expected_median = cv::Scalar::all(0),
            },
            {
                .values = (Matrix(1, 2) << cv::Scalar::all(0), cv::Scalar::all(0)),
                .expected_median = cv::Scalar::all(0),
            },
            {
                .values = (Matrix(3, 1) <<
                    cv::Scalar::all(0),
                    cv::Scalar::all(0),
                    cv::Scalar::all(0)
                ),
                .expected_median = cv::Scalar::all(0),
            },
            {
                .values = (Matrix(2, 2) <<
                    cv::Scalar::all(0), cv::Scalar::all(0),
                    cv::Scalar::all(0), cv::Scalar::all(0)
                ),
                .expected_median = cv::Scalar::all(0),
            },
            //  nonzero constants
            {
                .values = (Matrix(1, 1) << cv::Scalar::all(1)),
                .expected_median = cv::Scalar::all(1),
            },
            {
                .values = (Matrix(1, 2) << cv::Scalar::all(1), cv::Scalar::all(1)),
                .expected_median = cv::Scalar::all(1),
            },
            {
                .values = (Matrix(3, 1) <<
                    cv::Scalar::all(1),
                    cv::Scalar::all(1),
                    cv::Scalar::all(1)
                ),
                .expected_median = cv::Scalar::all(1),
            },
            {
                .values = (Matrix(2, 2) <<
                    cv::Scalar::all(1), cv::Scalar::all(1),
                    cv::Scalar::all(1), cv::Scalar::all(1)
                ),
                .expected_median = cv::Scalar::all(1),
            },
            //  nonzero constant per channels
            {
                .values = (Matrix(2, 2) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3)
                ),
                .expected_median = cv::Scalar(0, 1, 2, 3),
            },
            {
                .values = (Matrix(4, 1) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3)
                ),
                .expected_median = cv::Scalar(0, 1, 2, 3),
            },
            {
                .values = (Matrix(1, 4) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3)
                ),
                .expected_median = cv::Scalar(0, 1, 2, 3),
            },
            {
                .values = (Matrix(3, 3) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3)
                ),
                .expected_median = cv::Scalar(0, 1, 2, 3),
            },
            {
                .values = (Matrix(9, 1) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3)
                ),
                .expected_median = cv::Scalar(0, 1, 2, 3),
            },
            {
                .values = (Matrix(1, 9) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3),
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3), cv::Scalar(0, 1, 2, 3)
                ),
                .expected_median = cv::Scalar(0, 1, 2, 3),
            },
            //  permutation per channel
            {
                .values = (Matrix(2, 2) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 0),
                    cv::Scalar(2, 3, 0, 1), cv::Scalar(3, 0, 1, 2)
                ),
                .expected_median = cv::Scalar(1.5, 1.5, 1.5, 1.5),
            },
            {
                .values = (Matrix(4, 1) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 0),
                    cv::Scalar(2, 3, 0, 1), cv::Scalar(3, 0, 1, 2)
                ),
                .expected_median = cv::Scalar(1.5, 1.5, 1.5, 1.5),
            },
            {
                .values = (Matrix(1, 4) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 0),
                    cv::Scalar(2, 3, 0, 1), cv::Scalar(3, 0, 1, 2)
                ),
                .expected_median = cv::Scalar(1.5, 1.5, 1.5, 1.5),
            },
            {
                .values = (Matrix(3, 3) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
                    cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7), cv::Scalar(5, 6, 7, 8),
                    cv::Scalar(6, 7, 8, 0), cv::Scalar(7, 8, 0, 1), cv::Scalar(8, 0, 1, 2)
                ),
                .expected_median = cv::Scalar(4, 4, 4, 4),
            },
            {
                .values = (Matrix(9, 1) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
                    cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7), cv::Scalar(5, 6, 7, 8),
                    cv::Scalar(6, 7, 8, 0), cv::Scalar(7, 8, 0, 1), cv::Scalar(8, 0, 1, 2)
                ),
                .expected_median = cv::Scalar(4, 4, 4, 4),
            },
            {
                .values = (Matrix(1, 9) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
                    cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7), cv::Scalar(5, 6, 7, 8),
                    cv::Scalar(6, 7, 8, 0), cv::Scalar(7, 8, 0, 1), cv::Scalar(8, 0, 1, 2)
                ),
                .expected_median = cv::Scalar(4, 4, 4, 4),
            },
            //  different per channel
            {
                .values = (Matrix(2, 2) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4),
                    cv::Scalar(2, 3, 4, 5), cv::Scalar(3, 4, 5, 6)
                ),
                .expected_median = cv::Scalar(1.5, 2.5, 3.5, 4.5),
            },
            {
                .values = (Matrix(4, 1) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4),
                    cv::Scalar(2, 3, 4, 5), cv::Scalar(3, 4, 5, 6)
                ),
                .expected_median = cv::Scalar(1.5, 2.5, 3.5, 4.5),
            },
            {
                .values = (Matrix(1, 4) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4),
                    cv::Scalar(2, 3, 4, 5), cv::Scalar(3, 4, 5, 6)
                ),
                .expected_median = cv::Scalar(1.5, 2.5, 3.5, 4.5),
            },
            {
                .values = (Matrix(3, 3) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
                    cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7), cv::Scalar(5, 6, 7, 8),
                    cv::Scalar(6, 7, 8, 9), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11)
                ),
                .expected_median = cv::Scalar(4, 5, 6, 7),
            },
            {
                .values = (Matrix(9, 1) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
                    cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7), cv::Scalar(5, 6, 7, 8),
                    cv::Scalar(6, 7, 8, 9), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11)
                ),
                .expected_median = cv::Scalar(4, 5, 6, 7),
            },
            {
                .values = (Matrix(1, 9) <<
                    cv::Scalar(0, 1, 2, 3), cv::Scalar(1, 2, 3, 4), cv::Scalar(2, 3, 4, 5),
                    cv::Scalar(3, 4, 5, 6), cv::Scalar(4, 5, 6, 7), cv::Scalar(5, 6, 7, 8),
                    cv::Scalar(6, 7, 8, 9), cv::Scalar(7, 8, 9, 10), cv::Scalar(8, 9, 10, 11)
                ),
                .expected_median = cv::Scalar(4, 5, 6, 7),
            },
        };

        return params;
    }
};


TEST_P(MultiChannelMedianTest, MedianIsCorrect)
{
    auto param = GetParam();
    auto actual_median = median(param.values);

    EXPECT_THAT(actual_median, ScalarEq(param.expected_median));
}

TEST_P(MultiChannelMedianTest, MatrixIsUnaffected)
{
    auto param = GetParam();
    auto values_clone = param.values.clone();
    auto actual_median = median(param.values);

    EXPECT_THAT(param.values, MatrixEq(values_clone));
}

auto multi_channel_median_test_labels = MultiChannelMedianTest::create_labels();

INSTANTIATE_TEST_CASE_P(
    StatisticsGroup,
    MultiChannelMedianTest,
    testing::ValuesIn(MultiChannelMedianTest::create_params()),
    [&](const auto& info) { return multi_channel_median_test_labels[info.index]; }
);


//  ============================================================================
//  MAD
//  ============================================================================
const std::vector<int> PERMUTATION1 = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
const std::vector<int> PERMUTATION2 = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};

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
    StatisticsGroup,
    MadTest,
    testing::ValuesIn(MadTest::create_params())
);

