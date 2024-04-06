/**
 * Utils Unit Test
*/
#include <numeric>
#include <wavelet/utils.hpp>
#include "common.hpp"

using namespace cvwt;
using namespace testing;

/**
 * -----------------------------------------------------------------------------
 * Median Test
 * -----------------------------------------------------------------------------
*/
struct MedianTestParam
{
    std::vector<double> values;
    double expected_median;
};

void PrintTo(const MedianTestParam& param, std::ostream* stream)
{
    *stream << "{" << join(param.values, ", ") << "} => " << param.expected_median;
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
        for (int n = 2; n < MAX_SIZE; ++n)
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
                .values = {0},
                .expected_median = 0,
            },
            {
                .values = {0, 0},
                .expected_median = 0,
            },
            {
                .values = {0, 0, 0},
                .expected_median = 0,
            },
            {
                .values = {0, 0, 0, 0},
                .expected_median = 0,
            },
            //  nonzero constants
            {
                .values = {1},
                .expected_median = 1,
            },
            {
                .values = {1, 1},
                .expected_median = 1,
            },
            {
                .values = {1, 1, 1},
                .expected_median = 1,
            },
            {
                .values = {1, 1, 1, 1},
                .expected_median = 1,
            },
        };
        for (int n = 2; n < MAX_SIZE; ++n)
        {
            std::vector<int> values(n);
            std::iota(values.begin(), values.end(), 0);
            do {
                std::vector<double> x;
                std::ranges::copy(values, std::back_inserter(x));
                params.push_back({
                    .values = x,
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
    cv::Mat matrix(param.values);
    auto actual_median = median(matrix)[0];

    EXPECT_DOUBLE_EQ(actual_median, param.expected_median);
}

auto median_test_labels = MedianTest::create_labels();

INSTANTIATE_TEST_CASE_P(
    MedianGroup,
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

auto multi_channel_median_test_labels = MultiChannelMedianTest::create_labels();

INSTANTIATE_TEST_CASE_P(
    MedianGroup,
    MultiChannelMedianTest,
    testing::ValuesIn(MultiChannelMedianTest::create_params()),
    [&](const auto& info) { return multi_channel_median_test_labels[info.index]; }
);




/**
 * -----------------------------------------------------------------------------
 * Collect Masked Test
 * -----------------------------------------------------------------------------
*/
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




/**
 * -----------------------------------------------------------------------------
 * Negate Every Other Tests
 * -----------------------------------------------------------------------------
*/
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
    negate_evens(param.input, actual_output);

    EXPECT_THAT(actual_output, MatrixEq(expected_output));
}

TEST_P(NegateEveryOtherTest, NegateOdds)
{
    auto param = GetParam();
    auto expected_output = param.negated_at_evens_indices.empty()
                         ? cv::Mat()
                         : -param.negated_at_evens_indices;

    cv::Mat actual_output;
    negate_odds(param.input, actual_output);

    EXPECT_THAT(actual_output, MatrixEq(expected_output));
}

INSTANTIATE_TEST_CASE_P(
    NegateGroup,
    NegateEveryOtherTest,
    testing::ValuesIn(NegateEveryOtherTest::create_test_params())
);

