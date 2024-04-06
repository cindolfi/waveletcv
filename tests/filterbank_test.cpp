/**
 * Filter Bank Unit Tests
*/
#include <vector>
#include <cvwt/filterbank.hpp>
#include "common.hpp"
#include "base_dwt2d.hpp"

using namespace cvwt;
using namespace testing;

// =============================================================================
// Filter Bank
// =============================================================================
auto print_filter_bank_test_label  = [](const auto& info) {
    std::string wavelet_name = info.param.wavelet_name;
    std::ranges::replace(wavelet_name, '.', '_');
    return wavelet_name + "_"
        + info.param.input_name + "_"
        + get_type_name(info.param.type);
};

class FilterBankTest : public BaseDWT2DTest
{
protected:
    void get_subbands(
        const cv::Mat& coeffs,
        cv::Mat& approx,
        cv::Mat& horizontal_detail,
        cv::Mat& vertical_detail,
        cv::Mat& diagonal_detail
    )
    {
        BaseDWT2DTest::SetUp();

        cv::Size subband_size = coeffs.size() / 2;
        approx = coeffs(
            cv::Rect(0, 0, subband_size.width, subband_size.height)
        );
        horizontal_detail = coeffs(
            cv::Rect(0, subband_size.height, subband_size.width, subband_size.height)
        );
        vertical_detail = coeffs(
            cv::Rect(subband_size.width, 0, subband_size.width, subband_size.height)
        );
        diagonal_detail = coeffs(
            cv::Rect(subband_size.width, subband_size.height, subband_size.width, subband_size.height)
        );
    }

public:
    static std::vector<DWT2DTestParam> create_test_params()
    {
        init_test_params();
        std::vector<DWT2DTestParam> levels_params;
        std::ranges::copy_if(
            params,
            std::back_inserter(levels_params),
            [&](auto param) { return param.levels == 1; }
        );

        return levels_params;
    }
};


//  ----------------------------------------------------------------------------
//  Decompose
//  ----------------------------------------------------------------------------
class FilterBankDecomposeTest : public FilterBankTest
{
protected:
    void SetUp() override
    {
        FilterBankTest::SetUp();

        get_forward_input(input);
        get_forward_output(expected_output);
        get_subbands(
            expected_output,
            expected_approx,
            expected_horizontal_detail,
            expected_vertical_detail,
            expected_diagonal_detail
        );
    }

    cv::Mat input;
    cv::Mat expected_approx;
    cv::Mat expected_horizontal_detail;
    cv::Mat expected_vertical_detail;
    cv::Mat expected_diagonal_detail;
    cv::Mat expected_output;
};

TEST_P(FilterBankDecomposeTest, OutputSize)
{
    auto output_size = wavelet.filter_bank().output_size(input.size());

    EXPECT_EQ(output_size, expected_output.size());
}

TEST_P(FilterBankDecomposeTest, SubbandSize)
{
    auto subband_size = wavelet.filter_bank().subband_size(input.size());

    EXPECT_EQ(subband_size, expected_approx.size());
}

TEST_P(FilterBankDecomposeTest, Decompose)
{
    cv::Mat actual_approx;
    cv::Mat actual_horizontal_detail;
    cv::Mat actual_vertical_detail;
    cv::Mat actual_diagonal_detail;

    wavelet.filter_bank().decompose(
        input,
        actual_approx,
        actual_horizontal_detail,
        actual_vertical_detail,
        actual_diagonal_detail,
        cv::BORDER_REFLECT101
    );

    //  Clamping is only for readability of failure messages.  It does not
    //  impact the test because the clamp tolerance is smaller than the
    //  the nearness_tolerance.
    clamp_small_to_zero(
        actual_approx,
        actual_horizontal_detail,
        actual_vertical_detail,
        actual_diagonal_detail,
        expected_approx,
        expected_horizontal_detail,
        expected_vertical_detail,
        expected_diagonal_detail
    );

    EXPECT_THAT(
        actual_approx,
        MatrixNear(expected_approx, nearness_tolerance)
    ) << "approx is incorrect";
    EXPECT_THAT(
        actual_horizontal_detail,
        MatrixNear(expected_horizontal_detail, nearness_tolerance)
    ) << "horizontal_detail is incorrect";
    EXPECT_THAT(
        actual_vertical_detail,
        MatrixNear(expected_vertical_detail, nearness_tolerance)
    ) << "vertical_detail is incorrect";
    EXPECT_THAT(
        actual_diagonal_detail,
        MatrixNear(expected_diagonal_detail, nearness_tolerance)
    ) << "diagonal_detail is incorrect";
}


INSTANTIATE_TEST_CASE_P(
    FilterBankGroup,
    FilterBankDecomposeTest,
    testing::ValuesIn(FilterBankDecomposeTest::create_test_params()),
    print_filter_bank_test_label
);


//  ----------------------------------------------------------------------------
//  Reconstruct
//  ----------------------------------------------------------------------------
class FilterBankReconstructTest : public FilterBankTest
{
protected:
    void SetUp() override
    {
        FilterBankTest::SetUp();

        auto param = GetParam();
        get_subbands(
            param.coeffs,
            approx,
            horizontal_detail,
            vertical_detail,
            diagonal_detail
        );
        get_inverse_output(expected_output);
    }

    cv::Mat approx;
    cv::Mat horizontal_detail;
    cv::Mat vertical_detail;
    cv::Mat diagonal_detail;
    cv::Mat expected_output;
};

TEST_P(FilterBankReconstructTest, Reconstruct)
{
    cv::Mat actual_output;
    wavelet.filter_bank().reconstruct(
        approx,
        horizontal_detail,
        vertical_detail,
        diagonal_detail,
        actual_output,
        expected_output.size()
    );

    //  Clamping is only for readability of failure messages.  It does not
    //  impact the test because the clamp tolerance is smaller than the
    //  the nearness_tolerance.
    clamp_small_to_zero(actual_output, expected_output);

    EXPECT_THAT(actual_output, MatrixNear(expected_output, nearness_tolerance));
}


INSTANTIATE_TEST_CASE_P(
    FilterBankGroup,
    FilterBankReconstructTest,
    testing::ValuesIn(FilterBankReconstructTest::create_test_params()),
    print_filter_bank_test_label
);

