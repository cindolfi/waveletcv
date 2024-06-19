/**
 * Base Test Class For DWT2D and Filter Bank Unit Tests
*/
#include <fstream>
#include "common.hpp"
#include "json.hpp"
#include "base_dwt2d.hpp"

using namespace cvwt;
using namespace testing;

void PrintTo(const DWT2DTestParam& param, std::ostream* stream)
{
    *stream << "\nwavelet_name: " << param.wavelet_name
            << "\ninput_name: " << param.input_name
            << "\nlevels: " << param.levels
            << "\ncoeffs:\n";
    PrintTo(param.coeffs, stream);
}

std::map<std::string, cv::Mat> BaseDWT2DTest::inputs;
std::vector<DWT2DTestParam> BaseDWT2DTest::params;

BaseDWT2DTest::BaseDWT2DTest() :
    testing::TestWithParam<DWT2DTestParam>(),
    wavelet(Wavelet::create(GetParam().wavelet_name))
{
}

void BaseDWT2DTest::SetUp()
{
    levels = GetParam().levels;
    if (CV_MAT_DEPTH(GetParam().type) == CV_32F) {
        nearness_tolerance = SINGLE_NEARNESS_TOLERANCE;
    } else {
        nearness_tolerance = DOUBLE_NEARNESS_TOLERANCE;
    }
}

void BaseDWT2DTest::get_forward_input(cv::OutputArray input) const
{
    //  Inputs are stored as doubles, so we need to explicitly convert them
    //  here to test for various input types.
    auto param = GetParam();
    BaseDWT2DTest::inputs.at(param.input_name).convertTo(input, param.type);
}

void BaseDWT2DTest::get_forward_output(cv::OutputArray output) const
{
    //  Need to convert coeffs because filter bank kernels are stored as
    //  doubles, which forces a type promotion during decompose.
    auto param = GetParam();
    param.coeffs.convertTo(output, CV_64F);
}

void BaseDWT2DTest::get_inverse_output(cv::OutputArray output) const
{
    //  Need to convert output because filter bank kernels are stored as
    //  doubles, which forces a type promotion during inversion.
    auto param = GetParam();
    BaseDWT2DTest::inputs.at(param.input_name).convertTo(output, CV_64F);
}

void BaseDWT2DTest::init_test_params()
{
    if (!params.empty() && !inputs.empty())
        return;

    //  DWT2D_TEST_DATA_PATH is defined in CMakeLists.txt
    std::ifstream test_case_data_file(DWT2D_TEST_DATA_PATH);
    auto test_case_data = json::parse(test_case_data_file);

    for (auto& [input_name, input] : test_case_data["inputs"].items())
        inputs[input_name] = input.get<cv::Mat>();

    for (auto& test_case : test_case_data["test_cases"]) {
        auto double_precision_coeffs = test_case["coeffs"].get<cv::Mat>();
        cv::Mat single_precision_coeffs;
        double_precision_coeffs.convertTo(single_precision_coeffs, CV_32F);

        params.push_back({
            .wavelet_name = test_case["wavelet_name"],
            .input_name = test_case["input_name"],
            .levels = test_case["levels"],
            .type = double_precision_coeffs.type(),
            .coeffs = double_precision_coeffs,
        });
        params.push_back({
            .wavelet_name = test_case["wavelet_name"],
            .input_name = test_case["input_name"],
            .levels = test_case["levels"],
            .type = single_precision_coeffs.type(),
            .coeffs = single_precision_coeffs,
        });
    }
}

std::vector<DWT2DTestParam> BaseDWT2DTest::create_test_params()
{
    init_test_params();
    return params;
}

