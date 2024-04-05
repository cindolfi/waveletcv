#ifndef WAVELET_TEST_BASE_DWT2D_HPP
#define WAVELET_TEST_BASE_DWT2D_HPP

#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include <wavelet/dwt2d.hpp>
#include <wavelet/utils.hpp>
#include "common.hpp"

using json = nlohmann::json;

/**
 * =============================================================================
 * Transformation Tests
 * =============================================================================
*/
struct DWT2DTestParam
{
    std::string wavelet_name;
    std::string input_name;
    int levels;
    int type;
    cv::Mat coeffs;
};

void PrintTo(const DWT2DTestParam& param, std::ostream* stream);

namespace cv
{
    void from_json(const json& json_matrix, Mat& matrix);
}

class BaseDWT2DTest : public testing::TestWithParam<DWT2DTestParam>
{
public:
    const double SINGLE_NEARNESS_TOLERANCE = 1e-6;
    const double DOUBLE_NEARNESS_TOLERANCE = 1e-10;

protected:
    BaseDWT2DTest();

    void SetUp() override;
    void get_forward_input(cv::OutputArray input) const;
    void get_forward_output(cv::OutputArray output) const;
    void get_inverse_output(cv::OutputArray output) const;

    void clamp_small_to_zero(auto& ...matrices) const
    {
        (clamp_near_zero(matrices, 0.1 * nearness_tolerance), ...);
    }

    static void init_test_params();

    wavelet::Wavelet wavelet;
    int levels;
    double nearness_tolerance;
    static std::map<std::string, cv::Mat> inputs;
    static std::vector<DWT2DTestParam> params;

public:
    static std::vector<DWT2DTestParam> create_test_params();
};

#endif  // WAVELET_TEST_BASE_DWT2D_HPP

