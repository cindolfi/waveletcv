/**
 * Wavelet Unit Tests
*/
#include <fstream>
#include <nlohmann/json.hpp>
#include <cvwt/wavelet.hpp>
#include "common.hpp"

using namespace cvwt;
using namespace testing;
using json = nlohmann::json;

struct WaveletTestParam
{
    int vanishing_moments_psi;
    int vanishing_moments_phi;
    bool orthogonal;
    bool biorthogonal;
    Symmetry symmetry;
    std::string family;
    std::string name;
    std::vector<double> decompose_lowpass;
    std::vector<double> decompose_highpass;
    std::vector<double> reconstruct_lowpass;
    std::vector<double> reconstruct_highpass;
};

void from_json(const json& json_param, WaveletTestParam& param)
{
    param.vanishing_moments_psi = json_param["vanishing_moments_psi"];
    param.vanishing_moments_phi = json_param["vanishing_moments_phi"];
    param.orthogonal = json_param["orthogonal"];
    param.biorthogonal = json_param["biorthogonal"];
    if (json_param["symmetry"] == "symmetric")
        param.symmetry = Symmetry::SYMMETRIC;
    else if (json_param["symmetry"] == "asymmetric")
        param.symmetry = Symmetry::ASYMMETRIC;
    else if (json_param["symmetry"] == "near symmetric")
        param.symmetry = Symmetry::NEAR_SYMMETRIC;
    else
        assert(false);
    param.family = json_param["family"];
    param.name = json_param["name"];
    param.decompose_lowpass = json_param["decompose_lowpass"].get<std::vector<double>>();
    param.decompose_highpass = json_param["decompose_highpass"].get<std::vector<double>>();
    param.reconstruct_lowpass = json_param["reconstruct_lowpass"].get<std::vector<double>>();
    param.reconstruct_highpass = json_param["reconstruct_highpass"].get<std::vector<double>>();
}

void PrintTo(const WaveletTestParam& param, std::ostream* stream)
{
    std::string symmetry;
    switch (param.symmetry) {
    case Symmetry::ASYMMETRIC:
        symmetry = "ASYMMETRIC";
        break;
    case Symmetry::NEAR_SYMMETRIC:
        symmetry = "NEAR_SYMMETRIC";
        break;
    case Symmetry::SYMMETRIC:
        symmetry = "SYMMETRIC";
        break;
    }
    *stream << "\n"
        << "vanishing_moments_psi: " << param.vanishing_moments_psi << "\n"
        << "vanishing_moments_phi: " << param.vanishing_moments_phi << "\n"
        << "orthogonal: " << param.orthogonal << "\n"
        << "biorthogonal: " << param.biorthogonal << "\n"
        << "symmetry: " << symmetry << "\n"
        << "family: " << param.family << "\n"
        << "name: " << param.name << "\n";
}

class WaveletTest : public testing::TestWithParam<WaveletTestParam>
{
protected:
    WaveletTest() :
        testing::TestWithParam<WaveletTestParam>(),
        wavelet(cvwt::Wavelet::create(GetParam().name))
    {}

    cvwt::Wavelet wavelet;

public:
    static std::vector<WaveletTestParam> create_test_params()
    {
        //  WAVELET_TEST_DATA_PATH is defined in CMakeLists.txt
        std::ifstream test_case_data_file(WAVELET_TEST_DATA_PATH);
        auto test_case_data = json::parse(test_case_data_file);

        std::vector<WaveletTestParam> params;
        for (auto& test_case : test_case_data)
            params.push_back(test_case.get<WaveletTestParam>());

        return params;
    }
};

TEST_P(WaveletTest, VanisingMomentsPsi)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.vanishing_moments_psi(), param.vanishing_moments_psi);
}

TEST_P(WaveletTest, VanisingMomentsPhi)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.vanishing_moments_phi(), param.vanishing_moments_phi);
}

TEST_P(WaveletTest, Orthogonal)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.orthogonal(), param.orthogonal);
}

TEST_P(WaveletTest, Biorthogonal)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.biorthogonal(), param.biorthogonal);
}

TEST_P(WaveletTest, Symmetry)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.symmetry(), param.symmetry);
}

TEST_P(WaveletTest, Family)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.family(), param.family);
}

TEST_P(WaveletTest, Name)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.name(), param.name);
}

TEST_P(WaveletTest, DecomposeLowpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().decompose_kernels().lowpass(),
        MatrixFloatEq(cv::Mat(param.decompose_lowpass))
    );
}

TEST_P(WaveletTest, DecomposeHighpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().decompose_kernels().highpass(),
        MatrixFloatEq(cv::Mat(param.decompose_highpass))
    );
}

TEST_P(WaveletTest, ReconstructLowpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().reconstruct_kernels().lowpass(),
        MatrixFloatEq(cv::Mat(param.reconstruct_lowpass))
    );
}

TEST_P(WaveletTest, ReconstructHighpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().reconstruct_kernels().highpass(),
        MatrixFloatEq(cv::Mat(param.reconstruct_highpass))
    );
}


INSTANTIATE_TEST_CASE_P(
    WaveletGroup,
    WaveletTest,
    testing::ValuesIn(
        WaveletTest::create_test_params()
    ),
    [](const auto& info) {
        auto name = info.param.name;
        std::ranges::replace(name, '.', '_');
        return name;
    }
);

