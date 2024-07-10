/**
 * Wavelet Unit Tests
*/
#include <fstream>
#include <nlohmann/json.hpp>
#include <wtcv/wavelet.hpp>
#include "common.hpp"
#include "json.hpp"

using namespace wtcv;
using namespace testing;

struct WaveletTestParam
{
    int wavelet_vanishing_moments;
    int scaling_vanishing_moments;
    Orthogonality orthogonality;
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
    param.wavelet_vanishing_moments = json_param["vanishing_moments_psi"];
    param.scaling_vanishing_moments = json_param["vanishing_moments_phi"];
    if (json_param["orthogonal"].get<bool>())
        param.orthogonality = Orthogonality::ORTHOGONAL;
    else if (json_param["biorthogonal"].get<bool>())
        param.orthogonality = Orthogonality::BIORTHOGONAL;
    else
        param.orthogonality = Orthogonality::NONE;

    if (json_param["symmetry"] == "symmetric")
        param.symmetry = Symmetry::SYMMETRIC;
    else if (json_param["symmetry"] == "asymmetric")
        param.symmetry = Symmetry::ASYMMETRIC;
    else if (json_param["symmetry"] == "near symmetric")
        param.symmetry = Symmetry::NEARLY_SYMMETRIC;
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
    case Symmetry::NEARLY_SYMMETRIC:
        symmetry = "NEARLY_SYMMETRIC";
        break;
    case Symmetry::SYMMETRIC:
        symmetry = "SYMMETRIC";
        break;
    }

    std::string orthogonality;
    switch (param.orthogonality) {
    case Orthogonality::ORTHOGONAL:
        orthogonality = "ORTHOGONAL";
        break;
    case Orthogonality::BIORTHOGONAL:
        orthogonality = "BIORTHOGONAL";
        break;
    case Orthogonality::NONE:
        orthogonality = "NONE";
    }

    *stream << "\n"
        << "wavelet_vanishing_moments: " << param.wavelet_vanishing_moments << "\n"
        << "scaling_vanishing_moments: " << param.scaling_vanishing_moments << "\n"
        << "orthogonality: " << orthogonality << "\n"
        << "symmetry: " << symmetry << "\n"
        << "family: " << param.family << "\n"
        << "name: " << param.name << "\n";
}

class WaveletTest : public testing::TestWithParam<WaveletTestParam>
{
protected:
    WaveletTest() :
        testing::TestWithParam<WaveletTestParam>(),
        wavelet(wtcv::Wavelet::create(GetParam().name))
    {}

    wtcv::Wavelet wavelet;

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

TEST_P(WaveletTest, Orthogonality)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.orthogonality(), param.orthogonality);
}

TEST_P(WaveletTest, IsOrthogonal)
{
    auto param = GetParam();
    if (param.orthogonality == Orthogonality::ORTHOGONAL
        || param.name == "bior1.1"
        || param.name == "rbior1.1"
        ) {
        EXPECT_TRUE(wavelet.is_orthogonal());
        EXPECT_TRUE(wavelet.filter_bank().is_orthogonal());
    } else {
        EXPECT_FALSE(wavelet.is_orthogonal());
        EXPECT_FALSE(wavelet.filter_bank().is_orthogonal());
    }
}

TEST_P(WaveletTest, IsBiorthogonal)
{
    auto param = GetParam();
    if (param.orthogonality == Orthogonality::BIORTHOGONAL
        || param.orthogonality == Orthogonality::ORTHOGONAL) {
        EXPECT_TRUE(wavelet.is_biorthogonal());
        EXPECT_TRUE(wavelet.filter_bank().is_biorthogonal());
    } else {
        EXPECT_FALSE(wavelet.is_biorthogonal());
        EXPECT_FALSE(wavelet.filter_bank().is_biorthogonal());
    }
}

TEST_P(WaveletTest, Symmetry)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.symmetry(), param.symmetry);
}

TEST_P(WaveletTest, IsSymmetric)
{
    auto param = GetParam();
    if (param.symmetry == Symmetry::SYMMETRIC) {
        EXPECT_TRUE(wavelet.is_symmetric());
        EXPECT_TRUE(wavelet.filter_bank().is_linear_phase());
    } else {
        EXPECT_FALSE(wavelet.is_symmetric());
        if (param.name == "haar" || param.name == "db1")
            EXPECT_TRUE(wavelet.filter_bank().is_linear_phase());
        else
            EXPECT_FALSE(wavelet.filter_bank().is_linear_phase());
    }
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

TEST_P(WaveletTest, VanisingMomentsPsi)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.wavelet_vanishing_moments(), param.wavelet_vanishing_moments);
}

TEST_P(WaveletTest, VanisingMomentsPhi)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.scaling_vanishing_moments(), param.scaling_vanishing_moments);
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

