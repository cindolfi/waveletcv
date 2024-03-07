/**
 * Wavelet & DWT2D Unit Tests
*/
#include <wavelet/wavelet.hpp>
#include "common.hpp"

using namespace wavelet;
using namespace testing;

struct WaveletTestParam
{
    int order;
    int vanishing_moments_psi;
    int support_width;
    std::string short_name;
    std::vector<double> analysis_lowpass;
    std::vector<double> analysis_highpass;
    std::vector<double> synthesis_lowpass;
    std::vector<double> synthesis_highpass;
};

class WaveletTest : public testing::TestWithParam<WaveletTestParam>
{
protected:
    WaveletTest(const Wavelet& wavelet) :
        testing::TestWithParam<WaveletTestParam>(),
        wavelet(wavelet)
    {}

    Wavelet wavelet;
};


/**
 * -----------------------------------------------------------------------------
 * Daubechies
 * -----------------------------------------------------------------------------
*/
class DaubechiesTest : public WaveletTest
{
protected:
    DaubechiesTest() : WaveletTest(daubechies(GetParam().order))
    {}
};

TEST_P(DaubechiesTest, Order)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.order(), param.order);
}

TEST_P(DaubechiesTest, VanisingMomentsPsi)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.vanishing_moments_psi(), param.vanishing_moments_psi);
}

TEST_P(DaubechiesTest, VanisingMomentsPhi)
{
    ASSERT_EQ(wavelet.vanishing_moments_phi(), 0);
}

TEST_P(DaubechiesTest, SupportWidth)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.support_width(), param.support_width);
}

TEST_P(DaubechiesTest, Orthogonal)
{
    ASSERT_EQ(wavelet.orthogonal(), true);
}

TEST_P(DaubechiesTest, Biorthogonal)
{
    ASSERT_EQ(wavelet.biorthogonal(), true);
}

TEST_P(DaubechiesTest, Symmetry)
{
    ASSERT_EQ(wavelet.symmetry(), Wavelet::Symmetry::ASYMMETRIC);
}

TEST_P(DaubechiesTest, CompactSupport)
{
    ASSERT_EQ(wavelet.compact_support(), true);
}

TEST_P(DaubechiesTest, FamilyName)
{
    ASSERT_EQ(wavelet.family_name(), "Daubechies");
}

TEST_P(DaubechiesTest, ShortName)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.short_name(), param.short_name);
}

TEST_P(DaubechiesTest, CoeffsSize)
{
    auto param = GetParam();
    EXPECT_EQ(
        wavelet.filter_bank().analysis_kernels().lowpass().total(),
        2 * param.order
    );
    EXPECT_EQ(
        wavelet.filter_bank().analysis_kernels().highpass().total(),
        2 * param.order
    );
    EXPECT_EQ(
        wavelet.filter_bank().synthesis_kernels().lowpass().total(),
        2 * param.order
    );
    EXPECT_EQ(
        wavelet.filter_bank().synthesis_kernels().highpass().total(),
        2 * param.order
    );
}

TEST_P(DaubechiesTest, AnalysisLowpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().analysis_kernels().lowpass(),
        MatrixEq(cv::Mat(param.analysis_lowpass))
    );
}

TEST_P(DaubechiesTest, AnalysisHighpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().analysis_kernels().highpass(),
        MatrixEq(cv::Mat(param.analysis_highpass))
    );
}

TEST_P(DaubechiesTest, SynthesisLowpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().synthesis_kernels().lowpass(),
        MatrixEq(cv::Mat(param.synthesis_lowpass))
    );
}

TEST_P(DaubechiesTest, SynthesisHighpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().synthesis_kernels().highpass(),
        MatrixEq(cv::Mat(param.synthesis_highpass))
    );
}


INSTANTIATE_TEST_CASE_P(
    WaveletGroup,
    DaubechiesTest,
    testing::Values(
        WaveletTestParam{
            .order = 1,
            .vanishing_moments_psi = 2,
            .support_width = 1,
            .short_name = "db1",
            .analysis_lowpass = {
                7.071067811865475244008443621048490392848359376884740365883398e-01,
                7.071067811865475244008443621048490392848359376884740365883398e-01,
            },
            .analysis_highpass = {
                -7.071067811865475244008443621048490392848359376884740365883398e-01,
                7.071067811865475244008443621048490392848359376884740365883398e-01,
            },
            .synthesis_lowpass = {
                7.071067811865475244008443621048490392848359376884740365883398e-01,
                7.071067811865475244008443621048490392848359376884740365883398e-01,
            },
            .synthesis_highpass = {
                7.071067811865475244008443621048490392848359376884740365883398e-01,
                -7.071067811865475244008443621048490392848359376884740365883398e-01,
            },
        },
        WaveletTestParam{
            .order = 2,
            .vanishing_moments_psi = 4,
            .support_width = 3,
            .short_name = "db2",
            .analysis_lowpass = {
                -1.294095225512603811744494188120241641745344506599652569070016e-01,
                2.241438680420133810259727622404003554678835181842717613871683e-01,
                8.365163037378079055752937809168732034593703883484392934953414e-01,
                4.829629131445341433748715998644486838169524195042022752011715e-01,
            },
            .analysis_highpass = {
                -4.829629131445341433748715998644486838169524195042022752011715e-01,
                8.365163037378079055752937809168732034593703883484392934953414e-01,
                -2.241438680420133810259727622404003554678835181842717613871683e-01,
                -1.294095225512603811744494188120241641745344506599652569070016e-01,
            },
            .synthesis_lowpass = {
                4.829629131445341433748715998644486838169524195042022752011715e-01,
                8.365163037378079055752937809168732034593703883484392934953414e-01,
                2.241438680420133810259727622404003554678835181842717613871683e-01,
                -1.294095225512603811744494188120241641745344506599652569070016e-01,
            },
            .synthesis_highpass = {
                -1.294095225512603811744494188120241641745344506599652569070016e-01,
                -2.241438680420133810259727622404003554678835181842717613871683e-01,
                8.365163037378079055752937809168732034593703883484392934953414e-01,
                -4.829629131445341433748715998644486838169524195042022752011715e-01,
            },
        }
    )
);


/**
 * -----------------------------------------------------------------------------
 * Haar
 * -----------------------------------------------------------------------------
*/
class HaarTest : public WaveletTest
{
protected:
    HaarTest() : WaveletTest(haar())
    {}
};

TEST_P(HaarTest, Order)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.order(), param.order);
}

TEST_P(HaarTest, VanisingMomentsPsi)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.vanishing_moments_psi(), param.vanishing_moments_psi);
}

TEST_P(HaarTest, VanisingMomentsPhi)
{
    ASSERT_EQ(wavelet.vanishing_moments_phi(), 0);
}

TEST_P(HaarTest, SupportWidth)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.support_width(), param.support_width);
}

TEST_P(HaarTest, Orthogonal)
{
    ASSERT_EQ(wavelet.orthogonal(), true);
}

TEST_P(HaarTest, Biorthogonal)
{
    ASSERT_EQ(wavelet.biorthogonal(), true);
}

TEST_P(HaarTest, Symmetry)
{
    ASSERT_EQ(wavelet.symmetry(), Wavelet::Symmetry::ASYMMETRIC);
}

TEST_P(HaarTest, CompactSupport)
{
    ASSERT_EQ(wavelet.compact_support(), true);
}

TEST_P(HaarTest, FamilyName)
{
    ASSERT_EQ(wavelet.family_name(), "Haar");
}

TEST_P(HaarTest, ShortName)
{
    auto param = GetParam();
    ASSERT_EQ(wavelet.short_name(), param.short_name);
}

TEST_P(HaarTest, AnalysisLowpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().analysis_kernels().lowpass(),
        MatrixEq(cv::Mat(param.analysis_lowpass))
    );
}

TEST_P(HaarTest, AnalysisHighpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().analysis_kernels().highpass(),
        MatrixEq(cv::Mat(param.analysis_highpass))
    );
}

TEST_P(HaarTest, SynthesisLowpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().synthesis_kernels().lowpass(),
        MatrixEq(cv::Mat(param.synthesis_lowpass))
    );
}

TEST_P(HaarTest, SynthesisHighpassCoeffs)
{
    auto param = GetParam();
    EXPECT_THAT(
        wavelet.filter_bank().synthesis_kernels().highpass(),
        MatrixEq(cv::Mat(param.synthesis_highpass))
    );
}

INSTANTIATE_TEST_CASE_P(
    WaveletGroup,
    HaarTest,
    testing::Values(
        WaveletTestParam{
            .order = 1,
            .vanishing_moments_psi = 2,
            .support_width = 1,
            .short_name = "haar",
            .analysis_lowpass = {
                7.071067811865475244008443621048490392848359376884740365883398e-01,
                7.071067811865475244008443621048490392848359376884740365883398e-01,
            },
            .analysis_highpass = {
                -7.071067811865475244008443621048490392848359376884740365883398e-01,
                7.071067811865475244008443621048490392848359376884740365883398e-01,
            },
            .synthesis_lowpass = {
                7.071067811865475244008443621048490392848359376884740365883398e-01,
                7.071067811865475244008443621048490392848359376884740365883398e-01,
            },
            .synthesis_highpass = {
                7.071067811865475244008443621048490392848359376884740365883398e-01,
                -7.071067811865475244008443621048490392848359376884740365883398e-01,
            },
        }
    )
);

