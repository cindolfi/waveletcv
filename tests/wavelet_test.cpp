/**
 * Wavelet & DWT2D Unit Tests
*/
#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <wavelet/wavelet.hpp>
#include <wavelet/dwt2d.hpp>
#include <numeric>
#include <vector>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>

using namespace wavelet;
using namespace testing;

cv::Mat create_matrix(int rows, int cols, int type, float initial_value = 0)
{
    std::vector<float> elements(rows * cols);
    std::iota(elements.begin(), elements.end(), initial_value);
    auto result = cv::Mat(elements, true).reshape(0, rows);
    result.convertTo(result, type);

    return result;
}

void print_matrix(const cv::Mat& matrix, float zero_clamp=1e-7) {
    if (matrix.type() == CV_32F || matrix.type() == CV_64F) {
        // matrix
        cv::Mat clamped;
        matrix.copyTo(clamped);
        clamped.setTo(0, cv::abs(clamped) < zero_clamp);

        for (int row = 0; row < clamped.rows; ++row) {
            for (int col = 0; col < clamped.cols; ++col) {
                std::cout << std::setfill(' ') << std::setw(3) << std::setprecision(3) << clamped.at<float>(row, col) << " ";
            }
            std::cout << std::endl;
        }
    } else {
        for (int row = 0; row < matrix.rows; ++row) {
            for (int col = 0; col < matrix.cols; ++col) {
                std::cout << std::setfill(' ') << std::setw(3) << matrix.at<float>(row, col) << " ";
            }
            std::cout << std::endl;
        }
    }
}

bool matrix_equals(const cv::Mat& a, const cv::Mat& b) {
    return a.size() == b.size() && cv::countNonZero(a == b) == a.total();
}

bool matrix_equals(const cv::Mat& a, const cv::Scalar& b) {
    return cv::countNonZero(a == b) == a.total();
}

bool matrix_is_all_zeros(const cv::Mat& a) {
    return cv::countNonZero(a == 0.0) == a.total();
}

bool matrix_equals_float(const cv::Mat& a, const cv::Mat& b, float tolerance=0.0) {
    if (a.size() != b.size())
        return false;

    if (tolerance <= 0)
        tolerance = std::numeric_limits<float>::epsilon();

    cv::Mat diff;
    cv::absdiff(a, b, diff);
    cv::Mat mask = (diff <= tolerance);
    return cv::countNonZero(diff <= tolerance) == diff.total();
}


MATCHER_P(MatrixEq, other, "") { return matrix_equals(arg, other); }
MATCHER_P2(MatrixFloatEq, other, error, "") { return matrix_equals_float(arg, other, error); }


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




/**
 * -----------------------------------------------------------------------------
 * DWT2D::Coeffs
 * -----------------------------------------------------------------------------
*/
class Dwt2dCoeffsDefaultConstructorTest : public testing::Test
{
protected:
    DWT2D::Coeffs coeffs;
};

TEST_F(Dwt2dCoeffsDefaultConstructorTest, IsEmpty)
{
    ASSERT_TRUE(coeffs.empty());
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DepthIsZero)
{
    ASSERT_EQ(coeffs.depth(), 0);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, ApproxIsEmpty)
{
    ASSERT_TRUE(coeffs.approx().empty());
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, HorizontalDetailIsEmpty)
{
    ASSERT_TRUE(coeffs.horizontal_detail().empty());
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, VerticalDetailIsEmpty)
{
    ASSERT_TRUE(coeffs.vertical_detail().empty());
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DiagaonlDetailIsEmpty)
{
    ASSERT_TRUE(coeffs.diagonal_detail().empty());
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, CollectHorizontalDetailsIsEmpty)
{
    ASSERT_TRUE(coeffs.collect_horizontal_details().empty());
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, CollectVerticalDetailsIsEmpty)
{
    ASSERT_TRUE(coeffs.collect_vertical_details().empty());
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, CollectDiagonalDetailsIsEmpty)
{
    ASSERT_TRUE(coeffs.collect_diagonal_details().empty());
}

/**
 * -----------------------------------------------------------------------------
*/
class Dwt2dCoeffsTest : public testing::Test
{
protected:
    void SetUp() override
    {
        expected_matrix = create_matrix(rows, cols, type);
        coeffs = DWT2D::Coeffs(expected_matrix);
    }

    const int rows = 32;
    const int cols = 16;
    const int type = CV_32F;
    const int expected_depth = 4;
    const cv::Size size = cv::Size(cols, rows);
    cv::Mat expected_matrix;
    DWT2D::Coeffs coeffs;
};

TEST_F(Dwt2dCoeffsTest, SizeIsCorrect)
{
    ASSERT_EQ(coeffs.size(), size);
    ASSERT_EQ(coeffs.rows(), rows);
    ASSERT_EQ(coeffs.cols(), cols);
}

TEST_F(Dwt2dCoeffsTest, TypeIsCorrect)
{
    ASSERT_EQ(coeffs.type(), type);
}

TEST_F(Dwt2dCoeffsTest, DepthIsCorrect)
{
    ASSERT_EQ(coeffs.depth(), expected_depth);
}

TEST_F(Dwt2dCoeffsTest, CastToMatrix)
{
    cv::Mat matrix = coeffs;
    EXPECT_THAT(
        matrix,
        MatrixEq(expected_matrix)
    ) << "matrix is not equal to coeffs";

    EXPECT_TRUE(
        coeffs.shares_data(matrix)
    ) << "matrix does not share data with coeffs";
}

TEST_F(Dwt2dCoeffsTest, ClonedSizeIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    ASSERT_EQ(cloned_coeffs.size(), coeffs.size());
}

TEST_F(Dwt2dCoeffsTest, ClonedTypeIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    ASSERT_EQ(cloned_coeffs.type(), coeffs.type());
}

TEST_F(Dwt2dCoeffsTest, ClonedDepthIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    ASSERT_EQ(cloned_coeffs.depth(), coeffs.depth());
}

TEST_F(Dwt2dCoeffsTest, CloneCopiesData)
{
    auto cloned_coeffs = coeffs.clone();

    EXPECT_THAT(
        cloned_coeffs,
        MatrixEq(coeffs)
    ) << "cloned coeffs does not equal original";

    EXPECT_FALSE(
        coeffs.shares_data(cloned_coeffs)
    ) << "cloned coeffs shares data with original";
}

TEST_F(Dwt2dCoeffsTest, AssignmentFromMatrix)
{
    DWT2D::Coeffs new_coeffs(expected_matrix.size(), expected_matrix.type());
    new_coeffs = expected_matrix;

    EXPECT_THAT(new_coeffs, MatrixEq(expected_matrix));
    EXPECT_FALSE(new_coeffs.shares_data(expected_matrix));
}

TEST_F(Dwt2dCoeffsTest, AssignmentFromMatrixExpr)
{
    DWT2D::Coeffs new_coeffs(expected_matrix.size(), expected_matrix.type());
    new_coeffs = 1 + expected_matrix;

    EXPECT_THAT(new_coeffs, MatrixEq(1 + expected_matrix));
    EXPECT_FALSE(new_coeffs.shares_data(expected_matrix));
}

TEST_F(Dwt2dCoeffsTest, AssignmentFromScalar)
{
    DWT2D::Coeffs new_coeffs(size, type);
    new_coeffs = 1.0;

    EXPECT_THAT(new_coeffs, MatrixEq(cv::Mat::ones(size, type)));
}

#ifndef DISABLE_ARG_CHECKS
TEST_F(Dwt2dCoeffsTest, AssignmentFromWrongSizeMatrixIsError)
{
    EXPECT_THROW(
        {
            DWT2D::Coeffs new_coeffs(
                cv::Size(1, 1) + expected_matrix.size(),
                expected_matrix.type()
            );
            new_coeffs = expected_matrix;
        },
        cv::Exception
    );
}
#endif

TEST_F(Dwt2dCoeffsTest, CollectedHorizontalDetailsSizeEqualsDepth)
{
    EXPECT_EQ(coeffs.collect_horizontal_details().size(), expected_depth);
}

TEST_F(Dwt2dCoeffsTest, CollectedVerticalDetailsSizeEqualsDepth)
{
    EXPECT_EQ(coeffs.collect_vertical_details().size(), expected_depth);
}

TEST_F(Dwt2dCoeffsTest, CollectedDiagonalDetailsSizeEqualsDepth)
{
    EXPECT_EQ(coeffs.collect_diagonal_details().size(), expected_depth);
}

TEST_F(Dwt2dCoeffsTest, LevelIteratorConsistentWithAt)
{
    int level = 0;
    for (auto level_coeffs : coeffs) {
        EXPECT_THAT(
            level_coeffs,
            MatrixEq(coeffs.at(level))
        ) << "iterator and at() are inconsistent at level " << level;
        ++level;
    }

    EXPECT_EQ(
        level,
        coeffs.depth()
    ) << "iterator and depth() are inconsistent";
}

/**
 * -----------------------------------------------------------------------------
*/
class Dwt2dCoeffsLevelsTest : public testing::TestWithParam<int>
{
protected:
    void SetUp() override
    {
        expected_matrix = create_matrix(rows, cols, type);
        coeffs = DWT2D::Coeffs(expected_matrix);

        int level = GetParam();
        int level_size_factor = std::pow(2, level);
        int detail_size_factor = 2 * level_size_factor;

        expected_depth = full_depth - level;
        expected_size = full_size / level_size_factor;
        expected_detail_size = expected_size / 2;

        expected_horizontal_detail_rect = cv::Rect(
            horizontal_offset / detail_size_factor,
            expected_detail_size
        );
        expected_horizontal_detail = expected_matrix(expected_horizontal_detail_rect);

        expected_vertical_detail_rect = cv::Rect(
            vertical_offset / detail_size_factor,
            expected_detail_size
        );
        expected_vertical_detail = expected_matrix(expected_vertical_detail_rect);

        expected_diagonal_detail_rect = cv::Rect(
            diagonal_offset / detail_size_factor,
            expected_detail_size
        );
        expected_diagonal_detail = expected_matrix(expected_diagonal_detail_rect);

        expected_level_coeffs = DWT2D::Coeffs(
            expected_matrix(cv::Rect(cv::Point(0, 0), expected_size)),
            expected_depth
        );

        directions_and_expected_rects = {
            std::make_tuple(HORIZONTAL, "horizontal", expected_horizontal_detail_rect),
            std::make_tuple(VERTICAL, "vertical", expected_vertical_detail_rect),
            std::make_tuple(DIAGONAL, "diagonal", expected_diagonal_detail_rect),
        };
    }

    void assert_level_details_collected_correctly(
        const DWT2D::Coeffs& coeffs,
        const cv::Mat& actual_detail,
        const cv::Mat& expected_detail
    )
    {
        EXPECT_TRUE(
            coeffs.shares_data(actual_detail)
        ) << "collected detail was copied from coeffs";

        EXPECT_THAT(
            actual_detail,
            MatrixEq(expected_detail)
        ) << "detail values are incorrect";
    }

    void assert_set_detail(
        const DWT2D::Coeffs& full_coeffs,
        const cv::Mat& expected_detail,
        const cv::Mat& expected_modified_full_matrix,
        const cv::Mat& detail_from_full_coeffs_before_assign,
        const cv::Mat& detail_from_full_coeffs_after_assign,
        const cv::Mat& detail_from_level_coeffs_before_assign,
        const cv::Mat& detail_from_level_coeffs_after_assign,
        const std::string& direction_name
    )
    {
        EXPECT_THAT(
            detail_from_full_coeffs_before_assign,
            MatrixEq(expected_detail)
        ) << direction_name << " detail created from coeffs before set_details() is wrong";

        EXPECT_THAT(
            detail_from_full_coeffs_after_assign,
            MatrixEq(expected_detail)
        ) << direction_name << " detail created from coeffs after set_details() is wrong";

        EXPECT_THAT(
            detail_from_level_coeffs_before_assign,
            MatrixEq(expected_detail)
        ) << direction_name << " detail from level view created after set_details() is wrong";

        EXPECT_THAT(
            detail_from_level_coeffs_after_assign,
            MatrixEq(expected_detail)
        ) << direction_name << " detail from level view created before set_details() is wrong";

        EXPECT_THAT(
            full_coeffs,
            MatrixEq(expected_modified_full_matrix)
        ) << "full coeffs are wrong after setting " << direction_name << " details";

        EXPECT_TRUE(
            full_coeffs.shares_data(detail_from_full_coeffs_before_assign)
        ) << direction_name << " detail from full coeffs created before set_details() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(detail_from_full_coeffs_after_assign)
        ) << direction_name << " detail from full coeffs created after set_details() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(detail_from_level_coeffs_before_assign)
        ) << direction_name << " detail from level view created before set_details() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(detail_from_level_coeffs_after_assign)
        ) << direction_name << " detail from level view created after set_details() is a copy";
    }

    auto collect_and_clone_details(const DWT2D::Coeffs& coeffs)
    {
        auto clone_detail = [](const auto& detail) { return detail.clone(); };
        std::vector<cv::Mat> horizontal_details;
        std::ranges::transform(
            coeffs.collect_horizontal_details(),
            std::back_inserter(horizontal_details),
            clone_detail
        );
        std::vector<cv::Mat> vertical_details;
        std::ranges::transform(
            coeffs.collect_vertical_details(),
            std::back_inserter(vertical_details),
            clone_detail
        );
        std::vector<cv::Mat> diagonal_details;
        std::ranges::transform(
            coeffs.collect_diagonal_details(),
            std::back_inserter(diagonal_details),
            clone_detail
        );

        return std::make_tuple(horizontal_details, vertical_details, diagonal_details);
    }

    void assert_details_at_lower_levels_not_modified(
        const DWT2D::Coeffs& coeffs,
        const std::vector<cv::Mat>& expected_horizontal_details,
        const std::vector<cv::Mat>& expected_vertical_details,
        const std::vector<cv::Mat>& expected_diagonal_details
    )
    {
        int level = GetParam();
        for (int j = level - 1; j >= 1; --j) {
            EXPECT_THAT(
                coeffs.horizontal_detail(j),
                MatrixEq(expected_horizontal_details.at(j))
            ) << "setting level " << level << " modified horizontal detail at level " << j;

            EXPECT_THAT(
                coeffs.vertical_detail(j),
                MatrixEq(expected_vertical_details.at(j))
            ) << "setting level " << level << " modified vertical detail at level " << j;

            EXPECT_THAT(
                coeffs.diagonal_detail(j),
                MatrixEq(expected_diagonal_details.at(j))
            ) << "setting level " << level << " modified diagonal detail at level " << j;
        }
    }

public:
    static const int full_depth = 4;

protected:
    const int rows = 32;
    const int cols = 16;
    const int type = CV_32F;
    const cv::Size full_size = cv::Size(cols, rows);
    const cv::Point horizontal_offset = cv::Point(0, full_size.height);
    const cv::Point vertical_offset = cv::Point(full_size.width, 0);
    const cv::Point diagonal_offset = cv::Point(full_size.width, full_size.height);

    cv::Mat expected_matrix;
    DWT2D::Coeffs coeffs;
    cv::Size expected_size;
    int expected_depth;
    cv::Size expected_detail_size;
    DWT2D::Coeffs expected_level_coeffs;
    cv::Mat expected_horizontal_detail;
    cv::Mat expected_vertical_detail;
    cv::Mat expected_diagonal_detail;
    cv::Rect expected_horizontal_detail_rect;
    cv::Rect expected_vertical_detail_rect;
    cv::Rect expected_diagonal_detail_rect;
    std::vector<std::tuple<int, std::string, cv::Rect>> directions_and_expected_rects;
};

TEST_P(Dwt2dCoeffsLevelsTest, SizeIsCorrect)
{
    int level = GetParam();
    auto level_coeffs = coeffs.at(level);

    EXPECT_EQ(level_coeffs.size(), expected_size);
}

TEST_P(Dwt2dCoeffsLevelsTest, TypeIsCorrect)
{
    int level = GetParam();
    auto level_coeffs = coeffs.at(level);

    EXPECT_EQ(level_coeffs.type(), expected_level_coeffs.type());
}

TEST_P(Dwt2dCoeffsLevelsTest, DepthIsCorrect)
{
    int level = GetParam();
    auto level_coeffs = coeffs.at(level);

    EXPECT_EQ(level_coeffs.depth(), expected_depth);
}

TEST_P(Dwt2dCoeffsLevelsTest, ValuesAreCorrect)
{
    int level = GetParam();
    auto level_coeffs = coeffs.at(level);

    EXPECT_THAT(level_coeffs, MatrixEq(expected_level_coeffs));
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAccessDoesNotCauseCopy)
{
    int level = GetParam();
    auto level_coeffs = coeffs.at(level);

    EXPECT_TRUE(
        level_coeffs.shares_data(coeffs)
    ) << "coeffs.at(" << level << ") copied underlying matrix data";
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAccessIsTransitive)
{
    int i = GetParam();
    for (int j = 0; j < expected_depth - i; ++j) {
        auto level_coeffs1 = coeffs.at(i).at(j);
        auto level_coeffs2 = coeffs.at(i + j);

        EXPECT_THAT(
            level_coeffs1,
            MatrixEq(level_coeffs2)
        ) << "coeffs.at(" << i << ").at(" << j << ") != " << "coeffs.at(" << i + j << ")";

        EXPECT_TRUE(
            level_coeffs1.shares_data(level_coeffs2)
        ) << "coeffs.at(" << i << ").at(" << j << ") does not share data with " << "coeffs.at(" << i + j << ")";
    }
}

TEST_P(Dwt2dCoeffsLevelsTest, SetLevelToMatrix)
{
    int level = GetParam();
    DWT2D::Coeffs level_coeffs = coeffs.at(level);
    cv::Mat expected_level_matrix(expected_size, type, level);

    //  Set level coeffs
    coeffs.set_level(expected_level_matrix, level);

    EXPECT_THAT(level_coeffs, MatrixEq(expected_level_matrix));
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAssignmentFromMatrix)
{
    int level = GetParam();
    DWT2D::Coeffs level_coeffs = coeffs.at(level);
    cv::Mat expected_level_matrix(expected_size, type, level);

    //  Assign level coeffs
    level_coeffs = expected_level_matrix;

    EXPECT_THAT(level_coeffs, MatrixEq(expected_level_matrix));
}

#ifndef DISABLE_ARG_CHECKS
TEST_P(Dwt2dCoeffsLevelsTest, SetLevelToWrongSizeMatrixIsError)
{
    EXPECT_THROW(
        {
            int level = GetParam();
            coeffs.set_level(
                cv::Mat(cv::Size(1, 1) + expected_size, type, level),
                level
            );
        },
        cv::Exception
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAssignmentToWrongSizeMatrixIsError)
{
    EXPECT_THROW(
        {
            int level = GetParam();
            DWT2D::Coeffs level_coeffs = coeffs.at(level);
            level_coeffs = cv::Mat(cv::Size(1, 1) + expected_size, type, level);
        },
        cv::Exception
    );
}
#endif

TEST_P(Dwt2dCoeffsLevelsTest, SetLevelWritesIntoOriginalCoeffs)
{
    int level = GetParam();
    DWT2D::Coeffs level_coeffs = coeffs.at(level);

    cv::Mat expected_level_matrix(
        level_coeffs.rows(),
        level_coeffs.cols(),
        level_coeffs.type(),
        level
    );

    //  Set level coeffs
    coeffs.set_level(expected_level_matrix, level);

    EXPECT_TRUE(
        level_coeffs.shares_data(coeffs)
    ) << "assignment to level caused copy of original coeffs";

    EXPECT_THAT(
        coeffs.at(level),
        MatrixEq(expected_level_matrix)
    ) << "assignment to level not reflected in original coeffs";
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAssignmentWritesIntoOriginalCoeffs)
{
    int level = GetParam();
    DWT2D::Coeffs level_coeffs = coeffs.at(level);

    cv::Mat expected_level_matrix(
        level_coeffs.rows(),
        level_coeffs.cols(),
        level_coeffs.type(),
        level
    );

    //  Assign level coeffs
    level_coeffs = expected_level_matrix;

    EXPECT_EQ(
        level_coeffs.depth(),
        expected_depth
    ) << "assignment to level has wrong depth";

    EXPECT_TRUE(
        level_coeffs.shares_data(coeffs)
    ) << "assignment to level caused copy of original coeffs";

    EXPECT_THAT(
        coeffs.at(level),
        MatrixEq(expected_level_matrix)
    ) << "assignment to level not reflected in original coeffs";
}

TEST_P(Dwt2dCoeffsLevelsTest, SetLevelDoesNotModifyDetailsAtLowerLevels)
{
    int level = GetParam();
    DWT2D::Coeffs level_coeffs = coeffs.at(level);

    auto [expected_horizontal_details, expected_vertical_details, expected_diagonal_details] = collect_and_clone_details(coeffs);

    //  Set level coeffs
    coeffs.set_level(
        cv::Mat(level_coeffs.size(), level_coeffs.type(), level),
        level
    );

    assert_details_at_lower_levels_not_modified(
        coeffs,
        expected_horizontal_details,
        expected_vertical_details,
        expected_diagonal_details
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAssignmentDoesNotModifyDetailsAtLowerLevels)
{
    int level = GetParam();
    DWT2D::Coeffs level_coeffs = coeffs.at(level);

    auto [expected_horizontal_details, expected_vertical_details, expected_diagonal_details] = collect_and_clone_details(coeffs);

    //  Assign to level coeffs
    level_coeffs = cv::Mat(level_coeffs.size(), level_coeffs.type(), level);

    assert_details_at_lower_levels_not_modified(
        coeffs,
        expected_horizontal_details,
        expected_vertical_details,
        expected_diagonal_details
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, HorizontalDetailRect)
{
    int level = GetParam();

    EXPECT_EQ(
        coeffs.horizontal_rect(level),
        expected_horizontal_detail_rect
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, VerticalDetailRect)
{
    int level = GetParam();
    EXPECT_EQ(
        coeffs.vertical_rect(level),
        expected_vertical_detail_rect
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, DiagonalDetailRect)
{
    int level = GetParam();
    EXPECT_EQ(
        coeffs.diagonal_rect(level),
        expected_diagonal_detail_rect
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, HorizontalDetailValues)
{
    int level = GetParam();
    EXPECT_THAT(
        coeffs.horizontal_detail(level),
        MatrixEq(expected_horizontal_detail)
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, VerticalDetailValues)
{
    int level = GetParam();
    EXPECT_THAT(
        coeffs.vertical_detail(level),
        MatrixEq(expected_vertical_detail)
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, DiagonalDetailValues)
{
    int level = GetParam();
    EXPECT_THAT(
        coeffs.diagonal_detail(level),
        MatrixEq(expected_diagonal_detail)
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, HorizontalDetailSharesDataWithCoeffs)
{
    int level = GetParam();

    EXPECT_TRUE(
        coeffs.shares_data(coeffs.horizontal_detail(level))
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, VerticalDetailSharesDataWithCoeffs)
{
    int level = GetParam();
    EXPECT_TRUE(
        coeffs.shares_data(coeffs.vertical_detail(level))
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, DiagonalDetailSharesDataWithCoeffs)
{
    int level = GetParam();
    EXPECT_TRUE(
        coeffs.shares_data(coeffs.diagonal_detail(level))
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, SetDetailsToMatrix)
{
    int level = GetParam();
    for (auto [direction, name, rect] : directions_and_expected_rects) {
        auto expected_modified_full_matrix = expected_matrix.clone();

        DWT2D::Coeffs full_coeffs = expected_modified_full_matrix;
        DWT2D::Coeffs level_coeffs = full_coeffs.at(level);

        auto detail_from_full_coeffs_before_assign = full_coeffs.detail(direction, level);
        auto detail_from_level_coeffs_before_assign = level_coeffs.detail(direction);

        auto new_detail_values = cv::Mat(rect.size(), type, level);

        //  update expected
        expected_modified_full_matrix = expected_modified_full_matrix.clone();
        new_detail_values.copyTo(expected_modified_full_matrix(rect));

        //  fill details with value = level
        full_coeffs.set_detail(new_detail_values, direction, level);

        auto detail_from_full_coeffs_after_assign = full_coeffs.detail(direction, level);
        auto detail_from_level_coeffs_after_assign = level_coeffs.detail(direction);

        assert_set_detail(
            full_coeffs,
            new_detail_values,
            expected_modified_full_matrix,
            detail_from_full_coeffs_before_assign,
            detail_from_full_coeffs_after_assign,
            detail_from_level_coeffs_before_assign,
            detail_from_level_coeffs_after_assign,
            name
        );
    }
}

TEST_P(Dwt2dCoeffsLevelsTest, SetDetailsToScalar)
{
    int level = GetParam();
    for (auto [direction, name, rect] : directions_and_expected_rects) {
        auto expected_modified_full_matrix = expected_matrix.clone();

        DWT2D::Coeffs full_coeffs = expected_modified_full_matrix;
        DWT2D::Coeffs level_coeffs = full_coeffs.at(level);

        auto detail_from_full_coeffs_before_assign = full_coeffs.detail(direction, level);
        auto detail_from_level_coeffs_before_assign = level_coeffs.detail(direction);

        auto new_detail_values = cv::Mat(rect.size(), type, level);

        //  update expected
        expected_modified_full_matrix = expected_modified_full_matrix.clone();
        expected_modified_full_matrix(rect) = level;

        //  fill details with value = level
        full_coeffs.set_detail(level, direction, level);

        auto detail_from_full_coeffs_after_assign = full_coeffs.detail(direction, level);
        auto detail_from_level_coeffs_after_assign = level_coeffs.detail(direction);

        assert_set_detail(
            full_coeffs,
            new_detail_values,
            expected_modified_full_matrix,
            detail_from_full_coeffs_before_assign,
            detail_from_full_coeffs_after_assign,
            detail_from_level_coeffs_before_assign,
            detail_from_level_coeffs_after_assign,
            name
        );
    }
}

TEST_P(Dwt2dCoeffsLevelsTest, AssignScalarToDetails)
{
    int level = GetParam();
    for (auto [direction, name, rect] : directions_and_expected_rects) {
        auto expected_modified_full_matrix = expected_matrix.clone();

        DWT2D::Coeffs full_coeffs = expected_modified_full_matrix;
        DWT2D::Coeffs level_coeffs = full_coeffs.at(level);

        auto detail_from_full_coeffs_before_assign = full_coeffs.detail(direction, level);
        auto detail_from_level_coeffs_before_assign = level_coeffs.detail(direction);

        auto new_detail_values = cv::Mat(rect.size(), type, level);

        //  update expected
        expected_modified_full_matrix = expected_modified_full_matrix.clone();
        expected_modified_full_matrix(rect) = level;

        //  fill details with value = level
        full_coeffs.detail(direction, level) = level;

        auto detail_from_full_coeffs_after_assign = full_coeffs.detail(direction, level);
        auto detail_from_level_coeffs_after_assign = level_coeffs.detail(direction);

        assert_set_detail(
            full_coeffs,
            new_detail_values,
            expected_modified_full_matrix,
            detail_from_full_coeffs_before_assign,
            detail_from_full_coeffs_after_assign,
            detail_from_level_coeffs_before_assign,
            detail_from_level_coeffs_after_assign,
            name
        );
    }
}

#ifndef DISABLE_ARG_CHECKS
TEST_P(Dwt2dCoeffsLevelsTest, SetDetailsToWrongSizeMatrixIsError)
{
    int level = GetParam();
    for (auto [direction, name, rect] : directions_and_expected_rects) {
        EXPECT_THROW(
            {
                DWT2D::Coeffs full_coeffs = expected_matrix.clone();
                auto new_detail_values = cv::Mat(cv::Size(1, 1) + rect.size(), type, level);
                full_coeffs.set_detail(new_detail_values, direction, level);
            },
            cv::Exception
        ) << "did not throw exception when setting " << name << " details to an ill-sized matrix";
    }
}
#endif

TEST_P(Dwt2dCoeffsLevelsTest, CollectHorizontalDetails)
{
    int level = GetParam();
    assert_level_details_collected_correctly(
        coeffs,
        coeffs.collect_horizontal_details().at(level),
        expected_horizontal_detail
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, CollectVerticalDetails)
{
    int level = GetParam();
    assert_level_details_collected_correctly(
        coeffs,
        coeffs.collect_vertical_details().at(level),
        expected_vertical_detail
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, CollectDiagonalDetails)
{
    int level = GetParam();
    assert_level_details_collected_correctly(
        coeffs,
        coeffs.collect_diagonal_details().at(level),
        expected_diagonal_detail
    );
}


INSTANTIATE_TEST_CASE_P(
    Dwt2dCoeffsGroup,
    Dwt2dCoeffsLevelsTest,
    testing::Range(0, Dwt2dCoeffsLevelsTest::full_depth),
    testing::PrintToStringParamName()
);




/**
 * -----------------------------------------------------------------------------
 * Transformation Tests
 * -----------------------------------------------------------------------------
*/
using PatternPair = std::tuple<cv::Mat, cv::Mat>;

cv::Mat make_pattern(const std::vector<float>& pattern)
{
    int rows = std::sqrt(pattern.size());
    return cv::Mat(pattern, true).reshape(0, rows);
}

void print_test_patterns(const cv::Mat& actual_output, const cv::Mat& expected_output)
{
    std::cout << std::endl << "actual =" << std::endl;
    print_matrix(actual_output, 1e-5);
    std::cout << std::endl << "expected =" << std::endl;
    print_matrix(expected_output, 1e-5);
    std::cout << std::endl;
}

template <typename T1, typename T2>
std::vector<std::tuple<T1, T2>> zip(const std::vector<T1>& inputs, const std::vector<T2>& outputs)
{
    int size = std::min(inputs.size(), outputs.size());
    std::vector<std::tuple<T1, T2>> result(size);
    for (int i = 0; i < size; ++i)
        result[i] = std::make_tuple(inputs[i], outputs[i]);

    return result;
}

template <typename T1, typename T2>
std::vector<std::tuple<T1, T2>> zip(const T1& input, const T2& output)
{
    return std::vector<std::tuple<T1, T2>>({
        std::make_tuple(input, output),
    });
}


std::vector<std::string> pattern_names = {
    "ZerosPattern",
    "OnesPattern",
    "HorizontalLinesPattern",
    "InvertedHorizontalLinesPattern",
    "VerticalLinesPattern",
    "InvertedVerticalLinesPattern",
    "DiagonalLinesPattern",
    "InvertedDiagonalLinesPattern",
};

std::vector<cv::Mat> basic_test_patterns = {
    //  zeros
    make_pattern({
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  ones
    make_pattern({
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    }),
    //  horizontal lines
    make_pattern({
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  inverted horizontal lines
    make_pattern({
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    }),
    //  vertical lines
    make_pattern({
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    }),
    //  inverted vertical lines
    make_pattern({
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    }),
    //  checker board
    make_pattern({
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    }),
    //  inverted checker board
    make_pattern({
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    }),
};


/**
 * -----------------------------------------------------------------------------
 * WaveletFilterBank
 * -----------------------------------------------------------------------------
*/
class HaarPatternSingleLevelTests : public testing::TestWithParam<PatternPair>
{
};

TEST_P(HaarPatternSingleLevelTests, ForwardTransform) {
    auto wavelet = haar();
    auto [input_pattern, expected_output] = GetParam();

    cv::Size size = expected_output.size() / 2;
    int width = size.width;
    int height = size.height;

    auto expected_approx = expected_output(
        cv::Rect(0, 0, width, height)
    );
    auto expected_horizontal_detail = expected_output(
        cv::Rect(0, height, width, height)
    );
    auto expected_vertical_detail = expected_output(
        cv::Rect(width, 0, width, height)
    );
    auto expected_diagonal_detail = expected_output(
        cv::Rect(width, height, width, height)
    );

    cv::Mat approx;
    cv::Mat horizontal_detail;
    cv::Mat vertical_detail;
    cv::Mat diagonal_detail;
    wavelet.filter_bank().forward(
        input_pattern,
        approx,
        horizontal_detail,
        vertical_detail,
        diagonal_detail,
        cv::BORDER_REFLECT101
    );

    EXPECT_THAT(
        approx,
        MatrixFloatEq(expected_approx, 1e-5)
    ) << "approx is incorrect";
    EXPECT_THAT(
        horizontal_detail,
        MatrixFloatEq(expected_horizontal_detail, 1e-5)
    ) << "horizontal_detail is incorrect";
    EXPECT_THAT(
        vertical_detail,
        MatrixFloatEq(expected_vertical_detail, 1e-5)
    ) << "vertical_detail is incorrect";
    EXPECT_THAT(
        diagonal_detail,
        MatrixFloatEq(expected_diagonal_detail, 1e-5)
    ) << "diagonal_detail is incorrect";
}

TEST_P(HaarPatternSingleLevelTests, InverseTransform) {
    auto wavelet = haar();
    auto [expected_output, input_pattern] = GetParam();

    cv::Size size = input_pattern.size() / 2;
    int width = size.width;
    int height = size.height;

    auto approx = input_pattern(
        cv::Rect(0, 0, width, height)
    );
    auto horizontal_detail = input_pattern(
        cv::Rect(0, height, width, height)
    );
    auto vertical_detail = input_pattern(
        cv::Rect(width, 0, width, height)
    );
    auto diagonal_detail = input_pattern(
        cv::Rect(width, height, width, height)
    );

    cv::Mat actual_output;
    wavelet.filter_bank().inverse(
        approx,
        horizontal_detail,
        vertical_detail,
        diagonal_detail,
        actual_output,
        cv::BORDER_REFLECT101
    );

    EXPECT_THAT(actual_output, MatrixFloatEq(expected_output, 1e-5));
}


std::vector<cv::Mat> basic_test_patterns_haar_filter_bank_outputs = {
    //  zeros
    make_pattern({
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  ones
    make_pattern({
        2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  horizontal lines
    make_pattern({
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  inverted horizontal lines
    make_pattern({
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
    }),
    //  vertical lines
    make_pattern({
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  inverted vertical lines
    make_pattern({
         1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    }),
    //  checker board
    make_pattern({
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    }),
    //  inverted checker board
    make_pattern({
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
    }),
};

INSTANTIATE_TEST_CASE_P(
    HaarGroup,
    HaarPatternSingleLevelTests,
    testing::ValuesIn(
        zip(basic_test_patterns, basic_test_patterns_haar_filter_bank_outputs)
    ),
    [](const auto& info) { return pattern_names[info.index]; }
);


/**
 * -----------------------------------------------------------------------------
 * DWT2D
 * -----------------------------------------------------------------------------
*/
class HaarPatternDwtTests : public testing::TestWithParam<PatternPair>
{
};

TEST_P(HaarPatternDwtTests, ForwardTransform)
{
    DWT2D dwt(haar(), cv::BORDER_REFLECT101);

    auto [input_pattern, expected_output] = GetParam();
    auto actual_output = dwt(input_pattern);

    EXPECT_THAT(actual_output, MatrixFloatEq(expected_output, 1e-5));
}

TEST_P(HaarPatternDwtTests, InverseTransform)
{
    DWT2D dwt(haar(), cv::BORDER_REFLECT101);

    auto [expected_output, input_pattern] = GetParam();
    auto actual_output = dwt.inverse(input_pattern);

    EXPECT_THAT(actual_output, MatrixFloatEq(expected_output, 1e-6));
}


std::vector<cv::Mat> basic_test_patterns_haar_dwt_outputs = {
    //  zeros
    make_pattern({
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  ones
    make_pattern({
        16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  horizontal lines
    make_pattern({
        8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  inverted horizontal lines
    make_pattern({
         8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
    }),
    //  vertical lines
    make_pattern({
        8, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    }),
    //  inverted vertical lines
    make_pattern({
        8.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1., -1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1., -1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1., -1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1., -1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1., -1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1., -1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1., -1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1., -1., -1.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    }),
    //  checker board
    make_pattern({
        8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    }),
    //  inverted checker board
    make_pattern({
        8, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
    }),
};

INSTANTIATE_TEST_CASE_P(
    HaarGroup,
    HaarPatternDwtTests,
    testing::ValuesIn(
        zip(basic_test_patterns, basic_test_patterns_haar_dwt_outputs)
    ),
    [](const auto& info) { return pattern_names[info.index]; }
);

