/**
 * Wavelet & DWT2D Unit Tests
*/
#include <vector>
#include <map>
#include <fstream>
#include <ranges>
#include <nlohmann/json.hpp>
#include <wavelet/dwt2d.hpp>
#include <wavelet/utils.hpp>
#include "common.hpp"

using namespace wavelet;
using namespace testing;
using json = nlohmann::json;


/**
 * =============================================================================
 * DWT2D::Coeffs
 * =============================================================================
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

TEST_F(Dwt2dCoeffsDefaultConstructorTest, LevelsIsZero)
{
    ASSERT_EQ(coeffs.levels(), 0);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, InputSizeIsEmpty)
{
    ASSERT_TRUE(coeffs.input_size().empty());
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, WaveletIsInvalid)
{
    ASSERT_FALSE(coeffs.wavelet().is_valid());
}

//  ----------------------------------------------------------------------------
//  getters
TEST_F(Dwt2dCoeffsDefaultConstructorTest, AtLevelIsError)
{
    EXPECT_THROW({ coeffs.at_level(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, ApproxIsError)
{
    EXPECT_THROW({ coeffs.approx(); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, HorizontalDetailIsError)
{
    EXPECT_THROW({ coeffs.horizontal_detail(); }, cv::Exception);
    EXPECT_THROW({ coeffs.horizontal_detail(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, VerticalDetailIsError)
{
    EXPECT_THROW({ coeffs.vertical_detail(); }, cv::Exception);
    EXPECT_THROW({ coeffs.vertical_detail(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DiagaonlDetailIsError)
{
    EXPECT_THROW({ coeffs.diagonal_detail(); }, cv::Exception);
    EXPECT_THROW({ coeffs.diagonal_detail(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DetailIsError)
{
    EXPECT_THROW({ coeffs.detail(HORIZONTAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail(VERTICAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail(DIAGONAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail(0, HORIZONTAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail(0, VERTICAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail(0, DIAGONAL); }, cv::Exception);
}

//  ----------------------------------------------------------------------------
//  setters
TEST_F(Dwt2dCoeffsDefaultConstructorTest, SetLevelIsError)
{
    EXPECT_THROW({ coeffs.set_level(0, cv::Mat()); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, SetApproxIsError)
{
    EXPECT_THROW({ coeffs.set_approx(cv::Mat()); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, SetHorizontalDetailIsError)
{
    EXPECT_THROW({ coeffs.set_horizontal_detail(cv::Mat()); }, cv::Exception);
    EXPECT_THROW({ coeffs.set_horizontal_detail(0, cv::Mat()); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, SetVerticalDetailIsError)
{
    EXPECT_THROW({ coeffs.set_vertical_detail(cv::Mat()); }, cv::Exception);
    EXPECT_THROW({ coeffs.set_vertical_detail(0, cv::Mat()); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, SetDiagaonlDetailIsError)
{
    EXPECT_THROW({ coeffs.set_diagonal_detail(cv::Mat()); }, cv::Exception);
    EXPECT_THROW({ coeffs.set_diagonal_detail(0, cv::Mat()); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, SetDetailIsError)
{
    EXPECT_THROW({ coeffs.set_detail(HORIZONTAL, cv::Mat()); }, cv::Exception);
    EXPECT_THROW({ coeffs.set_detail(VERTICAL, cv::Mat()); }, cv::Exception);
    EXPECT_THROW({ coeffs.set_detail(DIAGONAL, cv::Mat()); }, cv::Exception);
    EXPECT_THROW({ coeffs.set_detail(0, HORIZONTAL, cv::Mat()); }, cv::Exception);
    EXPECT_THROW({ coeffs.set_detail(0, VERTICAL, cv::Mat()); }, cv::Exception);
    EXPECT_THROW({ coeffs.set_detail(0, DIAGONAL, cv::Mat()); }, cv::Exception);
}

//  ----------------------------------------------------------------------------
//  sizes & rects
TEST_F(Dwt2dCoeffsDefaultConstructorTest, LevelSizeIsError)
{
    EXPECT_THROW({ coeffs.level_size(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, LevelRectIsError)
{
    EXPECT_THROW({ coeffs.level_rect(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, ApproxRectIsError)
{
    EXPECT_THROW({ coeffs.approx_rect(); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, HorizontalDetailRectIsError)
{
    EXPECT_THROW({ coeffs.horizontal_detail_rect(); }, cv::Exception);
    EXPECT_THROW({ coeffs.horizontal_detail_rect(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, VerticalDetailRectIsError)
{
    EXPECT_THROW({ coeffs.vertical_detail_rect(); }, cv::Exception);
    EXPECT_THROW({ coeffs.vertical_detail_rect(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DiagaonlDetailRectIsError)
{
    EXPECT_THROW({ coeffs.diagonal_detail_rect(); }, cv::Exception);
    EXPECT_THROW({ coeffs.diagonal_detail_rect(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DetailSizeIsError)
{
    EXPECT_THROW({ coeffs.detail_size(); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail_size(0); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DetailRectIsError)
{
    EXPECT_THROW({ coeffs.detail_rect(HORIZONTAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail_rect(VERTICAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail_rect(DIAGONAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail_rect(0, HORIZONTAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail_rect(0, VERTICAL); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail_rect(0, DIAGONAL); }, cv::Exception);
}

//  ----------------------------------------------------------------------------
//  mask
TEST_F(Dwt2dCoeffsDefaultConstructorTest, ApproxMaskIsError)
{
    EXPECT_THROW({ coeffs.approx_mask(); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, HorizontalDetailMaskIsError)
{
    EXPECT_THROW({ coeffs.horizontal_detail_mask(); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, VerticalDetailMaskIsError)
{
    EXPECT_THROW({ coeffs.vertical_detail_mask(); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DiagaonlDetailMaskIsError)
{
    EXPECT_THROW({ coeffs.diagonal_detail_mask(); }, cv::Exception);
}

TEST_F(Dwt2dCoeffsDefaultConstructorTest, DetailMaskIsError)
{
    EXPECT_THROW({ coeffs.detail_mask(); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail_mask(); }, cv::Exception);
    EXPECT_THROW({ coeffs.detail_mask(); }, cv::Exception);
}

//  ----------------------------------------------------------------------------
//  collect details
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
class Dwt2dCoeffsTest : public testing::TestWithParam<int>
{
protected:
    const int rows = 32;
    const int cols = 16;
    const int expected_levels = 4;
    const cv::Size size = cv::Size(cols, rows);

protected:
    Dwt2dCoeffsTest() :
        testing::TestWithParam<int>(),
        dwt(create_haar())
    {
    }

    void SetUp() override
    {
        type = GetParam();
        expected_matrix = create_matrix(rows, cols, type);
        coeffs = dwt.create_coeffs(
            expected_matrix,
            expected_matrix.size(),
            expected_levels
        );
    }

    void assert_mask_and_rect_are_consistent(const cv::Mat& mask, const cv::Rect& rect, int level)
    {
        auto expected_nonzero_count = rect.width * rect.height;

        EXPECT_EQ(
            cv::countNonZero(mask),
            expected_nonzero_count
        ) << "mask and rect are inconsistent at level " << level;
        EXPECT_EQ(
            cv::countNonZero(mask(rect)),
            expected_nonzero_count
        ) << "mask and rect are inconsistent at level " << level;
    }

    int type;
    cv::Mat expected_matrix;
    DWT2D dwt;
    DWT2D::Coeffs coeffs;
};

TEST_P(Dwt2dCoeffsTest, SizeIsCorrect)
{
    ASSERT_EQ(coeffs.size(), size);
    ASSERT_EQ(coeffs.rows(), rows);
    ASSERT_EQ(coeffs.cols(), cols);
}

TEST_P(Dwt2dCoeffsTest, TypeIsCorrect)
{
    ASSERT_EQ(coeffs.type(), type);
}

TEST_P(Dwt2dCoeffsTest, InitializedToAllZeros)
{
    auto zero_initialized_coeffs = dwt.create_coeffs(rows, cols, type, expected_levels);
    auto expected_coeffs = cv::Mat(rows, cols, type, 0.0);

    EXPECT_THAT(zero_initialized_coeffs, MatrixEq(expected_coeffs));
}

TEST_P(Dwt2dCoeffsTest, InitializedCorrectly)
{
    auto expected_coeffs = create_matrix(rows, cols, type);

    EXPECT_THAT(coeffs, MatrixEq(expected_coeffs));
}

TEST_P(Dwt2dCoeffsTest, LevelsIsCorrect)
{
    ASSERT_EQ(coeffs.levels(), expected_levels);
}

TEST_P(Dwt2dCoeffsTest, CastToMatrix)
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

TEST_P(Dwt2dCoeffsTest, ClonedLevelsIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    ASSERT_EQ(cloned_coeffs.levels(), coeffs.levels());
}

TEST_P(Dwt2dCoeffsTest, ClonedSizeIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    ASSERT_EQ(cloned_coeffs.size(), coeffs.size());
}

TEST_P(Dwt2dCoeffsTest, ClonedTypeIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    ASSERT_EQ(cloned_coeffs.type(), coeffs.type());
}

TEST_P(Dwt2dCoeffsTest, ClonedInputSizeIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    for (int level = 0; level < coeffs.levels(); ++level) {
        ASSERT_EQ(
            cloned_coeffs.input_size(level),
            coeffs.input_size(level)
        ) << "cloned input size does not equal orginal detail size at level " << level;
    }
}

TEST_P(Dwt2dCoeffsTest, ClonedDetailSizeCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    for (int level = 0; level < coeffs.levels(); ++level) {
        ASSERT_EQ(
            cloned_coeffs.detail_size(level),
            coeffs.detail_size(level)
        ) << "cloned detail size does not equal orginal detail size at level " << level;
    }
}

TEST_P(Dwt2dCoeffsTest, ClonedWaveletIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    ASSERT_EQ(cloned_coeffs.wavelet(), coeffs.wavelet());
}

TEST_P(Dwt2dCoeffsTest, ClonedBorderTypeIsCorrect)
{
    auto cloned_coeffs = coeffs.clone();
    ASSERT_EQ(cloned_coeffs.border_type(), coeffs.border_type());
}

TEST_P(Dwt2dCoeffsTest, CloneCopiesData)
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

TEST_P(Dwt2dCoeffsTest, AssignmentFromMatrix)
{
    auto new_coeffs = dwt.create_coeffs(
        expected_matrix.size(),
        expected_matrix.type(),
        expected_levels
    );
    new_coeffs = expected_matrix;

    EXPECT_THAT(new_coeffs, MatrixEq(expected_matrix));
    EXPECT_FALSE(new_coeffs.shares_data(expected_matrix));
}

TEST_P(Dwt2dCoeffsTest, AssignmentFromMatrixExpr)
{
    auto new_coeffs = dwt.create_coeffs(
        expected_matrix.size(),
        expected_matrix.type(),
        expected_levels
    );
    new_coeffs = 1 + expected_matrix;
    cv::Mat h = 1 + expected_matrix;

    EXPECT_THAT(new_coeffs, MatrixEq(h));
    EXPECT_FALSE(new_coeffs.shares_data(expected_matrix));
}

TEST_P(Dwt2dCoeffsTest, AssignmentFromScalar)
{
    auto new_coeffs = dwt.create_coeffs(size, type, expected_levels);
    auto scalar = cv::Scalar(0.5, 1.5, 2.5, 3.5);
    new_coeffs = scalar;

    EXPECT_THAT(new_coeffs, MatrixAllEq(scalar));
}

#ifndef DISABLE_ARG_CHECKS
TEST_P(Dwt2dCoeffsTest, AssignmentFromWrongSizeMatrixIsError)
{
    EXPECT_THROW(
        {
            auto new_coeffs = dwt.create_coeffs(
                cv::Size(1, 1) + expected_matrix.size(),
                expected_matrix.type(),
                expected_levels
            );
            new_coeffs = expected_matrix;
        },
        cv::Exception
    );
}
#endif

TEST_P(Dwt2dCoeffsTest, CollectedHorizontalDetailsSizeEqualsLevels)
{
    EXPECT_EQ(coeffs.collect_horizontal_details().size(), expected_levels);
}

TEST_P(Dwt2dCoeffsTest, CollectedVerticalDetailsSizeEqualsLevels)
{
    EXPECT_EQ(coeffs.collect_vertical_details().size(), expected_levels);
}

TEST_P(Dwt2dCoeffsTest, CollectedDiagonalDetailsSizeEqualsLevels)
{
    EXPECT_EQ(coeffs.collect_diagonal_details().size(), expected_levels);
}

TEST_P(Dwt2dCoeffsTest, LevelIteratorConsistentWithAt)
{
    int level = 0;
    for (auto level_coeffs : coeffs) {
        EXPECT_THAT(
            level_coeffs,
            MatrixEq(coeffs.at_level(level))
        ) << "iterator and at() are inconsistent at level " << level;
        ++level;
    }

    EXPECT_EQ(
        level,
        coeffs.levels()
    ) << "iterator and levels() are inconsistent";
}

TEST_P(Dwt2dCoeffsTest, LevelRectWithNegativeLevel)
{
    for (int level = -1; level >= -expected_levels; --level) {
        EXPECT_EQ(
            coeffs.level_rect(level),
            coeffs.level_rect(level + expected_levels)
        ) << "level_rect() at level = " << level
          << " and level = " << level + expected_levels << " are inconsistent";
    }
}

TEST_P(Dwt2dCoeffsTest, HorizontalDetailRectWithNegativeLevel)
{
    for (int level = -1; level >= -expected_levels; --level) {
        EXPECT_EQ(
            coeffs.horizontal_detail_rect(level),
            coeffs.horizontal_detail_rect(level + expected_levels)
        ) << "horizontal_rect() at level = " << level
          << " and level = " << level + expected_levels << " are inconsistent";
    }
}

TEST_P(Dwt2dCoeffsTest, VerticalDetailRectWithNegativeLevel)
{
    for (int level = -1; level >= -expected_levels; --level) {
        EXPECT_EQ(
            coeffs.vertical_detail_rect(level),
            coeffs.vertical_detail_rect(level + expected_levels)
        ) << "vertical_rect() at level = " << level
          << " and level = " << level + expected_levels << " are inconsistent";
    }
}

TEST_P(Dwt2dCoeffsTest, DiagonalDetailRectWithNegativeLevel)
{
    for (int level = -1; level >= -expected_levels; --level) {
        EXPECT_EQ(
            coeffs.diagonal_detail_rect(level),
            coeffs.diagonal_detail_rect(level + expected_levels)
        ) << "diagonal_rect() at level = " << level
          << " and level = " << level + expected_levels << " are inconsistent";
    }
}

TEST_P(Dwt2dCoeffsTest, ApproxMaskAndRectAreConsistent)
{
    assert_mask_and_rect_are_consistent(
        coeffs.approx_mask(),
        coeffs.approx_rect(),
        0
    );
}

TEST_P(Dwt2dCoeffsTest, HorizontalDetailMaskAndRectAreConsistent)
{
    for (int level = 0; level < expected_levels; ++level) {
        assert_mask_and_rect_are_consistent(
            coeffs.horizontal_detail_mask(level),
            coeffs.horizontal_detail_rect(level),
            level
        );
    }
}

TEST_P(Dwt2dCoeffsTest, VerticalDetailMaskAndRectAreConsistent)
{
    for (int level = 0; level < expected_levels; ++level) {
        assert_mask_and_rect_are_consistent(
            coeffs.vertical_detail_mask(level),
            coeffs.vertical_detail_rect(level),
            level
        );
    }
}

TEST_P(Dwt2dCoeffsTest, DiagonalDetailMaskAndRectAreConsistent)
{
    for (int level = 0; level < expected_levels; ++level) {
        assert_mask_and_rect_are_consistent(
            coeffs.diagonal_detail_mask(level),
            coeffs.diagonal_detail_rect(level),
            level
        );
    }
}

INSTANTIATE_TEST_CASE_P(
    Dwt2dCoeffsGroup,
    Dwt2dCoeffsTest,
    testing::Values(CV_32F, CV_64FC4),
    [](const auto& info) { return get_type_name(info.param); }
);




/**
 * -----------------------------------------------------------------------------
 * Test access to coeffcients at specified levels
 *
 * This test is parameterized to run at each possible level for the specified
 * matrix size.
*/
class Dwt2dCoeffsLevelsTest : public testing::TestWithParam<std::tuple<int, int>>
{
public:
    static const int full_levels = 4;
    const int rows = 32;
    const int cols = 16;
    const cv::Size full_size = cv::Size(cols, rows);

protected:
    Dwt2dCoeffsLevelsTest() :
        testing::TestWithParam<ParamType>(),
        dwt(create_haar())
    {
    }

    void SetUp() override
    {
        level = std::get<0>(GetParam());
        type = std::get<1>(GetParam());

        expected_matrix = create_matrix(rows, cols, type);
        coeffs = dwt.create_coeffs(expected_matrix, full_size, full_levels);

        int level_size_factor = std::pow(2, level);
        int detail_size_factor = 2 * level_size_factor;

        expected_levels = full_levels - level;
        expected_size = full_size / level_size_factor;
        expected_detail_size = expected_size / 2;

        expected_approx_rect = cv::Rect(cv::Point(0, 0), cv::Size(1, 2));

        expected_subband_detail_rects[HORIZONTAL] = cv::Rect(
            cv::Point(0, full_size.height) / detail_size_factor,
            expected_detail_size
        );
        expected_horizontal_detail = expected_matrix(expected_subband_detail_rects[HORIZONTAL]);

        expected_subband_detail_rects[VERTICAL] = cv::Rect(
            cv::Point(full_size.width, 0) / detail_size_factor,
            expected_detail_size
        );
        expected_vertical_detail = expected_matrix(expected_subband_detail_rects[VERTICAL]);

        expected_subband_detail_rects[DIAGONAL] = cv::Rect(
            cv::Point(full_size.width, full_size.height) / detail_size_factor,
            expected_detail_size
        );
        expected_diagonal_detail = expected_matrix(expected_subband_detail_rects[DIAGONAL]);

        expected_level_coeffs = dwt.create_coeffs(
            expected_matrix(cv::Rect(cv::Point(0, 0), expected_size)),
            expected_size,
            expected_levels
        );
    }

    cv::Scalar make_scalar(double value) const
    {
        return cv::Scalar(value, value + 1, value + 2, value + 3);
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

    void assert_set_approx(
        const DWT2D::Coeffs& full_coeffs,
        const cv::Mat& expected_approx,
        const cv::Mat& expected_modified_full_matrix,
        const cv::Mat& approx_from_full_coeffs_before_assign,
        const cv::Mat& approx_from_full_coeffs_after_assign,
        const cv::Mat& approx_from_level_coeffs_before_assign,
        const cv::Mat& approx_from_level_coeffs_after_assign
    )
    {
        assert_set_approx_values_are_correct(
            expected_approx,
            approx_from_full_coeffs_before_assign,
            approx_from_full_coeffs_after_assign,
            approx_from_level_coeffs_before_assign,
            approx_from_level_coeffs_after_assign
        );
        assert_set_approx_does_not_modify_other_coefficients(
            full_coeffs,
            expected_modified_full_matrix
        );
        assert_set_approx_follows_view_semantics(
            full_coeffs,
            approx_from_full_coeffs_before_assign,
            approx_from_full_coeffs_after_assign,
            approx_from_level_coeffs_before_assign,
            approx_from_level_coeffs_after_assign
        );
    }

    void assert_set_approx_values_are_correct(
        const cv::Mat& expected_approx,
        const cv::Mat& approx_from_full_coeffs_before_assign,
        const cv::Mat& approx_from_full_coeffs_after_assign,
        const cv::Mat& approx_from_level_coeffs_before_assign,
        const cv::Mat& approx_from_level_coeffs_after_assign
    )
    {
        EXPECT_THAT(
            approx_from_full_coeffs_before_assign,
            MatrixEq(expected_approx)
        ) << " approx created from coeffs before set_approx() is wrong";

        EXPECT_THAT(
            approx_from_full_coeffs_after_assign,
            MatrixEq(expected_approx)
        ) << " approx created from coeffs after set_approx() is wrong";

        EXPECT_THAT(
            approx_from_level_coeffs_before_assign,
            MatrixEq(expected_approx)
        ) << " approx from level view created after set_approx() is wrong";

        EXPECT_THAT(
            approx_from_level_coeffs_after_assign,
            MatrixEq(expected_approx)
        ) << " approx from level view created before set_approx() is wrong";
    }

    void assert_set_approx_does_not_modify_other_coefficients(
        const DWT2D::Coeffs& full_coeffs,
        const cv::Mat& expected_modified_full_matrix
    )
    {
        EXPECT_THAT(
            full_coeffs,
            MatrixEq(expected_modified_full_matrix)
        ) << "full coeffs are wrong after setting approx";
    }

    void assert_set_approx_follows_view_semantics(
        const DWT2D::Coeffs& full_coeffs,
        const cv::Mat& approx_from_full_coeffs_before_assign,
        const cv::Mat& approx_from_full_coeffs_after_assign,
        const cv::Mat& approx_from_level_coeffs_before_assign,
        const cv::Mat& approx_from_level_coeffs_after_assign
    )
    {
        EXPECT_TRUE(
            full_coeffs.shares_data(approx_from_full_coeffs_before_assign)
        ) << " approx from full coeffs created before set_approx() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(approx_from_full_coeffs_after_assign)
        ) << " approx from full coeffs created after set_approx() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(approx_from_level_coeffs_before_assign)
        ) << " approx from level view created before set_approx() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(approx_from_level_coeffs_after_assign)
        ) << " approx from level view created after set_approx() is a copy";
    }


    void assert_set_detail(
        const DWT2D::Coeffs& full_coeffs,
        const cv::Mat& expected_detail,
        const cv::Mat& expected_modified_full_matrix,
        const cv::Mat& detail_from_full_coeffs_before_assign,
        const cv::Mat& detail_from_full_coeffs_after_assign,
        const cv::Mat& detail_from_level_coeffs_before_assign,
        const cv::Mat& detail_from_level_coeffs_after_assign,
        int subband
    )
    {
        std::string subband_name = get_subband_name(subband);
        assert_set_detail_values_are_correct(
            expected_detail,
            detail_from_full_coeffs_before_assign,
            detail_from_full_coeffs_after_assign,
            detail_from_level_coeffs_before_assign,
            detail_from_level_coeffs_after_assign,
            subband_name
        );
        assert_set_detail_does_not_modify_other_coefficients(
            full_coeffs,
            expected_modified_full_matrix,
            subband_name
        );
        assert_set_detail_follows_view_semantics(
            full_coeffs,
            detail_from_full_coeffs_before_assign,
            detail_from_full_coeffs_after_assign,
            detail_from_level_coeffs_before_assign,
            detail_from_level_coeffs_after_assign,
            subband_name
        );
    }

    void assert_set_detail_values_are_correct(
        const cv::Mat& expected_detail,
        const cv::Mat& detail_from_full_coeffs_before_assign,
        const cv::Mat& detail_from_full_coeffs_after_assign,
        const cv::Mat& detail_from_level_coeffs_before_assign,
        const cv::Mat& detail_from_level_coeffs_after_assign,
        const std::string& subband_name
    )
    {
        EXPECT_THAT(
            detail_from_full_coeffs_before_assign,
            MatrixEq(expected_detail)
        ) << subband_name << " detail created from coeffs before set_details() is wrong";

        EXPECT_THAT(
            detail_from_full_coeffs_after_assign,
            MatrixEq(expected_detail)
        ) << subband_name << " detail created from coeffs after set_details() is wrong";

        EXPECT_THAT(
            detail_from_level_coeffs_before_assign,
            MatrixEq(expected_detail)
        ) << subband_name << " detail from level view created after set_details() is wrong";

        EXPECT_THAT(
            detail_from_level_coeffs_after_assign,
            MatrixEq(expected_detail)
        ) << subband_name << " detail from level view created before set_details() is wrong";
    }

    void assert_set_detail_does_not_modify_other_coefficients(
        const DWT2D::Coeffs& full_coeffs,
        const cv::Mat& expected_modified_full_matrix,
        const std::string& subband_name
    )
    {
        EXPECT_THAT(
            full_coeffs,
            MatrixEq(expected_modified_full_matrix)
        ) << "full coeffs are wrong after setting " << subband_name << " details";
    }

    void assert_set_detail_follows_view_semantics(
        const DWT2D::Coeffs& full_coeffs,
        const cv::Mat& detail_from_full_coeffs_before_assign,
        const cv::Mat& detail_from_full_coeffs_after_assign,
        const cv::Mat& detail_from_level_coeffs_before_assign,
        const cv::Mat& detail_from_level_coeffs_after_assign,
        const std::string& subband_name
    )
    {
        EXPECT_TRUE(
            full_coeffs.shares_data(detail_from_full_coeffs_before_assign)
        ) << subband_name << " detail from full coeffs created before set_details() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(detail_from_full_coeffs_after_assign)
        ) << subband_name << " detail from full coeffs created after set_details() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(detail_from_level_coeffs_before_assign)
        ) << subband_name << " detail from level view created before set_details() is a copy";

        EXPECT_TRUE(
            full_coeffs.shares_data(detail_from_level_coeffs_after_assign)
        ) << subband_name << " detail from level view created after set_details() is a copy";
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

protected:
    int level;
    int type;
    cv::Mat expected_matrix;
    DWT2D dwt;
    DWT2D::Coeffs coeffs;
    cv::Size expected_size;
    int expected_levels;
    cv::Size expected_detail_size;
    DWT2D::Coeffs expected_level_coeffs;
    cv::Mat expected_horizontal_detail;
    cv::Mat expected_vertical_detail;
    cv::Mat expected_diagonal_detail;
    std::map<int, cv::Rect> expected_subband_detail_rects;
    cv::Rect expected_approx_rect;
};

TEST_P(Dwt2dCoeffsLevelsTest, SizeIsCorrect)
{
    auto level_coeffs = coeffs.at_level(level);

    EXPECT_EQ(level_coeffs.size(), expected_size);
}

TEST_P(Dwt2dCoeffsLevelsTest, TypeIsCorrect)
{
    auto level_coeffs = coeffs.at_level(level);

    EXPECT_EQ(level_coeffs.type(), expected_level_coeffs.type());
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelsIsCorrect)
{
    auto level_coeffs = coeffs.at_level(level);

    EXPECT_EQ(level_coeffs.levels(), expected_levels);
}

TEST_P(Dwt2dCoeffsLevelsTest, ValuesAreCorrect)
{
    auto level_coeffs = coeffs.at_level(level);

    EXPECT_THAT(level_coeffs, MatrixEq(expected_level_coeffs));
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAccessDoesNotCauseCopy)
{
    auto level_coeffs = coeffs.at_level(level);

    EXPECT_TRUE(
        level_coeffs.shares_data(coeffs)
    ) << "coeffs.at(" << level << ") copied underlying matrix data";
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAccessIsTransitive)
{
    int i = level;
    for (int j = 0; j < expected_levels - i; ++j) {
        auto level_coeffs1 = coeffs.at_level(i).at_level(j);
        auto level_coeffs2 = coeffs.at_level(i + j);

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
    DWT2D::Coeffs level_coeffs = coeffs.at_level(level);
    cv::Mat expected_level_matrix(expected_size, type, level);

    //  Set level coeffs
    coeffs.set_level(level, expected_level_matrix);

    EXPECT_THAT(level_coeffs, MatrixEq(expected_level_matrix));
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAssignmentFromMatrix)
{
    DWT2D::Coeffs level_coeffs = coeffs.at_level(level);
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
            coeffs.set_level(
                level,
                cv::Mat(cv::Size(1, 1) + expected_size, type, level)
            );
        },
        cv::Exception
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAssignmentToWrongSizeMatrixIsError)
{
    EXPECT_THROW(
        {
            DWT2D::Coeffs level_coeffs = coeffs.at_level(level);
            level_coeffs = cv::Mat(cv::Size(1, 1) + expected_size, type, level);
        },
        cv::Exception
    );
}
#endif

TEST_P(Dwt2dCoeffsLevelsTest, SetLevelWritesIntoOriginalCoeffs)
{
    DWT2D::Coeffs level_coeffs = coeffs.at_level(level);

    cv::Mat expected_level_matrix(
        level_coeffs.rows(),
        level_coeffs.cols(),
        level_coeffs.type(),
        level
    );

    //  Set level coeffs
    coeffs.set_level(level, expected_level_matrix);

    EXPECT_TRUE(
        level_coeffs.shares_data(coeffs)
    ) << "assignment to level caused copy of original coeffs";

    EXPECT_THAT(
        coeffs.at_level(level),
        MatrixEq(expected_level_matrix)
    ) << "assignment to level not reflected in original coeffs";
}

TEST_P(Dwt2dCoeffsLevelsTest, LevelAssignmentWritesIntoOriginalCoeffs)
{
    DWT2D::Coeffs level_coeffs = coeffs.at_level(level);

    cv::Mat expected_level_matrix(
        level_coeffs.rows(),
        level_coeffs.cols(),
        level_coeffs.type(),
        level
    );

    //  Assign level coeffs
    level_coeffs = expected_level_matrix;

    EXPECT_EQ(
        level_coeffs.levels(),
        expected_levels
    ) << "assignment to level has wrong levels";

    EXPECT_TRUE(
        level_coeffs.shares_data(coeffs)
    ) << "assignment to level caused copy of original coeffs";

    EXPECT_THAT(
        coeffs.at_level(level),
        MatrixEq(expected_level_matrix)
    ) << "assignment to level not reflected in original coeffs";
}

TEST_P(Dwt2dCoeffsLevelsTest, SetLevelDoesNotModifyDetailsAtLowerLevels)
{
    DWT2D::Coeffs level_coeffs = coeffs.at_level(level);

    auto [expected_horizontal_details, expected_vertical_details, expected_diagonal_details] = collect_and_clone_details(coeffs);

    //  Set level coeffs
    coeffs.set_level(
        level,
        cv::Mat(level_coeffs.size(), level_coeffs.type(), level)
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
    DWT2D::Coeffs level_coeffs = coeffs.at_level(level);

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
    EXPECT_EQ(
        coeffs.horizontal_detail_rect(level),
        expected_subband_detail_rects[HORIZONTAL]
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, VerticalDetailRect)
{
    EXPECT_EQ(
        coeffs.vertical_detail_rect(level),
        expected_subband_detail_rects[VERTICAL]
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, DiagonalDetailRect)
{
    EXPECT_EQ(
        coeffs.diagonal_detail_rect(level),
        expected_subband_detail_rects[DIAGONAL]
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, HorizontalDetailValues)
{
    EXPECT_THAT(
        coeffs.horizontal_detail(level),
        MatrixEq(expected_horizontal_detail)
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, VerticalDetailValues)
{
    EXPECT_THAT(
        coeffs.vertical_detail(level),
        MatrixEq(expected_vertical_detail)
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, DiagonalDetailValues)
{
    EXPECT_THAT(
        coeffs.diagonal_detail(level),
        MatrixEq(expected_diagonal_detail)
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, HorizontalDetailSharesDataWithCoeffs)
{
    EXPECT_TRUE(
        coeffs.shares_data(coeffs.horizontal_detail(level))
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, VerticalDetailSharesDataWithCoeffs)
{
    EXPECT_TRUE(
        coeffs.shares_data(coeffs.vertical_detail(level))
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, DiagonalDetailSharesDataWithCoeffs)
{
    EXPECT_TRUE(
        coeffs.shares_data(coeffs.diagonal_detail(level))
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, SetApproxToMatrix)
{
    auto expected_modified_full_matrix = expected_matrix.clone();

    auto full_coeffs = dwt.create_coeffs(
        expected_modified_full_matrix,
        full_size,
        full_levels
    );
    auto level_coeffs = full_coeffs.at_level(level);

    //  Get approx coefficients before assignment so that we can make sure
    //  view semantics are followed - i.e. assignment should be reflected
    //  in these objects, it should NOT force a copy of the underlying matrix.
    auto approx_from_full_coeffs_before_assign = full_coeffs.approx();
    auto approx_from_level_coeffs_before_assign = level_coeffs.approx();

    auto new_value = make_scalar(0.5 + level);
    auto new_approx_values = cv::Mat(expected_approx_rect.size(), type, new_value);

    //  update expected
    expected_modified_full_matrix = expected_modified_full_matrix.clone();
    new_approx_values.copyTo(expected_modified_full_matrix(expected_approx_rect));

    //  fill approx with new value
    full_coeffs.set_approx(new_approx_values);

    auto approx_from_full_coeffs_after_assign = full_coeffs.approx();
    auto approx_from_level_coeffs_after_assign = level_coeffs.approx();

    assert_set_approx(
        full_coeffs,
        new_approx_values,
        expected_modified_full_matrix,
        approx_from_full_coeffs_before_assign,
        approx_from_full_coeffs_after_assign,
        approx_from_level_coeffs_before_assign,
        approx_from_level_coeffs_after_assign
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, SetApproxToScalar)
{
    auto expected_modified_full_matrix = expected_matrix.clone();

    auto full_coeffs = dwt.create_coeffs(
        expected_modified_full_matrix,
        full_size,
        full_levels
    );
    auto level_coeffs = full_coeffs.at_level(level);

    //  Get approx coefficients before assignment so that we can make sure
    //  view semantics are followed - i.e. assignment should be reflected
    //  in these objects, it should NOT force a copy of the underlying matrix.
    auto approx_from_full_coeffs_before_assign = full_coeffs.approx();
    auto approx_from_level_coeffs_before_assign = level_coeffs.approx();

    auto new_value = make_scalar(0.5 + level);
    auto new_approx_values = cv::Mat(expected_approx_rect.size(), type, new_value);

    //  update expected
    expected_modified_full_matrix = expected_modified_full_matrix.clone();
    new_approx_values.copyTo(expected_modified_full_matrix(expected_approx_rect));

    //  fill approx with new value
    full_coeffs.set_approx(new_value);

    auto approx_from_full_coeffs_after_assign = full_coeffs.approx();
    auto approx_from_level_coeffs_after_assign = level_coeffs.approx();

    assert_set_approx(
        full_coeffs,
        new_approx_values,
        expected_modified_full_matrix,
        approx_from_full_coeffs_before_assign,
        approx_from_full_coeffs_after_assign,
        approx_from_level_coeffs_before_assign,
        approx_from_level_coeffs_after_assign
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, AssignScalarToApprox)
{
    auto expected_modified_full_matrix = expected_matrix.clone();
    auto full_coeffs = dwt.create_coeffs(
        expected_modified_full_matrix,
        full_size,
        full_levels
    );
    auto level_coeffs = full_coeffs.at_level(level);

    //  Get approx coefficients before assignment so that we can make sure
    //  view semantics are followed - i.e. assignment should be reflected
    //  in these objects, it should NOT force a copy of the underlying matrix.
    auto approx_from_full_coeffs_before_assign = full_coeffs.approx();
    auto approx_from_level_coeffs_before_assign = level_coeffs.approx();

    auto new_value = make_scalar(0.5 + level);
    auto new_approx_values = cv::Mat(expected_approx_rect.size(), type, new_value);

    //  update expected
    expected_modified_full_matrix = expected_modified_full_matrix.clone();
    new_approx_values.copyTo(expected_modified_full_matrix(expected_approx_rect));

    //  fill approx with new value
    full_coeffs.approx() = new_value;

    auto approx_from_full_coeffs_after_assign = full_coeffs.approx();
    auto approx_from_level_coeffs_after_assign = level_coeffs.approx();

    assert_set_approx(
        full_coeffs,
        new_approx_values,
        expected_modified_full_matrix,
        approx_from_full_coeffs_before_assign,
        approx_from_full_coeffs_after_assign,
        approx_from_level_coeffs_before_assign,
        approx_from_level_coeffs_after_assign
    );
}


TEST_P(Dwt2dCoeffsLevelsTest, SetDetailsToMatrix)
{
    for (auto [subband, detail_rect] : expected_subband_detail_rects) {
        auto expected_modified_full_matrix = expected_matrix.clone();
        auto full_coeffs = dwt.create_coeffs(
            expected_modified_full_matrix,
            full_size,
            full_levels
        );

        auto level_coeffs = full_coeffs.at_level(level);

        //  Get detail coefficients before assignment so that we can make sure
        //  view semantics are followed - i.e. assignment should be reflected
        //  in these objects, it should NOT force a copy of the underlying matrix.
        auto detail_from_full_coeffs_before_assign = full_coeffs.detail(level, subband);
        auto detail_from_level_coeffs_before_assign = level_coeffs.detail(subband);

        auto new_value = make_scalar(0.5 + level);
        auto new_detail_values = cv::Mat(detail_rect.size(), type, new_value);

        //  update expected
        expected_modified_full_matrix = expected_modified_full_matrix.clone();
        new_detail_values.copyTo(expected_modified_full_matrix(detail_rect));

        //  fill details with new value
        full_coeffs.set_detail(level, subband, new_detail_values);

        auto detail_from_full_coeffs_after_assign = full_coeffs.detail(level, subband);
        auto detail_from_level_coeffs_after_assign = level_coeffs.detail(subband);

        assert_set_detail(
            full_coeffs,
            new_detail_values,
            expected_modified_full_matrix,
            detail_from_full_coeffs_before_assign,
            detail_from_full_coeffs_after_assign,
            detail_from_level_coeffs_before_assign,
            detail_from_level_coeffs_after_assign,
            subband
        );
    }
}

TEST_P(Dwt2dCoeffsLevelsTest, SetDetailsToScalar)
{
    for (auto [subband, detail_rect] : expected_subband_detail_rects) {
        auto expected_modified_full_matrix = expected_matrix.clone();
        auto full_coeffs = dwt.create_coeffs(
            expected_modified_full_matrix,
            full_size,
            full_levels
        );
        auto level_coeffs = full_coeffs.at_level(level);

        //  Get detail coefficients before assignment so that we can make sure
        //  view semantics are followed - i.e. assignment should be reflected
        //  in these objects, it should NOT force a copy of the underlying matrix.
        auto detail_from_full_coeffs_before_assign = full_coeffs.detail(level, subband);
        auto detail_from_level_coeffs_before_assign = level_coeffs.detail(subband);

        auto new_value = make_scalar(0.5 + level);
        auto new_detail_values = cv::Mat(detail_rect.size(), type, new_value);

        //  update expected
        expected_modified_full_matrix = expected_modified_full_matrix.clone();
        expected_modified_full_matrix(detail_rect) = new_value;

        //  fill details with new value
        full_coeffs.set_detail(level, subband, new_value);

        auto detail_from_full_coeffs_after_assign = full_coeffs.detail(level, subband);
        auto detail_from_level_coeffs_after_assign = level_coeffs.detail(subband);

        assert_set_detail(
            full_coeffs,
            new_detail_values,
            expected_modified_full_matrix,
            detail_from_full_coeffs_before_assign,
            detail_from_full_coeffs_after_assign,
            detail_from_level_coeffs_before_assign,
            detail_from_level_coeffs_after_assign,
            subband
        );
    }
}

TEST_P(Dwt2dCoeffsLevelsTest, AssignScalarToDetails)
{
    for (auto [subband, detail_rect] : expected_subband_detail_rects) {
        auto expected_modified_full_matrix = expected_matrix.clone();
        auto full_coeffs = dwt.create_coeffs(
            expected_modified_full_matrix,
            full_size,
            full_levels
        );
        auto level_coeffs = full_coeffs.at_level(level);

        //  Get detail coefficients before assignment so that we can make sure
        //  view semantics are followed - i.e. assignment should be reflected
        //  in these objects, it should NOT force a copy of the underlying matrix.
        auto detail_from_full_coeffs_before_assign = full_coeffs.detail(level, subband);
        auto detail_from_level_coeffs_before_assign = level_coeffs.detail(subband);

        auto new_value = make_scalar(0.5 + level);
        auto new_detail_values = cv::Mat(detail_rect.size(), type, new_value);

        //  Update expected
        expected_modified_full_matrix = expected_modified_full_matrix.clone();
        expected_modified_full_matrix(detail_rect) = new_value;

        //  fill details with new value
        full_coeffs.detail(level, subband) = new_value;

        auto detail_from_full_coeffs_after_assign = full_coeffs.detail(level, subband);
        auto detail_from_level_coeffs_after_assign = level_coeffs.detail(subband);

        assert_set_detail(
            full_coeffs,
            new_detail_values,
            expected_modified_full_matrix,
            detail_from_full_coeffs_before_assign,
            detail_from_full_coeffs_after_assign,
            detail_from_level_coeffs_before_assign,
            detail_from_level_coeffs_after_assign,
            subband
        );
    }
}

#ifndef DISABLE_ARG_CHECKS
TEST_P(Dwt2dCoeffsLevelsTest, SetDetailsToWrongSizeMatrixIsError)
{
    for (auto [subband, detail_rect] : expected_subband_detail_rects) {
        EXPECT_THROW(
            {
                auto full_coeffs = dwt.create_coeffs(
                    expected_matrix.clone(),
                    expected_matrix.size(),
                    expected_levels
                );
                auto new_detail_values = cv::Mat(cv::Size(1, 1) + detail_rect.size(), type, level);
                full_coeffs.set_detail(level, subband, new_detail_values);
            },
            cv::Exception
        ) << "did not throw exception when setting " << get_subband_name(subband) << " details to an ill-sized matrix";
    }
}
#endif

TEST_P(Dwt2dCoeffsLevelsTest, CollectHorizontalDetails)
{
    assert_level_details_collected_correctly(
        coeffs,
        coeffs.collect_horizontal_details().at(level),
        expected_horizontal_detail
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, CollectVerticalDetails)
{
    assert_level_details_collected_correctly(
        coeffs,
        coeffs.collect_vertical_details().at(level),
        expected_vertical_detail
    );
}

TEST_P(Dwt2dCoeffsLevelsTest, CollectDiagonalDetails)
{
    assert_level_details_collected_correctly(
        coeffs,
        coeffs.collect_diagonal_details().at(level),
        expected_diagonal_detail
    );
}


INSTANTIATE_TEST_CASE_P(
    Dwt2dCoeffsGroup,
    Dwt2dCoeffsLevelsTest,
    testing::Combine(
        testing::Range(0, Dwt2dCoeffsLevelsTest::full_levels),
        testing::Values(CV_32F, CV_64FC4)
    ),
    [](const auto& info) {
        auto level = std::get<0>(info.param);
        auto type = std::get<1>(info.param);
        return (std::stringstream() << "level_" << level << "_" << get_type_name(type)).str();
    }
);


/**
 * -----------------------------------------------------------------------------
*/
struct NormalizeTestParam
{
    NormalizationMode approx_mode;
    NormalizationMode detail_mode;
    double min_approx_value;
    double max_approx_value;
    double min_detail_value;
    double max_detail_value;
    double expected_min_approx_value;
    double expected_max_approx_value;
    double expected_min_detail_value;
    double expected_max_detail_value;
};

class Dwt2dCoeffsNormalizeTest : public testing::TestWithParam<NormalizeTestParam>
{
protected:
    const int rows = 16;
    const int cols = 8;
    const int type = CV_32F;
    const cv::Size full_size = cv::Size(cols, rows);

protected:
    Dwt2dCoeffsNormalizeTest() :
        testing::TestWithParam<ParamType>(),
        dwt(create_haar())
    {
    }

    void SetUp() override
    {
        auto param = GetParam();
        coeffs = dwt.create_coeffs(full_size, type, 3);

        populate_test_case_matrix(param, coeffs);
    }

    void populate_test_case_matrix(const NormalizeTestParam& param, DWT2D::Coeffs& coeffs)
    {
        cv::Mat matrix = coeffs;
        auto set_value = [&](const auto& point, auto value) {
            if (matrix.type() == CV_32F) {
                matrix.at<float>(point) = value;
            } else if (matrix.type() == CV_64F) {
                matrix.at<double>(point) = value;
            } else {
                throw std::runtime_error("invalid type");
            }
        };

        auto approx_rect = coeffs.approx_rect();

        double num_approx_elements = approx_rect.width * approx_rect.height;
        double approx_value = param.min_approx_value;
        double approx_step = \
            (param.max_approx_value - param.min_approx_value) / (num_approx_elements - 1);

        double num_detail_elements = matrix.total() - num_approx_elements;
        double detail_value = param.min_detail_value;
        double detail_step = \
            (param.max_detail_value - param.min_detail_value) / (num_detail_elements - 1);

        for (int y = 0; y < coeffs.size().height; ++y) {
            for (int x = 0; x < coeffs.size().width; ++x) {
                auto point = cv::Point(x, y);
                if (approx_rect.contains(point)) {
                    set_value(point, approx_value);
                    approx_value += approx_step;
                } else {
                    set_value(point, detail_value);
                    detail_value += detail_step;
                }
            }
        }
    }

protected:
    cv::Mat expected_matrix;
    DWT2D dwt;
    DWT2D::Coeffs coeffs;
};

TEST_P(Dwt2dCoeffsNormalizeTest, ApproxValuesAreCorrect)
{
    auto param = GetParam();
    auto approx_mask = coeffs.approx_mask();

    coeffs.normalize(param.approx_mode, param.detail_mode);

    EXPECT_THAT(coeffs, IsMaskedMatrixMin(param.expected_min_approx_value, approx_mask));
    EXPECT_THAT(coeffs, IsMaskedMatrixMax(param.expected_max_approx_value, approx_mask));
}

TEST_P(Dwt2dCoeffsNormalizeTest, DetailValuesAreCorrect)
{
    auto param = GetParam();
    auto detail_mask = coeffs.detail_mask();

    coeffs.normalize(param.approx_mode, param.detail_mode);

    EXPECT_THAT(coeffs, IsMaskedMatrixMin(param.expected_min_detail_value, detail_mask));
    EXPECT_THAT(coeffs, IsMaskedMatrixMax(param.expected_max_detail_value, detail_mask));
}


std::string normalize_mode_string(int mode)
{
    switch (mode){
        case DWT_NO_NORMALIZE: return "DWT_NO_NORMALIZE";
        case DWT_ZERO_TO_HALF_NORMALIZE: return "DWT_ZERO_TO_HALF_NORMALIZE";
        case DWT_MAX_NORMALIZE: return "DWT_MAX_NORMALIZE";
    }

    return "";
}

void PrintTo(const Dwt2dCoeffsNormalizeTest::ParamType& param, std::ostream* stream)
{
    auto abs_max = std::max(
        std::max(std::fabs(param.min_approx_value), std::fabs(param.max_approx_value)),
        std::max(std::fabs(param.min_detail_value), std::fabs(param.max_detail_value))
    );

    *stream << std::setprecision(2)
        << "\n"
        << "APPROX: " << normalize_mode_string(param.approx_mode) << " "
        << "[" << param.min_approx_value << ", " << param.max_approx_value << "] "
        << "/ " << abs_max << " "
        << "-> [" << param.expected_min_approx_value << ", " << param.expected_max_approx_value << "] "
        << "\n"
        << "DETAIL: " << normalize_mode_string(param.detail_mode) << " "
        << "[" << param.min_detail_value << ", " << param.max_detail_value << "] "
        << "/ " << abs_max << " "
        << "-> [" << param.expected_min_detail_value << ", " << param.expected_max_detail_value << "] ";
}


std::vector<NormalizeTestParam> normalize_test_cases = {
    //  ========================================================================
    //  Do Not Normalize Approx, Do Not Normalize Detail
    //  ========================================================================
    //  Case 0
    //  DWT_NO_NORMALIZE
    //  [-10, 10] / 10 -> [-10, 10]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10.0,
        .expected_min_detail_value = -10.0,
        .expected_max_detail_value = 10.0,
    },
    //  ========================================================================
    //  Do Not Normalize Approx, Normalize Detail
    //  ========================================================================
    //  Case 1
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 10 -> [0, 1]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 1.0,
    },
    //  Case 2
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 10] / 10 -> [0.1, 1]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -8.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10.0,
        .expected_min_detail_value = 0.1,
        .expected_max_detail_value = 1.0,
    },
    //  Case 3
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 8] / 10 -> [0.0, 0.9]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -10.0,
        .max_detail_value = 8.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.9,
    },
    //  Case 4
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 20 -> [0.25, 0.75]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 20.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 20.0,
        .expected_min_detail_value = 0.25,
        .expected_max_detail_value = 0.75,
    },
    //  Case 5
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 10] / 20 -> [0.3, 0.75]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 20.0,
        .min_detail_value = -8.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 20.0,
        .expected_min_detail_value = 0.3,
        .expected_max_detail_value = 0.75,
    },
    //  Case 6
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 8] / 20 -> [0.25, 0.7]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 20.0,
        .min_detail_value = -10.0,
        .max_detail_value = 8.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 20.0,
        .expected_min_detail_value = 0.25,
        .expected_max_detail_value = 0.7,
    },
    //  Case 7
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 0.5 -> [0.0, 1.0]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 0.5,
        .min_detail_value = -0.5,
        .max_detail_value = 0.5,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 1.0,
    },
    //  Case 8
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 10] / 0.5 -> [0.1, 1.0]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 0.5,
        .min_detail_value = -0.4,
        .max_detail_value = 0.5,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = 0.1,
        .expected_max_detail_value = 1.0,
    },
    //  Case 9
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 8] / 0.5 -> [0.0, 0.9]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 0.5,
        .min_detail_value = -0.5,
        .max_detail_value = 0.4,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.9,
    },
    //  Case 10
    //  DWT_NO_NORMALIZE
    //  DWT_ZERO_TO_HALF_NORMALIZE [0, 0] / 10 -> [0.5, 0.5]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10,
        .min_detail_value = 0.0,
        .max_detail_value = 0.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10,
        .expected_min_detail_value = 0.5,
        .expected_max_detail_value = 0.5,
    },
    //  ------------------------------------------------------------------------
    //  Case 11
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-10, 10] / 10 -> [-1, 1]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10.0,
        .expected_min_detail_value = -1.0,
        .expected_max_detail_value = 1.0,
    },
    //  Case 12
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-8, 10] / 10 -> [-0.8, 1]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -8.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10.0,
        .expected_min_detail_value = -0.8,
        .expected_max_detail_value = 1.0,
    },
    //  Case 13
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-10, 8] / 10 -> [-1.0, 0.8]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -10.0,
        .max_detail_value = 8.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10.0,
        .expected_min_detail_value = -1.0,
        .expected_max_detail_value = 0.8,
    },
    //  Case 14
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-10, 10] / 20 -> [-0.5, 0.5]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 20.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 20.0,
        .expected_min_detail_value = -0.5,
        .expected_max_detail_value = 0.5,
    },
    //  Case 15
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-8, 10] / 20 -> [-0.4, 0.5]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 20.0,
        .min_detail_value = -8.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 20.0,
        .expected_min_detail_value = -0.4,
        .expected_max_detail_value = 0.5,
    },
    //  Case 16
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-10, 8] / 20 -> [-0.5, 0.4]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 20.0,
        .min_detail_value = -10.0,
        .max_detail_value = 8.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 20.0,
        .expected_min_detail_value = -0.5,
        .expected_max_detail_value = 0.4,
    },
    //  Case 17
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-0.5, 0.5] / 0.5 -> [-1.0, 1.0]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 0.5,
        .min_detail_value = -0.5,
        .max_detail_value = 0.5,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = -1.0,
        .expected_max_detail_value = 1.0,
    },
    //  Case 18
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-0.4, 0.5] / 0.5 -> [-0.8, 1.0]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 0.5,
        .min_detail_value = -0.4,
        .max_detail_value = 0.5,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = -0.8,
        .expected_max_detail_value = 1.0,
    },
    //  Case 19
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [-0.5, 0.4] / 0.5 -> [-1.0, 0.8]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 0.5,
        .min_detail_value = -0.5,
        .max_detail_value = 0.4,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = -1.0,
        .expected_max_detail_value = 0.8,
    },
    //  Case 20
    //  DWT_NO_NORMALIZE
    //  DWT_MAX_NORMALIZE [0, 0] / 10 -> [0.0, 0.0]
    {
        .approx_mode = DWT_NO_NORMALIZE,
        .detail_mode = DWT_MAX_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10,
        .min_detail_value = 0.0,
        .max_detail_value = 0.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 10,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.0,
    },
    //  ========================================================================
    //  Normalize Approx, Do Not Normalize Detail
    //  ========================================================================
    //  Case 21
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 10 -> [0, 1]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -10.0,
        .max_approx_value = 10.0,
        .min_detail_value = 0.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 10.0,
    },
    //  Case 22
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 10] / 10 -> [0.1, 1]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -8.0,
        .max_approx_value = 10.0,
        .min_detail_value = 0.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.1,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 10.0,
    },
    //  Case 23
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 8] / 10 -> [0.0, 0.9]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -10.0,
        .max_approx_value = 8.0,
        .min_detail_value = 0.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.9,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 10.0,
    },
    //  Case 24
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 20 -> [0.25, 0.75]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -10.0,
        .max_approx_value = 10.0,
        .min_detail_value = 0.0,
        .max_detail_value = 20.0,
        .expected_min_approx_value = 0.25,
        .expected_max_approx_value = 0.75,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 20.0,
    },
    //  Case 25
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 10] / 20 -> [0.3, 0.75]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -8.0,
        .max_approx_value = 10.0,
        .min_detail_value = 0.0,
        .max_detail_value = 20.0,
        .expected_min_approx_value = 0.3,
        .expected_max_approx_value = 0.75,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 20.0,
    },
    //  Case 26
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 8] / 20 -> [0.25, 0.7]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -10.0,
        .max_approx_value = 8.0,
        .min_detail_value = 0.0,
        .max_detail_value = 20.0,
        .expected_min_approx_value = 0.25,
        .expected_max_approx_value = 0.7,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 20.0,
    },
    //  Case 27
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 0.5 -> [0.0, 1.0]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -0.5,
        .max_approx_value = 0.5,
        .min_detail_value = 0.0,
        .max_detail_value = 0.5,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.5,
    },
    //  Case 28
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 10] / 0.5 -> [0.1, 1.0]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -0.4,
        .max_approx_value = 0.5,
        .min_detail_value = 0.0,
        .max_detail_value = 0.5,
        .expected_min_approx_value = 0.1,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.5,
    },
    //  Case 29
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 8] / 0.5 -> [0.0, 0.9]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -0.5,
        .max_approx_value = 0.4,
        .min_detail_value = 0.0,
        .max_detail_value = 0.5,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.9,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.5,
    },
    //  Case 30
    //  DWT_ZERO_TO_HALF_NORMALIZE [0, 0] / 10 -> [0.5, 0.5]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 0.0,
        .min_detail_value = 0.0,
        .max_detail_value = 10,
        .expected_min_approx_value = 0.5,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 10,
    },
    //  ------------------------------------------------------------------------
    //  Case 31
    //  DWT_MAX_NORMALIZE [-10, 10] / 10 -> [-1, 1]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -10.0,
        .max_approx_value = 10.0,
        .min_detail_value = 0.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = -1.0,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 10.0,
    },
    //  Case 32
    //  DWT_MAX_NORMALIZE [-8, 10] / 10 -> [-0.8, 1]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -8.0,
        .max_approx_value = 10.0,
        .min_detail_value = 0.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = -0.8,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 10.0,
    },
    //  Case 33
    //  DWT_MAX_NORMALIZE [-10, 8] / 10 -> [-1.0, 0.8]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -10.0,
        .max_approx_value = 8.0,
        .min_detail_value = 0.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = -1.0,
        .expected_max_approx_value = 0.8,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 10.0,
    },
    //  Case 34
    //  DWT_MAX_NORMALIZE [-10, 10] / 20 -> [-0.5, 0.5]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -10.0,
        .max_approx_value = 10.0,
        .min_detail_value = 0.0,
        .max_detail_value = 20.0,
        .expected_min_approx_value = -0.5,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 20.0,
    },
    //  Case 35
    //  DWT_MAX_NORMALIZE [-8, 10] / 20 -> [-0.4, 0.5]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -8.0,
        .max_approx_value = 10.0,
        .min_detail_value = 0.0,
        .max_detail_value = 20.0,
        .expected_min_approx_value = -0.4,
        .expected_max_approx_value = 0.5,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 20.0,
    },
    //  Case 36
    //  DWT_MAX_NORMALIZE [-10, 8] / 20 -> [-0.5, 0.4]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -10.0,
        .max_approx_value = 8.0,
        .min_detail_value = 0.0,
        .max_detail_value = 20.0,
        .expected_min_approx_value = -0.5,
        .expected_max_approx_value = 0.4,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 20.0,
    },
    //  Case 37
    //  DWT_MAX_NORMALIZE [-0.5, 0.5] / 0.5 -> [-1.0, 1.0]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -0.5,
        .max_approx_value = 0.5,
        .min_detail_value = 0.0,
        .max_detail_value = 0.5,
        .expected_min_approx_value = -1.0,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.5,
    },
    //  Case 38
    //  DWT_MAX_NORMALIZE [-0.4, 0.5] / 0.5 -> [-0.8, 1.0]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -0.4,
        .max_approx_value = 0.5,
        .min_detail_value = 0.0,
        .max_detail_value = 0.5,
        .expected_min_approx_value = -0.8,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.5,
    },
    //  Case 39
    //  DWT_MAX_NORMALIZE [-0.5, 0.4] / 0.5 -> [-1.0, 0.8]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = -0.5,
        .max_approx_value = 0.4,
        .min_detail_value = 0.0,
        .max_detail_value = 0.5,
        .expected_min_approx_value = -1.0,
        .expected_max_approx_value = 0.8,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 0.5,
    },
    //  Case 40
    //  DWT_MAX_NORMALIZE [0, 0] / 10 -> [0.0, 0.0]
    //  DWT_NO_NORMALIZE
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_NO_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 0.0,
        .min_detail_value = 0.0,
        .max_detail_value = 10,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 10,
    },
    //  ========================================================================
    //  Normalize Approx, Normalize Detail
    //  ========================================================================
    //  Case 41
    //  DWT_ZERO_TO_HALF_NORMALIZE [0, 10] / 10 -> [0.5, 1]
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 10 -> [0, 1]
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.5,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 1.0,
    },
    //  Case 42
    //  DWT_ZERO_TO_HALF_NORMALIZE [0, 10] / 10 -> [0.5, 1]
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 10] / 10 -> [0.1, 1]
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -8.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.5,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.1,
        .expected_max_detail_value = 1.0,
    },
    //  Case 43
    //  DWT_ZERO_TO_HALF_NORMALIZE [0, 8] / 10 -> [0, 0.8]
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 10 -> [0, 1]
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 8.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.5,
        .expected_max_approx_value = 0.9,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 1.0,
    },
    //  Case 44
    //  DWT_ZERO_TO_HALF_NORMALIZE [0, 10] / 10 -> [0.5, 1]
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, -8] / 10 -> [0.1, 0.9]
    {
        .approx_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -8.0,
        .max_detail_value = 8.0,
        .expected_min_approx_value = 0.5,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.1,
        .expected_max_detail_value = 0.9,
    },
    //  ------------------------------------------------------------------------
    //  Case 45
    //  DWT_MAX_NORMALIZE [0, 10] / 10 -> [0, 1]
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 10 -> [0, 1]
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 1.0,
    },
    //  Case 46
    //  DWT_MAX_NORMALIZE [0, 10] / 10 -> [0, 1]
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 10] / 10 -> [0.1, 1]
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -8.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.1,
        .expected_max_detail_value = 1.0,
    },
    //  Case 47
    //  DWT_MAX_NORMALIZE [0, 8] / 10 -> [0, 0.8]
    //  DWT_ZERO_TO_HALF_NORMALIZE [-10, 10] / 10 -> [0, 1]
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 8.0,
        .min_detail_value = -10.0,
        .max_detail_value = 10.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 0.8,
        .expected_min_detail_value = 0.0,
        .expected_max_detail_value = 1.0,
    },
    //  Case 48
    //  DWT_ZERO_TO_HALF_NORMALIZE [0, 10] / 10 -> [0.0, 1]
    //  DWT_ZERO_TO_HALF_NORMALIZE [-8, 8] / 10 -> [0.1, 0.9]
    {
        .approx_mode = DWT_MAX_NORMALIZE,
        .detail_mode = DWT_ZERO_TO_HALF_NORMALIZE,
        .min_approx_value = 0.0,
        .max_approx_value = 10.0,
        .min_detail_value = -8.0,
        .max_detail_value = 8.0,
        .expected_min_approx_value = 0.0,
        .expected_max_approx_value = 1.0,
        .expected_min_detail_value = 0.1,
        .expected_max_detail_value = 0.9,
    },
};


INSTANTIATE_TEST_CASE_P(
    Dwt2dCoeffsGroup,
    Dwt2dCoeffsNormalizeTest,
    testing::ValuesIn(normalize_test_cases)
);




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

void PrintTo(const DWT2DTestParam& param, std::ostream* stream)
{
    *stream << "\nwavelet_name: " << param.wavelet_name
            << "\ninput_name: " << param.input_name
            << "\nlevels: " << param.levels
            << "\ncoeffs: " << param.coeffs;
}

namespace cv
{
    void from_json(const json& json_matrix, Mat& matrix)
    {
        auto shape = json_matrix["shape"].get<std::vector<int>>();
        int type;
        if (json_matrix["dtype"] == "float64")
            type = CV_64F;
        else if (json_matrix["dtype"] == "float32")
            type = CV_32F;
        else
            assert(false);

        std::vector<double> data = json_matrix["data"];


        Mat data_matrix(data, true);
        matrix.create(data_matrix.size(), type);
        data_matrix.convertTo(matrix, CV_64F);
        if (!matrix.empty()) {
            int channels = (shape.size() == 3) ? shape[2] : 1;
            matrix = matrix.reshape(channels, shape[0]);
        }
    }
}

auto print_dwt2d_test_label  = [](const auto& info) {
    std::string wavelet_name = info.param.wavelet_name;
    std::ranges::replace(wavelet_name, '.', '_');
    return wavelet_name + "_"
        + info.param.input_name + "_"
        + std::to_string(info.param.levels) + "_"
        + get_type_name(info.param.type);
};

class BaseDWT2DTest : public testing::TestWithParam<DWT2DTestParam>
{
public:
    const double SINGLE_NEARNESS_TOLERANCE = 1e-6;
    const double DOUBLE_NEARNESS_TOLERANCE = 1e-10;

protected:
    BaseDWT2DTest() :
        testing::TestWithParam<DWT2DTestParam>(),
        wavelet(Wavelet::create(GetParam().wavelet_name))
    {}

    void SetUp() override
    {
        levels = GetParam().levels;
        if (CV_MAT_DEPTH(GetParam().type) == CV_32F) {
            nearness_tolerance = SINGLE_NEARNESS_TOLERANCE;
        } else {
            nearness_tolerance = DOUBLE_NEARNESS_TOLERANCE;
        }
    }

    void clamp_small_to_zero(auto& ...matrices) const
    {
        // (clamp_near_zero(matrices, CLAMP_TOLERANCE), ...);
        (clamp_near_zero(matrices, 0.1 * nearness_tolerance), ...);
    }

    void get_forward_input(cv::OutputArray input) const
    {
        //  Inputs are stored as doubles, so we need to explicitly convert them
        //  here to test for various input types.
        auto param = GetParam();
        BaseDWT2DTest::inputs.at(param.input_name).convertTo(input, param.type);
    }

    void get_forward_output(cv::OutputArray output) const
    {
        //  Need to convert coeffs because filter bank kernels are stored as
        //  doubles, which forces a type promotion during forward.
        auto param = GetParam();
        param.coeffs.convertTo(output, CV_64F);
    }

    void get_inverse_output(cv::OutputArray output) const
    {
        //  Need to convert output because filter bank kernels are stored as
        //  doubles, which forces a type promotion during inversion.
        auto param = GetParam();
        BaseDWT2DTest::inputs.at(param.input_name).convertTo(output, CV_64F);
    }

    Wavelet wavelet;
    int levels;
    double nearness_tolerance;
    static std::map<std::string, cv::Mat> inputs;
    static std::vector<DWT2DTestParam> params;

public:
    static void create_test_params()
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
                .coeffs = double_precision_coeffs
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

    static std::vector<DWT2DTestParam> get_test_params(int levels=0)
    {
        create_test_params();
        if (levels > 0) {
            std::vector<DWT2DTestParam> levels_params;
            std::ranges::copy_if(
                params,
                std::back_inserter(levels_params),
                [&](auto param) { return param.levels == levels; }
            );

            return levels_params;
        } else {
            return params;
        }
    }
};

std::map<std::string, cv::Mat> BaseDWT2DTest::inputs;
std::vector<DWT2DTestParam> BaseDWT2DTest::params;


/**
 * -----------------------------------------------------------------------------
 * Filter Bank
 * -----------------------------------------------------------------------------
*/
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
};

class ForwardFilterBankTest : public FilterBankTest
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

TEST_P(ForwardFilterBankTest, OutputSize)
{
    auto output_size = wavelet.filter_bank().output_size(input.size());

    EXPECT_EQ(output_size, expected_output.size());
}

TEST_P(ForwardFilterBankTest, SubbandSize)
{
    auto subband_size = wavelet.filter_bank().subband_size(input.size());

    EXPECT_EQ(subband_size, expected_approx.size());
}

TEST_P(ForwardFilterBankTest, Forward)
{
    cv::Mat actual_approx;
    cv::Mat actual_horizontal_detail;
    cv::Mat actual_vertical_detail;
    cv::Mat actual_diagonal_detail;

    wavelet.filter_bank().forward(
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
    ForwardFilterBankTest,
    testing::ValuesIn(ForwardFilterBankTest::get_test_params(1)),
    print_dwt2d_test_label
);


//  ----------------------------------------------------------------------------
class InverseFilterBankTest : public FilterBankTest
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

TEST_P(InverseFilterBankTest, Inverse)
{
    cv::Mat actual_output;
    wavelet.filter_bank().inverse(
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
    InverseFilterBankTest,
    testing::ValuesIn(InverseFilterBankTest::get_test_params(1)),
    print_dwt2d_test_label
);


/**
 * -----------------------------------------------------------------------------
 * DWT2D
 * -----------------------------------------------------------------------------
*/
struct DWT2DMaxLevelsTestParam
{
    std::string wavelet_name;
    cv::Size input_size;
    int expected_max_levels;
    int expected_max_levels_without_border_effects;
};

void PrintTo(const DWT2DMaxLevelsTestParam& param, std::ostream* stream)
{
    *stream << "\nwavelet_name: " << param.wavelet_name
            << "\ninput_size: " << param.input_size
            << "\nexpected_max_levels: " << param.expected_max_levels
            << "\nexpected_max_levels_without_border_effects: " << param.expected_max_levels_without_border_effects;
}

auto print_dwt2d_max_levels_test_label  = [](const auto& info) {
    std::string wavelet_name = info.param.wavelet_name;
    std::ranges::replace(wavelet_name, '.', '_');
    return wavelet_name + "_" + std::to_string(info.param.input_size.width) + "x" + std::to_string(info.param.input_size.height);
};

class DWT2DMaxLevelsTest : public testing::TestWithParam<DWT2DMaxLevelsTestParam>
{
protected:
    DWT2DMaxLevelsTest() :
        dwt(Wavelet::create(GetParam().wavelet_name), cv::BORDER_REFLECT101)
    {}

public:
    static std::vector<ParamType> create_test_params()
    {
        return {
            //  db1 (i.e. filter length == 2)
            {
                .wavelet_name = "db1",
                .input_size = cv::Size(),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db1",
                .input_size = cv::Size(2, 2),
                .expected_max_levels = 1,
                .expected_max_levels_without_border_effects = 1,
            },
            {
                .wavelet_name = "db1",
                .input_size = cv::Size(2, 1),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db1",
                .input_size = cv::Size(1, 2),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db1",
                .input_size = cv::Size(16, 16),
                .expected_max_levels = 4,
                .expected_max_levels_without_border_effects = 4,
            },
            {
                .wavelet_name = "db1",
                .input_size = cv::Size(16, 15),
                .expected_max_levels = 3,
                .expected_max_levels_without_border_effects = 3,
            },
            {
                .wavelet_name = "db1",
                .input_size = cv::Size(15, 16),
                .expected_max_levels = 3,
                .expected_max_levels_without_border_effects = 3,
            },
            //  db2 (i.e. filter length == 4)
            {
                .wavelet_name = "db2",
                .input_size = cv::Size(),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db2",
                .input_size = cv::Size(2, 2),
                .expected_max_levels = 1,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db2",
                .input_size = cv::Size(2, 1),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db2",
                .input_size = cv::Size(1, 2),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db2",
                .input_size = cv::Size(16, 16),
                .expected_max_levels = 4,
                .expected_max_levels_without_border_effects = 2,
            },
            {
                .wavelet_name = "db2",
                .input_size = cv::Size(16, 15),
                .expected_max_levels = 3,
                .expected_max_levels_without_border_effects = 2,
            },
            {
                .wavelet_name = "db2",
                .input_size = cv::Size(15, 16),
                .expected_max_levels = 3,
                .expected_max_levels_without_border_effects = 2,
            },
            {
                .wavelet_name = "db2",
                .input_size = cv::Size(24, 24),
                .expected_max_levels = 4,
                .expected_max_levels_without_border_effects = 3,
            },
            //  db4 (i.e. filter length == 8)
            {
                .wavelet_name = "db4",
                .input_size = cv::Size(),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db4",
                .input_size = cv::Size(2, 2),
                .expected_max_levels = 1,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db4",
                .input_size = cv::Size(2, 1),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db4",
                .input_size = cv::Size(1, 2),
                .expected_max_levels = 0,
                .expected_max_levels_without_border_effects = 0,
            },
            {
                .wavelet_name = "db4",
                .input_size = cv::Size(16, 16),
                .expected_max_levels = 4,
                .expected_max_levels_without_border_effects = 1,
            },
            {
                .wavelet_name = "db4",
                .input_size = cv::Size(16, 15),
                .expected_max_levels = 3,
                .expected_max_levels_without_border_effects = 1,
            },
            {
                .wavelet_name = "db4",
                .input_size = cv::Size(15, 16),
                .expected_max_levels = 3,
                .expected_max_levels_without_border_effects = 1,
            },
            {
                .wavelet_name = "db4",
                .input_size = cv::Size(28, 28),
                .expected_max_levels = 4,
                .expected_max_levels_without_border_effects = 2,
            },
        };
    }

    DWT2D dwt;
};

TEST_P(DWT2DMaxLevelsTest, MaxLevelsWithoutBorderEffects)
{
    auto param = GetParam();
    auto actual_max_levels_without_border_effects = dwt.max_levels_without_border_effects(param.input_size);

    EXPECT_EQ(
        actual_max_levels_without_border_effects,
        param.expected_max_levels_without_border_effects
    );
}

INSTANTIATE_TEST_CASE_P(
    DWT2DGroup,
    DWT2DMaxLevelsTest,
    testing::ValuesIn(DWT2DMaxLevelsTest::create_test_params()),
    print_dwt2d_max_levels_test_label
);


//  ----------------------------------------------------------------------------
class ForwardDWT2DTest : public BaseDWT2DTest
{
protected:
    ForwardDWT2DTest() :
        BaseDWT2DTest(),
        dwt(wavelet, cv::BORDER_REFLECT101)
    {}

    void SetUp() override
    {
        BaseDWT2DTest::SetUp();
        get_forward_input(input);
        get_forward_output(expected_output);
    }

    DWT2D dwt;
    cv::Mat input;
    cv::Mat expected_output;
};

TEST_P(ForwardDWT2DTest, CoeffSizeForInput)
{
    if (!expected_output.empty()) {
        auto coeffs_size = dwt.coeffs_size_for_input(input, levels);

        EXPECT_EQ(coeffs_size, expected_output.size());
    }
}

TEST_P(ForwardDWT2DTest, Forward)
{
    if (expected_output.empty()) {
        EXPECT_THROW({ dwt.forward(input, levels); }, cv::Exception);
    } else {
        auto actual_output = dwt.forward(input, levels);

        //  Clamping is only for readability of failure messages.  It does not
        //  impact the test because the clamp tolerance is smaller than the
        //  the nearness_tolerance.
        clamp_small_to_zero(actual_output, expected_output);

        EXPECT_EQ(actual_output.levels(), levels);
        EXPECT_EQ(actual_output.input_size(), input.size());
        EXPECT_EQ(actual_output.wavelet(), dwt.wavelet);
        EXPECT_EQ(actual_output.border_type(), dwt.border_type);
        EXPECT_THAT(actual_output, MatrixNear(expected_output, nearness_tolerance));
    }
}

TEST_P(ForwardDWT2DTest, CallOperator)
{
    if (expected_output.empty()) {
        EXPECT_THROW({ dwt(input, levels); }, cv::Exception);
    } else {
        auto actual_output = dwt(input, levels);

        //  Clamping is only for readability of failure messages.  It does not
        //  impact the test because the clamp tolerance is smaller than the
        //  the nearness_tolerance.
        clamp_small_to_zero(actual_output, expected_output);

        EXPECT_EQ(actual_output.levels(), levels);
        EXPECT_EQ(actual_output.input_size(), input.size());
        EXPECT_EQ(actual_output.wavelet(), dwt.wavelet);
        EXPECT_EQ(actual_output.border_type(), dwt.border_type);
        EXPECT_THAT(actual_output, MatrixNear(expected_output, nearness_tolerance));
    }
}


INSTANTIATE_TEST_CASE_P(
    DWT2DGroup,
    ForwardDWT2DTest,
    testing::ValuesIn(ForwardDWT2DTest::get_test_params()),
    print_dwt2d_test_label
);


//  ----------------------------------------------------------------------------
class InverseDWT2DTest : public BaseDWT2DTest
{
protected:
    InverseDWT2DTest() :
        BaseDWT2DTest(),
        dwt(wavelet, cv::BORDER_REFLECT101)
    {}

    void SetUp() override
    {
        BaseDWT2DTest::SetUp();
        auto param = GetParam();
        get_inverse_output(expected_output);
        if (!param.coeffs.empty())
            coeffs = dwt.create_coeffs(param.coeffs, expected_output.size(), levels);
    }

    DWT2D dwt;
    DWT2D::Coeffs coeffs;
    cv::Mat expected_output;
};

TEST_P(InverseDWT2DTest, Inverse)
{
    if (coeffs.empty()) {
        EXPECT_THROW({ dwt.inverse(coeffs); }, cv::Exception);
    } else {
        auto actual_output = dwt.inverse(coeffs);

        //  Clamping is only for readability of failure messages.  It does not
        //  impact the test because the clamp tolerance is smaller than the
        //  the nearness_tolerance.
        clamp_small_to_zero(actual_output, expected_output);

        EXPECT_THAT(actual_output, MatrixNear(expected_output, nearness_tolerance));
    }
}

TEST_P(InverseDWT2DTest, Invert)
{
    if (coeffs.empty()) {
        EXPECT_THROW({ coeffs.invert(); }, cv::Exception);
    } else {
        auto actual_output = coeffs.invert();

        //  Clamping is only for readability of failure messages.  It does not
        //  impact the test because the clamp tolerance is smaller than the
        //  the nearness_tolerance.
        clamp_small_to_zero(actual_output, expected_output);

        EXPECT_THAT(actual_output, MatrixNear(expected_output, nearness_tolerance));
    }
}


INSTANTIATE_TEST_CASE_P(
    DWT2DGroup,
    InverseDWT2DTest,
    testing::ValuesIn(InverseDWT2DTest::get_test_params()),
    print_dwt2d_test_label
);

