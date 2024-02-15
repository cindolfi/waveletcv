#ifndef TEST_WAVELET_H
#define TEST_WAVELET_H

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include <wavelet/wavelet.h>
#include <numeric>
#include <vector>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

// #include <read_numpy.hpp>

using json = nlohmann::json;

cv::Mat create_matrix(int rows, int cols, int type, float initial_value = 0)
{
    std::vector<float> elements(rows * cols);
    std::iota(elements.begin(), elements.end(), initial_value);
    auto result = cv::Mat(elements, true).reshape(0, rows);
    result.convertTo(result, type);

    return result;
}




void print_matrix(const cv::Mat& matrix, float zero_clamp=1e-7) {
    if (matrix.type() == CV_32F) {
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
    return cv::countNonZero(a == b) == a.total();
}

bool matrix_is_all_zeros(const cv::Mat& a) {
    return cv::countNonZero(a == 0.0) == a.total();
}

bool matrix_equals_float(const cv::Mat& a, const cv::Mat& b, float tolerance=0.0) {
    if (tolerance <= 0)
        tolerance = std::numeric_limits<float>::epsilon();

    cv::Mat diff;
    cv::absdiff(a, b, diff);
    cv::Mat mask = (diff <= tolerance);
    return cv::countNonZero(diff <= tolerance) == diff.total();
}



TEST(Daubechies, Construction) {
    Wavelet wavelet = daubechies(2);
    ASSERT_EQ(wavelet.order(), 2);
    ASSERT_EQ(wavelet.vanising_moments_psi(), 4);
    ASSERT_EQ(wavelet.vanising_moments_phi(), 0);
    ASSERT_EQ(wavelet.support_width(), 3);
    ASSERT_EQ(wavelet.orthogonal(), true);
    ASSERT_EQ(wavelet.biorthogonal(), true);
    ASSERT_EQ(wavelet.symmetry(), WaveletSymmetry::ASYMMETRIC);
    ASSERT_EQ(wavelet.compact_support(), true);
    ASSERT_EQ(wavelet.family_name(), "Daubechies");
    ASSERT_EQ(wavelet.short_name(), "db2");
}

TEST(Daubechies, CoeffsSize) {
    Wavelet wavelet = daubechies(2);

    ASSERT_EQ(wavelet.analysis_filter_bank().lowpass.size(), 1 + wavelet.support_width());
    ASSERT_EQ(wavelet.analysis_filter_bank().highpass.size(), 1 + wavelet.support_width());
    ASSERT_EQ(wavelet.synthesis_filter_bank().lowpass.size(), 1 + wavelet.support_width());
    ASSERT_EQ(wavelet.synthesis_filter_bank().highpass.size(), 1 + wavelet.support_width());
}

// w.dec_lo = [-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416]
TEST(Daubechies, AnalysisLowpassCoeffs) {
    Wavelet wavelet = daubechies(2);

    auto coeffs = wavelet.analysis_filter_bank().lowpass;

    ASSERT_DOUBLE_EQ(coeffs[0], -1.294095225512603811744494188120241641745344506599652569070016e-01);
    ASSERT_DOUBLE_EQ(coeffs[1], 2.241438680420133810259727622404003554678835181842717613871683e-01);
    ASSERT_DOUBLE_EQ(coeffs[2], 8.365163037378079055752937809168732034593703883484392934953414e-01);
    ASSERT_DOUBLE_EQ(coeffs[3], 4.829629131445341433748715998644486838169524195042022752011715e-01);
}

// w.dec_hi = [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037]
TEST(Daubechies, AnalysisHighpassCoeffs) {
    Wavelet wavelet = daubechies(2);

    auto coeffs = wavelet.analysis_filter_bank().highpass;

    ASSERT_DOUBLE_EQ(coeffs[0], -4.829629131445341433748715998644486838169524195042022752011715e-01);
    ASSERT_DOUBLE_EQ(coeffs[1], 8.365163037378079055752937809168732034593703883484392934953414e-01);
    ASSERT_DOUBLE_EQ(coeffs[2], -2.241438680420133810259727622404003554678835181842717613871683e-01);
    ASSERT_DOUBLE_EQ(coeffs[3], -1.294095225512603811744494188120241641745344506599652569070016e-01);
}

// w.rec_lo = [0.48296291314453416, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037],
TEST(Daubechies, SynthesisLowpassCoeffs) {
    Wavelet wavelet = daubechies(2);

    auto coeffs = wavelet.synthesis_filter_bank().lowpass;

    ASSERT_DOUBLE_EQ(coeffs[0], 4.829629131445341433748715998644486838169524195042022752011715e-01);
    ASSERT_DOUBLE_EQ(coeffs[1], 8.365163037378079055752937809168732034593703883484392934953414e-01);
    ASSERT_DOUBLE_EQ(coeffs[2], 2.241438680420133810259727622404003554678835181842717613871683e-01);
    ASSERT_DOUBLE_EQ(coeffs[3], -1.294095225512603811744494188120241641745344506599652569070016e-01);
}

//  w.rec_hi = [-0.12940952255126037, -0.2241438680420134, 0.8365163037378079, -0.48296291314453416]
TEST(Daubechies, SynthesisHighpassCoeffs) {
    Wavelet wavelet = daubechies(2);

    auto coeffs = wavelet.synthesis_filter_bank().highpass;

    ASSERT_DOUBLE_EQ(coeffs[0], -1.294095225512603811744494188120241641745344506599652569070016e-01);
    ASSERT_DOUBLE_EQ(coeffs[1], -2.241438680420133810259727622404003554678835181842717613871683e-01);
    ASSERT_DOUBLE_EQ(coeffs[2], 8.365163037378079055752937809168732034593703883484392934953414e-01);
    ASSERT_DOUBLE_EQ(coeffs[3], -4.829629131445341433748715998644486838169524195042022752011715e-01);
}



// w.dec_lo = [0.7071067811865476, 0.7071067811865476]
TEST(Haar, AnalysisLowpassCoeffs) {
    Wavelet wavelet = haar();

    auto coeffs = wavelet.analysis_filter_bank().lowpass;

    ASSERT_DOUBLE_EQ(coeffs[0], 7.071067811865475244008443621048490392848359376884740365883398e-01);
    ASSERT_DOUBLE_EQ(coeffs[1], 7.071067811865475244008443621048490392848359376884740365883398e-01);
}

// w.dec_hi = [-0.7071067811865476, 0.7071067811865476]
TEST(Haar, AnalysisHighpassCoeffs) {
    Wavelet wavelet = haar();

    auto coeffs = wavelet.analysis_filter_bank().highpass;

    ASSERT_DOUBLE_EQ(coeffs[0], -7.071067811865475244008443621048490392848359376884740365883398e-01);
    ASSERT_DOUBLE_EQ(coeffs[1], 7.071067811865475244008443621048490392848359376884740365883398e-01);
}

// w.rec_lo = [0.7071067811865476, 0.7071067811865476]
TEST(Haar, SynthesisLowpassCoeffs) {
    Wavelet wavelet = haar();

    auto coeffs = wavelet.synthesis_filter_bank().lowpass;

    ASSERT_DOUBLE_EQ(coeffs[0], 7.071067811865475244008443621048490392848359376884740365883398e-01);
    ASSERT_DOUBLE_EQ(coeffs[1], 7.071067811865475244008443621048490392848359376884740365883398e-01);
}

//  w.rec_hi = [0.7071067811865476, -0.7071067811865476]
TEST(Haar, SynthesisHighpassCoeffs) {
    Wavelet wavelet = haar();

    auto coeffs = wavelet.synthesis_filter_bank().highpass;

    ASSERT_DOUBLE_EQ(coeffs[0], 7.071067811865475244008443621048490392848359376884740365883398e-01);
    ASSERT_DOUBLE_EQ(coeffs[1], -7.071067811865475244008443621048490392848359376884740365883398e-01);
}



/**
 * -----------------------------------------------------------------------------
 * Dwt2dLevelCoeffs
 * -----------------------------------------------------------------------------
*/
class Dwt2dLevelCoeffsTest : public testing::Test {
protected:
    void SetUp() override {
        approx = create_matrix(rows, cols, type, 0);
        horizontal_detail = create_matrix(rows, cols, type, 1);
        vertical_detail = create_matrix(rows, cols, type, 2);
        diagonal_detail = create_matrix(rows, cols, type, 3);


        Dwt2dLevelCoeffs::CoeffArray coeffs = {
            approx,
            horizontal_detail,
            vertical_detail,
            diagonal_detail,
        };

        default_constructed = new Dwt2dLevelCoeffs();
        size_and_type_constructed = new Dwt2dLevelCoeffs(rows, cols, type);
        array_constructed = new Dwt2dLevelCoeffs(coeffs);
        four_matrices_constructed = new Dwt2dLevelCoeffs(
            approx,
            horizontal_detail,
            vertical_detail,
            diagonal_detail
        );
    }

    void TearDown() override {
        delete default_constructed;
        delete array_constructed;
        delete four_matrices_constructed;
    }

    void assert_coeffs_constructed_correctly(
        const Dwt2dLevelCoeffs& level_coeffs,
        const cv::Mat& approx,
        const cv::Mat& horizontal_detail,
        const cv::Mat& vertical_detail,
        const cv::Mat& diagonal_detail
    )
    {
        EXPECT_FALSE(
            level_coeffs.approx().empty()
        ) << "approx() is empty";
        EXPECT_TRUE(
            matrix_equals(level_coeffs.approx(), approx)
        ) << "approx() is not correct";

        EXPECT_FALSE(
            level_coeffs.horizontal_detail().empty()
        ) << "horizontal_detail() is empty";
        EXPECT_TRUE(
            matrix_equals(level_coeffs.horizontal_detail(), horizontal_detail)
        ) << "horizontal_detail() is not correct";

        EXPECT_FALSE(
            level_coeffs.vertical_detail().empty()
        ) << "vertical_detail() is empty";
        EXPECT_TRUE(
            matrix_equals(level_coeffs.vertical_detail(), vertical_detail)
        ) << "vertical_detail() is not correct";

        EXPECT_FALSE(
            level_coeffs.diagonal_detail().empty()
        ) << "diagonal_detail() is empty";
        EXPECT_TRUE(
            matrix_equals(level_coeffs.diagonal_detail(), diagonal_detail)
        ) << "diagonal_detail() is not correct";
    }

    void assert_level_coeffs_size(const Dwt2dLevelCoeffs& level_coeffs, int rows, int cols)
    {
        EXPECT_EQ(level_coeffs.rows(), rows) << "rows() is incorrect";
        EXPECT_EQ(level_coeffs.cols(), cols) << "cols() is incorrect";
        EXPECT_EQ(level_coeffs.size(), cv::Size(cols, rows)) << "size() is incorrect";
    }

    void assert_level_coeffs_type(const Dwt2dLevelCoeffs& level_coeffs, int type)
    {
        EXPECT_EQ(level_coeffs.type(), type) << "type() is incorrect";
    }

    int rows = 16;
    int cols = 8;
    int type = CV_32F;
    cv::Mat approx;
    cv::Mat horizontal_detail;
    cv::Mat vertical_detail;
    cv::Mat diagonal_detail;
    Dwt2dLevelCoeffs* default_constructed;
    Dwt2dLevelCoeffs* size_and_type_constructed;
    Dwt2dLevelCoeffs* array_constructed;
    Dwt2dLevelCoeffs* four_matrices_constructed;
};



//  Default Constructor
TEST_F(Dwt2dLevelCoeffsTest, DefaultConstructor_CoeffsAreEmpty) {
    EXPECT_TRUE(
        default_constructed->approx().empty()
    ) << "approx() is not empty";
    EXPECT_TRUE(
        default_constructed->horizontal_detail().empty()
    ) << "horizontal_detail() is not empty";
    EXPECT_TRUE(
        default_constructed->vertical_detail().empty()
    ) << "vertical_detail() is not empty";
    EXPECT_TRUE(
        default_constructed->diagonal_detail().empty()
    ) << "diagonal_detail() is not empty";
}

TEST_F(Dwt2dLevelCoeffsTest, DefaultConstructed_SizeIsZero) {
    assert_level_coeffs_size(*default_constructed, 0, 0);
}

TEST_F(Dwt2dLevelCoeffsTest, DefaultConstructed_TypeIsFloat) {
    assert_level_coeffs_type(*default_constructed, CV_32F);
}

//  Size & Type Constructor
TEST_F(Dwt2dLevelCoeffsTest, SizeAndTypeConstructor_CoeffsAreZero) {

    EXPECT_TRUE(
        matrix_is_all_zeros(size_and_type_constructed->approx())
    ) << "approx() is not all zeros";
    EXPECT_TRUE(
        matrix_is_all_zeros(size_and_type_constructed->horizontal_detail())
    ) << "horizontal_detail() is not all zeros";
    EXPECT_TRUE(
        matrix_is_all_zeros(size_and_type_constructed->vertical_detail())
    ) << "vertical_detail() is not all zeros";
    EXPECT_TRUE(
        matrix_is_all_zeros(size_and_type_constructed->diagonal_detail())
    ) << "diagonal_detail() is not all zeros";
}

TEST_F(Dwt2dLevelCoeffsTest, SizeAndTypeConstructor_SizeIsCorrect) {
    assert_level_coeffs_size(*size_and_type_constructed, rows, cols);
}

TEST_F(Dwt2dLevelCoeffsTest, SizeAndTypeConstructor_TypeIsCorrect) {
    assert_level_coeffs_type(*size_and_type_constructed, type);
}

//  Array Constructor
TEST_F(Dwt2dLevelCoeffsTest, ArrayConstructor_CoeffsConstructedCorrectly) {
    assert_coeffs_constructed_correctly(
        *array_constructed,
        approx,
        horizontal_detail,
        vertical_detail,
        diagonal_detail
    );
}

TEST_F(Dwt2dLevelCoeffsTest, ArrayConstructor_SizeMatchesCoeffs) {
    assert_level_coeffs_size(*array_constructed, rows, cols);
}

TEST_F(Dwt2dLevelCoeffsTest, ArrayConstructor_TypeMatchesCoeffs) {
    assert_level_coeffs_type(*array_constructed, type);
}

//  Four Matrices Constructor
TEST_F(Dwt2dLevelCoeffsTest, FourMatricesConstructor_CoeffsConstructedCorrectly) {
    assert_coeffs_constructed_correctly(
        *four_matrices_constructed,
        approx,
        horizontal_detail,
        vertical_detail,
        diagonal_detail
    );
}

TEST_F(Dwt2dLevelCoeffsTest, FourMatricesConstructor_SizeMatchesCoeffs) {
    assert_level_coeffs_size(*four_matrices_constructed, rows, cols);
}

TEST_F(Dwt2dLevelCoeffsTest, FourMatricesConstructor_TypeMatchesCoeffs) {
    assert_level_coeffs_type(*four_matrices_constructed, type);
}

//  Accessor
TEST_F(Dwt2dLevelCoeffsTest, AccessorsConsistentWitIndexing) {
    Dwt2dLevelCoeffs& level_coeffs = *four_matrices_constructed;

    EXPECT_TRUE(
        matrix_equals(level_coeffs.at(APPROXIMATION), level_coeffs.approx())
    ) << "approx() and at() are inconsistent";
    EXPECT_TRUE(
        matrix_equals(level_coeffs[APPROXIMATION], level_coeffs.approx())
    ) << "approx() and operaotor[]() are inconsistent";

    EXPECT_TRUE(
        matrix_equals(level_coeffs.at(HORIZONTAL), level_coeffs.horizontal_detail())
    ) << "horizontal_detail() and at() are inconsistent";
    EXPECT_TRUE(
        matrix_equals(level_coeffs[HORIZONTAL], level_coeffs.horizontal_detail())
    ) << "horizontal_detail() and operaotor[]() are inconsistent";

    EXPECT_TRUE(
        matrix_equals(level_coeffs.at(VERTICAL), level_coeffs.vertical_detail())
    ) << "vertical_detail() and at() are inconsistent";
    EXPECT_TRUE(
        matrix_equals(level_coeffs[VERTICAL], level_coeffs.vertical_detail())
    ) << "vertical_detail() and operaotor[]() are inconsistent";

    EXPECT_TRUE(
        matrix_equals(level_coeffs.at(DIAGONAL), level_coeffs.diagonal_detail())
    ) << "diagonal_detail() and at() are inconsistent";
    EXPECT_TRUE(
        matrix_equals(level_coeffs[DIAGONAL], level_coeffs.diagonal_detail())
    ) << "diagonal_detail() and operaotor[]() are inconsistent";
}

//  Exceptions
TEST_F(Dwt2dLevelCoeffsTest, DISABLED_MismatchedCoeffsSiseIsException) {
    // ASSERT_THROW()
}







/**
 * -----------------------------------------------------------------------------
 * Dwt2dResults
 * -----------------------------------------------------------------------------
*/
class Dwt2dResultsTest : public testing::Test {
protected:
    void SetUp() override {
        int rows = 32;
        int cols = 16;
        int type = CV_32F;

        // coeffs = Dwt2dResults::Coefficients{
        //     create_level_coeffs(rows, cols, type),
        //     create_level_coeffs(rows / 2, cols / 2, type),
        //     create_level_coeffs(rows / 4, cols / 4, type),
        //     create_level_coeffs(rows / 8, cols / 8, type),
        //     create_level_coeffs(rows / 16, cols / 16, type),
        // };

        // coeffs_constructed_results = Dwt2dResults(coeffs);

        expected_matrix = create_matrix(rows, cols, type);

        auto approx_0 = cv::Mat();
        auto horizontal_0 = expected_matrix(cv::Range(16, 32), cv::Range(0, 8));
        auto vertical_0 = expected_matrix(cv::Range(0, 16), cv::Range(8, 16));
        auto diagonal_0 = expected_matrix(cv::Range(16, 32), cv::Range(8, 16));

        auto approx_1 = cv::Mat();
        auto horizontal_1 = expected_matrix(cv::Range(8, 16), cv::Range(0, 4));
        auto vertical_1 = expected_matrix(cv::Range(0, 8), cv::Range(4, 8));
        auto diagonal_1 = expected_matrix(cv::Range(8, 16), cv::Range(4, 8));

        auto approx_2 = cv::Mat();
        auto horizontal_2 = expected_matrix(cv::Range(4, 8), cv::Range(0, 2));
        auto vertical_2 = expected_matrix(cv::Range(0, 4), cv::Range(2, 4));
        auto diagonal_2 = expected_matrix(cv::Range(4, 8), cv::Range(2, 4));

        auto approx_3 = expected_matrix(cv::Range(0, 2), cv::Range(0, 1));
        auto horizontal_3 = expected_matrix(cv::Range(2, 4), cv::Range(0, 1));
        auto vertical_3 = expected_matrix(cv::Range(0, 2), cv::Range(1, 2));
        auto diagonal_3 = expected_matrix(cv::Range(2, 4), cv::Range(1, 2));

        // level0 = Dwt2dLevelCoeffs(approx_0, horizontal_0, vertical_0, diagonal_0);
        // level1 = Dwt2dLevelCoeffs(approx_1, horizontal_1, vertical_1, diagonal_1);
        // level2 = Dwt2dLevelCoeffs(approx_2, horizontal_2, vertical_2, diagonal_2);
        // level3 = Dwt2dLevelCoeffs(approx_3, horizontal_3, vertical_3, diagonal_3);
        // level_coeffs = {level0, level1, level2, level3};

        level_coeffs = {
            Dwt2dLevelCoeffs(approx_0, horizontal_0, vertical_0, diagonal_0),
            Dwt2dLevelCoeffs(approx_1, horizontal_1, vertical_1, diagonal_1),
            Dwt2dLevelCoeffs(approx_2, horizontal_2, vertical_2, diagonal_2),
            Dwt2dLevelCoeffs(approx_3, horizontal_3, vertical_3, diagonal_3),
        };

        results = Dwt2dResults(level_coeffs);
    }

    void TearDown() override {
    }

    Dwt2dLevelCoeffs create_level_coeffs(int rows, int cols, int type)
    {
        cv::Mat approx = create_matrix(rows, cols, type, 0);
        cv::Mat horizontal_detail = create_matrix(rows, cols, type, 1);
        cv::Mat vertical_detail = create_matrix(rows, cols, type, 2);
        cv::Mat diagonal_detail = create_matrix(rows, cols, type, 3);

        return Dwt2dLevelCoeffs(approx, horizontal_detail, vertical_detail, diagonal_detail);
    }

    void assert_level_details_collected_correctly(const Dwt2dResults& results, int detail_type)
    {
        auto details = results.details(detail_type);
        ASSERT_EQ(
            details.size(),
            level_coeffs.size()
        ) << "size of collected " << detail_type_name(detail_type) << " details is incorrect";

        for (int level = 0; level < level_coeffs.size(); ++level) {
            EXPECT_TRUE(
                matrix_equals(details[level], level_coeffs[level][detail_type])
            ) << detail_type_name(detail_type) << " at level " << level << " are not collected correctly";
        }
    }

    std::string detail_type_name(int detail_type) {
        switch (detail_type) {
            case HORIZONTAL:
                return "horizontal";
            case VERTICAL:
                return "vertical";
            case DIAGONAL:
                return "diagonal";
            default:
                throw std::invalid_argument(
                    std::to_string(detail_type) + " is not a valid detail type"
                );
        };
    }

    Dwt2dResults default_results;
    Dwt2dResults coeffs_constructed_results;
    Dwt2dResults::Coefficients coeffs;
    Dwt2dResults::Coefficients level_coeffs;
    Dwt2dResults results;
    cv::Mat expected_matrix;
};

TEST_F(Dwt2dResultsTest, DefaultConstructor)
{
    ASSERT_TRUE(default_results.empty());
}

TEST_F(Dwt2dResultsTest, DefaultConstructor_DepthIsZero)
{
    ASSERT_EQ(default_results.depth(), 0);
}

TEST_F(Dwt2dResultsTest, DefaultConstructor_ApproxIsEmpty)
{
    ASSERT_TRUE(default_results.approx().empty());
}

TEST_F(Dwt2dResultsTest, DefaultConstructor_CollectDetailsIsEmpty)
{
    EXPECT_TRUE(
        default_results.horizontal_details().empty()
    ) << "horizontal_details() is not empty";

    EXPECT_TRUE(
        default_results.vertical_details().empty()
    ) << "vertical_details() is not empty";

    EXPECT_TRUE(
        default_results.diagonal_details().empty()
    ) << "diagonal_details() is not empty";
}

TEST_F(Dwt2dResultsTest, ApproxEqualsLastLevel)
{
    ASSERT_TRUE(matrix_equals(results.approx(), level_coeffs.back().approx()));
}

TEST_F(Dwt2dResultsTest, DepthEqualsSize)
{
    ASSERT_EQ(results.depth(), level_coeffs.size());
}

TEST_F(Dwt2dResultsTest, CollectDetails)
{
    assert_level_details_collected_correctly(results, HORIZONTAL);
    assert_level_details_collected_correctly(results, VERTICAL);
    assert_level_details_collected_correctly(results, DIAGONAL);
}

TEST_F(Dwt2dResultsTest, MatrixBuiltWithCorrectOrdering)
{
    ASSERT_TRUE(matrix_equals(expected_matrix, results.as_matrix()));
}



/**
 * -----------------------------------------------------------------------------
 * DWT2D
 * -----------------------------------------------------------------------------
*/
TEST(DWT2D, single_level_zeros) {
    Wavelet wavelet = daubechies(2);
    DWT2D dwt(wavelet);

    cv::Mat input(32, 32, CV_32F, 0.0);
    cv::Mat expected_approx(16, 16, CV_32F, 0.0);
    cv::Mat expected_horizontal_detail(16, 16, CV_32F, 0.0);
    cv::Mat expected_vertical_detail(16, 16, CV_32F, 0.0);
    cv::Mat expected_diagonal_detail(16, 16, CV_32F, 0.0);

    auto coeffs = dwt.forward_single_level(input);

    ASSERT_TRUE(matrix_equals(coeffs.approx(), expected_approx));
    ASSERT_TRUE(matrix_equals(coeffs.horizontal_detail(), expected_horizontal_detail));
    ASSERT_TRUE(matrix_equals(coeffs.vertical_detail(), expected_vertical_detail));
    ASSERT_TRUE(matrix_equals(coeffs.diagonal_detail(), expected_diagonal_detail));
}


TEST(DWT2D, zeros) {
    Wavelet wavelet = daubechies(2);

    DWT2D dwt(wavelet);

    cv::Mat input(32, 32, CV_32F, 0.0);
    cv::Mat expected_output(32, 32, CV_32F, 0.0);

    auto result = dwt(input);
    auto actual_output = result.as_matrix();

    ASSERT_EQ(result.depth(), 5);
    ASSERT_TRUE(matrix_equals(actual_output, expected_output));
}

TEST(DWT2D, zeros_two_levels) {
    Wavelet wavelet = daubechies(2);

    DWT2D dwt(wavelet);

    cv::Mat input(32, 32, CV_32F, 0.0);
    cv::Mat expected_matrix(32, 32, CV_32F, 0.0);

    cv::Mat expected_approx_0(16, 16, CV_32F, 0.0);
    cv::Mat expected_horizontal_detail_0(16, 16, CV_32F, 0.0);
    cv::Mat expected_vertical_detail_0(16, 16, CV_32F, 0.0);
    cv::Mat expected_diagonal_detail_0(16, 16, CV_32F, 0.0);

    cv::Mat expected_approx_1(8, 8, CV_32F, 0.0);
    cv::Mat expected_horizontal_detail_1(8, 8, CV_32F, 0.0);
    cv::Mat expected_vertical_detail_1(8, 8, CV_32F, 0.0);
    cv::Mat expected_diagonal_detail_1(8, 8, CV_32F, 0.0);

    auto result = dwt(input, 2);

    ASSERT_EQ(result.depth(), 2);
    ASSERT_TRUE(matrix_equals(result.as_matrix(), expected_matrix));

    auto coeffs_0 = result.at(0);
    ASSERT_TRUE(matrix_equals(coeffs_0.approx(), expected_approx_0));
    ASSERT_TRUE(matrix_equals(coeffs_0.horizontal_detail(), expected_horizontal_detail_0));
    ASSERT_TRUE(matrix_equals(coeffs_0.vertical_detail(), expected_vertical_detail_0));
    ASSERT_TRUE(matrix_equals(coeffs_0.diagonal_detail(), expected_diagonal_detail_0));

    auto coeffs_1 = result.at(1);
    ASSERT_TRUE(matrix_equals(coeffs_1.approx(), expected_approx_1));
    ASSERT_TRUE(matrix_equals(coeffs_1.horizontal_detail(), expected_horizontal_detail_1));
    ASSERT_TRUE(matrix_equals(coeffs_1.vertical_detail(), expected_vertical_detail_1));
    ASSERT_TRUE(matrix_equals(coeffs_1.diagonal_detail(), expected_diagonal_detail_1));
}
















class HaarPatternParameterizedTests : public testing::TestWithParam<std::tuple<cv::Mat, cv::Mat>>
{
};

static auto make_pattern_case_data(const std::vector<float>& pattern, const std::vector<float>& expected)
{
    int rows = std::sqrt(pattern.size());

    return std::make_tuple(
        cv::Mat(pattern, true).reshape(0, rows),
        cv::Mat(expected, true).reshape(0, rows)
    );
}


std::vector<std::tuple<cv::Mat, cv::Mat>> basic_test_patterns = {
    //  zeros
    make_pattern_case_data(
        {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        },
        {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        }
    ),
    //  ones
    make_pattern_case_data(
        {
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        },
        {
            16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        }
    ),
    //  checker board
    make_pattern_case_data(
        {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        },
        {
            8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        }
    ),
    //  inverted checker board
    make_pattern_case_data(
        {
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        },
        {
            8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        }
    ),
    //  vertical lines
    make_pattern_case_data(
        {
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
        },
        {
            8., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        }
    ),
    //  inverted vertical lines
    make_pattern_case_data(
        {
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
        },
        {
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
        }
    ),
    //  horizontal lines
    make_pattern_case_data(
        {
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        },
        {
            8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        }
    ),
    //  inverted horizontal lines
    make_pattern_case_data(
        {
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        },
        {
             8.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        }
    ),
};

std::vector<std::string> pattern_names = {
    "ZerosPattern",
    "OnesPattern",
    "CheckerBoardPattern",
    "InvertedCheckerBoardPattern",
    "VerticalLinesPattern",
    "InvertedVerticalLinesPattern",
    "HorizontalLinesPattern",
    "InvertedHorizontalLinesPattern",
};




TEST_P(HaarPatternParameterizedTests, ForwardTransform) {
    DWT2D dwt(haar(), cv::BORDER_REFLECT101);

    auto [input_pattern, expected_output] = GetParam();
    auto actual_output = dwt(input_pattern).as_matrix();

    ASSERT_TRUE(matrix_equals_float(actual_output, expected_output, 1e-5));
}


// TEST_P(HaarPatternParameterizedTests, InverseTransform) {
//     DWT2D dwt(haar(), cv::BORDER_REFLECT101);

//     auto [expected_output, input_pattern] = GetParam();
//     auto actual_output = dwt.inverse(expected_output).as_matrix();

//     ASSERT_TRUE(matrix_equals_float(actual_output, expected_output, 1e-6));
// }


INSTANTIATE_TEST_CASE_P(
    HaarPatternTests,
    HaarPatternParameterizedTests,
    testing::ValuesIn(basic_test_patterns),
    [](const auto& info) { return pattern_names[info.index]; }
);




































#endif


