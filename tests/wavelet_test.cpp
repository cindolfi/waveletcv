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
    return cv::Mat(elements, true).reshape(0, rows);
}

bool matrix_equals(const cv::Mat &a, const cv::Mat &b) {
    cv::Mat mask = a == b;
    return std::all_of(
        mask.begin<int>(), 
        mask.end<int>(), 
        [](int i) { return bool(i); }
    );
}

void print_matrix(const cv::Mat& matrix) {
    if (matrix.type() == CV_32F) {
        for (int row = 0; row < matrix.rows; ++row) {
            for (int col = 0; col < matrix.cols; ++col) {
                std::cout << std::setfill(' ') << std::setprecision(6) << matrix.at<float>(row, col) << " ";
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



// TEST(Daubechies, Construction) {
//     DaubechiesWavelet wavelet(2);
//     ASSERT_EQ(wavelet.order, 2);
//     ASSERT_EQ(wavelet.vanising_moments_psi, 4);
//     ASSERT_EQ(wavelet.vanising_moments_phi, 0);
//     ASSERT_EQ(wavelet.support_width, 3);
//     ASSERT_EQ(wavelet.orthogonal, true);
//     ASSERT_EQ(wavelet.biorthogonal, true);
//     ASSERT_EQ(wavelet.symmetry, WaveletSymmetry::ASYMMETRIC);
//     ASSERT_EQ(wavelet.compact_support, true);
//     ASSERT_EQ(wavelet.family_name, "Daubechies");
//     ASSERT_EQ(wavelet.short_name, "db");
// }

// TEST(Daubechies, CoeffsSize) {
//     DaubechiesWavelet wavelet(2);
//     ASSERT_EQ(wavelet.analysis_highpass_coeffs().size(), 1 + wavelet.support_width);
//     ASSERT_EQ(wavelet.analysis_lowpass_coeffs().size(), 1 + wavelet.support_width);
//     ASSERT_EQ(wavelet.synthesis_highpass_coeffs().size(), 1 + wavelet.support_width);
//     ASSERT_EQ(wavelet.synthesis_lowpass_coeffs().size(), 1 + wavelet.support_width);
// }

// // w.dec_lo = [-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416]
// TEST(Daubechies, AnalysisLowpassCoeffs) {
//     DaubechiesWavelet wavelet(2);
    
//     auto coeffs = wavelet.analysis_lowpass_coeffs();

//     ASSERT_DOUBLE_EQ(coeffs[0], -1.294095225512603811744494188120241641745344506599652569070016e-01);
//     ASSERT_DOUBLE_EQ(coeffs[1], 2.241438680420133810259727622404003554678835181842717613871683e-01);
//     ASSERT_DOUBLE_EQ(coeffs[2], 8.365163037378079055752937809168732034593703883484392934953414e-01);
//     ASSERT_DOUBLE_EQ(coeffs[3], 4.829629131445341433748715998644486838169524195042022752011715e-01);   
// }

// // w.dec_hi = [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037]
// TEST(Daubechies, AnalysisHighpassCoeffs) {
//     DaubechiesWavelet wavelet(2);
    
//     auto coeffs = wavelet.analysis_highpass_coeffs();

//     ASSERT_DOUBLE_EQ(coeffs[0], -4.829629131445341433748715998644486838169524195042022752011715e-01);
//     ASSERT_DOUBLE_EQ(coeffs[1], 8.365163037378079055752937809168732034593703883484392934953414e-01);
//     ASSERT_DOUBLE_EQ(coeffs[2], -2.241438680420133810259727622404003554678835181842717613871683e-01);
//     ASSERT_DOUBLE_EQ(coeffs[3], -1.294095225512603811744494188120241641745344506599652569070016e-01);   
// }

// // w.rec_lo = [0.48296291314453416, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037], 
// TEST(Daubechies, SynthesisLowpassCoeffs) {
//     DaubechiesWavelet wavelet(2);
    
//     auto coeffs = wavelet.synthesis_lowpass_coeffs();

//     ASSERT_DOUBLE_EQ(coeffs[0], 4.829629131445341433748715998644486838169524195042022752011715e-01);
//     ASSERT_DOUBLE_EQ(coeffs[1], 8.365163037378079055752937809168732034593703883484392934953414e-01);
//     ASSERT_DOUBLE_EQ(coeffs[2], 2.241438680420133810259727622404003554678835181842717613871683e-01);
//     ASSERT_DOUBLE_EQ(coeffs[3], -1.294095225512603811744494188120241641745344506599652569070016e-01);
// }

// //  w.rec_hi = [-0.12940952255126037, -0.2241438680420134, 0.8365163037378079, -0.48296291314453416] 
// TEST(Daubechies, SynthesisHighpassCoeffs) {
//     DaubechiesWavelet wavelet(2);
    
//     auto coeffs = wavelet.synthesis_highpass_coeffs();

//     ASSERT_DOUBLE_EQ(coeffs[0], -1.294095225512603811744494188120241641745344506599652569070016e-01);
//     ASSERT_DOUBLE_EQ(coeffs[1], -2.241438680420133810259727622404003554678835181842717613871683e-01);
//     ASSERT_DOUBLE_EQ(coeffs[2], 8.365163037378079055752937809168732034593703883484392934953414e-01);
//     ASSERT_DOUBLE_EQ(coeffs[3], -4.829629131445341433748715998644486838169524195042022752011715e-01);
// }



// /**
//  * -----------------------------------------------------------------------------
//  * Dwt2dLevelCoeffs
//  * -----------------------------------------------------------------------------
// */
// void assert_level_coeffs_accessors(
//     const Dwt2dLevelCoeffs &level_coeffs, 
//     const cv::Mat &approx, 
//     const cv::Mat &horizontal_detail, 
//     const cv::Mat &vertical_detail, 
//     const cv::Mat &diagonal_detail
// ) 
// {
//     ASSERT_TRUE(matrix_equals(level_coeffs.approx(), approx));
//     ASSERT_TRUE(matrix_equals(level_coeffs.horizontal_detail(), horizontal_detail));
//     ASSERT_TRUE(matrix_equals(level_coeffs.vertical_detail(), vertical_detail));
//     ASSERT_TRUE(matrix_equals(level_coeffs.diagonal_detail(), diagonal_detail));
// }

// void assert_level_coeffs_accessors_consistent_with_coeffs_method(const Dwt2dLevelCoeffs &level_coeffs) 
// {
//     ASSERT_TRUE(matrix_equals(level_coeffs.coeffs(APPROXIMATION), level_coeffs.approx()));
//     ASSERT_TRUE(matrix_equals(level_coeffs.coeffs(HORIZONTAL), level_coeffs.horizontal_detail()));
//     ASSERT_TRUE(matrix_equals(level_coeffs.coeffs(VERTICAL), level_coeffs.vertical_detail()));
//     ASSERT_TRUE(matrix_equals(level_coeffs.coeffs(DIAGONAL), level_coeffs.diagonal_detail()));
// }

// void assert_level_coeffs_all_coeffs_method_consistent_with_coeffs_method(const Dwt2dLevelCoeffs &level_coeffs) 
// {
//     auto coeffs = level_coeffs.all_coeffs();
//     ASSERT_TRUE(matrix_equals(coeffs[APPROXIMATION], level_coeffs.coeffs(APPROXIMATION)));
//     ASSERT_TRUE(matrix_equals(coeffs[HORIZONTAL], level_coeffs.coeffs(HORIZONTAL)));
//     ASSERT_TRUE(matrix_equals(coeffs[VERTICAL], level_coeffs.coeffs(VERTICAL)));
//     ASSERT_TRUE(matrix_equals(coeffs[DIAGONAL], level_coeffs.coeffs(DIAGONAL)));
// }

// void assert_level_coeffs_matrix_properties(const Dwt2dLevelCoeffs &level_coeffs, int rows, int cols, int type) {
//     ASSERT_EQ(level_coeffs.rows(), rows);
//     ASSERT_EQ(level_coeffs.cols(), cols);
//     ASSERT_EQ(level_coeffs.type(), type);
// }

// void assert_level_coeffs_constructed_correctly(
//     const Dwt2dLevelCoeffs &level_coeffs, 
//     const cv::Mat &approx, 
//     const cv::Mat &horizontal_detail, 
//     const cv::Mat &vertical_detail, 
//     const cv::Mat &diagonal_detail,
//     int rows,
//     int cols,
//     int type
// ) 
// {
//     assert_level_coeffs_accessors(level_coeffs, approx, horizontal_detail, vertical_detail, diagonal_detail);
//     assert_level_coeffs_accessors_consistent_with_coeffs_method(level_coeffs);
//     assert_level_coeffs_all_coeffs_method_consistent_with_coeffs_method(level_coeffs);
//     assert_level_coeffs_matrix_properties(level_coeffs, rows, cols, type);
// }

// TEST(Dwt2dLevelCoeffs, DefaultConstructor) {
//     Dwt2dLevelCoeffs level_coeffs;

//     ASSERT_TRUE(level_coeffs.approx().empty());
//     ASSERT_TRUE(level_coeffs.horizontal_detail().empty());
//     ASSERT_TRUE(level_coeffs.vertical_detail().empty());
//     ASSERT_TRUE(level_coeffs.diagonal_detail().empty());

//     ASSERT_TRUE(level_coeffs.coeffs(APPROXIMATION).empty());
//     ASSERT_TRUE(level_coeffs.coeffs(HORIZONTAL).empty());
//     ASSERT_TRUE(level_coeffs.coeffs(VERTICAL).empty());
//     ASSERT_TRUE(level_coeffs.coeffs(DIAGONAL).empty());

//     auto all_coeffs = level_coeffs.all_coeffs();
//     ASSERT_TRUE(all_coeffs[APPROXIMATION].empty());
//     ASSERT_TRUE(all_coeffs[HORIZONTAL].empty());
//     ASSERT_TRUE(all_coeffs[VERTICAL].empty());
//     ASSERT_TRUE(all_coeffs[DIAGONAL].empty());

//     ASSERT_EQ(level_coeffs.rows(), 0);
//     ASSERT_EQ(level_coeffs.cols(), 0);
//     ASSERT_EQ(level_coeffs.type(), 0);
// }

// TEST(Dwt2dLevelCoeffs, FourMatricesConstructor) {
//     int rows = 16;
//     int cols = 8;
//     int type = CV_32F;
//     cv::Mat approx = create_matrix(rows, cols, type, 0);
//     cv::Mat horizontal_detail = create_matrix(rows, cols, type, 1);
//     cv::Mat vertical_detail = create_matrix(rows, cols, type, 2);
//     cv::Mat diagonal_detail = create_matrix(rows, cols, type, 3);

//     Dwt2dLevelCoeffs level_coeffs(approx, horizontal_detail, vertical_detail, diagonal_detail);

//     assert_level_coeffs_constructed_correctly(
//         level_coeffs, 
//         approx, 
//         horizontal_detail, 
//         vertical_detail, 
//         diagonal_detail,
//         rows,
//         cols,
//         type
//     );
// }

// TEST(Dwt2dLevelCoeffs, ArrayConstructor) {
//     int rows = 16;
//     int cols = 8;
//     int type = CV_32F;
//     cv::Mat approx = create_matrix(rows, cols, type, 0);
//     cv::Mat horizontal_detail = create_matrix(rows, cols, type, 1);
//     cv::Mat vertical_detail = create_matrix(rows, cols, type, 2);
//     cv::Mat diagonal_detail = create_matrix(rows, cols, type, 3);

//     Dwt2dLevelCoeffs::Coefficients coeffs = {approx, horizontal_detail, vertical_detail, diagonal_detail};
//     Dwt2dLevelCoeffs level_coeffs(coeffs);

//     assert_level_coeffs_constructed_correctly(
//         level_coeffs, 
//         approx, 
//         horizontal_detail, 
//         vertical_detail, 
//         diagonal_detail,
//         rows,
//         cols,
//         type
//     );
// }

// TEST(Dwt2dLevelCoeffs, CopyConstructor) {
//     int rows = 16;
//     int cols = 8;
//     int type = CV_32F;
//     cv::Mat approx = create_matrix(rows, cols, type, 0);
//     cv::Mat horizontal_detail = create_matrix(rows, cols, type, 1);
//     cv::Mat vertical_detail = create_matrix(rows, cols, type, 2);
//     cv::Mat diagonal_detail = create_matrix(rows, cols, type, 3);

//     Dwt2dLevelCoeffs level_coeffs = Dwt2dLevelCoeffs(approx, horizontal_detail, vertical_detail, diagonal_detail);

//     assert_level_coeffs_constructed_correctly(
//         level_coeffs, 
//         approx, 
//         horizontal_detail, 
//         vertical_detail, 
//         diagonal_detail,
//         rows,
//         cols,
//         type
//     );
// }




// /**
//  * -----------------------------------------------------------------------------
//  * Dwt2dResults
//  * -----------------------------------------------------------------------------
// */
// Dwt2dLevelCoeffs create_level_coeffs(int rows, int cols, int type)
// {
//     cv::Mat approx = create_matrix(rows, cols, type, 0);
//     cv::Mat horizontal_detail = create_matrix(rows, cols, type, 1);
//     cv::Mat vertical_detail = create_matrix(rows, cols, type, 2);
//     cv::Mat diagonal_detail = create_matrix(rows, cols, type, 3);

//     return Dwt2dLevelCoeffs(approx, horizontal_detail, vertical_detail, diagonal_detail);
// }


// class Dwt2dResultsTest : public testing::Test {
// protected:
//     void SetUp() override {
//         int rows = 32;
//         int cols = 16;
//         int type = CV_32F;

//         coeffs = Dwt2dResults::Coefficients{
//             create_level_coeffs(rows, cols, type),
//             create_level_coeffs(rows / 2, cols / 2, type),
//             create_level_coeffs(rows / 4, cols / 4, type),
//             create_level_coeffs(rows / 8, cols / 8, type),
//             create_level_coeffs(rows / 16, cols / 16, type),
//         };
        
//         coeffs_constructed_results = Dwt2dResults(coeffs);


//         expected_matrix = create_matrix(rows, cols, type);

//         auto approx_0 = cv::Mat();
//         auto vertical_0 = expected_matrix(cv::Range(16, 32), cv::Range(0, 8));
//         auto horizontal_0 = expected_matrix(cv::Range(0, 16), cv::Range(8, 16));
//         // auto horizontal_0 = expected_matrix(cv::Range(16, 32), cv::Range(0, 8));
//         // auto vertical_0 = expected_matrix(cv::Range(0, 16), cv::Range(8, 16));
//         auto diagonal_0 = expected_matrix(cv::Range(16, 32), cv::Range(8, 16));

//         auto approx_1 = cv::Mat();
//         auto vertical_1 = expected_matrix(cv::Range(8, 16), cv::Range(0, 4));
//         auto horizontal_1 = expected_matrix(cv::Range(0, 8), cv::Range(4, 8));
//         // auto horizontal_1 = expected_matrix(cv::Range(8, 16), cv::Range(0, 4));
//         // auto vertical_1 = expected_matrix(cv::Range(0, 8), cv::Range(4, 8));
//         auto diagonal_1 = expected_matrix(cv::Range(8, 16), cv::Range(4, 8));

//         auto approx_2 = cv::Mat();
//         auto vertical_2 = expected_matrix(cv::Range(4, 8), cv::Range(0, 2));
//         auto horizontal_2 = expected_matrix(cv::Range(0, 4), cv::Range(2, 4));
//         // auto horizontal_2 = expected_matrix(cv::Range(4, 8), cv::Range(0, 2));
//         // auto vertical_2 = expected_matrix(cv::Range(0, 4), cv::Range(2, 4));
//         auto diagonal_2 = expected_matrix(cv::Range(4, 8), cv::Range(2, 4));

//         auto approx_3 = expected_matrix(cv::Range(0, 2), cv::Range(0, 1));
//         auto vertical_3 = expected_matrix(cv::Range(2, 4), cv::Range(0, 1));
//         auto horizontal_3 = expected_matrix(cv::Range(0, 2), cv::Range(1, 2));
//         // auto horizontal_3 = expected_matrix(cv::Range(2, 4), cv::Range(0, 1));
//         // auto vertical_3 = expected_matrix(cv::Range(0, 2), cv::Range(1, 2));
//         auto diagonal_3 = expected_matrix(cv::Range(2, 4), cv::Range(1, 2));

//         level0 = Dwt2dLevelCoeffs(approx_0, horizontal_0, vertical_0, diagonal_0);
//         level1 = Dwt2dLevelCoeffs(approx_1, horizontal_1, vertical_1, diagonal_1);
//         level2 = Dwt2dLevelCoeffs(approx_2, horizontal_2, vertical_2, diagonal_2);
//         level3 = Dwt2dLevelCoeffs(approx_3, horizontal_3, vertical_3, diagonal_3);

//         results.coeffs = {level0, level1, level2, level3};
//     }

//     void TearDown() override {
//     }

//     Dwt2dResults default_results;
//     Dwt2dResults coeffs_constructed_results;
//     Dwt2dResults::Coefficients coeffs;
//     Dwt2dResults results;
//     cv::Mat expected_matrix;
//     Dwt2dLevelCoeffs level0;
//     Dwt2dLevelCoeffs level1;
//     Dwt2dLevelCoeffs level2;
//     Dwt2dLevelCoeffs level3;
// };


// TEST_F(Dwt2dResultsTest, DefaultConstructor) {
//     ASSERT_TRUE(default_results.coeffs.empty());
// }

// TEST_F(Dwt2dResultsTest, CoeffsConstructor) {
//     ASSERT_EQ(coeffs_constructed_results.levels(), coeffs.size());
//     ASSERT_TRUE(matrix_equals(coeffs_constructed_results.approx(), coeffs.back().approx()));
// }

// TEST_F(Dwt2dResultsTest, details) {
//     //  horizontal
//     auto horizontal_details = results.details(HORIZONTAL);
//     ASSERT_EQ(horizontal_details.size(), 4);
//     ASSERT_TRUE(matrix_equals(horizontal_details[0], level0.horizontal_detail()));
//     ASSERT_TRUE(matrix_equals(horizontal_details[1], level1.horizontal_detail()));
//     ASSERT_TRUE(matrix_equals(horizontal_details[2], level2.horizontal_detail()));
//     ASSERT_TRUE(matrix_equals(horizontal_details[3], level3.horizontal_detail()));

//     auto horizontal_details2 = results.horizontal_details();
//     ASSERT_EQ(horizontal_details2.size(), 4);
//     ASSERT_TRUE(matrix_equals(horizontal_details2[0], level0.horizontal_detail()));
//     ASSERT_TRUE(matrix_equals(horizontal_details2[1], level1.horizontal_detail()));
//     ASSERT_TRUE(matrix_equals(horizontal_details2[2], level2.horizontal_detail()));
//     ASSERT_TRUE(matrix_equals(horizontal_details2[3], level3.horizontal_detail()));
    
//     //  vertical
//     auto vertical_details = results.details(VERTICAL);
//     ASSERT_EQ(vertical_details.size(), 4);
//     ASSERT_TRUE(matrix_equals(vertical_details[0], level0.vertical_detail()));
//     ASSERT_TRUE(matrix_equals(vertical_details[1], level1.vertical_detail()));
//     ASSERT_TRUE(matrix_equals(vertical_details[2], level2.vertical_detail()));
//     ASSERT_TRUE(matrix_equals(vertical_details[3], level3.vertical_detail()));

//     auto vertical_details2 = results.vertical_details();
//     ASSERT_EQ(vertical_details2.size(), 4);
//     ASSERT_TRUE(matrix_equals(vertical_details2[0], level0.vertical_detail()));
//     ASSERT_TRUE(matrix_equals(vertical_details2[1], level1.vertical_detail()));
//     ASSERT_TRUE(matrix_equals(vertical_details2[2], level2.vertical_detail()));
//     ASSERT_TRUE(matrix_equals(vertical_details2[3], level3.vertical_detail()));

//     //  diagonal
//     auto diagonal_details = results.details(DIAGONAL);
//     ASSERT_EQ(diagonal_details.size(), 4);
//     ASSERT_TRUE(matrix_equals(diagonal_details[0], level0.diagonal_detail()));
//     ASSERT_TRUE(matrix_equals(diagonal_details[1], level1.diagonal_detail()));
//     ASSERT_TRUE(matrix_equals(diagonal_details[2], level2.diagonal_detail()));
//     ASSERT_TRUE(matrix_equals(diagonal_details[3], level3.diagonal_detail()));

//     auto diagonal_details2 = results.diagonal_details();
//     ASSERT_EQ(diagonal_details2.size(), 4);
//     ASSERT_TRUE(matrix_equals(diagonal_details2[0], level0.diagonal_detail()));
//     ASSERT_TRUE(matrix_equals(diagonal_details2[1], level1.diagonal_detail()));
//     ASSERT_TRUE(matrix_equals(diagonal_details2[2], level2.diagonal_detail()));
//     ASSERT_TRUE(matrix_equals(diagonal_details2[3], level3.diagonal_detail()));
// }

// TEST_F(Dwt2dResultsTest, as_matrix) {
//     ASSERT_TRUE(matrix_equals(expected_matrix, results.as_matrix()));
// }



/**
 * -----------------------------------------------------------------------------
 * DWT2D
 * -----------------------------------------------------------------------------
*/

// class DWT2D {
// public:
//     DWT2D(const Wavelet& wavelet);
    
//     Dwt2dResults operator()(cv::InputArray x, int max_levels=0) const;
//     Dwt2dLevelCoeffs compute_single_level(cv::InputArray x) const;

// public:
//     const Wavelet& wavelet;

// protected:
//     cv::Mat convolve_rows_and_decimate_cols(const cv::Mat& data, cv::InputArray kernel) const;
//     cv::Mat convolve_cols_and_decimate_rows(const cv::Mat& data, cv::InputArray kernel) const;
//     cv::Mat convolve_and_decimate(
//         const cv::Mat& data, 
//         cv::InputArray kernel_x, 
//         cv::InputArray kernel_y, 
//         int final_rows, 
//         int final_cols
//     ) const;
// };


TEST(DWT2D, single_level_zeros) {
    DaubechiesWavelet wavelet(2);
    DWT2D dwt(wavelet);

    cv::Mat input(32, 32, CV_32F, 0.0);
    cv::Mat expected_approx(16, 16, CV_32F, 0.0);
    cv::Mat expected_horizontal_detail(16, 16, CV_32F, 0.0);
    cv::Mat expected_vertical_detail(16, 16, CV_32F, 0.0);
    cv::Mat expected_diagonal_detail(16, 16, CV_32F, 0.0);
    
    auto coeffs = dwt.compute_single_level(input);

    // std::cout << coeffs.approx().size() << std::endl;
    // std::cout << coeffs.horizontal_detail().size() << std::endl;
    // std::cout << coeffs.vertical_detail().size() << std::endl;
    // std::cout << coeffs.diagonal_detail().size() << std::endl;
    
    ASSERT_TRUE(matrix_equals(coeffs.approx(), expected_approx));
    ASSERT_TRUE(matrix_equals(coeffs.horizontal_detail(), expected_horizontal_detail));
    ASSERT_TRUE(matrix_equals(coeffs.vertical_detail(), expected_vertical_detail));
    ASSERT_TRUE(matrix_equals(coeffs.diagonal_detail(), expected_diagonal_detail));
}


TEST(DWT2D, zeros) {
    DaubechiesWavelet wavelet(2);

    DWT2D dwt(wavelet);

    cv::Mat input(32, 32, CV_32F, 0.0);
    cv::Mat expected_output(32, 32, CV_32F, 0.0);
    
    auto result = dwt(input);    
    auto actual_output = result.as_matrix();

    ASSERT_EQ(result.levels(), 5);
    ASSERT_TRUE(matrix_equals(actual_output, expected_output));
}

TEST(DWT2D, zeros_two_levels) {
    DaubechiesWavelet wavelet(2);

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

    ASSERT_EQ(result.levels(), 2);
    ASSERT_TRUE(matrix_equals(result.as_matrix(), expected_matrix));

    auto coeffs_0 = result.level_coeffs(0);
    ASSERT_TRUE(matrix_equals(coeffs_0.approx(), expected_approx_0));
    ASSERT_TRUE(matrix_equals(coeffs_0.horizontal_detail(), expected_horizontal_detail_0));
    ASSERT_TRUE(matrix_equals(coeffs_0.vertical_detail(), expected_vertical_detail_0));
    ASSERT_TRUE(matrix_equals(coeffs_0.diagonal_detail(), expected_diagonal_detail_0));

    auto coeffs_1 = result.level_coeffs(1);
    ASSERT_TRUE(matrix_equals(coeffs_1.approx(), expected_approx_1));
    ASSERT_TRUE(matrix_equals(coeffs_1.horizontal_detail(), expected_horizontal_detail_1));
    ASSERT_TRUE(matrix_equals(coeffs_1.vertical_detail(), expected_vertical_detail_1));
    ASSERT_TRUE(matrix_equals(coeffs_1.diagonal_detail(), expected_diagonal_detail_1));
}


TEST(DWT2D, from_files) {
    // std::cout << std::filesystem::current_path() << std::endl;

    std::ifstream manifest_file("/home/chris/projects/wavelet/tests/cases_manifest.json");
    json manifest = json::parse(manifest_file);

    // std::cout << manifest << std::endl;

    std::filesystem::path test_case_root = "/home/chris/projects/wavelet/tests";

    for (auto entry : manifest) {
        // std::cout << entry << std::endl;
        DaubechiesWavelet wavelet(1);
        DWT2D dwt(wavelet);

        std::filesystem::path case_path = test_case_root / entry["path"];
        int flags = entry["flags"];
        auto input_filename = case_path / entry["input_filename"];
        auto coeffs_filename = case_path / entry["coeffs_filename"];

        std::cout << input_filename << std::endl;
        std::cout << coeffs_filename << std::endl;
        std::cout << flags << std::endl;

        // flags = cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR;
        flags = cv::IMREAD_ANYCOLOR;
        
        auto input = cv::imread(input_filename, flags);
        auto expected_coeffs = cv::imread(coeffs_filename);

        std::cout << expected_coeffs.size() << "  " << expected_coeffs.type() << std::endl;

        ASSERT_TRUE(false);    
        // auto results = dwt(input);
        // auto actual_coeffs = results.as_matrix();

        // std::cout << actual_coeffs.size() << "  " << actual_coeffs.type() << std::endl;

        // std::cout << "-----------------------------" << std::endl;
        // ASSERT_TRUE(matrix_equals(actual_coeffs, expected_coeffs));
    }

    ASSERT_TRUE(false);
}



























#endif


