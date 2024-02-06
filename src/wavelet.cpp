/**
 * 
*/
#include "wavelet/wavelet.h"
#include <memory>
#include <opencv2/imgproc.hpp>

/**
 * -----------------------------------------------------------------------------
 * Wavelet
 * -----------------------------------------------------------------------------
*/
Wavelet::Wavelet(
        int order,
        int vanising_moments_psi,
        int vanising_moments_phi,
        int support_width,
        bool orthogonal,
        bool biorthogonal,
        bool symmetry,
        bool compact_support,
        const std::string& family_name,
        const std::string& short_name,
        const std::vector<double>& analysis_filter_coeffs,
        const std::vector<double>& synthesis_filter_coeffs
    ) :
    order(order),
    vanising_moments_psi(vanising_moments_psi),
    vanising_moments_phi(vanising_moments_phi),
    support_width(support_width),
    orthogonal(orthogonal),
    biorthogonal(biorthogonal),
    symmetry(symmetry),
    compact_support(compact_support),
    family_name(family_name),
    short_name(short_name),
    _analysis_filter_coeffs(analysis_filter_coeffs),
    _synthesis_filter_coeffs(synthesis_filter_coeffs),
    _highpass_analysis_filter(),
    _lowpass_analysis_filter(),
    _highpass_synthesis_filter(),
    _lowpass_synthesis_filter()
{
    // build_filter_bank(analysis_filter_coeffs, synthesis_filter_coeffs);
}

void Wavelet::build_filter_bank()
{
    _lowpass_synthesis_filter = _synthesis_filter_coeffs;
    std::vector<double> _lowpass_analysis_filter(_synthesis_filter_coeffs.size());
    std::reverse_copy(
        _synthesis_filter_coeffs.cbegin(), 
        _synthesis_filter_coeffs.cend(), 
        _lowpass_analysis_filter.begin()
    );

    _highpass_synthesis_filter = alternate_signs(_lowpass_analysis_filter);
    _highpass_analysis_filter = alternate_signs(_lowpass_synthesis_filter);

    // for(i = 0; i < w->rec_len; ++i){
    //     w->rec_lo_float[i] = db_float[coeffs_idx][i];
    //     w->dec_lo_float[i] = db_float[coeffs_idx][w->dec_len-1-i];
    //     w->rec_hi_float[i] = ((i % 2) ? -1 : 1)
    //         * db_float[coeffs_idx][w->dec_len-1-i];
    //     w->dec_hi_float[i] = (((w->dec_len-1-i) % 2) ? -1 : 1)
    //         * db_float[coeffs_idx][i];
    // }
}

std::vector<double> Wavelet::alternate_signs(const std::vector<double> &filter_coeffs) const
{
    std::vector<double> result(filter_coeffs.size());
    for (int i = 0; i < result.size(); ++i)
        result[i] = ((i % 2) ? -1 : 1) * filter_coeffs[i];
    
    return result;
}

DaubechiesWavelet::DaubechiesWavelet(int order) :
    Wavelet {
        order, // order
        2 * order, // vanising_moments_psi
        0, // vanising_moments_phi
        2 * order - 1, // support_width
        true, // orthogonal
        true, // biorthogonal
        true, // symmetry
        true, // compact_support
        "Daubechies", // family_name
        "db", // short_name
        ORDER_FILTER_COEFFS[order], // analysis_filter_coeffs
        ORDER_FILTER_COEFFS[order], // synthesis_filter_coeffs
    }
{   
}




/**
 * -----------------------------------------------------------------------------
 * Dwt2dResults
 * -----------------------------------------------------------------------------
*/
Dwt2dResults::Dwt2dResults(const Wavelet &wavelet) :
    wavelet(wavelet),
    _level_coeffs()
{
}

void Dwt2dResults::push_back(const Dwt2dLevelCoeffs &level_coeffs)
{
    _level_coeffs.push_back(level_coeffs);
}

const cv::Mat &Dwt2dResults::approx() const
{
    return _level_coeffs.back().approx();
}

std::vector<const cv::Mat &> Dwt2dResults::details(DWT2DCoeffCategory direction) const
{
    std::vector<const cv::Mat &> result(levels());
    std::transform(
        _level_coeffs.begin(),
        _level_coeffs.end(),
        result.begin(), 
        result.end(), 
        [](auto level) { level.coeffs(direction); }
    );

    return result;
}

cv::Mat Dwt2dResults::as_matrix() const
{
    int rows = _level_coeffs[0].rows();
    int cols = _level_coeffs[0].cols();
    int type = _level_coeffs[0].type();

    cv::Mat result(2 * rows, 2 * cols, type);
    for (auto level_coeffs : _level_coeffs) {
        cv::Mat approx_submatrix(
            result, 
            cv::Range(0, rows),
            cv::Range(0, cols)
        );
        approx_submatrix.setTo(level_coeffs.approx());
        
        cv::Mat horizontal_submatrix(
            result, 
            cv::Range(rows - 1, 2 * rows),
            cv::Range(0, cols)
        );
        horizontal_submatrix.setTo(level_coeffs.horizontal_detail());

        cv::Mat vertical_submatrix(
            result, 
            cv::Range(0, rows),
            cv::Range(cols - 1, 2 * cols)
        );
        vertical_submatrix.setTo(level_coeffs.vertical_detail());

        cv::Mat diagonal_submatrix(
            result, 
            cv::Range(rows - 1, 2 * rows),
            cv::Range(cols - 1, 2 * cols)
        );
        diagonal_submatrix.setTo(level_coeffs.diagonal_detail());
    }

    return result;
}



/**
 * -----------------------------------------------------------------------------
 * Dwt2dLevelCoeffs
 * -----------------------------------------------------------------------------
*/
Dwt2dLevelCoeffs::Dwt2dLevelCoeffs() :
    _coeffs{cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat()}
{
}

Dwt2dLevelCoeffs::Dwt2dLevelCoeffs(
        const cv::Mat &approx, 
        const cv::Mat &horizontal_detail, 
        const cv::Mat &vertical_detail, 
        const cv::Mat &diagonal_detail
    ) :
    _coeffs{approx, horizontal_detail, vertical_detail, diagonal_detail}
{
}

Dwt2dLevelCoeffs::Dwt2dLevelCoeffs(const Coefficients &coeffs) :
    _coeffs{coeffs}
{
}

Dwt2dLevelCoeffs::Dwt2dLevelCoeffs(const Dwt2dLevelCoeffs &coeffs) :
    _coeffs(coeffs._coeffs)
{
}

int Dwt2dLevelCoeffs::rows() const 
{
    auto first_nonempty = find_first_nonempty();
    return first_nonempty.empty() ? 0 : first_nonempty.rows;
}

int Dwt2dLevelCoeffs::cols() const 
{
    auto first_nonempty = find_first_nonempty();
    return first_nonempty.empty() ? 0 : first_nonempty.cols;
}

int Dwt2dLevelCoeffs::type() const 
{
    auto first_nonempty = find_first_nonempty();
    return first_nonempty.empty() ? 0 : first_nonempty.type();
}

cv::Mat Dwt2dLevelCoeffs::find_first_nonempty() const
{
    for (auto coeff : _coeffs) {
        if (!coeff.empty())
            return coeff;
    }
    return cv::Mat();
}




/**
 * -----------------------------------------------------------------------------
 * DWT2D
 * -----------------------------------------------------------------------------
*/
DWT2D::DWT2D(const Wavelet& wavelet) :
    wavelet(wavelet)
{
}

Dwt2dResults DWT2D::operator()(cv::InputArray x, int max_levels) const 
{
    Dwt2dResults result(wavelet);
    auto data = x.getMat();

    // int max_levels = 9;
    for (int level = 0; level < max_levels; ++level) {
        auto coeffs = compute_single_level(data);
        result.push_back(coeffs);
        data = coeffs.approx();
    }

    return result;
}

Dwt2dLevelCoeffs DWT2D::compute_single_level(cv::InputArray x) const
{
    auto data = x.getMat();
    auto lowpass_kernel = wavelet.analysis_lowpass_coeffs();
    auto highpass_kernel = wavelet.analysis_highpass_coeffs();

    auto approx = convolve_rows_and_decimate_cols(data, lowpass_kernel);
    auto detail = convolve_rows_and_decimate_cols(data, highpass_kernel);

    return Dwt2dLevelCoeffs{
        convolve_cols_and_decimate_rows(approx, lowpass_kernel),
        convolve_cols_and_decimate_rows(approx, highpass_kernel),
        convolve_cols_and_decimate_rows(detail, lowpass_kernel),
        convolve_cols_and_decimate_rows(detail, highpass_kernel),
    };
}

cv::Mat DWT2D::convolve_rows_and_decimate_cols(const cv::Mat& data, cv::InputArray kernel) const
{
    return convolve_and_decimate(
        data,
        kernel,
        std::vector<double>{1},
        data.rows / 2,
        data.cols
    );
}

cv::Mat DWT2D::convolve_cols_and_decimate_rows(const cv::Mat& data, cv::InputArray kernel) const
{
    return convolve_and_decimate(
        data,
        std::vector<double>{1},
        kernel,
        data.rows,
        data.cols / 2
    );
}

cv::Mat DWT2D::convolve_and_decimate(
        const cv::Mat& data, 
        cv::InputArray kernel_x, 
        cv::InputArray kernel_y, 
        int final_rows, 
        int final_cols
    ) const
{
    int type = data.type();
    
    auto filtered = cv::Mat(data.rows, data.cols, type);
    cv::sepFilter2D(data, filtered, -1, kernel_x, kernel_y);

    auto result = cv::Mat(final_rows, data.cols, type);
    cv::resize(filtered, result, result.size(), 0.0, 0.0, cv::INTER_NEAREST);

    return result;
}


