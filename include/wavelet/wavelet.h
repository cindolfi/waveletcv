#ifndef WAVELET_WAVELET_H
#define WAVELET_WAVELET_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <array>
#include <span>

/**
 * The plan
 * 1.  implement dwt
 * 2.  tests
 * 3.  examples
 *     a. denoise
 *     b. feature detection
 * 4.  docs
 * 5. lint
 *
 *
 * TODO
 *  inverse
 *  non 2^n size
 *  from_matrix
 *  denoise example
 *  docs
*/

enum class WaveletSymmetry {
    SYMMETRIC,
    ASYMMETRIC,
};


struct WaveletFilterBank
{
    using Coeffs = std::vector<double>;

    WaveletFilterBank(const Coeffs& lowpass, const Coeffs& highpass);
    WaveletFilterBank() = default;
    WaveletFilterBank(const WaveletFilterBank& filters);
    WaveletFilterBank(const WaveletFilterBank&& filters);

    Coeffs lowpass;
    Coeffs highpass;

    static WaveletFilterBank build_analysis_filter_bank(const Coeffs& analysis_coeffs);
    static WaveletFilterBank build_synthesis_filter_bank(const Coeffs& synthesis_coeffs);

    static void negate_odds(Coeffs& filter_coeffs);
    static void negate_evens(Coeffs& filter_coeffs);
};

struct WaveletImpl
{
    int order;
    int vanising_moments_psi;
    int vanising_moments_phi;
    int support_width;
    bool orthogonal;
    bool biorthogonal;
    WaveletSymmetry symmetry;
    bool compact_support;
    std::string family_name;
    std::string short_name;
    WaveletFilterBank analysis_coeffs;
    WaveletFilterBank synthesis_coeffs;
};

class Wavelet
{
public:
    Wavelet(
        int order,
        int vanising_moments_psi,
        int vanising_moments_phi,
        int support_width,
        bool orthogonal,
        bool biorthogonal,
        WaveletSymmetry symmetry,
        bool compact_support,
        const std::string& family_name,
        const std::string& short_name,
        const WaveletFilterBank& analysis_coeffs,
        const WaveletFilterBank& synthesis_coeffs
    );
    Wavelet() = delete;

    const WaveletFilterBank::Coeffs& analysis_lowpass_coeffs() const {
        return analysis_coeffs().lowpass;
    }
    const WaveletFilterBank::Coeffs& analysis_highpass_coeffs() const {
        return analysis_coeffs().highpass;
    }
    const WaveletFilterBank::Coeffs& synthesis_lowpass_coeffs() const {
        return synthesis_coeffs().lowpass;
    }
    const WaveletFilterBank::Coeffs& synthesis_highpass_coeffs() const {
        return synthesis_coeffs().highpass;
    }

    const WaveletFilterBank& synthesis_filter_bank() const {
        return synthesis_coeffs();
    }

    const WaveletFilterBank& analysis_filter_bank() const {
        return analysis_coeffs();
    }

    int order() const { return _p->order; }
    int vanising_moments_psi() const { return _p->vanising_moments_psi; }
    int vanising_moments_phi() const { return _p->vanising_moments_phi; }
    int support_width() const { return _p->support_width; }
    bool orthogonal() const { return _p->orthogonal; }
    bool biorthogonal() const { return _p->biorthogonal; }
    WaveletSymmetry symmetry() const { return _p->symmetry; }
    bool compact_support() const { return _p->compact_support; }
    std::string family_name() const { return _p->family_name; }
    std::string short_name() const { return _p->short_name; }
    const WaveletFilterBank& analysis_coeffs() const { return _p->analysis_coeffs; }
    const WaveletFilterBank& synthesis_coeffs() const { return _p->synthesis_coeffs; }

private:
    std::shared_ptr<WaveletImpl> _p;
};




/**
 * Wavelt factory that returns a wavelet object from a wavelet name
*/
Wavelet create_wavelet(const std::string& id);

using WaveletFactory = std::function<Wavelet()>;
void register_wavelet_factory(const std::string& name, const WaveletFactory& factory);
std::vector<std::string> registered_wavelets();

/**
 * Wavelt factories
*/
Wavelet daubechies(int order);
Wavelet haar();



/**
 * -----------------------------------------------------------------------------
*/
enum DWT2DCoeffCategory {
    APPROXIMATION = 0,
    HORIZONTAL = 1,
    VERTICAL = 2,
    DIAGONAL = 3,
};

class Dwt2dLevelCoeffs {
public:
    using CoeffArray = const std::array<cv::Mat, 4>;
public:
    Dwt2dLevelCoeffs();
    Dwt2dLevelCoeffs(int rows, int cols, int type);
    Dwt2dLevelCoeffs(
        const cv::Mat& approx,
        const cv::Mat& horizontal_detail,
        const cv::Mat& vertical_detail,
        const cv::Mat& diagonal_detail
    );
    Dwt2dLevelCoeffs(const CoeffArray& coeffs);
    Dwt2dLevelCoeffs(const Dwt2dLevelCoeffs& coeffs) = default;
    Dwt2dLevelCoeffs(Dwt2dLevelCoeffs&& coeffs) = default;
    Dwt2dLevelCoeffs& operator=(const Dwt2dLevelCoeffs& coeffs) = default;
    Dwt2dLevelCoeffs& operator=(Dwt2dLevelCoeffs&& coeffs) = default;

    void check_for_consistent_types() const;
    bool has_consistent_types() const noexcept;
    void check_for_consistent_sizes() const;
    bool has_consistent_sizes() const noexcept;

    const cv::Mat& horizontal_detail() const { return _coeffs[HORIZONTAL]; }
    cv::Mat& horizontal_detail() { return _coeffs[HORIZONTAL]; }

    const cv::Mat& vertical_detail() const { return _coeffs[VERTICAL]; }
    cv::Mat& vertical_detail() { return _coeffs[VERTICAL]; }

    const cv::Mat& diagonal_detail() const { return _coeffs[DIAGONAL]; }
    cv::Mat& diagonal_detail() { return _coeffs[DIAGONAL]; }

    const cv::Mat& approx() const { return _coeffs[APPROXIMATION]; }
    cv::Mat& approx() { return _coeffs[APPROXIMATION]; }

    std::span<cv::Mat> details() {
        return std::span<cv::Mat>(_coeffs.begin() + HORIZONTAL, 3);
    }
    std::span<cv::Mat> coeffs() {
        return std::span<cv::Mat>(_coeffs);
    }

    cv::Mat& at(std::size_t i) { return _coeffs.at(i); }
    const cv::Mat& at(std::size_t i) const { return _coeffs.at(i); }

    cv::Mat& operator[](std::size_t i) { return _coeffs[i]; }
    const cv::Mat& operator[](std::size_t i) const { return _coeffs[i]; }

    auto begin() { return _coeffs.begin(); }
    auto end() { return _coeffs.end(); }
    const auto begin() const { return _coeffs.begin(); }
    const auto end() const { return _coeffs.end(); }

    int rows() const;
    int cols() const;
    cv::Size size() const;
    int type() const;

protected:
    cv::Mat find_first_nonempty() const;

private:
    std::array<cv::Mat, 4> _coeffs;
};




/**
 * -----------------------------------------------------------------------------
*/
enum {
    DWT_NORMALIZE_NONE = 0,
    DWT_NORMALIZE_ZERO_TO_HALF,
    DWT_NORMALIZE_MAX,
};

struct Dwt2dResults {
public:
    using Coefficients = std::vector<Dwt2dLevelCoeffs>;

public:
    Dwt2dResults();
    explicit Dwt2dResults(const Coefficients& coeffs);
    Dwt2dResults(const cv::Mat& matrix, int depth=0);
    Dwt2dResults(int rows, int cols, int type, int max_depth=0);
    Dwt2dResults(const cv::Size& size, int type, int max_depth=0);

    Dwt2dResults(const Dwt2dResults& other) = default;
    Dwt2dResults(Dwt2dResults&& other) = default;
    Dwt2dResults& operator=(const Dwt2dResults& coeffs) = default;
    Dwt2dResults& operator=(Dwt2dResults&& coeffs) = default;

    bool has_consistent_sizes_accross_levels() const noexcept;
    void check_for_consistent_sizes_accross_levels() const;

    cv::Mat approx() const;
    std::vector<cv::Mat> details(int direction) const;
    std::vector<cv::Mat> horizontal_details() const { return details(HORIZONTAL); }
    std::vector<cv::Mat> vertical_details() const { return details(VERTICAL); }
    std::vector<cv::Mat> diagonal_details() const { return details(DIAGONAL); }

    cv::Mat as_matrix() const;
    operator cv::Mat() const { return as_matrix(); }

    Dwt2dLevelCoeffs& at(std::size_t level) { return coeffs.at(level); }
    const Dwt2dLevelCoeffs& at(std::size_t level) const { return coeffs.at(level); }
    Dwt2dLevelCoeffs& operator[](std::size_t level) { return coeffs[level]; }
    const Dwt2dLevelCoeffs& operator[](std::size_t level) const { return coeffs[level]; }
    std::size_t size() const { return coeffs.size(); }
    std::size_t depth() const { return size(); }
    bool empty() const { return coeffs.empty(); }

    auto begin() const { return coeffs.begin(); }
    auto end() const { return coeffs.end(); }
    auto begin() { return coeffs.begin(); }
    auto end() { return coeffs.end(); }

    void normalize(int approx_mode=DWT_NORMALIZE_MAX, int detail_mode=DWT_NORMALIZE_ZERO_TO_HALF);
    double maximum_abs_value() const;

    cv::Rect approx_roi() const;
    cv::Rect horizontal_roi(int level) const;
    cv::Rect vertical_roi(int level) const;
    cv::Rect diagonal_roi(int level) const;

    static cv::Rect approx_rect(const cv::Size& size, int level);
    static cv::Rect horizontal_rect(const cv::Size& size, int level);
    static cv::Rect vertical_rect(const cv::Size& size, int level);
    static cv::Rect diagonal_rect(const cv::Size& size, int level);

    // static void approx_submatrix(cv::InputArray matrix, cv::InputArray output, int level);
    // static void horizontal_submatrix(cv::InputArray matrix, cv::InputArray output, int level);
    // static void vertical_submatrix(cv::InputArray matrix, cv::InputArray output, int level);
    // static void diagonal_submatrix(cv::InputArray matrix, cv::InputArray output, int level);

protected:
    std::pair<double, double> normalization_constants(int normalization_mode, double max_abs_value) const;


private:
    Coefficients coeffs;
};




/**
 * -----------------------------------------------------------------------------
*/
class DWT2D {
public:
    DWT2D(const Wavelet& wavelet, int border_type=cv::BORDER_DEFAULT);
    DWT2D() = delete;
    DWT2D(const DWT2D& other) = default;

    Dwt2dResults operator()(cv::InputArray x, int max_depth=0) const;

    Dwt2dResults forward(cv::InputArray x, int max_depth=0) const;
    Dwt2dLevelCoeffs forward_single_level(cv::InputArray x) const;

    void inverse(const Dwt2dResults& coeffs, cv::OutputArray output) const;
    void inverse_single_level(const Dwt2dLevelCoeffs& x, cv::OutputArray output) const;
    void inverse_single_level(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray output
    ) const;

    static int max_possible_depth(cv::InputArray x);

public:
    Wavelet wavelet;
    int border_type;

protected:
    cv::Mat convolve_rows_and_downsample_cols(const cv::Mat& data, cv::InputArray kernel) const;
    cv::Mat convolve_cols_and_downsample_rows(const cv::Mat& data, cv::InputArray kernel) const;
    cv::Mat convolve_and_downsample(
        const cv::Mat& data,
        cv::InputArray kernel_x,
        cv::InputArray kernel_y,
        int final_rows,
        int final_cols
    ) const;

    cv::Mat convolve_rows_and_upsample_cols(const cv::Mat& data, cv::InputArray kernel) const;
    cv::Mat convolve_cols_and_upsample_rows(const cv::Mat& data, cv::InputArray kernel) const;
    cv::Mat convolve_and_upsample(
        const cv::Mat& data,
        cv::InputArray kernel_x,
        cv::InputArray kernel_y,
        int final_rows,
        int final_cols
    ) const;
};


Dwt2dResults dwt2d(
    cv::InputArray input,
    const Wavelet& wavelet,
    int max_depth=0,
    int border_type=cv::BORDER_DEFAULT
);

Dwt2dResults dwt2d(
    cv::InputArray input,
    const std::string& wavelet,
    int max_depth=0,
    int border_type=cv::BORDER_DEFAULT
);

void idwt2d(
    const Dwt2dResults& input,
    cv::OutputArray output,
    const Wavelet& wavelet,
    int border_type=cv::BORDER_DEFAULT
);

void idwt2d(
    const Dwt2dResults& input,
    cv::OutputArray output,
    const std::string& wavelet,
    int border_type=cv::BORDER_DEFAULT
);

#endif
