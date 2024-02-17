#ifndef WAVELET_WAVELET_H
#define WAVELET_WAVELET_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <functional>
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

/**
 * -----------------------------------------------------------------------------
*/
enum FilterSubband {
    APPROXIMATION = 0,
    HORIZONTAL = 1,
    VERTICAL = 2,
    DIAGONAL = 3,
};

enum NormalizationMode {
    DWT_NORMALIZE_NONE = 0,
    DWT_NORMALIZE_ZERO_TO_HALF,
    DWT_NORMALIZE_MAX,
};


namespace internal
{
class WaveletFilterBank
{
public:
    class KernelPair
    {
    public:
        KernelPair(const cv::Mat& lowpass, const cv::Mat& highpass) :
            _lowpass(lowpass),
            _highpass(highpass)
        {}

        template <typename T>
        KernelPair(const std::vector<T>& lowpass, const std::vector<T>& highpass) :
            _lowpass(cv::Mat(lowpass, true)),
            _highpass(cv::Mat(highpass, true))
        {}

        template <typename T, int N>
        KernelPair(const std::array<T, N>& lowpass, const std::array<T, N>& highpass) :
            _lowpass(cv::Mat(lowpass, true)),
            _highpass(cv::Mat(highpass, true))
        {}

        const cv::Mat& lowpass() const { return _lowpass; }
        const cv::Mat& highpass() const { return _highpass; }

        KernelPair flipped() const;
    private:
        cv::Mat _lowpass;
        cv::Mat _highpass;
    };

    WaveletFilterBank(
        const KernelPair& synthesis_kernels,
        const KernelPair& analysis_kernels
    );
    WaveletFilterBank(
        const cv::Mat& synthesis_lowpass,
        const cv::Mat& synthesis_highpass,
        const cv::Mat& analysis_lowpass,
        const cv::Mat& analysis_highpass
    );
    WaveletFilterBank() = default;
    WaveletFilterBank(const WaveletFilterBank& other) = default;
    WaveletFilterBank(WaveletFilterBank&& other) = default;

    void forward(
        cv::InputArray x,
        cv::OutputArray approx,
        cv::OutputArray horizontal_detail,
        cv::OutputArray vertical_detail,
        cv::OutputArray diagonal_detail,
        int border_type=cv::BORDER_DEFAULT
    ) const;

    cv::Mat forward_stage1_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void forward_stage1_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat forward_stage1_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void forward_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat forward_stage2_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void forward_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat forward_stage2_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void forward_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    void inverse(
        cv::InputArray approx,
        cv::InputArray horizontal_detail,
        cv::InputArray vertical_detail,
        cv::InputArray diagonal_detail,
        cv::OutputArray output,
        int border_type=cv::BORDER_DEFAULT
    ) const;

    cv::Mat inverse_stage1_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage1_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat inverse_stage1_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage1_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat inverse_stage2_lowpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage2_lowpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    cv::Mat inverse_stage2_highpass(cv::InputArray data, int border_type=cv::BORDER_DEFAULT) const;
    void inverse_stage2_highpass(cv::InputArray data, cv::OutputArray output, int border_type=cv::BORDER_DEFAULT) const;

    template <typename T>
    static KernelPair build_analysis_kernels(const std::vector<T>& coeffs);
    template <typename T>
    static KernelPair build_synthesis_kernels(const std::vector<T>& coeffs);

    template <typename T>
    static void negate_odds(std::vector<T>& coeffs);
    template <typename T>
    static void negate_evens(std::vector<T>& coeffs);

    KernelPair synthesis_kernels() const { return _synthesis_kernels.flipped(); }
    KernelPair analysis_kernels() const { return _analysis_kernels.flipped(); }

protected:
    void downsample_rows(cv::InputArray data, cv::OutputArray output) const;
    void downsample_cols(cv::InputArray data, cv::OutputArray output) const;
    void upsample_rows(cv::InputArray data, cv::OutputArray output) const;
    void upsample_cols(cv::InputArray data, cv::OutputArray output) const;
    void filter_rows(cv::InputArray data, cv::OutputArray output, const cv::Mat& kernel, int border_type) const;
    void filter_cols(cv::InputArray data, cv::OutputArray output, const cv::Mat& kernel, int border_type) const;

    KernelPair _synthesis_kernels;
    KernelPair _analysis_kernels;
};


/*
    |----------------------------------|----------------------------------|
    |  2a   |  2v    |                 |                                  |
    |-------+--------|      1v         |                                  |
    |  2h   |  2d    |                 |                                  |
    -----------------+-----------------|               0v                 |
    |                |                 |                                  |
    |      1h        |      1d         |                                  |
    |                |                 |                                  |
    |                |                 |                                  |
    |----------------------------------+----------------------------------|
    |                                  |                                  |
    |                                  |                                  |
    |                                  |                                  |
    |               0h                 |               0d                 |
    |                                  |                                  |
    |                                  |                                  |
    |                                  |                                  |
    |                                  |                                  |
    |----------------------------------|----------------------------------|
*/
// class Dwt2dCoeffs
// {
// public:
//     class LevelIterator
//     {
//     public:
//         using value_type = Dwt2dCoeffs;
//         using iterator_category = std::bidirectional_iterator_tag;

//         LevelIterator(const Dwt2dCoeffs* coeffs, int level) : coeffs(coeffs), level(level)
//         {}

//         value_type operator*() const
//         {
//             return coeffs->at(level);
//         }

//         LevelIterator& operator++()
//         {
//             ++level;
//             return *this;
//         }
//         LevelIterator operator++(int)
//         {
//             auto copy = *this;
//             ++*this;
//             return copy;
//         }

//         LevelIterator& operator--()
//         {
//             --level;
//             return *this;
//         }
//         LevelIterator operator--(int)
//         {
//             auto copy = *this;
//             --*this;
//             return copy;
//         }

//         bool operator==(const LevelIterator& other) const
//         {
//             return coeffs == other.coeffs && level == other.level;
//         }

//     private:
//         const Dwt2dCoeffs* coeffs;
//         int level;
//     };

// public:
//     Dwt2dCoeffs();
//     Dwt2dCoeffs(const cv::Mat& matrix);
//     Dwt2dCoeffs(cv::Mat&& matrix);
//     Dwt2dCoeffs(const cv::Mat& matrix, int depth);
//     Dwt2dCoeffs(int rows, int cols, int type, int depth=-1);
//     Dwt2dCoeffs(const cv::Size& size, int type, int depth=-1);
//     Dwt2dCoeffs(const Dwt2dCoeffs& other) = default;
//     Dwt2dCoeffs(Dwt2dCoeffs&& other) = default;

//     Dwt2dCoeffs& operator=(const Dwt2dCoeffs& coeffs) = default;
//     Dwt2dCoeffs& operator=(Dwt2dCoeffs&& coeffs) = default;
//     Dwt2dCoeffs& operator=(const cv::Mat& matrix);
//     Dwt2dCoeffs& operator=(const cv::MatExpr& matrix);
//     Dwt2dCoeffs& operator=(const cv::Scalar& scalar);
//     Dwt2dCoeffs& operator=(cv::Mat&& matrix);

//     operator cv::Mat() const { return _coeff_matrix; }
//     operator cv::_InputArray() const { return _coeff_matrix; }
//     operator cv::_OutputArray() const { return _coeff_matrix; }
//     operator cv::_InputOutputArray() const { return _coeff_matrix; }

//     Dwt2dCoeffs clone() const;

//     cv::Mat approx() const;
//     cv::Mat detail(int direction, int level=0) const;
//     cv::Mat horizontal_detail(int level=0) const;
//     cv::Mat vertical_detail(int level=0) const;
//     cv::Mat diagonal_detail(int level=0) const;

//     void set_level(const cv::Mat& coeffs, int level=0);
//     void set_level(const cv::Scalar& scalar, int level=0);
//     void set_approx(const cv::Mat& coeffs);
//     void set_approx(const cv::Scalar& scalar);
//     void set_detail(const cv::Mat& coeffs, int direction, int level=0);
//     void set_detail(const cv::Scalar& scalar, int direction, int level=0);
//     void set_horizontal_detail(const cv::Mat& coeffs, int level=0);
//     void set_horizontal_detail(const cv::Scalar& scalar, int level=0);
//     void set_vertical_detail(const cv::Mat& coeffs, int level=0);
//     void set_vertical_detail(const cv::Scalar& scalar, int level=0);
//     void set_diagonal_detail(const cv::Mat& coeffs, int level=0);
//     void set_diagonal_detail(const cv::Scalar& scalar, int level=0);

//     Dwt2dCoeffs at(int level) const;
//     Dwt2dCoeffs operator[](int level) const { return at(level); }

//     std::vector<cv::Mat> collect_details(int direction) const;
//     std::vector<cv::Mat> collect_horizontal_details() const { return collect_details(HORIZONTAL); }
//     std::vector<cv::Mat> collect_vertical_details() const { return collect_details(VERTICAL); }
//     std::vector<cv::Mat> collect_diagonal_details() const { return collect_details(DIAGONAL); }

//     int depth() const { return _depth; }
//     int rows() const { return _coeff_matrix.rows; }
//     int cols() const { return _coeff_matrix.cols; }
//     cv::Size size() const { return _coeff_matrix.size(); }
//     int type() const { return _coeff_matrix.type(); }
//     bool empty() const { return _coeff_matrix.empty(); }

//     LevelIterator begin() const;
//     LevelIterator end() const;
//     LevelIterator begin();
//     LevelIterator end();

//     void normalize(int approx_mode=DWT_NORMALIZE_MAX, int detail_mode=DWT_NORMALIZE_ZERO_TO_HALF);
//     double maximum_abs_value() const;

//     cv::Size level_size(int level) const;
//     cv::Rect level_rect(int level) const;
//     cv::Size detail_size(int level) const;
//     cv::Rect approx_rect(int level) const;
//     cv::Rect horizontal_rect(int level) const;
//     cv::Rect vertical_rect(int level) const;
//     cv::Rect diagonal_rect(int level) const;

//     bool shares_data(const Dwt2dCoeffs& other) const;
//     bool shares_data(const cv::Mat& matrix) const;

// protected:
//     template <typename MatrixLike>
//     void check_size_for_assignment(const MatrixLike& matrix) const;

//     template <typename MatrixLike>
//     void check_size_for_set_level(const MatrixLike& matrix, int level) const;

//     template <typename MatrixLike>
//     void check_size_for_set_detail(const MatrixLike& matrix, int level) const;

//     std::pair<double, double> normalization_constants(int normalization_mode, double max_abs_value) const;

//     void convert_and_copy(const cv::Mat& source, const cv::Mat& destination);

// private:
//     cv::Mat _coeff_matrix;
//     int _depth;
// };
} // namespace internal



class Wavelet
{
public:
    using FilterBank = internal::WaveletFilterBank;
    enum class Symmetry {
        SYMMETRIC,
        NEAR_SYMMETRIC,
        ASYMMETRIC,
    };

protected:
    struct WaveletImpl
    {
        int order;
        int vanising_moments_psi;
        int vanising_moments_phi;
        int support_width;
        bool orthogonal;
        bool biorthogonal;
        Symmetry symmetry;
        bool compact_support;
        std::string family_name;
        std::string short_name;
        FilterBank filter_bank;
    };

public:
    Wavelet(
        int order,
        int vanising_moments_psi,
        int vanising_moments_phi,
        int support_width,
        bool orthogonal,
        bool biorthogonal,
        Symmetry symmetry,
        bool compact_support,
        const std::string& family_name,
        const std::string& short_name,
        const FilterBank& filter_bank
    );
    Wavelet() = delete;

    int order() const { return _p->order; }
    int vanising_moments_psi() const { return _p->vanising_moments_psi; }
    int vanising_moments_phi() const { return _p->vanising_moments_phi; }
    int support_width() const { return _p->support_width; }
    bool orthogonal() const { return _p->orthogonal; }
    bool biorthogonal() const { return _p->biorthogonal; }
    Symmetry symmetry() const { return _p->symmetry; }
    bool compact_support() const { return _p->compact_support; }
    std::string family_name() const { return _p->family_name; }
    std::string short_name() const { return _p->short_name; }
    const FilterBank& filter_bank() const { return _p->filter_bank; }

    static Wavelet create(const std::string& name);
    template<class... Args>
    static void register_factory(const std::string& name, Wavelet factory(Args...), const Args&... args);
    static std::vector<std::string> registered_wavelets();

private:
    std::shared_ptr<WaveletImpl> _p;
    static std::map<std::string, std::function<Wavelet()>> _wavelet_factories;
};


/**
 * Wavelt factories
*/
Wavelet daubechies(int order);
Wavelet haar();




/**
 * -----------------------------------------------------------------------------
*/
// class DWT2D {
// public:
//     using Coeffs = internal::Dwt2dCoeffs;

// public:
//     DWT2D(const Wavelet& wavelet, int border_type=cv::BORDER_DEFAULT);
//     DWT2D() = delete;
//     DWT2D(const DWT2D& other) = default;

//     Coeffs operator()(cv::InputArray x, int levels=0) const;

//     Coeffs forward(cv::InputArray x, int levels=0) const;
//     void inverse(const Coeffs& coeffs, cv::OutputArray output, int levels=0) const;
//     cv::Mat inverse(const Coeffs& coeffs, int levels=0) const;

//     static int max_possible_depth(cv::InputArray x);
//     static int max_possible_depth(int rows, int cols);
//     static int max_possible_depth(const cv::Size& size);

// public:
//     Wavelet wavelet;
//     int border_type;
// };


// DWT2D::Coeffs dwt2d(
//     cv::InputArray input,
//     const Wavelet& wavelet,
//     int levels=0,
//     int border_type=cv::BORDER_DEFAULT
// );

// DWT2D::Coeffs dwt2d(
//     cv::InputArray input,
//     const std::string& wavelet,
//     int levels=0,
//     int border_type=cv::BORDER_DEFAULT
// );

// void dwt2d(
//     cv::InputArray input,
//     cv::OutputArray output,
//     const Wavelet& wavelet,
//     int levels=0,
//     int border_type=cv::BORDER_DEFAULT
// );

// void dwt2d(
//     cv::InputArray input,
//     cv::OutputArray output,
//     const std::string& wavelet,
//     int levels=0,
//     int border_type=cv::BORDER_DEFAULT
// );

// void idwt2d(
//     const DWT2D::Coeffs& input,
//     cv::OutputArray output,
//     const Wavelet& wavelet,
//     int levels=0,
//     int border_type=cv::BORDER_DEFAULT
// );

// void idwt2d(
//     const DWT2D::Coeffs& input,
//     cv::OutputArray output,
//     const std::string& wavelet,
//     int levels=0,
//     int border_type=cv::BORDER_DEFAULT
// );

// cv::Mat idwt2d(
//     const DWT2D::Coeffs& input,
//     const Wavelet& wavelet,
//     int levels=0,
//     int border_type=cv::BORDER_DEFAULT
// );

// cv::Mat idwt2d(
//     const DWT2D::Coeffs& input,
//     const std::string& wavelet,
//     int levels=0,
//     int border_type=cv::BORDER_DEFAULT
// );

#endif
