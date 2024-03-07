#ifndef WAVELET_DWT2D_HPP
#define WAVELET_DWT2D_HPP

#include "wavelet/wavelet.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <iostream>

namespace wavelet
{
enum Dwt2DSubband {
    HORIZONTAL = 0,
    VERTICAL = 1,
    DIAGONAL = 2,
};

enum NormalizationMode {
    DWT_NO_NORMALIZE = 0,
    DWT_ZERO_TO_HALF_NORMALIZE,
    DWT_MAX_NORMALIZE,
};

namespace internal
{
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
class Dwt2dCoeffs
{
public:
    class LevelIterator
    {
    public:
        using value_type = Dwt2dCoeffs;
        using difference_type = int;

        LevelIterator() = default;
        LevelIterator(Dwt2dCoeffs* coeffs, int level) : coeffs(coeffs), level(level) {}
        value_type operator*() const { return coeffs->at_level(level); }
        auto& operator++(){ ++level; return *this; }
        auto operator++(int) { auto copy = *this; ++*this; return copy; }
        auto& operator--() { --level; return *this; }
        auto operator--(int) { auto copy = *this; --*this; return copy; }
        bool operator==(const LevelIterator& rhs) const { return coeffs == rhs.coeffs && level == rhs.level; }
        difference_type operator-(const LevelIterator& rhs) const { return level - rhs.level; }
    private:
        Dwt2dCoeffs* coeffs;
        int level;
    };

    class ConstLevelIterator
    {
    public:
        using value_type = const Dwt2dCoeffs;
        using difference_type = int;

        ConstLevelIterator() = default;
        ConstLevelIterator(const Dwt2dCoeffs* coeffs, int level) : coeffs(coeffs), level(level) {}
        value_type operator*() const { return coeffs->at_level(level); }
        auto& operator++(){ ++level; return *this; }
        auto operator++(int) { auto copy = *this; ++*this; return copy; }
        auto& operator--() { --level; return *this; }
        auto operator--(int) { auto copy = *this; --*this; return copy; }
        bool operator==(const ConstLevelIterator& other) const { return coeffs == other.coeffs && level == other.level; }
        difference_type operator-(const ConstLevelIterator& rhs) const { return level - rhs.level; }
    private:
        const Dwt2dCoeffs* coeffs;
        int level;
    };

public:
    Dwt2dCoeffs();
    Dwt2dCoeffs(const cv::Mat& matrix);
    Dwt2dCoeffs(cv::Mat&& matrix);
    Dwt2dCoeffs(const cv::Mat& matrix, int depth);
    Dwt2dCoeffs(int rows, int cols, int type, int depth=-1);
    Dwt2dCoeffs(const cv::Size& size, int type, int depth=-1);
    Dwt2dCoeffs(const Dwt2dCoeffs& other) = default;
    Dwt2dCoeffs(Dwt2dCoeffs&& other) = default;

    Dwt2dCoeffs& operator=(const Dwt2dCoeffs& coeffs) = default;
    Dwt2dCoeffs& operator=(Dwt2dCoeffs&& coeffs) = default;
    Dwt2dCoeffs& operator=(const cv::Mat& matrix);
    Dwt2dCoeffs& operator=(const cv::MatExpr& matrix);
    Dwt2dCoeffs& operator=(const cv::Scalar& scalar);
    Dwt2dCoeffs& operator=(cv::Mat&& matrix);

    operator cv::Mat() const { return _coeff_matrix; }
    operator cv::_InputArray() const { return _coeff_matrix; }
    operator cv::_OutputArray() const { return _coeff_matrix; }
    operator cv::_InputOutputArray() const { return _coeff_matrix; }

    Dwt2dCoeffs clone() const;

    cv::Mat approx() const;
    cv::Mat detail(int subband, int level=0) const;
    cv::Mat horizontal_detail(int level=0) const;
    cv::Mat vertical_detail(int level=0) const;
    cv::Mat diagonal_detail(int level=0) const;

    void set_level(const cv::Mat& coeffs, int level=0);
    void set_level(const cv::Scalar& scalar, int level=0);
    void set_approx(const cv::Mat& coeffs);
    void set_approx(const cv::Scalar& scalar);
    void set_detail(const cv::Mat& coeffs, int subband, int level=0);
    void set_detail(const cv::MatExpr& coeffs, int subband, int level=0);
    void set_detail(const cv::Scalar& scalar, int subband, int level=0);
    void set_horizontal_detail(const cv::Mat& coeffs, int level=0);
    void set_horizontal_detail(const cv::MatExpr& coeffs, int level=0);
    void set_horizontal_detail(const cv::Scalar& scalar, int level=0);
    void set_vertical_detail(const cv::Mat& coeffs, int level=0);
    void set_vertical_detail(const cv::MatExpr& coeffs, int level=0);
    void set_vertical_detail(const cv::Scalar& scalar, int level=0);
    void set_diagonal_detail(const cv::Mat& coeffs, int level=0);
    void set_diagonal_detail(const cv::MatExpr& coeffs, int level=0);
    void set_diagonal_detail(const cv::Scalar& scalar, int level=0);

    Dwt2dCoeffs at_level(int level) const;
    // Dwt2dCoeffs at_level(int level);

    std::vector<cv::Mat> collect_details(int subband) const;
    std::vector<cv::Mat> collect_horizontal_details() const { return collect_details(HORIZONTAL); }
    std::vector<cv::Mat> collect_vertical_details() const { return collect_details(VERTICAL); }
    std::vector<cv::Mat> collect_diagonal_details() const { return collect_details(DIAGONAL); }

    int levels() const { return _levels; }
    int rows() const { return _coeff_matrix.rows; }
    int cols() const { return _coeff_matrix.cols; }
    cv::Size size() const { return _coeff_matrix.size(); }
    int type() const { return _coeff_matrix.type(); }
    bool empty() const { return _coeff_matrix.empty(); }
    int total() const { return _coeff_matrix.total(); }
    int channels() const { return _coeff_matrix.channels(); }
    size_t elemSize() const { return _coeff_matrix.elemSize(); }
    size_t elemSize1() const { return _coeff_matrix.elemSize1(); }
    void copyTo(cv::OutputArray other) const { _coeff_matrix.copyTo(other); }
    void copyTo(cv::OutputArray other, cv::InputArray mask) const { _coeff_matrix.copyTo(other, mask); }
    void convertTo(cv::OutputArray other, int type, double alpha=1.0, double beta=0.0) const { _coeff_matrix.convertTo(other, type, alpha, beta); }
    bool isContinuous() const { return _coeff_matrix.isContinuous(); }
    bool isSubmatrix() const { return _coeff_matrix.isSubmatrix(); }

    ConstLevelIterator begin() const;
    ConstLevelIterator end() const;
    LevelIterator begin();
    LevelIterator end();

    ConstLevelIterator cbegin() const;
    ConstLevelIterator cend() const;
    ConstLevelIterator cbegin();
    ConstLevelIterator cend();

    void normalize(int approx_mode=DWT_MAX_NORMALIZE, int detail_mode=DWT_ZERO_TO_HALF_NORMALIZE);
    double maximum_abs_value() const;

    cv::Size level_size(int level) const;
    cv::Rect level_rect(int level) const;
    cv::Size detail_size(int level) const;
    cv::Rect approx_rect() const;
    cv::Rect horizontal_detail_rect(int level=0) const;
    cv::Rect vertical_detail_rect(int level=0) const;
    cv::Rect diagonal_detail_rect(int level=0) const;

    cv::Mat approx_mask() const;
    cv::Mat detail_mask(int lower_level=0, int upper_level=-1) const;
    cv::Mat horizontal_detail_mask(int level=0) const;
    cv::Mat vertical_detail_mask(int level=0) const;
    cv::Mat diagonal_detail_mask(int level=0) const;

    bool shares_data(const Dwt2dCoeffs& other) const;
    bool shares_data(const cv::Mat& matrix) const;

    friend std::ostream& operator<<(std::ostream& stream, const Dwt2dCoeffs& coeffs);

protected:
    int resolve_level(int level) const
    {
        return (level >= 0) ? level : level + levels();
    }
    template <typename MatrixLike>
    void check_size_for_assignment(const MatrixLike& matrix) const;

    void check_size_for_set_level(const cv::Mat& matrix, int level) const;
    void check_size_for_set_detail(const cv::Mat& matrix, int level, int subband) const;
    void check_size_for_set_approx(const cv::Mat& matrix) const;
    void check_level_nonnegative(int level, const std::string level_name = "level") const;
    void check_level_in_range(int level, const std::string level_name = "level") const;
    void check_nonempty() const;

    std::pair<double, double> normalization_constants(int normalization_mode, double max_abs_value) const;

    void convert_and_copy(const cv::Mat& source, const cv::Mat& destination);

private:
    cv::Mat _coeff_matrix;
    int _levels;
};

std::ostream& operator<<(std::ostream& stream, const Dwt2dCoeffs& coeffs);
} // namespace internal




class DWT2D {
public:
    using Coeffs = internal::Dwt2dCoeffs;

public:
    DWT2D(const Wavelet& wavelet, int border_type=cv::BORDER_DEFAULT);
    DWT2D() = delete;
    DWT2D(const DWT2D& other) = default;
    DWT2D(DWT2D&& other) = default;

    Coeffs operator()(cv::InputArray x, int levels=0) const;
    Coeffs forward(cv::InputArray x, int levels=0) const;
    void inverse(const Coeffs& coeffs, cv::OutputArray output, int levels=0) const;
    cv::Mat inverse(const Coeffs& coeffs, int levels=0) const;

    static int max_possible_levels(cv::InputArray x);
    static int max_possible_levels(int rows, int cols);
    static int max_possible_levels(const cv::Size& size);

private:
    void check_levels_nonnegative(int levels, const std::string levels_name="levels") const;

public:
    Wavelet wavelet;
    int border_type;
};


DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const Wavelet& wavelet,
    int levels=0,
    int border_type=cv::BORDER_DEFAULT
);

DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const std::string& wavelet,
    int levels=0,
    int border_type=cv::BORDER_DEFAULT
);

void dwt2d(
    cv::InputArray input,
    cv::OutputArray output,
    const Wavelet& wavelet,
    int levels=0,
    int border_type=cv::BORDER_DEFAULT
);

void dwt2d(
    cv::InputArray input,
    cv::OutputArray output,
    const std::string& wavelet,
    int levels=0,
    int border_type=cv::BORDER_DEFAULT
);

void idwt2d(
    const DWT2D::Coeffs& input,
    cv::OutputArray output,
    const Wavelet& wavelet,
    int levels=0,
    int border_type=cv::BORDER_DEFAULT
);

void idwt2d(
    const DWT2D::Coeffs& input,
    cv::OutputArray output,
    const std::string& wavelet,
    int levels=0,
    int border_type=cv::BORDER_DEFAULT
);

cv::Mat idwt2d(
    const DWT2D::Coeffs& input,
    const Wavelet& wavelet,
    int levels=0,
    int border_type=cv::BORDER_DEFAULT
);

cv::Mat idwt2d(
    const DWT2D::Coeffs& input,
    const std::string& wavelet,
    int levels=0,
    int border_type=cv::BORDER_DEFAULT
);

} // namespace wavelet

#endif  // WAVELET_DWT2D_HPP

