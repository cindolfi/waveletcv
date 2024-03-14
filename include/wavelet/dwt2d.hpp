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

class DWT2D;

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
    friend class wavelet::DWT2D;

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
    Dwt2dCoeffs(const cv::Mat& matrix, int levels);
    Dwt2dCoeffs(int rows, int cols, int type);
    Dwt2dCoeffs(int rows, int cols, int type, int levels);
    Dwt2dCoeffs(const cv::Size& size, int type);
    Dwt2dCoeffs(const cv::Size& size, int type, int levels);
    Dwt2dCoeffs(cv::Mat&& matrix);
    Dwt2dCoeffs(cv::Mat&& matrix, int levels);
    Dwt2dCoeffs(const Dwt2dCoeffs& other) = default;
    Dwt2dCoeffs(Dwt2dCoeffs&& other) = default;

    //  assignment
    Dwt2dCoeffs& operator=(const Dwt2dCoeffs& coeffs) = default;
    Dwt2dCoeffs& operator=(Dwt2dCoeffs&& coeffs) = default;
    Dwt2dCoeffs& operator=(const cv::Mat& matrix);
    Dwt2dCoeffs& operator=(const cv::MatExpr& matrix);
    Dwt2dCoeffs& operator=(const cv::Scalar& scalar);
    Dwt2dCoeffs& operator=(cv::Mat&& matrix);

    //  casting
    operator cv::Mat() const { return _coeff_matrix; }
    operator cv::_InputArray() const { return _coeff_matrix; }
    operator cv::_OutputArray() const { return _coeff_matrix; }
    operator cv::_InputOutputArray() const { return _coeff_matrix; }

    //  copy
    Dwt2dCoeffs clone() const;

    //  get & set sub-coefficients
    Dwt2dCoeffs at_level(int level) const;
    void set_level(int level, const cv::Mat& coeffs)
    {
        check_size_for_set_level(coeffs, level);
        convert_and_copy(coeffs, _coeff_matrix(level_rect(level)));
    }
    void set_level(int level, const cv::Scalar& scalar) { _coeff_matrix(level_rect(level)) = scalar; }

    //  get & set approx
    cv::Mat approx() const
    {
        return _coeff_matrix(approx_rect());
    }
    void set_approx(const cv::Mat& coeffs)
    {
        check_size_for_set_approx(coeffs);
        convert_and_copy(coeffs, approx());
    }
    void set_approx(const cv::Scalar& scalar) { approx() = scalar; }

    //  get & set detail coefficients via parameter
    cv::Mat detail(int level, int subband) const;
    cv::Mat detail(int subband) const { return detail(0, subband); }
    void set_detail(int level, int subband, const cv::Mat& coeffs)
    {
        check_size_for_set_detail(coeffs, level, subband);
        check_subband(subband);
        convert_and_copy(coeffs, detail(level, subband));
    }
    void set_detail(int subband, const cv::Mat& coeffs) { set_detail(0, subband, coeffs); }
    void set_detail(int level, int subband, const cv::MatExpr& coeffs) { set_detail(level, subband, cv::Mat(coeffs)); }
    void set_detail(const cv::MatExpr& coeffs, int subband) { set_detail(subband, cv::Mat(coeffs)); }
    void set_detail(int level, int subband, const cv::Scalar& scalar) { detail(level, subband) = scalar; }
    void set_detail(int subband, const cv::Scalar& scalar) { set_detail(0, subband, scalar); }

    //  get & set horizontal details
    cv::Mat horizontal_detail(int level) const
    {
        return _coeff_matrix(horizontal_detail_rect(level));
    }
    cv::Mat horizontal_detail() const { return horizontal_detail(0); }
    void set_horizontal_detail(int level, const cv::Mat& coeffs)
    {
        check_size_for_set_detail(coeffs, level, HORIZONTAL);
        convert_and_copy(coeffs, horizontal_detail(level));
    }
    void set_horizontal_detail(const cv::Mat& coeffs) { set_horizontal_detail(0, coeffs); }
    void set_horizontal_detail(int level, const cv::MatExpr& coeffs) { set_horizontal_detail(level, cv::Mat(coeffs)); }
    void set_horizontal_detail(const cv::MatExpr& coeffs) { set_horizontal_detail(cv::Mat(coeffs)); }
    void set_horizontal_detail(int level, const cv::Scalar& scalar) { horizontal_detail(level) = scalar; }
    void set_horizontal_detail(const cv::Scalar& scalar) { set_horizontal_detail(0, scalar); }

    //  get & set vertical details
    cv::Mat vertical_detail(int level) const
    {
        return _coeff_matrix(vertical_detail_rect(level));
    }
    cv::Mat vertical_detail() const { return vertical_detail(0); }
    void set_vertical_detail(int level, const cv::Mat& coeffs)
    {
        check_size_for_set_detail(coeffs, level, VERTICAL);
        convert_and_copy(coeffs, vertical_detail(level));
    }
    void set_vertical_detail(const cv::Mat& coeffs) { set_vertical_detail(0, coeffs); }
    void set_vertical_detail(int level, const cv::MatExpr& coeffs) { set_vertical_detail(level, cv::Mat(coeffs)); }
    void set_vertical_detail(const cv::MatExpr& coeffs) { set_vertical_detail(cv::Mat(coeffs)); }
    void set_vertical_detail(int level, const cv::Scalar& scalar) { vertical_detail(level) = scalar; }
    void set_vertical_detail(const cv::Scalar& scalar) { set_vertical_detail(0, scalar); }

    //  get & set diagonal details
    cv::Mat diagonal_detail(int level) const
    {
        return _coeff_matrix(diagonal_detail_rect(level));
    }
    cv::Mat diagonal_detail() const { return diagonal_detail(0); }
    void set_diagonal_detail(int level, const cv::Mat& coeffs)
    {
        check_size_for_set_detail(coeffs, level, DIAGONAL);
        convert_and_copy(coeffs, diagonal_detail(level));
    }
    void set_diagonal_detail(const cv::Mat& coeffs) { set_diagonal_detail(0, coeffs); }
    void set_diagonal_detail(int level, const cv::MatExpr& coeffs) { set_diagonal_detail(level, cv::Mat(coeffs)); }
    void set_diagonal_detail(const cv::MatExpr& coeffs) { set_diagonal_detail(cv::Mat(coeffs)); }
    void set_diagonal_detail(int level, const cv::Scalar& scalar) { diagonal_detail(level) = scalar; }
    void set_diagonal_detail(const cv::Scalar& scalar) { set_diagonal_detail(0, scalar); }

    //  collect details
    std::vector<cv::Mat> collect_details(int subband) const;
    std::vector<cv::Mat> collect_horizontal_details() const { return collect_details(HORIZONTAL); }
    std::vector<cv::Mat> collect_vertical_details() const { return collect_details(VERTICAL); }
    std::vector<cv::Mat> collect_diagonal_details() const { return collect_details(DIAGONAL); }

    //  sizes & rects
    cv::Size level_size(int level) const;
    cv::Rect level_rect(int level) const;
    cv::Size detail_size(int level=0) const;
    cv::Rect detail_rect(int level, int subband) const;
    cv::Rect detail_rect(int subband) const { return detail_rect(0, subband); }
    cv::Rect approx_rect() const;
    cv::Rect horizontal_detail_rect(int level=0) const;
    cv::Rect vertical_detail_rect(int level=0) const;
    cv::Rect diagonal_detail_rect(int level=0) const;

    //  masks
    cv::Mat approx_mask() const;
    cv::Mat detail_mask(int lower_level=0, int upper_level=-1) const;
    cv::Mat horizontal_detail_mask(int level=0) const;
    cv::Mat vertical_detail_mask(int level=0) const;
    cv::Mat diagonal_detail_mask(int level=0) const;

    //  convience cv::Mat wrappers
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

    //  level coefficients iterators
    auto begin() const { return ConstLevelIterator(this, 0); }
    auto end() const { return ConstLevelIterator(this, levels()); }
    auto begin() { return LevelIterator(this, 0); }
    auto end() { return LevelIterator(this, levels()); }
    auto cbegin() const { return ConstLevelIterator(this, 0); }
    auto cend() const { return ConstLevelIterator(this, levels()); }
    auto cbegin() { return ConstLevelIterator(this, 0); }
    auto cend() { return ConstLevelIterator(this, levels()); }

    //  other
    void normalize(
        NormalizationMode approx_mode=DWT_MAX_NORMALIZE,
        NormalizationMode detail_mode=DWT_ZERO_TO_HALF_NORMALIZE
    );

    bool shares_data(const Dwt2dCoeffs& other) const;
    bool shares_data(const cv::Mat& matrix) const;

    friend std::ostream& operator<<(std::ostream& stream, const Dwt2dCoeffs& wavelet);

protected:
    //  argument checkers - these all raise execeptions and can be disabled by
    //  defining DISABLE_ARG_CHECKS
    void check_size_for_assignment(cv::InputArray matrix) const;
    void check_size_for_set_level(const cv::Mat& matrix, int level) const;
    void check_size_for_set_detail(const cv::Mat& matrix, int level, int subband) const;
    void check_size_for_set_approx(const cv::Mat& matrix) const;
    void check_level_in_range(int level, const std::string level_name = "level") const;
    void check_constructor_level(int level, int max_level) const;
    void check_nonempty() const;
    void check_subband(int subband) const;

    //  helpers
    int resolve_level(int level) const { return (level >= 0) ? level : level + levels(); }
    double maximum_abs_value() const;

    std::pair<double, double> normalization_constants(
        NormalizationMode normalization_mode,
        double max_abs_value
    ) const;

    void convert_and_copy(const cv::Mat& source, const cv::Mat& destination);

private:
    cv::Mat _coeff_matrix;
    int _levels;
};

std::ostream& operator<<(std::ostream& stream, const Dwt2dCoeffs& wavelet);
} // namespace internal




class DWT2D {
public:
    using Coeffs = internal::Dwt2dCoeffs;

public:
    DWT2D(const Wavelet& wavelet, cv::BorderTypes border_type=cv::BORDER_DEFAULT);
    DWT2D() = delete;
    DWT2D(const DWT2D& other) = default;
    DWT2D(DWT2D&& other) = default;

    Coeffs operator()(cv::InputArray x) const { return forward(x); }
    Coeffs operator()(cv::InputArray x, int levels) const { return forward(x, levels); }
    void operator()(cv::InputArray x, Coeffs& output) const { forward(x, output); }
    void operator()(cv::InputArray x, Coeffs& output, int levels) const { forward(x, output, levels); }

    Coeffs forward(cv::InputArray x) const;
    Coeffs forward(cv::InputArray x, int levels) const;
    void forward(cv::InputArray x, Coeffs& output) const;
    void forward(cv::InputArray x, Coeffs& output, int levels) const;
    Coeffs running_forward(const Coeffs& coeffs, int levels) const;
    Coeffs running_forward(const cv::Mat& x, int levels) const;
    void running_forward(const Coeffs& coeffs, Coeffs& output, int levels) const;
    void running_forward(const cv::Mat& x, Coeffs& output, int levels) const;

    cv::Mat inverse(const Coeffs& coeffs) const;
    void inverse(const Coeffs& coeffs, cv::OutputArray output) const;
    Coeffs running_inverse(const Coeffs& coeffs, int levels) const;
    void running_inverse(const Coeffs& coeffs, Coeffs& output, int levels) const;

    static int max_possible_levels(cv::InputArray x);
    static int max_possible_levels(int rows, int cols);
    static int max_possible_levels(const cv::Size& size);

    static int max_levels_without_border_effects(int rows, int cols, int filter_length);
    static int max_levels_without_border_effects(const cv::Size& size, int filter_length);
    static int max_levels_without_border_effects(cv::InputArray x, int filter_length);

protected:
    void _forward(cv::InputArray x, DWT2D::Coeffs& output, int levels) const;

    //  argument checkers - these all raise execeptions and can be disabled by
    //  defining DISABLE_ARG_CHECKS
    void check_levels_in_range(int levels, int min_levels, int max_levels) const;
    void check_levels_in_range(int levels, int min_levels, cv::InputArray x) const;

    void copy_if_not_identical(const DWT2D::Coeffs& x, DWT2D::Coeffs& output) const;
    void create_like(cv::InputArray x, DWT2D::Coeffs& output, int levels) const;
    void create_like(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& output) const;

public:
    Wavelet wavelet;
    cv::BorderTypes border_type;
};


/**
 * -----------------------------------------------------------------------------
 * Functional Interface
 * -----------------------------------------------------------------------------
*/
DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void dwt2d(
    cv::InputArray input,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void dwt2d(
    cv::InputArray input,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void dwt2d(
    cv::InputArray input,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void dwt2d(
    cv::InputArray input,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

DWT2D::Coeffs running_dwt2d(
    const DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

DWT2D::Coeffs running_dwt2d(
    const DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void running_dwt2d(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void running_dwt2d(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);




void idwt2d(
    const DWT2D::Coeffs& input,
    cv::OutputArray output,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void idwt2d(
    const DWT2D::Coeffs& input,
    cv::OutputArray output,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

cv::Mat idwt2d(
    const DWT2D::Coeffs& input,
    const Wavelet& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

cv::Mat idwt2d(
    const DWT2D::Coeffs& input,
    const std::string& wavelet,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void running_idwt2d(
    const DWT2D::Coeffs& input,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

void running_idwt2d(
    const DWT2D::Coeffs& input,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

DWT2D::Coeffs running_idwt2d(
    const DWT2D::Coeffs& input,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);

DWT2D::Coeffs running_idwt2d(
    const DWT2D::Coeffs& input,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type=cv::BORDER_DEFAULT
);


} // namespace wavelet

#endif  // WAVELET_DWT2D_HPP

