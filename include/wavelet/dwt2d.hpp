#ifndef WAVELET_DWT2D_HPP
#define WAVELET_DWT2D_HPP

#include "wavelet/wavelet.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>

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
class Dwt2dCoeffsImpl
{
public:
    Dwt2dCoeffsImpl() :
        coeff_matrix(),
        levels(0),
        input_size(),
        diagonal_subband_rects(),
        wavelet(),
        border_type(cv::BORDER_DEFAULT)
    {
    }

    Dwt2dCoeffsImpl(
        const cv::Mat& coeff_matrix,
        int levels,
        const cv::Size& input_size,
        const std::vector<cv::Rect>& diagonal_subband_rects,
        const Wavelet& wavelet,
        cv::BorderTypes border_type
    ) :
        coeff_matrix(coeff_matrix),
        levels(levels),
        input_size(input_size),
        diagonal_subband_rects(diagonal_subband_rects),
        wavelet(wavelet),
        border_type(border_type)
    {
    }

    Dwt2dCoeffsImpl(
        const cv::Mat& coeff_matrix,
        int levels,
        const cv::Size& input_size,
        const std::vector<cv::Size>& subband_sizes,
        const Wavelet& wavelet,
        cv::BorderTypes border_type
    ) :
        coeff_matrix(coeff_matrix),
        levels(levels),
        input_size(input_size),
        diagonal_subband_rects(),
        wavelet(wavelet),
        border_type(border_type)
    {
        build_diagonal_subband_rects(subband_sizes);
    }

    void build_diagonal_subband_rects(const std::vector<cv::Size>& subband_sizes)
    {
        cv::Point offset(
            coeff_matrix.size().width,
            coeff_matrix.size().height
        );
        diagonal_subband_rects.clear();
        for (const auto& size : subband_sizes) {
            offset.x = offset.x - size.width;
            offset.y = offset.y - size.height;
            diagonal_subband_rects.emplace_back(offset, size);
        }
    }

public:
    cv::Mat coeff_matrix;
    int levels;
    cv::Size input_size;
    std::vector<cv::Rect> diagonal_subband_rects;
    Wavelet wavelet;
    cv::BorderTypes border_type;
};
} // namespace internal


class DWT2D
{
public:
    /*
        |---------------------------------------------------------------------|
        |  2a   |  2v    |                 |                                  |
        |-------+--------|      1v         |                                  |
        |  2h   |  2d    |                 |                                  |
        |----------------+-----------------|               0v                 |
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
        |---------------------------------------------------------------------|
    */
    class Coeffs
    {
        friend class DWT2D;

    public:
        class LevelIterator
        {
        public:
            using value_type = Coeffs;
            using difference_type = int;

            LevelIterator() = default;
            LevelIterator(Coeffs* coeffs, int level) : coeffs(coeffs), level(level) {}
            value_type operator*() const { return coeffs->at_level(level); }
            auto& operator++(){ ++level; return *this; }
            auto operator++(int) { auto copy = *this; ++*this; return copy; }
            auto& operator--() { --level; return *this; }
            auto operator--(int) { auto copy = *this; --*this; return copy; }
            bool operator==(const LevelIterator& rhs) const { return coeffs == rhs.coeffs && level == rhs.level; }
            difference_type operator-(const LevelIterator& rhs) const { return level - rhs.level; }
        private:
            Coeffs* coeffs;
            int level;
        };

        class ConstLevelIterator
        {
        public:
            using value_type = const Coeffs;
            using difference_type = int;

            ConstLevelIterator() = default;
            ConstLevelIterator(const Coeffs* coeffs, int level) : coeffs(coeffs), level(level) {}
            value_type operator*() const { return coeffs->at_level(level); }
            auto& operator++(){ ++level; return *this; }
            auto operator++(int) { auto copy = *this; ++*this; return copy; }
            auto& operator--() { --level; return *this; }
            auto operator--(int) { auto copy = *this; --*this; return copy; }
            bool operator==(const ConstLevelIterator& other) const { return coeffs == other.coeffs && level == other.level; }
            difference_type operator-(const ConstLevelIterator& rhs) const { return level - rhs.level; }
        private:
            const Coeffs* coeffs;
            int level;
        };

    protected:
        Coeffs(
            const cv::Mat& matrix,
            int levels,
            const cv::Size& input_size,
            const std::vector<cv::Size>& subband_sizes,
            const Wavelet& wavelet,
            cv::BorderTypes border_type
        );

        Coeffs(
            const cv::Mat& matrix,
            int levels,
            const cv::Size& input_size,
            const std::vector<cv::Rect>& diagonal_subband_rects,
            const Wavelet& wavelet,
            cv::BorderTypes border_type
        );

        void reset(
            const cv::Size& size,
            int type,
            int levels,
            const cv::Size& input_size,
            const std::vector<cv::Size>& subband_sizes,
            const Wavelet& wavelet,
            cv::BorderTypes border_type
        );

    public:
        Coeffs();
        Coeffs(const Coeffs& other) = default;
        Coeffs(Coeffs&& other) = default;

        //  Assignment
        Coeffs& operator=(const Coeffs& coeffs) = default;
        Coeffs& operator=(Coeffs&& coeffs) = default;
        Coeffs& operator=(const cv::Mat& matrix);
        Coeffs& operator=(const cv::MatExpr& matrix);
        Coeffs& operator=(const cv::Scalar& scalar);

        //  Casting
        operator cv::Mat() const { return _p->coeff_matrix; }
        operator cv::_InputArray() const { return _p->coeff_matrix; }
        operator cv::_OutputArray() const { return _p->coeff_matrix; }
        operator cv::_InputOutputArray() const { return _p->coeff_matrix; }

        //  Copy
        Coeffs clone() const;

        //  Get & Set Sub-coefficients
        Coeffs at_level(int level) const;
        void set_level(int level, const cv::Mat& coeffs)
        {
            check_size_for_set_level(coeffs, level);
            convert_and_copy(coeffs, _p->coeff_matrix(level_rect(level)));
        }
        void set_level(int level, const cv::Scalar& scalar) { _p->coeff_matrix(level_rect(level)) = scalar; }

        //  Get & Set Approximation Coefficients
        cv::Mat approx() const
        {
            return _p->coeff_matrix(approx_rect());
        }
        void set_approx(const cv::Mat& coeffs)
        {
            check_size_for_set_approx(coeffs);
            convert_and_copy(coeffs, approx());
        }
        void set_approx(const cv::Scalar& scalar) { approx() = scalar; }

        //  Get & Set Detail Coefficients (via parameter)
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

        //  Get & Set Horizontal Detail Coefficients
        cv::Mat horizontal_detail(int level) const
        {
            return _p->coeff_matrix(horizontal_detail_rect(level));
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

        //  Get & Set Vertical Detail Coefficients
        cv::Mat vertical_detail(int level) const
        {
            return _p->coeff_matrix(vertical_detail_rect(level));
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

        //  Get & Set Diagonal Detail Coefficients
        cv::Mat diagonal_detail(int level) const
        {
            return _p->coeff_matrix(diagonal_detail_rect(level));
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

        //  Collect Detail Coefficients
        std::vector<cv::Mat> collect_details(int subband) const;
        std::vector<cv::Mat> collect_horizontal_details() const { return collect_details(HORIZONTAL); }
        std::vector<cv::Mat> collect_vertical_details() const { return collect_details(VERTICAL); }
        std::vector<cv::Mat> collect_diagonal_details() const { return collect_details(DIAGONAL); }

        //  Sizes & Rects
        cv::Size level_size(int level) const;
        cv::Rect level_rect(int level) const;
        cv::Size detail_size(int level=0) const;
        cv::Rect detail_rect(int level, int subband) const;
        cv::Rect detail_rect(int subband) const { return detail_rect(0, subband); }
        cv::Rect approx_rect() const;
        cv::Rect horizontal_detail_rect(int level=0) const;
        cv::Rect vertical_detail_rect(int level=0) const;
        cv::Rect diagonal_detail_rect(int level=0) const;

        //  Masks
        cv::Mat approx_mask() const;
        cv::Mat detail_mask(int lower_level=0, int upper_level=-1) const;
        cv::Mat detail_mask(const cv::Range& levels) const;
        cv::Mat horizontal_detail_mask(int level=0) const;
        cv::Mat vertical_detail_mask(int level=0) const;
        cv::Mat diagonal_detail_mask(int level=0) const;

        //  Convenience cv::Mat Wrappers
        int levels() const { return _p->levels; }
        int rows() const { return _p->coeff_matrix.rows; }
        int cols() const { return _p->coeff_matrix.cols; }
        cv::Size size() const { return _p->coeff_matrix.size(); }
        int type() const { return _p->coeff_matrix.type(); }
        int depth() const { return _p->coeff_matrix.depth(); }
        int channels() const { return _p->coeff_matrix.channels(); }
        bool empty() const { return _p->coeff_matrix.empty(); }
        int total() const { return _p->coeff_matrix.total(); }
        size_t elemSize() const { return _p->coeff_matrix.elemSize(); }
        size_t elemSize1() const { return _p->coeff_matrix.elemSize1(); }
        void copyTo(cv::OutputArray other) const { _p->coeff_matrix.copyTo(other); }
        void copyTo(cv::OutputArray other, cv::InputArray mask) const { _p->coeff_matrix.copyTo(other, mask); }
        void convertTo(cv::OutputArray other, int type, double alpha=1.0, double beta=0.0) const { _p->coeff_matrix.convertTo(other, type, alpha, beta); }
        bool isContinuous() const { return _p->coeff_matrix.isContinuous(); }
        bool isSubmatrix() const { return _p->coeff_matrix.isSubmatrix(); }

        //  Level Iterators
        auto begin() const { return ConstLevelIterator(this, 0); }
        auto end() const { return ConstLevelIterator(this, levels()); }
        auto begin() { return LevelIterator(this, 0); }
        auto end() { return LevelIterator(this, levels()); }
        auto cbegin() const { return ConstLevelIterator(this, 0); }
        auto cend() const { return ConstLevelIterator(this, levels()); }
        auto cbegin() { return ConstLevelIterator(this, 0); }
        auto cend() { return ConstLevelIterator(this, levels()); }

        //  DWT
        Wavelet wavelet() const { return _p->wavelet; }
        cv::BorderTypes border_type() const { return _p->border_type; }
        DWT2D dwt() const;
        cv::Size input_size(int level=0) const { return level == 0 ? _p->input_size : diagonal_detail_rect(level - 1).size(); }

        cv::Mat invert() const;
        void invert(cv::OutputArray output) const;

        //  Other
        void normalize(
            NormalizationMode approx_mode=DWT_MAX_NORMALIZE,
            NormalizationMode detail_mode=DWT_ZERO_TO_HALF_NORMALIZE
        );

        bool shares_data(const Coeffs& other) const;
        bool shares_data(const cv::Mat& matrix) const;

        friend std::vector<Coeffs> split(const Coeffs& coeffs);
        friend Coeffs merge(const std::vector<Coeffs>& coeffs);
        friend std::ostream& operator<<(std::ostream& stream, const Coeffs& wavelet);

    protected:
        //  Argument Checkers - these all raise execeptions and can be disabled by
        //  defining DISABLE_ARG_CHECKS
        void check_size_for_assignment(cv::InputArray matrix) const;
        void check_size_for_set_level(const cv::Mat& matrix, int level) const;
        void check_size_for_set_detail(const cv::Mat& matrix, int level, int subband) const;
        void check_size_for_set_approx(const cv::Mat& matrix) const;
        void check_level_in_range(int level, const std::string level_name = "level") const;
        void check_constructor_level(int level, int max_level) const;
        void check_nonempty() const;
        void check_subband(int subband) const;

        //  Helpers
        double maximum_abs_value() const;
        std::pair<double, double> normalization_constants(
            NormalizationMode normalization_mode,
            double max_abs_value
        ) const;
        void convert_and_copy(const cv::Mat& source, const cv::Mat& destination);
        int resolve_level(int level) const { return (level >= 0) ? level : level + levels(); }

    private:
        std::shared_ptr<internal::Dwt2dCoeffsImpl> _p;
    };

public:
    DWT2D(const Wavelet& wavelet, cv::BorderTypes border_type=cv::BORDER_DEFAULT);
    DWT2D() = delete;
    DWT2D(const DWT2D& other) = default;
    DWT2D(DWT2D&& other) = default;

    Coeffs operator()(cv::InputArray x) const { return decompose(x); }
    Coeffs operator()(cv::InputArray x, int levels) const { return decompose(x, levels); }
    void operator()(cv::InputArray x, Coeffs& output) const { decompose(x, output); }
    void operator()(cv::InputArray x, Coeffs& output, int levels) const { decompose(x, output, levels); }

    void decompose(cv::InputArray x, Coeffs& output, int levels) const;
    Coeffs decompose(cv::InputArray x) const
    {
        Coeffs coeffs;
        decompose(x, coeffs);
        return coeffs;
    }
    void decompose(cv::InputArray x, Coeffs& output) const
    {
        decompose(x, output, max_levels_without_border_effects(x));
    }
    Coeffs decompose(cv::InputArray x, int levels) const
    {
        DWT2D::Coeffs coeffs;
        decompose(x, coeffs, levels);
        return coeffs;
    }

    void reconstruct(const Coeffs& coeffs, cv::OutputArray output) const;
    cv::Mat reconstruct(const Coeffs& coeffs) const
    {
        cv::Mat output;
        reconstruct(coeffs, output);
        return output;
    }

    Coeffs create_coeffs(cv::InputArray coeffs_matrix, const cv::Size& input_size, int levels) const;
    Coeffs create_coeffs(const cv::Size& input_size, int type, int levels) const
    {
        auto size = coeffs_size_for_input(input_size, levels);
        return create_coeffs(cv::Mat(size, type, 0.0), input_size, levels);
    }
    Coeffs create_coeffs(int input_rows, int input_cols, int type, int levels) const
    {
        return create_coeffs(cv::Size(input_cols, input_rows), type, levels);
    }

    Coeffs create_coeffs_for_input(const cv::Size& size, int type, int levels) const;
    Coeffs create_coeffs_for_input(cv::InputArray input, int levels) const
    {
        return create_coeffs_for_input(input.size(), input.type(), levels);
    }
    Coeffs create_coeffs_for_input(int rows, int cols, int type, int levels) const
    {
        return create_coeffs_for_input(cv::Size(cols, rows), type, levels);
    }

    cv::Size coeffs_size_for_input(const cv::Size& input_size, int levels) const;
    cv::Size coeffs_size_for_input(cv::InputArray input, int levels) const
    {
        return coeffs_size_for_input(input.size(), levels);
    }
    cv::Size coeffs_size_for_input(int rows, int cols, int levels) const
    {
        return coeffs_size_for_input(cv::Size(cols, rows), levels);
    }

    int max_levels_without_border_effects(int rows, int cols) const;
    int max_levels_without_border_effects(const cv::Size& size) const
    {
        return max_levels_without_border_effects(size.height, size.width);
    }
    int max_levels_without_border_effects(cv::InputArray x) const
    {
        return max_levels_without_border_effects(x.size());
    }
protected:
    //  Argument Checkers - these all raise execeptions and can be disabled by
    //  defining DISABLE_ARG_CHECKS
    void check_levels_in_range(int levels) const;
    void check_coeffs_size(cv::InputArray coeffs, const cv::Size& input_size, int levels) const;

    void warn_if_border_effects_will_occur(int levels, const cv::Size& input_size) const;
    void warn_if_border_effects_will_occur(int levels, cv::InputArray x) const;
    void warn_if_border_effects_will_occur(const Coeffs& coeffs) const;

    std::vector<cv::Size> calc_subband_sizes(const cv::Size& input_size, int levels) const;
public:
    Wavelet wavelet;
    cv::BorderTypes border_type;
};

std::vector<DWT2D::Coeffs> split(const DWT2D::Coeffs& coeffs);
DWT2D::Coeffs merge(const std::vector<DWT2D::Coeffs>& coeffs);
std::ostream& operator<<(std::ostream& stream, const DWT2D::Coeffs& wavelet);


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

} // namespace wavelet

#endif  // WAVELET_DWT2D_HPP

