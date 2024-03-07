#include <iostream>
#include <opencv2/imgproc.hpp>
#include "wavelet/dwt2d.hpp"
#include "wavelet/utils.hpp"

namespace wavelet
{

namespace internal
{
Dwt2dCoeffs::Dwt2dCoeffs() :
    _coeff_matrix(),
    _levels(0)
{
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Mat& matrix) :
    Dwt2dCoeffs(matrix, 0)
{
}

Dwt2dCoeffs::Dwt2dCoeffs(cv::Mat&& matrix) :
    _coeff_matrix(matrix),
    _levels(DWT2D::max_possible_levels(_coeff_matrix))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Mat& matrix, int levels) :
    _coeff_matrix(matrix),
    _levels(levels > 0 ? levels : DWT2D::max_possible_levels(matrix))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(int rows, int cols, int type, int levels) :
    _coeff_matrix(rows, cols, type, 0.0),
    _levels(levels > 0 ? levels : DWT2D::max_possible_levels(rows, cols))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Size& size, int type, int levels) :
    Dwt2dCoeffs(size.height, size.width, type, levels)
{
}

Dwt2dCoeffs Dwt2dCoeffs::clone() const
{
    return Dwt2dCoeffs(_coeff_matrix.clone(), _levels);
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::Mat& matrix)
{
    check_size_for_assignment(matrix);

    if (matrix.type() != _coeff_matrix.type()) {
        cv::Mat matrix2;
        matrix.convertTo(matrix2, _coeff_matrix.type());
        matrix2.copyTo(_coeff_matrix);
    } else {
        matrix.copyTo(_coeff_matrix);
    }

    _levels = DWT2D::max_possible_levels(_coeff_matrix);

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(cv::Mat&& matrix)
{
    check_size_for_assignment(matrix);
    _coeff_matrix = matrix;
    _levels = DWT2D::max_possible_levels(_coeff_matrix);

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::MatExpr& matrix)
{
    check_size_for_assignment(matrix);
    _coeff_matrix = matrix;
    _levels = DWT2D::max_possible_levels(_coeff_matrix);

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::Scalar& scalar)
{
    _coeff_matrix = scalar;
    _levels = DWT2D::max_possible_levels(_coeff_matrix);

    return *this;
}

std::vector<cv::Mat> Dwt2dCoeffs::collect_details(int subband) const
{
    std::vector<cv::Mat> result;
    for (int level = 0; level < levels(); ++level)
        result.push_back(detail(subband, level));

    return result;
}

Dwt2dCoeffs Dwt2dCoeffs::at_level(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);
    if (level == 0)
        return *this;

    return Dwt2dCoeffs(
        _coeff_matrix(level_rect(level)),
        levels() - level
    );
}

cv::Mat Dwt2dCoeffs::approx() const
{
    if (empty())
        return cv::Mat(0, 0, type());

    return _coeff_matrix(approx_rect());
}

cv::Mat Dwt2dCoeffs::detail(int subband, int level) const
{
    switch (subband) {
        case HORIZONTAL: return horizontal_detail(level);
        case VERTICAL: return vertical_detail(level);
        case DIAGONAL: return diagonal_detail(level);
        default:
            std::stringstream message;
            message
                << "Invalid subband.  "
                << "Must be 0 (HORIZONTAL), 1 (VERTICAL), or 2 (DIAGONAL) - "
                << "got " << subband << ".";
            CV_Error(cv::Error::StsBadArg, message.str());
    }
}

cv::Mat Dwt2dCoeffs::horizontal_detail(int level) const
{
    if (empty())
        return cv::Mat(0, 0, type());

    return _coeff_matrix(horizontal_detail_rect(level));
}

cv::Mat Dwt2dCoeffs::vertical_detail(int level) const
{
    if (empty())
        return cv::Mat(0, 0, type());

    return _coeff_matrix(vertical_detail_rect(level));
}

cv::Mat Dwt2dCoeffs::diagonal_detail(int level) const
{
    if (empty())
        return cv::Mat(0, 0, type());

    return _coeff_matrix(diagonal_detail_rect(level));
}

void Dwt2dCoeffs::set_level(const cv::Mat& coeffs, int level)
{
    check_size_for_set_level(coeffs, level);
    convert_and_copy(coeffs, _coeff_matrix(level_rect(level)));
}

void Dwt2dCoeffs::set_level(const cv::Scalar& scalar, int level)
{
    _coeff_matrix(level_rect(level)) = scalar;
}

void Dwt2dCoeffs::set_approx(const cv::Mat& coeffs)
{
    check_size_for_set_approx(coeffs);
    convert_and_copy(coeffs, approx());
}

void Dwt2dCoeffs::set_approx(const cv::Scalar& scalar)
{
    approx() = scalar;
}

void Dwt2dCoeffs::set_detail(const cv::Mat& coeffs, int subband, int level)
{
    check_size_for_set_detail(coeffs, level, subband);
    convert_and_copy(coeffs, detail(subband, level));
}

void Dwt2dCoeffs::set_detail(const cv::MatExpr& coeffs, int subband, int level)
{
    set_detail(cv::Mat(coeffs), subband, level);
}

void Dwt2dCoeffs::set_detail(const cv::Scalar& scalar, int subband, int level)
{
    detail(subband, level) = scalar;
}

void Dwt2dCoeffs::set_horizontal_detail(const cv::Mat& coeffs, int level)
{
    check_size_for_set_detail(coeffs, level, HORIZONTAL);
    convert_and_copy(coeffs, horizontal_detail(level));
}

void Dwt2dCoeffs::set_horizontal_detail(const cv::MatExpr& coeffs, int level)
{
    set_horizontal_detail(cv::Mat(coeffs), level);
}

void Dwt2dCoeffs::set_horizontal_detail(const cv::Scalar& scalar, int level)
{
    horizontal_detail(level) = scalar;
}

void Dwt2dCoeffs::set_vertical_detail(const cv::Mat& coeffs, int level)
{
    check_size_for_set_detail(coeffs, level, VERTICAL);
    convert_and_copy(coeffs, vertical_detail(level));
}

void Dwt2dCoeffs::set_vertical_detail(const cv::MatExpr& coeffs, int level)
{
    set_vertical_detail(cv::Mat(coeffs), level);
}

void Dwt2dCoeffs::set_vertical_detail(const cv::Scalar& scalar, int level)
{
    vertical_detail(level) = scalar;
}

void Dwt2dCoeffs::set_diagonal_detail(const cv::Mat& coeffs, int level)
{
    check_size_for_set_detail(coeffs, level, DIAGONAL);
    convert_and_copy(coeffs, diagonal_detail(level));
}

void Dwt2dCoeffs::set_diagonal_detail(const cv::MatExpr& coeffs, int level)
{
    set_diagonal_detail(cv::Mat(coeffs), level);
}

void Dwt2dCoeffs::set_diagonal_detail(const cv::Scalar& scalar, int level)
{
    diagonal_detail(level) = scalar;
}

void Dwt2dCoeffs::convert_and_copy(const cv::Mat& source, const cv::Mat& destination)
{
    if (source.type() != destination.type()) {
        cv::Mat source2;
        source.convertTo(source2, type());
        source2.copyTo(destination);
    } else {
        source.copyTo(destination);
    }
}

#ifndef DISABLE_ARG_CHECKS
template <typename MatrixLike>
void Dwt2dCoeffs::check_size_for_assignment(const MatrixLike& matrix) const
{
    if (matrix.size() != size()) {
        std::stringstream message;
        message
            << "Cannot assign the matrix to this Dwt2dCoeffs.  "
            << "The size of the matrix must be " << size() << ") - "
            << "got " << matrix.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}

void Dwt2dCoeffs::check_size_for_set_level(const cv::Mat& matrix, int level) const
{
    if (matrix.size() != level_size(level)) {
        std::stringstream message;
        message
            << "Cannot set the coeffs at level " << level << ".  "
            << "The size of the matrix must be " << level_size(level) << " - "
            << "got " << matrix.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}

void Dwt2dCoeffs::check_size_for_set_detail(const cv::Mat& matrix, int level, int subband) const
{
    if (matrix.size() != detail_size(level)) {
        std::string subband_name;
        switch (subband) {
            case HORIZONTAL:
                subband_name = "horizontal";
                break;
            case VERTICAL:
                subband_name = "vertical";
                break;
            case DIAGONAL:
                subband_name = "diagonal";
                break;
            default:
                assert("Unknown subband identifier");
        }
        std::stringstream message;
        message
            << "Cannot set the " << subband_name << " detail coefficients at level " << level << ".  "
            << "The size of the matrix must be " << detail_size(level) << " - "
            << "got " << matrix.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}

void Dwt2dCoeffs::check_size_for_set_approx(const cv::Mat& matrix) const
{
    if (matrix.size() != detail_size(levels() - 1)) {
        std::stringstream message;
        message
            << "Cannot set the approx coefficients.  "
            << "The size of the matrix must be " << detail_size(levels() - 1) << " - "
            << "got " << matrix.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}

void Dwt2dCoeffs::check_level_nonnegative(int level, const std::string level_name) const
{
    if (level < 0 || level >= levels()) {
        std::stringstream message;
        message
            << level_name << " is out of range. "
            << "Must be " << 0 << " <= " << level_name << " < " << levels() << " - "
            << "got " << level << ".";
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }
}

void Dwt2dCoeffs::check_level_in_range(int level, const std::string level_name) const
{
    if (level < -levels() || level >= levels()) {
        std::stringstream message;
        message
            << level_name << " is out of range. "
            << "Must be " << -levels() << " <= " << level_name << " < " << levels() << " - "
            << "got " << level << ".";
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }
}

void Dwt2dCoeffs::check_nonempty() const
{
    if (empty()) {
        CV_Error(cv::Error::StsBadSize, "Coefficients are empty.");
    }
}
#else
template <typename MatrixLike>
inline void Dwt2dCoeffs::check_size_for_assignment(const MatrixLike& matrix) const {}

inline void Dwt2dCoeffs::check_size_for_set_level(const MatrixLike& matrix, int level) const {}
inline void Dwt2dCoeffs::check_size_for_set_approx(const cv::Mat& matrix) const {}
inline void Dwt2dCoeffs::check_size_for_set_detail(const MatrixLike& matrix, int level, int subband) const {}
inline void Dwt2dCoeffs::check_nonnegative_level(int level, const std::string level_name) const {}
inline void Dwt2dCoeffs::check_level_in_range(int level, const std::string level_name) const {}
#endif

cv::Size Dwt2dCoeffs::level_size(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);
    return _coeff_matrix.size() / int(std::pow(2, level));
}

cv::Rect Dwt2dCoeffs::level_rect(int level) const
{
    auto size = level_size(level);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(0, 0), size);
}

cv::Size Dwt2dCoeffs::detail_size(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);
    return _coeff_matrix.size() / int(std::pow(2, 1 + level));
}

cv::Rect Dwt2dCoeffs::approx_rect() const
{
    auto size = detail_size(-1);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(0, 0), size);
}

cv::Rect Dwt2dCoeffs::horizontal_detail_rect(int level) const
{
    auto size = detail_size(level);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(0, size.height), size);
}

cv::Rect Dwt2dCoeffs::vertical_detail_rect(int level) const
{
    auto size = detail_size(level);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(size.width, 0), size);
}

cv::Rect Dwt2dCoeffs::diagonal_detail_rect(int level) const
{
    auto size = detail_size(level);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(size.width, size.height), size);
}

cv::Mat Dwt2dCoeffs::approx_mask() const
{
    if (empty())
        return cv::Mat();

    auto mask = cv::Mat(size(), CV_8U, cv::Scalar(0));
    mask(approx_rect()) = 255;

    return mask;
}

cv::Mat Dwt2dCoeffs::detail_mask(int lower_level, int upper_level) const
{
    check_level_in_range(lower_level, "lower_level");
    lower_level = resolve_level(lower_level);

    check_level_in_range(upper_level, "upper_level");
    upper_level = resolve_level(upper_level);

    if (empty())
        return cv::Mat();

    cv::Mat mask;
    if (lower_level == 0 && upper_level == levels() - 1) {
        mask = cv::Mat(size(), CV_8U, cv::Scalar(255));
        mask(approx_rect()) = 0;
    } else {
        mask = cv::Mat(size(), CV_8U, cv::Scalar(0));
        for (int level = lower_level; level <= upper_level; ++level) {
            mask(horizontal_detail_rect(level)) = 255;
            mask(vertical_detail_rect(level)) = 255;
            mask(diagonal_detail_rect(level)) = 255;
        }
    }

    return mask;
}

cv::Mat Dwt2dCoeffs::horizontal_detail_mask(int level) const
{
    if (empty())
        return cv::Mat();

    auto mask = cv::Mat(size(), CV_8U, cv::Scalar(0));
    mask(horizontal_detail_rect(level)) = 255;

    return mask;
}

cv::Mat Dwt2dCoeffs::vertical_detail_mask(int level) const
{
    if (empty())
        return cv::Mat();

    auto mask = cv::Mat(size(), CV_8U, cv::Scalar(0));
    mask(vertical_detail_rect(level)) = 255;

    return mask;
}

cv::Mat Dwt2dCoeffs::diagonal_detail_mask(int level) const
{
    if (empty())
        return cv::Mat();

    auto mask = cv::Mat(size(), CV_8U, cv::Scalar(0));
    mask(diagonal_detail_rect(level)) = 255;

    return mask;
}

Dwt2dCoeffs::ConstLevelIterator Dwt2dCoeffs::begin() const
{
    return ConstLevelIterator(this, 0);
}

Dwt2dCoeffs::ConstLevelIterator Dwt2dCoeffs::end() const
{
    return ConstLevelIterator(this, levels());
}

Dwt2dCoeffs::LevelIterator Dwt2dCoeffs::begin()
{
    return LevelIterator(this, 0);
}

Dwt2dCoeffs::LevelIterator Dwt2dCoeffs::end()
{
    return LevelIterator(this, levels());
}

Dwt2dCoeffs::ConstLevelIterator Dwt2dCoeffs::cbegin() const
{
    return ConstLevelIterator(this, 0);
}

Dwt2dCoeffs::ConstLevelIterator Dwt2dCoeffs::cend() const
{
    return ConstLevelIterator(this, levels());
}

Dwt2dCoeffs::ConstLevelIterator Dwt2dCoeffs::cbegin()
{
    return ConstLevelIterator(this, 0);
}

Dwt2dCoeffs::ConstLevelIterator Dwt2dCoeffs::cend()
{
    return ConstLevelIterator(this, levels());
}


bool Dwt2dCoeffs::shares_data(const Dwt2dCoeffs& other) const
{
    return shares_data(other._coeff_matrix);
}

bool Dwt2dCoeffs::shares_data(const cv::Mat& matrix) const
{
    return _coeff_matrix.datastart == matrix.datastart;
}

void Dwt2dCoeffs::normalize(int approx_mode, int detail_mode)
{
    if (approx_mode == DWT_NO_NORMALIZE && detail_mode == DWT_NO_NORMALIZE)
        return;

    auto max_abs_value = maximum_abs_value();
    if (approx_mode == detail_mode) {
        auto [alpha, beta] = normalization_constants(detail_mode, max_abs_value);
        _coeff_matrix = alpha * _coeff_matrix + beta;
    } else {
        auto original_approx = approx().clone();

        if (detail_mode != DWT_NO_NORMALIZE) {
            auto [alpha, beta] = normalization_constants(detail_mode, max_abs_value);
            _coeff_matrix = alpha * _coeff_matrix + beta;
        }

        if (approx_mode != DWT_NO_NORMALIZE) {
            auto [alpha, beta] = normalization_constants(approx_mode, max_abs_value);
            set_approx(alpha * original_approx + beta);
        } else {
            set_approx(original_approx);
        }
    }
}

double Dwt2dCoeffs::maximum_abs_value() const
{
    double min = 0;
    double max = 0;
    cv::minMaxLoc(_coeff_matrix, &min, &max);
    return std::max({std::abs(min), std::abs(max)});
}

std::pair<double, double> Dwt2dCoeffs::normalization_constants(int normalization_mode, double max_abs_value) const
{
    double alpha = 1.0;
    double beta = 0.0;
    switch (normalization_mode) {
        case DWT_ZERO_TO_HALF_NORMALIZE:
            alpha = 0.5 / max_abs_value;
            beta = 0.5;
            break;
        case DWT_MAX_NORMALIZE:
            alpha = 1.0 / max_abs_value;
            beta = 0.0;
            break;
    }

    return std::make_pair(alpha, beta);
}

std::ostream& operator<<(std::ostream& stream, const Dwt2dCoeffs& coeffs)
{
    stream << coeffs._coeff_matrix;
    return stream;
}
} // namespace internal




/**
 * =============================================================================
 * Public API
 * =============================================================================
*/
DWT2D::DWT2D(const Wavelet& wavelet, int border_type) :
    wavelet(wavelet),
    border_type(border_type)
{
}

void DWT2D::check_levels_nonnegative(int levels, const std::string levels_name) const
{
    if (levels < 0) {
        std::stringstream message;
        message
            << levels_name << " is out of range. "
            << "Must be " << levels_name << " > 0 - "
            << "got " << levels << ".";
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }
}

int DWT2D::max_possible_levels(cv::InputArray x)
{
    return x.empty() ? 0 : max_possible_levels(x.size());
}

int DWT2D::max_possible_levels(int rows, int cols)
{
    return std::log2(std::min(rows, cols));
}

int DWT2D::max_possible_levels(const cv::Size& size)
{
    return max_possible_levels(size.height, size.width);
}

DWT2D::Coeffs DWT2D::operator()(cv::InputArray x, int levels) const
{
    return forward(x, levels);
}




DWT2D::Coeffs DWT2D::forward(cv::InputArray x, int levels) const
{
    check_levels_nonnegative(levels);

    int max_levels = max_possible_levels(x);
    levels = (levels == 0) ? max_levels : std::min(max_levels, levels);

    auto data = x.getMat();
    DWT2D::Coeffs coeffs(data.size(), data.type(), levels);
    for (int level = 0; level < levels; ++level) {
        cv::Mat approx;
        cv::Mat horizontal_detail;
        cv::Mat vertical_detail;
        cv::Mat diagonal_detail;
        wavelet.filter_bank().forward(
            data,
            approx,
            horizontal_detail,
            vertical_detail,
            diagonal_detail
        );

        data = approx;
        coeffs.set_horizontal_detail(horizontal_detail, level);
        coeffs.set_vertical_detail(vertical_detail, level);
        coeffs.set_diagonal_detail(diagonal_detail, level);
    }

    coeffs.set_approx(data);

    return coeffs;
}

void DWT2D::inverse(const DWT2D::Coeffs& coeffs, cv::OutputArray output, int levels) const
{
    check_levels_nonnegative(levels);

    levels = (levels > 0) ? std::max(coeffs.levels() - levels, 0) : 0;
    cv::Mat approx = coeffs.approx();
    for (int level = coeffs.levels() - 1; level >= levels; --level) {
        cv::Mat result;
        wavelet.filter_bank().inverse(
            approx,
            coeffs.horizontal_detail(level),
            coeffs.vertical_detail(level),
            coeffs.diagonal_detail(level),
            result
        );
        approx = result;
    }

    output.assign(approx);
}

cv::Mat DWT2D::inverse(const DWT2D::Coeffs& coeffs, int levels) const
{
    cv::Mat output;
    inverse(coeffs, output, levels);
    return output;
}




/**
 * -----------------------------------------------------------------------------
 * Functional Interface
 * -----------------------------------------------------------------------------
*/
DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const Wavelet& wavelet,
    int levels,
    int border_type
)
{
    DWT2D dwt(wavelet, border_type);
    return dwt(input, levels);
}

DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const std::string& wavelet,
    int levels,
    int border_type
)
{
    return dwt2d(input, Wavelet::create(wavelet), levels, border_type);
}

void dwt2d(
    cv::InputArray input,
    cv::OutputArray output,
    const Wavelet& wavelet,
    int levels,
    int border_type
)
{
    output.assign(dwt2d(input, wavelet, levels, border_type));
}

void dwt2d(
    cv::InputArray input,
    cv::OutputArray output,
    const std::string& wavelet,
    int levels,
    int border_type
)
{
    dwt2d(input, output, Wavelet::create(wavelet), levels, border_type);
}

void idwt2d(
    const DWT2D::Coeffs& input,
    cv::OutputArray output,
    const Wavelet& wavelet,
    int levels,
    int border_type
)
{
    DWT2D dwt(wavelet, border_type);
    dwt.inverse(input, output);
}

void idwt2d(
    const DWT2D::Coeffs& input,
    cv::OutputArray output,
    const std::string& wavelet,
    int levels,
    int border_type
)
{
    idwt2d(input, output, Wavelet::create(wavelet), border_type);
}

cv::Mat idwt2d(
    const DWT2D::Coeffs& input,
    const Wavelet& wavelet,
    int levels,
    int border_type
)
{
    DWT2D dwt(wavelet, border_type);
    return dwt.inverse(input, levels);
}

cv::Mat idwt2d(
    const DWT2D::Coeffs& input,
    const std::string& wavelet,
    int levels,
    int border_type
)
{
    return idwt2d(input, Wavelet::create(wavelet), border_type);
}

} // namespace wavelet

