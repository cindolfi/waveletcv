#include "wavelet/dwt2d.hpp"
#include <opencv2/imgproc.hpp>

namespace wavelet
{

namespace internal
{

Dwt2dCoeffs::Dwt2dCoeffs() :
    _coeff_matrix(),
    _depth(0)
{
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Mat& matrix) :
    Dwt2dCoeffs(matrix, 0)
{
}

Dwt2dCoeffs::Dwt2dCoeffs(cv::Mat&& matrix) :
    _coeff_matrix(matrix),
    _depth(DWT2D::max_possible_depth(_coeff_matrix))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Mat& matrix, int depth) :
    _coeff_matrix(matrix),
    _depth(depth > 0 ? depth : DWT2D::max_possible_depth(matrix))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(int rows, int cols, int type, int depth) :
    _coeff_matrix(rows, cols, type, 0.0),
    _depth(depth > 0 ? depth : DWT2D::max_possible_depth(rows, cols))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Size& size, int type, int depth) :
    Dwt2dCoeffs(size.height, size.width, type, depth)
{
}

Dwt2dCoeffs Dwt2dCoeffs::clone() const
{
    return Dwt2dCoeffs(_coeff_matrix.clone(), _depth);
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

    _depth = DWT2D::max_possible_depth(_coeff_matrix);

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(cv::Mat&& matrix)
{
    check_size_for_assignment(matrix);
    _coeff_matrix = matrix;
    _depth = DWT2D::max_possible_depth(_coeff_matrix);

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::MatExpr& matrix)
{
    check_size_for_assignment(matrix);
    _coeff_matrix = matrix;
    _depth = DWT2D::max_possible_depth(_coeff_matrix);

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::Scalar& scalar)
{
    _coeff_matrix = scalar;
    _depth = DWT2D::max_possible_depth(_coeff_matrix);

    return *this;
}

std::vector<cv::Mat> Dwt2dCoeffs::collect_details(int direction) const
{
    std::vector<cv::Mat> result;
    for (int level = 0; level < depth(); ++level)
        result.push_back(detail(direction, level));

    return result;
}

Dwt2dCoeffs Dwt2dCoeffs::at(int level) const
{
    if (level == 0)
        return *this;

    //  TODO throw exception
    if (level >= _depth)
        return Dwt2dCoeffs(0, 0, type());

    return Dwt2dCoeffs(
        _coeff_matrix(level_rect(level)),
        _depth - level
    );
}

cv::Mat Dwt2dCoeffs::approx() const
{
    return _coeff_matrix(approx_rect(depth() - 1));
}

cv::Mat Dwt2dCoeffs::detail(int direction, int level) const
{
    switch (direction) {
        case HORIZONTAL: return horizontal_detail(level);
        case VERTICAL: return vertical_detail(level);
        case DIAGONAL: return diagonal_detail(level);
        default:
            //  TODO throw exception
            break;
    }

    return cv::Mat();
}

cv::Mat Dwt2dCoeffs::horizontal_detail(int level) const
{
    return _coeff_matrix(horizontal_rect(level));
}

cv::Mat Dwt2dCoeffs::vertical_detail(int level) const
{
    return _coeff_matrix(vertical_rect(level));
}

cv::Mat Dwt2dCoeffs::diagonal_detail(int level) const
{
    return _coeff_matrix(diagonal_rect(level));
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
    check_size_for_set_level(coeffs, depth());
    convert_and_copy(coeffs, approx());
}

void Dwt2dCoeffs::set_approx(const cv::Scalar& scalar)
{
    approx() = scalar;
}

void Dwt2dCoeffs::set_detail(const cv::Mat& coeffs, int direction, int level)
{
    check_size_for_set_detail(coeffs, level);
    convert_and_copy(coeffs, detail(direction, level));
}

void Dwt2dCoeffs::set_detail(const cv::Scalar& scalar, int direction, int level)
{
    detail(direction, level) = scalar;
}

void Dwt2dCoeffs::set_horizontal_detail(const cv::Mat& coeffs, int level)
{
    check_size_for_set_detail(coeffs, level);
    convert_and_copy(coeffs, horizontal_detail(level));
}

void Dwt2dCoeffs::set_horizontal_detail(const cv::Scalar& scalar, int level)
{
    horizontal_detail(level) = scalar;
}

void Dwt2dCoeffs::set_vertical_detail(const cv::Mat& coeffs, int level)
{
    check_size_for_set_detail(coeffs, level);
    convert_and_copy(coeffs, vertical_detail(level));
}

void Dwt2dCoeffs::set_vertical_detail(const cv::Scalar& scalar, int level)
{
    vertical_detail(level) = scalar;
}

void Dwt2dCoeffs::set_diagonal_detail(const cv::Mat& coeffs, int level)
{
    check_size_for_set_detail(coeffs, level);
    convert_and_copy(coeffs, diagonal_detail(level));
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
            << "got " << matrix.size();
        CV_Error(cv::Error::BadImageSize, message.str());
    }
}

template <typename MatrixLike>
void Dwt2dCoeffs::check_size_for_set_level(const MatrixLike& matrix, int level) const
{
    if (level < 0 || level > depth()) {
        std::stringstream message;
        message
            << "Level is out of range. "
            << "Must be 0 <= level < " << depth() << " - "
            << "got " << level;
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }

    auto level_size = size() / int(std::pow(2, level));
    if (matrix.size() != level_size) {
        std::stringstream message;
        message
            << "Cannot set the coeffs at level " << level << ".  "
            << "The size of the matrix must be must be " << level_size << " - "
            << "got " << matrix.size() << ")";
        CV_Error(cv::Error::BadImageSize, message.str());
    }
}

template <typename MatrixLike>
void Dwt2dCoeffs::check_size_for_set_detail(const MatrixLike& matrix, int level) const
{
    if (level < 0 || level >= depth()) {
        std::stringstream message;
        message
            << "Level is out of range. "
            << "Must be 0 <= level < " << depth() << " - "
            << "got " << level;
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }

    auto detail_size = size() / int(std::pow(2, 1 + level));
    if (matrix.size() != detail_size) {
        std::stringstream message;
        message
            << "Cannot set the details at level " << level << ".  "
            << "The size of the matrix must be must be " << detail_size << " - "
            << "got " << matrix.size() << ")";
        CV_Error(cv::Error::BadImageSize, message.str());
    }
}
#else
template <typename MatrixLike>
inline void Dwt2dCoeffs::check_size_for_assignment(const MatrixLike& matrix) const {}

template <typename MatrixLike>
inline void Dwt2dCoeffs::check_size_for_set_level(const MatrixLike& matrix, int level) const {}

template <typename MatrixLike>
inline void Dwt2dCoeffs::check_size_for_set_detail(const MatrixLike& matrix, int level) const {}
#endif

cv::Size Dwt2dCoeffs::level_size(int level) const
{
    //  TODO: throw exception
    if (level < 0 || level >= depth())
        return cv::Size(0, 0);

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
    //  TODO: throw exception
    if (level < 0 || level >= depth())
        return cv::Size(0, 0);

    return _coeff_matrix.size() / int(std::pow(2, 1 + level));
}

cv::Rect Dwt2dCoeffs::approx_rect(int level) const
{
    auto size = detail_size(level);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(0, 0), size);
}

cv::Rect Dwt2dCoeffs::horizontal_rect(int level) const
{
    auto size = detail_size(level);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(0, size.height), size);
}

cv::Rect Dwt2dCoeffs::vertical_rect(int level) const
{
    auto size = detail_size(level);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(size.width, 0), size);
}

cv::Rect Dwt2dCoeffs::diagonal_rect(int level) const
{
    auto size = detail_size(level);
    if (size.empty())
        return cv::Rect();

    return cv::Rect(cv::Point(size.width, size.height), size);
}

Dwt2dCoeffs::LevelIterator Dwt2dCoeffs::begin() const
{
    return LevelIterator(this, 0);
}

Dwt2dCoeffs::LevelIterator Dwt2dCoeffs::end() const
{
    return LevelIterator(this, depth());
}

Dwt2dCoeffs::LevelIterator Dwt2dCoeffs::begin()
{
    return LevelIterator(this, 0);
}

Dwt2dCoeffs::LevelIterator Dwt2dCoeffs::end()
{
    return LevelIterator(this, depth());
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
    // if (approx_mode == DWT_NORMALIZE_NONE && detail_mode == DWT_NORMALIZE_NONE)
    //     return;

    // auto max_abs_value = maximum_abs_value();

    // if (approx_mode != DWT_NORMALIZE_NONE) {
    //     auto [alpha, beta] = normalization_constants(approx_mode, max_abs_value);
    //     for (auto& level : coeffs) {
    //         level.approx() = alpha * level.approx() + beta;
    //     }
    // }

    // if (approx_mode != DWT_NORMALIZE_NONE) {
    //     auto [alpha, beta] = normalization_constants(detail_mode, max_abs_value);
    //     for (auto& level : coeffs) {
    //         level.horizontal_detail() = alpha * level.horizontal_detail() + beta;
    //         level.vertical_detail() = alpha * level.vertical_detail() + beta;
    //         level.diagonal_detail() = alpha * level.diagonal_detail() + beta;
    //     }
    // }
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
        case DWT_NORMALIZE_ZERO_TO_HALF:
            alpha = 0.5 / max_abs_value;
            beta = 0.5;
            break;
        case DWT_NORMALIZE_MAX:
            alpha = 1.0 / max_abs_value;
            beta = 0.0;
            break;
    }

    return std::make_pair(alpha, beta);
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

int DWT2D::max_possible_depth(cv::InputArray x)
{
    return x.empty() ? 0 : max_possible_depth(x.size());
}

int DWT2D::max_possible_depth(int rows, int cols)
{
    return std::log2(std::min(rows, cols));
}

int DWT2D::max_possible_depth(const cv::Size& size)
{
    return max_possible_depth(size.height, size.width);
}

DWT2D::Coeffs DWT2D::operator()(cv::InputArray x, int levels) const
{
    return forward(x, levels);
}

DWT2D::Coeffs DWT2D::forward(cv::InputArray x, int levels) const
{
    int depth = max_possible_depth(x);
    if (levels > 0)
        depth = std::min(depth, levels);

    if (depth <= 0)
        return DWT2D::Coeffs(cv::Size(), x.type());

    auto data = x.getMat();
    DWT2D::Coeffs coeffs(data.size(), data.type(), depth);
    for (int level = 0; level < depth; ++level) {
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
    int depth = 0;
    if (levels > 0)
        depth = std::max(coeffs.depth() - levels, 0);

    cv::Mat approx = coeffs.approx();
    for (int level = coeffs.depth() - 1; level >= depth; --level) {
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

