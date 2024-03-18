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

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Mat& matrix, int levels) :
    _coeff_matrix(matrix),
    _levels(levels)
{
    check_constructor_level(_levels, DWT2D::max_possible_levels(_coeff_matrix));
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Mat& matrix) :
    Dwt2dCoeffs(matrix, DWT2D::max_possible_levels(matrix))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(int rows, int cols, int type) :
    Dwt2dCoeffs(cv::Mat(rows, cols, type, 0.0))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(int rows, int cols, int type, int levels) :
    Dwt2dCoeffs(cv::Mat(rows, cols, type, 0.0), levels)
{
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Size& size, int type, int levels) :
    Dwt2dCoeffs(cv::Mat(size, type, 0.0), levels)
{
}

Dwt2dCoeffs::Dwt2dCoeffs(const cv::Size& size, int type) :
    Dwt2dCoeffs(cv::Mat(size, type, 0.0))
{
}

Dwt2dCoeffs::Dwt2dCoeffs(cv::Mat&& matrix, int levels) :
    _coeff_matrix(matrix),
    _levels(levels)
{
}

Dwt2dCoeffs::Dwt2dCoeffs(cv::Mat&& matrix) :
    Dwt2dCoeffs(matrix, DWT2D::max_possible_levels(matrix))
{
}

Dwt2dCoeffs Dwt2dCoeffs::clone() const
{
    return Dwt2dCoeffs(_coeff_matrix.clone(), _levels);
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::Mat& matrix)
{
    check_size_for_assignment(matrix);

    if (matrix.size() != size())
        _levels = DWT2D::max_possible_levels(_coeff_matrix);

    if (matrix.type() != _coeff_matrix.type()) {
        cv::Mat matrix2;
        matrix.convertTo(matrix2, _coeff_matrix.type());
        matrix2.copyTo(_coeff_matrix);
    } else {
        matrix.copyTo(_coeff_matrix);
    }

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(cv::Mat&& matrix)
{
    check_size_for_assignment(matrix);
    if (matrix.size() != size())
        _levels = DWT2D::max_possible_levels(_coeff_matrix);

    _coeff_matrix = matrix;

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::MatExpr& matrix)
{
    return this->operator=(cv::Mat(matrix));
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::Scalar& scalar)
{
    _coeff_matrix = scalar;
    return *this;
}

std::vector<cv::Mat> Dwt2dCoeffs::collect_details(int subband) const
{
    std::vector<cv::Mat> result;
    for (int level = 0; level < levels(); ++level)
        result.push_back(detail(level, subband));

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

cv::Mat Dwt2dCoeffs::detail(int level, int subband) const
{
    check_subband(subband);
    switch (subband) {
        case HORIZONTAL: return horizontal_detail(level);
        case VERTICAL: return vertical_detail(level);
        case DIAGONAL: return diagonal_detail(level);
    }

    return cv::Mat();
}

void Dwt2dCoeffs::convert_and_copy(const cv::Mat& source, const cv::Mat& destination)
{
    if (source.type() != destination.type()) {
        cv::Mat converted;
        source.convertTo(converted, type());
        converted.copyTo(destination);
    } else {
        source.copyTo(destination);
    }
}

cv::Size Dwt2dCoeffs::level_size(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);
    return _coeff_matrix.size() / int(std::pow(2, level));
}

cv::Rect Dwt2dCoeffs::level_rect(int level) const
{
    auto size = level_size(level);
    return cv::Rect(cv::Point(0, 0), size);
}

cv::Size Dwt2dCoeffs::detail_size(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);
    return _coeff_matrix.size() / int(std::pow(2, 1 + level));
}

cv::Rect Dwt2dCoeffs::detail_rect(int level, int subband) const
{
    check_subband(subband);
    switch (subband) {
        case HORIZONTAL: return horizontal_detail_rect(level);
        case VERTICAL: return vertical_detail_rect(level);
        case DIAGONAL: return diagonal_detail_rect(level);
    }

    return cv::Rect();
}

cv::Rect Dwt2dCoeffs::approx_rect() const
{
    auto size = detail_size(-1);
    return cv::Rect(cv::Point(0, 0), size);
}

cv::Rect Dwt2dCoeffs::horizontal_detail_rect(int level) const
{
    auto size = detail_size(level);
    return cv::Rect(cv::Point(0, size.height), size);
}

cv::Rect Dwt2dCoeffs::vertical_detail_rect(int level) const
{
    auto size = detail_size(level);
    return cv::Rect(cv::Point(size.width, 0), size);
}

cv::Rect Dwt2dCoeffs::diagonal_detail_rect(int level) const
{
    auto size = detail_size(level);
    return cv::Rect(cv::Point(size.width, size.height), size);
}

cv::Mat Dwt2dCoeffs::approx_mask() const
{
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
    auto mask = cv::Mat(size(), CV_8U, cv::Scalar(0));
    mask(horizontal_detail_rect(level)) = 255;

    return mask;
}

cv::Mat Dwt2dCoeffs::vertical_detail_mask(int level) const
{
    auto mask = cv::Mat(size(), CV_8U, cv::Scalar(0));
    mask(vertical_detail_rect(level)) = 255;

    return mask;
}

cv::Mat Dwt2dCoeffs::diagonal_detail_mask(int level) const
{
    auto mask = cv::Mat(size(), CV_8U, cv::Scalar(0));
    mask(diagonal_detail_rect(level)) = 255;

    return mask;
}

bool Dwt2dCoeffs::shares_data(const Dwt2dCoeffs& other) const
{
    return shares_data(other._coeff_matrix);
}

bool Dwt2dCoeffs::shares_data(const cv::Mat& matrix) const
{
    return _coeff_matrix.datastart == matrix.datastart;
}

void Dwt2dCoeffs::normalize(NormalizationMode approx_mode, NormalizationMode detail_mode)
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

std::pair<double, double> Dwt2dCoeffs::normalization_constants(
    NormalizationMode normalization_mode,
    double max_abs_value
) const
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

#ifndef DISABLE_ARG_CHECKS
void Dwt2dCoeffs::check_size_for_assignment(cv::InputArray matrix) const
{
    if (matrix.size() != size()) {
        std::stringstream message;
        message
            << "DWT2D::Coeffs: Cannot assign the matrix to this Dwt2dCoeffs.  "
            << "The size of the matrix must be " << size() << "), "
            << "got size = " << matrix.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}

void Dwt2dCoeffs::check_size_for_set_level(const cv::Mat& matrix, int level) const
{
    if (matrix.size() != level_size(level)) {
        std::stringstream message;
        message
            << "DWT2D::Coeffs: Cannot set the coeffs at level " << level << ".  "
            << "The size of the matrix must be " << level_size(level) << ", "
            << "got size = " << matrix.size() << ".";
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
            << "DWT2D::Coeffs: Cannot set the " << subband_name << " detail coefficients at level " << level << ".  "
            << "The size of the matrix must be " << detail_size(level) << ", "
            << "got size = " << matrix.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}

void Dwt2dCoeffs::check_size_for_set_approx(const cv::Mat& matrix) const
{
    if (matrix.size() != detail_size(levels() - 1)) {
        std::stringstream message;
        message
            << "DWT2D::Coeffs: Cannot set the approx coefficients.  "
            << "The size of the matrix must be " << detail_size(levels() - 1) << ", "
            << "got size = " << matrix.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}

void Dwt2dCoeffs::check_level_in_range(int level, const std::string level_name) const
{
    if (level < -levels() || level >= levels()) {
        std::stringstream message;
        message
            << "DWT2D::Coeffs: " << level_name << " is out of range. "
            << "Must be " << -levels() << " <= " << level_name << " < " << levels() << ", "
            << "got " << level_name << " = " << level << ".";
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }
}

void Dwt2dCoeffs::check_constructor_level(int level, int max_level) const
{
    if (level > max_level) {
        std::stringstream message;
        message
            << "DWT2D::Coeffs: level is out of range. "
            << "Must be " << -levels() << " <= level < " << levels() << ", "
            << "got level = " << level << ".";
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }
}

void Dwt2dCoeffs::check_nonempty() const
{
    if (empty()) {
        CV_Error(cv::Error::StsBadSize, "DWT2D::Coeffs: Coefficients are empty.");
    }
}

void Dwt2dCoeffs::check_subband(int subband) const
{
    switch (subband) {
        case HORIZONTAL:
        case VERTICAL:
        case DIAGONAL:
            break;
        default:
            std::stringstream message;
            message
                << "DWT2D::Coeffs: Invalid subband.  "
                << "Must be 0 (HORIZONTAL), 1 (VERTICAL), or 2 (DIAGONAL), "
                << "got " << subband << ".";
            CV_Error(cv::Error::StsBadArg, message.str());
    }
}
#else
inline void Dwt2dCoeffs::check_size_for_assignment(cv::InputArray matrix) const {}
inline void Dwt2dCoeffs::check_size_for_set_level(const MatrixLike& matrix, int level) const {}
inline void Dwt2dCoeffs::check_size_for_set_approx(const cv::Mat& matrix) const {}
inline void Dwt2dCoeffs::check_size_for_set_detail(const MatrixLike& matrix, int level, int subband) const {}
inline void Dwt2dCoeffs::check_nonnegative_level(int level, const std::string level_name) const {}
inline void Dwt2dCoeffs::check_level_in_range(int level, const std::string level_name) const {}
inline void Dwt2dCoeffs::check_subband(int subband) const {}
#endif

std::ostream& operator<<(std::ostream& stream, const Dwt2dCoeffs& coeffs)
{
    stream << coeffs._coeff_matrix << "\n(" << coeffs.levels() << " levels)";
    return stream;
}
} // namespace internal




/**
 * =============================================================================
 * Public API
 * =============================================================================
*/
DWT2D::DWT2D(const Wavelet& wavelet, cv::BorderTypes border_type) :
    wavelet(wavelet),
    border_type(border_type)
{
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

DWT2D::Coeffs DWT2D::forward(cv::InputArray x) const
{
    DWT2D::Coeffs coeffs;
    forward(x, coeffs);
    return coeffs;
}

void DWT2D::forward(cv::InputArray x, DWT2D::Coeffs& output) const
{
    forward(x, output, max_possible_levels(x));
}

DWT2D::Coeffs DWT2D::forward(cv::InputArray x, int levels) const
{
    DWT2D::Coeffs coeffs;
    forward(x, coeffs, levels);
    return coeffs;
}

void DWT2D::forward(cv::InputArray x, DWT2D::Coeffs& output, int levels) const
{
    check_levels_in_range(levels, 1, x);
    create_like(x, output, levels);
    _forward(x, output, levels);
}

DWT2D::Coeffs DWT2D::running_forward(const DWT2D::Coeffs& coeffs, int levels) const
{
    DWT2D::Coeffs result;
    running_forward(coeffs, result, levels);
    return result;
}

void DWT2D::running_forward(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& output, int levels) const
{
    check_levels_in_range(levels, 0, coeffs.approx());
    copy_if_not_identical(coeffs, output);

    //  We are using output.approx() as a buffer to write into here.  That is,
    //  we are not interpreting output.approx() as a full set of cofficients.
    //  Its just a matter of not incurring another allocation.
    auto new_levels_coeffs = DWT2D::Coeffs(output.approx(), levels);

    _forward(coeffs.approx(), new_levels_coeffs, levels);
    output._levels = coeffs.levels() + levels;
    output.set_level(coeffs.levels(), new_levels_coeffs);
}

DWT2D::Coeffs DWT2D::running_forward(const cv::Mat& x, int levels) const
{
    return (levels == 1) ? forward(x, 1) : running_forward(forward(x, 1), levels - 1);
}

void DWT2D::running_forward(const cv::Mat& x, Coeffs& output, int levels) const
{
    if (levels == 1) {
        forward(x, output, 1);
    } else {
        forward(x, output, 1);
        running_forward(output, output, levels - 1);
    }
}

void DWT2D::_forward(cv::InputArray x, DWT2D::Coeffs& output, int levels) const
{
    auto running_approx = x.getMat();
    for (int level = 0; level < levels; ++level) {
        cv::Mat approx;
        cv::Mat horizontal_detail;
        cv::Mat vertical_detail;
        cv::Mat diagonal_detail;

        wavelet.filter_bank().forward(
            running_approx,
            approx,
            horizontal_detail,
            vertical_detail,
            diagonal_detail
        );

        running_approx = approx;
        output.set_horizontal_detail(level, horizontal_detail);
        output.set_vertical_detail(level, vertical_detail);
        output.set_diagonal_detail(level, diagonal_detail);
    }

    output.set_approx(running_approx);
}

cv::Mat DWT2D::inverse(const DWT2D::Coeffs& coeffs) const
{
    cv::Mat output;
    inverse(coeffs, output);
    return output;
}

void DWT2D::inverse(const DWT2D::Coeffs& coeffs, cv::OutputArray output) const
{
    cv::Mat approx = coeffs.approx();
    for (int level = coeffs.levels() - 1; level >= 0; --level) {
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
    if (output.isContinuous())
        output.assign(approx);
    else
        approx.copyTo(output);
}

DWT2D::Coeffs DWT2D::running_inverse(const DWT2D::Coeffs& coeffs, int levels) const
{
    DWT2D::Coeffs output;
    running_inverse(coeffs, output, levels);
    return output;
}

void DWT2D::running_inverse(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& output, int levels) const
{
    check_levels_in_range(levels, 0, coeffs.levels());

    if (levels == 0) {
        copy_if_not_identical(coeffs, output);
    } else {
        int stop_level = coeffs.levels() - levels;
        if (stop_level == 0) {
            create_like(coeffs, output);
            inverse(coeffs, output);
        } else {
            copy_if_not_identical(coeffs, output);
            inverse(coeffs.at_level(stop_level), output.at_level(stop_level));
        }
        output._levels = stop_level;
    }
}

void DWT2D::copy_if_not_identical(const DWT2D::Coeffs& x, DWT2D::Coeffs& output) const
{
    if (&output != &x) {
        if (output.size() == x.size() && output.type() == x.type())
            x.copyTo(output);
        else
            output = x.clone();
    }
}

void DWT2D::create_like(cv::InputArray x, DWT2D::Coeffs& output, int levels) const
{
    if (output.size() != x.size() || output.type() != x.type())
        output = DWT2D::Coeffs(x.size(), x.type(), levels);
    else
        output._levels = levels;
}

void DWT2D::create_like(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& output) const
{
    if (&output != &coeffs)
        create_like(coeffs, output, coeffs.levels());
}

#ifndef DISABLE_ARG_CHECKS
void DWT2D::check_levels_in_range(int levels, int min_levels, int max_levels) const
{
    if (levels < min_levels || levels > max_levels) {
        std::stringstream message;
        message
            << "DWT2D: levels is out of range. "
            << "Must be " << min_levels << " <= levels <= " << max_levels << ", "
            << "got levels = " << levels << ".";
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }
}

void DWT2D::check_levels_in_range(int levels, int min_levels, cv::InputArray x) const
{
    check_levels_in_range(levels, min_levels, max_possible_levels(x));
}
#else
inline void DWT2D::check_levels_in_range(int levels, int min_levels, int max_levels) const {}
inline void DWT2D::check_levels_in_range(int levels, int min_levels, cv::InputArray x) const {}
#endif // DISABLE_ARG_CHECKS



/**
 * -----------------------------------------------------------------------------
 * Functional Interface
 * -----------------------------------------------------------------------------
*/
DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).forward(input);
}

DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const std::string& wavelet,
    cv::BorderTypes border_type
)
{
    return dwt2d(input, Wavelet::create(wavelet), border_type);
}

DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).forward(input, levels);
}

DWT2D::Coeffs dwt2d(
    cv::InputArray input,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return dwt2d(input, Wavelet::create(wavelet), levels, border_type);
}

void dwt2d(
    cv::InputArray input,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).forward(input, output);
}

void dwt2d(
    cv::InputArray input,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    cv::BorderTypes border_type
)
{
    dwt2d(input, output, Wavelet::create(wavelet), border_type);
}

void dwt2d(
    cv::InputArray input,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).forward(input, output, levels);
}

void dwt2d(
    cv::InputArray input,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    dwt2d(input, output, Wavelet::create(wavelet), levels, border_type);
}

DWT2D::Coeffs running_dwt2d(
    const DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).running_forward(coeffs, levels);
}

DWT2D::Coeffs running_dwt2d(
    const DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return running_dwt2d(coeffs, Wavelet::create(wavelet), levels, border_type);
}

void running_dwt2d(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    DWT2D(wavelet, border_type).running_forward(coeffs, output, levels);
}

void running_dwt2d(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    running_dwt2d(coeffs, output, Wavelet::create(wavelet), levels, border_type);
}

void idwt2d(
    const DWT2D::Coeffs& coeffs,
    cv::OutputArray output,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    DWT2D(wavelet, border_type).inverse(coeffs, output);
}

void idwt2d(
    const DWT2D::Coeffs& coeffs,
    cv::OutputArray output,
    const std::string& wavelet,
    cv::BorderTypes border_type
)
{
    idwt2d(coeffs, output, Wavelet::create(wavelet), border_type);
}

cv::Mat idwt2d(
    const DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).inverse(coeffs);
}

cv::Mat idwt2d(
    const DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    cv::BorderTypes border_type
)
{
    return idwt2d(coeffs, Wavelet::create(wavelet), border_type);
}

void running_idwt2d(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& output,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    DWT2D(wavelet, border_type).running_inverse(coeffs, output, levels);
}

void running_idwt2d(
    const DWT2D::Coeffs& coeffs,
    DWT2D::Coeffs& output,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    running_idwt2d(coeffs, output, Wavelet::create(wavelet), border_type);
}

DWT2D::Coeffs running_idwt2d(
    const DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).running_inverse(coeffs, levels);
}

DWT2D::Coeffs running_idwt2d(
    const DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return running_idwt2d(coeffs, Wavelet::create(wavelet), border_type);
}

} // namespace wavelet

