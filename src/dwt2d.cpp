#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "wavelet/dwt2d.hpp"
#include "wavelet/utils.hpp"

namespace wavelet
{

namespace internal
{
Dwt2dCoeffs::Dwt2dCoeffs() :
    _p(std::make_shared<Dwt2dCoeffsImpl>())
{
}

Dwt2dCoeffs::Dwt2dCoeffs(
    const cv::Mat& coeff_matrix,
    int levels,
    const cv::Size& input_size,
    const std::vector<cv::Size>& subband_sizes,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
) :
    _p(
        std::make_shared<Dwt2dCoeffsImpl>(
            coeff_matrix,
            levels,
            input_size,
            subband_sizes,
            wavelet,
            border_type
        )
    )
{
}

Dwt2dCoeffs::Dwt2dCoeffs(
    const cv::Mat& coeff_matrix,
    int levels,
    const cv::Size& input_size,
    const std::vector<cv::Rect>& diagonal_subband_rects,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
) :
    _p(
        std::make_shared<Dwt2dCoeffsImpl>(
            coeff_matrix,
            levels,
            input_size,
            diagonal_subband_rects,
            wavelet,
            border_type
        )
    )
{
}

DWT2D Dwt2dCoeffs::dwt() const
{
    return DWT2D(wavelet(), border_type());
}

cv::Mat Dwt2dCoeffs::invert() const
{
    return dwt().inverse(*this);
}

void Dwt2dCoeffs::invert(cv::OutputArray output) const
{
    dwt().inverse(*this, output);
}

Dwt2dCoeffs Dwt2dCoeffs::clone() const
{
    return Dwt2dCoeffs(
        _p->coeff_matrix.clone(),
        _p->levels,
        _p->input_size,
        _p->diagonal_subband_rects,
        _p->wavelet,
        _p->border_type
    );
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::Mat& matrix)
{
    check_size_for_assignment(matrix);

    if (matrix.type() != type()) {
        cv::Mat converted;
        matrix.convertTo(converted, type());
        converted.copyTo(_p->coeff_matrix);
    } else {
        matrix.copyTo(_p->coeff_matrix);
    }

    // _coeff_matrix = matrix;

    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::MatExpr& matrix)
{
    check_size_for_assignment(matrix);
    _p->coeff_matrix = matrix;
    return *this;
}

Dwt2dCoeffs& Dwt2dCoeffs::operator=(const cv::Scalar& scalar)
{
    _p->coeff_matrix = scalar;
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

    std::vector<cv::Rect> detail_rects;
    detail_rects.insert(
        detail_rects.begin(),
        _p->diagonal_subband_rects.begin() + level,
        _p->diagonal_subband_rects.end()
    );

    return Dwt2dCoeffs(
        _p->coeff_matrix(level_rect(level)),
        levels() - level,
        input_size(level),
        detail_rects,
        wavelet(),
        border_type()
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
    return level_rect(level).size();
}

cv::Rect Dwt2dCoeffs::level_rect(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);

    auto rect = _p->diagonal_subband_rects[level];
    return cv::Rect(cv::Point(0, 0), rect.br());
}

cv::Size Dwt2dCoeffs::detail_size(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);

    return _p->diagonal_subband_rects[level].size();
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
    check_nonempty();
    auto rect = _p->diagonal_subband_rects[levels() - 1];
    return rect - rect.tl();
}

cv::Rect Dwt2dCoeffs::horizontal_detail_rect(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);

    auto detail_rect = _p->diagonal_subband_rects[level];
    detail_rect.x = 0;

    return detail_rect;
}

cv::Rect Dwt2dCoeffs::vertical_detail_rect(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);

    auto detail_rect = _p->diagonal_subband_rects[level];
    detail_rect.y = 0;

    return detail_rect;
}

cv::Rect Dwt2dCoeffs::diagonal_detail_rect(int level) const
{
    check_level_in_range(level);
    level = resolve_level(level);

    return _p->diagonal_subband_rects[level];
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

cv::Mat Dwt2dCoeffs::detail_mask(const cv::Range& levels) const
{
    return levels == cv::Range::all() ? detail_mask() : detail_mask(levels.start, levels.end);
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
    return shares_data(other._p->coeff_matrix);
}

bool Dwt2dCoeffs::shares_data(const cv::Mat& matrix) const
{
    return _p->coeff_matrix.datastart == matrix.datastart;
}

void Dwt2dCoeffs::normalize(NormalizationMode approx_mode, NormalizationMode detail_mode)
{
    if (approx_mode == DWT_NO_NORMALIZE && detail_mode == DWT_NO_NORMALIZE)
        return;

    auto max_abs_value = maximum_abs_value();
    if (approx_mode == detail_mode) {
        auto [alpha, beta] = normalization_constants(detail_mode, max_abs_value);
        _p->coeff_matrix = alpha * _p->coeff_matrix + beta;
    } else {
        auto original_approx = approx().clone();

        if (detail_mode != DWT_NO_NORMALIZE) {
            auto [alpha, beta] = normalization_constants(detail_mode, max_abs_value);
            _p->coeff_matrix = alpha * _p->coeff_matrix + beta;
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
    cv::minMaxLoc(_p->coeff_matrix, &min, &max);
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
            << "DWT2D::Coeffs: Cannot assign matrix to this.  "
            << "The size of matrix must be " << size() << "), "
            << "got matrix.size() = " << matrix.size() << ".";
        CV_Error(cv::Error::StsBadSize, message.str());
    }

    if (matrix.channels() != channels()) {
        std::stringstream message;
        message
            << "DWT2D::Coeffs: Cannot assign matrix to this.  "
            << "The number of channels of matrix must be " << channels() << "), "
            << "got matrix.channels() = " << matrix.channels() << ".";
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
inline void Dwt2dCoeffs::check_size_for_assignment(cv::InputArray other) const {}
inline void Dwt2dCoeffs::check_size_for_set_level(const MatrixLike& matrix, int level) const {}
inline void Dwt2dCoeffs::check_size_for_set_approx(const cv::Mat& matrix) const {}
inline void Dwt2dCoeffs::check_size_for_set_detail(const MatrixLike& matrix, int level, int subband) const {}
inline void Dwt2dCoeffs::check_nonnegative_level(int level, const std::string level_name) const {}
inline void Dwt2dCoeffs::check_level_in_range(int level, const std::string level_name) const {}
inline void Dwt2dCoeffs::check_subband(int subband) const {}
#endif

std::ostream& operator<<(std::ostream& stream, const Dwt2dCoeffs& coeffs)
{
    stream << coeffs._p->coeff_matrix << "\n(" << coeffs.levels() << " levels)";
    return stream;
}
} // namespace internal

// std::vector<internal::Dwt2dCoeffs> split(const internal::Dwt2dCoeffs& coeffs)
// {
//     std::vector<cv::Mat> coeffs_channels;
//     cv::split(coeffs, coeffs_channels);

//     std::vector<DWT2D::Coeffs> result;
//     for (const auto& coeff_matrix : coeffs_channels) {
//         result.emplace_back(
//             coeff_matrix,
//             coeffs._p->levels,
//             coeffs._p->input_size,
//             coeffs._p->diagonal_subband_rects,
//             coeffs._p->wavelet,
//             coeffs._p->border_type
//         );
//     }

//     return result;
// }

// internal::Dwt2dCoeffs merge(const std::vector<internal::Dwt2dCoeffs>& coeffs)
// {
//     if (coeffs.empty())
//         return internal::Dwt2dCoeffs();

//     cv::Mat coeff_matrix;
//     cv::merge(coeffs, coeff_matrix);

//     return internal::Dwt2dCoeffs(
//         coeff_matrix,
//         coeffs[0]._p->levels,
//         coeffs[0]._p->input_size,
//         coeffs[0]._p->diagonal_subband_rects,
//         coeffs[0]._p->wavelet,
//         coeffs[0]._p->border_type
//     );
// }


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

void DWT2D::forward(cv::InputArray x, DWT2D::Coeffs& output, int levels) const
{
    // check_levels_in_range(levels, x);
    warn_if_border_effects_will_occur(levels, x);
    resolve_forward_output(output, x, levels);

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

void DWT2D::resolve_forward_output(DWT2D::Coeffs& output, cv::InputArray x, int levels) const
{
    bool is_compatible = !output.empty()
        && output.levels() == levels
        && output.channels() == x.channels()
        && output.input_size() == x.size()
        && output.size() == coeffs_size_for_input(x, levels);

    if (!is_compatible)
        output = create_coeffs_for_input(x, levels);
    else if (output.type() != x.type())
        output.convertTo(output, x.type());

    output._p->wavelet = wavelet;
    output._p->border_type = border_type;
    output._p->input_size = x.size();
}

void DWT2D::inverse(const DWT2D::Coeffs& coeffs, cv::OutputArray output) const
{
    warn_if_border_effects_will_occur(coeffs);

    cv::Mat approx = coeffs.approx();
    for (int level = coeffs.levels() - 1; level >= 0; --level) {
        cv::Mat result;
        wavelet.filter_bank().inverse(
            approx,
            coeffs.horizontal_detail(level),
            coeffs.vertical_detail(level),
            coeffs.diagonal_detail(level),
            result,
            coeffs.input_size(level)
        );
        approx = result;
    }

    if (output.isContinuous())
        output.assign(approx);
    else
        approx.copyTo(output);
}

DWT2D::Coeffs DWT2D::create_coeffs(cv::InputArray coeffs_matrix, const cv::Size& input_size, int levels) const
{
    check_levels_in_range(levels);
    check_coeffs_size(coeffs_matrix, input_size, levels);

    std::vector<cv::Size> subband_sizes;
    cv::Size subband_size = input_size;
    for (int i = 0; i < levels; ++i) {
        subband_size = wavelet.filter_bank().subband_size(subband_size);
        subband_sizes.push_back(subband_size);
    }

    return Coeffs(coeffs_matrix.getMat(), levels, input_size, subband_sizes, wavelet, border_type);
}

DWT2D::Coeffs DWT2D::create_coeffs_for_input(const cv::Size& input_size, int type, int levels) const
{
    auto size = coeffs_size_for_input(input_size, levels);

    cv::Mat coeffs_matrix(size, type, 0.0);

    std::vector<cv::Size> subband_sizes;
    cv::Size subband_size = input_size;
    for (int i = 0; i < levels; ++i) {
        subband_size = wavelet.filter_bank().subband_size(subband_size);
        subband_sizes.push_back(subband_size);
    }

    return Coeffs(coeffs_matrix, levels, input_size, subband_sizes, wavelet, border_type);
}

cv::Size DWT2D::coeffs_size_for_input(const cv::Size& input_size, int levels) const
{
    check_levels_in_range(levels);

    cv::Size level_subband_size = wavelet.filter_bank().subband_size(input_size);
    cv::Size accumulator = level_subband_size;
    for (int i = 1; i < levels; ++i) {
        level_subband_size = wavelet.filter_bank().subband_size(level_subband_size);
        accumulator += level_subband_size;
    }

    //  add once more to account for approximation coeffcients
    accumulator += level_subband_size;

    return accumulator;
}

int DWT2D::max_levels_without_border_effects(int rows, int cols) const
{
    double data_length = std::min(rows, cols);
    if (data_length <= 0)
        return 0;

    int max_levels = std::floor(std::log2(data_length / (wavelet.filter_length() - 1.0)));
    return std::max(max_levels, 0);
}

// int DWT2D::max_levels(int rows, int cols) const
// {
//     double data_length = std::min(rows, cols);
//     if (data_length <= 0)
//         return 0;

//     int max_levels = std::floor(std::log2(data_length));
//     return std::max(max_levels, 0);
// }

#ifndef DISABLE_ARG_CHECKS
void DWT2D::check_levels_in_range(int levels) const
{
    if (levels < 1) {
        std::stringstream message;
        message
            << "DWT2D: levels is out of range. "
            << "Must be levels >= 1, "
            << "got levels = " << levels << ".";
        CV_Error(cv::Error::StsOutOfRange, message.str());
    }
}

// void DWT2D::check_levels_in_range(int levels, int max_levels) const
// {
//     if (levels < 1 || levels > max_levels) {
//         std::stringstream message;
//         message
//             << "DWT2D: levels is out of range. "
//             << "Must be 1 <= levels <= " << max_levels << ", "
//             << "got levels = " << levels << ".";
//         CV_Error(cv::Error::StsOutOfRange, message.str());
//     }
// }

// void DWT2D::check_levels_in_range(int levels, cv::InputArray x) const
// {
//     check_levels_in_range(levels, max_levels(x));
// }

void DWT2D::check_coeffs_size(cv::InputArray coeffs, const cv::Size& input_size, int levels) const
{
    auto required_coeffs_size = coeffs_size_for_input(input_size, levels);
    if (coeffs.size() != required_coeffs_size) {
        std::stringstream message;
        message
            << "DWT2D: coefficients size is not consistent with input size. "
            << "Coefficients size must be " << required_coeffs_size << " "
            << "for input size = " << input_size << " and levels = " << levels << ", "
            << "got coeffs.size() = " << coeffs.size() << ". "
            << "(Note: use DWT2D::coeffs_size_for_input() to get the required size)";
        CV_Error(cv::Error::StsBadSize, message.str());
    }
}
#else
inline void DWT2D::check_levels_in_range(int levels, int max_levels) const {}
inline void DWT2D::check_levels_in_range(int levels, cv::InputArray x) const {}
inline void DWT2D::check_coeffs_size(cv::InputArray coeffs, const cv::Size& input_size, int levels) const {}
#endif // DISABLE_ARG_CHECKS

#ifndef DISABLE_DWT_WARNINGS
void DWT2D::warn_if_border_effects_will_occur(int levels, const cv::Size& input_size) const
{
    int max_levels = max_levels_without_border_effects(input_size);
    if (levels > max_levels) {
        std::stringstream message;
        message
            << "DWT2D: border effects will occur for a " << levels << " level DWT "
            << "of a " << input_size << " input using the " << wavelet.name() << " wavelet. "
            << "Must have levels <= " << max_levels << " to avoid border effects.";
        CV_LOG_WARNING(NULL, message.str());
    }
}

void DWT2D::warn_if_border_effects_will_occur(int levels, cv::InputArray x) const
{
    warn_if_border_effects_will_occur(levels, x.size());
}

void DWT2D::warn_if_border_effects_will_occur(const Coeffs& coeffs) const
{
    warn_if_border_effects_will_occur(
        coeffs.levels(),
        coeffs.input_size()
    );
}

#else
inline void DWT2D::warn_if_border_effects_will_occur(int levels, int max_levels) const {}
inline void DWT2D::warn_if_border_effects_will_occur(int levels, cv::InputArray x) const {}
#endif // DISABLE_DWT_WARNINGS


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

} // namespace wavelet

