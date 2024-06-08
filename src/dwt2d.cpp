#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "cvwt/dwt2d.hpp"
#include "cvwt/utils.hpp"
#include "cvwt/exception.hpp"
#include <iostream>

namespace cvwt
{
DWT2D::Coeffs::Coeffs() :
    _p(std::make_shared<internal::Dwt2dCoeffsImpl>())
{
}

DWT2D::Coeffs::Coeffs(
    const cv::Mat& coeff_matrix,
    int levels,
    const cv::Size& image_size,
    const std::vector<cv::Size>& subband_sizes,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
) :
    _p(
        std::make_shared<internal::Dwt2dCoeffsImpl>(
            coeff_matrix,
            levels,
            image_size,
            subband_sizes,
            wavelet,
            border_type
        )
    )
{
}

DWT2D::Coeffs::Coeffs(
    const cv::Mat& coeff_matrix,
    int levels,
    const cv::Size& image_size,
    const std::vector<cv::Rect>& diagonal_subband_rects,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
) :
    _p(
        std::make_shared<internal::Dwt2dCoeffsImpl>(
            coeff_matrix,
            levels,
            image_size,
            diagonal_subband_rects,
            wavelet,
            border_type
        )
    )
{
}

void DWT2D::Coeffs::reset(
    const cv::Size& size,
    int type,
    int levels,
    const cv::Size& image_size,
    const std::vector<cv::Size>& subband_sizes,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    _p->coeff_matrix.create(size, type);
    _p->levels = levels;
    _p->image_size = image_size;
    _p->wavelet = wavelet;
    _p->border_type = border_type;
    _p->build_diagonal_subband_rects(subband_sizes);
}

CoeffsExpr DWT2D::Coeffs::mul(cv::InputArray matrix, double scale) const
{
    return CoeffsExpr(*this, _p->coeff_matrix.mul(matrix, scale));
}

CoeffsExpr DWT2D::Coeffs::mul(const CoeffsExpr& expression, double scale) const
{
    return expression.mul(*this, scale);
}

CoeffsExpr DWT2D::Coeffs::mul(const Coeffs& coeffs, double scale) const
{
    return CoeffsExpr(
        *this,
        coeffs,
        _p->coeff_matrix.mul(coeffs._p->coeff_matrix, scale)
    );
}

DWT2D DWT2D::Coeffs::dwt() const
{
    return DWT2D(wavelet(), border_type());
}

cv::Mat DWT2D::Coeffs::reconstruct() const
{
    return dwt().reconstruct(*this);
}

void DWT2D::Coeffs::reconstruct(cv::OutputArray image) const
{
    dwt().reconstruct(*this, image);
}

DWT2D::Coeffs DWT2D::Coeffs::clone() const
{
    return DWT2D::Coeffs(
        _p->coeff_matrix.clone(),
        _p->levels,
        _p->image_size,
        _p->diagonal_subband_rects,
        _p->wavelet,
        _p->border_type
    );
}

DWT2D::Coeffs DWT2D::Coeffs::empty_clone() const
{
    return DWT2D::Coeffs(
        cv::Mat(0, 0, type()),
        _p->levels,
        _p->image_size,
        _p->diagonal_subband_rects,
        _p->wavelet,
        _p->border_type
    );
}

DWT2D::Coeffs DWT2D::Coeffs::clone_and_assign(cv::InputArray coeff_matrix) const
{
    throw_if_wrong_size_for_assignment(coeff_matrix);
    return DWT2D::Coeffs(
        coeff_matrix.getMat(),
        _p->levels,
        _p->image_size,
        _p->diagonal_subband_rects,
        _p->wavelet,
        _p->border_type
    );
}

std::vector<cv::Mat> DWT2D::Coeffs::collect_details(int subband) const
{
    std::vector<cv::Mat> result;
    for (int level = 0; level < levels(); ++level)
        result.push_back(detail(level, subband));

    return result;
}

DWT2D::Coeffs DWT2D::Coeffs::at_level(int level) const
{
    throw_if_level_out_of_range(level);
    level = resolve_level(level);
    if (level == 0)
        return *this;

    std::vector<cv::Rect> detail_rects;
    detail_rects.insert(
        detail_rects.begin(),
        _p->diagonal_subband_rects.begin() + level,
        _p->diagonal_subband_rects.end()
    );

    return DWT2D::Coeffs(
        _p->coeff_matrix(level_rect(level)),
        levels() - level,
        image_size(level),
        detail_rects,
        wavelet(),
        border_type()
    );
}

cv::Mat DWT2D::Coeffs::detail(int level, int subband) const
{
    throw_if_invalid_subband(subband);
    switch (subband) {
        case HORIZONTAL: return horizontal_detail(level);
        case VERTICAL: return vertical_detail(level);
        case DIAGONAL: return diagonal_detail(level);
    }

    return cv::Mat();
}

cv::Size DWT2D::Coeffs::level_size(int level) const
{
    return level_rect(level).size();
}

cv::Rect DWT2D::Coeffs::level_rect(int level) const
{
    throw_if_level_out_of_range(level);
    level = resolve_level(level);

    auto rect = _p->diagonal_subband_rects[level];
    return cv::Rect(cv::Point(0, 0), rect.br());
}

cv::Size DWT2D::Coeffs::detail_size(int level) const
{
    throw_if_level_out_of_range(level);
    level = resolve_level(level);

    return _p->diagonal_subband_rects[level].size();
}

cv::Rect DWT2D::Coeffs::detail_rect(int level, int subband) const
{
    throw_if_invalid_subband(subband);
    switch (subband) {
        case HORIZONTAL: return horizontal_detail_rect(level);
        case VERTICAL: return vertical_detail_rect(level);
        case DIAGONAL: return diagonal_detail_rect(level);
    }

    return cv::Rect();
}

cv::Rect DWT2D::Coeffs::approx_rect() const
{
    throw_if_this_is_empty();
    auto rect = _p->diagonal_subband_rects[levels() - 1];
    return rect - rect.tl();
}

cv::Rect DWT2D::Coeffs::horizontal_detail_rect(int level) const
{
    throw_if_level_out_of_range(level);
    level = resolve_level(level);

    auto detail_rect = _p->diagonal_subband_rects[level];
    detail_rect.x = 0;

    return detail_rect;
}

cv::Rect DWT2D::Coeffs::vertical_detail_rect(int level) const
{
    throw_if_level_out_of_range(level);
    level = resolve_level(level);

    auto detail_rect = _p->diagonal_subband_rects[level];
    detail_rect.y = 0;

    return detail_rect;
}

cv::Rect DWT2D::Coeffs::diagonal_detail_rect(int level) const
{
    throw_if_level_out_of_range(level);
    level = resolve_level(level);

    return _p->diagonal_subband_rects[level];
}

cv::Mat DWT2D::Coeffs::approx_mask() const
{
    auto mask = cv::Mat(size(), CV_8UC1, cv::Scalar(0));
    mask(approx_rect()) = 255;

    return mask;
}

cv::Mat DWT2D::Coeffs::invalid_detail_mask() const
{
    cv::Mat mask(size(), CV_8UC1, cv::Scalar(255));
    for (int level = 0; level < levels(); ++level) {
        mask(horizontal_detail_rect(level)) = 0;
        mask(vertical_detail_rect(level)) = 0;
        mask(diagonal_detail_rect(level)) = 0;
    }
    mask(approx_rect()) = 0;

    return mask;
}

int DWT2D::Coeffs::total_valid() const
{
    cv::Rect approx_rect = this->approx_rect();
    return total_details() + approx_rect.area();
}

int DWT2D::Coeffs::total_details() const
{
    int result = 0;
    for (int level = 0; level < levels(); ++level) {
        auto subband_rect = diagonal_detail_rect(level);
        result += 3 * subband_rect.area();
    }

    return result;
}

cv::Mat DWT2D::Coeffs::detail_mask() const
{
    return detail_mask(cv::Range(0, levels()));
}

cv::Mat DWT2D::Coeffs::detail_mask(int level) const
{
    throw_if_level_out_of_range(level);

    cv::Mat mask(size(), CV_8UC1, cv::Scalar(0));
    mask(horizontal_detail_rect(level)) = 255;
    mask(vertical_detail_rect(level)) = 255;
    mask(diagonal_detail_rect(level)) = 255;

    return mask;
}

cv::Mat DWT2D::Coeffs::detail_mask(const cv::Range& levels) const
{
    if (levels == cv::Range::all())
        return detail_mask();

    auto resolved_levels = resolve_level_range(levels);
    throw_if_levels_out_of_range(resolved_levels.start, resolved_levels.end);

    cv::Mat mask(size(), CV_8UC1, cv::Scalar(0));
    for (int level = resolved_levels.start; level < resolved_levels.end; ++level) {
        mask(horizontal_detail_rect(level)) = 255;
        mask(vertical_detail_rect(level)) = 255;
        mask(diagonal_detail_rect(level)) = 255;
    }

    return mask;
}

cv::Mat DWT2D::Coeffs::detail_mask(int level, int subband) const
{
    throw_if_level_out_of_range(level);
    throw_if_invalid_subband(subband);

    cv::Mat mask(size(), CV_8UC1, cv::Scalar(0));
    mask(detail_rect(resolve_level(level), subband)) = 255;

    return mask;
}

cv::Mat DWT2D::Coeffs::detail_mask(int lower_level, int upper_level, int subband) const
{
    return detail_mask(cv::Range(lower_level, upper_level), subband);
}

cv::Mat DWT2D::Coeffs::detail_mask(const cv::Range& levels, int subband) const
{
    auto resolved_levels = resolve_level_range(levels);
    throw_if_invalid_subband(subband);
    throw_if_levels_out_of_range(resolved_levels.start, resolved_levels.end);

    cv::Mat mask(size(), CV_8UC1, cv::Scalar(0));
    for (int level = resolved_levels.start; level < resolved_levels.end; ++level)
        mask(detail_rect(level, subband)) = 255;

    return mask;
}

cv::Mat DWT2D::Coeffs::horizontal_detail_mask(int level) const
{
    return detail_mask(level, HORIZONTAL);
}

cv::Mat DWT2D::Coeffs::horizontal_detail_mask(const cv::Range& levels) const
{
    return detail_mask(levels, HORIZONTAL);
}

cv::Mat DWT2D::Coeffs::horizontal_detail_mask(int lower_level, int upper_level) const
{
    return detail_mask(lower_level, upper_level, HORIZONTAL);
}

cv::Mat DWT2D::Coeffs::vertical_detail_mask(int level) const
{
    return detail_mask(level, VERTICAL);
}

cv::Mat DWT2D::Coeffs::vertical_detail_mask(const cv::Range& levels) const
{
    return detail_mask(levels, VERTICAL);
}

cv::Mat DWT2D::Coeffs::vertical_detail_mask(int lower_level, int upper_level) const
{
    return detail_mask(lower_level, upper_level, VERTICAL);
}

cv::Mat DWT2D::Coeffs::diagonal_detail_mask(int level) const
{
    return detail_mask(level, DIAGONAL);
}

cv::Mat DWT2D::Coeffs::diagonal_detail_mask(const cv::Range& levels) const
{
    return detail_mask(levels, DIAGONAL);
}

cv::Mat DWT2D::Coeffs::diagonal_detail_mask(int lower_level, int upper_level) const
{
    return detail_mask(lower_level, upper_level, DIAGONAL);
}

DWT2D::Coeffs DWT2D::Coeffs::map_details_to_unit_interval(
    cv::InputArray read_mask,
    cv::InputArray write_mask
) const
{
    double dummy;
    return map_details_to_unit_interval(dummy, read_mask, write_mask);
}

DWT2D::Coeffs DWT2D::Coeffs::map_details_to_unit_interval(
    double& scale_output,
    cv::InputArray read_mask,
    cv::InputArray write_mask
) const
{
    DWT2D::Coeffs normalized_coeffs;

    internal::throw_if_empty(_p->coeff_matrix, "Coefficients are empty.");
    throw_if_bad_mask_for_normalize(write_mask, "write");

    scale_output = map_detail_to_unit_interval_scale(read_mask);
    cv::Mat normalized_coeffs_matrix = scale_output * _p->coeff_matrix + 0.5;
    if (is_not_array(write_mask)) {
        normalized_coeffs = empty_clone();
        normalized_coeffs._p->coeff_matrix = normalized_coeffs_matrix;
        normalized_coeffs.set_approx(approx());
    } else {
        //  Make sure to include any unused half rows/columns resulting from
        //  odd image size or kernel lengths.
        auto write_mask_matrix = write_mask.getMat() | invalid_detail_mask();

        normalized_coeffs = clone();
        normalized_coeffs_matrix.copyTo(normalized_coeffs, write_mask_matrix);
        normalized_coeffs.set_approx(approx());
    }

    return normalized_coeffs;
}


// DWT2D::Coeffs DWT2D::Coeffs::map_details_to_unit_interval(
//     cv::InputArray read_mask,
//     cv::InputArray write_mask
// ) const
// {
//     DWT2D::Coeffs normalized_coeffs;
//     map_details_to_unit_interval(normalized_coeffs, read_mask, write_mask);
//     return normalized_coeffs;
// }

// double DWT2D::Coeffs::map_details_to_unit_interval(
//     DWT2D::Coeffs& normalized_coeffs,
//     cv::InputArray read_mask,
//     cv::InputArray write_mask
// ) const
// {
//     internal::throw_if_empty(_p->coeff_matrix, "Coefficients are empty.");
//     throw_if_bad_mask_for_normalize(write_mask, "write");

//     double alpha = map_detail_to_unit_interval_scale(read_mask);
//     cv::Mat normalized_coeffs_matrix = alpha * _p->coeff_matrix + 0.5;
//     if (is_not_array(write_mask)) {
//         normalized_coeffs = empty_clone();
//         normalized_coeffs._p->coeff_matrix = normalized_coeffs_matrix;
//         normalized_coeffs.set_approx(approx());
//     } else {
//         //  Make sure to include any unused half rows/columns resulting from
//         //  odd image size or kernel lengths.
//         auto write_mask_matrix = write_mask.getMat() | invalid_detail_mask();

//         normalized_coeffs = clone();
//         normalized_coeffs_matrix.copyTo(normalized_coeffs, write_mask_matrix);
//         normalized_coeffs.set_approx(approx());
//     }

//     return alpha;
// }

DWT2D::Coeffs DWT2D::Coeffs::map_details_from_unit_interval(
    double scale,
    cv::InputArray write_mask
) const
{
    internal::throw_if_empty(_p->coeff_matrix, "Coefficients are empty.");
    throw_if_bad_mask_for_normalize(write_mask, "write");

    cv::Mat unnormalized_coeffs_matrix = (_p->coeff_matrix - 0.5) / scale;

    if (is_not_array(write_mask)) {
        auto unnormalized_coeffs = empty_clone();
        unnormalized_coeffs._p->coeff_matrix = unnormalized_coeffs_matrix;
        unnormalized_coeffs.set_approx(approx());

        return unnormalized_coeffs;
    } else {
        //  Make sure to include any unused half rows/columns resulting from
        //  odd image size or kernel lengths.
        auto write_mask_matrix = write_mask.getMat() | invalid_detail_mask();

        auto unnormalized_coeffs = clone();
        unnormalized_coeffs_matrix.copyTo(unnormalized_coeffs, write_mask_matrix);
        unnormalized_coeffs.set_approx(approx());

        return unnormalized_coeffs;
    }
}

double DWT2D::Coeffs::map_detail_to_unit_interval_scale(cv::InputArray read_mask) const
{
    internal::throw_if_empty(_p->coeff_matrix, "Coefficients are empty.");
    throw_if_bad_mask_for_normalize(read_mask, "read");
    double max_value = maximum_abs_value(
        _p->coeff_matrix,
        is_not_array(read_mask) ? read_mask : detail_mask()
    );

    return 0.5 / max_value;
}

template <typename Matrix>
requires std::same_as<std::remove_cvref_t<Matrix>, cv::Mat>
void DWT2D::Coeffs::convert_and_copy(
    cv::InputArray source,
    Matrix&& destination,
    cv::InputArray mask
)
{
    assert(destination.type() == type());

    if (is_scalar(source)) {
        destination.setTo(source, mask);
    } else if (source.type() != destination.type()) {
        cv::Mat converted;
        source.getMat().convertTo(converted, type());
        converted.copyTo(destination, mask);
    } else {
        source.copyTo(destination, mask);
    }
}

#if CVWT_ARGUMENT_CHECKING_ENABLED
void DWT2D::Coeffs::throw_if_bad_mask_for_normalize(
    cv::InputArray mask,
    const std::string mask_name
) const
{
    if (is_not_array(mask))
        return;

    internal::throw_if_bad_mask_type(
        mask,
        "The ", mask_name, " mask has the wrong data type."
    );
    internal::throw_if_empty(
        mask,
        "The ", mask_name, " mask is empty. Use cv::noArray() to indicate no mask."
    );
    if (mask.size() != size())
        internal::throw_bad_size(
            "The ", mask_name, " mask size must be ", size(),
            ", got ", mask.size(), "."
        );
}

void DWT2D::Coeffs::throw_if_wrong_size_for_assignment(cv::InputArray matrix) const
{
    if (empty()) {
        if (!is_scalar(matrix) && matrix.size() == level_size(0))
            return;
    } else {
        if (is_scalar(matrix) || (matrix.size() == size() && matrix.channels() == channels()))
            return;

        if (matrix.channels() != channels()) {
            internal::throw_bad_size(
                "DWT2D::Coeffs: Cannot assign matrix to this.  ",
                "The number of channels of matrix must be ", channels(), "), ",
                "got matrix.channels() = ", matrix.channels(), "."
            );
        }
    }

    auto required_size = empty() ? level_size(0) : size();
    if (matrix.size() != required_size) {
        internal::throw_bad_size(
            "DWT2D::Coeffs: Cannot assign matrix to this.  ",
            "The size of matrix must be ", required_size, "), ",
            "got matrix.size() = ", matrix.size(), "."
        );
    }
}

void DWT2D::Coeffs::throw_if_wrong_size_for_set_level(cv::InputArray matrix, int level) const
{
    if (is_scalar(matrix) || matrix.size() == level_size(level))
        return;

    internal::throw_bad_size(
        "DWT2D::Coeffs: Cannot set the coeffs at level ", level, ".  "
        "The size of the matrix must be ", level_size(level), ", ",
        "got size = ", matrix.size(), "."
    );
}

void DWT2D::Coeffs::throw_if_wrong_size_for_set_detail(
    cv::InputArray matrix,
    int level,
    int subband
) const
{
    if (is_scalar(matrix)
        || (matrix.size() == detail_size(level) && matrix.channels() == channels()))
        return;

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

    if (matrix.size() != detail_size(level)) {
        internal::throw_bad_size(
            "DWT2D::Coeffs: Cannot set the ", subband_name, " detail coefficients at level ", level, ".  ",
            "The size of the matrix must be ", detail_size(level), ", "
            "got size = ", matrix.size(), "."
        );
    }

    if (matrix.channels() != channels()) {
        internal::throw_bad_size(
            "DWT2D::Coeffs: Cannot set the ", subband_name, " detail coefficients at level ", level, ".  ",
            "The number of channels must be ", channels(), ", "
            "got channels = ", matrix.channels(), "."
        );
    }
}

void DWT2D::Coeffs::throw_if_wrong_size_for_set_all_detail_levels(cv::InputArray matrix) const
{
    if (is_scalar(matrix) || (matrix.size() == size() && matrix.channels() == channels()))
        return;

    if (matrix.size() != size()) {
        std::string subband_name;
        internal::throw_bad_size(
            "DWT2D::Coeffs: Cannot set all the detail coefficients. ",
            "The size of the matrix must be ", size(), ", "
            "got size = ", matrix.size(), "."
        );
    }

    if (matrix.channels() != channels()) {
        std::string subband_name;
        internal::throw_bad_size(
            "DWT2D::Coeffs: Cannot set all the detail coefficients. ",
            "The number of channels must be ", channels(), ", "
            "got channels = ", matrix.channels(), "."
        );
    }
}

void DWT2D::Coeffs::throw_if_wrong_size_for_set_approx(cv::InputArray matrix) const
{
    if (is_scalar(matrix) || matrix.size() == detail_size(levels() - 1))
        return;

    internal::throw_bad_size(
        "DWT2D::Coeffs: Cannot set the approx coefficients.  "
        "The matrix must be a scalar or its size must be ", detail_size(levels() - 1), ", ",
        "got size = ", matrix.size(), "."
    );
}

void DWT2D::Coeffs::throw_if_level_out_of_range(int level) const
{
    if (level < -levels() || level >= levels()) {
        internal::throw_out_of_range(
            "DWT2D::Coeffs: level is out of range. ",
            "Must be ", -levels(), " <= level < ", levels(), ", ",
            "got level = ", level, "."
        );
    }
}

void DWT2D::Coeffs::throw_if_levels_out_of_range(int lower_level, int upper_level) const
{
    if (lower_level < -levels() || lower_level >= levels()) {
        internal::throw_out_of_range(
            "DWT2D::Coeffs: lower_level is out of range. ",
            "Must be ", -levels(), " <= lower_level < ", levels(), ", ",
            "got lower_level = ", lower_level, "."
        );
    }

    if (upper_level < -levels() || upper_level > levels()) {
        internal::throw_out_of_range(
            "DWT2D::Coeffs: upper_level is out of range. ",
            "Must be ", -levels(), " <= upper_level < ", levels(), ", ",
            "got upper_level = ", upper_level, "."
        );
    }
}

void DWT2D::Coeffs::throw_if_this_is_empty() const
{
    if (empty()) {
        internal::throw_bad_size(
            "DWT2D::Coeffs: Coefficients are empty."
        );
    }
}

void DWT2D::Coeffs::throw_if_invalid_subband(int subband) const
{
    switch (subband) {
        case HORIZONTAL:
        case VERTICAL:
        case DIAGONAL:
            break;
        default:
            internal::throw_bad_arg(
                "DWT2D::Coeffs: Invalid subband.  ",
                "Must be 0 (HORIZONTAL), 1 (VERTICAL), or 2 (DIAGONAL), ",
                "got ", subband, "."
            );
    }
}
#endif  // CVWT_ARGUMENT_CHECKING_ENABLED

std::ostream& operator<<(std::ostream& stream, const DWT2D::Coeffs& coeffs)
{
    stream << coeffs._p->coeff_matrix << "\n(" << coeffs.levels() << " levels)";
    return stream;
}

std::vector<DWT2D::Coeffs> split(const DWT2D::Coeffs& coeffs)
{
    std::vector<cv::Mat> coeffs_channels;
    cv::split(coeffs, coeffs_channels);

    std::vector<DWT2D::Coeffs> result;
    for (const auto& coeff_matrix : coeffs_channels) {
        result.push_back(
            DWT2D::Coeffs(
                coeff_matrix,
                coeffs._p->levels,
                coeffs._p->image_size,
                coeffs._p->diagonal_subband_rects,
                coeffs._p->wavelet,
                coeffs._p->border_type
            )
        );
    }

    return result;
}

DWT2D::Coeffs merge(const std::vector<DWT2D::Coeffs>& coeffs)
{
    if (coeffs.empty())
        return DWT2D::Coeffs();

    std::vector<cv::Mat> coeff_matrices(coeffs.size());
    std::ranges::transform(
        coeffs,
        coeff_matrices.begin(),
        [](const auto& coeff) -> cv::Mat { return coeff; }
    );
    cv::Mat coeff_matrix;
    cv::merge(coeff_matrices, coeff_matrix);

    return DWT2D::Coeffs(
        coeff_matrix,
        coeffs[0]._p->levels,
        coeffs[0]._p->image_size,
        coeffs[0]._p->diagonal_subband_rects,
        coeffs[0]._p->wavelet,
        coeffs[0]._p->border_type
    );
}


//  ============================================================================
//  Public API
//  ============================================================================
DWT2D::DWT2D(const Wavelet& wavelet, cv::BorderTypes border_type) :
    wavelet(wavelet),
    border_type(border_type)
{
}

void DWT2D::decompose(cv::InputArray image, DWT2D::Coeffs& coeffs, int levels) const
{
    throw_if_levels_out_of_range(levels);
    warn_if_border_effects_will_occur(levels, image);
    coeffs.reset(
        coeffs_size_for_image(image, levels),
        wavelet.filter_bank().promote_type(image.type()),
        levels,
        image.size(),
        calc_subband_sizes(image.size(), levels),
        wavelet,
        border_type
    );

    //  Must initialize to zero here in case of odd subband width/height.
    //  Whenever there is an odd subband width/height there is a half row/column
    //  of unused horizontal/vertical detail elements that need to be forced to
    //  zero.
    coeffs = 0.0;

    auto running_approx = image.getMat();
    for (int level = 0; level < levels; ++level) {
        cv::Mat approx;
        cv::Mat horizontal_detail;
        cv::Mat vertical_detail;
        cv::Mat diagonal_detail;
        wavelet.filter_bank().decompose(
            running_approx,
            approx,
            horizontal_detail,
            vertical_detail,
            diagonal_detail
        );

        running_approx = approx;
        coeffs.set_horizontal_detail(level, horizontal_detail);
        coeffs.set_vertical_detail(level, vertical_detail);
        coeffs.set_diagonal_detail(level, diagonal_detail);
    }

    coeffs.set_approx(running_approx);
}

void DWT2D::reconstruct(const DWT2D::Coeffs& coeffs, cv::OutputArray image) const
{
    warn_if_border_effects_will_occur(coeffs);

    cv::Mat approx = coeffs.approx();
    for (int level = coeffs.levels() - 1; level >= 0; --level) {
        cv::Mat result;
        wavelet.filter_bank().reconstruct(
            approx,
            coeffs.horizontal_detail(level),
            coeffs.vertical_detail(level),
            coeffs.diagonal_detail(level),
            result,
            coeffs.image_size(level)
        );
        approx = result;
    }

    if (image.isContinuous())
        image.assign(approx);
    else
        approx.copyTo(image);
}

DWT2D::Coeffs DWT2D::create_coeffs(
    cv::InputArray coeffs_matrix,
    const cv::Size& image_size,
    int levels
) const
{
    throw_if_levels_out_of_range(levels);
    throw_if_inconsistent_coeffs_and_image_sizes(coeffs_matrix, image_size, levels);

    return Coeffs(
        coeffs_matrix.getMat(),
        levels,
        image_size,
        calc_subband_sizes(image_size, levels),
        wavelet,
        border_type
    );
}

DWT2D::Coeffs DWT2D::create_empty_coeffs(
    const cv::Size& image_size,
    int levels
) const
{
    throw_if_levels_out_of_range(levels);

    return Coeffs(
        cv::Mat(0, 0, wavelet.filter_bank().depth()),
        levels,
        image_size,
        calc_subband_sizes(image_size, levels),
        wavelet,
        border_type
    );
}

DWT2D::Coeffs DWT2D::create_coeffs(const cv::Size& image_size, int type, int levels) const
{
    auto size = coeffs_size_for_image(image_size, levels);

    return create_coeffs(
        cv::Mat(size, type, cv::Scalar::all(0.0)),
        image_size,
        levels
    );
}

std::vector<cv::Size> DWT2D::calc_subband_sizes(const cv::Size& image_size, int levels) const
{
    std::vector<cv::Size> subband_sizes;
    cv::Size subband_size = image_size;
    for (int i = 0; i < levels; ++i) {
        subband_size = wavelet.filter_bank().subband_size(subband_size);
        subband_sizes.push_back(subband_size);
    }

    return subband_sizes;
}

cv::Size DWT2D::coeffs_size_for_image(const cv::Size& image_size, int levels) const
{
    throw_if_levels_out_of_range(levels);

    cv::Size level_subband_size = wavelet.filter_bank().subband_size(image_size);
    cv::Size accumulator = level_subband_size;
    for (int i = 1; i < levels; ++i) {
        level_subband_size = wavelet.filter_bank().subband_size(level_subband_size);
        accumulator += level_subband_size;
    }

    //  add once more to account for approximation coefficients
    accumulator += level_subband_size;

    return accumulator;
}

int DWT2D::max_levels_without_border_effects(int image_rows, int image_cols) const
{
    double data_length = std::min(image_rows, image_cols);
    if (data_length <= 0)
        return 0;

    int max_levels = std::floor(std::log2(data_length / (wavelet.filter_length() - 1.0)));
    return std::max(max_levels, 0);
}

#if CVWT_ARGUMENT_CHECKING_ENABLED
void DWT2D::throw_if_levels_out_of_range(int levels) const
{
    if (levels < 1) {
        internal::throw_out_of_range(
            "DWT2D: levels is out of range. ",
            "Must be levels >= 1, ",
            "got levels = ", levels,  "."
        );
    }
}

void DWT2D::throw_if_inconsistent_coeffs_and_image_sizes(
    cv::InputArray coeffs,
    const cv::Size& image_size,
    int levels
) const
{
    auto required_coeffs_size = coeffs_size_for_image(image_size, levels);
    if (coeffs.size() != required_coeffs_size) {
        internal::throw_bad_size(
            "DWT2D: coefficients size is not consistent with image size. ",
            "Coefficients size must be ", required_coeffs_size, " ",
            "for image size = ", image_size, " and levels = ", levels, ", ",
            "got coeffs.size() = ", coeffs.size(), ". ",
            "(Note: use DWT2D::coeffs_size_for_input() to get the required size)"
        );
    }
}
#endif  // CVWT_ARGUMENT_CHECKING_ENABLED

#if CVWT_DISABLE_DWT_WARNINGS_ENABLED
void DWT2D::warn_if_border_effects_will_occur(int levels, const cv::Size& image_size) const noexcept
{
    int max_levels = max_levels_without_border_effects(image_size);
    if (levels > max_levels) {
        std::stringstream message;
        message
            << "DWT2D: border effects will occur for a " << levels << " level DWT "
            << "of a " << image_size << " image using the " << wavelet.name() << " wavelet. "
            << "Must have levels <= " << max_levels << " to avoid border effects.";
        CV_LOG_WARNING(NULL, message.str());
    }
}

void DWT2D::warn_if_border_effects_will_occur(int levels, cv::InputArray image) const noexcept
{
    warn_if_border_effects_will_occur(levels, image.size());
}

void DWT2D::warn_if_border_effects_will_occur(const Coeffs& coeffs) const noexcept
{
    warn_if_border_effects_will_occur(
        coeffs.levels(),
        coeffs.image_size()
    );
}
#else
inline void DWT2D::warn_if_border_effects_will_occur(int levels, const cv::Size& image_size) const noexcept {}
inline void DWT2D::warn_if_border_effects_will_occur(int levels, cv::InputArray image) const noexcept {}
inline void DWT2D::warn_if_border_effects_will_occur(const Coeffs& coeffs) const noexcept {}
#endif  // CVWT_DISABLE_DWT_WARNINGS_ENABLED


//  ----------------------------------------------------------------------------
//  Functional Interface
//  ----------------------------------------------------------------------------
DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).decompose(image);
}

DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const std::string& wavelet,
    cv::BorderTypes border_type
)
{
    return dwt2d(image, Wavelet::create(wavelet), border_type);
}

DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).decompose(image, levels);
}

DWT2D::Coeffs dwt2d(
    cv::InputArray image,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return dwt2d(image, Wavelet::create(wavelet), levels, border_type);
}

void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).decompose(image, coeffs);
}

void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    cv::BorderTypes border_type
)
{
    dwt2d(image, coeffs, Wavelet::create(wavelet), border_type);
}

void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).decompose(image, coeffs, levels);
}

void dwt2d(
    cv::InputArray image,
    DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    int levels,
    cv::BorderTypes border_type
)
{
    dwt2d(image, coeffs, Wavelet::create(wavelet), levels, border_type);
}

cv::Mat idwt2d(
    const DWT2D::Coeffs& coeffs,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    return DWT2D(wavelet, border_type).reconstruct(coeffs);
}

cv::Mat idwt2d(
    const DWT2D::Coeffs& coeffs,
    const std::string& wavelet,
    cv::BorderTypes border_type
)
{
    return idwt2d(coeffs, Wavelet::create(wavelet), border_type);
}

void idwt2d(
    const DWT2D::Coeffs& coeffs,
    cv::OutputArray image,
    const Wavelet& wavelet,
    cv::BorderTypes border_type
)
{
    DWT2D(wavelet, border_type).reconstruct(coeffs, image);
}

void idwt2d(
    const DWT2D::Coeffs& coeffs,
    cv::OutputArray image,
    const std::string& wavelet,
    cv::BorderTypes border_type
)
{
    idwt2d(coeffs, image, Wavelet::create(wavelet), border_type);
}

} // namespace cvwt

