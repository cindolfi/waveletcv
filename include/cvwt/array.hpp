#ifndef CVWT_ARRAY_HPP
#define CVWT_ARRAY_HPP

#include <opencv2/core.hpp>

namespace cvwt
{
/**
 * @name Array Ops
 * @{
 */
/**
 * @brief Collect values indicated by the given mask.
 *
 * @param[in] array
 * @param[out] collected
 * @param[in] mask
 */
void collect_masked(cv::InputArray array, cv::OutputArray collected, cv::InputArray mask);

/**
 * @brief Returns true if the two matrices refer to the same data.
 *
 * @param[in] a
 * @param[in] b
 */
bool is_data_shared(cv::InputArray a, cv::InputArray b);

/**
 * @brief Negates all even indexed values.
 *
 * @param[in] vector A row or column vector.
 * @param[out] result The input vector with the even indexed values negated.
 */
void negate_even_indices(cv::InputArray vector, cv::OutputArray result);

/**
 * @brief Negates all odd indexed values.
 *
 * @param[in] vector A row or column vector.
 * @param[out] result The input vector with the odd indexed values negated.
 */
void negate_odd_indices(cv::InputArray vector, cv::OutputArray result);

/**
 * @brief Returns true if array is cv::noArray().
 *
 * @param[in] array
 */
bool is_not_array(cv::InputArray array);

/**
 * @brief Replace all NaN values.
 *
 * This is a version of cv::patch_nans() that accepts arrays of any depth, not
 * just CV_32F.
 *
 * @param[inout] array The array containing NaN values.
 * @param[in] value The value used to replace NaN.
 */
void patch_nans(cv::InputOutputArray array, double value = 0.0);

/**
 * @brief Returns true if the given value can be used as a scalar for the given array.
 *
 * Scalars can be added to or subtracted from the array, be assigned to all or
 * some array elements, or be used with comparison functions (e.g. compare(),
 * less_than(), etc.).
 *
 * A scalar is defined to be one of the following:
 *  - A fundamental type (e.g. float, double, etc.)
 *  - A vector containing @pref{array,channels(),cv::Mat::channels} elements
 *    (e.g. cv::Vec, std::vector, array, etc.)
 *  - A cv::Scalar if @pref{array,channels(),cv::Mat::channels} is less than or
 *    equal to 4
 *
 * @param[in] scalar
 * @param[in] array
 */
bool is_scalar_for_array(cv::InputArray scalar, cv::InputArray array);

/**
 * @brief Returns true if the array is a row vector or column vector.
 *
 * The @pref{array} is a vector if it has a single row or column and is
 * continuous.
 *
 * @param array The potential vector.
 */
bool is_vector(cv::InputArray array);

/**
 * @brief Returns true if the array is a row vector or column vector.
 *
 * The @pref{array} is a vector if it has a single row or column, is continuous,
 * and has @pref{channels} number of channels.
 *
 * @param array The potential vector.
 * @param channels The required number of channels.
 */
bool is_vector(cv::InputArray array, int channels);

/**
 * @brief Returns true if the array is a row vector or column vector.
 *
 * The @pref{array} is a vector if it has a single column and is continuous.
 *
 * @param array The potential vector.
 */
bool is_column_vector(cv::InputArray array);

/**
 * @brief Returns true if the array is a column vector.
 *
 * The @pref{array} is a vector if it has a single column, is continuous, and
 * has @pref{channels} number of channels.
 *
 * @param array The potential vector.
 * @param channels The required number of channels.
 */
bool is_column_vector(cv::InputArray array, int channels);

/**
 * @brief Returns true if the array is a row vector or column vector.
 *
 * The @pref{array} is a vector if it has a single row and is continuous.
 *
 * @param array The potential vector.
 */
bool is_row_vector(cv::InputArray array);

/**
 * @brief Returns true if the array is a row vector.
 *
 * The @pref{array} is a vector if it has a single row, is continuous, and
 * has @pref{channels} number of channels.
 *
 * @param array The potential vector.
 * @param channels The required number of channels.
 */
bool is_row_vector(cv::InputArray array, int channels);
/** @}*/
}   // namespace cvwt

#endif  // CVWT_ARRAY_HPP

