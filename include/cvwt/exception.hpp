#ifndef CVWT_EXCEPTION_HPP
#define CVWT_EXCEPTION_HPP

#include <span>
#include <ranges>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <iostream>

namespace cvwt
{
//  forward declarations (defined in utils.cpp)
bool is_vector(cv::InputArray vector, int channels);

/**
 * @brief Throws an error.
 *
 * @param[in] code A cv::Error::Code.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_error(code, ...) internal::_throw_error(code, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a bad size exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_bad_size(...) internal::_throw_bad_size(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a bad size exception the array is empty.
 *
 * @param[in] array The array to check.  This must be convertible to cv::InputArray.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_empty(array, ...) internal::_throw_if_empty(array, #array, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a bad argument exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_bad_arg(...) internal::_throw_bad_arg(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws an out of range exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_out_of_range(...) internal::_throw_out_of_range(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a bad mask exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_bad_mask(...) internal::_throw_bad_mask(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a bad mask exception if the mask type is not CV_8UC1 or CV_8SC1.
 *
 * @param[in] mask The mask to check.  This must be convertible to cv::InputArray.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_bad_mask_type(mask, ...) internal::_throw_if_bad_mask_type(mask, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a bad mask exception if the mask depth is not CV_8U or CV_8S.
 *
 * @param[in] mask The mask to check.  This must be convertible to cv::InputArray.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_bad_mask_depth(mask, ...) internal::_throw_if_bad_mask_depth(mask, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Allowed number of mask channels that can be used with given array.
 */
enum AllowedMaskChannels
{
    SINGLE,
    SAME,
    SINGLE_OR_SAME,
};

/**
 * @brief Throws a bad mask exception if mask cannot be used with the array.
 *
 * The mask must have the same size as the array.
 * If `allowed_channels` is
 *  - AllowedMaskChannels::SINGLE: the mask must be a single channel
 *  - AllowedMaskChannels::SAME: the mask must have the same number of
 *    channels as the array
 *  - AllowedMaskChannels::SINGLE_OR_SAME: the mask must be a single channel or
 *    have the same number of channels as the array
 *
 * @param[in] array The array to check.  This must be convertible to cv::InputArray.
 * @param[in] mask The mask to check.  This must be convertible to cv::InputArray.
 * @param[in] allowed_channels The number of allowed mask channels:
 *                             AllowedMaskChannels::SINGLE,
 *                             AllowedMaskChannels::SAME,
 *                             or AllowedMaskChannels::SINGLE_OR_SAME.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_bad_mask_for_array(array, mask, allowed_channels, ...) internal::_throw_if_bad_mask_for_array(array, mask, #array, #mask, allowed_channels, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a not implemented exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_not_implemented(...) internal::_throw_not_implemented(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a member function is not implemented exception.
 *
 * @param[in] class_name The name of the class.
 * @param[in] function_name The name of the member function.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_member_not_implemented(class_name, function_name, ...) internal::_throw_member_not_implemented(class_name, function_name, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Throws a bad size exception if the array is not a single channel row or column vector.
 *
 * @param[in] array The array to check.  This must be convertible to cv::InputArray.
 * @param[in] channels The number of channels the array must have.  If this is
                       less than or equal to zero, any number of channels is
                       allowed.
 */
#define throw_if_not_vector(array, channels, ...) internal::_throw_if_not_vector(array, channels, #array, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)




namespace internal
{
//  forward declarations (defined in utils.cpp)
std::string get_type_name(int type);

CVWT_NORETURN
void _throw_error(
    cv::Error::Code code,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(!CVWT_EXCEPTIONS_ENABLED)
{
#if CVWT_EXCEPTIONS_ENABLED
    std::stringstream message;
    (message << ... << message_parts);

    cv::error(code, message.str(), function, file, line);
#endif
}

CVWT_NORETURN
void _throw_bad_size(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(!CVWT_EXCEPTIONS_ENABLED)
{
#if CVWT_EXCEPTIONS_ENABLED
    _throw_error(cv::Error::StsBadSize, function, file, line, message_parts...);
#endif
}

void _throw_if_empty(
    cv::InputArray array,
    const char* array_name,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) CVWT_NOEXCEPT
{
#if CVWT_EXCEPTIONS_ENABLED
    if (array.empty()) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_size(
                function, file, line,
                "The array `", array_name, "` cannot be empty."
            );
        else
            _throw_bad_size(function, file, line, message_parts...);
    }
#endif
}

CVWT_NORETURN
void _throw_bad_arg(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(!CVWT_EXCEPTIONS_ENABLED)
{
#if CVWT_EXCEPTIONS_ENABLED
    _throw_error(cv::Error::StsBadArg, function, file, line, message_parts...);
#endif
}

CVWT_NORETURN
void _throw_out_of_range(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(!CVWT_EXCEPTIONS_ENABLED)
{
#if CVWT_EXCEPTIONS_ENABLED
    _throw_error(cv::Error::StsOutOfRange, function, file, line, message_parts...);
#endif
}

CVWT_NORETURN
void _throw_bad_mask(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(!CVWT_EXCEPTIONS_ENABLED)
{
#if CVWT_EXCEPTIONS_ENABLED
    _throw_error(cv::Error::StsBadMask, function, file, line, message_parts...);
#endif
}

void _throw_if_bad_mask_type(
    cv::InputArray mask,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) CVWT_NOEXCEPT
{
#if CVWT_EXCEPTIONS_ENABLED
    if (mask.type() != CV_8UC1 && mask.type() != CV_8SC1) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_mask(
                function, file, line,
                "Mask type must be CV_8UC1 or CV_8SC1, got ",
                get_type_name(mask.type()), ". "
            );
        else
            _throw_bad_mask(
                function, file, line,
                message_parts...,
                " [Mask type must be CV_8UC1 or CV_8SC1, got ",
                get_type_name(mask.type()), "]"
            );
    }
#endif
}

void _throw_if_bad_mask_depth(
    cv::InputArray mask,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
)
{
#if CVWT_EXCEPTIONS_ENABLED
    if (mask.depth() != CV_8U && mask.depth() != CV_8S) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_mask(
                function, file, line,
                "Mask depth must be CV_8U or CV_8S, got ",
                get_type_name(mask.type()), ". "
            );
        else
            _throw_bad_mask(
                function, file, line,
                message_parts...,
                " [Mask type must be CV_8U or CV_8S, got ",
                get_type_name(mask.type()), "]"
            );
    }
#endif
}

void _throw_if_bad_mask_for_array(
    cv::InputArray array,
    cv::InputArray mask,
    const std::string& array_name,
    const std::string& mask_name,
    AllowedMaskChannels allowed_channels,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) CVWT_NOEXCEPT
{
#if CVWT_EXCEPTIONS_ENABLED
    if (!array.sameSize(mask)) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_mask(
                function, file, line,
                "The `", mask_name, "` must be the same size as `", array_name, "`. "
                "Got ", array_name, ".size() = ", array.size(),
                " and ", mask_name, ".size() = ", mask.size(), ". "
            );
        else
            _throw_bad_mask(function, file, line, message_parts...);
    }

    switch (allowed_channels) {
    case AllowedMaskChannels::SINGLE:
        if (mask.channels() != 1) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    function, file, line,
                    "Wrong number of `", mask_name, "` channels for `", array_name, "`. "
                    "The number of `", mask_name, "` channels must be 1. "
                    "Got ", mask_name, ".channels() = ", mask.channels(), ". "
                );
            else
                _throw_bad_mask(function, file, line, message_parts...);
        }
        break;
    case AllowedMaskChannels::SAME:
        if (mask.channels() != array.channels()) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    function, file, line,
                    "Wrong number of `", mask_name, "` channels for `", array_name, "`. ",
                    "The number of `", mask_name, "` channels must be the same as `", array_name, "`. ",
                    "Got ", array_name, ".channels() = ", array.channels(),
                    " and ", mask_name, ".channels() = ", mask.channels(), "."
                );
            else
                _throw_bad_mask(function, file, line, message_parts...);
        }
        break;
    case AllowedMaskChannels::SINGLE_OR_SAME:
        if (mask.channels() != 1 && mask.channels() != array.channels()) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    function, file, line,
                    "The number of `", mask_name, "` channels must be 1 or the same as `", array_name, "`. ",
                    "Got ", array_name, ".channels() = ", array.channels(),
                    " and ", mask_name, ".channels() = ", mask.channels(), "."
                );
            else
                _throw_bad_mask(function, file, line, message_parts...);
        }
        break;
    }
#endif
}

CVWT_NORETURN
void _throw_not_implemented(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(!CVWT_EXCEPTIONS_ENABLED)
{
#if CVWT_EXCEPTIONS_ENABLED
    _throw_error(cv::Error::StsNotImplemented, function, file, line, message_parts...);
#endif
}

CVWT_NORETURN
void _throw_member_not_implemented(
    const std::string& class_name,
    const std::string& function_name,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(!CVWT_EXCEPTIONS_ENABLED)
{
#if CVWT_EXCEPTIONS_ENABLED
    _throw_not_implemented(
        function, file, line,
        class_name, "::", function_name, " is not implemented. ",
        message_parts...
    );
#endif
}

void _throw_if_not_vector(
    cv::InputArray array,
    int channels,
    const std::string& array_name,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) CVWT_NOEXCEPT
{
#if CVWT_EXCEPTIONS_ENABLED
    if (!is_vector(array, channels <= 0 ? array.channels() : channels)) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_size(
                function, file, line,
                "Vectors must have rows == 1 or cols == 1 and channels == ", channels, ". ",
                "Got ", array_name, ".size() = ", array.size(), " and ",
                array_name, ".channels() = ", array.channels(), "."
            );
        else
            _throw_bad_size(
                function, file, line,
                message_parts...,
                " [Vectors must have rows == 1 or cols == 1 and channels == ", channels, ". ",
                "Got ", array_name, ".size() = ", array.size(), " and ",
                array_name, ".channels() = ", array.channels(), ".]"
            );
    }
#endif
}
}   // namespace internal
}   // namespace cvwt

#endif  // CVWT_EXCEPTION_HPP

