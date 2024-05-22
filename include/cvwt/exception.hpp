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
namespace internal
{
//  forward declaration (defined in utils.cpp)
std::string get_type_name(int type);

#define throw_error(code, ...) _throw_error(code, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_error(
    cv::Error::Code code,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    std::stringstream message;
    (message << ... << message_parts);

    cv::error(code, message.str(), function, file, line);
}

#define throw_bad_size(...) _throw_bad_size(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_bad_size(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(cv::Error::StsBadSize, function, file, line, message_parts...);
}

#define throw_if_empty(array, ...) _throw_if_empty(array, #array, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

void _throw_if_empty(
    cv::InputArray array,
    const char* array_name,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
)
{
    if (array.empty()) {
        if constexpr (sizeof...(message_parts) == 0)
            _throw_bad_size(
                function, file, line,
                "The array `", array_name, "` cannot be empty."
            );
        else
            _throw_bad_size(function, file, line, message_parts...);
    }
}

#define throw_bad_arg(...) _throw_bad_arg(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_bad_arg(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(
        cv::Error::StsBadArg,
        function, file, line,
        message_parts...
    );
}

#define throw_out_of_range(...) _throw_out_of_range(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_out_of_range(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(
        cv::Error::StsOutOfRange,
        function, file, line,
        message_parts...
    );
}

#define throw_bad_mask(...) _throw_bad_mask(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_bad_mask(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(
        cv::Error::StsBadMask,
        function, file, line,
        message_parts...
    );
}

#define throw_if_bad_mask_type(mask, ...) _throw_if_bad_mask_type(mask, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

void _throw_if_bad_mask_type(
    cv::InputArray mask,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
)
{
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
}

enum AllowedMaskChannels
{
    SINGLE,
    SAME,
    SINGLE_OR_SAME,
};

#define throw_if_bad_mask_for_array(array, mask, allowed_channels, ...) _throw_if_bad_mask_for_array(array, mask, #array, #mask, allowed_channels, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

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
)
{
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
}

#define throw_not_implemented(...) _throw_not_implemented(CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_not_implemented(
    const char* function,
    const char* file,
    int line,
    auto... message_parts
) noexcept(false)
{
    throw_error(cv::Error::StsNotImplemented, function, file, line, message_parts...);
}

#define throw_member_not_implemented(class_name, function_name, ...) _throw_member_not_implemented(class_name, function_name, CV_Func, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

[[noreturn]]
void _throw_member_not_implemented(
    const std::string& class_name,
    const std::string& function_name,
    const char* function,
    const char* file,
    int line,
    auto... message_parts
)  noexcept(false)
{
    _throw_not_implemented(
        function, file, line,
        class_name, "::", function_name, " is not implemented. ",
        message_parts...
    );
}
}   // namespace internal
}   // namespace cvwt

#endif  // CVWT_EXCEPTION_HPP

