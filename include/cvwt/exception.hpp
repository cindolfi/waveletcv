#ifndef CVWT_EXCEPTION_HPP
#define CVWT_EXCEPTION_HPP

#include <string>
#include <source_location>
#include <opencv2/core.hpp>
#include "cvwt/utils.hpp"
#include "cvwt/array.hpp"

namespace cvwt
{
/**
 * @brief Allowed number of mask channels that can be used with given array.
 */
enum AllowedMaskChannels
{
    SINGLE,
    SAME,
    SINGLE_OR_SAME,
};


namespace internal
{
template <std::size_t... Indices>
struct concatable_index_sequence : std::index_sequence<Indices...>
{
    using std::index_sequence<Indices...>::index_sequence;

    template<std::size_t... J>
    concatable_index_sequence<Indices..., J...> operator+(concatable_index_sequence<J...>) const
    {
        return concatable_index_sequence<Indices..., J...>{};
    }
};

struct SourceLocationsRemover
{
    template <typename... Args>
    constexpr auto operator()(Args&&... args) const
    {
        auto select_args = [&]<size_t... Indices>(std::index_sequence<Indices...>) {
            return std::make_tuple(
                std::get<Indices>(std::forward_as_tuple(args...))...
            );
        };

        return select_args(
            non_source_location_indices(std::forward<Args>(args)...)
        );
    }

private:
    template <std::size_t... Indices, typename... Args>
    constexpr auto non_source_location_indices2(
        std::index_sequence<Indices...> indices,
        Args&&... args
    ) const
    {
        static_assert(sizeof...(Indices) == sizeof...(args));

        constexpr auto filter_index = [&]<std::size_t index>(std::index_sequence<index>) constexpr {
            using T = std::remove_cvref_t<
                std::tuple_element_t<index, std::tuple<Args...>>
            >;
            if constexpr (std::is_same_v<T, std::source_location>)
                return concatable_index_sequence{};
            else
                return concatable_index_sequence<index>{};
        };

        if constexpr (sizeof...(Args) > 0)
            return (filter_index(std::index_sequence<Indices>{}) + ...);
        else
            return std::index_sequence<>{};
    }

    template <typename... Args>
    constexpr auto non_source_location_indices(Args&&... args) const
    {
        return non_source_location_indices2(
            std::index_sequence_for<Args...>{},
            args...
        );
    }
};

constexpr auto remove_source_locations = SourceLocationsRemover();


struct SourceLocationsGetter
{
    template <typename... Args>
    constexpr auto operator()(Args&&... args) const
    {
        auto select_first = [&]<size_t Index, size_t... Indices>(std::index_sequence<Index, Indices...>) {
            return std::get<Index>(std::forward_as_tuple(args...));
        };

        return select_first(
            source_location_indices(std::forward<Args>(args)...)
        );
    }

private:
    template <std::size_t... Indices, typename... Args>
    constexpr auto source_location_indices2(
        std::index_sequence<Indices...> indices,
        Args&&... args
    ) const
    {
        static_assert(sizeof...(Indices) == sizeof...(args));

        constexpr auto filter_index = [&]<std::size_t index>(std::index_sequence<index>) constexpr {
            using T = std::remove_cvref_t<
                std::tuple_element_t<index, std::tuple<Args...>>
            >;
            if constexpr (std::is_same_v<T, std::source_location>)
                return concatable_index_sequence<index>{};
            else
                return concatable_index_sequence{};
        };

        if constexpr (sizeof...(Args) > 0)
            return (filter_index(std::index_sequence<Indices>{}) + ...);
        else
            return std::index_sequence<>{};
    }

    template <typename... Args>
    constexpr auto source_location_indices(Args&&... args) const
    {
        return source_location_indices2(
            std::index_sequence_for<Args...>{},
            args...
        );
    }
};

constexpr auto get_source_location = SourceLocationsGetter();


template <class Derived, typename... Args>
struct ErrorThrower
{
    ErrorThrower(
        Args&&... args,
        const std::source_location& location = std::source_location::current()
    ) noexcept(!CVWT_EXCEPTIONS_ENABLED) CVWT_NORETURN
    {
    #if CVWT_EXCEPTIONS_ENABLED
        auto derived = static_cast<Derived*>(this);
        if constexpr (sizeof...(Args) == 0) {
            derived->throw_error(location);
        } else {
            std::apply(
                [&](auto&&... non_location_args) {
                    derived->throw_error(
                        internal::get_source_location(std::forward<Args>(args)..., location),
                        std::forward<decltype(non_location_args)>(non_location_args)...
                    );
                },
                internal::remove_source_locations(std::forward<Args>(args)..., location)
            );
        }
    #endif
    }
};

template <class Derived, typename... Args>
struct ConditionalErrorThrower
{
    ConditionalErrorThrower(
        Args&&... args,
        const std::source_location& location = std::source_location::current()
    ) noexcept(!CVWT_EXCEPTIONS_ENABLED)
    {
    #if CVWT_EXCEPTIONS_ENABLED
        auto derived = static_cast<Derived*>(this);
        if constexpr (sizeof...(Args) == 0) {
            derived->throw_if_error(location);
        } else {
            std::apply(
                [&](auto&&... non_location_args) CVWT_NORETURN {
                    derived->throw_if_error(
                        internal::get_source_location(std::forward<Args>(args)..., location),
                        std::forward<decltype(non_location_args)>(non_location_args)...
                    );
                },
                internal::remove_source_locations(std::forward<Args>(args)..., location)
            );
        }
    #endif
    }
};

template <cv::Error::Code code, typename... Args>
struct _throw_error : public ErrorThrower<
    _throw_error<code, Args...>,
    Args...
>
{
    using ErrorThrower<_throw_error<code, Args...>, Args...>::ErrorThrower;

    template <typename... MessageParts>
    void throw_error CVWT_NORETURN (
        const std::source_location& location,
        MessageParts&&... message_parts
    ) const noexcept(!CVWT_EXCEPTIONS_ENABLED)
    {
        std::stringstream message;
        (message << ... << message_parts);
        cv::error(
            code,
            message.str(),
            location.function_name(),
            location.file_name(),
            location.line()
        );
    }
};

template <cv::Error::Code code, typename... MessageParts>
_throw_error(MessageParts&&...) -> _throw_error<code, MessageParts...>;


template <typename... MessageParts>
using _throw_bad_size = _throw_error<cv::Error::StsBadSize, MessageParts...>;

template <typename... MessageParts>
using _throw_bad_mask = _throw_error<cv::Error::StsBadMask, MessageParts...>;

template <typename... MessageParts>
using _throw_out_of_range = _throw_error<cv::Error::StsOutOfRange, MessageParts...>;

template <typename... MessageParts>
using _throw_bad_arg = _throw_error<cv::Error::StsBadArg, MessageParts...>;

template <typename... MessageParts>
using _throw_not_implemented = _throw_error<cv::Error::StsNotImplemented, MessageParts...>;


template <typename... Args>
struct _throw_if_empty : ConditionalErrorThrower<
    _throw_if_empty<Args...>,
    Args...
>
{
    using ConditionalErrorThrower<_throw_if_empty<Args...>, Args...>::ConditionalErrorThrower;

    template <typename... MessageParts>
    void throw_if_error(
        const std::source_location& location,
        cv::InputArray array,
        const std::string& array_name,
        MessageParts&&... message_parts
    ) const noexcept(!CVWT_EXCEPTIONS_ENABLED)
    {
        if (array.empty()) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_size(
                    "The array `", array_name, "` cannot be empty.",
                    location
                );
            else
                _throw_bad_size(message_parts..., location);
        }
    }
};

template <typename... Args>
_throw_if_empty(Args&&...) -> _throw_if_empty<Args...>;


template <typename... Args>
struct _throw_if_empty_mask : ConditionalErrorThrower<
    _throw_if_empty_mask<Args...>,
    Args...
>
{
    using ConditionalErrorThrower<_throw_if_empty_mask<Args...>, Args...>::ConditionalErrorThrower;

    template <typename... MessageParts>
    void throw_if_error(
        const std::source_location& location,
        cv::InputArray mask,
        const std::string& mask_name,
        MessageParts&&... message_parts
    ) const noexcept(!CVWT_EXCEPTIONS_ENABLED)
    {
        if (mask.empty()) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    "The mask `", mask_name, "` is empty. Use cv::noArray() to indicate no mask.",
                    location
                );
            else
                _throw_bad_mask(message_parts..., location);
        }
    }
};

template <typename... Args>
_throw_if_empty_mask(Args&&...) -> _throw_if_empty_mask<Args...>;


template <typename... Args>
struct _throw_if_bad_mask_type : ConditionalErrorThrower<
    _throw_if_bad_mask_type<Args...>,
    Args...
>
{
    using ConditionalErrorThrower<_throw_if_bad_mask_type<Args...>, Args...>::ConditionalErrorThrower;

    template <typename... MessageParts>
    void throw_if_error(
        const std::source_location& location,
        cv::InputArray mask,
        const std::string& mask_name,
        MessageParts&&... message_parts
    ) const noexcept(!CVWT_EXCEPTIONS_ENABLED)
    {
        if (mask.type() != CV_8UC1 && mask.type() != CV_8SC1) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    "The mask `", mask_name, "` type must be CV_8UC1 or CV_8SC1. Got ",
                    internal::get_type_name(mask.type()), ". ",
                    location
                );
            else
                _throw_bad_mask(
                    message_parts...,
                    " [The mask `", mask_name, "` type must be CV_8UC1 or CV_8SC1. Got ",
                    internal::get_type_name(mask.type()), "]",
                    location
                );
        }
    }
};

template <typename... Args>
_throw_if_bad_mask_type(Args&&...) -> _throw_if_bad_mask_type<Args...>;


template <typename... Args>
struct _throw_if_bad_mask_depth : ConditionalErrorThrower<
    _throw_if_bad_mask_depth<Args...>,
    Args...
>
{
    using ConditionalErrorThrower<_throw_if_bad_mask_depth<Args...>, Args...>::ConditionalErrorThrower;

    template <typename... MessageParts>
    void throw_if_error(
        const std::source_location& location,
        cv::InputArray mask,
        const std::string& mask_name,
        MessageParts&&... message_parts
    ) const noexcept(!CVWT_EXCEPTIONS_ENABLED)
    {
        if (mask.depth() != CV_8U && mask.depth() != CV_8S) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    "Mask `", mask_name, "` depth must be CV_8U or CV_8S. Got ",
                    get_type_name(mask.type()), ". ",
                    location
                );
            else
                _throw_bad_mask(
                    message_parts...,
                    " [Mask `", mask_name, "` depth must be CV_8U or CV_8S. Got ",
                    get_type_name(mask.type()), "]",
                    location
                );
        }
    }
};

template <typename... Args>
_throw_if_bad_mask_depth(Args&&...) -> _throw_if_bad_mask_depth<Args...>;


template <typename... Args>
struct _throw_if_bad_mask_for_array : ConditionalErrorThrower<
    _throw_if_bad_mask_for_array<Args...>,
    Args...
>
{
    using ConditionalErrorThrower<_throw_if_bad_mask_for_array<Args...>, Args...>::ConditionalErrorThrower;

    template <typename... MessageParts>
    void throw_if_error(
        const std::source_location& location,
        cv::InputArray array,
        cv::InputArray mask,
        const std::string& array_name,
        const std::string& mask_name,
        AllowedMaskChannels allowed_channels,
        MessageParts&&... message_parts
    ) const noexcept(!CVWT_EXCEPTIONS_ENABLED)
    {
        _throw_if_bad_mask_depth(mask, mask_name, message_parts...);
        _throw_if_empty_mask(mask, mask_name, message_parts...);

        if (!array.sameSize(mask)) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_mask(
                    "The `", mask_name, "` must be the same size as `", array_name, "`. "
                    "Got ", array_name, ".size() = ", array.size(),
                    " and ", mask_name, ".size() = ", mask.size(), ". ",
                    location
                );
            else
                _throw_bad_mask(message_parts..., location);
        }

        switch (allowed_channels) {
        case AllowedMaskChannels::SINGLE:
            if (mask.channels() != 1) {
                if constexpr (sizeof...(message_parts) == 0)
                    _throw_bad_mask(
                        "Wrong number of `", mask_name, "` channels for `", array_name, "`. "
                        "The number of `", mask_name, "` channels must be 1. "
                        "Got ", mask_name, ".channels() = ", mask.channels(), ". ",
                        location
                    );
                else
                    _throw_bad_mask(message_parts..., location);
            }
            break;
        case AllowedMaskChannels::SAME:
            if (mask.channels() != array.channels()) {
                if constexpr (sizeof...(message_parts) == 0)
                    _throw_bad_mask(
                        "Wrong number of `", mask_name, "` channels for `", array_name, "`. ",
                        "The number of `", mask_name, "` channels must be the same as `", array_name, "`. ",
                        "Got ", array_name, ".channels() = ", array.channels(),
                        " and ", mask_name, ".channels() = ", mask.channels(), ".",
                        location
                    );
                else
                    _throw_bad_mask(message_parts..., location);
            }
            break;
        case AllowedMaskChannels::SINGLE_OR_SAME:
            if (mask.channels() != 1 && mask.channels() != array.channels()) {
                if constexpr (sizeof...(message_parts) == 0)
                    _throw_bad_mask(
                        "The number of `", mask_name, "` channels must be 1 or the same as `", array_name, "`. ",
                        "Got ", array_name, ".channels() = ", array.channels(),
                        " and ", mask_name, ".channels() = ", mask.channels(), ".",
                        location
                    );
                else
                    _throw_bad_mask(message_parts..., location);
            }
            break;
        }
    }
};

template <typename... Args>
_throw_if_bad_mask_for_array(Args&&...) -> _throw_if_bad_mask_for_array<Args...>;


template <typename... Args>
struct _throw_if_not_vector : ConditionalErrorThrower<
    _throw_if_not_vector<Args...>,
    Args...
>
{
    using ConditionalErrorThrower<_throw_if_not_vector<Args...>, Args...>::ConditionalErrorThrower;

    template <typename... MessageParts>
    void throw_if_error(
        const std::source_location& location,
        cv::InputArray array,
        int channels,
        const std::string& array_name,
        MessageParts&&... message_parts
    ) const noexcept(!CVWT_EXCEPTIONS_ENABLED)
    {
        if (!is_vector(array, channels <= 0 ? array.channels() : channels)) {
            if constexpr (sizeof...(message_parts) == 0)
                _throw_bad_size(
                    "Vectors must have rows == 1 or cols == 1 and channels == ", channels, ". ",
                    "Got ", array_name, ".size() = ", array.size(), " and ",
                    array_name, ".channels() = ", array.channels(), ".",
                    location
                );
            else
                _throw_bad_size(
                    message_parts...,
                    " [Vectors must have rows == 1 or cols == 1 and channels == ", channels, ". ",
                    "Got ", array_name, ".size() = ", array.size(), " and ",
                    array_name, ".channels() = ", array.channels(), ".]",
                    location
                );
        }
    }
};

template <typename... Args>
_throw_if_not_vector(Args&&...) -> _throw_if_not_vector<Args...>;
}   // namespace internal


/**
 * @brief Throws an error.
 *
 * @param[in] code A cv::Error::Code.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_error(code, ...) internal::_throw_error<code>{__VA_ARGS__}

/**
 * @brief Throws a bad size exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_bad_size(...) internal::_throw_bad_size{__VA_ARGS__}

/**
 * @brief Throws a bad size exception the array is empty.
 *
 * @param[in] array The array to check.  This must be convertible to cv::InputArray.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_empty(array, ...) internal::_throw_if_empty{array, #array __VA_OPT__(,) __VA_ARGS__}

/**
 * @brief Throws a bad argument exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_bad_arg(...) internal::_throw_bad_arg{__VA_ARGS__}

/**
 * @brief Throws an out of range exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_out_of_range(...) internal::_throw_out_of_range{__VA_ARGS__}

/**
 * @brief Throws a bad mask exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_bad_mask(...) internal::_throw_bad_mask{__VA_ARGS__}

/**
 * @brief Throws a bad mask exception the mask is empty.
 *
 * @param[in] mask The mask to check.  This must be convertible to cv::InputArray.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_empty_mask(mask, ...) internal::_throw_if_empty_mask{mask, #mask __VA_OPT__(,) __VA_ARGS__}

/**
 * @brief Throws a bad mask exception if the mask type is not CV_8UC1 or CV_8SC1.
 *
 * @param[in] mask The mask to check.  This must be convertible to cv::InputArray.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_bad_mask_type(mask, ...) internal::_throw_if_bad_mask_type{mask, #mask __VA_OPT__(,) __VA_ARGS__}

/**
 * @brief Throws a bad mask exception if the mask depth is not CV_8U or CV_8S.
 *
 * @param[in] mask The mask to check.  This must be convertible to cv::InputArray.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_bad_mask_depth(mask, ...) internal::_throw_if_bad_mask_depth{mask, #mask __VA_OPT__(,) __VA_ARGS__}

/**
 * @brief Throws a bad mask exception if mask cannot be used with the array.
 *
 * The mask must have the same size as the array.
 *
 * If @pref{allowed_channels} is
 *  - AllowedMaskChannels::SINGLE: the mask must be a single channel
 *  - AllowedMaskChannels::SAME: the mask must have the same number of
 *    channels as the array
 *  - AllowedMaskChannels::SINGLE_OR_SAME: the mask must be a single channel or
 *    have the same number of channels as the array
 *
 * @param[in] array The array to check.  This must be convertible to cv::InputArray.
 * @param[in] mask The mask to check.  This must be convertible to cv::InputArray.
 * @param[in] allowed_channels The number of allowed mask channels.
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_if_bad_mask_for_array(array, mask, allowed_channels, ...) internal::_throw_if_bad_mask_for_array{array, mask, #array, #mask, allowed_channels __VA_OPT__(,) __VA_ARGS__}

/**
 * @brief Throws a not implemented exception.
 *
 * @param[in] ... Message strings, values, or objects.  These are combined using
 *            std::stringstream to generate the error message.
 */
#define throw_not_implemented(...) internal::_throw_not_implemented{__VA_ARGS__}

/**
 * @brief Throws a bad size exception if the array is not a single channel row or column vector.
 *
 * @param[in] array The array to check.  This must be convertible to cv::InputArray.
 * @param[in] channels The number of channels the array must have.  If this is
                       less than or equal to zero, any number of channels is
                       allowed.
 */
#define throw_if_not_vector(array, channels, ...) internal::_throw_if_not_vector{array, channels, #array __VA_OPT__(,) __VA_ARGS__}
}   // namespace cvwt

#endif  // CVWT_EXCEPTION_HPP

