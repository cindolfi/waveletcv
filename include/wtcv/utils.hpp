#ifndef CVWT_UTILS_HPP
#define CVWT_UTILS_HPP

#include <utility>
#include <opencv2/core.hpp>

namespace std
{
#ifndef __cpp_lib_unreachable
[[noreturn]] inline void unreachable()
{
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
    __assume(false);
#else // GCC, Clang
    __builtin_unreachable();
#endif
}
#endif
}

namespace wtcv
{
namespace internal
{
std::string get_type_name(int type);

cv::Scalar set_unused_channels(const cv::Scalar& scalar, int channels, double value = 0.0);

struct Index {
    int row;
    int col;
};

inline Index unravel_index(const cv::Mat& array, int flat_index)
{
    return {
        .row = flat_index / array.cols,
        .col = flat_index % array.cols
    };
}
}   // namespace internal
}   // namespace wtcv

#endif  // CVWT_UTILS_HPP

