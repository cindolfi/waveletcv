#include "wavelet/utils.hpp"

namespace wavelet
{
void flatten(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat matrix;
    if (input.isSubmatrix())
        input.copyTo(matrix);
    else
        matrix = input.getMat();

    matrix.reshape(0, 1).copyTo(output);
}

void collect_masked(cv::InputArray input, cv::OutputArray output, cv::InputArray mask)
{
    internal::dispatch_on_pixel_type<internal::collect_masked>(
        input.type(),
        input,
        output,
        mask
    );
}

bool equals(const cv::Mat& a, const cv::Mat& b)
{
    return &a == &b || (
        a.size() == b.size()
        && (a.data == b.data || cv::sum(a != b) == cv::Scalar(0, 0, 0, 0))
    );
}

cv::Scalar median(cv::InputArray input)
{
    cv::Scalar result;
    internal::dispatch_on_pixel_type<internal::median>(
        input.type(),
        input,
        result
    );

    return result;
}

cv::Scalar median(cv::InputArray input, cv::InputArray mask)
{
    cv::Scalar result;
    internal::dispatch_on_pixel_type<internal::median>(
        input.type(),
        input,
        mask,
        result
    );

    return result;
}
}   // namespace wavelet
