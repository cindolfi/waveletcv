#include "wavelet/utils.hpp"

namespace cvwt
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
    if (a.dims != b.dims || a.size != b.size)
        return false;

    const cv::Mat* matrices[2] = {&a, &b};
    cv::Mat planes[2];
    cv::NAryMatIterator it(matrices, planes, 2);
    for (int p = 0; p < it.nplanes; ++p, ++it)
        if (cv::countNonZero(it.planes[0] != it.planes[1]) != 0)
            return false;

    return true;
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

void negate_evens(cv::InputArray input, cv::OutputArray output)
{
    internal::dispatch_on_pixel_type<internal::NegateEveryOther, 0>(
        input.type(),
        input,
        output
    );
}

void negate_odds(cv::InputArray input, cv::OutputArray output)
{
    internal::dispatch_on_pixel_type<internal::NegateEveryOther, 1>(
        input.type(),
        input,
        output
    );
}
}   // namespace cvwt
