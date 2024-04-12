#include "cvwt/utils.hpp"

namespace cvwt
{
void flatten(cv::InputArray array, cv::OutputArray result)
{
    cv::Mat matrix;
    if (array.isSubmatrix())
        array.copyTo(matrix);
    else
        matrix = array.getMat();

    matrix.reshape(0, 1).copyTo(result);
}

void collect_masked(cv::InputArray array, cv::OutputArray result, cv::InputArray mask)
{
    internal::dispatch_on_pixel_type<internal::collect_masked>(
        array.type(),
        array,
        result,
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

cv::Scalar median(cv::InputArray array)
{
    cv::Scalar result;
    internal::dispatch_on_pixel_type<internal::median>(
        array.type(),
        array,
        result
    );

    return result;
}

cv::Scalar median(cv::InputArray array, cv::InputArray mask)
{
    cv::Scalar result;
    internal::dispatch_on_pixel_type<internal::median>(
        array.type(),
        array,
        mask,
        result
    );

    return result;
}

void negate_evens(cv::InputArray vector, cv::OutputArray result)
{
    internal::dispatch_on_pixel_type<internal::NegateEveryOther, 0>(
        vector.type(),
        vector,
        result
    );
}

void negate_odds(cv::InputArray vector, cv::OutputArray result)
{
    internal::dispatch_on_pixel_type<internal::NegateEveryOther, 1>(
        vector.type(),
        vector,
        result
    );
}
}   // namespace cvwt
