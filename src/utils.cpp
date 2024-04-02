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
    if (a.dims != b.dims || a.size != b.size)
        return false;

    const cv::Mat* matrices[2] = {&a, &b};
    cv::Mat planes[2];
    cv::NAryMatIterator it(matrices, planes, 2);
    for (int p = 0; p < it.nplanes; ++p, ++it)
        if (cv::countNonZero(it.planes[0] != it.planes[1]) != 0)
            return false;

    return true;

    // //  We're using cv::sum() instead of cv::countNonzero() because the latter
    // //  only works with single channel matrices.
    // return &a == &b || (
    //     a.size() == b.size()
    //     && a.channels() == b.channels()
    //     && (a.data == b.data || cv::sum(a != b) == cv::Scalar(0, 0, 0, 0))
    // );
}

// bool matIsEqual(const cv::Mat Mat1, const cv::Mat Mat2)
// {
//   if( Mat1.dims == Mat2.dims &&
//     Mat1.size == Mat2.size &&
//     Mat1.elemSize() == Mat2.elemSize())
//   {
//     if( Mat1.isContinuous() && Mat2.isContinuous())
//     {
//       return 0==memcmp( Mat1.ptr(), Mat2.ptr(), Mat1.total()*Mat1.elemSize());
//     }
//     else
//     {
//       const cv::Mat* arrays[] = {&Mat1, &Mat2, 0};
//       uchar* ptrs[2];
//       cv::NAryMatIterator it( arrays, ptrs, 2);
//       for(unsigned int p = 0; p < it.nplanes; p++, ++it)
//         if( 0!=memcmp( it.ptrs[0], it.ptrs[1], it.size*Mat1.elemSize()) )
//           return false;

//       return true;
//     }
//   }

//   return false;
// }

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
