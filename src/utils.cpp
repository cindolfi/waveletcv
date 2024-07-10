#include "wtcv/utils.hpp"

namespace wtcv
{
namespace internal
{
std::string get_type_name(int type)
{
    std::string channels = std::to_string(CV_MAT_CN(type));
    switch (CV_MAT_DEPTH(type)){
        case CV_64F: return "CV_64FC" + channels;
        case CV_32F: return "CV_32FC" + channels;
        case CV_32S: return "CV_32SC" + channels;
        case CV_16S: return "CV_16SC" + channels;
        case CV_16U: return "CV_16UC" + channels;
        case CV_8S: return "CV_8SC" + channels;
        case CV_8U: return "CV_8UC" + channels;
    }

    return std::to_string(type);
}

cv::Scalar set_unused_channels(const cv::Scalar& scalar, int channels, double value)
{
    cv::Scalar result(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < channels; ++i)
        result[i] = scalar[i];

    return result;
}
}   // namespace internal
}   // namespace wtcv
