#ifndef CVWT_TEST_JSON_HPP
#define CVWT_TEST_JSON_HPP

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <wtcv/dwt2d.hpp>

using json = nlohmann::json;

namespace cv
{
    void from_json(const json& json_matrix, Mat& matrix);
    void from_json(const json& json_matrix, Mat4d& matrix);
    void from_json(const json& json_scalar, Scalar& scalar);
}

namespace wtcv
{
    void from_json(const json& json_coeffs, DWT2D::Coeffs& coeffs);
}

#endif  // CVWT_TEST_JSON_HPP
