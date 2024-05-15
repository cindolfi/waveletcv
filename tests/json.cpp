/**
*/
#include <vector>
#include <array>
#include <string>
#include "common.hpp"
#include "json.hpp"

namespace cv
{
    void from_json(const json& json_matrix, Mat& matrix)
    {
        Mat data(json_matrix["data"].get<std::vector<double>>(), true);
        auto shape = json_matrix["shape"].get<std::vector<int>>();
        int type;
        if (json_matrix["dtype"] == "float64")
            type = CV_64F;
        else if (json_matrix["dtype"] == "float32")
            type = CV_32F;
        else
            assert(false);

        if (data.empty()) {
            matrix.create(0, 0, type);
        } else {
            data.convertTo(matrix, type);
            int channels = (shape.size() == 3) ? shape[2] : 1;
            matrix = matrix.reshape(channels, {shape[0], shape[1]});
        }
    }

    void from_json(const json& json_matrix, Mat4d& matrix)
    {
        Mat data = json_matrix.get<Mat>();
        matrix.create(data.size());
        if (!matrix.empty()) {
            matrix = 0.0;
            std::vector<int> from_to(2 * data.channels());
            for (int i = 0; i < data.channels(); ++i) {
                from_to[2 * i] = i;
                from_to[2 * i + 1] = i;
            }
            cv::mixChannels(data, matrix, from_to);
        }
    }

    void from_json(const json& json_scalar, Scalar& scalar)
    {
        auto shape = json_scalar["shape"].get<std::vector<int>>();
        auto data = json_scalar["data"].get<std::vector<double>>();
        scalar = 0;
        for (int i = 0; i < data.size(); ++i)
            scalar[i] = data[i];
    }
}

namespace cvwt
{
    void from_json(const json& json_coeffs, DWT2D::Coeffs& coeffs)
    {
        auto image_size = json_coeffs["image_size"].get<std::array<int, 2>>();

        DWT2D dwt(Wavelet::create(json_coeffs["wavelet"].get<std::string>()));

        coeffs = dwt.create_coeffs(
            json_coeffs["matrix"].get<cv::Mat>(),
            cv::Size(image_size[1], image_size[0]),
            json_coeffs["levels"].get<int>()
        );
    }
}

