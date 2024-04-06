/**
 * Performs the discrete wavelet transform of an image
*/
#include <filesystem>
#include <cxxopts.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cvwt/dwt2d.hpp>
#include "common.hpp"

using namespace cvwt;

const std::string PROGRAM_NAME = "dwt2d";

void show(
    const cv::Mat& image,
    const DWT2D::Coeffs& coeffs,
    const std::filesystem::path& filepath,
    const Wavelet& wavelet
)
{
    cv::imshow(make_title("Image (", filepath.filename(), ")"), image);

    auto normalized_coeffs = coeffs.clone();
    normalized_coeffs.normalize();
    cv::imshow(
        make_title("DWT Coefficients(", wavelet.name(), ", ",
                   coeffs.levels(), " levels)"),
        normalized_coeffs
    );
}

void main_program(const cxxopts::ParseResult& args)
{
    auto [image, filepath] = open_image(args["image_file"].as<std::string>());
    auto wavelet = Wavelet::create(args["wavelet"].as<std::string>());
    auto levels = args["levels"].as<int>();
    auto coeffs = dwt2d(image, wavelet, args["levels"].as<int>());

    if (args.count("out"))
        save_coeffs(coeffs, args["out"].as<std::string>());

    if (args.count("show")) {
        show(image, coeffs, filepath, wavelet);
        cv::waitKey(0);
    }
}

int main(int argc, char* argv[])
{
    cxxopts::Options options(PROGRAM_NAME);
    add_common_options(options);
    options.add_options()
        (
            "s, show",
            "Show image and DWT"
        )
        (
            "o, out",
            "Output file path"
        );

    return execute(argc, argv, options, main_program);
}

