/**
 * Denoises image using DWT
*/
#include <iostream>
#include <set>
#include <ranges>
#include <filesystem>
#include <cxxopts.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <wavelet/dwt2d.hpp>
#include <wavelet/shrinkage.hpp>
#include "common.hpp"

using namespace wavelet;

const std::string PROGRAM_NAME = "denoise";
const std::set<std::string> AVAILABLE_SHRINK_METHODS = {
    "sure",
    "sure-levelwise",
    "hybrid-sure",
    "hybrid-sure-levelwise",
    "bayes",
    "visu-soft",
    "visu-hard",
};

void show_images(
    const cv::Mat& image,
    const cv::Mat& noisy_image,
    const cv::Mat& denoised_image,
    const std::filesystem::path& filepath,
    double stdev
)
{
    cv::imshow(make_title("Image (", filepath.filename(), ")"), image);

    if (stdev > 0)
        cv::imshow(make_title("Image With Extra Noise (stdev =", stdev, ")"), noisy_image);
    cv::imshow(make_title("Denoised Image"), denoised_image);
}

void show_coeffs(
    const DWT2D::Coeffs& coeffs,
    const DWT2D::Coeffs& denoised_coeffs,
    const std::string& method,
    const Wavelet& wavelet
)
{
    auto normalized_coeffs = coeffs.clone();
    normalized_coeffs.normalize();
    cv::imshow(
        make_title("DWT Coefficients(", wavelet.short_name(), ", ",
                   coeffs.levels(), " levels)"),
        normalized_coeffs
    );

    auto normalized_denoised_coeffs = denoised_coeffs.clone();
    normalized_denoised_coeffs.normalize();
    cv::imshow(
        make_title("Shrunk DWT Coefficients(", wavelet.short_name(), ", ",
                   coeffs.levels(), " levels, ", method, ")"),
        normalized_denoised_coeffs
    );
}

void add_noise(cv::InputArray input, cv::OutputArray output, double stdev)
{
    auto image = input.getMat();
    cv::Mat noise(image.size(), image.type(), 0.0);
    cv::randn(noise, 0, stdev);
    cv::add(image, noise, output);
}

void verify_args(const cxxopts::ParseResult& args)
{
    if (!args.count("method"))
        throw InvalidOptions("missing method");

    if(!AVAILABLE_SHRINK_METHODS.contains(args["method"].as<std::string>())) {
        throw InvalidOptions(
            "invalid method - must be one of: "
            + join(AVAILABLE_SHRINK_METHODS, ", ")
        );
    }

    if (args["add-noise"].as<double>() < 0)
        throw InvalidOptions("add-noise must be a positive number");
}

void main_program(const cxxopts::ParseResult& args)
{
    verify_args(args);
    auto [image, filepath] = open_image(args["image_file"].as<std::string>());
    auto wavelet = Wavelet::create(args["wavelet"].as<std::string>());
    auto depth = args["levels"].as<int>();
    double stdev = args["add-noise"].as<double>();

    cv::Mat noisy_image;
    if (stdev > 0)
        add_noise(image, noisy_image, stdev);
    else
        noisy_image = image;

    auto coeffs = dwt2d(noisy_image, wavelet, depth);
    auto shrunk_coeffs = coeffs.clone();
    auto shrink_method = args["method"].as<std::string>();
    if (shrink_method == "visu-hard")
        visu_hard_shrink(shrunk_coeffs);
    else if (shrink_method == "visu-soft")
        visu_soft_shrink(shrunk_coeffs);
    else if (shrink_method == "sure")
        sure_shrink(shrunk_coeffs);
    else if (shrink_method == "sure-levelwise")
        sure_shrink_levelwise(shrunk_coeffs);
    else if (shrink_method == "hybrid-sure")
        hybrid_sure_shrink(shrunk_coeffs);
    else if (shrink_method == "hybrid-sure-levelwise")
        hybrid_sure_shrink_levelwise(shrunk_coeffs);
    else if (shrink_method == "bayes")
        bayes_shrink(shrunk_coeffs);

    auto denoised_image = idwt2d(shrunk_coeffs, wavelet);

    if (args.count("out"))
        save_image(denoised_image, args["out"].as<std::string>());

    if (args.count("show"))
        show_images(image, noisy_image, denoised_image, filepath, stdev);

    if (args.count("show-coeffs"))
        show_coeffs(coeffs, shrunk_coeffs, shrink_method, wavelet);

    if (args.count("show") || args.count("show-coeffs"))
        cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    cxxopts::Options options(PROGRAM_NAME);
    add_common_options(options);
    options.add_options()
        (
            "m, method",
            "The shrinkage algorithm [" + join(AVAILABLE_SHRINK_METHODS, ", ") + "]",
            cxxopts::value<std::string>()->default_value("sure")
        )
        (
            "s, show",
            "Show original and denoised images"
        )
        (
            "show-coeffs",
            "Show original and shrunk DWT coefficients"
        )
        (
            "add-noise",
            "Add extra gaussian noise with given standard deviation",
            cxxopts::value<double>()->default_value("0.0")
        )
        (
            "o, out",
            "Output file path"
        );

    return execute(argc, argv, options, main_program);
}


