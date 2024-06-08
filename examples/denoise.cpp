/**
 * Denoises image using DWT
*/
#include <iostream>
#include <set>
#include <ranges>
#include <filesystem>
#include <chrono>
#include <cxxopts.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cvwt/dwt2d.hpp>
#include <cvwt/shrinkage.hpp>
#include "common.hpp"

using namespace cvwt;

const std::string PROGRAM_NAME = "dwt2d-denoise";
const std::set<std::string> AVAILABLE_SHRINK_METHODS = {
    "visu",
    "sure",
    "sure-levelwise",
    "sure-global",
    "sure-strict",
    "sure-strict-levelwise",
    "sure-strict-global",
    "bayes",
    "bayes-levelwise",
    "bayes-global",
};
const std::map<std::string, SureShrink::Optimizer> AVAILABLE_OPTIMIZERS = {
    {"auto", SureShrink::AUTO},
    {"sbplx", SureShrink::SBPLX},
    {"nelder-mead", SureShrink::NELDER_MEAD},
    {"cobyla", SureShrink::COBYLA},
    {"bobyqa", SureShrink::BOBYQA},
    {"direct", SureShrink::DIRECT},
    {"direct-l", SureShrink::DIRECT_L},
    {"brute-force", SureShrink::BRUTE_FORCE},
};

void show_images(
    const cv::Mat& image,
    const cv::Mat& noisy_image,
    const cv::Mat& denoised_image,
    const std::filesystem::path& filepath,
    double stdev,
    bool split_channels
)
{
    show_image(image, split_channels, "Image", filepath.filename());
    show_image(denoised_image, split_channels, "Denoised Image");
    if (stdev > 0)
        show_image(
            noisy_image,
            split_channels,
            "Image With Extra Noise",
            make_title("stdev = ", stdev)
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
        throw InvalidOptions("Missing --method.");

    auto method = args["method"].as<std::string>();
    if(!AVAILABLE_SHRINK_METHODS.contains(method)) {
        throw InvalidOptions(
            "Invalid --method: `"
            + method
            + "`. Must be one of: "
            + join(AVAILABLE_SHRINK_METHODS, ", ")
            + "."
        );
    }

    auto optimizer = args["optimizer"].as<std::string>();
    if(!AVAILABLE_OPTIMIZERS.contains(optimizer)) {
        throw InvalidOptions(
            "Invalid --optimizer: `"
            + optimizer
            + "`. Must be one of: "
            + join(std::views::keys(AVAILABLE_OPTIMIZERS), ", ")
            + "."
        );
    }

    if (args.count("add-noise") && args["add-noise"].as<double>() < 0)
        throw InvalidOptions("Invalid --add-noise. Must be a positive number.");
}


void main_program(const cxxopts::ParseResult& args)
{
    verify_args(args);
    auto [image, filepath] = open_image(args["image_file"].as<std::string>());
    auto wavelet = Wavelet::create(args["wavelet"].as<std::string>());
    int levels = args.count("levels") ? args["levels"].as<int>() : 0;
    double stdev = args.count("add-noise") ? args["add-noise"].as<double>() : 0.0;
    auto shrink_method = args["method"].as<std::string>();
    auto optimizer = AVAILABLE_OPTIMIZERS.at(args["optimizer"].as<std::string>());
    auto split_channels = args["split-channels"].as<bool>();

    cv::Mat noisy_image;
    if (stdev > 0)
        add_noise(image, noisy_image, stdev);
    else
        noisy_image = image;

    std::unique_ptr<Shrink> shrinker;
    if (shrink_method == "visu")
        shrinker = std::make_unique<VisuShrink>();
    else if (shrink_method == "sure")
        shrinker = std::make_unique<SureShrink>(
            SureShrink::SUBBANDS,
            SureShrink::HYBRID,
            optimizer
        );
    else if (shrink_method == "sure-levelwise")
        shrinker = std::make_unique<SureShrink>(
            Shrink::LEVELS,
            SureShrink::HYBRID,
            optimizer
        );
    else if (shrink_method == "sure-global")
        shrinker = std::make_unique<SureShrink>(
            Shrink::GLOBALLY,
            SureShrink::HYBRID,
            optimizer
        );
    else if (shrink_method == "sure-strict")
        shrinker = std::make_unique<SureShrink>(
            SureShrink::SUBBANDS,
            SureShrink::STRICT,
            optimizer
        );
    else if (shrink_method == "sure-strict-levelwise")
        shrinker = std::make_unique<SureShrink>(
            Shrink::LEVELS,
            SureShrink::STRICT,
            optimizer
        );
    else if (shrink_method == "sure-strict-global")
        shrinker = std::make_unique<SureShrink>(
            Shrink::GLOBALLY,
            SureShrink::STRICT,
            optimizer
        );
    else if (shrink_method == "bayes")
        shrinker = std::make_unique<BayesShrink>();
    else if (shrink_method == "bayes-levelwise")
        shrinker = std::make_unique<BayesShrink>(Shrink::LEVELS);
    else if (shrink_method == "bayes-global")
        shrinker = std::make_unique<BayesShrink>(Shrink::GLOBALLY);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto coeffs = levels > 0 ? dwt2d(noisy_image, wavelet, levels)
                             : dwt2d(noisy_image, wavelet);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> decompose_duration = end_time - start_time;

    cv::Mat thresholds;
    start_time = std::chrono::high_resolution_clock::now();
    auto shrunk_coeffs = shrinker->shrink(coeffs, thresholds);
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> shrink_duration = end_time - start_time;

    start_time = std::chrono::high_resolution_clock::now();
    auto denoised_image = shrunk_coeffs.reconstruct();
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> reconstruct_duration = end_time - start_time;

    CV_LOG_IF_INFO(NULL, shrink_method.starts_with("sure"), "SureShrink optimizer = " << args["optimizer"].as<std::string>());
    CV_LOG_INFO(NULL, "DWT decomposition took " << decompose_duration.count() << " milliseconds");
    CV_LOG_INFO(NULL, "Shrink coefficients took " << shrink_duration.count() << " milliseconds");
    CV_LOG_INFO(NULL, "DWT reconstruction took " << reconstruct_duration.count() << " milliseconds");
    CV_LOG_INFO(NULL, "Denoise took " << (decompose_duration + shrink_duration + reconstruct_duration).count() << " milliseconds");
    CV_LOG_INFO(NULL, "Thresholds = \n" << thresholds);

    bool wait_for_key_press = false;
    if (args.count("out"))
        save_image(denoised_image, args["out"].as<std::string>());

    if (args.count("show")) {
        show_images(
            image,
            noisy_image,
            denoised_image,
            filepath,
            stdev,
            split_channels
        );
        wait_for_key_press = true;
    }

    if (args.count("show-coeffs")) {
        auto normalization_method = args["show-coeffs"].as<std::string>();
        show_coeffs(
            coeffs,
            wavelet,
            normalization_method,
            split_channels,
            "DWT Coefficients",
            shrink_method
        );
        show_coeffs(
            shrunk_coeffs,
            wavelet,
            normalization_method,
            split_channels,
            "Shrunk DWT Coefficients",
            shrink_method
        );
        wait_for_key_press = true;
    }

    if (args.count("show-threshold-mask")) {
        cv::Mat thresholded_mask;
        greater_than(
            cv::abs(coeffs),
            shrinker->expand_thresholds(coeffs, thresholds),
            thresholded_mask,
            coeffs.detail_mask()
        );
        show_image(thresholded_mask, split_channels, "DWT Coefficient > Threshold Mask");
        wait_for_key_press = true;
    }

    if (wait_for_key_press)
        cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    cxxopts::Options options(PROGRAM_NAME);
    add_common_options(options);
    options.add_options()
        (
            "m, method",
            "The shrinkage algorithm.",
            cxxopts::value<std::string>()->default_value("bayes"),
            "[" + join(AVAILABLE_SHRINK_METHODS, "|") + "]"
        )
        (
            "s, show",
            "Show original and denoised images."
        )
        (
            "show-threshold-mask",
            "Show the mask indicating which coefficients are less than the shrink threshold.",
            cxxopts::value<bool>()->default_value("false")
        )
        (
            "add-noise",
            "Add extra gaussian noise to the image.",
            cxxopts::value<double>()->no_implicit_value(),
            "STDEV"
        )
        (
            "noise-stdev",
            "The known noise level corrupting the image. "
            "When not given, the noise is estimated from the image. "
            "Note: this does not add noise to the image (see --add-noise). "
            "The value passed to --add-noise is NOT added to this value automatically - "
            "users should typically add it manually.",
            cxxopts::value<double>()->no_implicit_value(),
            "STDEV"
        )
        (
            "optimizer",
            "The optimization algorithm used for computing SureShrink thresholds.",
            cxxopts::value<std::string>()->default_value("auto"),
            "[" + join(std::views::keys(AVAILABLE_OPTIMIZERS), "|") + "]"
        )
        (
            "o, out",
            "Save the denoised image.",
            cxxopts::value<double>()->no_implicit_value(),
            "FILE"
        );

    return execute(argc, argv, options, main_program);
}
