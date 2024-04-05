#include <iostream>
#include <sstream>
#include <set>
#include <opencv2/imgcodecs.hpp>
#include <wavelet/wavelet.hpp>
#include "common.hpp"

using namespace wavelet;

std::pair<cv::Mat, std::filesystem::path> open_image(const std::filesystem::path& input_filename, int type)
{
    auto filepath = std::filesystem::canonical(input_filename);
    auto image = cv::imread(filepath);
    if (image.empty()) {
        auto error_code = std::filesystem::exists(filepath)
            ? std::make_error_code(std::errc::no_message)
            : std::make_error_code(std::errc::no_such_file_or_directory);

        throw std::filesystem::filesystem_error(
            "Failed to open image",
            filepath,
            error_code
        );
    }

    cv::Mat result;
    image.convertTo(result, type, 1.0 / 255.0);

    return std::make_pair(result, filepath);
}

void save_image(const cv::Mat& image, const std::filesystem::path& filepath)
{
    cv::imwrite(std::filesystem::canonical(filepath), image);
}

void save_coeffs(const DWT2D::Coeffs& coeffs, const std::filesystem::path& filepath)
{
    auto normalized_coeffs = coeffs.clone();
    normalized_coeffs.normalize();
    save_image(normalized_coeffs, filepath);
}

void add_common_options(cxxopts::Options& options)
{
    options.add_options()
        (
            "image_file",
            "Input image file", cxxopts::value<std::string>()
        )
        (
            "w, wavelet",
            "Wavelet identifier [use --available-wavelets to see choices]", cxxopts::value<std::string>()
        )
        (
            "l, levels",
            "The maximum number of DWT levels", cxxopts::value<int>()->default_value("0")
        )
        (
            "a, available-wavelets",
            "Print available wavelets identifiers"
        )
        (
            "log-level",
            "Set the opencv logging level [" + join(std::views::keys(AVAILABLE_LOG_LEVELS), ", ") + "]",
            cxxopts::value<std::string>()->default_value("warning")
        )
        (
            "h, help",
            "Print usage"
        );

    options.parse_positional({"image_file"});
}

void verify_common_args(const cxxopts::ParseResult& args)
{
    if (!args.count("image_file"))
        throw InvalidOptions("missing input file");

    if (!args.count("wavelet"))
        throw InvalidOptions("missing wavelet");

    auto wavelet = args["wavelet"].as<std::string>();
    auto available_wavelets = Wavelet::registered_wavelets();
    if (std::ranges::find(available_wavelets, wavelet) == available_wavelets.end()) {
        throw InvalidOptions(
            "invalid wavelet - must be one of: "
            + join(available_wavelets, ", ")
        );
    }

    auto log_level = args["log-level"].as<std::string>();
    if(!AVAILABLE_LOG_LEVELS.contains(log_level)) {
        throw InvalidOptions(
            "invalid log-level - must be one of: "
            + join(std::views::keys(AVAILABLE_LOG_LEVELS), ", ")
        );
    }
}

void set_log_level_from_args(const cxxopts::ParseResult& args)
{
    auto log_level = args["log-level"].as<std::string>();
    cv::utils::logging::setLogLevel(AVAILABLE_LOG_LEVELS.at(log_level));
}

void print_available_wavelets()
{
    std::cout << "Wavelets: ";
    for (const auto& name : Wavelet::registered_wavelets())
        std::cout << name << " ";
    std::cout << "\n";
}

int execute(
    int argc,
    char* argv[],
    cxxopts::Options& options,
    void body(const cxxopts::ParseResult&)
)
{
    try {
        auto args = options.parse(argc, argv);
        if (args.count("help")) {
            std::cout << options.help() << "\n";
        } else if (args.count("available-wavelets")) {
            print_available_wavelets();
        } else {
            verify_common_args(args);
            set_log_level_from_args(args);

            body(args);
        }
    } catch (const cxxopts::exceptions::exception& error) {
        std::cerr
            << options.program() << ": " << error.what() << "\n"
            << options.help() << "\n";
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

