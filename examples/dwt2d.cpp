/**
 * Performs the discrete wavelet transform of an image
*/
#include <filesystem>
#include <cxxopts.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <wtcv/dwt2d.hpp>
#include "common.hpp"

using namespace wtcv;

void main_program(const cxxopts::ParseResult& args)
{
    auto [image, filepath] = open_image(args["image_file"].as<std::string>());
    auto wavelet = args["wavelet"].as<std::string>();
    auto levels = args["levels"].as<int>();
    auto split_channels = args["split-channels"].as<bool>();
    auto coeffs = levels > 0 ? dwt2d(image, wavelet, levels)
                             : dwt2d(image, wavelet);

    if (args.count("out"))
        save_coeffs(coeffs, args["out"].as<std::string>());

    bool wait_for_key_press = false;
    if (args.count("show")) {
        show_image(image, split_channels, "Image", filepath.filename());
        wait_for_key_press = true;
    }

    if (args.count("show-coeffs")) {
        auto normalization_method = args["show-coeffs"].as<std::string>();
        show_coeffs(coeffs, normalization_method, split_channels);
        wait_for_key_press = true;
    }

    if (wait_for_key_press)
        cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    //  PROGRAM_NAME is defined in CMakeLists.txt
    cxxopts::Options options(PROGRAM_NAME);
    add_common_options(options);
    options.add_options()
        (
            "s, show",
            "Show image."
        )
        (
            "o, out",
            "Save the denoised image.",
            cxxopts::value<double>()->no_implicit_value(),
            "FILE"
        );

    return execute(argc, argv, options, main_program);
}

