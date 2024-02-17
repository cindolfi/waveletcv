/**
 * Performs the discrete wavelet transform of an image
*/
#include <iostream>
#include <numeric>
#include <vector>
#include <filesystem>
#include <cxxopts.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// #include <wavelet/wavelet.h>
#include <wavelet/dwt2d.hpp>


const std::string PROGRAM_NAME = "dwt2d";

cv::Mat open_image(const std::filesystem::path& input_filename)
{
    auto image = cv::imread(input_filename);
    if (image.empty()) {
        auto error_code = std::filesystem::exists(input_filename)
            ? std::make_error_code(std::errc::no_message)
            : std::make_error_code(std::errc::no_such_file_or_directory);

        throw std::filesystem::filesystem_error(
            "Failed to open image",
            std::filesystem::path(input_filename),
            error_code
        );
    }

    cv::Mat result(image.size(), CV_32FC3);
    image.convertTo(result, result.type(), 1.0 / 255.0);
    // image.convertTo(result, result.type());

    return result;
}

void save_image(const cv::Mat& image, const std::filesystem::path& output_filename)
{
    cv::imwrite(output_filename, image);
}

void show(
    const cv::Mat& image,
    const cv::Mat& dwt_image,
    // const Dwt2dResults& dwt_results,
    const DWT2D::Coeffs& dwt_coeffs,
    const std::filesystem::path& input_filename,
    const Wavelet& wavelet
)
{
    cv::imshow(
        "Image (" + input_filename.string() + ")",
        image
    );
    cv::imshow(
        // "DWT (" + wavelet.short_name() + ", " + std::to_string(dwt_results.depth()) + " levels)",
        "DWT (" + wavelet.short_name() + ", " + std::to_string(dwt_coeffs.depth()) + " levels)",
        dwt_image
    );
    cv::waitKey(0);
}

void print_available_wavelets()
{
    // std::cout << "Wavelets: ";
    // for (auto name : registered_wavelets())
    //     std::cout << name << ' ';
    // std::cout << std::endl;
}


cv::Mat standardize(const cv::Mat& image)
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(image.reshape(1, 1), mean, stddev);
    return (image - mean[0]) / stddev[0];
}

class invalid_options : public cxxopts::exceptions::exception {
public:
    invalid_options(const std::string& what_arg) : cxxopts::exceptions::exception(what_arg)
    {}
};

void verify_options(const cxxopts::Options& options, const cxxopts::ParseResult& result)
{
    if (!result.count("image_file"))
        throw invalid_options("missing input file");

    if (!result.count("wavelet"))
        throw invalid_options("missing wavelet");
}

template <typename T>
void print_minmax(const T& x, const std::string& header)
{
    if (!header.empty())
        std::cout << header << " ";

    const auto [min, max] = std::minmax_element(x.begin(), x.end());
    std::cout << "min = " << *min << "  max = " << *max << std::endl;
}

template <>
void print_minmax(const cv::Mat& x, const std::string& header)
{
    if (!header.empty())
        std::cout << header << " ";

    double min = 0.0;
    double max = 0.0;
    cv::minMaxLoc(x, &min, &max);
    std::cout << "min = " << min << "  max = " << max << std::endl;
    // const auto [min, max] = std::minmax_element(x.begin<double>(), x.end<double>());
    // std::cout << "min = " << *min << "  max = " << *max << std::endl;

}







// (512, 512, 3) uint8
// 255 3
// original:
//     type = uint8
//     min = 3 max = 255
// image:
//     type = float32
//     min = -2.1231012 max = 2.149264

// (133, 133, 3)
// dwt_image:
//     dwt_image = float32 (512, 512, 3)
//     min = -9.546701 max = 10.078802
//     h = 20.157604217529297

// final_image:
//     type = uint8 (525, 525, 3)
//     min = 6 max = 255



int main(int argc, char* argv[])
{
    cxxopts::Options options(PROGRAM_NAME);

    options.add_options()
        ("image_file", "Input image file", cxxopts::value<std::string>())
        ("w, wavelet", "Wavelet identifier", cxxopts::value<std::string>())
        ("d, depth", "The maximum number of DWT levels", cxxopts::value<int>()->default_value("0"))
        ("a, available-wavelets", "Print available wavelets identifiers")
        ("s, show", "Show image and DWT")
        ("o, out", "Output file path")
        ("h, help", "Print usage");

    options.parse_positional({"image_file"});

    try {
        auto parsed_options = options.parse(argc, argv);

        if (parsed_options.count("help")) {
            std::cout << options.help() << std::endl;
        } else if (parsed_options.count("available-wavelets")) {
            print_available_wavelets();
        } else {
            verify_options(options, parsed_options);

            auto input_filename = std::filesystem::canonical(
                parsed_options["image_file"].as<std::string>()
            );
            auto wavelet_name = parsed_options["wavelet"].as<std::string>();
            auto depth = parsed_options["depth"].as<int>();

            auto image = open_image(input_filename);

            auto wavelet = Wavelet::create(wavelet_name);
            // auto dwt_results = dwt2d(image, wavelet, depth);
            auto dwt_results = dwt2d(image, wavelet_name, depth);
            dwt_results.normalize();
            // auto dwt_image = dwt_results.as_matrix();
            cv::Mat dwt_image = dwt_results;

            // for (auto& level : dwt_results) {
            //     threshold(level.horizontal, t);
            //     threshold(level.horizontal, t);
            //     threshold(level.horizontal, t);
            //     threshold(level.horizontal, t);

            //     for (auto& detail : level.details())
            //         threshold(detail, h);
            // }

            if (parsed_options.count("out")) {
                auto output_filename = std::filesystem::canonical(
                    parsed_options["out"].as<std::string>()
                );
                save_image(dwt_image, output_filename);
            }

            if (parsed_options.count("show"))
                show(image, dwt_image, dwt_results, input_filename, wavelet);
        }
    } catch (const cxxopts::exceptions::exception& error) {
        // std::cerr << error.what() << std::endl;
        std::cerr << options.program() << ": " << error.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

