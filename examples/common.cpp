#include "common.hpp"

#include <iostream>
#include <sstream>
#include <set>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cvwt/wavelet.hpp>
#include <cvwt/utils.hpp>
#include <cvwt/array.hpp>

using namespace cvwt;

const std::array<std::string, 4> channel_names = {
    "blue",
    "green",
    "red",
    "alpha",
};
const int WINDOW_FLAGS = cv::WINDOW_NORMAL | cv::WINDOW_FREERATIO | cv::WINDOW_GUI_EXPANDED;

std::pair<cv::Mat, std::filesystem::path> open_image(
    const std::filesystem::path& filename,
    int type
)
{
    const int channels = CV_MAT_CN(type);
    const int depth = CV_MAT_DEPTH(type);
    auto filepath = std::filesystem::canonical(filename);
    int flags = cv::IMREAD_ANYDEPTH;
    if (channels == 1)
        flags |= cv::IMREAD_GRAYSCALE;
    else if (channels == 3)
        flags |= cv::IMREAD_COLOR;
    else if (channels == 4)
        flags |= cv::IMREAD_ANYCOLOR;
    else
        throw_bad_arg(
            "Invalid number of channels, got ", channels, ". ",
            "Must be 1, 3, or 4."
        );

    auto image = cv::imread(filepath, flags);
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

    //  The image might be BGR or grayscale when read using cv::IMREAD_ANYCOLOR
    //  (i.e. when the requested number of channels is 4).  Convert the image
    //  to the requested number of channels.
    if (flags & cv::IMREAD_ANYCOLOR) {
        assert(channels == 4);
        if (image.channels() == 3)
            cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
        else if (image.channels() == 1)
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGRA);
    }

    //  Since the image was read using cv::IMREAD_ANYDEPTH, we need to convert
    //  it to the requested depth.
    if (image.depth() < CV_32F && depth >= CV_32F)
        image.convertTo(image, type, 1.0 / 255.0);
    else if (image.depth() >= CV_32F && depth < CV_32F)
        image.convertTo(image, type, 255.0);
    else
        image.convertTo(image, type);

    return std::make_pair(image, filepath);
}

void save_image(const cv::Mat& image, const std::filesystem::path& filepath)
{
    cv::imwrite(std::filesystem::canonical(filepath), image);
}

void save_coeffs(const DWT2D::Coeffs& coeffs, const std::filesystem::path& filepath)
{
    save_image(coeffs.map_details_to_unit_interval(), filepath);
}

void finalize_normalize_details(cv::Mat& normalized_coeffs, const DWT2D::Coeffs& coeffs, bool scaled)
{
    if (scaled) {
        auto maximum = maximum_abs_value(normalized_coeffs, coeffs.detail_mask());
        normalized_coeffs = normalized_coeffs / maximum;
    }

    if (normalized_coeffs.channels() == 1 && coeffs.channels() != 1) {
        auto original_type = normalized_coeffs.type();
        normalized_coeffs.convertTo(normalized_coeffs, CV_32F);
        if (coeffs.channels() == 3)
            cv::cvtColor(normalized_coeffs, normalized_coeffs, cv::COLOR_GRAY2BGR);
        else if (coeffs.channels() == 4)
            cv::cvtColor(normalized_coeffs, normalized_coeffs, cv::COLOR_GRAY2BGRA);

        normalized_coeffs.convertTo(normalized_coeffs, original_type);
    }

    cv::normalize(
        coeffs.approx(),
        normalized_coeffs(coeffs.approx_rect()),
        1.0,
        0.0,
        cv::NORM_MINMAX
    );
}

cv::Mat normalize_details_abs(const DWT2D::Coeffs& coeffs, bool scaled)
{
    cv::Mat result = cv::abs(coeffs);
    finalize_normalize_details(result, coeffs, scaled);

    return result;
}

cv::Mat sum_channels(const cv::Mat& coeffs)
{
    std::vector<int> vector_shape = {(int)coeffs.total(), coeffs.channels()};
    cv::Mat reduced;
    cv::reduce(coeffs.reshape(1, vector_shape), reduced, 1, cv::REDUCE_SUM);

    return reduced.reshape(1, coeffs.rows);
}

cv::Mat normalize_details_l2(const DWT2D::Coeffs& coeffs, bool scaled)
{
    if (coeffs.channels() == 1)
        return normalize_details_abs(coeffs, scaled);

    cv::Mat result;
    cv::sqrt(sum_channels(static_cast<cv::Mat>(coeffs).mul(coeffs)), result);
    finalize_normalize_details(result, coeffs, scaled);

    return result;
}

cv::Mat normalize_details_l1(const DWT2D::Coeffs& coeffs, bool scaled)
{
    if (coeffs.channels() == 1)
        return normalize_details_abs(coeffs, scaled);

    cv::Mat abs_coeffs = cv::abs(coeffs);
    auto result = sum_channels(abs_coeffs);
    finalize_normalize_details(result, coeffs, scaled);

    return result;
}

cv::Mat normalize_details(
    const DWT2D::Coeffs& coeffs,
    const std::string& normalization_method
)
{
    if (normalization_method == "affine")
        return coeffs.map_details_to_unit_interval();
    else if (normalization_method == "l2-clip")
        return normalize_details_l2(coeffs, false);
    else if (normalization_method == "l2-scale")
        return normalize_details_l2(coeffs, true);
    else if (normalization_method == "l1-clip")
        return normalize_details_l1(coeffs, false);
    else if (normalization_method == "l1-scale")
        return normalize_details_l1(coeffs, true);
    else if (normalization_method == "abs-clip")
        return normalize_details_abs(coeffs, false);
    else if (normalization_method == "abs-scale")
        return normalize_details_abs(coeffs, true);

    assert(false);
    return cv::Mat();
}

void show_coeffs(
    const DWT2D::Coeffs& coeffs,
    const std::string& normalization_method,
    bool split_channels,
    const std::string& title,
    const std::string& title_info
)
{
    auto normalized_coeffs = normalize_details(coeffs, normalization_method);
    auto subtitle = title_info.empty() ? "" : ", " + title_info;

    show_image(
        normalized_coeffs,
        split_channels,
        title,
        make_title(
            coeffs.wavelet().name(), ", ",
            coeffs.levels(), " levels, ",
            normalization_method, (title_info.empty() ? "" : ", "),
            title_info
        )
    );
}

void show_image(
    const cv::Mat& image,
    bool split_channels,
    const std::string& title,
    const std::string& subtitle
)
{
    if (split_channels && image.channels() > 1) {
        std::vector<cv::Mat> image_channels;
        cv::split(image, image_channels);
        cv::Mat tiled_image(
            image.size().height,
            image.channels() * image.size().width,
            image.type(),
            cv::Scalar::all(0.0)
        );
        cv::Rect tile_rect(cv::Point(0, 0), image.size());
        for (int i = 0; i < image_channels.size(); ++i) {
            std::vector<int> from_to = {i, i};
            cv::mixChannels(image_channels, tiled_image(tile_rect), from_to);
            tile_rect += cv::Point(tile_rect.width, 0);
        }
        auto window_id = make_title(
            title,
            (subtitle.empty() ? "" : " ("),
            subtitle,
            (subtitle.empty() ? "" : ")")
        );
        cv::namedWindow(window_id, WINDOW_FLAGS);
        cv::imshow(window_id, tiled_image);
    } else {
        auto window_id = make_title(
            title,
            (subtitle.empty() ? "" : " ("),
            subtitle,
            (subtitle.empty() ? "" : ")")
        );
        cv::namedWindow(window_id, WINDOW_FLAGS);
        cv::imshow(window_id, image);
    }
}

void add_common_options(cxxopts::Options& options)
{
    options.add_options()
        (
            "h, help",
            ""
        )
        (
            "image_file",
            "The input image file.",
            cxxopts::value<std::string>(),
            "FILE"
        )
        (
            "w, wavelet",
            "The wavelet identifier (use --available-wavelets to see choices).",
            cxxopts::value<std::string>(),
            "WAVELET ID"
        )
        (
            "l, levels",
            "The maximum number of DWT levels. Computed from image size if not given.",
            cxxopts::value<int>()->default_value("0"),
            "LEVELS"
        )
        (
            "a, available-wavelets",
            "Print available wavelets identifiers."
        )
        (
            "show-coeffs",
            "Show the original and shrunk normalized DWT coefficients. "
            "Detail coefficients are mapped to the interval [0, 1]. "
            "The available maps are:\n"
            "  - affine: shift 0.0 to 0.5 and scale\n"
            "  - abs-clip: absolute value and clip\n"
            "  - abs-scale: absolute value and scale\n"
            "  - l2-clip: channel-wise L2 norm and clip\n"
            "  - l2-scale: channel-wise L2 norm and scale\n"
            "  - l1-clip: channel-wise L1 norm clip\n"
            "  - l1-scale: channel-wise L1 norm and scale",
            cxxopts::value<std::string>()->implicit_value("affine"),
            join(AVAILABLE_NORMALIZATION_METHODS, "|")
        )
        (
            "split-channels",
            "Show separate red, green, and blue channels for colored images.",
            cxxopts::value<bool>()->default_value("false")
        )
        (
            "log-level",
            "Set the OpenCV logging level.",
            cxxopts::value<std::string>()->default_value("warning"),
            "[" + join(std::views::keys(AVAILABLE_LOG_LEVELS), "|") + "]"
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
    auto available_wavelets = Wavelet::available_wavelets();
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

    if (args.count("show-coeffs")) {
        auto normlization_method = args["show-coeffs"].as<std::string>();
        if (!AVAILABLE_NORMALIZATION_METHODS.contains(normlization_method)) {
            throw InvalidOptions(
                "Invalid --show-coeffs: \""
                + normlization_method
                + "\". Must be one of: "
                + join(AVAILABLE_NORMALIZATION_METHODS, ", ")
                + "."
            );
        }
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
    for (const auto& name : Wavelet::available_wavelets())
        std::cout << name << " ";
    std::cout << "\n";
}

int execute(
    int argc,
    char* argv[],
    cxxopts::Options& options,
    void main_program(const cxxopts::ParseResult&)
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

            main_program(args);
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

