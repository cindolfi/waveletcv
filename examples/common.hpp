#ifndef CVWT_EXAMPLES_COMMON_HPP
#define CVWT_EXAMPLES_COMMON_HPP

#include <map>
#include <set>
#include <ranges>
#include <string>
#include <filesystem>
#include <experimental/iterator>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <wtcv/dwt2d.hpp>
#include <cxxopts.hpp>

const std::map<std::string, cv::utils::logging::LogLevel> AVAILABLE_LOG_LEVELS = {
    {"fatal", cv::utils::logging::LOG_LEVEL_FATAL},
    {"error", cv::utils::logging::LOG_LEVEL_ERROR},
    {"warning", cv::utils::logging::LOG_LEVEL_WARNING},
    {"info", cv::utils::logging::LOG_LEVEL_INFO},
    {"debug", cv::utils::logging::LOG_LEVEL_DEBUG},
    {"verbose", cv::utils::logging::LOG_LEVEL_VERBOSE},
};
const std::set<std::string> AVAILABLE_NORMALIZATION_METHODS = {
    "affine",
    "l2-clip",
    "l2-scale",
    "l1-clip",
    "l1-scale",
    "abs-clip",
    "abs-scale",
};

std::pair<cv::Mat, std::filesystem::path> open_image(
    const std::filesystem::path& filename,
    int type = CV_64FC3
);
void save_image(const cv::Mat& image, const std::filesystem::path& filepath);
void save_coeffs(const wtcv::DWT2D::Coeffs& coeffs, const std::filesystem::path& filepath);
cv::Mat normalize_details(const wtcv::DWT2D::Coeffs& coeffs, const std::string& normalization_method);
void show_image(
    const cv::Mat& image,
    bool split_channels,
    const std::string& title,
    const std::string& title_info = ""
);
void show_coeffs(
    const wtcv::DWT2D::Coeffs& coeffs,
    const std::string& normalization_method,
    bool split_channels,
    const std::string& title = "DWT Coefficients",
    const std::string& title_info = ""
);

void add_common_options(cxxopts::Options& options);
void verify_common_args(const cxxopts::ParseResult& args);
void set_log_level_from_args(const cxxopts::ParseResult& args);
void print_available_wavelets();
int execute(
    int argc,
    char* argv[],
    cxxopts::Options& options,
    void body(const cxxopts::ParseResult&)
);

class InvalidOptions : public cxxopts::exceptions::exception {
public:
    InvalidOptions(const std::string& what_arg) : cxxopts::exceptions::exception(what_arg)
    {}
};

std::string join(const std::ranges::range auto& items, const std::string& delim)
{
    std::stringstream stream;
    auto joiner = std::experimental::make_ostream_joiner(stream, delim);
    std::copy(items.begin(), items.end(), joiner);

    return stream.str();
}

std::string make_title(auto&& ...parts)
{
    std::stringstream title;
    (title << ... << parts);

    return title.str();
}

#endif  // CVWT_EXAMPLES_COMMON_HPP

