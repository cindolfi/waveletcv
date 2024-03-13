#ifndef WAVELET_DWT2D_TEST_HPP
#define WAVELET_DWT2D_TEST_HPP

#include <wavelet/dwt2d.hpp>
#include <vector>
#include <map>
#include "common.hpp"


using DWT2DTestInputs = std::map<
    std::string, // input name
    std::vector<double> // values
>;
using DWT2DTestCases = std::map<
    int, // level
    std::map<
        std::string, // wavelet name
        std::map<
            std::string, // input name
            std::vector<double> // coefficients
        >
    >
>;

using Suckit = std::map<
    std::string, // wavelet name
    std::map<
        std::string, // input name
        std::vector<double> // coefficients
    >
>;

#endif  // WAVELET_DWT2D_TEST_HPP

