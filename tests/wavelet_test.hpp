/**
 * Wavelet & DWT2D Unit Tests
*/
#ifndef WAVELET_TEST_HPP
#define WAVELET_TEST_HPP

#include <wavelet/wavelet.hpp>
#include "common.hpp"

struct WaveletTestParam
{
    int vanishing_moments_psi;
    int vanishing_moments_phi;
    bool orthogonal;
    bool biorthogonal;
    wavelet::Wavelet::Symmetry symmetry;
    std::string family;
    std::string name;
    std::vector<double> decompose_lowpass;
    std::vector<double> decompose_highpass;
    std::vector<double> reconstruct_lowpass;
    std::vector<double> reconstruct_highpass;
};

//  map(family -> map(wavelet name -> param))
using WaveletTestCases = std::map<std::string, std::map<std::string, WaveletTestParam>>;

void PrintTo(const WaveletTestParam& param, std::ostream* stream);

class WaveletTest : public testing::TestWithParam<WaveletTestParam>
{
protected:
    WaveletTest() :
        testing::TestWithParam<WaveletTestParam>(),
        wavelet(wavelet::Wavelet::create(GetParam().name))
    {}

    wavelet::Wavelet wavelet;
};

std::vector<WaveletTestParam> concat(auto&& ...test_cases)
{
    std::vector<WaveletTestParam> result;
    (result.insert(result.end(), test_cases.begin(), test_cases.end()), ...);

    return result;
}

#endif  // WAVELET_TEST_HPP

