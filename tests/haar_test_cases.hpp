#ifndef WAVELET_HAAR_WAVELET_TEST_HPP
#define WAVELET_HAAR_WAVELET_TEST_HPP

#include <vector>
#include <map>
#include <string>

#include "wavelet_test.hpp"

std::map<std::string, WaveletTestParam> haar_coeffs_test_cases = {
    //  ----------------------------------------------------------------------------
    { // haar
        "haar",
        {
            .vanishing_moments_psi = 1,
            .vanishing_moments_phi = 0,
            .orthogonal = true,
            .biorthogonal = true,
            .symmetry = wavelet::Wavelet::Symmetry::ASYMMETRIC,
            .family = "Haar",
            .name = "haar",
            .decompose_lowpass = {
                7.0710678118654757274e-1,
                7.0710678118654757274e-1,
            },
            .decompose_highpass = {
                -7.0710678118654757274e-1,
                7.0710678118654757274e-1,
            },
            .reconstruct_lowpass = {
                7.0710678118654757274e-1,
                7.0710678118654757274e-1,
            },
            .reconstruct_highpass = {
                7.0710678118654757274e-1,
                -7.0710678118654757274e-1,
            },
        },
    }, // haar
};

#endif  // WAVELET_HAAR_WAVELET_TEST_HPP

