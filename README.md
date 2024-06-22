
[![Latest Docs](https://readthedocs.org/projects/nlopt/badge/?version=latest)](http://nlopt.readthedocs.io/en/latest/)
[![Build Status](https://github.com/stevengj/nlopt/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/stevengj/nlopt/actions/workflows/build.yml)

# WaveletCV

- [Features](#features)
- [Build & Install](#build)
- [Usage](#usage)
- [Documentation](#documentation)
- [License](#license)

WaveletCV is a wavelet library.  It is a companion to OpenCV.

### Features {#features}
- 2D multilevel discrete wavelet transform
- Built-in Daubechies, Symlets, Coiflets, and B-spline Biorthogonal wavelet families
- Support for custom wavelets
- DWT coefficient shrinking algorithms for denoising and compression applications
- Double or single precision


### Build & Install {#build}

WaveletCV is built and installed using the [cmake](https://cmake.org/cmake/help/latest/manual/cmake.1.html) build system (version 3.24.0 or newer).

The latest source code is available from the [WaveletCV releases](https://github.com/cindolfi/waveletcv/releases) page on Github. Alternatively, clone the [WaveletCV repository](https://github.com/cindolfi/cvwt.git) and checkout the `latest` branch.

```bash
# Download
git clone https://github.com/cindolfi/cvwt.git
cd cvwt
git checkout latest

# Configure
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake ---build build

# Install
sudo cmake --install build

# Uninstall
cd build
sudo make uninstall
```

> :memo: **Note:** For single configuration providers (e.g. gcc, clang), the `CMAKE_BUILD_TYPE` variable **must** be set to one of:
> - Debug
> - Release


WaveletCV is installed to `/usr/local` by default. To install to a specified directory use:
```bash
cmake --install build --install-prefix <install-path>
```

See the [build and install documentation](http://waveletcv.readthedocs.org/install) for more information.

### Usage {#usage}

#### Wavelet Objects
```cpp
#include <cvwt/wavelet.hpp>
```
Builtin wavelets are created by name
```cpp
cvwt::Wavelet wavelet = cvwt::Wavelet::create("db2");
```
or by factory
```cpp
cvwt::Wavelet wavelet = cvwt::create_daubuchies(2);
```
Accessing the filter banks decomposition and reconstruction kernels
```cpp
wavelet.filter_bank().decompose_kernels().lowpass()
wavelet.filter_bank().decompose_kernels().highpass()
wavelet.filter_bank().reconstruct_kernels().lowpass()
wavelet.filter_bank().reconstruct_kernels().highpass()
```
See
- The [Wavelet API](http://waveletcv.readthedocs.org/wavelet) documentation.

#### Discrete Wavelet Transform (DWT)
```cpp
#include <cvwt/dwt2d.hpp>
```
Performing a discrete wavelet transformation (DWT) of an image is done using a functional style
```cpp
cv::Mat image = cv::imread(filename);
DWT2D::Coeffs coeffs = cvwt::dwt2d(image, "db2");
```
or an object oriented approach
```cpp
cvwt::Wavelet wavelet = cvwt::Wavelet::create("db2");
int levels = 2;
cvwt::DWT2D dwt(wavelet);
cvwt::DWT2D::Coeffs coeffs = dwt(image, levels);
```
Reconstruct the image by inverting the DWT
```cpp
cv::Mat reconstructed_image = coeffs.reconstruct();
```
##### Accessing Coefficients
Access the approximation coefficients
```cpp
cv::Mat approx_coeffs = coeffs.approx();
```
Access the finest scale (i.e. highest resolution) horizontal subband coefficients
```cpp
cv::Mat finest_horizontal_coeffs = coeffs.horizontal_detail(0);
coeffs.set_horizontal_detail(0, finest_horizontal_coeffs);
```
Or use the parameterized subband version
```cpp
cv::Mat finest_horizontal_coeffs = coeffs.detail(cvwt::HORIZONTAL, 0);
coeffs.set_detail(0, cvwt::HORIZONTAL, finest_horizontal_coeffs);
```
Use negative level indexing to access the coarsest scale (i.e. lowest resolution) horizontal subband coefficients
```cpp
// Equivalent to coeffs.horizontal_detail(coeffs.levels() - 1)
cv::Mat coarsest_horizontal_coeffs = coeffs.horizontal_detail(-1);
coeffs.set_horizontal_detail(-1, coarsest_horizontal_coeffs);
```
Get horizontal detail coefficients at every scale
```cpp
std::vector<cv::Mat> horizontal_details = coeffs.collect_horizontal_details();
```
> :memo: **Note:** The horizontal detail coefficients are used for illustration.  There are corresponding accessors for vertical and diagonal detail coefficients.

Get detail coefficients at every scale and subband
```cpp
std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>> details = coeffs.details();
std::vector<cvwt::DWT2D::Coeffs::DetailTuple>> details = coeffs.details();
```

See
- The [cvwt-dwt2d example](https://github.com/cindolfi/waveletcv/examples/dwt2d.cpp) for a complete example.
- The [Discrete Wavelet Transform API](http://waveletcv.readthedocs.org/dwt2d) documentation.

#### Shrink DWT Coefficients
```cpp
#include <cvwt/shrink.hpp>
```
Shrinking DWT coefficients is the basis for many denoising and compression applications. There are several shrinking algorithms implemented. Take the BayesShrink algorithm as an example
```cpp
coeffs = cvwt::bayes_shrink(coeffs);
```
Alternatively, the object oriented API can be used in a polymorphic way
```cpp
cvwt::Shrinker* shrinker = new cvwt::BayesShrinker();
coeffs = shrinker->shrink(coeffs);
```
or as a function object
```cpp
cvwt::BayesShrinker shrink;
coeffs = shrink(coeffs);
```
The functional API is simpler and more succinct, whereas the object oriented API offers more options to fine tune the algorithm.

See
- The [cvwt-denoise example](https://github.com/cindolfi/waveletcv/examples/denoise.cpp) for a complete example.
- The [Shrinker API](http://waveletcv.readthedocs.org/shrink) documentation.


### Documentation {#documentation}
Documentation is hosted at http://waveletcv.readthedocs.org.

### License {#license}
WaveletsCV is free open source software released under the MIT license.

