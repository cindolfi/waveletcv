.. |.installation| replace:: Installation
.. _.installation: https://wavletcv.readthedocs.io/en/latest/installation.html#installation


        .. |release| replace:: 0.1.0
        .. |author| replace:: Christopher Indolfi
        .. |cmake_version| replace:: 3.24
        .. |github_url| replace:: https://github.com/cindolfi/waveletcv
        .. |github_repo| replace:: waveletcv
        .. |github_version| replace:: 0.1.0
        .. |github_version_tag| replace:: v0.1.0
    

.. |Build Status| image:: https://img.shields.io/github/actions/workflow/status/cindolfi/waveletcv/build-multi-platform.yml?branch=master&event=push&logo=github&label=Build
   :alt: GitHub Actions Workflow Status
   :target: https://github.com/cindolfi/waveletcv/actions

.. |Release| image:: https://img.shields.io/github/v/release/cindolfi/waveletcv?logo=github&label=Latest%20Release
   :alt: GitHub Release
   :target: https://github.com/cindolfi/waveletcv/releases/latest

.. |Documentation Status| image:: https://img.shields.io/readthedocs/waveletcv?logo=read%20the%20docs&label=Docs
   :alt: Read the Docs
   :target: https://waveletcv.readthedocs.io/en/latest/

.. |MIT license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/cindolfi/waveletcv/blob/master/LICENSE

.. list-table::
   :width: 100%
   :class: borderless

   * - |Build Status|
     - |Release|
     - |Documentation Status|
     - |MIT license|

WaveletCV
=========

WaveletCV is a wavelet library.  It is a companion to OpenCV.

Features
--------

- 2D multilevel discrete wavelet transform
- Built-in Daubechies, Symlets, Coiflets, and B-spline Biorthogonal wavelet families
- Support for custom wavelets
- DWT coefficient shrinking algorithms for denoising and compression applications
- Double or single precision
- Functional and Object Oriented Interfaces


Build & Install
---------------

WaveletCV is built and installed using
`cmake <https://cmake.org/cmake/help/latest/manual/cmake.1.html>`_
|cmake_version| or newer.

.. code-block:: bash
    :substitutions:

    # Download
    git clone |github_url|.git
    cd |github_repo|
    git checkout |github_version_tag|

    # Configure
    mkdir build
    cmake -B build -DCMAKE_BUILD_TYPE=Release

    # Build
    cmake ---build build

    # Install To /usr/local
    sudo cmake --install build


See the |.installation|_ documentation for more details.

Usage
-----

Wavelet Objects
^^^^^^^^^^^^^^^

.. code-block:: cpp

    #include <wtcv/wavelet.hpp>

Builtin wavelets are created by name

.. code-block:: cpp

    wtcv::Wavelet wavelet = wtcv::Wavelet::create("db2");

or by factory

.. code-block:: cpp

    wtcv::Wavelet wavelet = wtcv::create_daubuchies(2);

Accessing the filter banks decomposition and reconstruction kernels

.. code-block:: cpp

    wavelet.filter_bank().decompose_kernels().lowpass()
    wavelet.filter_bank().decompose_kernels().highpass()
    wavelet.filter_bank().reconstruct_kernels().lowpass()
    wavelet.filter_bank().reconstruct_kernels().highpass()


Discrete Wavelet Transform (DWT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    #include <wtcv/dwt2d.hpp>

Performing a discrete wavelet transformation (DWT) of an image is done using a functional style

.. code-block:: cpp

    cv::Mat image = cv::imread(filename);
    DWT2D::Coeffs coeffs = wtcv::dwt2d(image, "db2");

or an object oriented approach

.. code-block:: cpp

    wtcv::Wavelet wavelet = wtcv::Wavelet::create("db2");
    int levels = 2;
    wtcv::DWT2D dwt(wavelet);
    wtcv::DWT2D::Coeffs coeffs = dwt(image, levels);

Reconstruct the image by inverting the DWT

.. code-block:: cpp

    cv::Mat reconstructed_image = coeffs.reconstruct();


Accessing DWT Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^


.. raw:: html

   <table>
       <tr align="left">
           <th>

📝 Note

.. raw:: html

   </th>
   <tr><td>

The horizontal detail coefficients are used for illustration.
There are corresponding accessors for vertical and diagonal detail coefficients.

.. raw:: html

   </td></tr>
   </table>


Access the approximation coefficients

.. code-block:: cpp

    cv::Mat approx_coeffs = coeffs.approx();

Access the finest scale (i.e. highest resolution) horizontal subband coefficients

.. code-block:: cpp

    cv::Mat finest_horizontal_coeffs = coeffs.horizontal_detail(0);
    coeffs.set_horizontal_detail(0, finest_horizontal_coeffs);

Or use the parameterized subband version

.. code-block:: cpp

    cv::Mat finest_horizontal_coeffs = coeffs.detail(wtcv::HORIZONTAL, 0);
    coeffs.set_detail(0, wtcv::HORIZONTAL, finest_horizontal_coeffs);

**Negative Level Indexing**

Use negative level indexing to access the coarsest scale (i.e. lowest resolution) horizontal subband coefficients

.. code-block:: cpp

    // Equivalent to coeffs.horizontal_detail(coeffs.levels() - 1)
    cv::Mat coarsest_horizontal_coeffs = coeffs.horizontal_detail(-1);
    coeffs.set_horizontal_detail(-1, coarsest_horizontal_coeffs);

**Collect Details At Multiple Scales**

Get horizontal detail coefficients at every scale

.. code-block:: cpp

    std::vector<cv::Mat> horizontal_details = coeffs.collect_horizontal_details();

Get detail coefficients at every scale and subband

.. code-block:: cpp

    std::vector<wtcv::DWT2D::Coeffs::DetailTuple>> details = coeffs.details();


Shrink DWT Coefficients
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    #include <wtcv/shrink.hpp>
    wtcv::DWT2D::Coeffs coeffs = ...;

Shrinking DWT coefficients is the basis for many denoising and compression
applications.  There are several shrinking algorithms implemented.  Take the
BayesShrink algorithm as an example

.. code-block:: cpp

    coeffs = wtcv::bayes_shrink(coeffs);

Alternatively, the object oriented API can be used in a polymorphic way

.. code-block:: cpp

    wtcv::Shrinker* shrinker = new wtcv::BayesShrinker();
    coeffs = shrinker->shrink(coeffs);

or as a function object

.. code-block:: cpp

    wtcv::BayesShrinker shrink;
    coeffs = shrink(coeffs);

The functional API is simpler and more succinct, whereas the object oriented API
offers more options to fine tune the algorithm.



Documentation
-------------

Documentation is hosted at `read the docs <https://waveletcv.readthedocs.org>`_.

License
-------


WaveletsCV is free open source software released under the MIT license.


