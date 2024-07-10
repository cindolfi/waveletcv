.. |.dwt| replace:: Discrete Wavelet Transform (DWT)
.. _.dwt: https://wavletcv.readthedocs.io/en/latest/dwt2d.html#dwt
.. |.dwt_api| replace:: DWT API
.. _.dwt_api: https://wavletcv.readthedocs.io/en/latest/api/dwt.html#dwt-api
.. |.installation| replace:: Installation
.. _.installation: https://wavletcv.readthedocs.io/en/latest/installation.html#installation
.. |.shrink| replace:: Shrinking DWT Coefficients
.. _.shrink: https://wavletcv.readthedocs.io/en/latest/shrink.html#shrink
.. |.shrink_api| replace:: Shrinker API
.. _.shrink_api: https://wavletcv.readthedocs.io/en/latest/api/shrink.html#shrink-api
.. |.wavelet| replace:: Wavelets
.. _.wavelet: https://wavletcv.readthedocs.io/en/latest/wavelet.html#wavelet
.. |.wavelet_api| replace:: Wavelet API
.. _.wavelet_api: https://wavletcv.readthedocs.io/en/latest/api/wavelet.html#wavelet-api



.. [![Latest Docs](https://readthedocs.org/projects/nlopt/badge/?version=latest)](http://nlopt.readthedocs.io/en/latest/)
.. [![Build Status](https://github.com/stevengj/nlopt/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/stevengj/nlopt/actions/workflows/build.yml)

|Build Status|

.. |Build Status| image:: https://github.com/stevengj/nlopt/actions/workflows/build.yml/badge.svg?branch=master
   :target: https://github.com/stevengj/nlopt/actions/workflows/build.yml

|GitHub release|

.. |GitHub release| image:: https://img.shields.io/github/release/Naereen/StrapDown.js.svg
   :target: https://GitHub.com/Naereen/StrapDown.js/releases/

|Documentation Status|

.. |Documentation Status| image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
   :target: http://ansicolortags.readthedocs.io/?badge=latest

|MIT license|

.. |MIT license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://lbesson.mit-license.org/


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

WaveletCV is built and installed using the
`cmake <https://cmake.org/cmake/help/latest/manual/cmake.1.html>`_
build system (version 3.24.0 or newer).

.. code-block:: bash

    # Download
    git clone https://github.com/cindolfi/cvwt.git
    cd cvwt
    git checkout stable

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

    #include <cvwt/wavelet.hpp>

Builtin wavelets are created by name

.. code-block:: cpp

    cvwt::Wavelet wavelet = cvwt::Wavelet::create("db2");

or by factory

.. code-block:: cpp

    cvwt::Wavelet wavelet = cvwt::create_daubuchies(2);

Accessing the filter banks decomposition and reconstruction kernels

.. code-block:: cpp

    wavelet.filter_bank().decompose_kernels().lowpass()
    wavelet.filter_bank().decompose_kernels().highpass()
    wavelet.filter_bank().reconstruct_kernels().lowpass()
    wavelet.filter_bank().reconstruct_kernels().highpass()



.. raw:: html

   <table>
       <tr align="left">
           <th>

📄 Seealso

.. raw:: html

   </th>
   <tr><td>

- |.wavelet|_
- |.wavelet_api|_

.. raw:: html

   </td></tr>
   </table>


Discrete Wavelet Transform (DWT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    #include <cvwt/dwt2d.hpp>

Performing a discrete wavelet transformation (DWT) of an image is done using a functional style

.. code-block:: cpp

    cv::Mat image = cv::imread(filename);
    DWT2D::Coeffs coeffs = cvwt::dwt2d(image, "db2");

or an object oriented approach

.. code-block:: cpp

    cvwt::Wavelet wavelet = cvwt::Wavelet::create("db2");
    int levels = 2;
    cvwt::DWT2D dwt(wavelet);
    cvwt::DWT2D::Coeffs coeffs = dwt(image, levels);

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

    cv::Mat finest_horizontal_coeffs = coeffs.detail(cvwt::HORIZONTAL, 0);
    coeffs.set_detail(0, cvwt::HORIZONTAL, finest_horizontal_coeffs);

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

    std::vector<cvwt::DWT2D::Coeffs::DetailTuple>> details = coeffs.details();


.. raw:: html

   <table>
       <tr align="left">
           <th>

📄 Seealso

.. raw:: html

   </th>
   <tr><td>

- |.dwt|_
- |.dwt_api|_
- `cvwt-dwt2d <https://github.com/cindolfi/waveletcv/examples/dwt2d.cpp>`_ for a complete example

.. raw:: html

   </td></tr>
   </table>



Shrink DWT Coefficients
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    #include <cvwt/shrink.hpp>
    cvwt::DWT2D::Coeffs coeffs = ...;

Shrinking DWT coefficients is the basis for many denoising and compression
applications.  There are several shrinking algorithms implemented.  Take the
BayesShrink algorithm as an example

.. code-block:: cpp

    coeffs = cvwt::bayes_shrink(coeffs);

Alternatively, the object oriented API can be used in a polymorphic way

.. code-block:: cpp

    cvwt::Shrinker* shrinker = new cvwt::BayesShrinker();
    coeffs = shrinker->shrink(coeffs);

or as a function object

.. code-block:: cpp

    cvwt::BayesShrinker shrink;
    coeffs = shrink(coeffs);

The functional API is simpler and more succinct, whereas the object oriented API
offers more options to fine tune the algorithm.



.. raw:: html

   <table>
       <tr align="left">
           <th>

📄 Seealso

.. raw:: html

   </th>
   <tr><td>

- |.shrink|_
- |.shrink_api|_
- `cvwt-denoise <https://github.com/cindolfi/waveletcv/examples/denoise.cpp>`_ for a complete example

.. raw:: html

   </td></tr>
   </table>



Documentation
-------------

Documentation is hosted at `http://waveletcv.readthedocs.org`_.

License
-------


WaveletsCV is free open source software released under the MIT license.


