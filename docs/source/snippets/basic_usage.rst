
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


.. seealso::

    - :ref:`wavelet`
    - :ref:`wavelet_api`

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

.. note::

    The horizontal detail coefficients are used for illustration.
    There are corresponding accessors for vertical and diagonal detail coefficients.

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

.. rubric:: Negative Level Indexing

Use negative level indexing to access the coarsest scale (i.e. lowest resolution) horizontal subband coefficients

.. code-block:: cpp

    // Equivalent to coeffs.horizontal_detail(coeffs.levels() - 1)
    cv::Mat coarsest_horizontal_coeffs = coeffs.horizontal_detail(-1);
    coeffs.set_horizontal_detail(-1, coarsest_horizontal_coeffs);

.. rubric:: Collect Details At Multiple Scales

Get horizontal detail coefficients at every scale

.. code-block:: cpp

    std::vector<cv::Mat> horizontal_details = coeffs.collect_horizontal_details();

Get detail coefficients at every scale and subband

.. code-block:: cpp

    std::vector<cvwt::DWT2D::Coeffs::DetailTuple>> details = coeffs.details();

.. seealso::

    - :ref:`dwt`
    - :ref:`dwt_api`
    - `cvwt-dwt2d <https://github.com/cindolfi/waveletcv/examples/dwt2d.cpp>`_ for a complete example


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


.. seealso::

    - :ref:`shrink`
    - :ref:`shrink_api`
    - `cvwt-denoise <https://github.com/cindolfi/waveletcv/examples/denoise.cpp>`_ for a complete example
