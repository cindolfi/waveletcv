

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

    cv::Mat finest_horizontal_coeffs = coeffs.detail(wtcv::HORIZONTAL, 0);
    coeffs.set_detail(0, wtcv::HORIZONTAL, finest_horizontal_coeffs);

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

    std::vector<wtcv::DWT2D::Coeffs::DetailTuple>> details = coeffs.details();

.. seealso::

    - :ref:`dwt`
    - :ref:`dwt_api`
    - `wtcv-dwt2d <https://github.com/cindolfi/waveletcv/examples/dwt2d.cpp>`_ for a complete example

