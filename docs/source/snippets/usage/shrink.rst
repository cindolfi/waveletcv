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
