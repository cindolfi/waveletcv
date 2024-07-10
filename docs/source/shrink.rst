.. _shrink:

Shrinking DWT Coefficients
==========================
.. cpp:namespace:: wtcv
.. cpp:namespace-push:: Shrinker

Many DWT based denoising and compression algorithms rely on shrinking
(i.e. thresholding) DWT coefficients.  This is accomplished by transforming
each coefficient with a threshold parameterized :cpp:type:`shrinking function<ShrinkFunction>`.
Two of the most common methods are :cpp:func:`soft thresholding<soft_threshold>`
and :cpp:func:`hard thresholding<hard_threshold>`.
The coefficients are partitioned into subsets where each subset uses a different
threshold. The partitioning scheme is typically global, per decomposition level, or
per subband. Various algorithms differ in

    - How the threshold is calculated
    - Which shrinking (i.e. thresholding) function is applied
    - How the coefficients are partitioned

The :cpp:class:`Shrinker` class is the base class for all shrinking algorithms.
WaveletCV has implementations of:

    - :cpp:class:`BayesShrink<BayesShrinker>`
    - :cpp:class:`SUREShrink<SureShrinker>`
    - :cpp:class:`VisuShrink<VisuShrinker>`
    - :cpp:class:`Universal Threshold Shrinking<UniversalShrinker>`

The details of these algorithms can be found in
`Denoising Through Wavelet Shrinkage: An Empircal Study <https://computing.llnl.gov/sites/default/files/jei2001.pdf>`_.

Shrinking Coefficients
----------------------

To shrink all detail coefficients:

.. code-block:: cpp

    #include <wtcv/shrink.hpp>
    #include <wtcv/dwt2d.hpp>
    using namespace wtcv;

    DWT2D::Coeffs coeffs = ...;
    Shrinker* shrinker = new ...;

    DWT2D::Coeffs shrunken_coeffs = shrinker->shrink(coeffs);

To shrink only the highest resolution detail coefficients:

.. code-block:: cpp

    shrunken_coeffs = shrinker->shrink(coeffs, 1);

To shrink only the first two decomposition levels:

.. code-block:: cpp

    shrunken_coeffs = shrinker->shrink(coeffs, 2);

To shrink all but the highest resolution detail coefficients:

.. code-block:: cpp

    shrunken_coeffs = shrinker->shrink(coeffs, cv::Range(1, coeffs.levels()));

Shrinker objects are also functors:

.. code-block:: cpp

    BayesShrink shrink;
    shrunken_coeffs = shrink(coeffs);


Working With Thresholds
-----------------------

To compute the thresholds:

.. code-block:: cpp

    cv::Mat4d thresholds = shrinker->compute_thresholds(coeffs);

To shrink detail coefficients and get the thresholds in a single call:

.. code-block:: cpp

    cv::Mat4d thresholds;
    shrunken_coeffs = shrinker->shrink(coeffs, thresholds);

To get an array containing the threshold for each corresponding coefficient:

.. code-block:: cpp

    cv::Mat coeff_thresholds = shrinker->expand_thresholds(coeffs, thresholds);
    assert(coeff_thresholds.size() == coeffs.size());
    assert(coeff_thresholds.channels() == coeffs.channels());


To compute a mask that indicates which coefficients were shrunk to zero:

.. code-block:: cpp

    #include <wtcv/array.hpp>

    cv::Mat shrunk_coeffs_mask;
    less_than_or_equal(
        cv::abs(coeffs),
        coeffs_thresholds,
        shrunk_coeffs_mask,
        coeffs.detail_mask()
    );

Note the use of :cpp:func:`coeffs.detail_mask()<DWT2D::Coeffs::detail_mask>`
above.  This is necassary for two reasons.  First, approximation coefficients
are never shrunk so they should be excluded from the operation.  Second, some of
the elements in the coefficient matrix may be placeholders that are always set
to zero (i.e. they are not actual coefficients computed by the DWT).  These
locations should also be ignored.


Thresholds Format
^^^^^^^^^^^^^^^^^

TODO


Noise Estimation
----------------

Many algorithms assume that the coefficients are additively corrupted by i.i.d.
noise drawn from a distribution with zero mean and some fixed, and generally
unknown, variance. For algorithms that require the noise variance to compute the
thresholds, an estimate of the noise standard deviation is provided by
:cpp:func:`compute_noise_stdev`.

The default implementation of :cpp:func:`compute_noise_stdev` calls
:cpp:func:`stdev_function` on the diagonal subband at the finest resolution
(i.e. :cpp:func:`coeffs.diagonal_detail(0)<DWT2D::Coeffs::diagonal_detail>`).
The default :cpp:func:`stdev_function` is :cpp:func:`mad_stdev`, which gives a
statistically robust estimate of the standard deviation.

.. note::

    In the unlikely situation that the coefficient noise variance is known
    (e.g. from a knowlegdge about image acquisition) use the versions of
    :cpp:func:`shrink` and :cpp:func:`compute_thresholds` that
    take a ``stdev`` argument.
    In most cases the noise variance must be estimated from the
    coefficients and the versions of :cpp:func:`shrink` and
    :cpp:func:`compute_thresholds` that do not accept a ``stdev``
    argument should be used, in which case it is estimated internally by
    :cpp:func:`compute_noise_stdev`.


Implementing Shrinking Algorithms
---------------------------------

Implement a shrinking algorithm by subclassing :cpp:class:`Shrinker`.

Algorithms that use a **single threshold** must implement:

    - A constructor that passes, or allows the user to pass, :cpp:enumerator:`Shrinker::GLOBALLY<Shrinker::Partition::GLOBALLY>` to :cpp:class:`Shrinker`
    - :cpp:func:`compute_global_threshold`

Algorithms that use a **separate threshold for each level** must implement:

    - A constructor that passes, or allows the user to pass, :cpp:enumerator:`Shrinker::LEVELS<Shrinker::Partition::LEVELS>` to :cpp:class:`Shrinker`
    - :cpp:func:`compute_level_threshold`

Algorithms that use a **separate threshold for each subband** must implement:

    - A constructor that passes, or allows the user to pass, :cpp:enumerator:`Shrinker::SUBBAND<Shrinker::Partition::SUBBAND>` to :cpp:class:`Shrinker`
    - :cpp:func:`compute_subband_threshold`

Algorithms that use a **custom partitioning scheme** different to those listed above must implement:

    - A constructor that passes, or allows the user to pass, :cpp:enumerator:`Shrinker::SUBSETS<Shrinker::Partition::SUBSETS>` to :cpp:class:`Shrinker`

    - :cpp:func:`compute_subset_thresholds`
    - :cpp:func:`expand_subset_thresholds`
    - :cpp:func:`shrink_subsets`

.. note::

    Algorithms can support multiple partitioning schemes.

Customizing Noise Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Subclasses can override :cpp:func:`compute_noise_stdev` to change which
coefficients are used to esitmate the coefficient noise standard deviation.
Subclasses should change the standard deviation estimator (i.e.
:cpp:func:`stdev_function`) by passing a different estimator to the
:cpp:class:`Shrinker` constructor.

For performance reasons, algorithms that do not require an estimate of the
noise variance should override :cpp:func:`compute_noise_stdev` to do nothing.


Temporaries
^^^^^^^^^^^

TODO

Low Level API
-------------

TODO

Custom Shrink Functions
^^^^^^^^^^^^^^^^^^^^^^^

TODO

.. :cpp:namespace-pop::

.. admonition:: API Reference

    :ref:`shrink_api`

