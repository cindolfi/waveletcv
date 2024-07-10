.. _shrink_api:

Shrinker API
============

.. doxygenclass:: wtcv::Shrinker
   :project: WaveletCV
   :members:
   :protected-members:


.. doxygentypedef:: wtcv::ShrinkFunction
   :project: WaveletCV

.. doxygentypedef:: wtcv::PrimitiveShrinkFunction
   :project: WaveletCV

.. doxygentypedef:: wtcv::StdDevFunction
   :project: WaveletCV


Universal Threshold Shrinker
----------------------------

.. doxygenclass:: wtcv::UniversalShrinker
   :project: WaveletCV
   :members:


VisuShrink Algorithm
--------------------

.. doxygenclass:: wtcv::VisuShrinker
   :project: WaveletCV
   :members:

Functional Interface
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: wtcv::visu_shrink(const DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::visu_shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::visu_shrink(const DWT2D::Coeffs&, int)
   :project: WaveletCV

.. doxygenfunction:: wtcv::visu_shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, int)
   :project: WaveletCV


SureShrink Algorithm
--------------------

.. doxygenclass:: wtcv::SureShrinker
   :project: WaveletCV
   :members:

Functional Interface
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: wtcv::sure_shrink(const DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::sure_shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::sure_shrink(const DWT2D::Coeffs&, int)
   :project: WaveletCV

.. doxygenfunction:: wtcv::sure_shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, int)
   :project: WaveletCV

.. doxygenfunction:: wtcv::sure_shrink_levelwise(const DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::sure_shrink_levelwise(const DWT2D::Coeffs&, DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::sure_shrink_levelwise(const DWT2D::Coeffs&, int)
   :project: WaveletCV

.. doxygenfunction:: wtcv::sure_shrink_levelwise(const DWT2D::Coeffs&, DWT2D::Coeffs&, int)
   :project: WaveletCV


BayesShrink Algorithm
---------------------

.. doxygenclass:: wtcv::BayesShrinker
   :project: WaveletCV
   :members:


Functional Interface
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: wtcv::bayes_shrink(const DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::bayes_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
   :project: WaveletCV


Low Level API
-------------

.. doxygenfunction:: wtcv::soft_threshold(cv::InputArray, cv::OutputArray, const cv::Scalar&, cv::InputArray)
   :project: WaveletCV

.. doxygenfunction:: wtcv::hard_threshold(cv::InputArray, cv::OutputArray, const cv::Scalar&, cv::InputArray)
   :project: WaveletCV

.. doxygenfunction:: wtcv::make_shrink_function()
   :project: WaveletCV

.. doxygenfunction:: wtcv::make_shrink_function(PrimitiveShrinkFunction<T, W>)
   :project: WaveletCV

.. doxygenfunction:: wtcv::shrink_globally(DWT2D::Coeffs&, const cv::Scalar&, ShrinkFunction, const cv::Range&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::shrink_levels(DWT2D::Coeffs&, cv::InputArray, ShrinkFunction, const cv::Range&)
   :project: WaveletCV

.. doxygenfunction:: wtcv::shrink_subbands(DWT2D::Coeffs&, cv::InputArray, ShrinkFunction, const cv::Range&)
   :project: WaveletCV


