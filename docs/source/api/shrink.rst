.. _shrink_api:

Shrinker API
============

.. doxygenclass:: cvwt::Shrinker
   :project: WaveletCV
   :members:
   :protected-members:


.. doxygentypedef:: cvwt::ShrinkFunction
   :project: WaveletCV

.. doxygentypedef:: cvwt::PrimitiveShrinkFunction
   :project: WaveletCV

.. doxygentypedef:: cvwt::StdDevFunction
   :project: WaveletCV


Universal Threshold Shrinker
----------------------------

.. doxygenclass:: cvwt::UniversalShrinker
   :project: WaveletCV
   :members:


VisuShrink Algorithm
--------------------

.. doxygenclass:: cvwt::VisuShrinker
   :project: WaveletCV
   :members:

Functional Interface
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: cvwt::visu_shrink(const DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::visu_shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::visu_shrink(const DWT2D::Coeffs&, int)
   :project: WaveletCV

.. doxygenfunction:: cvwt::visu_shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, int)
   :project: WaveletCV


SureShrink Algorithm
--------------------

.. doxygenclass:: cvwt::SureShrinker
   :project: WaveletCV
   :members:

Functional Interface
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: cvwt::sure_shrink(const DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::sure_shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::sure_shrink(const DWT2D::Coeffs&, int)
   :project: WaveletCV

.. doxygenfunction:: cvwt::sure_shrink(const DWT2D::Coeffs&, DWT2D::Coeffs&, int)
   :project: WaveletCV

.. doxygenfunction:: cvwt::sure_shrink_levelwise(const DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::sure_shrink_levelwise(const DWT2D::Coeffs&, DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::sure_shrink_levelwise(const DWT2D::Coeffs&, int)
   :project: WaveletCV

.. doxygenfunction:: cvwt::sure_shrink_levelwise(const DWT2D::Coeffs&, DWT2D::Coeffs&, int)
   :project: WaveletCV


BayesShrink Algorithm
---------------------

.. doxygenclass:: cvwt::BayesShrinker
   :project: WaveletCV
   :members:


Functional Interface
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: cvwt::bayes_shrink(const DWT2D::Coeffs&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::bayes_shrink(const DWT2D::Coeffs& coeffs, DWT2D::Coeffs& shrunk_coeffs)
   :project: WaveletCV


Low Level API
-------------

.. doxygenfunction:: cvwt::soft_threshold(cv::InputArray, cv::OutputArray, const cv::Scalar&, cv::InputArray)
   :project: WaveletCV

.. doxygenfunction:: cvwt::hard_threshold(cv::InputArray, cv::OutputArray, const cv::Scalar&, cv::InputArray)
   :project: WaveletCV

.. doxygenfunction:: cvwt::make_shrink_function()
   :project: WaveletCV

.. doxygenfunction:: cvwt::make_shrink_function(PrimitiveShrinkFunction<T, W>)
   :project: WaveletCV

.. doxygenfunction:: cvwt::shrink_globally(DWT2D::Coeffs&, const cv::Scalar&, ShrinkFunction, const cv::Range&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::shrink_levels(DWT2D::Coeffs&, cv::InputArray, ShrinkFunction, const cv::Range&)
   :project: WaveletCV

.. doxygenfunction:: cvwt::shrink_subbands(DWT2D::Coeffs&, cv::InputArray, ShrinkFunction, const cv::Range&)
   :project: WaveletCV


