.. _wavelet_api:

Wavelet API
===========

.. doxygenclass:: cvwt::Wavelet
    :project: WaveletCV
    :members:

.. doxygenclass:: cvwt::FilterBank
    :project: WaveletCV
    :members:

.. doxygenclass:: cvwt::KernelPair
    :project: WaveletCV
    :members:

.. doxygenenum:: cvwt::DetailSubband
    :project: WaveletCV

.. doxygenfunction:: std::to_string(cvwt::DetailSubband)
    :project: WaveletCV

.. doxygenenum:: cvwt::Symmetry
    :project: WaveletCV

.. doxygenenum:: cvwt::Orthogonality
    :project: WaveletCV

Wavelet Factories
-----------------

.. doxygenfunction:: cvwt::create_haar
    :project: WaveletCV

.. doxygenfunction:: cvwt::create_daubechies
    :project: WaveletCV

.. doxygenfunction:: cvwt::create_symlets
    :project: WaveletCV

.. doxygenfunction:: cvwt::create_coiflets
    :project: WaveletCV

.. doxygenfunction:: cvwt::create_biorthogonal
    :project: WaveletCV

.. doxygenfunction:: cvwt::create_reverse_biorthogonal
    :project: WaveletCV



