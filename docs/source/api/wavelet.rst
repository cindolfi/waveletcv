.. _wavelet_api:

Wavelet API
===========

.. doxygenclass:: wtcv::Wavelet
    :project: WaveletCV
    :members:

.. doxygenclass:: wtcv::FilterBank
    :project: WaveletCV
    :members:

.. doxygenclass:: wtcv::KernelPair
    :project: WaveletCV
    :members:

.. doxygenenum:: wtcv::DetailSubband
    :project: WaveletCV

.. doxygenfunction:: std::to_string(wtcv::DetailSubband)
    :project: WaveletCV

.. doxygenenum:: wtcv::Symmetry
    :project: WaveletCV

.. doxygenenum:: wtcv::Orthogonality
    :project: WaveletCV

Wavelet Factories
-----------------

.. doxygenfunction:: wtcv::create_haar
    :project: WaveletCV

.. doxygenfunction:: wtcv::create_daubechies
    :project: WaveletCV

.. doxygenfunction:: wtcv::create_symlets
    :project: WaveletCV

.. doxygenfunction:: wtcv::create_coiflets
    :project: WaveletCV

.. doxygenfunction:: wtcv::create_biorthogonal
    :project: WaveletCV

.. doxygenfunction:: wtcv::create_reverse_biorthogonal
    :project: WaveletCV



