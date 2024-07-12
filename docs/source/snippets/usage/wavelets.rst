
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


.. seealso::

    - :ref:`wavelet`
    - :ref:`wavelet_api`

