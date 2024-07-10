.. _installation:

Installation
============
.. cpp:namespace:: cvwt

Overview
---------

.. include:: build_and_install.rst

Source Code
-----------

The latest source code is available from the
`WaveletCV releases <https://github.com/cindolfi/waveletcv/releases>`_
page on Github.

.. code-block:: bash

    wget https://github.com/cindolfi/waveletcv/releases/stable.gz.tar
    tar -axf cvwt.gz.tar
    cd cvwt


Alternatively, clone the
`WaveletCV repository <https://github.com/cindolfi/cvwt.git>`_ and checkout the
``stable`` tag.

.. code-block:: bash

    git clone https://github.com/cindolfi/cvwt.git
    cd cvwt
    git checkout stable


Configure
---------

Build Type
^^^^^^^^^^

For single configuration providers (e.g. make, ninja), the `CMAKE_BUILD_TYPE <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_
variable **must** be set to one of:

    - Debug
    - Release
    - RelWithDebInfo
    - MinSizeRel

Exceptions
^^^^^^^^^^

Exceptions can be **conditionally compiled out** by turning off one or more of the
following configuration variables.

    - ``CVWT_ENABLE_EXCEPTIONS``: Disable all exceptions
    - ``CVWT_ENABLE_DWT2D_EXCEPTIONS``: Disable exceptions thrown by :cpp:class:`DWT2D`
    - ``CVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS``: Disable exceptions thrown by :cpp:class:`DWT2D::Coeffs`
    - ``CVWT_ENABLE_FILTER_BANK_EXCEPTIONS``: Disable exceptions thrown by :cpp:class:`FilterBank`
    - ``CVWT_ENABLE_WAVELET_EXCEPTIONS``: Disable exceptions thrown by :cpp:class:`Wavelet` and related functions


.. note::

    This only applies to exceptions thrown **directly** by WaveletCV.
    Exceptions thrown by OpenCV or any other library are not affected.

For example, to remove argument checking from :cpp:class:`DWT2D::Coeffs` functions use

.. code-block:: bash

    cmake -B build -DCVWT_ENABLE_DWT2D_COEFFS_EXCEPTIONS=OFF

Warnings
^^^^^^^^

Warnings can be **conditionally compiled out** by turning off one or more of the
following configuration variables.

    - ``CVWT_ENABLE_DWT2D_WARNINGS``: Disable warnings logged by :cpp:class:`DWT2D`


In Source Building
^^^^^^^^^^^^^^^^^^

Attempting to set the build directory to the project directory is an error by default.
This ensures that the source directory isn't accidentally littered with build files
if cmake is invoked with a missing build directory.

.. code-block:: bash

    # This fails!
    # Running cmake without a -B option sets both the build directory and the
    # source directory to the current directory.
    cmake

The recommended approach is to build to a separate directory.

.. code-block:: bash

    mkdir build
    cmake -B build

But, if building in place is absolutely necessary set the
``CVWT_ENABLE_BUILD_IN_PLACE`` configuration variable.

.. code-block:: bash

    # This no longer fails, but it will leave a mess inside the source directory.
    cmake -DCVWT_ENABLE_BUILD_IN_PLACE

Build
-----

.. code-block:: bash

    cmake ---build build


.. Examples
.. ^^^^^^^^

.. Build all examples with

.. .. code-block:: bash

..     cmake --build build --target examples

.. or build examples individually with

.. .. code-block:: bash

..     cmake --build build --target cvwt-dwt2d
..     cmake --build build --target cvwt-denoise


Install
-------

WaveletCV is installed to `/usr/local` by default. To install to a specified directory use

.. code-block:: bash

    cmake --install build --install-prefix <install-path>

.. Examples
.. ^^^^^^^^

.. Install all examples with

.. .. code-block:: bash

..     sudo cmake --install build --component examples

.. or install examples individually with

.. .. code-block:: bash

..     sudo cmake --install build --component cvwt-dwt2d
..     sudo cmake --install build --component cvwt-denoise


Uninstall
^^^^^^^^^

To uninstall the library run

.. code-block:: bash

    cd build
    sudo make uninstall


Examples
--------

Build and install all examples with

.. code-block:: bash

    cmake --build build --target examples
    sudo cmake --install build --component examples

or build and install examples individually with

.. code-block:: bash

    # Build and install the cvwt-dwt2d program.
    cmake --build build --target cvwt-dwt2d
    sudo cmake --install build --component cvwt-dwt2d

    # Build and install the cvwt-denoise program.
    cmake --build build --target cvwt-denoise
    sudo cmake --install build --component cvwt-denoise



.. Tests
.. -----

.. .. code-block:: bash

..     # Build
..     cmake --build build --target cvwt_test

..     # Run
..     ctest --test-dir build

.. or

.. .. code-block:: bash

..     # Build
..     ctest --build-and-test . build


.. Documentation
.. -------------

.. .. code-block:: bash

..     # Build
..     cmake --build build --target docs

