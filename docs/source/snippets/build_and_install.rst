WaveletCV is built and installed using the
`cmake <https://cmake.org/cmake/help/latest/manual/cmake.1.html>`_
build system (version 3.24.0 or newer).

.. code-block:: bash

    # Download
    git clone https://github.com/cindolfi/wtcv.git
    cd wtcv
    git checkout stable

    # Configure
    mkdir build
    cmake -B build -DCMAKE_BUILD_TYPE=Release

    # Build
    cmake ---build build

    # Install To /usr/local
    sudo cmake --install build