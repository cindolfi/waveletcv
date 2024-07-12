WaveletCV is built and installed using
`cmake <https://cmake.org/cmake/help/latest/manual/cmake.1.html>`_
|cmake_version| or newer.

.. code-block:: bash
    :substitutions:

    # Download
    git clone |github_url|.git
    cd |github_repo|
    git checkout |github_version_tag|

    # Configure
    mkdir build
    cmake -B build -DCMAKE_BUILD_TYPE=Release

    # Build
    cmake ---build build

    # Install To /usr/local
    sudo cmake --install build
