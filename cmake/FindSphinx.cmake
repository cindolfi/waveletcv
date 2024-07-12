#   Look for an executable called sphinx-build
find_program(
    SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Path to sphinx-build executable"
)

include(FindPackageHandleStandardArgs)

#   Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(
    Sphinx
    "Failed to find sphinx-build executable"
    SPHINX_EXECUTABLE
)

#   Get the version
execute_process(
    COMMAND ${SPHINX_EXECUTABLE} --version
    OUTPUT_VARIABLE SPHINX_VERSION
)
string(REGEX MATCH "[0-9.]+" SPHINX_VERSION "${SPHINX_VERSION}")
