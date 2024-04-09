#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
find_package(Doxygen)

if (DOXYGEN_FOUND)
    message(STATUS "Doxygen version ${DOXYGEN_VERSION}")

    set(DOXYGEN_EXCLUDE_SYMBOLS "cvwt::internal")
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    # set(DOXYGEN_WARNINGS NO)
    set(DOXYGEN_TAGFILES "doxygen-tags/opencv.tag=http://docs.opencv.org/4.9.0")
    set(DOXYGEN_HTML_EXTRA_STYLESHEET "docs.css")

    doxygen_add_docs(
        docs
        ALL
        ${CVWT_INCLUDE_DIR}/cvwt
    )
else()
    message(STATUS "Could not find Doxygen")
endif()

