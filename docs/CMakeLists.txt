#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
find_package(Doxygen)
find_package(Sphinx)

if (DOXYGEN_FOUND)
    message(STATUS "Found Doxygen ${DOXYGEN_VERSION}")
else()
    message(STATUS "Doxygen not found")
endif()
if (SPHINX_FOUND)
    message(STATUS "Found Sphinx ${SPHINX_VERSION}")
else()
    message(STATUS "Sphinx not found")
endif()

if (DOXYGEN_FOUND AND SPHINX_FOUND)
    #   Get references to external OpenCV docs
    set(OPENCV_DOCS_URL "https://docs.opencv.org/${OpenCV_VERSION}")
    set(OPENCV_DOCS_TAG_FILE "${CMAKE_CURRENT_BINARY_DIR}/doxygen-tags/${OpenCV_VERSION}/opencv.tag")
    if(NOT EXISTS "${OPENCV_DOCS_TAG_FILE}")
        message(CHECK_START "Downloading ${OPENCV_DOCS_URL}/opencv.tag")
        file(
            DOWNLOAD
            "${OPENCV_DOCS_URL}/opencv.tag"
            "${OPENCV_DOCS_TAG_FILE}"
            STATUS OPENCV_DOCS_TAG_DOWNLOADED
        )
        if (OPENCV_DOCS_TAG_DOWNLOADED)
            message(CHECK_PASS "Done")
        else()
            message(CHECK_FAIL "Failed")
        endif()
    endif()

    #   ------------------------------------------------------------------------
    #   Doxygen Configuration
    #   ------------------------------------------------------------------------
    #   Symbols
    set(DOXYGEN_EXTRACT_ALL YES)
    set(
        DOXYGEN_EXCLUDE_SYMBOLS
        "wtcv::internal"
        "cv"
        "cvv"
        "cvflann"
        "simd512"
        "common_*"
        "CoeffsExpr"
    )

    #   General
    set(DOXYGEN_USE_MATHJAX YES)

    #   Warnings
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    # set(DOXYGEN_WARNINGS NO)

    #   External Libraries
    set(DOXYGEN_TAGFILES "${OPENCV_DOCS_TAG_FILE}=${OPENCV_DOCS_URL}")
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

    #   Extra Files
    set(DOXYGEN_FORMULA_MACROFILE "${CMAKE_CURRENT_SOURCE_DIR}/formulas.tex")
    set(DOXYGEN_HTML_EXTRA_STYLESHEET "docs.css")
    set(DOXYGEN_CITE_BIB_FILES "dwt2d.bib")
    set(DOXYGEN_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

    #   Grouping & Sorting
    set(DOXYGEN_SHOW_GROUPED_MEMB_INC YES)
    set(DOXYGEN_SORT_MEMBER_DOCS NO)

    #   Path Names
    set(DOXYGEN_STRIP_FROM_PATH "${CVWT_INCLUDE_DIR}")
    set(DOXYGEN_STRIP_FROM_INC_PATH "${CVWT_INCLUDE_DIR}")
    set(DOXYGEN_FULL_PATH_NAMES YES)

    #   Aliases
    set(DOXYGEN_ALIASES
    [[rst=^^@verbatim embed:rst:leading-asterisk^^"]]
    [[endrst=@endverbatim"]]
    [[pref{1}="<span class=\"paramref paramname\">\1</span>"]]
    [[pref{2}="<span class=\"paramref paramname\">\1</span>.@ref \2"]]
    [[pref{3}="<span class=\"paramref paramname\">\1</span>.@ref \3 \"\2\""]]
    [[equivalentto="This function is equivalent to:"]]
    [[when_noise_unknown{2}="@note Use this overload when the coefficient noise must be *estimated* from the coefficients.  If the noise is *known* use this @ref \1 \"\2\" instead."]]
    [[when_noise_known{2}="@note Use this overload when the coefficient noise is *known*.  If the noise is must be *estimated* from the coefficients use this @ref \1 \"\2\" instead."]]
    )
    set(DOXYGEN_VERBATIM_VARS DOXYGEN_ALIASES)

    #   Output
    set(DOXYGEN_GENERATE_XML YES)

    #   ------------------------------------------------------------------------
    #   Sphinx
    #   ------------------------------------------------------------------------
    set(CVWT_SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
    if (WTCV_DOCS_OUTPUT)
        set(CVWT_SPHINX_BUILD ${WTCV_DOCS_OUTPUT})
    else()
        set(CVWT_SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
    endif()

    set(
        SPHINK_COMMAND
        ${SPHINX_EXECUTABLE}
        -b html
        -Dbreathe_projects.WaveletCV="${CMAKE_CURRENT_BINARY_DIR}/xml"
        -Dproject="${CMAKE_PROJECT_NAME}"
        -Dversion="${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}"
        -Drelease="${CMAKE_PROJECT_VERSION}"
        -Dhtml_context.cmake_version="${CMAKE_MINIMUM_REQUIRED_VERSION}"
        -Dhtml_context.github_version="${CMAKE_PROJECT_VERSION}"
        \"${CVWT_SPHINX_SOURCE}\"
        \"${CVWT_SPHINX_BUILD}\"
    )

    message(VERBOSE "CVWT_SPHINX_SOURCE ${CVWT_SPHINX_SOURCE}")
    message(VERBOSE "CVWT_SPHINX_BUILD ${CVWT_SPHINX_BUILD}")

    #   ------------------------------------------------------------------------
    #   Add docs target
    #   ------------------------------------------------------------------------
    doxygen_add_docs(
        docs
        ${CVWT_INCLUDE_DIR}/wtcv
    )

    #   Run sphinx-build after generating doxygen xml
    add_custom_command(
        TARGET docs
        POST_BUILD
        COMMAND ${SPHINK_COMMAND}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating documentation with Sphinx"
    )
elseif(WTCV_BUILD_DOCS_ONLY)
    message(FATAL_ERROR "Failed to add docs target")
else()
    message(WARNING "Failed to add docs target")
endif()
