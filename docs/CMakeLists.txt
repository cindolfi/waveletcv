#   ----------------------------------------------------------------------------
#   Find or Fetch Dependencies
#   ----------------------------------------------------------------------------
find_package(Doxygen)

if (DOXYGEN_FOUND)
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

    message(STATUS "Doxygen version ${DOXYGEN_VERSION}")

    set(
        DOXYGEN_EXCLUDE_SYMBOLS
        "cvwt::internal"
        "cv"
        "cvv"
        "cvflann"
        "simd512"
        "common_*"
        "CoeffsExpr"
    )
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    # set(DOXYGEN_WARNINGS NO)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_TAGFILES "${OPENCV_DOCS_TAG_FILE}=${OPENCV_DOCS_URL}")
    set(DOXYGEN_HTML_EXTRA_STYLESHEET "docs.css")
    set(DOXYGEN_FORMULA_MACROFILE "${CMAKE_CURRENT_SOURCE_DIR}/formulas.tex")
    # set(DOXYGEN_FORMULA_FONTSIZE 24)
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

    set(DOXYGEN_SHOW_GROUPED_MEMB_INC YES)
    set(DOXYGEN_SORT_MEMBER_DOCS NO)
    # set(DOXYGEN_SHOW_INCLUDE_FILES YES)
    # set(DOXYGEN_TOC_INCLUDE_HEADINGS 5)

    set(DOXYGEN_STRIP_FROM_PATH "${CVWT_INCLUDE_DIR}")
    set(DOXYGEN_STRIP_FROM_INC_PATH "${CVWT_INCLUDE_DIR}")
    set(DOXYGEN_CITE_BIB_FILES "dwt2d.bib")

    set(DOXYGEN_FULL_PATH_NAMES YES)
    set(DOXYGEN_ALIASES
        [[pref{1}="<span class=\"paramref paramname\">\1</span>"]]
        [[pref{2}="<span class=\"paramref paramname\">\1</span>.@ref \2"]]
        [[pref{3}="<span class=\"paramref paramname\">\1</span>.@ref \3 \"\2\""]]
        [[equivalentto="This function is equivalent to:"]]
        [[when_noise_unknown{2}="@note Use this overload when the coefficient noise must be *estimated* from the coefficients.  If the noise is *known* use this @ref \1 \"\2\" instead."]]
        [[when_noise_known{2}="@note Use this overload when the coefficient noise is *known*.  If the noise is must be *estimated* from the coefficients use this @ref \1 \"\2\" instead."]]
    )
    set(DOXYGEN_VERBATIM_VARS DOXYGEN_ALIASES)

    # SHOW_INCLUDE_FILES     = YES
    # SHOW_GROUPED_MEMB_INC  = YES

    set(DOXYGEN_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

    # set(DOXYGEN_MATHJAX_EXTENSIONS ams)

    doxygen_add_docs(
        docs
        ${CVWT_INCLUDE_DIR}/cvwt
    )
else()
    message(STATUS "Could not find Doxygen")
endif()

