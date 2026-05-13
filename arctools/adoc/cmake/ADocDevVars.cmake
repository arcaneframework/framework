if (NOT DOXYGEN_PROJECT_BRIEF)
  if (${ADOC_LANGUAGE} STREQUAL "French")
    set(DOXYGEN_PROJECT_BRIEF
      "Documentation développeur"
    )
  else ()
    set(DOXYGEN_PROJECT_BRIEF
      "Developer documentation"
    )
  endif ()
endif ()

# ----------------------------------------------------------------------------

set(DOXYGEN_HTML_EXTRA_STYLESHEET ${DOXYGEN_HTML_EXTRA_STYLESHEET}
  "${ADOC_SOURCE_DIR}/theme/css/dev_colors.css"
)

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_EXTRACT_PRIVATE)
  set(DOXYGEN_EXTRACT_PRIVATE
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_INTERNAL_DOCS)
  set(DOXYGEN_INTERNAL_DOCS
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_CALL_GRAPH)
  set(DOXYGEN_CALL_GRAPH
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_CALLER_GRAPH)
  set(DOXYGEN_CALLER_GRAPH
    "YES"
  )
endif ()

# ----------------------------------------------------------------------------
