if (NOT DEFINED DOXYGEN_PROJECT_BRIEF)
  if (${ADOC_LANGUAGE} STREQUAL "French")
    set(DOXYGEN_PROJECT_BRIEF
      "Documentation utilisateur"
    )
  else ()
    set(DOXYGEN_PROJECT_BRIEF
      "User documentation"
    )
  endif ()
endif ()

# ----------------------------------------------------------------------------

set(DOXYGEN_HTML_EXTRA_STYLESHEET ${DOXYGEN_HTML_EXTRA_STYLESHEET}
  "${ADOC_SOURCE_DIR}/theme/css/user_colors.css"
)

# ----------------------------------------------------------------------------

if (NOT DEFINED DOXYGEN_GENERATE_TODOLIST)
  set(DOXYGEN_GENERATE_TODOLIST
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DEFINED DOXYGEN_STRIP_CODE_COMMENTS)
  set(DOXYGEN_STRIP_CODE_COMMENTS
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DEFINED DOXYGEN_CLASS_GRAPH)
  set(DOXYGEN_CLASS_GRAPH
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DEFINED DOXYGEN_COLLABORATION_GRAPH)
  set(DOXYGEN_COLLABORATION_GRAPH
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DEFINED DOXYGEN_GROUP_GRAPHS)
  set(DOXYGEN_GROUP_GRAPHS
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------
