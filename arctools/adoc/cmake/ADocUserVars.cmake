if (NOT DOXYGEN_PROJECT_BRIEF)
  set(DOXYGEN_PROJECT_BRIEF
    "Documentation utilisateur"
  )
endif ()

# ----------------------------------------------------------------------------

set(DOXYGEN_HTML_EXTRA_STYLESHEET ${DOXYGEN_HTML_EXTRA_STYLESHEET}
  "${ADOC_SOURCE_DIR}/theme/css/user_colors.css"
)

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_GENERATE_TODOLIST)
  set(DOXYGEN_GENERATE_TODOLIST
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_STRIP_CODE_COMMENTS)
  set(DOXYGEN_STRIP_CODE_COMMENTS
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_CLASS_GRAPH)
  set(DOXYGEN_CLASS_GRAPH
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_COLLABORATION_GRAPH)
  set(DOXYGEN_COLLABORATION_GRAPH
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------

if (NOT DOXYGEN_GROUP_GRAPHS)
  set(DOXYGEN_GROUP_GRAPHS
    "NO"
  )
endif ()

# ----------------------------------------------------------------------------
