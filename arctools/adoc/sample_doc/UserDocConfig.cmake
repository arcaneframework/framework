# Remplacer "${CMAKE_SOURCE_DIR}/src" par l'emplacement des sources.
set(DOXYGEN_STRIP_FROM_PATH ${DOXYGEN_STRIP_FROM_PATH}
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

# ----------------------------------------------------------------------------

# Remplacer "${CMAKE_SOURCE_DIR}/src" par l'emplacement des sources.
set(DOXYGEN_STRIP_FROM_INC_PATH ${ DOXYGEN_STRIP_FROM_INC_PATH}
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_EXAMPLE_PATH ${DOXYGEN_EXAMPLE_PATH}
  "${CMAKE_BINARY_DIR}/share/axl/dox/snippets"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_EXAMPLE_RECURSIVE
  "YES"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_IMAGE_PATH ${DOXYGEN_IMAGE_PATH}
  "${DOC_DIR}/theme/img"
  "${DOC_DIR}/doc_user/chap_core_types/img"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
  "${DOC_DIR}/doc_user/0_usermanual.md"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_RECURSIVE
  "NO"
)

# ----------------------------------------------------------------------------

# Remplacer "${CMAKE_SOURCE_DIR}/sample" par les emplacements des sources (DOXYGEN_RECURSIVE = NO juste au-dessus).
set(ADOC_DOXYGEN_INPUT ${ADOC_DOXYGEN_INPUT}
  "${CMAKE_SOURCE_DIR}/sample"
  "${DOC_DIR}/doc_common"
  "${DOC_DIR}/doc_common/chap_news"
  "${DOC_DIR}/doc_user"
  "${DOC_DIR}/doc_user/chap_getting_started"
  "${DOC_DIR}/doc_user/chap_services_modules"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

# ----------------------------------------------------------------------------
