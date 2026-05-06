# Remplacer "${CMAKE_SOURCE_DIR}/src" par l'emplacement des sources.
set(DOXYGEN_STRIP_FROM_PATH ${DOXYGEN_STRIP_FROM_PATH}
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

# ----------------------------------------------------------------------------

# Remplacer "${CMAKE_SOURCE_DIR}/src" par l'emplacement des sources.
set(DOXYGEN_STRIP_FROM_INC_PATH ${DOXYGEN_STRIP_FROM_INC_PATH}
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_IMAGE_PATH ${DOXYGEN_IMAGE_PATH}
  "${ADOC_DOC_CONFIG_DIR}/theme/img"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
  "${DOC_DIR}/doc_dev/0_devmanual.md"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_RECURSIVE
  "YES"
)

# ----------------------------------------------------------------------------

# Remplacer "${CMAKE_SOURCE_DIR}/src" par l'emplacement des sources.
set(ADOC_DOXYGEN_INPUT ${ADOC_DOXYGEN_INPUT}
  "${CMAKE_SOURCE_DIR}/src"
  "${DOC_DIR}/doc_dev"
  "${DOC_DIR}/doc_common"
  "${DOC_DIR}/doc_common/chap_news"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

# ----------------------------------------------------------------------------
