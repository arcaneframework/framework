set(DOXYGEN_STRIP_FROM_PATH ${DOXYGEN_STRIP_FROM_PATH}
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_STRIP_FROM_INC_PATH ${DOXYGEN_STRIP_FROM_INC_PATH}
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

# ----------------------------------------------------------------------------

set(DOXYGEN_IMAGE_PATH ${DOXYGEN_IMAGE_PATH}
  "${DOC_DIR}/theme/img"
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

# Remplacer "${CMAKE_SOURCE_DIR}/sample" par l'emplacement des sources.
set(ADOC_DOXYGEN_INPUT ${ADOC_DOXYGEN_INPUT}
  "${DOC_DIR}/doc_dev"
  "${DOC_DIR}/doc_common"
  "${DOC_DIR}/doc_common/chap_news"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
  "${CMAKE_SOURCE_DIR}/sample"
)

# ----------------------------------------------------------------------------
