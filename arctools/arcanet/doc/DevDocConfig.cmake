
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_devdoc_config_adoc_variables)

  # Remplacer "${CMAKE_SOURCE_DIR}/src" par l'emplacement des sources.
  set(ADOC_DOXYGEN_INPUT ${ADOC_DOXYGEN_INPUT}
    "${DOC_DIR}/doc_dev"
    "${DOC_DIR}/doc_common"
    "${DOC_DIR}/doc_common/chap_news"
  )

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_devdoc_config_doxygen_variables)

  # Remplacer "${CMAKE_SOURCE_DIR}/src" par l'emplacement des sources.
  set(DOXYGEN_STRIP_FROM_PATH ${DOXYGEN_STRIP_FROM_PATH}
  )

  # ----------------------------------------------------------------------------

  # Remplacer "${CMAKE_SOURCE_DIR}/src" par l'emplacement des sources.
  set(DOXYGEN_STRIP_FROM_INC_PATH ${DOXYGEN_STRIP_FROM_INC_PATH}
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

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
