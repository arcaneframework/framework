
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_devdoc_config_adoc_variables)

  set(ADOC_DOXYGEN_INPUT ${ADOC_DOXYGEN_INPUT}
    ${ARCANESRCROOT}/doc/doc_dev
    ${ARCANESRCROOT}/doc/doc_common
    ${ARCANESRCROOT}/doc/doc_common/chap_news
    ${ARCANESRCROOT}/doc/doc_common/chap_build_install
    ${ARCANESRCROOT}/doc/doc_common/chap_build_install/subchap_prerequisites
    ${ARCANE_SRC_PATH}/arcane
    ${ARCANE_ADDITIONAL_SUBDIR_DOC}
    ${CMAKE_BINARY_DIR}/arcane
    ${Arccore_ROOT}/src
    ${ARCANE_CEA_SOURCE_PATH}/src/arcane
    ${CMAKE_BINARY_DIR}/share/axl/dox
  )

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_devdoc_config_doxygen_variables)

  set(DOXYGEN_STRIP_FROM_PATH ${DOXYGEN_STRIP_FROM_PATH}
    "${Arcane_SOURCE_DIR}/src"
    "${ARCANE_CEA_SOURCE_PATH}/src"
    "${CMAKE_BINARY_DIR}/share/axl/dox"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_STRIP_FROM_INC_PATH ${DOXYGEN_STRIP_FROM_INC_PATH}
    "${Arcane_SOURCE_DIR}/src"
    "${ARCANE_CEA_SOURCE_PATH}/src"
    "${CMAKE_BINARY_DIR}/share/axl/dox"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_EXCLUDE ${DOXYGEN_EXCLUDE}
    "${CMAKE_BINARY_DIR}/arcane/tests"
    "${ARCANE_SRC_PATH}/arcane/tests"
    "${ARCANE_CEA_SOURCE_PATH}/src/arcane/nabla"
    "${ARCANESRCROOT}/doc/doc_common/chap_build_install/subchap_prerequisites/snippets"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_EXAMPLE_PATH ${DOXYGEN_EXAMPLE_PATH}
    "${ARCANESRCROOT}/doc/doc_common/chap_build_install/subchap_prerequisites/snippets"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_IMAGE_PATH ${DOXYGEN_IMAGE_PATH}
    "${ARCANESRCROOT}/doc/specifs/images"
    "${ARCANESRCROOT}/doc/theme/img"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
    "${ARCANESRCROOT}/doc/doc_dev/0_devmanual.md"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_INCLUDE_PATH ${DOXYGEN_INCLUDE_PATH}
    "${ARCANE_SRC_PATH}/arcane/utils"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_RECURSIVE
    "YES"
  )

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
