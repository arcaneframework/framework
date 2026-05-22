
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_devdoc_config_adoc_variables)

  set(ADOC_DOXYGEN_INPUT ${ADOC_DOXYGEN_INPUT}
    "${CMAKE_SOURCE_DIR}/alien/standalone/src"
    "${DOC_DIR}/doc_common"
    "${DOC_DIR}/doc_dev"
  )

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_devdoc_config_doxygen_variables)

  set(DOXYGEN_EXCLUDE
    "${CMAKE_SOURCE_DIR}/alien/standalone/src/test_framework"
    "${CMAKE_SOURCE_DIR}/alien/standalone/src/core/tests"
    "${CMAKE_SOURCE_DIR}/alien/standalone/src/movesemantic/tests"
    "${CMAKE_SOURCE_DIR}/alien/standalone/src/refsemantic/tests"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_STRIP_FROM_PATH ${DOXYGEN_STRIP_FROM_PATH}
    "${CMAKE_SOURCE_DIR}/alien/standalone/src"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_STRIP_FROM_INC_PATH ${DOXYGEN_STRIP_FROM_INC_PATH}
    "${CMAKE_SOURCE_DIR}/alien/standalone/src"
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
