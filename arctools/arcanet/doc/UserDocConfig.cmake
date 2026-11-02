
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_userdoc_config_adoc_variables)

  # Replace "${CMAKE_SOURCE_DIR}/src" with the locations of the sources (it is possible to set the
  # DOXYGEN_RECURSIVE variable to `YES` to avoid having to specify all the directories).
  set(ADOC_DOXYGEN_INPUT ${ADOC_DOXYGEN_INPUT}
    "${DOC_DIR}/doc_common"
    "${DOC_DIR}/doc_common/chap_news"
    "${DOC_DIR}/doc_user"
    "${DOC_DIR}/doc_user/chap_getting_started"
    "${DOC_DIR}/doc_user/chap_arct"
    "${DOC_DIR}/doc_user/chap_arct/subchap_cmake"
    "${DOC_DIR}/doc_user/chap_arccher"
    "${DOC_DIR}/../arct/specifications.md"
  )

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_userdoc_config_doxygen_variables)

  # Replace "${CMAKE_SOURCE_DIR}/src" with the location of the sources.
  set(DOXYGEN_STRIP_FROM_PATH ${DOXYGEN_STRIP_FROM_PATH}
  )

  # ----------------------------------------------------------------------------

  # Replace "${CMAKE_SOURCE_DIR}/src" with the location of the sources.
  set(DOXYGEN_STRIP_FROM_INC_PATH ${DOXYGEN_STRIP_FROM_INC_PATH}
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_EXAMPLE_PATH ${DOXYGEN_EXAMPLE_PATH}
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_EXAMPLE_RECURSIVE
    "YES"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_IMAGE_PATH ${DOXYGEN_IMAGE_PATH}
    "${ADOC_DOC_CONFIG_DIR}/theme/img"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
    "${DOC_DIR}/doc_user/0_usermanual.md"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_RECURSIVE
    "NO"
  )

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
