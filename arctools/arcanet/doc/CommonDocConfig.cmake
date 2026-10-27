
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_commondoc_config_adoc_variables)
  set(ADOC_LANGUAGE
    "French"
  )
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_commondoc_config_doxygen_variables)

  # Il est possible de remplacer le lien.
  set(DOXYGEN_ALIASES ${DOXYGEN_ALIASES}
    "pr{1}=[PR #\\1](https://github.com/arcaneframework/framework/pull/\\1)"
  )

  # ----------------------------------------------------------------------------

  ## Pour mettre un autre logo (taille : 200x100).
  #set(DOXYGEN_PROJECT_LOGO
  #  "${DOC_DIR}/theme/img/arcane_framework_small.webp"
  #)

  # ----------------------------------------------------------------------------

  set(DOXYGEN_EXAMPLE_PATTERNS ${DOXYGEN_EXAMPLE_PATTERNS}
    ""
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_GENERATE_LATEX
    "NO"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_PREDEFINED ${DOXYGEN_PREDEFINED}
    "DOXYGEN_DOC=1"
  )

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
