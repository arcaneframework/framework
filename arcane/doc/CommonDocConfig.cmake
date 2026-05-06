
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_commondoc_config_adoc_variables)

  # La variable ADOC_LEGACY_THEME permet de passer au thème Doxygen classique.
  if (NOT ADOC_LEGACY_THEME)
    if (ARCANEDOC_LEGACY_THEME AND ARCANEDOC_LEGACY_THEME STREQUAL ON)
      set(ADOC_LEGACY_THEME "ON")
    else ()
      set(ADOC_LEGACY_THEME "OFF")
    endif ()
  endif ()

  # ----------------------------------------------------------------------------

  # Variable pour savoir si la doc aura accès à internet ou non.
  if (NOT ADOC_MATHJAX)
    if (ARCANEDOC_OFFLINE AND ARCANEDOC_OFFLINE STREQUAL ON)
      set(ADOC_MATHJAX "OFF")
      message(STATUS "Offline mode for user and dev documentations")
    else ()
      set(ADOC_MATHJAX "ON")
    endif ()
  endif ()

  # ----------------------------------------------------------------------------

  if (NOT ADOC_PROJECT_REPO_LINK)
    set(ADOC_PROJECT_REPO_LINK "https://github.com/arcaneframework/framework")
  endif ()

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

macro(adoc_commondoc_config_doxygen_variables)

  set(DOXYGEN_ALIASES ${DOXYGEN_ALIASES}
    "arcane{1}=\\ref Arcane::\\1 \\\"\\1\\\""
    "arccore{1}=\\ref Arccore::\\1 \\\"\\1\\\""
    "arcaneacc{1}=\\ref Arcane::Accelerator::\\1 \\\"\\1\\\""
    "arcanemat{1}=\\ref Arcane::Materials::\\1 \\\"\\1\\\""
    "pr{1}=[PR #\\1](https://github.com/arcaneframework/framework/pull/\\1)"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_LATEX_CMD_NAME
    "latex"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_LATEX_BIB_STYLE
    "plain"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_PREDEFINED ${DOXYGEN_PREDEFINED}
    "DOXYGEN_DOC=1"
    "ARCCORE_BASE_EXPORT="
    "ARCCORE_COMMON_EXPORT="
    "ARCCORE_COLLECTION_EXPORT="
    "ARCCORE_ACCELERATOR_EXPORT="
    "ARCCORE_MESSAGEPASSING_EXPORT="
    "ARCANE_ALIGNAS_PACKED"
    "ARCANE_ALIGNAS"
    "ARCANE_BEGIN_NAMESPACE=namespace Arcane {"
    "ARCANE_END_NAMESPACE=}"
    "ARCANE_CORE_EXPORT="
    "ARCANE_ACCELERATOR_EXPORT="
    "ARCANE_ACCELERATOR_CORE_EXPORT="
    "ARCANE_UTILS_EXPORT="
    "ARCANE_IMPL_EXPORT="
    "ARCANE_GEOMETRY_EXPORT="
    "ARCCORE_HOST_DEVICE=__host__ __device__"
    "ARCCORE_DEPRECATED_2019(a)=[[deprecated(a)]]"
    "ARCCORE_DEPRECATED_2020(a)=[[deprecated(a)]]"
    "ARCCORE_DEPRECATED_2021(a)=[[deprecated(a)]]"
    "ARCCORE_DEPRECATED_REASON(a)=[[deprecated(a)]]"
    "ARCANE_DEPRECATED_REASON(a)=[[deprecated(a)]]"
  )

  # ----------------------------------------------------------------------------

  set(DOXYGEN_EXPAND_AS_DEFINED ${DOXYGEN_EXPAND_AS_DEFINED}
    "ARCANE_MATERIALS_EXPORT"
    "ARCANE_HAS_CXX11"
    "GEOMETRIC_BEGIN_NAMESPACE"
    "GEOMETRIC_END_NAMESPACE"
    "GEOMETRIC_BEGIN_NAMESPACE"
    "GEOMETRIC_END_NAMESPACE"
    "MATERIALS_BEGIN_NAMESPACE"
    "MATERIALS_END_NAMESPACE"
    "ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE"
    "ARCCORE_NORETURN"
    "ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS"
    "ARCCORE_DEFINE_ARRAY_PODTYPE"
  )

endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
