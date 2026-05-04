

set(DOXYGEN_PROJECT_LOGO
  "${ARCANESRCROOT}/doc/theme/img/arcane_framework_small.webp"
)

set(DOXYGEN_PROJECT_BRIEF
  "Documentation utilisateur"
)

set(DOXYGEN_OUTPUT_LANGUAGE
  "French"
)
set(DOXYGEN_ABBREVIATE_BRIEF
  "The $name class"
  "The $name widget"
  "The $name file"
  "is"
  "provides"
  "specifies"
  "contains"
  "represents"
  "a"
  "an"
  "the"
)

set(DOXYGEN_STRIP_FROM_PATH
  "${Arcane_SOURCE_DIR}/src"
  "${ARCANE_CEA_SOURCE_PATH}/src"
  "${Arccore_ROOT}/src/base"
  "${Arccore_ROOT}/src/message_passing"
  "${Arccore_ROOT}/src/collections"
  "${Arccore_ROOT}/src/common"
  "${Arccore_ROOT}/src/accelerator"
  "${Arccore_ROOT}/src/concurrency"
  "${Arccore_ROOT}/src/trace"
  "${Arccore_ROOT}/src/message_passing_mpi"
  "${Arccore_ROOT}/src/serialize"
  "${Arccore_ROOT}"
  "${Arcane_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

set(DOXYGEN_STRIP_FROM_INC_PATH
  "${Arcane_SOURCE_DIR}/src"
  "${ARCANE_CEA_SOURCE_PATH}/src"
  "${Arccore_ROOT}/src/base"
  "${Arccore_ROOT}/src/message_passing"
  "${Arccore_ROOT}/src/collections"
  "${Arccore_ROOT}/src/common"
  "${Arccore_ROOT}/src/accelerator"
  "${Arccore_ROOT}/src/concurrency"
  "${Arccore_ROOT}/src/trace"
  "${Arccore_ROOT}/src/message_passing_mpi"
  "${Arccore_ROOT}/src/serialize"
  "${Arccore_ROOT}"
  "${Arcane_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/share/axl/dox"
)

set(DOXYGEN_ALIASES
  "arcane{1}=\\ref Arcane::\\1 \\\"\\1\\\""
  "arccore{1}=\\ref Arccore::\\1 \\\"\\1\\\""
  "arcaneacc{1}=\\ref Arcane::Accelerator::\\1 \\\"\\1\\\""
  "arcanemat{1}=\\ref Arcane::Materials::\\1 \\\"\\1\\\""
  "pr{1}=[PR #\\1](https://github.com/arcaneframework/framework/pull/\\1)"
)

set(DOXYGEN_LAYOUT_FILE
  "${ARCANESRCROOT}/doc/doc_user/layout.xml"
)

set(DOXYGEN_EXAMPLE_PATH
  "${Arcane_SOURCE_DIR}/src/arcane/tests/accelerator"
  "${Arcane_SOURCE_DIR}/src/arcane/tests/cartesianmesh"
  "${Arcane_SOURCE_DIR}/src/arcane/tests/material"
  "${Arcane_SOURCE_DIR}/src/arcane/tests"
  "${ARCANESRCROOT}/samples_build/samples"
  "${ARCANESRCROOT}/doc/doc_common/chap_build_install/subchap_prerequisites/snippets"
  "${CMAKE_BINARY_DIR}/share/axl/dox/snippets"
)
set(DOXYGEN_EXAMPLE_PATTERNS
  ""
)
set(DOXYGEN_EXAMPLE_RECURSIVE
  "YES"
)
set(DOXYGEN_IMAGE_PATH
  "${ARCANESRCROOT}/doc/specifs/images"
  "${ARCANESRCROOT}/doc/theme/img"
  "${ARCANESRCROOT}/doc/doc_user/chap_core_types/img"
  "${ARCANESRCROOT}/doc/doc_user/chap_entities/img"
  "${ARCANESRCROOT}/doc/doc_user/chap_entities/subchap_amr_cartesianmesh/img"
  "${ARCANESRCROOT}/doc/doc_user/chap_examples/subchap_simple_example/img"
  "${ARCANESRCROOT}/doc/doc_user/chap_examples/subchap_concret_example/img"
  "${ARCANESRCROOT}/doc/doc_user/chap_io/subchap_timehistory/img"
  "${ARCANESRCROOT}/doc/doc_user/chap_getting_started/img"
)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
  "${ARCANESRCROOT}/doc/doc_user/0_usermanual.md"
)

set(DOXYGEN_HTML_EXTRA_FILES ${DOXYGEN_HTML_EXTRA_FILES}
  "${ARCANESRCROOT}/doc/theme/img/logo_arcane.svg"
)

set(DOXYGEN_GENERATE_LATEX
  "NO"
)
set(DOXYGEN_LATEX_CMD_NAME
  "latex"
)
set(DOXYGEN_LATEX_BIB_STYLE
  "plain"
)

set(DOXYGEN_INCLUDE_PATH
  "${ARCANE_SRC_PATH}/arcane/utils"
  "${ARCANE_CEA_SOURCE_PATH}/src"
)
set(DOXYGEN_PREDEFINED
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
set(DOXYGEN_EXPAND_AS_DEFINED
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

