# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Backend Arccore pour SYCL

set(ARCCORE_SOURCES
  SyclAccelerator.cc
  SyclAccelerator.h
)

# Pour compatibilité avec l'existant
if (DEFINED ARCANE_CXX_SYCL_FLAGS AND NOT DEFINED ARCCORE_CXX_SYCL_FLAGS)
  set(ARCCORE_CXX_SYCL_FLAGS "${ARCANE_CXX_SYCL_FLAGS}")
endif()

# Créé une cible interface pour propager les options de compilation
# communes pour la compilation SYCL

add_library(arccore_sycl_compile_flags INTERFACE)

target_compile_options(arccore_sycl_compile_flags INTERFACE
  # Pas d'option spécifique pour l'instant
)
if (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
  target_link_options(arccore_sycl_compile_flags INTERFACE "-lsycl")
endif()

install(TARGETS arccore_sycl_compile_flags EXPORT ArccoreTargets)

arccore_add_component_library(accelerator_native
  LIB_NAME arccore_accelerator_sycl
  FILES ${ARCCORE_SOURCES}
)
target_compile_options(arccore_accelerator_sycl PRIVATE "${ARCCORE_CXX_SYCL_FLAGS}")

target_link_libraries(arccore_accelerator_sycl PUBLIC
  arccore_base
  arccore_sycl_compile_flags
)
target_link_options(arccore_accelerator_sycl PUBLIC "${ARCCORE_CXX_SYCL_FLAGS}")

# Détecte oneDPL si on utilise DPC++
# On fait uniquement la détection et on suppose que c'est cohérent avec le
# compilateur. Il ne faut pas ajouter la cible 'oneDPL' à 'arccore_accelerator_sycl'
# car cela ajoute des flags de compilation (notamment liés à OpenMP) qui
# peuvent provoquer des erreurs de compilation. En plus de cela, il peut y avoir
# des incohérences avec la version de TBB utilisée ailleurs dans Arccore.
if (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
  find_package(oneDPL CONFIG)
  message(STATUS "[Sycl] oneDPL found?=${oneDPL_FOUND} Version=${oneDPL_VERSION}")
  if (oneDPL_FOUND)
    target_compile_definitions(arccore_accelerator_sycl PUBLIC ARCCORE_HAS_ONEDPL ARCANE_HAS_ONEDPL)
  endif()
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
