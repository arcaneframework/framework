# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Backend Arcane pour SYCL
# Cette composante est interne à Arccore/Arcane

set(ARCANE_SOURCES
  SyclAccelerator.cc
  SyclAccelerator.h
  )

# Créé une cible interface pour propager les options de compilation
# communes pour la compilation SYCL

add_library(arcane_sycl_compile_flags INTERFACE)

target_compile_options(arcane_sycl_compile_flags INTERFACE
  # Pas d'option spécifique pour l'instant
)
if (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
  target_link_options(arcane_sycl_compile_flags INTERFACE "-lsycl")
endif()

install(TARGETS arcane_sycl_compile_flags EXPORT ArcaneTargets)

arcane_add_library(arcane_accelerator_sycl
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator/sycl
  FILES ${ARCANE_SOURCES}
)
target_compile_options(arcane_accelerator_sycl PRIVATE "${ARCANE_CXX_SYCL_FLAGS}")

# Pour que le runtime associé trouve les fichiers d'en-tête de cette composante
target_include_directories(arcane_accelerator_sycl PUBLIC $<BUILD_INTERFACE:${Arcane_SOURCE_DIR}/src>)

target_link_libraries(arcane_accelerator_sycl PUBLIC
  Arccore::arccore_common arcane_sycl_compile_flags
)
target_link_options(arcane_accelerator_sycl PUBLIC "${ARCANE_CXX_SYCL_FLAGS}")

# Détecte oneDPL si on utilise DPC++
# On fait uniquement la détection et on suppose que c'est cohérent avec le
# compilateur. Il ne faut pas ajouter la cible 'oneDPL' à 'arcane_accelerator_sycl'
# car cela ajoute des flags de compilation (notamment liés à OpenMP) qui
# peuvent provoquer des erreurs de compilation. En plus de cela, il peut y avoir
# des incohérences avec la version de TBB utilisée ailleurs dans Arcane.
if (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
  find_package(oneDPL CONFIG)
  message(STATUS "[Sycl] oneDPL found?=${oneDPL_FOUND} Version=${oneDPL_VERSION}")
  if (oneDPL_FOUND)
    target_compile_definitions(arcane_accelerator_sycl PUBLIC ARCANE_HAS_ONEDPL)
  endif()
endif()

# ----------------------------------------------------------------------------

arcane_register_library(arcane_accelerator_sycl OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
