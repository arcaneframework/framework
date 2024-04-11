# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Backend Arcane pour SYCL

set(ARCANE_SOURCES
  SyclAccelerator.cc
  SyclAccelerator.h
  internal/SyclAcceleratorInternal.h
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

target_link_libraries(arcane_accelerator_sycl PUBLIC
  arcane_core arcane_sycl_compile_flags
)
target_link_options(arcane_accelerator_sycl PUBLIC "${ARCANE_CXX_SYCL_FLAGS}")

# ----------------------------------------------------------------------------

arcane_register_library(arcane_accelerator_sycl OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
