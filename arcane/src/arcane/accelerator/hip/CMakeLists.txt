# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Backend Arcane pour ROCM/HIP

set(ARCANE_SOURCES
  HipAccelerator.cc
  HipAccelerator.h
)

arcane_find_package(Hip REQUIRED)

# Créé une cible interface pour propager les options de compilation
# communes pour la compilation HIP

add_library(arcane_hip_compile_flags INTERFACE)
# Normalement il ne devrait pas y avoir besoin d'ajouter cette ligne mais si on le fait
# pas avec 'ROCM 4.3', alors il ne trouve pas 'libclang_rt.builtins-x86_64.a'
target_link_directories(arcane_hip_compile_flags INTERFACE ${HIP_CLANG_INCLUDE_PATH}/lib/linux)

target_compile_options(arcane_hip_compile_flags INTERFACE
# Pas d'option spécifique pour l'instant
)
install(TARGETS arcane_hip_compile_flags EXPORT ArcaneTargets)

arcane_add_library(arcane_accelerator_hip
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator/hip
  FILES ${ARCANE_SOURCES}
)

# Pour que le runtime associé trouve les fichiers d'en-tête de cette composante
target_include_directories(arcane_accelerator_hip PUBLIC $<BUILD_INTERFACE:${Arcane_SOURCE_DIR}/src>)

target_link_libraries(arcane_accelerator_hip PUBLIC
  Arccore::arccore_common arcane_hip_compile_flags hip::host
)

# ----------------------------------------------------------------------------

arcane_register_library(arcane_accelerator_hip  OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
