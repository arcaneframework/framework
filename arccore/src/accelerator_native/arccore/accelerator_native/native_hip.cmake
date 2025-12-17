# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Backend Arccore pour ROCM/HIP

set(ARCCORE_SOURCES
  HipAccelerator.cc
  HipAccelerator.h
)

find_package(Hip REQUIRED)

# Créé une cible interface pour propager les options de compilation
# communes pour la compilation HIP

add_library(arccore_hip_compile_flags INTERFACE)
# Normalement il ne devrait pas y avoir besoin d'ajouter cette ligne mais si on le fait
# pas avec 'ROCM 4.3', alors il ne trouve pas 'libclang_rt.builtins-x86_64.a'
target_link_directories(arccore_hip_compile_flags INTERFACE ${HIP_CLANG_INCLUDE_PATH}/lib/linux)

target_compile_options(arccore_hip_compile_flags INTERFACE
# Pas d'option spécifique pour l'instant
)
install(TARGETS arccore_hip_compile_flags EXPORT ArccoreTargets)

arccore_add_component_library(accelerator_native
  LIB_NAME arccore_accelerator_hip
  FILES ${ARCCORE_SOURCES}
)

target_link_libraries(arccore_accelerator_hip PUBLIC
  arccore_base
  arccore_hip_compile_flags
  hip::host
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
