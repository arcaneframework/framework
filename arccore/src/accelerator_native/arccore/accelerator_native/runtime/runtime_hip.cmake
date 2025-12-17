# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arccore pour HIP

message(STATUS "Adding Arccore Runtime for Hip")

set(ARCCORE_SOURCES
  HipAcceleratorRuntime.cc
)

arccore_add_library(arccore_accelerator_hip_runtime
  INPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/runtime
  FILES ${ARCCORE_SOURCES}
)

target_link_libraries(arccore_accelerator_hip_runtime PRIVATE
  arccore_accelerator_hip
  arccore_common
  arccore_hip_compile_flags
)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Recherche 'roctx' (qui est dans roctracer)
# il n'y a pas l'air d'avoir de fichier de configuration associé
# Normalement c'est dans un sous-répertoire 'roctracer' de 'rocm'
find_path(ROCTX_INCLUDE NAMES roctx.h PATH_SUFFIXED roctracer)
find_library(ROCTX_LIBRARIES NAMES roctx64)
find_library(ROCTRACER_LIBRARIES NAMES roctracer64)
message(STATUS "ROCTX_INCLUDE=${ROCTX_INCLUDE}")
message(STATUS "ROCTX_LIBRARIES=${ROCTX_LIBRARIES}")
if (ROCTX_INCLUDE AND ROCTX_LIBRARIES)
  target_link_libraries(arccore_accelerator_hip_runtime PRIVATE ${ROCTX_LIBRARIES})
  target_include_directories(arccore_accelerator_hip_runtime PRIVATE ${ROCTX_INCLUDE}/roctracer)
  target_compile_definitions(arccore_accelerator_hip_runtime PRIVATE ARCCORE_HAS_ROCTX)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
