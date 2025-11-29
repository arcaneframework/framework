# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arcane pour HIP

set(ARCANE_SOURCES
  HipAcceleratorRuntime.cc
  )

arcane_add_library(arcane_accelerator_hip_runtime
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator/hip/runtime
  FILES ${ARCANE_SOURCES}
  )

target_link_libraries(arcane_accelerator_hip_runtime PRIVATE
  arcane_accelerator_hip
)
target_link_libraries(arcane_accelerator_hip_runtime PUBLIC
  arcane_hip_compile_flags
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
  target_link_libraries(arcane_accelerator_hip_runtime PRIVATE ${ROCTX_LIBRARIES})
  target_include_directories(arcane_accelerator_hip_runtime PRIVATE ${ROCTX_INCLUDE}/roctracer)
  target_compile_definitions(arcane_accelerator_hip_runtime PRIVATE ARCANE_HAS_ROCTX)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_hip_testlib Test.cu.cc TestCpp.cc)
set_source_files_properties(Test.cu.cc PROPERTIES LANGUAGE HIP)
target_link_libraries(arcane_accelerator_hip_testlib PUBLIC  arcane_accelerator arcane_core arcane_hip_compile_flags)

add_executable(arcane_accelerator_hip_test TestMain.cc)
target_link_libraries(arcane_accelerator_hip_test PUBLIC arcane_accelerator_hip_testlib arcane_accelerator_hip_runtime)

arcane_register_library(arcane_accelerator_hip_runtime OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
