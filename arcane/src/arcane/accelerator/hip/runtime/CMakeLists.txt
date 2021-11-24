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

target_link_libraries(arcane_accelerator_hip_runtime PUBLIC
  arcane_accelerator arcane_hip_compile_flags
  )

#if (TARGET CUDA::nvToolsExt)
#  target_compile_definitions(arcane_accelerator_hip_runtime PRIVATE ARCANE_HAS_CUDA_NVTOOLSEXT)
#  target_link_libraries(arcane_accelerator_hip_runtime PRIVATE HIP::nvToolsExt)
#endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_hip_testlib Test.cu.cc TestCpp.cc)
set_source_files_properties(Test.cu.cc PROPERTIES LANGUAGE HIP)
#set_target_properties(arcane_accelerator_hip_testlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(arcane_accelerator_hip_testlib PUBLIC arcane_core)
target_link_libraries(arcane_accelerator_hip_testlib PUBLIC arcane_core arcane_hip_compile_flags)

add_executable(arcane_accelerator_hip_test TestMain.cc)
target_link_libraries(arcane_accelerator_hip_test PUBLIC arcane_accelerator_hip_testlib arcane_accelerator_hip_runtime)

arcane_register_library(arcane_accelerator_hip_runtime OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
