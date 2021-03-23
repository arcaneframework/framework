# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arcane pour CUDA

set(ARCANE_SOURCES
  CudaAcceleratorRuntime.cc
  )

find_package(CUDAToolkit REQUIRED)

arcane_add_library(arcane_accelerator_cuda_runtime
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator/cuda/runtime
  FILES ${ARCANE_SOURCES}
  )

target_link_libraries(arcane_accelerator_cuda_runtime PUBLIC
  arcane_accelerator arcane_cuda_compile_flags CUDA::cudart
  )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_cuda_testlib Test.cu TestCpp.cc)
set_target_properties(arcane_accelerator_cuda_testlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(arcane_accelerator_cuda_testlib PUBLIC arcane_core arcane_cuda_compile_flags)

add_executable(arcane_accelerator_cuda_test TestMain.cc)
target_link_libraries(arcane_accelerator_cuda_test PUBLIC arcane_accelerator_cuda_testlib arcane_accelerator_cuda_runtime)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
