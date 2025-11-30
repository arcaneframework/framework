# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arcane pour CUDA

set(ARCANE_SOURCES
  CudaAcceleratorRuntime.cc
  internal/Cupti.h
  )

arcane_find_package(CUDAToolkit REQUIRED)

arcane_add_library(arcane_accelerator_cuda_runtime
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator/cuda/runtime
  FILES ${ARCANE_SOURCES}
  )

target_include_directories(arcane_accelerator_cuda_runtime PRIVATE $<BUILD_INTERFACE:${Arcane_SOURCE_DIR}/src>)

target_link_libraries(arcane_accelerator_cuda_runtime PRIVATE Arccore::arccore_accelerator_cuda)

target_link_libraries(arcane_accelerator_cuda_runtime PUBLIC
  arccore_cuda_compile_flags
  CUDA::cudart
  CUDA::cuda_driver
)

# ----------------------------------------------------------------------------

if (TARGET CUDA::nvtx3)
  message(STATUS "CUDA: 'nvtx3' is available")
  target_compile_definitions(arcane_accelerator_cuda_runtime PRIVATE ARCANE_HAS_CUDA_NVTOOLSEXT)
  target_link_libraries(arcane_accelerator_cuda_runtime PRIVATE CUDA::nvtx3)
endif()

# ----------------------------------------------------------------------------

if (TARGET CUDA::cupti)
  message(STATUS "CUDA: 'cupti' is available")
  target_sources(arcane_accelerator_cuda_runtime PRIVATE Cupti.cc)
  target_compile_definitions(arcane_accelerator_cuda_runtime PRIVATE ARCANE_HAS_CUDA_CUPTI)
  target_link_libraries(arcane_accelerator_cuda_runtime PRIVATE CUDA::cupti)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_cuda_testlib Test.cu Test2.cc TestCpp.cc)
arcane_accelerator_add_source_files(Test2.cc)
set_target_properties(arcane_accelerator_cuda_testlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(arcane_accelerator_cuda_testlib PUBLIC arcane_accelerator arcane_core arccore_cuda_compile_flags)

add_executable(arcane_accelerator_cuda_test TestMain.cc)
target_link_libraries(arcane_accelerator_cuda_test PUBLIC arcane_accelerator_cuda_testlib arcane_accelerator_cuda_runtime)

arcane_register_library(arcane_accelerator_cuda_runtime OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
