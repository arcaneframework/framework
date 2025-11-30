# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arccore pour CUDA

message(STATUS "Adding Arccore Runtime for Cuda")

set(ARCCORE_SOURCES
  CudaAcceleratorRuntime.cc
  Cupti.h
)
if (TARGET CUDA::cupti)
  list(APPEND ARCCORE_SOURCES Cupti.cc)
endif()

arccore_add_library(arccore_accelerator_cuda_runtime
  INPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/runtime
  RELATIVE_PATH .
  FILES ${ARCCORE_SOURCES}
  )

target_link_libraries(arccore_accelerator_cuda_runtime PRIVATE Arccore::arccore_accelerator_cuda)

target_link_libraries(arccore_accelerator_cuda_runtime PUBLIC
  arccore_cuda_compile_flags
  CUDA::cudart
  CUDA::cuda_driver
)

# ----------------------------------------------------------------------------

if (TARGET CUDA::nvtx3)
  message(STATUS "CUDA: 'nvtx3' is available")
  target_compile_definitions(arccore_accelerator_cuda_runtime PRIVATE ARCCORE_HAS_CUDA_NVTOOLSEXT ARCANE_HAS_CUDA_NVTOOLSEXT)
  target_link_libraries(arccore_accelerator_cuda_runtime PRIVATE CUDA::nvtx3)
endif()

# ----------------------------------------------------------------------------

if (TARGET CUDA::cupti)
  message(STATUS "CUDA: 'cupti' is available")
  #target_sources(arccore_accelerator_cuda_runtime PRIVATE Cupti.cc)
  target_compile_definitions(arccore_accelerator_cuda_runtime PRIVATE ARCCORE_HAS_CUDA_CUPTI ARCANE_HAS_CUDA_CUPTI)
  target_link_libraries(arccore_accelerator_cuda_runtime PRIVATE CUDA::cupti)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
