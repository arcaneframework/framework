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

target_link_libraries(arcane_accelerator_cuda_runtime PUBLIC
  arcane_accelerator arcane_cuda_compile_flags CUDA::cudart CUDA::cuda_driver
  )

# Note: à partir de CMake 3.25 'nvToolsExt' devient 'nvtx3'
#
# NB: Avec CMake 3.25.2 le remplacement de 'nvToolsExt' par 'nvtx3' produit des erreurs de link :
# /usr/bin/ld : ../../../../../../../../lib/libarcane_accelerator_cuda_runtime.so : référence indéfinie vers « nvtxRangePop »
# /usr/bin/ld : ../../../../../../../../lib/libarcane_accelerator_cuda_runtime.so : référence indéfinie vers « nvtxRangePushA »
#
#if (TARGET CUDA::nvtx3)
#  target_compile_definitions(arcane_accelerator_cuda_runtime PRIVATE ARCANE_HAS_CUDA_NVTOOLSEXT)
#  target_link_libraries(arcane_accelerator_cuda_runtime PRIVATE CUDA::nvtx3)
if (TARGET CUDA::nvToolsExt)
  message(STATUS "CUDA: 'nvToolsExt' is available")
  target_compile_definitions(arcane_accelerator_cuda_runtime PRIVATE ARCANE_HAS_CUDA_NVTOOLSEXT)
  target_link_libraries(arcane_accelerator_cuda_runtime PRIVATE CUDA::nvToolsExt)
endif()
if (TARGET CUDA::cupti)
  message(STATUS "CUDA: 'cupti' is available")
  target_sources(arcane_accelerator_cuda_runtime PRIVATE Cupti.cc)
  target_compile_definitions(arcane_accelerator_cuda_runtime PRIVATE ARCANE_HAS_CUDA_CUPTI)
  target_link_libraries(arcane_accelerator_cuda_runtime PRIVATE CUDA::cupti)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_cuda_testlib Test.cu TestCpp.cc)
set_target_properties(arcane_accelerator_cuda_testlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(arcane_accelerator_cuda_testlib PUBLIC arcane_core arcane_cuda_compile_flags)

add_executable(arcane_accelerator_cuda_test TestMain.cc)
target_link_libraries(arcane_accelerator_cuda_test PUBLIC arcane_accelerator_cuda_testlib arcane_accelerator_cuda_runtime)

arcane_register_library(arcane_accelerator_cuda_runtime OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
