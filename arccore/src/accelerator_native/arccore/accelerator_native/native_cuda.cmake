# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Arcane Backend for CUDA

set(ARCCORE_SOURCES
  CudaAccelerator.cc
  CudaAccelerator.h
  )

find_package(CUDAToolkit REQUIRED)

if (ARCCORE_HAS_CXX23 AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0))
  message(FATAL_ERROR "CUDA <=12 doesn't support C++23. Add -DARCCORE_CXX_STANDARD=20 to the configuration")
endif()

# Create an interface target to propagate compilation options
# common for CUDA compilation

add_library(arccore_cuda_compile_flags INTERFACE)
add_library(arccore_cuda_build_compile_flags INTERFACE)

option(ARCCORE_CUDA_DEVICE_DEBUG "If True, add '--device-debug' to cuda compiler flags" OFF)
set(_CUDA_DEBUG_FLAGS "-lineinfo")
if (ARCCORE_CUDA_DEVICE_DEBUG)
  set(_CUDA_DEBUG_FLAGS "--device-debug")
endif()

if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  # When compiling with clang, certain debug information should not be generated
  # because it causes a compilation error
  # (see https://github.com/llvm/llvm-project/issues/58491)
  # This might be fixed in clang version 19
  target_compile_options(arccore_cuda_compile_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xarch_device>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-g0>"
  )
else()
  # Classic CUDA compiler (NVCC or NVHPC)
  target_compile_options(arccore_cuda_compile_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-g>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Werror>"
    "$<$<COMPILE_LANGUAGE:CUDA>:cross-execution-space-call>"
  )

  target_compile_options(arccore_cuda_build_compile_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:${_CUDA_DEBUG_FLAGS}>"
  )
endif()

install(TARGETS arccore_cuda_compile_flags EXPORT ArccoreTargets)
install(TARGETS arccore_cuda_build_compile_flags EXPORT ArccoreTargets)

arccore_add_component_library(accelerator_native
  LIB_NAME arccore_accelerator_cuda
  FILES ${ARCCORE_SOURCES}
)

target_link_libraries(arccore_accelerator_cuda PUBLIC
  arccore_base
  arccore_cuda_compile_flags
  $<BUILD_INTERFACE:arccore_cuda_build_compile_flags>
  CUDA::cudart
)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Implementation of CUDA-specific routines.
# For this to work, this library must be static
# (otherwise, it seems CMake does not add it to the link edit of targets
# that depend on it, but perhaps this is an 'nvcc' limitation)
# It also allows propagating compilation options to users
# of this library
# add_library(arccore_accelerator_cuda_impl STATIC
#   Reduce.cu
#   )
# set_target_properties(arccore_accelerator_cuda_impl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(arccore_accelerator_cuda_impl PROPERTIES LINKER_LANGUAGE CUDA)
# target_link_libraries(arccore_accelerator_cuda_impl PUBLIC arccore_core arccore_cuda_compile_flags)
# target_link_libraries(arccore_accelerator_cuda_impl PUBLIC CUDA::cudart)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
