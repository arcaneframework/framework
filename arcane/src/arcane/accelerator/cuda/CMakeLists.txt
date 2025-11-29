# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Backend Arcane pour CUDA
# Cette composante est interne à Arccore/Arcane

set(ARCANE_SOURCES
  CudaAccelerator.cc
  CudaAccelerator.h
  )

arcane_find_package(CUDAToolkit REQUIRED)

if ((ARCANE_HAS_CXX23) AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0))
  message(FATAL_ERROR "CUDA <=12 doesn't support C++23. Add -DARCCORE_CXX_STANDARD=20 to the configuration")
endif()

# Créé une cible interface pour propager les options de compilation
# communes pour la compilation CUDA

add_library(arcane_cuda_compile_flags INTERFACE)
add_library(arcane_cuda_build_compile_flags INTERFACE)

option(ARCANE_CUDA_DEVICE_DEBUG "If True, add '--device-debug' to cuda compiler flags" OFF)
set(_CUDA_DEBUG_FLAGS "-lineinfo")
if (ARCANE_CUDA_DEVICE_DEBUG)
  set(_CUDA_DEBUG_FLAGS "--device-debug")
endif()

if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  # Lorsqu'on compile avec clang, il ne faut pas générer certaines informations
  # de debug car cela provoque une erreur de compilation
  # (voir https://github.com/llvm/llvm-project/issues/58491)
  # Cela sera peut-être corrigé avec la version 19 de clang
  target_compile_options(arcane_cuda_compile_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xarch_device>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-g0>"
  )
else()
  # Compilateur CUDA classique (NVCC ou NVHPC)
  target_compile_options(arcane_cuda_compile_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-g>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Werror>"
    "$<$<COMPILE_LANGUAGE:CUDA>:cross-execution-space-call>"
  )

  target_compile_options(arcane_cuda_build_compile_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:${_CUDA_DEBUG_FLAGS}>"
  )
endif()

install(TARGETS arcane_cuda_compile_flags EXPORT ArcaneTargets)
install(TARGETS arcane_cuda_build_compile_flags EXPORT ArcaneTargets)

arcane_add_library(arcane_accelerator_cuda
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator/cuda
  FILES ${ARCANE_SOURCES}
  )

# Pour que le runtime associé trouve les fichiers d'en-tête de cette composante
target_include_directories(arcane_accelerator_cuda PUBLIC $<BUILD_INTERFACE:${Arcane_SOURCE_DIR}/src>)

target_link_libraries(arcane_accelerator_cuda PUBLIC
  Arccore::arccore_common
  arcane_cuda_compile_flags
  $<BUILD_INTERFACE:arcane_cuda_build_compile_flags>
  CUDA::cudart
)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Implémentation des routines spécifiques CUDA.
# Pour que cela fonctionne, cette bibliothèque doit être statique
# (sinon, il semble que CMake ne l'ajoute pas à l'édition de lien des cibles
# qui en dépende mais peut-être s'agit-il d'une limitation de 'nvcc')
# Elle permet aussi de propager les options de compilation aux utilisateurs
# de cette bibliothèque
add_library(arcane_accelerator_cuda_impl STATIC
  Reduce.cu
#  CudaReduceImpl.h
  )
set_target_properties(arcane_accelerator_cuda_impl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(arcane_accelerator_cuda_impl PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(arcane_accelerator_cuda_impl PUBLIC arcane_core arcane_cuda_compile_flags)
target_link_libraries(arcane_accelerator_cuda_impl PUBLIC CUDA::cudart)

# ----------------------------------------------------------------------------

arcane_register_library(arcane_accelerator_cuda  OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
