# NOTE: Ce FindCUDA n'est utilisé que pour 'aleph_cuda' et il est obsolète.
# Il faudrait utiliser find_package(CUDAToolkit) à la place

# A partir de CMake 3.0, le package CUDA est disponible
# Pour l'instant on ne s'en sert que pour récupérer le numéro
# de version de CUDA et s'il est inférieur à 7 on n'utilise
# par CUDA car il n'y aura pas le support du C++11.
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(CUDA QUIET)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})
message("CUDA_VERSION ${CUDA_VERSION_MAJOR}")

#
# Find the Cuda includes and library
#
# This module defines
# CUDA_INCLUDE_DIR, where to find headers,
# CUDA_LIBRARIES, the libraries to link against to use cuda.

FIND_PATH(CUDA_INCLUDE_DIR cuda.h)
FIND_LIBRARY(CUDART_LIBRARY cudart)
FIND_LIBRARY(CUBLAS_LIBRARY cublas)
FIND_PROGRAM(CUDA_BIN_NVCC nvcc)

message(STATUS "CUDA_VERSION_MAJOR = ${CUDA_VERSION_MAJOR}")
MESSAGE(STATUS "CUDA_INCLUDE_DIR = ${CUDA_INCLUDE_DIR}")
MESSAGE(STATUS "CUDART_LIBRARY = ${CUDART_LIBRARY}")
MESSAGE(STATUS "CUBLAS_LIBRARY = ${CUBLAS_LIBRARY}")
MESSAGE(STATUS "CUDA_BIN_NVCC = ${CUDA_BIN_NVCC}")

set(CUDA_FOUND NO)

if (CUDA_VERSION_MAJOR GREATER 6)
  if(CUDA_INCLUDE_DIR AND CUDART_LIBRARY AND CUBLAS_LIBRARY)
    set(CUDA_FOUND TRUE)
    set(CUDA_LIBRARIES ${CUDART_LIBRARY} ${CUBLAS_LIBRARY})
    set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIR})
    get_filename_component(CUDA_NVCC_PATH ${CUDA_BIN_NVCC} PATH)
    message(STATUS "CUDA_NVCC_PATH = ${CUDA_NVCC_PATH}")
    message(STATUS "ADD_CUDA_LIBRARY")
    arcane_add_package_library(CUDA CUDA)
  endif(CUDA_INCLUDE_DIR AND CUDART_LIBRARY AND CUBLAS_LIBRARY)
else ()
  message(STATUS "Disabling cuda because version is too old (version 7+ required)")
endif ()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
