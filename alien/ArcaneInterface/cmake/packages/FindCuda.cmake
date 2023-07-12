#
# Find the CUDA includes and library
#
# This module uses
# CUDA_ROOT
#
# This module defines
# CUDA_FOUND
# CUDA_INCLUDE_DIRS
# CUDA_LIBRARIES
#
# Target cuda

if(NOT CUDA_ROOT)
  set(CUDA_ROOT $ENV{CUDA_ROOT})
endif()

if(CUDA_ROOT)
  set(_CUDA_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_CUDA_SEARCH_OPTS)
endif()

if(NOT CUDA_FOUND) 

  find_library(CUBLAS_LIBRARY 
    NAMES cublas
    HINTS ${CUDA_ROOT}
    PATH_SUFFIXES lib64
    ${_CUDA_SEARCH_OPTS}
    )
  mark_as_advanced(CUBLAS_LIBRARY)
  
  find_library(CUSPARSE_LIBRARY 
    NAMES cusparse
    HINTS ${CUDA_ROOT}
    PATH_SUFFIXES lib64
    ${_CUDA_SEARCH_OPTS}
    )
  mark_as_advanced(CUSPARSE_LIBRARY)
  
  find_library(CUDART_LIBRARY 
    NAMES cudart
    HINTS ${CUDA_ROOT}
    PATH_SUFFIXES lib64
    ${_CUDA_SEARCH_OPTS}
    )
  mark_as_advanced(CUDART_LIBRARY)
  
  find_path(CUDA_INCLUDE_DIR cuda.h
    HINTS ${CUDA_ROOT} 
    PATH_SUFFIXES include
    ${_CUDA_SEARCH_OPTS}
    )
  mark_as_advanced(CUDA_INCLUDE_DIR)
  
  find_path(HELPER_CUDA_INCLUDE_DIR helper_cuda.h
      HINTS ${CUDA_ROOT} 
      PATH_SUFFIXES samples/common/inc
      ${_CUDA_SEARCH_OPTS}
      )
  mark_as_advanced(HELPER_CUDA_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(CUDA_FIND_QUIETLY ON)

find_package_handle_standard_args(CUDA DEFAULT_MSG
  CUDA_INCLUDE_DIR 
  CUBLAS_LIBRARY
  CUDART_LIBRARY
  CUSPARSE_LIBRARY)

if(CUDA_FOUND AND NOT TARGET cuda)
  
  set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIR} ${HELPER_CUDA_INCLUDE_DIR})

  set(CUDA_LIBRARIES ${CUBLAS_LIBRARY}
                     ${CUDART_LIBRARY}
                     ${CUSPARSE_LIBRARY})
    
  # cublas
  
  add_library(cublas UNKNOWN IMPORTED)
  
  set_target_properties(cublas PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
  
  set_target_properties(cublas PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${CUBLAS_LIBRARY}")
  
  # cusparse
  
  add_library(cusparse UNKNOWN IMPORTED)
  
  set_target_properties(cusparse PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
  
  set_target_properties(cusparse PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${CUSPARSE_LIBRARY}")
  
  # cuda rt
  
  add_library(cudart UNKNOWN IMPORTED)
  
  set_target_properties(cudart PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
  
  set_target_properties(cudart PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${CUDART_LIBRARY}")
  
  # cuda
  
  add_library(cuda INTERFACE IMPORTED)
  
  set_property(TARGET cuda APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "cublas")

  set_property(TARGET cuda APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "cusparse")

  set_property(TARGET cuda APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "cudart")
 
endif()
