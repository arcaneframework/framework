#
# Find the BLAS includes and library
#
# This module uses
# BLAS_ROOT
#
# This module defines
# BLAS_FOUND
# BLAS_INCLUDE_DIRS
# BLAS_LIBRARIES
#
# Target blas

if(NOT BLAS_ROOT)
  set(BLAS_ROOT $ENV{BLAS_ROOT})
endif()

if(BLAS_ROOT)
  set(_BLAS_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_BLAS_SEARCH_OPTS)
endif()

if(NOT BLAS_FOUND) 

  find_library(BLAS_LIBRARY 
    NAMES openblas
		HINTS ${BLAS_ROOT}
		PATH_SUFFIXES lib
		${_BLAS_SEARCH_OPTS}
    )
  mark_as_advanced(BLAS_LIBRARY)
  
  find_path(BLAS_INCLUDE_DIR cblas.h
    HINTS ${BLAS_ROOT} 
		PATH_SUFFIXES include
    ${_BLAS_SEARCH_OPTS}
    )
  mark_as_advanced(BLAS_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(BLAS_FIND_QUIETLY ON)

find_package_handle_standard_args(BLAS 
	DEFAULT_MSG 
	BLAS_INCLUDE_DIR 
	BLAS_LIBRARY)

if(BLAS_FOUND AND NOT TARGET blas)

  set(BLAS_INCLUDE_DIRS ${BLAS_INCLUDE_DIR})
  
  set(BLAS_LIBRARIES ${BLAS_LIBRARY})
  
  # BLAS
	
  add_library(blas UNKNOWN IMPORTED)
	
  set_target_properties(blas PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIRS}")
  
	set_target_properties(blas PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${BLAS_LIBRARY}")

endif()
