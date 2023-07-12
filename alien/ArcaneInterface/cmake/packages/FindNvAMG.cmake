#
# Find the NVAMG includes and library
#
# This module uses
# NVAMG_ROOT
#
# This module defines
# NVAMG_FOUND
# NVAMG_INCLUDE_DIRS
# NVAMG_LIBRARIES
#
# Target nvamg 

if(NOT NVAMG_ROOT)
  set(NVAMG_ROOT $ENV{NVAMG_ROOT})
endif()

if(NVAMG_ROOT)
  set(_NVAMG_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_NVAMG_SEARCH_OPTS)
endif()

if(NOT NVAMG_FOUND) 

  find_library(NVAMG_LIBRARY 
    NAMES amgxsh
		HINTS ${NVAMG_ROOT}
		PATH_SUFFIXES lib
		${_NVAMG_SEARCH_OPTS}
    )
  mark_as_advanced(NVAMG_LIBRARY)
  
  find_path(NVAMG_INCLUDE_DIR amgx_c.h
    HINTS ${NVAMG_ROOT} 
		PATH_SUFFIXES include
    ${_NVAMG_SEARCH_OPTS}
    )
  mark_as_advanced(NVAMG_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(NVAMG_FIND_QUIETLY ON)

find_package_handle_standard_args(NVAMG 
	DEFAULT_MSG 
	NVAMG_INCLUDE_DIR 
	NVAMG_LIBRARY)

if(NVAMG_FOUND AND NOT TARGET nvamg)
    
  set(NVAMG_INCLUDE_DIRS ${NVAMG_INCLUDE_DIR})
  
  set(NVAMG_LIBRARIES ${NVAMG_LIBRARY})
	  
  add_library(nvamg UNKNOWN IMPORTED)
	  
  set_target_properties(nvamg PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${NVAMG_INCLUDE_DIRS}")
    
	set_target_properties(nvamg PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${NVAMG_LIBRARY}")
    
endif()
