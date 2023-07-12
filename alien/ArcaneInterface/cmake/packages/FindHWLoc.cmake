#
# Find the HWLOC includes and library
#
# This module uses
# HWLOC_ROOT
#
# This module defines
# HWLOC_FOUND
# HWLOC_INCLUDE_DIRS
# HWLOC_LIBRARIES
#
# Target hwloc

if(NOT HWLOC_ROOT)
  set(HWLOC_ROOT $ENV{HWLOC_ROOT})
endif()

if(HWLOC_ROOT)
  set(_HWLOC_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_HWLOC_SEARCH_OPTS)
endif()

if(NOT HWLOC_FOUND) 

  find_library(HWLOC_LIBRARY 
    NAMES  hwloc
		HINTS ${HWLOC_ROOT}
		PATH_SUFFIXES lib
		${_HWLOC_SEARCH_OPTS}
    )
  mark_as_advanced(HWLOC_LIBRARY)
  
  find_path(HWLOC_INCLUDE_DIR hwloc.h
    HINTS ${HWLOC_ROOT} 
		PATH_SUFFIXES include
    ${_HWLOC_SEARCH_OPTS}
    )
  mark_as_advanced(HWLOC_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(HWLOC_FIND_QUIETLY ON)

find_package_handle_standard_args(HWLOC 
	DEFAULT_MSG 
	HWLOC_INCLUDE_DIR 
	HWLOC_LIBRARY)

if(HWLOC_FOUND AND NOT TARGET hwloc)
  
  set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})
  
  set(HWLOC_LIBRARIES ${HWLOC_LIBRARY})
  
  # hwloc
	  
  add_library(hwloc UNKNOWN IMPORTED)
	 
  set_target_properties(hwloc PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIRS}")
    
	set_target_properties(hwloc PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${HWLOC_LIBRARY}")
    
endif()
