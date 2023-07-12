#
# Find the NUMA includes and library
#
# This module uses
# NUMA_ROOT
#
# This module defines
# NUMA_FOUND
# NUMA_INCLUDE_DIRS
# NUMA_LIBRARIES
#
# Target numa 

if(NOT NUMA_ROOT)
  set(NUMA_ROOT $ENV{NUMA_ROOT})
endif()

if(NUMA_ROOT)
  set(_NUMA_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_NUMA_SEARCH_OPTS)
endif()

if(NOT NUMA_FOUND) 

  find_library(NUMA_LIBRARY 
    NAMES numa
		HINTS ${NUMA_ROOT}
		PATH_SUFFIXES lib64
		${_NUMA_SEARCH_OPTS}
    )
  mark_as_advanced(NUMA_LIBRARY)
  
  find_path(NUMA_INCLUDE_DIR numa.h
    HINTS ${NUMA_ROOT} 
		PATH_SUFFIXES include
    ${_NUMA_SEARCH_OPTS}
    )
  mark_as_advanced(NUMA_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(NUMA_FIND_QUIETLY ON)

find_package_handle_standard_args(NUMA 
	DEFAULT_MSG 
	NUMA_INCLUDE_DIR 
	NUMA_LIBRARY)

if(NUMA_FOUND AND NOT TARGET numa)
    
  set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR})
  
  set(NUMA_LIBRARIES ${NUMA_LIBRARY})
  	  
  add_library(numa UNKNOWN IMPORTED)
	  
  set_target_properties(numa PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIRS}")
    
	set_target_properties(numa PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${NUMA_LIBRARY}")
    
endif()


