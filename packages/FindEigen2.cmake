#
# Find the EIGEN2 includes and library
#
# This module uses
# EIGEN2_ROOT
#
# This module defines
# EIGEN2_FOUND
# EIGEN2_INCLUDE_DIRS
#
# Target eigen2

if(NOT EIGEN2_ROOT)
  set(EIGEN2_ROOT $ENV{EIGEN2_ROOT})
endif()

if(EIGEN2_ROOT)
  set(_EIGEN2_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_EIGEN2_SEARCH_OPTS)
endif()

if(NOT EIGEN2_FOUND) 

  find_path(EIGEN2_INCLUDE_DIR Eigen
    HINTS ${EIGEN2_ROOT} 
	PATH_SUFFIXES include/eigen2
    ${_EIGEN2_SEARCH_OPTS}
    )
  mark_as_advanced(EIGEN2_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(EIGEN2_FIND_QUIETLY ON)

find_package_handle_standard_args(EIGEN2 
	DEFAULT_MSG 
	EIGEN2_INCLUDE_DIR)

if(EIGEN2_FOUND AND NOT TARGET eigen2)

  set(EIGEN2_INCLUDE_DIRS ${EIGEN2_INCLUDE_DIR})
  
  add_library(eigen2 INTERFACE IMPORTED)
	  
  set_target_properties(eigen2 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN2_INCLUDE_DIRS}")
    
endif()
