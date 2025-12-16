#
# Find the EIGEN3 includes and library
#
# This module uses
# EIGEN3_ROOT
#
# This module defines
# EIGEN3_FOUND
# EIGEN3_INCLUDE_DIRS
#
# Target eigen3 

if(NOT EIGEN3_ROOT)
  set(EIGEN3_ROOT $ENV{EIGEN_ROOT})
endif()

if(EIGEN3_ROOT)
  set(_EIGEN3_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_EIGEN3_SEARCH_OPTS)
endif()

if(NOT Eigen3_FOUND) 

  find_path(EIGEN3_INCLUDE_DIR Eigen
    HINTS ${EIGEN3_ROOT} 
    PATH_SUFFIXES include include/eigen3
    ${_EIGEN3_SEARCH_OPTS}
    )
  mark_as_advanced(EIGEN3_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(EIGEN3_FIND_QUIETLY ON)

find_package_handle_standard_args(Eigen3 
	DEFAULT_MSG 
	EIGEN3_INCLUDE_DIR)

if(Eigen3_FOUND AND NOT TARGET eigen3)

  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
    
  add_library(eigen3 INTERFACE IMPORTED)
	  
  set_target_properties(eigen3 PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIRS}")
    
endif()
