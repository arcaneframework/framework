#
# Find the MTL includes and library
#
# This module uses
# MTL_ROOT
#
# This module defines
# MTL_FOUND
# MTL_INCLUDE_DIRS
#
# Target mtl 

if(NOT MTL_ROOT)
  set(MTL_ROOT $ENV{MTL_ROOT})
endif()

if(MTL_ROOT)
  set(_MTL_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_MTL_SEARCH_OPTS)
endif()

if(NOT MTL_FOUND) 

  find_path(MTL_INCLUDE_DIR boost/numeric/mtl/mtl.hpp
    HINTS ${MTL_ROOT} 
		PATH_SUFFIXES include
    ${_MTL_SEARCH_OPTS}
    )
  mark_as_advanced(MTL_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(MTL_FIND_QUIETLY ON)

find_package_handle_standard_args(MTL 
	DEFAULT_MSG 
	MTL_INCLUDE_DIR)

if(MTL_FOUND AND NOT TARGET mtl)
    
  set(MTL_INCLUDE_DIRS ${MTL_INCLUDE_DIR})
  
  set(MTL_FLAGS USE_MTL4)

  add_library(mtl INTERFACE IMPORTED)
	  
  set_target_properties(mtl PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${MTL_INCLUDE_DIRS}")

  set_target_properties(mtl PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${MTL_FLAGS}")
    
endif()
