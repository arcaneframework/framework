#
# Find the HYPRE includes and library
#
# This module uses
# HYPRE_ROOT
#
# This module defines
# HYPRE_FOUND
# HYPRE_INCLUDE_DIRS
# HYPRE_LIBRARIES
#
# Target hypre

if(NOT HYPRE_ROOT)
  set(HYPRE_ROOT $ENV{HYPRE_ROOT})
endif()

if(HYPRE_ROOT)
  set(_HYPRE_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_HYPRE_SEARCH_OPTS)
endif()

if(NOT HYPRE_FOUND)

  if(NOT WIN32)
    
    find_library(HYPRE_LIBRARY
      NAMES HYPRE
      HINTS ${HYPRE_ROOT} 
      PATH_SUFFIXES lib lib64
      ${_HYPRE_SEARCH_OPTS}
      )
    mark_as_advanced(HYPRE_LIBRARY)
    
  else()
    
    find_library(HYPRE_LIBRARY
      NAMES libHYPRE
      HINTS ${HYPRE_ROOT} 
      PATH_SUFFIXES lib lib64
      ${_HYPRE_SEARCH_OPTS}
      )
    mark_as_advanced(HYPRE_LIBRARY)
    
  endif()
 
  find_path(HYPRE_INCLUDE_DIR HYPRE.h
    HINTS ${HYPRE_ROOT} 
    PATH_SUFFIXES include
    ${_HYPRE_SEARCH_OPTS}
    )
  mark_as_advanced(HYPRE_INCLUDE_DIR)
  
endif()
 
# pour limiter le mode verbose
set(HYPRE_FIND_QUIETLY ON)

find_package_handle_standard_args(HYPRE
	DEFAULT_MSG 
	HYPRE_INCLUDE_DIR 
	HYPRE_LIBRARY)

if(HYPRE_FOUND AND NOT TARGET hypre)
   
  set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})
  
  set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})
   
  add_library(hypre UNKNOWN IMPORTED)
	  
  set_target_properties(hypre PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}")
     
	set_target_properties(hypre PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${HYPRE_LIBRARY}")
    
endif()
