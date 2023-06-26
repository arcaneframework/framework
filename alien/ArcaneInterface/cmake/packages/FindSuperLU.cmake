#
# Find the SUPERLU includes and library
#
# This module uses
# SUPERLU_ROOT
#
# This module defines
# SUPERLU_FOUND
# SUPERLU_INCLUDE_DIRS
# SUPERLU_LIBRARIES
#
# Target superlu 

if(NOT SUPERLU_ROOT)
  set(SUPERLU_ROOT $ENV{SUPERLU_ROOT})
endif()


if(NOT SUPERLU_VERSION)
  set(SUPERLU_VERSION $ENV{SUPERLU_VERSION})
endif()

if(SUPERLU_ROOT)
  set(_SUPERLU_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_SUPERLU_SEARCH_OPTS)
endif()

if(NOT SUPERLU_FOUND)

  find_library(SUPERLU_LIBRARY
    NAMES superlu${SUPERLU_VERSION}
    HINTS ${SUPERLU_ROOT} 
    PATH_SUFFIXES lib
    ${_SUPERLU_SEARCH_OPTS}
    )
  mark_as_advanced(SUPERLU_LIBRARY)

  find_path(SUPERLU_INCLUDE_DIR slu_cdefs.h
    HINTS ${SUPERLU_ROOT} 
    PATH_SUFFIXES include
    ${_SUPERLU_SEARCH_OPTS}
    )
  mark_as_advanced(SUPERLU_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(SUPERLU_FIND_QUIETLY ON)

find_package_handle_standard_args(SUPERLU
  DEFAULT_MSG 
  SUPERLU_INCLUDE_DIR
  SUPERLU_LIBRARY
  )

if(SUPERLU_FOUND AND NOT TARGET superlu)

  set(SUPERLU_INCLUDE_DIRS ${SUPERLU_INCLUDE_DIR})
  
  set(SUPERLU_LIBRARIES ${SUPERLU_LIBRARY})

  add_library(superlu UNKNOWN IMPORTED)
  
  set_target_properties(superlu PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${SUPERLU_INCLUDE_DIRS}")
    
  set_target_properties(superlu PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${SUPERLU_LIBRARY}")
  
  if(TARGET mkl)
     set_property(TARGET superlu APPEND PROPERTY 
                  INTERFACE_LINK_LIBRARIES "mkl")
  endif()
  
  if(TARGET fortran)
     set_property(TARGET superlu APPEND PROPERTY 
                  INTERFACE_LINK_LIBRARIES "fortran")
  endif()
endif()

