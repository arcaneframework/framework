#
# Find the BINUTILS includes and library
#
# This module uses
# BINUTILS_ROOT
#
# This module defines
# BINUTILS_FOUND
# BINUTILS_INCLUDE_DIRS
# BINUTILS_LIBRARIES
#
# Target binutils

if(NOT BINUTILS_ROOT)
  set(BINUTILS_ROOT $ENV{BINUTILS_ROOT})
endif()

if(BINUTILS_ROOT)
  set(_BINUTILS_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_BINUTILS_SEARCH_OPTS)
endif()

if(NOT BINUTILS_FOUND) 

  find_library(BINUTILS_IBERTY_LIBRARY 
    NAMES iberty
		HINTS ${BINUTILS_ROOT}
		PATH_SUFFIXES lib
		${_BINUTILS_SEARCH_OPTS}
    )
  mark_as_advanced(BINUTILS_IBERTY_LIBRARY)
  
  find_library(BINUTILS_BFD_LIBRARY 
	  NAMES bfd
		HINTS ${BINUTILS_ROOT}
		PATH_SUFFIXES lib
		${_BINUTILS_SEARCH_OPTS}
    )
  mark_as_advanced(BINUTILS_BFD_LIBRARY)
  
  find_path(BINUTILS_INCLUDE_DIR libiberty.h
    HINTS ${BINUTILS_ROOT} 
		PATH_SUFFIXES include
    ${_BINUTILS_SEARCH_OPTS}
    )
  mark_as_advanced(BINUTILS_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(BINUTILS_FIND_QUIETLY ON)

find_package_handle_standard_args(BINUTILS 
	DEFAULT_MSG 
	BINUTILS_INCLUDE_DIR 
	BINUTILS_IBERTY_LIBRARY
  BINUTILS_BFD_LIBRARY)

if(BINUTILS_FOUND AND NOT TARGET binutils)

  set(BINUTILS_INCLUDE_DIRS ${BINUTILS_INCLUDE_DIR})
  
  set(BINUTILS_LIBRARIES ${BINUTILS_IBERTY_LIBRARY}
                         ${BINUTILS_BFD_LIBRARY})
  
  # BINUTILS main
	  
  add_library(binutils_iberty UNKNOWN IMPORTED)
	  
  set_target_properties(binutils_iberty PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${BINUTILS_INCLUDE_DIRS}")
    
	set_target_properties(binutils_iberty PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${BINUTILS_IBERTY_LIBRARY}")
    
  # BINUTILS utils
    
	add_library(binutils_bfd UNKNOWN IMPORTED)
	  
	set_target_properties(binutils_bfd PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${BINUTILS_BFD_LIBRARY}")
    
  # BINUTILS
    
	add_library(binutils INTERFACE IMPORTED)
	  
	set_property(TARGET binutils APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "binutils_iberty")

  set_property(TARGET binutils APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "binutils_bfd")
  
endif()
