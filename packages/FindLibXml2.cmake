#
# Find the LIBXML2 includes and library
#
# This module uses
# LIBXML2_ROOT
#
# This module defines
# LIBXML2_FOUND
# LIBXML2_INCLUDE_DIRS
# LIBXML2_LIBRARIES
#
# Target libxml2

find_package(LibXml2)
arccon_return_if_package_found(LibXml2)

message(STATUS "Warning: using obsolete FindLibXml2")

if(NOT LIBXML2_ROOT)
  set(LIBXML2_ROOT $ENV{LIBXML2_ROOT})
endif()

if(LIBXML2_ROOT)
  set(_LIBXML2_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_LIBXML2_SEARCH_OPTS)
endif()

if(NOT LIBXML2_FOUND) 

  find_library(LIBXML2_LIBRARY 
    NAMES xml2
		HINTS ${LIBXML2_ROOT}
		PATH_SUFFIXES lib
		${_LIBXML2_SEARCH_OPTS}
    )
  mark_as_advanced(LIBXML2_LIBRARY)
  
  find_path(LIBXML2_INCLUDE_DIR libxml/parser.h
    HINTS ${LIBXML2_ROOT} 
		PATH_SUFFIXES include/libxml2
    ${_LIBXML2_SEARCH_OPTS}
    )
  mark_as_advanced(LIBXML2_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(Libxml2_FIND_QUIETLY ON)

find_package_handle_standard_args(Libxml2 
	DEFAULT_MSG 
	LIBXML2_INCLUDE_DIR 
	LIBXML2_LIBRARY)

if(LIBXML2_FOUND AND NOT TARGET libxml2)

  set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR})
   
  set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARY})
  
  # libxml2 
	  
  add_library(libxml2 UNKNOWN IMPORTED)
	  
  set_target_properties(libxml2 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIRS}")
    
  set_target_properties(libxml2 PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${LIBXML2_LIBRARY}")
   
endif()
