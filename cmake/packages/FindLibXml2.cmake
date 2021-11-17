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

if(NOT LIBXML2_ROOT)
  set(LIBXML2_ROOT $ENV{LIBXML2_ROOT})
endif()

if(LIBXML2_ROOT)
  set(_LIBXML2_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_LIBXML2_SEARCH_OPTS)
endif()

if(NOT LIBXML2_FOUND) 

  if(WIN32)
    find_library(LIBXML2_LIBRARY 
      NAMES xml2
      HINTS ${LIBXML2_ROOT}
      PATH_SUFFIXES lib VC12/lib
      ${_LIBXML2_SEARCH_OPTS}
      )
    mark_as_advanced(LIBXML2_LIBRARY)
  
    find_path(LIBXML2_INCLUDE_DIR libxml/parser.h
      HINTS ${LIBXML2_ROOT} 
      PATH_SUFFIXES include
      ${_LIBXML2_SEARCH_OPTS}
      )
    mark_as_advanced(LIBXML2_INCLUDE_DIR)
  
    find_library(ICONV_LIBRARY 
      NAMES iconv
      HINTS ${ICONV_ROOT}
      PATH_SUFFIXES lib VC12/lib
      ${_LIBXML2_SEARCH_OPTS}
      )
    mark_as_advanced(ICONV_LIBRARY)
  
    find_path(ICONV_INCLUDE_DIR iconv.h
      HINTS ${ICONV_ROOT} 
      PATH_SUFFIXES include
      ${_LIBXML2_SEARCH_OPTS}
      )
    mark_as_advanced(ICONV_INCLUDE_DIR)
  
  else()
  
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
  
endif()

# pour limiter le mode verbose
set(Libxml2_FIND_QUIETLY ON)

find_package_handle_standard_args(Libxml2 
	DEFAULT_MSG 
	LIBXML2_INCLUDE_DIR 
	LIBXML2_LIBRARY)

if(LIBXML2_FOUND AND NOT TARGET libxml2)

  set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR} ${ICONV_INCLUDE_DIR})
   
  set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARY} ${ICONV_LIBRARY})
  
  # libxml2 

  if(WIN32)
	  add_library(iconv UNKNOWN IMPORTED)
	  
  	set_target_properties(iconv PROPERTIES 
    	INTERFACE_INCLUDE_DIRECTORIES "${ICONV_INCLUDE_DIRS}")
    
  	set_target_properties(iconv PROPERTIES
    	IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    	IMPORTED_LOCATION "${ICONV_LIBRARY}")
  	endif(WIN32)
  add_library(libxml2 UNKNOWN IMPORTED)
	  
  set_target_properties(libxml2 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIRS}")
    
  set_target_properties(libxml2 PROPERTIES
     IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
     IMPORTED_LOCATION "${LIBXML2_LIBRARY}")
  
  if(WIN32)
    set_property(TARGET libxml2 APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "iconv")
  endif()
  
  add_definitions(-DUSE_XML2)
     
endif()
