#
# Find the METIS includes and library
#
# This module uses
# METIS_ROOT
#
# This module defines
# METIS_FOUND
# METIS_INCLUDE_DIRS
# METIS_LIBRARIES
#
# Target metis 

if(NOT METIS_ROOT)
  set(METIS_ROOT $ENV{METIS_ROOT})
endif()

if(METIS_ROOT)
  set(_METIS_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_METIS_SEARCH_OPTS)
endif()

if(NOT METIS_FOUND) 

  find_library(METIS_LIBRARY 
    NAMES metis
		HINTS ${METIS_ROOT}
		PATH_SUFFIXES lib
		${_METIS_SEARCH_OPTS}
    )
  mark_as_advanced(METIS_LIBRARY)

  find_library(PARMETIS_LIBRARY 
    NAMES parmetis
		HINTS ${METIS_ROOT}
		PATH_SUFFIXES lib
		${_METIS_SEARCH_OPTS}
    )
  mark_as_advanced(PARMETIS_LIBRARY)
  
  find_path(METIS_INCLUDE_DIR metis.h
    HINTS ${METIS_ROOT} 
		PATH_SUFFIXES include
    ${_METIS_SEARCH_OPTS}
    )
  mark_as_advanced(METIS_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(METIS_FIND_QUIETLY ON)

find_package_handle_standard_args(METIS 
	DEFAULT_MSG 
	METIS_INCLUDE_DIR 
	METIS_LIBRARY)

if(METIS_FOUND AND NOT TARGET metis)
    
  set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
  
  if(PARMETIS_LIBRARY)
    set(METIS_LIBRARIES ${METIS_LIBRARY} ${PARMETIS_LIBRARY})
  else()
    set(METIS_LIBRARIES ${METIS_LIBRARY})
  endif()

  add_library(metis UNKNOWN IMPORTED)
	  
  set_target_properties(metis PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIRS}")
  
  set_target_properties(metis PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${METIS_LIBRARY}")

  if(PARMETIS_LIBRARY)
    set_property(TARGET metis APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "${PARMETIS_LIBRARY}")
  endif()

endif()
