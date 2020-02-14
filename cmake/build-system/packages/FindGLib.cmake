#
# Find the GLIB includes and library
#
# This module uses
# GLIB_ROOT
#
# This module defines
# GLIB_FOUND
# GLIB_INCLUDE_DIRS
# GLIB_LIBRARIES
#
# Target glib 

if(NOT GLIB_ROOT)
  set(GLIB_ROOT $ENV{GLIB_ROOT})
endif()

if(GLIB_ROOT)
  set(_GLIB_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_GLIB_SEARCH_OPTS)
endif()

if(NOT GLIB_FOUND)

  find_library(GLIB_LIBRARY 
    NAMES glib-2.0
    HINTS ${GLIB_ROOT} 
    PATH_SUFFIXES lib
    ${_GLIB_SEARCH_OPTS}
    )
  mark_as_advanced(GLIB_LIBRARY)
  
  find_library(GTHREAD_LIBRARY 
    NAMES gthread-2.0
    HINTS ${GLIB_ROOT} 
    PATH_SUFFIXES lib
    ${_GLIB_SEARCH_OPTS}
    )
  mark_as_advanced(GTHREAD_LIBRARY)
  
  find_library(GMODULE_LIBRARY
    NAMES gmodule-2.0
    HINTS ${GLIB_ROOT} 
    PATH_SUFFIXES lib
    ${_GLIB_SEARCH_OPTS}
    )
  mark_as_advanced(GMODULE_LIBRARY)

  find_path(GLIB_INCLUDE_DIR glib.h
    HINTS ${GLIB_ROOT} 
    PATH_SUFFIXES include include/glib-2.0
    ${_GLIB_SEARCH_OPTS}
    )
  mark_as_advanced(GLIB_INCLUDE_DIR)
 
  get_filename_component(GLIB_ROOT_PATH ${GLIB_LIBRARY} PATH)
  
  find_path(GLIBCONFIG_INCLUDE_DIR glibconfig.h
    HINTS ${GLIB_ROOT_PATH} 
    PATH_SUFFIXES include glib-2.0/include
    ${_GLIB_SEARCH_OPTS}
    )
  mark_as_advanced(GLIB_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(GLIB_FIND_QUIETLY ON)

find_package_handle_standard_args(GLIB
  DEFAULT_MSG 
  GLIB_INCLUDE_DIR
  GLIBCONFIG_INCLUDE_DIR
  GLIB_LIBRARY
  GTHREAD_LIBRARY
  GMODULE_LIBRARY
  )

if(GLIB_FOUND AND NOT TARGET glib)

  set(GLIB_INCLUDE_DIRS ${GLIB_INCLUDE_DIR}
                        ${GLIBCONFIG_INCLUDE_DIR})
  
  set(GLIB_LIBRARIES ${GLIB_LIBRARY}
                     ${GTHREAD_LIBRARY}
                     ${GMODULE_LIBRARY})
  
  # glib_main
  
  add_library(glib_main UNKNOWN IMPORTED)
  
  set_target_properties(glib_main PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${GLIB_INCLUDE_DIRS}")
  
  set_target_properties(glib_main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${GLIB_LIBRARY}")
  
  # gthread
  
  add_library(gthread UNKNOWN IMPORTED)
  
  set_target_properties(gthread PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${GLIB_INCLUDE_DIRS}")
  
  set_target_properties(gthread PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${GTHREAD_LIBRARY}")
  
  # gmodule
  
  add_library(gmodule UNKNOWN IMPORTED)
  
  set_target_properties(gmodule PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${GLIB_INCLUDE_DIRS}")
  
  set_target_properties(gmodule PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${GMODULE_LIBRARY}")
  
  # glib
  
  add_library(glib INTERFACE IMPORTED)
  
  set_property(TARGET glib APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "glib_main")

  set_property(TARGET glib APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "gmodule")

  set_property(TARGET glib APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "gthread")

endif()

