#
# Find the HPDDM includes and library
# Permet de linker Hpddm et ses dépendances : Mumps et Arpack
#
# This module uses
# HPDDM_ROOT
# MUMPS_ROOT
# ARPACK_ROOT
#
# This module defines
# HPDDM_FOUND
# HPDDM_INCLUDE_DIRS
#
# Target hpddm

if(NOT HPDDM_ROOT)
  set(HPDDM_ROOT $ENV{HPDDM_ROOT})
endif()

if(HPDDM_ROOT)
  set(_HPDDM_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_HPDDM_SEARCH_OPTS)
endif()

if(NOT HPDDM_FOUND)

       
#Hpddm
find_path(HPDDM_INCLUDE_DIR HPDDM.hpp
    HINTS ${HPDDM_ROOT} 
    PATH_SUFFIXES include
    ${_HPDDM_SEARCH_OPTS}
    )
  mark_as_advanced(HPDDM_INCLUDE_DIR)
  
endif()
 
# pour limiter le mode verbose
set(HPDDM_FIND_QUIETLY ON)

find_package_handle_standard_args(HPDDM
                                  DEFAULT_MSG 
                                  HPDDM_INCLUDE_DIR)

if(HPDDM_FOUND AND NOT TARGET hpddm)

  set(HPDDM_INCLUDE_DIRS ${HPDDM_INCLUDE_DIR})
  


  #Construction de Hpddm
  add_library(hpddm INTERFACE IMPORTED)
  set_target_properties(hpddm PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${HPDDM_INCLUDE_DIRS}")
    
  if(TARGET mumps)
    set_property(TARGET hpddm APPEND PROPERTY 
                 INTERFACE_LINK_LIBRARIES "mumps")
  endif()
  
  if(TARGET arpack)
    set_property(TARGET hpddm APPEND PROPERTY 
                 INTERFACE_LINK_LIBRARIES "arpack")
  endif()
     
  #target_compile_definitions(Hpddm PUBLIC -DDMUMPS -DMUMPSSUB)
    
endif()
