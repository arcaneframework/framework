#
# Find the MUMPS includes and library
#
# This module uses
# MUMPS_ROOT
#
# This module defines
# MUMPS_FOUND
# MUMPS_INCLUDE_DIRS
# MUMPS_LIBRARIES
#
# Target mumps

message(status "MUMPS PACKAGE : ${MUMPS_ROOT}")
if(NOT MUMPS_ROOT)
  set(MUMPS_ROOT $ENV{MUMPS_ROOT})
endif()

message(status "MUMPS PACKAGE : ${MUMPS_ROOT}")
if(MUMPS_ROOT)
  set(_MUMPS_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_MUMPS_SEARCH_OPTS)
endif()

if(NOT MUMPS_FOUND)

  if(NOT WIN32)
    
    find_library(DMUMPS_LIBRARY
      NAMES dmumps
      HINTS ${MUMPS_ROOT} 
      PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(DMUMPS_LIBRARY)
    message(status "DMUMPS LIBRARY : ${DMUMPS_LIBRARY}")
    
    
    find_library(CMUMPS_LIBRARY
      NAMES cmumps
      HINTS ${MUMPS_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(CMUMPS_LIBRARY)
    message(status "CMUMPS LIBRARY : ${CMUMPS_LIBRARY}")
    
    find_library(SMUMPS_LIBRARY
      NAMES smumps
      HINTS ${MUMPS_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(SMUMPS_LIBRARY)
    message(status "SMUMPS LIBRARY : ${SMUMPS_LIBRARY}")
    
    find_library(ZMUMPS_LIBRARY
      NAMES zmumps
      HINTS ${MUMPS_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(ZMUMPS_LIBRARY)
    message(status "ZMUMPS LIBRARY : ${ZMUMPS_LIBRARY}")
    
    find_library(MUMPS_COMMON_LIBRARY
      NAMES mumps_common
      HINTS ${MUMPS_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(MUMPS_COMMON_LIBRARY)
    message(status "MUMPS COMMON LIBRARY : ${MUMPS_COMMON_LIBRARY}")
    
    find_library(MUMPS_PORD_LIBRARY
      NAMES pord
      HINTS ${MUMPS_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(MUMPS_PORD_LIBRARY)
    message(status "MUMPS PORD LIBRARY : ${MUMPS_PORD_LIBRARY}")
    
    
    find_library(ESMUMPS_LIBRARY
      NAMES esmumps
      HINTS $ENV{SCOTCH_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(ESMUMPS_LIBRARY)
    message(status "ESMUMPS LIBRARY : ${ESMUMPS_LIBRARY}")
    
    find_library(PTESMUMPS_LIBRARY
                 NAMES ptesmumps
                 HINTS $ENV{SCOTCH_ROOT} 
                 PATH_SUFFIXES lib
                 ${_MUMPS_SEARCH_OPTS}
                )
    mark_as_advanced(PTESMUMPS_LIBRARY)
    message(status "PTESMUMPS LIBRARY : ${PTESMUMPS_LIBRARY}")
    
    
    find_library(SCALAPACK_LIBRARY
                 NAMES scalapack
                 HINTS $ENV{MUMPS_ROOT} 
                 PATH_SUFFIXES lib
                 ${_MUMPS_SEARCH_OPTS})
      
    find_library(GFORTRAN_LIBRARY
      NAMES libgfortran.so
      HINTS $ENV{GFORTRAN_ROOT} 
      PATH_SUFFIXES lib64
      ${_MUMPS_SEARCH_OPTS})
    
  else()
    
    find_library(DMUMPS_LIBRARY
      NAMES MUMPS libdmumps
      HINTS ${MUMPS_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
      
    mark_as_advanced(MUMPS_LIBRARY)
    
    find_library(MUMPS_COMMON_LIBRARY
      NAMES libmumps_common
      HINTS ${MUMPS_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(MUMPS_COMMON_LIBRARY)
    
    
    find_library(MUMPS_PORD_LIBRARY
      NAMES libpord
      HINTS ${MUMPS_ROOT} 
		  PATH_SUFFIXES lib
      ${_MUMPS_SEARCH_OPTS}
      )
    mark_as_advanced(MUMPS_PORD_LIBRARY)
    
  endif()
 
  find_path(MUMPS_INCLUDE_DIR dmumps_c.h
    HINTS ${MUMPS_ROOT} 
		PATH_SUFFIXES include
    ${_MUMPS_SEARCH_OPTS}
    )
  mark_as_advanced(MUMPS_INCLUDE_DIR)
  
endif()
 
# pour limiter le mode verbose
set(MUMPS_FIND_QUIETLY ON)
set(MUMPS_COMMON_FIND_QUIETLY ON)
set(MUMPS_PORD_FIND_QUIETLY ON)

find_package_handle_standard_args(MUMPS
                                  DEFAULT_MSG 
                                  MUMPS_INCLUDE_DIR 
                                  DMUMPS_LIBRARY)
                                  
find_package_handle_standard_args(MUMPS_COMMON
                                  DEFAULT_MSG 
                                  MUMPS_INCLUDE_DIR 
                                  MUMPS_COMMON_LIBRARY)
                                  
find_package_handle_standard_args(MUMPS_PORD
                                  DEFAULT_MSG 
                                  MUMPS_INCLUDE_DIR 
                                  MUMPS_PORD_LIBRARY)

if(MUMPS_FOUND AND NOT TARGET mumps)
   
  set(MUMPS_INCLUDE_DIRS ${MUMPS_INCLUDE_DIR})
  
  set(MUMPS_LIBRARIES ${MUMPS_LIBRARY} ${MUMPS_COMMON_LIBRARY} ${MUMPS_PORD})
   
   add_library(mumps INTERFACE IMPORTED)
   
  add_library(dmumps UNKNOWN IMPORTED)
  set_target_properties(dmumps PROPERTIES 
                         INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIRS}")
  set_target_properties(dmumps PROPERTIES
                         IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                         IMPORTED_LOCATION "${DMUMPS_LIBRARY}")
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "dmumps")
    
    
  add_library(cmumps UNKNOWN IMPORTED)
  set_target_properties(cmumps PROPERTIES 
                        INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIRS}")
  set_target_properties(cmumps PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${CMUMPS_LIBRARY}")
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "cmumps")
    
    
  add_library(smumps UNKNOWN IMPORTED)
  set_target_properties(smumps PROPERTIES 
                        INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIRS}")
  set_target_properties(smumps PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${SMUMPS_LIBRARY}") 
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "smumps")
  
  
  add_library(zmumps UNKNOWN IMPORTED)
  set_target_properties(zmumps PROPERTIES 
                        INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIRS}")
  set_target_properties(zmumps PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${ZMUMPS_LIBRARY}")
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "zmumps")
    
  add_library(mumps_common UNKNOWN IMPORTED)
  set_target_properties(mumps_common PROPERTIES 
                        INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIRS}")
  set_target_properties(mumps_common PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${MUMPS_COMMON_LIBRARY}")
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "mumps_common")
    
  add_library(pord UNKNOWN IMPORTED)
  
  set_target_properties(pord PROPERTIES 
                        INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIRS}")
     
  set_target_properties(pord PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${MUMPS_PORD_LIBRARY}")
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "pord")
    
    
  add_library(esmumps UNKNOWN IMPORTED)
  set_target_properties(esmumps PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${ESMUMPS_LIBRARY}") 
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "esmumps")
    
  add_library(ptesmumps UNKNOWN IMPORTED)
  set_target_properties(ptesmumps PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${PTESMUMPS_LIBRARY}") 
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "ptesmumps")

  add_library(scalapack UNKNOWN IMPORTED)
  set_target_properties(scalapack PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${SCALAPACK_LIBRARY}")
    
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "scalapack")
  #if(TARGET mkl)
  #  set_property(TARGET mumps APPEND PROPERTY
  #               INTERFACE_LINK_LIBRARIES "mkl")
  #endif()
               
  add_library(gfortran UNKNOWN IMPORTED)
  set_target_properties(gfortran PROPERTIES
                        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                        IMPORTED_LOCATION "${GFORTRAN_LIBRARY}")
    
  set_property(TARGET mumps APPEND PROPERTY
               INTERFACE_LINK_LIBRARIES "gfortran")
  
    
endif()
