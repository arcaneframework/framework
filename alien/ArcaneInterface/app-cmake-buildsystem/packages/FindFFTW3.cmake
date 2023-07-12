#
# Find the FFTW3 includes and library
#
# This module uses
# FFTW3_ROOT
#
# This module defines
# FFTW3_FOUND
# FFTW3_INCLUDE_DIRS
# FFTW3_LIBRARIES
#
# Target fftw3 

if(NOT FFTW3_ROOT)
  set(FFTW3_ROOT $ENV{FFTW3_ROOT})
endif()

if(FFTW3_ROOT)
  set(_FFTW3_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_FFTW3_SEARCH_OPTS)
endif()

if(NOT FFTW3_FOUND)

  find_library(FFTW3_LIBRARY
    NAMES fftw3
    HINTS ${FFTW3_ROOT} 
    PATH_SUFFIXES lib
    ${_FFTW3_SEARCH_OPTS}
    )
  mark_as_advanced(FFTW3_LIBRARY)

  find_library(FFTW3_MPI_LIBRARY
    NAMES fftw3_mpi
    HINTS ${FFTW3_ROOT} 
    PATH_SUFFIXES lib
    ${_FFTW3_SEARCH_OPTS}
    )
  mark_as_advanced(FFTW3_LIBRARY)

  find_path(FFTW3_INCLUDE_DIR fftw3.h
    HINTS ${FFTW3_ROOT} 
    PATH_SUFFIXES include
    ${_FFTW3_SEARCH_OPTS}
    )
  mark_as_advanced(FFTW3_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(FFTW3_FIND_QUIETLY ON)

find_package_handle_standard_args(FFTW3
  DEFAULT_MSG 
  FFTW3_INCLUDE_DIR
  FFTW3_LIBRARY
  FFTW3_MPI_LIBRARY
  )

if(FFTW3_FOUND AND NOT TARGET fftw3)

  set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
  
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})

  add_library(fftw3_main UNKNOWN IMPORTED)
  
  set_target_properties(fftw3_main PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}")
    
  set_target_properties(fftw3_main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${FFTW3_LIBRARY}")
  
  add_library(fftw3_mpi UNKNOWN IMPORTED)
  
  set_target_properties(fftw3_mpi PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}")
    
  set_target_properties(fftw3_mpi PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${FFTW3_MPI_LIBRARY}")
  
  add_library(fftw3 INTERFACE IMPORTED)
	  
  set_property(TARGET fftw3 APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "fftw3_main;fftw3_mpi")
  
endif()

