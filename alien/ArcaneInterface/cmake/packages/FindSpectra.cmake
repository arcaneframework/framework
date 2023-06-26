#
# Find SPECTRA includes
#
# This module defines
# SPECTRA_INCLUDE_DIRS, where to find headers,
# SPECTRA_LIBRARIES, the libraries to link against to use eigen.
# SPECTRA_FOUND If false, do not try to use eigen.


if(NOT SPECTRA_ROOT)
  set(SPECTRA_ROOT $ENV{SPECTRA_ROOT})
  if(NOT SPECTRA_ROOT)
    set(SPECTRA_ROOT $ENV{SPECTRALIB_ROOT})
  endif()
endif()

if(SPECTRA_ROOT)
  set(_SPECTRA_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_SPECTRA_SEARCH_OPTS)
endif()

if(NOT SPECTRA_FOUND) 

  find_path(SPECTRA_INCLUDE_DIR SymGEigsSolver.h
    HINTS ${SPECTRA_ROOT} 
		PATH_SUFFIXES include include
    ${_SPECTRA_SEARCH_OPTS}
    )
  mark_as_advanced(SPECTRA_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(SPECTRA_FIND_QUIETLY ON)

find_package_handle_standard_args(SPECTRA 
	DEFAULT_MSG 
	SPECTRA_INCLUDE_DIR)

if(SPECTRA_FOUND AND NOT TARGET spectra)

  set(SPECTRA_INCLUDE_DIRS ${SPECTRA_INCLUDE_DIR})
    
  add_library(spectra INTERFACE IMPORTED)
	  
  set_target_properties(spectra PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${SPECTRA_INCLUDE_DIRS}")
    
endif()
