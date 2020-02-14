#
# Find the UMFPACK includes and library
#
# This module uses
# UMFPACK_ROOT
#
# This module defines
# UMFPACK_FOUND
# UMFPACK_INCLUDE_DIRS
# UMFPACK_LIBRARIES
#
# Target umfpack 

if(NOT UMFPACK_ROOT)
  set(UMFPACK_ROOT $ENV{UMFPACK_ROOT})
endif()

if(UMFPACK_ROOT)
  set(_UMFPACK_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_UMFPACK_SEARCH_OPTS)
endif()

if(NOT UMFPACK_FOUND) 

  find_library(UMFPACK_CCOLAMD_LIBRARY 
    NAMES ccolamd
    HINTS ${UMFPACK_ROOT}
    PATH_SUFFIXES lib
    ${_UMFPACK_SEARCH_OPTS}
    )
  mark_as_advanced(UMFPACK_CCOLAMD_LIBRARY)
  
  find_library(UMFPACK_COLAMD_LIBRARY 
    NAMES colamd
    HINTS ${UMFPACK_ROOT}
    PATH_SUFFIXES lib
    ${_UMFPACK_SEARCH_OPTS}
    )
  mark_as_advanced(UMFPACK_COLAMD_LIBRARY)
  
  find_library(UMFPACK_CAMD_LIBRARY 
    NAMES camd
    HINTS ${UMFPACK_ROOT}
    PATH_SUFFIXES lib
    ${_UMFPACK_SEARCH_OPTS}
    )
  mark_as_advanced(UMFPACK_CAMD_LIBRARY)
  
  find_library(UMFPACK_AMD_LIBRARY 
    NAMES amd
    HINTS ${UMFPACK_ROOT}
    PATH_SUFFIXES lib
    ${_UMFPACK_SEARCH_OPTS}
    )
  mark_as_advanced(UMFPACK_AMD_LIBRARY)
  
  find_library(UMFPACK_LIBRARY 
    NAMES umfpack
    HINTS ${UMFPACK_ROOT}
    PATH_SUFFIXES lib
    ${_UMFPACK_SEARCH_OPTS}
    )
  mark_as_advanced(UMFPACK_LIBRARY)
  
  find_library(UMFPACK_CHOLDMOD_LIBRARY 
    NAMES cholmod
    HINTS ${UMFPACK_ROOT}
    PATH_SUFFIXES lib
    ${_UMFPACK_SEARCH_OPTS}
    )
  mark_as_advanced(UMFPACK_CHOLDMOD_LIBRARY)
  
  find_path(UMFPACK_INCLUDE_DIR umfpack.h
    HINTS ${UMFPACK_ROOT} 
    PATH_SUFFIXES include
    ${_UMFPACK_SEARCH_OPTS}
    )
  mark_as_advanced(UMFPACK_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(UMFPACK_FIND_QUIETLY ON)

find_package_handle_standard_args(UMFPACK DEFAULT_MSG
  UMFPACK_INCLUDE_DIR 
  UMFPACK_LIBRARY
  UMFPACK_CHOLDMOD_LIBRARY
  UMFPACK_AMD_LIBRARY
  UMFPACK_CAMD_LIBRARY
  UMFPACK_COLAMD_LIBRARY
  UMFPACK_CCOLAMD_LIBRARY)

if(UMFPACK_FOUND AND NOT TARGET umfpack)

  set(UMFPACK_INCLUDE_DIRS ${UMFPACK_INCLUDE_DIR})
  
  set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARY} 
                        ${UMFPACK_CHOLDMOD_LIBRARY}
                        ${UMFPACK_AMD_LIBRARY}
                        ${UMFPACK_CAMD_LIBRARY}
                        ${UMFPACK_COLAMD_LIBRARY}
                        ${UMFPACK_CCOLAMD_LIBRARY})

  # umfpack main
    
  add_library(umfpack_main SHARED IMPORTED)
    
  set_target_properties(umfpack_main PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${UMFPACK_INCLUDE_DIRS}")

  set_target_properties(umfpack_main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${UMFPACK_LIBRARY}" 
    IMPORTED_NO_SONAME ON)
    
  # umfpack cholmod
    
  add_library(umfpack_cholmod SHARED IMPORTED)
    
  set_target_properties(umfpack_cholmod PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${UMFPACK_CHOLDMOD_LIBRARY}"
    IMPORTED_NO_SONAME ON)
    
  # umfpack amd
    
  add_library(umfpack_amd SHARED IMPORTED)
    
  set_target_properties(umfpack_amd PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${UMFPACK_AMD_LIBRARY}"
    IMPORTED_NO_SONAME ON)

  # umfpack camd
    
  add_library(umfpack_camd SHARED IMPORTED)
    
  set_target_properties(umfpack_camd PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${UMFPACK_CAMD_LIBRARY}"
    IMPORTED_NO_SONAME ON)
    
  # umfpack colamd
    
  add_library(umfpack_colamd SHARED IMPORTED)
    
  set_target_properties(umfpack_colamd PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${UMFPACK_COLAMD_LIBRARY}"
    IMPORTED_NO_SONAME ON)

  # umfpack ccolamd
    
  add_library(umfpack_ccolamd SHARED IMPORTED)
    
  set_target_properties(umfpack_ccolamd PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${UMFPACK_CCOLAMD_LIBRARY}"
    IMPORTED_NO_SONAME ON)
    
  # umfpack
    
  add_library(umfpack INTERFACE IMPORTED)
    
  set_property(TARGET umfpack APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "umfpack_main")

  set_property(TARGET umfpack APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "umfpack_cholmod")

  set_property(TARGET umfpack APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "umfpack_amd")

  set_property(TARGET umfpack APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "umfpack_camd")

  set_property(TARGET umfpack APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "umfpack_colamd")

  set_property(TARGET umfpack APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "umfpack_ccolamd")
 
  set_target_properties(umfpack PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "MTL_HAS_UMFPACK")

endif()
