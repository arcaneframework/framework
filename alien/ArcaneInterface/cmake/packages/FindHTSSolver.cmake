#
# Find the HTSSOLVER includes and library
#
# This module uses
# HTSSOLVER_ROOT
#
# This module defines
# HTSSOLVER_FOUND
# HTSSOLVER_INCLUDE_DIRS
# HTSSOLVER_LIBRARIES
#
# Target mcgsolver
 
if(NOT HTSSOLVER_ROOT)
  set(HTSSOLVER_ROOT $ENV{HTSSOLVER_ROOT})
endif()

if(HTSSOLVER_ROOT)
  set(_HTSSOLVER_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_HTSSOLVER_SEARCH_OPTS)
endif()

if(NOT HTSSOLVER_FOUND) 

  find_library(HTSSOLVER_LIBRARY 
               NAMES HARTSSolver
               HINTS ${HTSSOLVER_ROOT}
               PATH_SUFFIXES lib
               ${_HTSSOLVER_SEARCH_OPTS}
               )
  mark_as_advanced(HTSSOLVER_LIBRARY)
  
  find_library(HTSSOLVERCORE_LIBRARY 
               NAMES HARTSSolverCore
               HINTS ${HTSSOLVER_ROOT}
               PATH_SUFFIXES lib
               ${_HTSSOLVER_SEARCH_OPTS}
               )
  mark_as_advanced(HTSSOLVERCORE_LIBRARY)
  
  find_library(HTSSOLVERSIMD_LIBRARY 
               NAMES HARTSSolver_simd
               HINTS ${HTSSOLVER_ROOT}
               PATH_SUFFIXES lib
               ${_HTSSOLVER_SEARCH_OPTS}
               )
  mark_as_advanced(HTSSOLVERSIMD_LIBRARY)
  
  find_path(HTSSOLVER_INCLUDE_DIR HARTSSolver/Driver/HTSSolver.h
    HINTS ${HTSSOLVER_ROOT} 
		PATH_SUFFIXES include
    ${_HTSSOLVER_SEARCH_OPTS}
    )
  mark_as_advanced(HTSSOLVER_INCLUDE_DIR)
  message("HTSSOLVE : ${HTSSOLVER_INCLUDE_DIR} ${HTSSOLVER_ROOT}")
  
endif()

# pour limiter le mode verbose
set(HTSSOLVER_FIND_QUIETLY ON)

find_package_handle_standard_args(HTSSOLVER
	DEFAULT_MSG 
	HTSSOLVER_INCLUDE_DIR 
	HTSSOLVER_LIBRARY)

find_package_handle_standard_args(HTSSOLVERCORE
	DEFAULT_MSG
	HTSSOLVERCORE_LIBRARY)
	
find_package_handle_standard_args(HTSSOLVERSIMD
	DEFAULT_MSG
	HTSSOLVERSIMD_LIBRARY)
	
if(HTSSOLVER_FOUND AND NOT TARGET htssolver)
    
  set(HTSSOLVER_INCLUDE_DIRS ${HTSSOLVER_INCLUDE_DIR})
  
  set(HTSSOLVER_LIBRARIES ${HTSSOLVER_LIBRARY} ${HTSSOLVERCORE_LIBRARY})

  # htssolver main
	  
  add_library(htssolver_main UNKNOWN IMPORTED)
  
  set_target_properties(htssolver_main PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${HTSSOLVER_INCLUDE_DIRS}")
  
  set_target_properties(htssolver_main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${HTSSOLVER_LIBRARY}")
  
  
  # htssolver
  add_library(htssolver INTERFACE IMPORTED)
  set_property(TARGET htssolver APPEND PROPERTY 
                 INTERFACE_LINK_LIBRARIES "htssolver_main")
                 
                 
  if(HTSSOLVERCORE_FOUND)

    list(APPEND HTSSOLVER_LIBRARIES ${HTSSOLVERCORE_LIBRARY})
    add_library(htssolver_core UNKNOWN IMPORTED)
  
    set_target_properties(htssolver_core PROPERTIES 
                          INTERFACE_INCLUDE_DIRECTORIES "${HTSSOLVER_INCLUDE_DIRS}")
  
    set_target_properties(htssolver_core PROPERTIES
                          IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                          IMPORTED_LOCATION "${HTSSOLVERCORE_LIBRARY}")
                          
    set_property(TARGET htssolver APPEND PROPERTY 
                 INTERFACE_LINK_LIBRARIES "htssolver_core")
  endif()
  
  if(HTSSOLVERSIMD_FOUND)
    
    list(APPEND HTSSOLVER_LIBRARIES ${HTSSOLVERSIMD_LIBRARY})
    add_library(htssolver_simd UNKNOWN IMPORTED)
  
    set_target_properties(htssolver_simd PROPERTIES 
                          INTERFACE_INCLUDE_DIRECTORIES "${HTSSOLVER_INCLUDE_DIRS}")
  
    set_target_properties(htssolver_simd PROPERTIES
                          IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                          IMPORTED_LOCATION "${HTSSOLVERSIMD_LIBRARY}")
                          
    set_property(TARGET htssolver APPEND PROPERTY 
                 INTERFACE_LINK_LIBRARIES "htssolver_simd")
  endif()
  
  set_target_properties(htssolver PROPERTIES
                        INTERFACE_COMPILE_DEFINITIONS "${HTSSOLVER_FLAGS}")

  if(TARGET harts)
     set_property(TARGET htssolver APPEND PROPERTY 
                  INTERFACE_LINK_LIBRARIES "harts")
  endif()
  
  if(TARGET tbb)
     set_property(TARGET htssolver APPEND PROPERTY 
                  INTERFACE_LINK_LIBRARIES "tbb")
  endif()
  

endif()
