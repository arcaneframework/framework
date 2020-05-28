#
# Find the IFPSOLVER includes and library
#
# This module uses
# IFPSOLVER_ROOT
#
# This module defines
# IFPSOLVER_FOUND
# IFPSOLVER_INCLUDE_DIRS
# IFPSOLVER_LIBRARIES
#
# Target ifpsolver

if(NOT IFPSOLVER_ROOT)
  set(IFPSOLVER_ROOT $ENV{IFPSOLVER_ROOT})
endif()

if(IFPSOLVER_ROOT)
  set(_IFPSOLVER_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_IFPSOLVER_SEARCH_OPTS)
endif()

if(NOT IFPSOLVER_FOUND) 

  find_library(IFPSOLVER_LIBRARY 
    NAMES IFPSolver libIFPSolver
		HINTS ${IFPSOLVER_ROOT}
		PATH_SUFFIXES lib
		${_IFPSOLVER_SEARCH_OPTS}
    )
  mark_as_advanced(IFPSOLVER_LIBRARY)
  
  find_library(IFPSOLVER_DOMAINDECOMP_LIBRARY 
    NAMES DomainDecomp libDomainDecomp
		HINTS ${IFPSOLVER_ROOT}
		PATH_SUFFIXES lib
		${_IFPSOLVER_SEARCH_OPTS}
    )
  mark_as_advanced(IFPSOLVER_DOMAINDECOMP_LIBRARY)
  
  find_path(IFPSOLVER_INCLUDE_DIR m_bcgssolver_module.mod
    HINTS ${IFPSOLVER_ROOT} 
		PATH_SUFFIXES include
    ${_IFPSOLVER_SEARCH_OPTS}
    )
  mark_as_advanced(IFPSOLVER_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(IFPSOLVER_FIND_QUIETLY ON)

find_package_handle_standard_args(IFPSOLVER
	DEFAULT_MSG 
	IFPSOLVER_INCLUDE_DIR 
	IFPSOLVER_LIBRARY
	IFPSOLVER_DOMAINDECOMP_LIBRARY)

if(IFPSOLVER_FOUND AND NOT TARGET ifpsolver)
    
  set(IFPSOLVER_INCLUDE_DIRS ${IFPSOLVER_INCLUDE_DIR})
  
  set(IFPSOLVER_LIBRARIES ${IFPSOLVER_LIBRARY}
                          ${IFPSOLVER_DOMAINDECOMP_LIBRARY})
  
  set(IFPSOLVER_FLAGS PRECDB
                      SCHEMA1
                      _MPI
                      OPTEXT
                      PCBS
                      USE_MPIINC
                      USE_AMG
                      USE_HYPRE)
  
  # ifpsolver main
	  
  add_library(ifpsolver_main UNKNOWN IMPORTED)
	  
  set_target_properties(ifpsolver_main PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${IFPSOLVER_INCLUDE_DIRS}")
  
	set_target_properties(ifpsolver_main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IFPSOLVER_LIBRARY}")
  
  # domain decomp
  
  add_library(ifpsolver_domaindecomp UNKNOWN IMPORTED)
	  
  set_target_properties(ifpsolver_domaindecomp PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IFPSOLVER_DOMAINDECOMP_LIBRARY}")
  
  # ifpsolver
  
  add_library(ifpsolver INTERFACE IMPORTED)
	  
  set_property(TARGET ifpsolver APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ifpsolver_main")

  set_property(TARGET ifpsolver APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ifpsolver_domaindecomp")

  set_target_properties(ifpsolver PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${IFPSOLVER_FLAGS}")
  
endif()
