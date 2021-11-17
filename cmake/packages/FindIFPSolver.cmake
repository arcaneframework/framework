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

# ifpsolver_intel [POUR ANCIENNE CHAINE DE COMPILATION ou WIN32]

# Si gimkl ou foss, INTEL_ROOT non défini

# On va créer une target ifpsolver_intel que l'on ajoutera à ifpsolver

if(NOT INTEL_ROOT)
  set(INTEL_ROOT $ENV{INTEL_ROOT})
endif()

if(INTEL_ROOT)
  set(_INTEL_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_INTEL_SEARCH_OPTS)
endif()

if(NOT WIN32)
  
  if(NOT IFPSOLVER_INTEL_FOUND)
    
    find_library(IFPSOLVER_INTEL_IRC_LIBRARY
      NAMES irc
      HINTS ${INTEL_ROOT} 
	  PATH_SUFFIXES lib lib/intel64
      ${_INTEL_SEARCH_OPTS}
      )
    mark_as_advanced(INTEL_IRC_LIBRARY)
    
  endif()
 
  # pour limiter le mode verbose
  set(IFPSOLVER_INTEL_FIND_QUIETLY ON)

  find_package_handle_standard_args(IFPSOLVER_INTEL 
	  DEFAULT_MSG 
	  IFPSOLVER_INTEL_IRC_LIBRARY)
  
  if(IFPSOLVER_INTEL_FOUND AND NOT TARGET ifpsolver_intel)
  
    set(IFPSOLVER_INTEL_LIBRARIES ${IFPSOLVER_INTEL_IRC_LIBRARY})

    add_library(ifpsolver_intel UNKNOWN IMPORTED)
            
    set_target_properties(ifpsolver_intel PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${IFPSOLVER_INTEL_IRC_LIBRARY}")
      
  endif()
  
else()
  
  find_library(IFPSOLVER_INTEL_IRC_LIBRARY libirc)
  find_library(IFPSOLVER_INTEL_SVLM_DISPMT_LIBRARY svml_dispmt)
  find_library(IFPSOLVER_INTEL_SVLM_DISPMD_LIBRARY svml_dispmd)
  find_library(IFPSOLVER_INTEL_M_LIBRARY libm)
  find_library(IFPSOLVER_INTEL_MMT_LIBRARY libmmt)
  find_library(IFPSOLVER_INTEL_MMD_LIBRARY libmmd)
  find_library(IFPSOLVER_INTEL_DECIMAL_LIBRARY libdecimal)
  find_library(IFPSOLVER_INTEL_IFCONSOL_LIBRARY ifconsol)
  find_library(IFPSOLVER_INTEL_IFCOREMD_LIBRARY libifcoremd)
  find_library(IFPSOLVER_INTEL_IFPORTMD_LIBRARY libifportmd)
  
  include(FindPackageHandleStandardArgs)

  # pour limiter le mode verbose
  set(IFPSOLVER_INTEL_FIND_QUIETLY ON)

  find_package_handle_standard_args(IFPSOLVER_INTEL
    DEFAULT_MSG 
    IFPSOLVER_INTEL_M_LIBRARY 
	  IFPSOLVER_INTEL_MMT_LIBRARY 
	  IFPSOLVER_INTEL_MMD_LIBRARY
	  IFPSOLVER_INTEL_IRC_LIBRARY
	  IFPSOLVER_INTEL_SVLM_DISPMT_LIBRARY 
	  IFPSOLVER_INTEL_SVLM_DISPMD_LIBRARY
	  IFPSOLVER_INTEL_DECIMAL_LIBRARY 
	  IFPSOLVER_INTEL_IFCONSOL_LIBRARY 
	  IFPSOLVER_INTEL_IFCOREMD_LIBRARY
	  IFPSOLVER_INTEL_IFPORTMD_LIBRARY
    )
	
  if(IFPSOLVER_INTEL_FOUND AND NOT TARGET ifpsolver_intel)
    # création de la cible
    add_library(ifpsolver_intel INTERFACE IMPORTED)
    # librairie
    # SD: Gros doute sur ce qui suit...
    # D'après moi, on écrase la propriété INTERFACE_LINK_LIBRARIES à chaque ligne...
    # Donc seule la dernière gagne...
    set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_M_LIBRARY})
	set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_MMT_LIBRARY} APPEND)
	set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_MMD_LIBRARY} APPEND)
	set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_IRC_LIBRARY} APPEND)
	set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_SVLM_DISPMT_LIBRARY} APPEND)
    set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_SVLM_DISPMD_LIBRARY} APPEND)
	set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_DECIMAL_LIBRARY} APPEND)
	set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_IFCONSOL_LIBRARY} APPEND)
	set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_IFCOREMD_LIBRARY} APPEND)
	# SD: la gagnante...
    set_property(TARGET ifpsolver_intel PROPERTY INTERFACE_LINK_LIBRARIES ${IFPSOLVER_INTEL_IFPORTMD_LIBRARY} APPEND)
  endif()

endif()

# [POUR NOUVELLE CHAINE DE COMPILATION]

# Les macros définies sont MPI_ROOT et IFORT_ROOT

# D'abord on cherche mpiifort

if(MPI_ROOT)
  set(_MPI_SEARCH_OPTS NO_DEFAULT_PATH)
else ()
  set(_MPI_SEARCH_OPTS)
endif()

# Si on a déjà le fortran, on ne cherche pas
if(NOT MPI_FORTRAN_LIBRARY)

  find_library(MPI_FORTRAN_LIBRARY
    NAMES mpifort
    HINTS ${MPI_ROOT}
    PATH_SUFFIXES lib lib64
    ${_MPI_SEARCH_OPTS}
    )
  mark_as_advanced(MPI_FORTRAN_LIBRARY)

endif()

# pour limiter le mode verbose
set(IFPSOLVER_MPIFORT_FIND_QUIETLY ON)

find_package_handle_standard_args(IFPSOLVER_MPIFORT
	DEFAULT_MSG 
  MPI_FORTRAN_LIBRARY
  )

if(IFPSOLVER_MPIFORT_FOUND AND NOT TARGET ifpsolver_mpifort)

  add_library(ifpsolver_mpifort INTERFACE IMPORTED)
  
  set_target_properties(ifpsolver_mpifort PROPERTIES
    INTERFACE_LINK_LIBRARIES "${MPI_FORTRAN_LIBRARY}")

endif()

# Ensuite les librairies intel ifort

if(NOT IFORT_ROOT)
  set(IFORT_ROOT $ENV{IFORT_ROOT})
endif()

if(IFORT_ROOT)
  set(_IFORT_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_IFORT_SEARCH_OPTS)
endif()
 
if(NOT IFPSOLVER_IFORT_FOUND)
  
  # Pour IFPPartitioner et non IFPSolver !!
  find_library(IFPSOLVER_SVML_LIBRARY
    NAMES svml 
    HINTS ${IFORT_ROOT}
    PATH_SUFFIXES lib lib/intel64
    ${_IFORT_SEARCH_OPTS}
    )
  mark_as_advanced(IFPSOLVER_SVML_LIBRARY)

  find_library(IFPSOLVER_IFCORE_LIBRARY
    NAMES ifcore 
    HINTS ${IFORT_ROOT}
    PATH_SUFFIXES lib lib/intel64
    ${_IFORT_SEARCH_OPTS}
    )
  mark_as_advanced(IFPSOLVER_IFCORE_LIBRARY)

  find_library(IFPSOLVER_IRC_LIBRARY
    NAMES irc
    HINTS ${IFORT_ROOT} 
		PATH_SUFFIXES lib lib/intel64
    ${_IFORT_SEARCH_OPTS}
    )
  mark_as_advanced(IFPSOLVER_IRC_LIBRARY)
  
  find_library(IFPSOLVER_IFPORT_LIBRARY
    NAMES ifport
    HINTS ${IFORT_ROOT}
    PATH_SUFFIXES lib lib/intel64
    ${_IFORT_SEARCH_OPTS}
    )
  mark_as_advanced(IFPSOLVER_IFPORT_LIBRARY)
  
endif()

# pour limiter le mode verbose
set(IFPSOLVER_IFORT_FIND_QUIETLY ON)

find_package_handle_standard_args(IFPSOLVER_IFORT
	DEFAULT_MSG 
	IFPSOLVER_IFCORE_LIBRARY
  IFPSOLVER_IRC_LIBRARY
  IFPSOLVER_IFPORT_LIBRARY
  IFPSOLVER_SVML_LIBRARY
  )

if(IFPSOLVER_IFORT_FOUND AND NOT TARGET ifpsolver_ifort)

  add_library(ifpsolver_ifcore UNKNOWN IMPORTED)
	
  set_target_properties(ifpsolver_ifcore PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IFPSOLVER_IFCORE_LIBRARY}")
	
  add_library(ifpsolver_irc UNKNOWN IMPORTED)
	
  set_target_properties(ifpsolver_irc PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IFPSOLVER_IRC_LIBRARY}")
	    
  add_library(ifpsolver_ifport UNKNOWN IMPORTED)
	
  set_target_properties(ifpsolver_ifport PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IFPSOLVER_IFPORT_LIBRARY}")
  
  add_library(ifpsolver_svml UNKNOWN IMPORTED)
	
  set_target_properties(ifpsolver_svml PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IFPSOLVER_SVML_LIBRARY}")
  
  add_library(ifpsolver_ifort INTERFACE IMPORTED)
	  
  set_property(TARGET ifpsolver_ifort APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ifpsolver_ifcore")
  
  set_property(TARGET ifpsolver_ifort APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ifpsolver_irc")
  
  set_property(TARGET ifpsolver_ifort APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ifpsolver_ifport")
 
  set_property(TARGET ifpsolver_ifort APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ifpsolver_svml")
    
endif()

# Enfin ifpsolver

if(NOT IFPSOLVER_ROOT)
  set(IFPSOLVER_ROOT $ENV{IFPSOLVER_ROOT})
endif()

if(IFPSOLVER_ROOT)
  set(_IFPSOLVER_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_IFPSOLVER_SEARCH_OPTS)
endif()

if(NOT IFPSOLVER_FOUND) 
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
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
	IFPSOLVER_DOMAINDECOMP_LIBRARY
  )

if(IFPSOLVER_FOUND AND NOT TARGET ifpsolver)
    
  set(IFPSOLVER_INCLUDE_DIRS ${IFPSOLVER_INCLUDE_DIR})
  
  set(IFPSOLVER_LIBRARIES ${IFPSOLVER_LIBRARY}
                          ${IFPSOLVER_DOMAINDECOMP_LIBRARY})

  if(TARGET ifpsolver_mpifort)
    list(APPEND IFPSOLVER_LIBRARIES ${MPI_FORTRAN_LIBRARY})
  endif()
  if(TARGET ifpsolver_ifort)
    list(APPEND IFPSOLVER_LIBRARIES ${IFPSOLVER_IFCORE_LIBRARY})
    list(APPEND IFPSOLVER_LIBRARIES ${IFPSOLVER_IRC_LIBRARY})
    list(APPEND IFPSOLVER_LIBRARIES ${IFPSOLVER_IFPORT_LIBRARY})
    list(APPEND IFPSOLVER_LIBRARIES ${IFPSOLVER_SVML_LIBRARY})
  endif()
  # NB: on ajoute pas ifpsolver_intel (car cas particuliers à gérer et bientot déprécié...)

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
  
  if(TARGET ifpsolver_mpifort)
    set_property(TARGET ifpsolver APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "ifpsolver_mpifort")
  endif()

  if(TARGET ifpsolver_ifort)
    set_property(TARGET ifpsolver APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "ifpsolver_ifort")
  endif()
  
  if(TARGET ifpsolver_intel)
    set_property(TARGET ifpsolver APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "ifpsolver_intel")
  endif()
  
endif()

# vérification finale :
# ifpsolver_intel
# OU
# ifpsolver_ifort ET ifpsolver_mpifort

if(IFPSOLVER_FOUND)
if(TARGET ifpsolver_intel AND TARGET ifpsolver_ifort)
  logFatalError("ifpsolver_intel and ifpsolver_ifort defined at same time : dont use INTEL_ROOT and IFORT_ROOT")
endif()
if(TARGET ifpsolver_intel AND TARGET ifpsolver_mpifort)
  logFatalError("ifpsolver_intel and ifpsolver_mpifort defined at same time : dont use INTEL_ROOT and MPI_ROOT")
endif()
if(NOT TARGET ifpsolver_ifort AND TARGET ifpsolver_mpifort)
  logFatalError("ifpsolver_mpifort and ifpsolver_ifort not defined at same time : use IFORT_ROOT and MPI_ROOT")
endif()
if(TARGET ifpsolver_ifort AND NOT TARGET ifpsolver_mpifort)
  logFatalError("ifpsolver_mpifort and ifpsolver_ifort not defined at same time : use IFORT_ROOT and MPI_ROOT")
endif()
endif()

