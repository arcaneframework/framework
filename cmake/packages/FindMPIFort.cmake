#
# Find the MPIFORT includes and library
#
# This module uses
# MPI_ROOT
#
# This module defines
# MPIFORT_FOUND
# MPIFORT_INCLUDE_DIRS
# MPIFORT_LIBRARIES
#
# Target mpifort

# intel [POUR ANCIENNE CHAINE DE COMPILATION ou WIN32]

# Si gimkl ou foss, INTEL_ROOT non défini

# On va créer une target intel que l'on ajoutera à ifpsolver

if(NOT INTEL_ROOT)
  set(INTEL_ROOT $ENV{INTEL_ROOT})
endif()

if(INTEL_ROOT)
  set(_INTEL_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_INTEL_SEARCH_OPTS)
endif()

if(NOT WIN32)
  
  if(NOT INTEL_FOUND)
    
    find_library(INTEL_IRC_LIBRARY
      NAMES irc
      HINTS ${INTEL_ROOT} 
		  PATH_SUFFIXES lib lib/intel64
      ${_INTEL_SEARCH_OPTS}
      )
    mark_as_advanced(INTEL_IRC_LIBRARY)
    
  endif()
 
  # pour limiter le mode verbose
  set(INTEL_FIND_QUIETLY ON)

  find_package_handle_standard_args(INTEL 
	  DEFAULT_MSG 
	  INTEL_IRC_LIBRARY)
  
  if(INTEL_FOUND AND NOT TARGET intel)
  
    set(INTEL_LIBRARIES ${INTEL_IRC_LIBRARY})

    add_library(intel UNKNOWN IMPORTED)
            
    set_target_properties(intel PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${INTEL_IRC_LIBRARY}")
      
  endif()
  
else()
  
  find_library(INTEL_IRC_LIBRARY libirc)
  find_library(INTEL_SVLM_DISPMT_LIBRARY svml_dispmt)
  find_library(INTEL_SVLM_DISPMD_LIBRARY svml_dispmd)
  find_library(INTEL_M_LIBRARY libm)
  find_library(INTEL_MMT_LIBRARY libmmt)
  find_library(INTEL_MMD_LIBRARY libmmd)
  find_library(INTEL_DECIMAL_LIBRARY libdecimal)
  find_library(INTEL_IFCONSOL_LIBRARY ifconsol)
  find_library(INTEL_IFCOREMD_LIBRARY libifcoremd)
  find_library(INTEL_IFPORTMD_LIBRARY libifportmd)
  
  include(FindPackageHandleStandardArgs)

  # pour limiter le mode verbose
  set(INTEL_FIND_QUIETLY ON)

  find_package_handle_standard_args(INTEL
    DEFAULT_MSG 
    INTEL_M_LIBRARY 
	  INTEL_MMT_LIBRARY 
	  INTEL_MMD_LIBRARY
	  INTEL_IRC_LIBRARY
	  INTEL_SVLM_DISPMT_LIBRARY 
	  INTEL_SVLM_DISPMD_LIBRARY
	  INTEL_DECIMAL_LIBRARY 
	  INTEL_IFCONSOL_LIBRARY 
	  INTEL_IFCOREMD_LIBRARY
	  INTEL_IFPORTMD_LIBRARY
    )
	
  if(INTEL_FOUND AND NOT TARGET intel)
    # création de la cible
    add_library(intel INTERFACE IMPORTED)
    # librairie
    # SD: Gros doute sur ce qui suit...
    # D'après moi, on écrase la propriété INTERFACE_LINK_LIBRARIES à chaque ligne...
    # Donc seule la dernière gagne...
    set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_M_LIBRARY})
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_MMT_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_MMD_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_IRC_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_SVLM_DISPMT_LIBRARY} APPEND)
    set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_SVLM_DISPMD_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_DECIMAL_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_IFCONSOL_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_IFCOREMD_LIBRARY} APPEND)
	  # SD: la gagnante...
    set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${INTEL_IFPORTMD_LIBRARY} APPEND)
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
    NAMES mpifort mpi_mpifh
    HINTS ${MPI_ROOT}
    PATH_SUFFIXES lib lib64
    ${_MPI_SEARCH_OPTS}
    )
  mark_as_advanced(MPI_FORTRAN_LIBRARY)
  message(status "MPI_FORTRAN_LIBRARY : ${MPI_FORTRAN_LIBRARY}")

endif()

# pour limiter le mode verbose
set(MPIFORT_FIND_QUIETLY ON)

find_package_handle_standard_args(MPIFORT
	DEFAULT_MSG 
  MPI_FORTRAN_LIBRARY
  )

if(MPIFORT_FOUND AND NOT TARGET mpifort)

  add_library(mpifort INTERFACE IMPORTED)
  
  set_target_properties(mpifort PROPERTIES
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
 
if(NOT IFORT_FOUND)
  
  # Pour IFPPartitioner et non IFPSolver !!
  find_library(SVML_LIBRARY
    NAMES svml 
    HINTS ${IFORT_ROOT}
    PATH_SUFFIXES lib lib/intel64
    ${_IFORT_SEARCH_OPTS}
    )
  mark_as_advanced(SVML_LIBRARY)

  find_library(IFCORE_LIBRARY
    NAMES ifcore 
    HINTS ${IFORT_ROOT}
    PATH_SUFFIXES lib lib/intel64
    ${_IFORT_SEARCH_OPTS}
    )
  mark_as_advanced(IFCORE_LIBRARY)

  find_library(IRC_LIBRARY
    NAMES irc
    HINTS ${IFORT_ROOT} 
		PATH_SUFFIXES lib lib/intel64
    ${_IFORT_SEARCH_OPTS}
    )
  mark_as_advanced(IRC_LIBRARY)
  
  find_library(IFPORT_LIBRARY
    NAMES ifport
    HINTS ${IFORT_ROOT}
    PATH_SUFFIXES lib lib/intel64
    ${_IFORT_SEARCH_OPTS}
    )
  mark_as_advanced(IFPORT_LIBRARY)
  
endif()

# pour limiter le mode verbose
set(IFORT_FIND_QUIETLY ON)

find_package_handle_standard_args(IFORT
	DEFAULT_MSG 
	IFCORE_LIBRARY
  IRC_LIBRARY
  IFPORT_LIBRARY
  SVML_LIBRARY
  )

if(IFORT_FOUND AND NOT TARGET ifort)

  add_library(ifcore UNKNOWN IMPORTED)
	
  set_target_properties(ifcore PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IFCORE_LIBRARY}")
	
  add_library(irc UNKNOWN IMPORTED)
	
  set_target_properties(irc PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IRC_LIBRARY}")
	    
  add_library(ifport UNKNOWN IMPORTED)
	
  set_target_properties(ifport PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${IFPORT_LIBRARY}")
  
  add_library(svml UNKNOWN IMPORTED)
	
  set_target_properties(svml PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${SVML_LIBRARY}")
  
  add_library(ifort INTERFACE IMPORTED)
	  
  set_property(TARGET ifort APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ifcore")
  
  set_property(TARGET ifort APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "irc")
  
  set_property(TARGET ifort APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ifport")
 
  set_property(TARGET ifort APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "svml")
    
endif()


