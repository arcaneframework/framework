#
# Find the MPI includes and library
#
# This module uses
# MPI_ROOT
#
# This module defines
# MPI_FOUND
# MPI_INCLUDE_DIRS
# MPI_LIBRARIES
#
# Target mpi 
if(MPI_FOUND)
   return()
endif()

if(NOT MPI_ROOT)
  set(MPI_ROOT $ENV{MPI_ROOT})
endif()

if(MPI_ROOT)
  set(_MPI_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_MPI_SEARCH_OPTS)
endif()

if(NOT MPI_EXEC_NAME)
  set(MPI_EXEC_NAME mpiexec)
endif()

if(NOT MPI_LIBRARY_NAME)
  set(MPI_LIBRARY_NAME mpi impi mpich mpich2 msmpi)
endif()

if(NOT MPI_PROC_FLAG)
  set(MPI_PROC_FLAG "-np")
endif()

set(Mpi_FIND_QUIETLY ON)
logStatus("MPI_FOUND            : ${MPI_FOUND}")
logStatus("MPI_BIN_FROM_ENV     : ${MPI_BIN_FROM_ENV}")
logStatus("MPI_EXEC             : ${MPI_EXEC}")
if(NOT MPI_FOUND)

  if(MPI_BIN_FROM_ENV)
    # on cherche mpiexec dans l'environnement
	find_program(MPI_EXEC ${MPI_EXEC_NAME})
  else()
    # on cherche mpiexec dans MPI_ROOT
    find_program(MPI_EXEC
      NAMES ${MPI_EXEC_NAME}
      HINTS ${MPI_ROOT}
      PATH_SUFFIXES bin bin64
      ${_MPI_SEARCH_OPTS}
      )
  endif()
  mark_as_advanced(MPI_EXEC)
  
  find_library(MPI_LIBRARY 
    NAMES ${MPI_LIBRARY_NAME}
    HINTS ${MPI_ROOT}
		PATH_SUFFIXES lib lib64 lib/${CMAKE_BUILD_TYPE} lib64/${CMAKE_BUILD_TYPE}  lib/release lib/debug
		${_MPI_SEARCH_OPTS}
  )
  mark_as_advanced(MPI_LIBRARY)
  
  find_path(MPI_INCLUDE_DIR mpi.h
    HINTS ${MPI_ROOT} 
		PATH_SUFFIXES include include64
    ${_MPI_SEARCH_OPTS}
  )
  mark_as_advanced(MPI_INCLUDE_DIR)
  
endif()

if(MPI_EXEC)
  get_filename_component(MPI_BIN_PATH ${MPI_EXEC} PATH)
  set(MPI_EXEC_PATH ${MPI_BIN_PATH})
endif()

# pour limiter le mode verbose
set(Mpi_FIND_QUIETLY ON)
logStatus("MPI_INCLUDE_DIR : ${MPI_INCLUDE_DIR}")
logStatus("MPI_LIBRARY     : ${MPI_LIBRARY}")
logStatus("MPI_EXEC        : ${MPI_EXEC}")
find_package_handle_standard_args(Mpi 
	DEFAULT_MSG 
	MPI_INCLUDE_DIR 
	MPI_LIBRARY 
	MPI_EXEC)

if(MPI_FOUND AND NOT TARGET mpi)

  set(MPI_INCLUDE_DIRS "${MPI_INCLUDE_DIR}")

  set(MPI_LIBRARIES "${MPI_LIBRARY}")

  add_library(mpi UNKNOWN IMPORTED)
  
  set_target_properties(mpi PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${MPI_INCLUDE_DIRS}")
    
  set_target_properties(mpi PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MPI_LIBRARY}")
    
endif()
