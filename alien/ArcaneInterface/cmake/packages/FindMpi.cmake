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

if (NOT MPI_ROOT)
  set(MPI_ROOT $ENV{MPI_ROOT})
endif ()

if (NOT MPI_HOME)
  set(MPI_HOME $ENV{MPI_HOME})
  if (NOT MPI_HOME)
      set(MPI_HOME ${MPI_ROOT})
  endif (NOT MPI_HOME)
endif ()

if(NOT MPI_SKIP_FINDPACKAGE)
  # pour limiter le mode verbose
  set(MPI_FIND_QUIETLY ON)
  find_package(MPI)
endif()

if(MPI_FOUND)

  set(MPI_INCLUDE_DIRS ${MPI_C_INCLUDE_PATH})
  set(MPI_LIBRARIES ${MPI_C_LIBRARIES})
  set(MPI_EXEC ${MPIEXEC})
  set(MPI_PROC_FLAG ${MPIEXEC_NUMPROC_FLAG})

else()

  if(MPI_ROOT)
    set(_MPI_SEARCH_OPTS NO_DEFAULT_PATH)
  else ()
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

  find_library(MPI_C_LIBRARY
    NAMES ${MPI_LIBRARY_NAME}
    HINTS ${MPI_ROOT}
    PATH_SUFFIXES lib lib64 lib/${CMAKE_BUILD_TYPE}
    ${_MPI_SEARCH_OPTS}
    )
  mark_as_advanced(MPI_LIBRARY)

  find_path(MPI_INCLUDE_DIR mpi.h
    HINTS ${MPI_ROOT}
    PATH_SUFFIXES include include64
    ${_MPI_SEARCH_OPTS}
    )
  mark_as_advanced(MPI_INCLUDE_DIR)
  
  # pour limiter le mode verbose
  set(Mpi_FIND_QUIETLY ON)

  find_package_handle_standard_args(Mpi
    DEFAULT_MSG
    MPI_INCLUDE_DIR
    MPI_C_LIBRARY
    MPI_EXEC)

endif()

if(MPI_EXEC)
  get_filename_component(MPI_BIN_PATH ${MPI_EXEC} PATH)
  set(MPI_EXEC_PATH ${MPI_BIN_PATH})
endif()

createOption(COMMANDLINE MpiStdIOConflict
  NAME        MPI_STDIO_CONFLICT
  MESSAGE     "Mpi stdio confict"
  DEFAULT     ON)

if(MPI_FOUND AND NOT TARGET mpi)

  if(NOT MPI_INCLUDE_DIRS)
    set(MPI_INCLUDE_DIRS "${MPI_INCLUDE_DIR}")
  endif(NOT MPI_INCLUDE_DIRS)

  if(NOT MPI_LIBRARIES)
    set(MPI_LIBRARIES "${MPI_C_LIBRARY}")
  endif(NOT MPI_LIBRARIES)

  add_library(mpi INTERFACE IMPORTED)

  set_target_properties(mpi PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MPI_INCLUDE_DIRS}")

  set(MPI_CXX_DEFINITIONS OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
  if(MPI_STDIO_CONFLICT)
    LIST(APPEND MPI_CXX_DEFINITIONS MPICH_IGNORE_CXX_SEEK)
  endif(MPI_STDIO_CONFLICT)

  foreach (MPI_DEF ${MPI_CXX_DEFINITIONS})
		if(MSVC)
		  set_property(TARGET mpi APPEND PROPERTY 
			  INTERFACE_COMPILE_DEFINITIONS ${MPI_DEF})
		else()
	    # Ne fonctionne qu'avec les générateurs Makefile et Ninja
     	set_property(TARGET mpi APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
        $<$<COMPILE_LANGUAGE:CXX>:${MPI_DEF}>)
	  endif()
  endforeach ()

#  set_target_properties(mpi PROPERTIES
#    IMPORTED_LINK_INTERFACE_LANGUAGES "C")
  set_target_properties(mpi PROPERTIES
    INTERFACE_LINK_LIBRARIES "${MPI_LIBRARIES}")

endif()
