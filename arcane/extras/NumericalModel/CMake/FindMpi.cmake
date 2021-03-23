#
# Find the Mpi includes and library
#
# This module defines
# MPI_EXECUTABLE, where to find valgrind executable
# MPI_INCLUDE_DIRS, where to find headers,
# MPI_LIBRARIES, the libraries to link against to use Mpi.
# MPI_FOUND, If false, do not try to use Mpi.

find_program(MPI_EXECUTABLE mpiexec
  PATHS ${MPI_BIN_PATH} NO_DEFAULT_PATH)

find_path(MPI_INCLUDE_DIR mpi.h
  PATHS ${MPI_INCLUDE_PATH} NO_DEFAULT_PATH)

find_library(MPI_LIBRARY NAMES mpi mpich mpich2 msmpi
  PATHS ${MPI_LIBRARY_PATH} NO_DEFAULT_PATH)

set(MPI_FOUND "NO")
if(MPI_INCLUDE_DIR AND MPI_LIBRARY AND MPI_EXECUTABLE)
  set(MPI_FOUND "YES")
  if(WIN32)
    set(MPI_LIBRARIES ${MPI_LIBRARY})
  else(WIN32)
    set(MPI_LIBRARIES ${MPI_LIBRARY} rt)
  endif(WIN32)
  set(MPI_EXECUTABLE ${MPI_EXECUTABLE})
  set(MPI_LIBRARIES ${MPI_LIBRARIES})
  set(MPI_INCLUDE_DIRS ${MPI_INCLUDE_DIR})
  set(MPI_FLAGS "_MPI")
  get_filename_component(MPI_EXEC_PATH ${MPI_EXECUTABLE} PATH)
endif(MPI_INCLUDE_DIR AND MPI_LIBRARY AND MPI_EXECUTABLE)
