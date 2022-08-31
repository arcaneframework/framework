#
# Find the TrilinosMpi includes and library
#
# This module defines
# TRILINOS_INCLUDE_DIRS, where to find headers,
# TRILINOS_LIBRARIES, the libraries to link against to use Trilinos.
# TRILINOS_FOUND, If false, do not try to use Trilinos.

arccon_return_if_package_found(Trilinos)

set(Trilinos_TARGETS_IMPORTED 1)
find_package(AztecOO QUIET)
find_package(ML QUIET)
find_package(Ifpack QUIET)
find_package(Belos QUIET)

MESSAGE(STATUS "AztecOO_FOUND " ${AztecOO_FOUND})
MESSAGE(STATUS "ML_FOUND " ${ML_FOUND})
MESSAGE(STATUS "Ifpack_FOUND " ${Ifpack_FOUND})
MESSAGE(STATUS "Belos_FOUND " ${Belos_FOUND})

set(TRILINOS_FOUND "NO")

if (AztecOO_FOUND AND ML_FOUND AND Ifpack_FOUND AND Belos_FOUND)
   SET(TRILINOS_INCLUDE_DIRS "${AztecOO_INCLUDE_DIRS}")
   if (TRILINOS_INCLUDE_DIRS)
    set(TRILINOS_FOUND "YES")
  endif()
endif()

foreach(T_LIBNAME ${ML_LIBRARIES} ${Belos_LIBRARIES})
  find_library(T_LIB_${T_LIBNAME} ${T_LIBNAME} PATHS ${AztecOO_LIBRARY_DIRS})
  if (T_LIB_${T_LIBNAME})
      LIST(APPEND TRILINOS_LIBRARIES ${T_LIB_${T_LIBNAME}})
  else()
      set(TRILINOS_FOUND "NO")
  endif()
endforeach()

#Remove duplicate libraries, keeping the last (for linking)
if (TRILINOS_FOUND)
  set(Trilinos_FOUND TRUE)
  LIST(REVERSE TRILINOS_LIBRARIES)
  LIST(REMOVE_DUPLICATES TRILINOS_LIBRARIES)
  LIST(REVERSE TRILINOS_LIBRARIES)
  arccon_register_package_library(Trilinos TRILINOS)
endif (TRILINOS_FOUND)

message(STATUS "TRILINOS_LIBRARIES = ${TRILINOS_LIBRARIES}")
message(STATUS "TRILINOS_INCLUDE_DIRS = '${TRILINOS_INCLUDE_DIRS}'")
message(STATUS "TRILINOS_FOUND = '${TRILINOS_FOUND}'")

if(NOT TRILINOS_FOUND)
 unset(TRILINOS_INCLUDE_DIRS)
 unset(TRILINOS_LIBRARIES)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
