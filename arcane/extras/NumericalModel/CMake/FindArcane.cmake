#
# Find the Arcane includes and library
#
# This module defines
# ARCANE_INCLUDE_DIRS, where to find headers,
# ARCANE_LIBRARIES, the libraries to link
# ARCANE_AXL2CC, the axl to cc compiler
# ARCANE_FOUND, If false, do not try to use Arcane.

set(ARCANE_INCLUDE_PATH ${ARCANE_PATH}/include)
set(ARCANE_LIBRARY_PATH ${ARCANE_PATH}/lib)
set(ARCANE_BIN_PATH     ${ARCANE_PATH}/bin)

find_path(ARCANE_INCLUDE_DIR arcane_config.h
  PATHS ${ARCANE_INCLUDE_PATH} NO_DEFAULT_PATH)
if(NOT ARCANE_INCLUDE_DIR)
  message(STATUS "Arcane include not found in ${ARCANE_INCLUDE_PATH}")
endif(NOT ARCANE_INCLUDE_DIR)

find_program(AXL2CC_COMPILER axl2cc
  PATHS ${ARCANE_BIN_PATH} NO_DEFAULT_PATH)
if(NOT AXL2CC_COMPILER)
  message(STATUS "Arcane axl2cc not found in ${ARCANE_BIN_PATH}")
endif(NOT AXL2CC_COMPILER)

set(ARCANE_VERSION_BACKUP ${ARCANE_VERSION})
# il vaudrait mieux utiliser pkg_check_modules aussi pour les libs arcane
# doc http://www.cmake.org/cmake/help/cmake2.6docs.html#module:FindPkgConfig
set(ENV{PKG_CONFIG_PATH} ${ARCANE_PATH})
pkg_check_modules(ARCANE arcane)
set(ARCANE_FLAGS ${ARCANE_CFLAGS})
set(ARCANE_VERSION ${ARCANE_VERSION_BACKUP}) # car le .pc contient une version erronée pour l'instant

set(ARCANE_LIB_LIST arcane_core arcane_impl arcane_mesh arcane_std arcane_utils arcane_mpi arcane_ios)

if(ARCANE_FOUND)
  foreach(ARCANE_LIB ${ARCANE_LIB_LIST})
    find_library(ARCANE_SUB_${ARCANE_LIB} ${ARCANE_LIB}
                 PATHS ${ARCANE_LIBRARY_PATH} NO_DEFAULT_PATH)
    if(ARCANE_SUB_${ARCANE_LIB})
      set(ARCANE_LIBRARY ${ARCANE_LIBRARY} ${ARCANE_SUB_${ARCANE_LIB}})
    else(ARCANE_SUB_${ARCANE_LIB})
      set(ARCANE_LIBRARY_FAILED "YES")
      message(STATUS "  sub library ${ARCANE_LIB} not found")
    endif(ARCANE_SUB_${ARCANE_LIB})
  endforeach(ARCANE_LIB)
else(ARCANE_FOUND)
      set(ARCANE_LIBRARY_FAILED "YES")
endif(ARCANE_FOUND)

set(ARCANE_FOUND "NO")
if(ARCANE_INCLUDE_DIR AND AXL2CC_COMPILER)
  if(NOT ARCANE_LIBRARY_FAILED)
    set(ARCANE_FOUND "YES")
    set(AXL2CC_COMPILER ${AXL2CC_COMPILER})
    set(ARCANE_INCLUDE_DIRS ${ARCANE_INCLUDE_DIR})
    set(ARCANE_LIBRARIES ${ARCANE_LIBRARY})
  endif(NOT ARCANE_LIBRARY_FAILED)
endif(ARCANE_INCLUDE_DIR AND AXL2CC_COMPILER)

