#
# Find the 'libdw' includes and library
#
# This module defines
# DW_INCLUDE_DIR, where to find headers,
# DW_LIBRARIES, the libraries to link against to use 'libdw'.
# DW_FOUND, If false, do not try to use 'libdw'.

arccon_return_if_package_found(DW)

message(STATUS "Search for DW")

# Il n'y a pas de find_package correspondant à 'DW' dans 'CMake'
# On cherche à l'ancienne
find_library(DW_LIBRARY dw)
find_path(DW_INCLUDE_DIR elfutils/libdwfl.h)

message(STATUS "DW_INCLUDE_DIR = ${DW_INCLUDE_DIR}")
message(STATUS "DW_LIBRARY     = ${DW_LIBRARY}")

set(DW_FOUND FALSE)
if(DW_INCLUDE_DIR AND DW_LIBRARY)
  set(DW_FOUND TRUE)
  set(DW_LIBRARIES ${DW_LIBRARY} )
  set(DW_INCLUDE_DIRS ${DW_INCLUDE_DIR})
endif()

arccon_register_package_library(DW DW)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
