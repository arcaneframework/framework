#
# Find the 'bzip2' includes and library
#
# This module defines
# BZIP2_INCLUDE_DIR, where to find headers,
# BZIP2_LIBRARIES, the libraries to link against to use bzip2.
# BZIP2_FOUND, If false, do not try to use bzip2.

arccon_return_if_package_found(BZip2)

# Essaie de trouver un fichier de configuration ou un FindBZip2.cmake.
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(BZip2)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

if (TARGET BZip2::BZip2)
  arccon_register_cmake_config_target(BZip2 CONFIG_TARGET_NAME BZip2::BZip2)
  return()
endif()

# TODO: utiliser le package correspondant de CMake.
find_library(BZip2_LIBRARY bz2)
find_path(BZip2_INCLUDE_DIR bzlib.h)

message(STATUS "BZip2_INCLUDE_DIR        = ${BZip2_INCLUDE_DIR}")
message(STATUS "BZip2_LIBRARY            = ${BZip2_LIBRARY}")

set(BWip2_FOUND FALSE)
if(BZip2_INCLUDE_DIR AND BZip2_LIBRARY)
  set(BZip2_FOUND TRUE)
  set(BZip2_LIBRARIES ${BZip2_LIBRARY} )
  set(BZip2_INCLUDE_DIRS ${BZip2_INCLUDE_DIR})
endif()

arccon_register_package_library(BZip2 BZip2)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
