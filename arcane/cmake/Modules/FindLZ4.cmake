#
# Find the 'lz4' includes and library
#
# This module defines
# LZ4_INCLUDE_DIR, where to find headers,
# LZ4_LIBRARIES, the libraries to link against to use lz4.
# LZ4_FOUND, If false, do not try to use lz4.

arccon_return_if_package_found(LZ4)

# Essaie de trouver le fichier de configuration correspondant.
# 'LZ4' ne livre normalement pas de fichier de configuration mais
# par exemple 'vcpkg' en génère un
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(lz4 CONFIG)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

if (TARGET lz4::lz4)
  arccon_register_cmake_config_target(LZ4 CONFIG_TARGET_NAME lz4::lz4)
  return()
endif()

# Il n'y a pas de find_package correspondant à 'LZ4' dans 'CMake'
find_library(LZ4_LIBRARY lz4)
find_path(LZ4_INCLUDE_DIR lz4.h)

message(STATUS "LZ4_INCLUDE_DIR = ${LZ4_INCLUDE_DIR}")
message(STATUS "LZ4_LIBRARY     = ${LZ4_LIBRARY}")

set(LZ4_FOUND FALSE)
if(LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
  set(LZ4_FOUND TRUE)
  set(LZ4_LIBRARIES ${LZ4_LIBRARY} )
  set(LZ4_INCLUDE_DIRS ${LZ4_INCLUDE_DIR})
endif()

arccon_register_package_library(LZ4 LZ4)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
