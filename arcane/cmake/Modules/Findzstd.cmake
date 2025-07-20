#
# Find the 'zstd' includes and library
#
# This module defines
# zstd_INCLUDE_DIR, where to find headers,
# zstd_LIBRARIES, the libraries to link against to use zstd.
# zstd_FOUND, If false, do not try to use zstd.

arccon_return_if_package_found(zstd)

# Essaie de trouver le fichier de configuration correspondant.
# 'zstd' ne livre normalement pas de fichier de configuration mais
# par exemple 'vcpkg' en génère un
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(zstd CONFIG)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

if (TARGET zstd::libzstd)
  arccon_register_cmake_config_target(zstd CONFIG_TARGET_NAME zstd::libzstd)
  return()
endif()

# TODO: utiliser pkg_check_module()

# Il n'y a pas de find_package correspondant à 'zstd' dans 'CMake'
find_library(zstd_LIBRARY zstd)
find_path(zstd_INCLUDE_DIR zstd.h)

message(STATUS "zstd_INCLUDE_DIR = ${zstd_INCLUDE_DIR}")
message(STATUS "zstd_LIBRARY     = ${zstd_LIBRARY}")

set(zstd_FOUND FALSE)
if(zstd_INCLUDE_DIR AND zstd_LIBRARY)
  set(zstd_FOUND TRUE)
  set(zstd_LIBRARIES ${zstd_LIBRARY} )
  set(zstd_INCLUDE_DIRS ${zstd_INCLUDE_DIR})
endif()

arccon_register_package_library(zstd zstd)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
