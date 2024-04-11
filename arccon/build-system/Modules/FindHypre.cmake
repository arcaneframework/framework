include (${CMAKE_CURRENT_LIST_DIR}/../commands/commands.cmake)
arccon_return_if_package_found(Hypre)

# Les versions récentes de Hypre permettent d'utiliser un CMakeLists.txt
# Tente d'utiliser le module correspondant de CMake s'il existe.
# A noter que ce module s'appelle 'HYPRE' et pas 'Hypre'.

set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(HYPRE CONFIG QUIET)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

# Les versions de Hypre à partir de la version 2.20 utilisent toujours
# un CMakeLists.txt et doivent avoir la cible HYPRE::HYPRE qui est définie
# NOTE: Normalement le package devrait s'appeler HYPRE si on veut être cohérent
# avec le fichier fournit par Hypre.
if (TARGET HYPRE::HYPRE)
  if (NOT ARCCON_USE_LEGACY_FIND)
    arccon_register_cmake_config_target(Hypre CONFIG_TARGET_NAME HYPRE::HYPRE)
    return()
  endif()
  arccon_register_package_library(Hypre HYPRE)
  set(ARCCON_TARGET_Hypre HYPRE::HYPRE CACHE STRING "Target for package HYPRE" FORCE)
endif()

find_library(Hypre_LIBRARY
  NAMES HYPRE)

set(Hypre_LIBRARIES ${Hypre_LIBRARY})

# On debian/ubuntu, headers can be found in a /usr/include/"pkg"
find_path(Hypre_INCLUDE_DIRS HYPRE.h
  PATH_SUFFIXES Hypre hypre)
mark_as_advanced(Hypre_INCLUDE_DIRS)

set(Hypre_INCLUDE_DIRS ${Hypre_INCLUDE_DIR})

# SD: others libraries should be found by find_library
# Look for other lib for Hypre
# It is the case on debian based platform
get_filename_component(Hypre_LIBRARY_DIR ${Hypre_LIBRARY} DIRECTORY)
# extra : .so .a .dylib, etc
file(GLOB Hypre_EXTRA_LIBRARIES ${Hypre_LIBRARY_DIR}/libHYPRE*.*)

list(APPEND Hypre_LIBRARIES ${Hypre_EXTRA_LIBRARIES})

message(STATUS "Hypre_INCLUDE_DIRS=${Hypre_INCLUDE_DIRS}")
message(STATUS "Hypre_LIBRARIES=${Hypre_LIBRARIES}")
message(STATUS "Hypre_LIBRARY=${Hypre_LIBRARY}")
message(STATUS "Hypre_LIBRARY_DIR=${Hypre_LIBRARY_DIR}")
mark_as_advanced(Hypre_LIBRARIES)

find_package_handle_standard_args(Hypre
        DEFAULT_MSG
        Hypre_INCLUDE_DIRS
        Hypre_LIBRARIES)

arccon_register_package_library(Hypre Hypre)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
