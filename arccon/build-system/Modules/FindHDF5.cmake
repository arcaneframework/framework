#
# Find the Hdf5 includes and library
#
# This module defines
# HDF5_INCLUDE_DIR, where to find headers,
# HDF5_LIBRARIES, the libraries to link against to use Hdf5.
# HDF5_FOUND, If false, do not try to use Hdf5.

include (${CMAKE_CURRENT_LIST_DIR}/../commands/commands.cmake)

arccon_return_if_package_found(HDF5)

# Tente d'utiliser le module correspondant de CMake
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(HDF5 CONFIG QUIET)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

# Note: à partir de HDF5 1.10, il faudrait toujours
# utiliser le mode configuration de HDF5 ce qui
# permettrait de supprimer tout le code qui n'utilise
# pas de target
unset(_ARCCON_HDF5_TARGET)

if (TARGET hdf5::hdf5-shared)
  set(_ARCCON_HDF5_TARGET hdf5::hdf5-shared)
elseif(TARGET hdf5::hdf5-static)
  set(_ARCCON_HDF5_TARGET hdf5::hdf5-static)
endif()

if (_ARCCON_HDF5_TARGET)
  message(STATUS "Found HDF5 via configuration target=${_ARCCON_HDF5_TARGET}")

  # Par défaut on utilise la cible importée sauf si on demande l'ancien mécanisme
  if (NOT ARCCON_USE_LEGACY_FIND)
    arccon_register_cmake_config_target(HDF5 CONFIG_TARGET_NAME ${_ARCCON_HDF5_TARGET})
    return()
  endif()

  # NOTE: il existe deux bibliothèques suivant si on
  # est en débug ou release. On les récupère via les
  # propriétés IMPORTED_LOCATION_{DEBUG|RELEASE|RELWITHDEBINFO}

  if (WIN32)
    get_target_property(HDF5_LIBRARIES_DEBUG ${_ARCCON_HDF5_TARGET} IMPORTED_IMPLIB_DEBUG)
    get_target_property(HDF5_LIBRARIES_RELEASE ${_ARCCON_HDF5_TARGET} IMPORTED_IMPLIB_RELEASE)
    get_target_property(HDF5_LIBRARIES_RELWITHDEBINFO ${_ARCCON_HDF5_TARGET} IMPORTED_IMPLIB_RELWITHDEBINFO)
  else()
    get_target_property(HDF5_LIBRARIES_DEBUG ${_ARCCON_HDF5_TARGET} IMPORTED_LOCATION_DEBUG)
    get_target_property(HDF5_LIBRARIES_RELEASE ${_ARCCON_HDF5_TARGET} IMPORTED_LOCATION_RELEASE)
    get_target_property(HDF5_LIBRARIES_RELWITHDEBINFO ${_ARCCON_HDF5_TARGET} IMPORTED_LOCATION_RELWITHDEBINFO)
  endif()

  get_target_property(HDF5_INCLUDE_DIRS ${_ARCCON_HDF5_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(HDF5_COMPILE_DEFINITIONS ${_ARCCON_HDF5_TARGET} INTERFACE_COMPILE_DEFINITIONS)
  message(STATUS "HDF5_LIBRARIES_DEBUG = ${HDF5_LIBRARIES_DEBUG}")
  message(STATUS "HDF5_LIBRARIES_RELEASE = ${HDF5_LIBRARIES_RELEASE}")
  message(STATUS "HDF5_LIBRARIES_RELWITHDEBINFO = ${HDF5_LIBRARIES_RELWITHDEBINFO}")
  message(STATUS "HDF5_INCLUDE_DIRS = ${HDF5_INCLUDE_DIRS}")
  message(STATUS "HDF5_COMPILE_DEFINITIONS = ${HDF5_COMPILE_DEFINITIONS}")
  # Note: il faudrait uniquement indiquer qu'on dépend de hdf5::hdf5-shared
  # via set(HDF5_LIBRARIES hdf5::hdf5-shared) mais
  # cela ne fonctionne pas avec le pkglist.xml généré car ce dernier a besoin
  # explicitement de la liste des includes et des bibliothèques
  if (HDF5_LIBRARIES_RELWITHDEBINFO)
    set(HDF5_LIBRARIES "${HDF5_LIBRARIES_RELWITHDEBINFO}")
  else()
    set(HDF5_LIBRARIES "${HDF5_LIBRARIES_RELEASE}")
  endif()

  message(STATUS "HDF5_LIBRARIES = ${HDF5_LIBRARIES}")

  if (HDF5_LIBRARIES AND HDF5_INCLUDE_DIRS)
    set(HDF5_FOUND TRUE)

    # On ne sais pas si HDF5 a besoin de libz. Dans le doute, on le cherche
    # et s'il existe on l'ajoute.
    find_library(GZIP_LIBRARY z)
    if(GZIP_LIBRARY)
      list(APPEND HDF5_LIBRARIES ${GZIP_LIBRARY})
    endif()

    arccon_register_package_library(HDF5 HDF5)
    set(ARCCON_TARGET_HDF5 ${_ARCCON_HDF5_TARGET} CACHE STRING "Target for package HDF5" FORCE)
    set_property(TARGET arcconpkg_HDF5 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${HDF5_COMPILE_DEFINITIONS})
  endif()
  return()
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Tente de trouver HDF5 via le FindHdf5 de cmake
# On a besoin d'au moins un 'COMPONENTS' pour déterminer la version de HDF5.

set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(HDF5 MODULE)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})
if (HDF5_FOUND)
  if (TARGET hdf5::hdf5)
    arccon_register_cmake_config_target(HDF5 CONFIG_TARGET_NAME hdf5::hdf5)
    return()
  else()
    message(STATUS "Found HDF5 with CMake FindHdf5")
    message(STATUS "HDF5_VERSION = ${HDF5_VERSION}")
    message(STATUS "HDF5_LIBRARIES = ${HDF5_LIBRARIES}")
    message(STATUS "HDF5_INCLUDE_DIRS = ${HDF5_INCLUDE_DIRS}")
    message(STATUS "HDF5_DEFINITIONS = ${HDF5_DEFINITIONS}")
    arccon_register_package_library(HDF5 HDF5)
    if (HDF5_DEFINITIONS)
      set_target_properties(${_TARGET_NAME}
        PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "${HDF5_DEFINITIONS}"
        )
    endif()
  endif()
  # Il semble que HDF5_VERSION ne soit pas toujours mis dans le cache. On le
  # force
  set(HDF5_VERSION ${HDF5_VERSION} CACHE STRING "Hdf5 version" FORCE)
  return()
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# On arrive ici si on n'a pas trouvé hdf5 via la configuration CMake
# ni le find_package de CMake (a priori dans ce cas on ne trouvera pas
# hdf5 donc on doit pouvoir supprimer les lignes ci-dessous.

find_library(HDF5_LIBRARY NAMES hdf5dll hdf5ddll hdf5)
find_library(GZIP_LIBRARY z)

message(STATUS "HDF5_VERSION = ${HDF5_VERSION}")
message(STATUS "HDF5_LIBRARY = ${HDF5_LIBRARY}")
if(HDF5_LIBRARY)
  GET_FILENAME_COMPONENT(HDF5_LIB_PATH ${HDF5_LIBRARY} PATH)
  GET_FILENAME_COMPONENT(HDF5_ROOT_PATH ${HDF5_LIB_PATH} PATH)
  MESSAGE(STATUS "HDF5 ROOT PATH = ${HDF5_ROOT_PATH}")
endif()

find_path(HDF5_INCLUDE_DIRS hdf5.h
  ${HDF5_ROOT_PATH}/include
)
 
set(HDF5_FOUND)
if (HDF5_INCLUDE_DIRS AND HDF5_LIBRARY)
  set(HDF5_FOUND TRUE)
  set(HDF5_LIBRARIES ${HDF5_LIBRARY})
  if(GZIP_LIBRARY)
    list(APPEND HDF5_LIBRARIES ${GZIP_LIBRARY})
  endif()
  get_filename_component(HDF5_LIB_PATH ${HDF5_LIBRARY} PATH)
endif()

arccon_register_package_library(HDF5 HDF5)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
