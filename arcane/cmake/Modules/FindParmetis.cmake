#
# Find the Parmetis includes and library
#
# This module defines
# Parmetis_INCLUDE_DIRS, where to find headers,
# Parmetis_LIBRARIES, the libraries to link against to use Parmetis.
# Parmetis_FOUND, If false, do not try to use Parmetis.
 
arccon_return_if_package_found(Parmetis)

set(Parmetis_FOUND FALSE)

set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(parmetis CONFIG QUIET)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

# Si on a un fichier ParmetisConfig.cmake, alors cette cible est définie et on
# l'utilise directement
if (TARGET parmetis)
  message(STATUS "Found 'parmetis' CMake configuration file")
  set(Parmetis_FOUND TRUE)
  arccon_register_cmake_config_target(Parmetis CONFIG_TARGET_NAME parmetis PACKAGE_NAME Parmetis)
  return()
endif()

find_path(Parmetis_INCLUDE_DIR parmetis.h)

if(Parmetis_INCLUDE_DIR)
  set(Parmetis_FOUND TRUE)
endif(Parmetis_INCLUDE_DIR)

find_library(Parmetis_LIBRARY parmetis)
if(NOT Parmetis_LIBRARY)
  SET( Parmetis_FOUND FALSE)
else()
  # This way we are looking for libmetis that is in the same directory than libparmetis.
  get_filename_component(Parmetis_LIBPATH ${Parmetis_LIBRARY} PATH)
  find_library(Metis_LIBRARY NAMES metis PATHS ${Parmetis_LIBPATH} NO_DEFAULT_PATH)
  message(STATUS "METIS_LIBRARY = ${Metis_LIBRARY}")
  # Si on ne trouve pas metis, recherche dans les répertoires classiques
  if (NOT Metis_LIBRARY)
    find_library(Metis_LIBRARY metis)
    message(STATUS "Metis_LIBRARY (2) = ${Metis_LIBRARY}")
  endif()

  find_path(Metis_INCLUDE_DIR NAMES metis.h PATHS ${Parmetis_INCLUDE_DIR})
  message(STATUS "Metis_INCLUDE_DIR = ${Metis_INCLUDE_DIR}")
endif()

message(STATUS "Parmetis_LIBRARY = ${Parmetis_LIBRARY}")
message(STATUS "Parmetis_INCLUDE_DIR = ${Parmetis_INCLUDE_DIR}")

if (Parmetis_FOUND)
  set(Parmetis_LIBRARIES ${Parmetis_LIBRARY} ${Metis_LIBRARY})
  set(Parmetis_INCLUDE_DIRS ${Parmetis_INCLUDE_DIR} ${Metis_INCLUDE_DIR})
  arccon_register_package_library(Parmetis Parmetis)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
