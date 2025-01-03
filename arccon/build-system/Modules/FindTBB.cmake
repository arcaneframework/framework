#
# Find the TBB (Intel Thread Building blocks) includes and library
#
# This module defines
# TBB_INCLUDE_DIR, where to find headers,
# TBB_LIBRARIES, the libraries to link against to use TBB.
# TBB_FOUND, If false, do not try to use TBB.

include(${CMAKE_CURRENT_LIST_DIR}/../commands/commands.cmake)
arccon_return_if_package_found(TBB)

# A partir des versions OneTBB (2020+), il existe un fichier de configuration
# CMake pour TBB. On tente de l'utiliser si possible

# Tente d'utiliser le module correspondant de CMake, sauf si la variable ARCCON_NO_TBB_CONFIG est activée
if (NOT ARCCON_NO_TBB_CONFIG)
  set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
  unset(CMAKE_MODULE_PATH)
  find_package(TBB CONFIG QUIET COMPONENTS tbb)
  set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})
endif ()

# vcpkg ne positionne ni TBB_tbb_found ni TBB_IMPORTED_TARGETS
# On utilise donc direction 'TBB::tbb'
if (TBB_tbb_FOUND OR TARGET TBB::tbb)
  if (NOT TBB_IMPORTED_TARGETS)
    set(TBB_IMPORTED_TARGETS TBB::tbb)
  endif()
  message(STATUS "[TBB] TBB_DIR = ${TBB_DIR}")
  message(STATUS "[TBB] TBB_IMPORTED_TARGETS = ${TBB_IMPORTED_TARGETS}")
  foreach(_component ${TBB_IMPORTED_TARGETS})
    get_target_property(_INC_DIRS ${_component} INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "[TBB] INCLUDE_DIR: ${_component} : ${_INC_DIRS}")
  endforeach()
  arccon_register_cmake_config_target(TBB CONFIG_TARGET_NAME ${TBB_IMPORTED_TARGETS})
  return()
endif()

find_library(TBB_LIBRARY_DEBUG NAMES tbb_debug)
find_library(TBB_LIBRARY_RELEASE NAMES tbb)
message(STATUS "TBB DEBUG ${TBB_LIBRARY_DEBUG}")
message(STATUS "TBB RELEASE ${TBB_LIBRARY_RELEASE}")

find_path(TBB_INCLUDE_DIR tbb/task.h)

message(STATUS "TBB_INCLUDE_DIR = ${TBB_INCLUDE_DIR}")

if (TBB_LIBRARY_DEBUG)
  set(_TBB_HAS_DEBUG_LIB TRUE)
else()
  set(TBB_LIBRARY_DEBUG ${TBB_LIBRARY_RELEASE})
endif()

set(TBB_FOUND NO)
if (TBB_INCLUDE_DIR AND TBB_LIBRARY_RELEASE AND TBB_LIBRARY_DEBUG)
  set(TBB_FOUND YES)
  if (WIN32)
    set(TBB_LIBRARIES "$<$<CONFIG:Debug>:${TBB_LIBRARY_DEBUG}>$<$<CONFIG:Release>:${TBB_LIBRARY_RELEASE}>")
  else()
    set(TBB_LIBRARIES ${TBB_LIBRARY_RELEASE} )
  endif()
  set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Créé une interface cible pour référencer facilement le package dans les dépendances.
# Utilise TBB_LIBRARY_DEBUG si on compile en mode Debug, TBB_LIBRARY_RELEASE sinon
# En débug, il faut aussi définir TBB_USE_DEBUG=1 pour que les vérifications soient
# activées.
# TODO: il faudrait pouvoir spécifier la version Debug même en compilation
# en mode optimisé.
if (TBB_FOUND)
  arccon_register_package_library(TBB TBB)
  if (CMAKE_BUILD_TYPE STREQUAL Debug)
    if (_TBB_HAS_DEBUG_LIB)
      target_compile_definitions(arcconpkg_TBB INTERFACE TBB_USE_DEBUG=1)
    endif()
  endif()
  # Sous Win32, utilise les generator-expression pour spécifier le choix de la bibliothèque
  # en fonction de la cible 'Debug' ou 'Release'.
  # Sous Unix, on devrait faire la même chose mais cela pose problème avec le fichier
  # .pc généré donc pour l'instant on laisse comme ci dessous.
  if (NOT WIN32)
    if (CMAKE_BUILD_TYPE STREQUAL Debug)
      target_link_libraries(arcconpkg_TBB INTERFACE ${TBB_LIBRARY_DEBUG})
    else()
      target_link_libraries(arcconpkg_TBB INTERFACE ${TBB_LIBRARY_RELEASE})
    endif()
  endif()
endif ()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
