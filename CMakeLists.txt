cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# ----------------------------------------------------------------------------
# Indique que le projet n'a pas de language
# Cela est indispensable pour que CMake ne recherche pas les
# compilateurs avant d'avoir inclus les fichiers de configuration.
Project(ArcaneRepo LANGUAGES NONE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Positionne CMAKE_BUILD_TYPE si ARCANE_BUILD_TYPE est défini ou
# ARCANE_BUILD_TYPE en fonction de CMAKE_BUILD_TYPE dans le cas contraire
# TODO: utiliser CMAKE_CONFIGURATION_TYPE pour modifier les valeurs possibles
# CMAKE_BUILD_TYPE et ainsi les faire correspondre à celles de Arcane.

# NOTE: a terme, ARCANE_BUILD_TYPE ne sera plus utilisé et seul
# ARCCORE_BUILD_MODE restera

# Si 'ARCCORE_BUILD_MODE' est défini, on s'en sert pour positionner
# ARCANE_BUILD_TYPE même si ce dernier est défini.

if (DEFINED ARCCORE_BUILD_MODE)
  set(ARCANE_BUILD_TYPE ${ARCCORE_BUILD_MODE})
endif()

if (NOT DEFINED ARCANE_BUILD_TYPE)
  if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  if (CMAKE_BUILD_TYPE STREQUAL Debug)
    set(ARCANE_BUILD_TYPE Debug)
  else()
    set(ARCANE_BUILD_TYPE Release)
  endif()
else()
  if (ARCANE_BUILD_TYPE STREQUAL Debug)
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Debug)
    endif()
    set(ARCANE_WANT_CHECK TRUE)
  endif()

  if (ARCANE_BUILD_TYPE STREQUAL Check)
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release)
    endif()
    set(ARCANE_WANT_CHECK TRUE)
  endif()

  if (ARCANE_BUILD_TYPE STREQUAL Release)
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release)
    endif()
  endif()
endif()

if (NOT DEFINED ARCCORE_BUILD_MODE)
  set(ARCCORE_BUILD_MODE ${ARCANE_BUILD_TYPE})
endif()

#set(ARCCORE_BUILD_MODE ${ARCCORE_BUILD_MODE} CACHE STRING "Arccore build mode (debug,check,release)" FORCE)
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "The type of build, options are: None Debug Release RelWithDebInfo" FORCE)

message(STATUS "ARCANE_BUILD_TYPE = ${ARCANE_BUILD_TYPE}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# ----------------------------------------------------------------------------

enable_testing()

# ----------------------------------------------------------------------------
# Le vrai projet se trouve dans un sous-répertoire.
# A terme, on déplacera le CMakeLists.txt de ce sous-répertoire ici
add_subdirectory(arcane)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
