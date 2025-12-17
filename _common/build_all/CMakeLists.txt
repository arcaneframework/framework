cmake_minimum_required(VERSION 3.15)

# ----------------------------------------------------------------------------
# Indique que le projet n'a pas de language
# Cela est indispensable pour que CMake ne recherche pas les
# compilateurs avant d'avoir inclus les fichiers de configuration.
Project(ArcaneFrameworkFull LANGUAGES NONE)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0111)
  cmake_policy(SET CMP0111 NEW)
endif()

# ----------------------------------------------------------------------------

# Liste des composants à compiler
if (NOT ARCANEFRAMEWORK_BUILD_COMPONENTS)
  set (ARCANEFRAMEWORK_BUILD_COMPONENTS Arcane;Alien)
endif()
set(ARCANEFRAMEWORK_BUILD_COMPONENTS ${ARCANEFRAMEWORK_BUILD_COMPONENTS} CACHE STRING "List of ArcaneFramework components to build" FORCE)

# ----------------------------------------------------------------------------
# Macro pour positionner la variable 'var_name' dans le cache
# avec sa valeur actuelle ou la valeur 'var_value' si elle n'a
# pas de valeur.
macro(arcane_set_if_unset var_name var_value var_type description)
  if (NOT DEFINED ${var_name})
    set(${var_name} ${var_value})
  endif()
  set(${var_name} ${${var_name}} CACHE ${var_type} "${description}" FORCE)
  option(${var_name} "${description}")
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

arcane_set_if_unset(ARCANE_WANT_NUMERICS TRUE BOOL "Compile numerical library")
arcane_set_if_unset(ARCANE_WANT_TOTALVIEW TRUE BOOL "Compile total view display")
arcane_set_if_unset(ARCANE_DEFAULT_PARTITIONER Auto STRING "Default partitioner")

arcane_set_if_unset(ARCANE_WANT_LIBXML2 TRUE BOOL "Use libxml2")
arcane_set_if_unset(ARCANE_WANT_LEGACY_CONNECTIVITY FALSE BOOL "Use legacy item connectivity accessor")
arcane_set_if_unset(ARCANE_WANT_CHECK FALSE BOOL "Compilation with additional check (array bounds, ...)")

# Force l'utilisation de bibliothèques dynamiques
set(BUILD_SHARED_LIBS TRUE)

# Compile en release par défaut si rien n'est spécifié
if (NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED ARCCORE_BUILD_MODE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------------------------------------------------------

set(ARCFRAMEWORK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

arcane_set_if_unset(Arccon_ROOT ${ARCFRAMEWORK_ROOT}/arccon PATH "Arccon root path")
arcane_set_if_unset(AXLSTAR_ROOT ${ARCFRAMEWORK_ROOT}/axlstar PATH "Axlstar root")
arcane_set_if_unset(Arccore_ROOT ${ARCFRAMEWORK_ROOT}/arccore PATH "Arccore root")
arcane_set_if_unset(ARCDEPENDENCIES_ROOT ${ARCFRAMEWORK_ROOT}/dependencies PATH "Arcdependencies root")
arcane_set_if_unset(Arctools_ROOT ${ARCFRAMEWORK_ROOT}/arctools PATH "Arctools root")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

enable_testing()

# ----------------------------------------------------------------------------

if (NOT ALIEN_BUILD_COMPONENT)
  # Deux valeurs possibles: 'alien_standalone' ou 'all'
  set(ALIEN_BUILD_COMPONENT alien_standalone) # alien/ArcaneInterface (=Alien_legacy_plugins) temporarily deactivated. To remove.
endif()

# ----------------------------------------------------------------------------

message(STATUS "ARCANEFRAMEWORK_BUILD_COMPONENTS are: '${ARCANEFRAMEWORK_BUILD_COMPONENTS}'")

# ----------------------------------------------------------------------------

if (Arccore IN_LIST ARCANEFRAMEWORK_BUILD_COMPONENTS)
  add_subdirectory(${ARCFRAMEWORK_ROOT}/arccore arccore)
  set(Arccore_FOUND YES)
endif()

# ----------------------------------------------------------------------------

if (Arcane IN_LIST ARCANEFRAMEWORK_BUILD_COMPONENTS)
  message(STATUS "Configuring Arcane")
  configure_file(nuget.config.in ${CMAKE_BINARY_DIR}/nuget.config @ONLY)

  add_subdirectory(${ARCFRAMEWORK_ROOT}/arcane arcane)

  # ----------------------------------------------------------------------------
  # Recopie les packages nuget de 'dependencies' dans le répertoire
  # contenant les exemples. On recopie les packages nécessaires pour
  # 'netstandard2.0'.
  #
  set(DEPENDENCIES_NUGET ${ARCDEPENDENCIES_ROOT}/nuget_fallback)
  set(SAMPLES_NUGET_DIR ${CMAKE_INSTALL_PREFIX}/samples/_nuget_fallback)
  install(FILES ${DEPENDENCIES_NUGET}/microsoft.netcore.platforms.1.1.0.nupkg DESTINATION ${SAMPLES_NUGET_DIR})
  install(FILES ${DEPENDENCIES_NUGET}/netstandard.library.2.0.3.nupkg DESTINATION ${SAMPLES_NUGET_DIR})
endif()

# ----------------------------------------------------------------------------

if (Alien IN_LIST ARCANEFRAMEWORK_BUILD_COMPONENTS)
  message(STATUS "Configuring Alien")
  set(ARCANEFRAMEWORK_BINARY_DIR ${CMAKE_BINARY_DIR})
  add_subdirectory(${ARCFRAMEWORK_ROOT}/alien alien)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
