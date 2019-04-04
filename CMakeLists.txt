# Il faut au minimun cmake 3.11.0
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "Including arccore/CMakeLists.txt")

# ----------------------------------------------------------------------------
# Récupère numéro de version de Arcane à partir du fichier 'version' et
# génère un fichier 'arcane_version.h' contenant ces informations.

file(READ "version" ARCCORE_VERSION_STR_FULL)
string(REPLACE "_dev" "" ARCCORE_VERSION ${ARCCORE_VERSION_STR_FULL})
message(STATUS "ArccoreVersion = ${ARCCORE_VERSION}")

project(Arccore VERSION ${ARCCORE_VERSION} LANGUAGES CXX)

math(EXPR ARCCORE_VERSION_NUMERIC "((10000 * ${Arccore_VERSION_MAJOR}) + 100 * ${Arccore_VERSION_MINOR}) + ${Arccore_VERSION_PATCH}")
configure_file(arccore_version.h.in ${CMAKE_BINARY_DIR}/arccore_version.h @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/arccore_version.h DESTINATION include)

# ----------------------------------------------------------------------------

if (NOT ARCCORE_EXPORT_TARGET)
  set(ARCCORE_EXPORT_TARGET ArccoreTargets)
endif()

# ----------------------------------------------------------------------------

if (NOT ARCCON_ROOT)
  message(FATAL_ERROR "Variable 'ARCCON_ROOT' is not set")
endif()
if (NOT ARCCON_EXPORT_TARGET)
  set(ARCCON_EXPORT_TARGET ${ARCCORE_EXPORT_TARGET})
endif()
include(${ARCCON_ROOT}/Arccon.cmake)

# ----------------------------------------------------------------------------
# Indique que si une variable <PackageName>_ROOT est définie, considère
# qu'il s'agit du chemin de base du package <PackageName>.
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(ARCCORE_DEBUG YES)
endif()

# ----------------------------------------------------------------------------
# Arccore nécessite un support complet du C++17. Malheureusement, mettre
# cxx_std_17 via target_compile_features() ne suffit pas pour savoir
# si le compilateur supporte toute la norme et certaines fonctionnalités de la
# norme n'ont pas de compile-features dans CMake.
# Par exemple, c'est le cas de la classe std::string_view. Pour GCC, cette
# classe n'existe que dans les versions 7+ mais pourtant CMake considère que
# GCC 6 supporte 'cxx_std_17'. Il n'y a donc pas d'autres solutions que
# de tester la version du compilateur

if (CMAKE_COMPILER_IS_GNUCXX)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0.0")
    message(FATAL_ERROR "You version of GCC (${CMAKE_CXX_COMPILER_VERSION}) is too old. Version 7+ is required")
  endif()
endif()

# ----------------------------------------------------------------------------
# Définit une cible fictive 'arccore_build_compile_flags' et qui contient les
# options de compilation internes aux cibles de Arccore. Ce sont par exemple
# les avertissements de compilation ou les options sur la visibilité des symboles.

add_library(arccore_build_compile_flags INTERFACE)
install(TARGETS arccore_build_compile_flags EXPORT ${ARCCORE_EXPORT_TARGET})

# ----------------------------------------------------------------------------
# Les symboles des bibliothèques ne sont pas visibles sauf si demandé
# explicitement

if (CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(arccore_build_compile_flags INTERFACE -fvisibility-inlines-hidden -fvisibility=hidden)
endif()
# ----------------------------------------------------------------------------

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/CheckCompileFlags.cmake)

# ----------------------------------------------------------------------------
# Fonctions utilitaires diverses
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/Functions.cmake)

# ----------------------------------------------------------------------------
# Regarde la taille des différents types (short,int,long,long long) du C++
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/CheckArccoreTypeSize.cmake)

# ----------------------------------------------------------------------------

configure_file(${CMAKE_CURRENT_LIST_DIR}/arccore_config.h.in ${CMAKE_BINARY_DIR}/arccore/arccore_config.h)
install(FILES ${CMAKE_BINARY_DIR}/arccore/arccore_config.h DESTINATION include/arccore)

# ----------------------------------------------------------------------------

#set(Arccore_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#set(Arccore_BINARY_DIR ${CMAKE_BINARY_DIR}/arccore)

message(STATUS "Arccore_SOURCE_DIR=${Arccore_SOURCE_DIR}")
message(STATUS "Arccore_BINARY_DIR=${Arccore_BINARY_DIR}")

# ----------------------------------------------------------------------------
# Ajoute support des tests. Il faut le faire avant d'ajouter les componsants
# car ces derniers en ont besoin.

# Par défaut on exécute les tests sauf si cela est explicitement pas demandé.
if (NOT DEFINED ARCCORE_WANT_TEST)
  set(ARCCORE_WANT_TEST TRUE)
endif()

if (ARCCORE_WANT_TEST)
  find_package(GTest)
  if (GTEST_FOUND)
    include(GoogleTest)
  endif()
endif()

# ----------------------------------------------------------------------------

arccore_add_component_directory(base)
arccore_add_component_directory(collections)
arccore_add_component_directory(concurrency)
arccore_add_component_directory(trace)
arccore_add_component_directory(message_passing)
arccore_add_component_directory(message_passing_mpi)

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
