cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# ----------------------------------------------------------------------------
# Récupère numéro de version de Arccore à partir du fichier 'version' et
# génère un fichier 'arcane_version.h' contenant ces informations.

file(READ "version" ARCCORE_VERSION_STR_FULL)
string(REPLACE "_dev" "" ARCCORE_VERSION ${ARCCORE_VERSION_STR_FULL})
message(STATUS "ArccoreVersion = ${ARCCORE_VERSION}")
project(Arccore VERSION ${ARCCORE_VERSION} LANGUAGES CXX)

math(EXPR ARCCORE_VERSION_NUMERIC "((10000 * ${Arccore_VERSION_MAJOR}) + 100 * ${Arccore_VERSION_MINOR}) + ${Arccore_VERSION_PATCH}")
configure_file(arccore_version.h.in ${CMAKE_BINARY_DIR}/arccore_version.h @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/arccore_version.h DESTINATION include)

# ----------------------------------------------------------------------------

if (NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "Arccore has to be compiled with BUILD_SHARED_LIBS=TRUE. Static linking does not work.")
endif()

# ----------------------------------------------------------------------------

if (NOT ARCCORE_EXPORT_TARGET)
  set(ARCCORE_EXPORT_TARGET ArccoreTargets)
endif()

# ----------------------------------------------------------------------------

# Use Arccon functions and macros
find_package(Arccon REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})
include(LoadBuildSystem)
include(ArcconSetInstallDirs)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Positionne les variables 'ARCCORE_DEBUG' et 'ARCCORE_CHECK'.
# en fonction des valeurs de 'CMAKE_BUILD_TYPE' ou 'ARCCORE_BUILD_MODE'

message(STATUS "[arccore] CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "[arccore] ARCCORE_BUILD_MODE (initial) = ${ARCCORE_BUILD_MODE}")

# Si ARCCORE_BUILD_MODE n'est pas défini, positionne sa valeur en fonction
# de la valeur de 'CMAKE_BUILD_TYPE'
if (NOT ARCCORE_BUILD_MODE)
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(ARCCORE_BUILD_MODE Debug)
  else()
    set(ARCCORE_BUILD_MODE Release)
  endif()
  message(STATUS "Forcing 'ARCCORE_BUILD_MODE' to '${ARCCORE_BUILD_MODE}' from value of 'CMAKE_BUILD_TYPE'")
endif()

if(ARCCORE_BUILD_MODE STREQUAL "Debug")
  set(ARCCORE_DEBUG YES)
  set(ARCCORE_CHECK YES)
elseif(ARCCORE_BUILD_MODE STREQUAL "Check")
  set(ARCCORE_DEBUG NO)
  set(ARCCORE_CHECK YES)
elseif(ARCCORE_BUILD_MODE STREQUAL "Release")
  set(ARCCORE_DEBUG NO)
  set(ARCCORE_CHECK NO)
else()
  message(FATAL_ERROR "Invalid value '${ARCCORE_BUILD_MODE}' for 'ARCCORE_BUILD_MODE'. Valid values are 'Debug', 'Check' or 'Release'")
endif()

# S'assure que le mode debug contient le mode check
if(ARCCORE_DEBUG)
  set(ARCCORE_CHECK YES)
endif(ARCCORE_DEBUG)

set(ARCCORE_BUILD_MODE ${ARCCORE_BUILD_MODE} CACHE STRING "Arccore build type (debug,check,release)" FORCE)
set(ARCCORE_DEBUG ${ARCCORE_DEBUG} CACHE BOOL "Is Arccore in debug mode ?" FORCE)
set(ARCCORE_CHECK ${ARCCORE_CHECK} CACHE BOOL "Is Arccore in check mode ?" FORCE)

message(STATUS "[arccore] CMAKE_BUILD_TYPE (final) = ${CMAKE_BUILD_TYPE}")
message(STATUS "[arccore] ARCCORE_BUILD_MODE (final) = ${ARCCORE_BUILD_MODE}")
message(STATUS "[arccore] Is Check mode enabled (final) ? ${ARCCORE_CHECK}")
message(STATUS "[arccore] Is Debug mode enabled (final) ? ${ARCCORE_DEBUG}")

# ----------------------------------------------------------------------------

option(ARCCORE_USE_MPI "Whether or not MPI support is enabled" ON)
option(ARCCORE_ENABLE_CODE_COVERAGE "Whether we use compiler option for code coverage" OFF)

# Pour l'instant c'est obligatoire d'avoir la GLIB.
set(ARCCORE_ENABLE_GLIB TRUE CACHE BOOL "Is Glib enabled in Arccon?" FORCE)

# ----------------------------------------------------------------------------
# Il ne faut surtout pas inclure le répertoire courante automatiquement
# car cela pose des problèmes avec certains fichiers tels que 'String.h' qui
# peut être inclus à la place de 'string.h' sous windows par exemple.

set(CMAKE_INCLUDE_CURRENT_DIR FALSE)

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
# ----------------------------------------------------------------------------
# Version du standard C++ qu'on souhaite utiliser.
# Les valeurs valides sont 17, 20 ou 23
if (NOT DEFINED ARCCORE_CXX_STANDARD)
  set(ARCCORE_CXX_STANDARD 17)
endif()
set (ARCCORE_CXX_STANDARD_LIST 17 20 23)
if (NOT ${ARCCORE_CXX_STANDARD} IN_LIST ARCCORE_CXX_STANDARD_LIST)
  message(FATAL_ERROR "Invalid value for '${ARCCORE_CXX_STANDARD}' for ARCCORE_CXX_STANDARD."
    " Valid values are '${ARCCORE_CXX_STANDARD_LIST}'")
endif()
message(STATUS "Compiling with C++ standard: ${ARCCORE_CXX_STANDARD}")

# ----------------------------------------------------------------------------
# Définit une cible fictive 'arccore_build_compile_flags' et qui contient les
# options de compilation internes aux cibles de Arccore. Ce sont par exemple
# les avertissements de compilation ou les options sur la visibilité des symboles.

add_library(arccore_build_compile_flags INTERFACE)
install(TARGETS arccore_build_compile_flags EXPORT ${ARCCORE_EXPORT_TARGET})

add_library(Arccore::arccore_build_compile_flags ALIAS arccore_build_compile_flags)

# ----------------------------------------------------------------------------
# Meme si on n'est pas en mode verif, il faut toujours avoir les symboles de debug
# pour le profiling ou pour les debuggers
if (UNIX)
  target_compile_options(arccore_build_compile_flags INTERFACE -g)
endif ()
if (MSVC)
  # '/Zi' permet de créér des fichiers '.pdb' mais n'est pas compatible avec 'ccache'.
  # On utilise donc '/Z7' qui place les informations de debug dans les '.obj'.
  target_compile_options(arccore_build_compile_flags INTERFACE /Z7)
  target_link_options(arccore_build_compile_flags INTERFACE /debug)
endif()

# ----------------------------------------------------------------------------
# Create an interface library 'arccore_full' that groups all Arccore components.
add_library(arccore_full INTERFACE)
install(TARGETS arccore_full EXPORT ${ARCCORE_EXPORT_TARGET})
# FIXME: whether or not arccore_build_compile_flags has to be added to arccore_full.

# ----------------------------------------------------------------------------
# Les symboles des bibliothèques ne sont pas visibles sauf si demandé
# explicitement

# ----------------------------------------------------------------------------
# Options pour la couverture de test

if (ARCCORE_ENABLE_CODE_COVERAGE)
  message(STATUS "Code coverage is enabled")
  if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(arccore_build_compile_flags INTERFACE --coverage -fprofile-abs-path -ftest-coverage)
    target_link_options(arccore_build_compile_flags INTERFACE --coverage -fprofile-abs-path -ftest-coverage)
  else()
    message(FATAL_ERROR "Code coverage is only enabled for GCC compiler")
  endif()
endif()

# ----------------------------------------------------------------------------

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckCompileFlags.cmake)

# ----------------------------------------------------------------------------
# Fonctions utilitaires diverses
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Functions.cmake)

# ----------------------------------------------------------------------------
# Regarde la taille des différents types (short,int,long,long long) du C++
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckArccoreTypeSize.cmake)

# ----------------------------------------------------------------------------

configure_file(${CMAKE_CURRENT_LIST_DIR}/arccore_config.h.in ${CMAKE_BINARY_DIR}/arccore/arccore_config.h)
install(FILES ${CMAKE_BINARY_DIR}/arccore/arccore_config.h DESTINATION include/arccore)

# ----------------------------------------------------------------------------
# Ajoute support des tests. Il faut le faire avant d'ajouter les composants
# car ces derniers en ont besoin.

# Par défaut on exécute les tests sauf si cela est explicitement pas demandé.
if (NOT DEFINED ARCCORE_WANT_TEST)
  set(ARCCORE_WANT_TEST TRUE)
endif()

if (ARCCORE_WANT_TEST)
  enable_testing()
  find_package(GTest)
  if (GTEST_FOUND)
    include(GoogleTest)
  endif()
endif()

# ----------------------------------------------------------------------------
# Configure installation layout.
# Layout. This works for all platforms:
#   * <prefix>/lib*/cmake/<PROJECT-NAME>
#   * <prefix>/lib*/
#   * <prefix>/include/
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(namespace "Arccore::")

# ----------------------------------------------------------------------------

arccore_add_component_directory(base)
arccore_add_component_directory(collections)
arccore_add_component_directory(concurrency)
arccore_add_component_directory(trace)
arccore_add_component_directory(serialize)
arccore_add_component_directory(message_passing)
if (ARCCORE_USE_MPI)
  arccore_add_component_directory(message_passing_mpi)
endif()
# ----------------------------------------------------------------------------

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Use:
#   * PROJECT_VERSION
write_basic_package_version_file(
  "${version_config}" COMPATIBILITY AnyNewerVersion
)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * TARGETS_EXPORT_NAME
#   * PROJECT_NAME
configure_package_config_file(
  "cmake/ArccoreConfig.cmake.in"
  "${project_config}"
  INSTALL_DESTINATION "${config_install_dir}"
  )

set(Arccore_project_config "${project_config}" CACHE FILEPATH "Arccore project config" FORCE)
get_filename_component(Arccore_config_build_dir "${project_config}" DIRECTORY CACHE)
set(Arccore_config_install_dir "${config_install_dir}" CACHE STRING "Arccore config install dir" FORCE)
set(Arccore_VERSION "${Arccore_VERSION}" CACHE STRING "Arccore version" FORCE)

install(FILES "${project_config}" "${version_config}"
  DESTINATION "${config_install_dir}"
)

install(EXPORT ${ARCCORE_EXPORT_TARGET}
  DESTINATION ${config_install_dir}
  NAMESPACE "${namespace}")

if (ARCCORE_EXPORT_TARGET STREQUAL ArccoreTargets)
  message(STATUS "Exporting 'ArccoreTargets'")
  export(EXPORT ${ARCCORE_EXPORT_TARGET} FILE ${Arccore_config_install_dir}/ArccoreTargets.cmake)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
