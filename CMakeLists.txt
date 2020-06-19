cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# For support of VERSION
cmake_policy(SET CMP0048 NEW)
# Check libraries with FOO_ROOT (CMake 3.12)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
# target_link_libraries from different libraries (CMake 3.13)
if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()
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

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(ARCCORE_DEBUG YES)
endif()

option(ARCCORE_USE_MPI "Whether or not MPI support must be set" ON)


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
# Définit une cible fictive 'arccore_build_compile_flags' et qui contient les
# options de compilation internes aux cibles de Arccore. Ce sont par exemple
# les avertissements de compilation ou les options sur la visibilité des symboles.

add_library(arccore_build_compile_flags INTERFACE)
install(TARGETS arccore_build_compile_flags EXPORT ${ARCCORE_EXPORT_TARGET})

add_library(Arccore::arccore_build_compile_flags ALIAS arccore_build_compile_flags)

# ----------------------------------------------------------------------------
# Create an interface library 'arccore_full' that groups all Arccore components.
add_library(arccore_full INTERFACE)
install(TARGETS arccore_full EXPORT ${ARCCORE_EXPORT_TARGET})
# FIXME: whether or not arccore_build_compile_flags has to be added to arccore_full.

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
# Ajoute support des tests. Il faut le faire avant d'ajouter les componsants
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
  "${version_config}" COMPATIBILITY SameMajorVersion
)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * TARGETS_EXPORT_NAME
#   * PROJECT_NAME
configure_package_config_file(
  "CMake/ArccoreConfig.cmake.in"
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
