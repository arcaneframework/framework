cmake_minimum_required(VERSION 3.11)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# For support of VERSION
cmake_policy(SET CMP0048 NEW)
# Check libraries with FOO_ROOT
cmake_policy(SET CMP0074 NEW)

# ----------------------------------------------------------------------------

project(Arccore VERSION 1.0 LANGUAGES CXX)

# ----------------------------------------------------------------------------

if (NOT ARCCORE_EXPORT_TARGET)
  set(ARCCORE_EXPORT_TARGET ArccoreTargets)
endif()
include(GNUInstallDirs)

# ----------------------------------------------------------------------------


# Use Arccon functions and macros
find_package(Arccon REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})
include(LoadBuildSystem)

# if (NOT ARCCON_ROOT)
#   message(FATAL_ERROR "Variable 'ARCCON_ROOT' is not set")
# endif()
# if (NOT ARCCON_EXPORT_TARGET)
#   set(ARCCON_EXPORT_TARGET ${ARCCORE_EXPORT_TARGET})
# endif()
# include(${ARCCON_ROOT}/Arccon.cmake)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(ARCCORE_DEBUG YES)
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
set(namespace "${Arccore}::")

# ----------------------------------------------------------------------------

arccore_add_component_directory(base)
arccore_add_component_directory(collections)
arccore_add_component_directory(concurrency)
arccore_add_component_directory(trace)
arccore_add_component_directory(message_passing)
arccore_add_component_directory(message_passing_mpi)

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

install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)

install(EXPORT ${ARCCORE_EXPORT_TARGET}
        DESTINATION ${config_install_dir})
 


# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
