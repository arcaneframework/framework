cmake_minimum_required(VERSION 3.21)

# ----------------------------------------------------------------------------
# ArcaNet: environnement de tests (fichiers .arct + lanceur ArcCher).
#
# Ce projet ne dépend pas d'Arcane. Il peut être:
#   - construit en même temps que le framework Arcane (voir
#     '_common/build_all/CMakeLists.txt'),
#   - copié dans un autre projet (seul le répertoire 'arct' est alors utile),
#   - installé puis utilisé via 'find_package(ArcaNet)'.

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version" ARCANET_VERSION_STR_FULL)
string(REPLACE "_dev" "" ArcaNet_VERSION ${ARCANET_VERSION_STR_FULL})
message(STATUS "ArcaNetVersion = ${ArcaNet_VERSION}")

project(ArcaNet VERSION ${ArcaNet_VERSION} LANGUAGES NONE)

# ----------------------------------------------------------------------------

option(ARCANET_BUILD_ARCCHER "Build the ArcCher launcher" ON)
if (ARCANET_BUILD_ARCCHER)
  add_subdirectory(arccher)
endif ()

add_subdirectory(arct)


# ----------------------------------------------------------------------------
# Installation

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ARCANET_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/ArcaNet)
set(ARCANET_CMAKE_INSTALL_DIR ${ARCANET_CONFIG_INSTALL_DIR})
set(ARCANET_SPEC_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/arcanet)

# Répertoire d'installation du fichier de configuration. Utilisé par Arcane
# pour positionner les hints de 'find_package(ArcaNet)'.
set(ArcaNet_config_install_dir "${ARCANET_CONFIG_INSTALL_DIR}"
  CACHE STRING "ArcaNet config install dir" FORCE)

set(_arcanet_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/arcanet-generated")
configure_package_config_file(cmake/ArcaNetConfig.cmake.in
  "${_arcanet_generated_dir}/ArcaNetConfig.cmake"
  INSTALL_DESTINATION "${ARCANET_CONFIG_INSTALL_DIR}"
  PATH_VARS ARCANET_CMAKE_INSTALL_DIR ARCANET_SPEC_INSTALL_DIR)
write_basic_package_version_file("${_arcanet_generated_dir}/ArcaNetConfigVersion.cmake"
  VERSION ${ArcaNet_VERSION}
  COMPATIBILITY AnyNewerVersion)
install(FILES "${_arcanet_generated_dir}/ArcaNetConfig.cmake"
                "${_arcanet_generated_dir}/ArcaNetConfigVersion.cmake"
  DESTINATION ${ARCANET_CONFIG_INSTALL_DIR})

if (ARCANET_BUILD_ARCCHER AND Arccore_FOUND)
  install(EXPORT ArcaNetTargets
    NAMESPACE ArcaNet::
    DESTINATION ${ARCANET_CONFIG_INSTALL_DIR})
endif ()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
