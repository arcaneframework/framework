cmake_minimum_required(VERSION 3.21)

# ----------------------------------------------------------------------------
# Installation

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ARCANET_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/ArcaNet)
set(ARCANET_CMAKE_INSTALL_DIR ${ARCANET_CONFIG_INSTALL_DIR}/ArcCher)
set(ARCANET_SPEC_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/arcanet)

install(FILES
  cmake/arct_edit.cmake
  cmake/arct_functions.cmake
  cmake/arct_to_ctest.cmake
  DESTINATION ${ARCANET_CMAKE_INSTALL_DIR})
install(FILES
  cmake/internal/internal_arct_edit.cmake
  cmake/internal/internal_arct_to_ctest.cmake
  DESTINATION ${ARCANET_CMAKE_INSTALL_DIR}/internal)
install(FILES specifications.md
  DESTINATION ${ARCANET_SPEC_INSTALL_DIR})


include(cmake/arct_functions.cmake)
include(cmake/test.cmake)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
