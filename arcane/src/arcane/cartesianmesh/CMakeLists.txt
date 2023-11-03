include(srcs.cmake)

arcane_add_library(arcane_cartesianmesh
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/cartesianmesh
  FILES ${ARCANE_SOURCES}
  )

arcane_add_arccon_packages(arcane_cartesianmesh PRIVATE ${PKGS})

target_include_directories(arcane_cartesianmesh PUBLIC $<BUILD_INTERFACE:${ARCANEBUILDROOT}>)
target_include_directories(arcane_cartesianmesh PUBLIC $<BUILD_INTERFACE:${ARCANECEA_SRC_PATH}> $<INSTALL_INTERFACE:include>)

target_link_libraries(arcane_cartesianmesh PUBLIC arcane_core)
target_link_libraries(arcane_cartesianmesh PRIVATE arcane_mesh)

arcane_register_library(arcane_cartesianmesh)

# ----------------------------------------------------------------------------

if (GTEST_FOUND)
  add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
