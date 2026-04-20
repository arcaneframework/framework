message(STATUS "Parsing '${CMAKE_CURRENT_SOURCE_DIR}' binary='${CMAKE_CURRENT_BINARY_DIR}'")

set(PKGS MPI)
arcane_find_package(MPI)

include(srcs.cmake)

arcane_add_library(arcane_cea
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/cea
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
  )

arcane_add_arccon_packages(arcane_cea PRIVATE ${PKGS})

target_include_directories(arcane_cea PUBLIC $<BUILD_INTERFACE:${ARCANEBUILDROOT}>)
target_include_directories(arcane_cea PUBLIC $<BUILD_INTERFACE:${ARCANECEA_SRC_PATH}> $<INSTALL_INTERFACE:include>)

target_link_libraries(arcane_cea PUBLIC arcane_cartesianmesh arcane_core)

arcane_register_library(arcane_cea)
