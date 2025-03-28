message(STATUS "Parsing '${CMAKE_CURRENT_SOURCE_DIR}' binary='${CMAKE_CURRENT_BINARY_DIR}'")

set(PKGS MPI Lima)
arcane_find_package(MPI)
arcane_find_package(Lima)

include(srcs.cmake)

if (Lima_FOUND)
  list(APPEND ARCANE_SOURCES Lima.cc)
  if (ARCANE_LIMA_HAS_MLI)
    list(APPEND ARCANE_SOURCES LimaMalipp1.cc LimaMalippT.h)
  endif()
  if (ARCANE_LIMA_HAS_MLI2)
    list(APPEND ARCANE_SOURCES LimaMalipp2.cc)
  endif()
endif()

arcane_add_library(arcane_cea
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/cea
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
  )

arcane_add_arccon_packages(arcane_cea PRIVATE ${PKGS})

target_include_directories(arcane_cea PUBLIC $<BUILD_INTERFACE:${ARCANEBUILDROOT}>)
target_include_directories(arcane_cea PUBLIC $<BUILD_INTERFACE:${ARCANECEA_SRC_PATH}> $<INSTALL_INTERFACE:include>)

if(TARGET Lima::Lima)
  message(STATUS "Add Lima::Lima to 'arcane_cea' target")
  target_link_libraries(arcane_cea PRIVATE Lima::Lima)
elseif(TARGET arcane::lima)
  target_link_libraries(arcane_cea PRIVATE arcane::lima)
  if (ARCANE_LIMA_HAS_MLI2)
    arcane_add_arccon_packages(arcane_cea PRIVATE HDF5)
  endif()
endif()

if (ARCANE_LIMA_HAS_MLI2)
  target_compile_definitions(arcane_cea PRIVATE ARCANE_LIMA_HAS_MLI2)
endif()
if (ARCANE_LIMA_HAS_MLI)
  target_compile_definitions(arcane_cea PRIVATE ARCANE_LIMA_HAS_MLI)
endif()

target_link_libraries(arcane_cea PUBLIC arcane_cartesianmesh arcane_core)

arcane_register_library(arcane_cea)
