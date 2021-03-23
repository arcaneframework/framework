message(STATUS "Parsing ${CMAKE_CURRENT_SOURCE_DIR}")
set(PKGS MPI Sloop SuperLU)
find_package(Sloop)
if (NOT TARGET arccon::sloop)
  message(STATUS "Removing aleph/sloop because package 'sloop' is not available")
  return()
endif()
if (NOT TARGET arccon::MPI)
  message(STATUS "Removing aleph/sloop because package 'MPI' is not available")
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_aleph_sloop
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/aleph/sloop
  FILES ${ARCANE_SOURCES}
  )

target_include_directories(arcane_aleph_sloop PUBLIC $<BUILD_INTERFACE:${ARCANECEA_SRC_PATH}> $<INSTALL_INTERFACE:include>)
target_link_libraries(arcane_aleph_sloop PUBLIC arcane_core)
target_link_libraries(arcane_aleph_sloop PRIVATE arccon::sloop)
if (TARGET arccon::superlu)
  target_link_libraries(arcane_aleph_sloop PRIVATE arccon::superlu)
endif()
target_link_libraries(arcane_aleph_sloop PRIVATE arccon::MPI)

arcane_register_library(arcane_aleph_sloop OPTIONAL)
