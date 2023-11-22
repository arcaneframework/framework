message(STATUS "Parsing ${CMAKE_CURRENT_SOURCE_DIR}")
set(PKGS Sloop SuperLU MPI)
foreach(package ${PKGS})
  arcane_find_package(${package})
endforeach()

if (NOT Sloop_FOUND)
  message(STATUS "Removing aleph/sloop because package 'Sloop' is not available")
  return()
endif()
if (NOT MPI_FOUND)
  message(STATUS "Removing aleph/sloop because package 'MPI' is not available")
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_aleph_sloop
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/sloop
  FILES ${ARCANE_SOURCES}
  )

target_include_directories(arcane_aleph_sloop PUBLIC $<BUILD_INTERFACE:${ARCANECEA_SRC_PATH}> $<INSTALL_INTERFACE:include>)
target_link_libraries(arcane_aleph_sloop PUBLIC arcane_core)

arcane_add_arccon_packages(arcane_aleph_sloop PRIVATE ${PKGS})
arcane_register_library(arcane_aleph_sloop OPTIONAL)
