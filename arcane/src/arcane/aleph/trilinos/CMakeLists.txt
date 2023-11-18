message(STATUS "'arcane_aleph_trilinos' subdirectory")

set(PKGS TrilinosAleph MPI)
foreach(package ${PKGS})
  arcane_find_package(${package})
  if (NOT ${package}_FOUND)
    message(STATUS "Disabling 'arcane_aleph_trilinos' because package '${package}' is not found")
    return()
  endif()
endforeach()

include(srcs.cmake)

arcane_add_library(arcane_aleph_trilinos
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/trilinos
  FILES ${ARCANE_SOURCES}
)

arcane_add_arccon_packages(arcane_aleph_trilinos PRIVATE ${PKGS})
target_link_libraries(arcane_aleph_trilinos PUBLIC arcane_core)
arcane_register_library(arcane_aleph_trilinos OPTIONAL)
