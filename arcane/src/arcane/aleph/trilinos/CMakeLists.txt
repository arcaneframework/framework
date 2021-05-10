message(STATUS "ALEPH_TRILINOS subdirectory")
set(PKGS Trilinos MPI)
arcane_find_package(Trilinos)
if (NOT TARGET arcconpkg_Trilinos)
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_aleph_trilinos
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/trilinos
  FILES ${ARCANE_SOURCES}
)

arcane_add_arccon_packages(arcane_aleph_trilinos PRIVATE ${PKGS})
target_link_libraries(arcane_aleph_trilinos PRIVATE arccon::Trilinos arccon::MPI)
target_link_libraries(arcane_aleph_trilinos PUBLIC arcane_core)
arcane_register_library(arcane_aleph_trilinos OPTIONAL)
