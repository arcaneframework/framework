message(STATUS "ALEPH_PETSC subdirectory")
set(PKGS PETSc MPI)
arcane_find_package(PETSc)
if (NOT PETSc_FOUND)
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_aleph_petsc
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/petsc
  FILES ${ARCANE_SOURCES}
)

arcane_add_arccon_packages(arcane_aleph_petsc PRIVATE ${PKGS})
target_link_libraries(arcane_aleph_petsc PUBLIC arcane_core)
target_link_libraries(arcane_aleph_petsc PRIVATE arcane_aleph)

arcane_register_library(arcane_aleph_petsc OPTIONAL)
