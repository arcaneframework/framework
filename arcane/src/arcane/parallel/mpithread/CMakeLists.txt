set(PKGS MPI)
if (NOT TARGET arcane_mpi)
  message(STATUS "Disabling 'arcane_mpithread' because 'arcane_mpi' is not available")
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_mpithread
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/parallel/mpithread
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_mpithread PRIVATE ARCANE_COMPONENT_arcane_mpithread)

arcane_add_arccon_packages(arcane_mpithread PRIVATE ${PKGS})

target_link_libraries(arcane_mpithread PUBLIC arcane_mpi arcane_thread arcane_core arcane_utils)

arcane_register_library(arcane_mpithread)
