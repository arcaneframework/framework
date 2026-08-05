cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(DoFsOnCells LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(DoFsOnCells main.cc)
arcane_add_arcane_libraries_to_target(DoFsOnCells)

configure_file(bar_hybrid_quad_tria.msh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(data.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

enable_testing()

# ----------------------------------------------------------------------------
# Add tests with hybrid mesh
add_test(NAME dofs_on_cells_test1 COMMAND DoFsOnCells data.arc)

if(MPIEXEC_EXECUTABLE)
  add_test(NAME dofs_on_cells_test1_p4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ./DoFsOnCells data.arc
  )
endif()

# ----------------------------------------------------------------------------

# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
