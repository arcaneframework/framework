set(PKGS MPI)
arcane_find_package(MPI)

if (NOT TARGET Arccore::arccore_message_passing_mpi)
  message(STATUS "Disabling 'arcane_mpi' because 'Arccore::arccore_message_passing_mpi' is not available")
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_mpi
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/parallel/mpi
  FILES ${ARCANE_SOURCES}
  )

target_compile_definitions(arcane_mpi PRIVATE ARCANE_COMPONENT_arcane_mpi)

target_link_libraries(arcane_mpi PUBLIC arcane_impl Arccore::arccore_message_passing_mpi)

arcane_register_library(arcane_mpi)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:

