set(PKGS MPI)
arcane_find_package(MPI)

if (NOT TARGET Arccore::arccore_message_passing_mpi)
  message(STATUS "Disabling 'arcane_mpi' because 'Arccore::arccore_message_passing_mpi' is not available")
  return()
endif()

include(srcs.cmake)

set(ARCANE_HAS_MPI_NEIGHBOR FALSE)
if (MPI_CXX_VERSION VERSION_GREATER_EQUAL "3.1")
  list(APPEND ARCANE_SOURCES
    MpiNeighborVariableSynchronizeDispatcher.cc
    )
  set(ARCANE_HAS_MPI_NEIGHBOR TRUE)
endif()
set(ARCANE_HAS_MPI_NEIGHBOR "${ARCANE_HAS_MPI_NEIGHBOR}" CACHE BOOL "Are MPI 3.1 MPI_Neighbor_* functions available" FORCE)

arcane_add_library(arcane_mpi
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/parallel/mpi
  FILES ${ARCANE_SOURCES}
  )

target_compile_definitions(arcane_mpi PRIVATE ARCANE_COMPONENT_arcane_mpi)
if (ARCANE_HAS_MPI_NEIGHBOR)
  target_compile_definitions(arcane_mpi PRIVATE ARCANE_HAS_MPI_NEIGHBOR)
endif()
target_link_libraries(arcane_mpi PUBLIC arcane_impl Arccore::arccore_message_passing_mpi)

arcane_register_library(arcane_mpi)

# Pour NCCL
if (ARCANE_TEST_NCCL)
  if (TARGET Arccore::arccore_accelerator_cuda)
    arcane_find_package(NCCL)
    if (NCCL_FOUND)
      target_sources(arcane_mpi PRIVATE NCCLVariableSynchronizeDispatcher.cc)
      target_link_libraries(arcane_mpi PRIVATE Arccore::arccore_accelerator)
      arcane_add_arccon_packages(arcane_mpi PRIVATE NCCL)
    endif()
  endif()
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:

