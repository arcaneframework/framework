add_executable(nullspace nullspace.cc)
target_link_libraries(nullspace PUBLIC arccore_alina)

add_executable(nullspace_hybrid nullspace_hybrid.cc)
target_link_libraries(nullspace_hybrid PUBLIC arccore_alina)

add_executable(nullspace_block nullspace_block.cc)
target_link_libraries(nullspace_block PUBLIC arccore_alina)

if (TARGET mpi_target)
  add_executable(nullspace_mpi nullspace_mpi.cc)
  target_link_libraries(nullspace_mpi PUBLIC arccore_alina mpi_target)

  if (TARGET Metis::metis)
    target_link_libraries(nullspace_mpi Metis::metis)
  endif ()
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
