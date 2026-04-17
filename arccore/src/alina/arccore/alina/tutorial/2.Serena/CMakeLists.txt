add_executable(serena serena.cc)
target_link_libraries(serena PUBLIC arccore_alina)

if (TARGET mpi_target)
  add_executable(serena_mpi serena_mpi.cc)
  target_link_libraries(serena_mpi PUBLIC arccore_alina mpi_target)

  if (TARGET Metis::metis)
    target_link_libraries(serena_mpi Metis::metis)
  endif ()
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
