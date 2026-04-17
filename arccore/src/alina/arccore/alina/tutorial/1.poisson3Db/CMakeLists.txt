add_executable(poisson3Db poisson3Db.cc)
target_link_libraries(poisson3Db PUBLIC arccore_alina)

add_executable(poisson3Db_mixed poisson3Db.cc)
target_link_libraries(poisson3Db_mixed PUBLIC arccore_alina)
target_compile_definitions(poisson3Db_mixed PRIVATE MIXED_PRECISION)

if (TARGET cuda_target)
  enable_language(CUDA)
  add_executable(poisson3Db_cuda poisson3Db_cuda.cu)
  #target_link_libraries(poisson3Db_cuda PUBLIC arccore_alina Arccore::arccore_accelerator)
  target_link_libraries(poisson3Db_cuda PUBLIC arccore_alina cuda_target)
endif()

if(TARGET mpi_target)
  add_executable(poisson3Db_mpi poisson3Db_mpi.cc)
  target_link_libraries(poisson3Db_mpi PUBLIC arccore_alina mpi_target)

  if (TARGET Metis::metis)
    target_link_libraries(poisson3Db_mpi Metis::metis)
  endif ()
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
