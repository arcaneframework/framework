
set(ARCCORE_ALINA_BLOCK_SIZES "3;4" CACHE STRING "Block sizes for block solvers in examples")
set(ARCCORE_ALINA_BLOCK_SIZES_SEQ "")
foreach(B ${ARCCORE_ALINA_BLOCK_SIZES})
  set(ARCCORE_ALINA_BLOCK_SIZES_SEQ "${ARCCORE_ALINA_BLOCK_SIZES_SEQ}(${B})")
endforeach()

# Add a library for common samples functions and types
add_library(arccore_alina_samples_lib
  AlinaSamplesCommon.h
  AlinaSamplesCommon.cc
)
arcane_target_set_standard_path(arccore_alina_samples_lib)
target_link_libraries(arccore_alina_samples_lib PRIVATE arcane_launcher arcane_full)
target_link_libraries(arccore_alina_samples_lib PUBLIC arccore_alina)
target_include_directories(arccore_alina_samples_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

set(ALINA_SAMPLE_BUILD_DIR "${CMAKE_BINARY_DIR}/alina_samples")
function(add_arccore_alina_example_no_test EXAMPLE SOURCES)
  add_executable(${EXAMPLE} ${SOURCES})
  set_target_properties(${EXAMPLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${ALINA_SAMPLE_BUILD_DIR}")
  target_link_libraries(${EXAMPLE} PUBLIC arccore_alina_samples_lib Boost::program_options)
  target_include_directories(${EXAMPLE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../tests")
  target_compile_definitions(${EXAMPLE} PUBLIC "ARCCORE_ALINA_BLOCK_SIZES=${ARCCORE_ALINA_BLOCK_SIZES_SEQ}")
endfunction()

# NOTE:
# Certains tests utilisent OpenMP
# On fixe en dur le nombre de threads sinon on utilise tous
# les coeurs disponibles par défaut.
function(add_arccore_alina_example EXAMPLE SOURCES)
  add_arccore_alina_example_no_test(${EXAMPLE} ${SOURCES})
  set(TEST_NAME "alina_sample_seq_${EXAMPLE}")
  add_test(NAME ${TEST_NAME}
    COMMAND ${EXAMPLE} ${ARGN}
    WORKING_DIRECTORY "${ALINA_SAMPLE_BUILD_DIR}"
  )
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "ARCANE_NB_TASK=2")
endfunction()

add_arccore_alina_example_no_test(mm2bin MatrixMarkerToBinary.cc)
add_arccore_alina_example_no_test(bin2mm BinaryToMatrixMarket.cc)

arcane_find_package(Hypre)

add_arccore_alina_example(basic_solver BasicSolver.cc)
target_sources(basic_solver PUBLIC HypreComparer.cc)
arcane_add_arccon_packages(basic_solver PRIVATE Hypre)

add_arccore_alina_example(solver Solver.cc)
target_sources(solver PUBLIC HypreComparer.cc)
arcane_add_arccon_packages(solver PRIVATE Hypre)

add_arccore_alina_example(solver_complex SolverComplex.cc)
add_arccore_alina_example(crs_builder AdapterCSRBuilder.cc)
add_arccore_alina_example(custom_adapter CustomAdapter.cc)
add_arccore_alina_example(mixed_precision MixedPrecision.cc)

set(SAMPLE_MATRIX_ARGS "--matrix" "${CMAKE_CURRENT_SOURCE_DIR}/../tests/bcsstk02.mtx")

add_arccore_alina_example_no_test(block_crs BackendBlockCSR.cc)
add_arccore_alina_example(cpr CPR.cc ${SAMPLE_MATRIX_ARGS})
add_arccore_alina_example(cpr_drs CPRDynamicRowSum.cc ${SAMPLE_MATRIX_ARGS})
add_arccore_alina_example(ns_search NonScalarSearch.cc ${SAMPLE_MATRIX_ARGS})
add_arccore_alina_example_no_test(schur_pressure_correction SchurPressureCorrection.cc)
add_arccore_alina_example_no_test(schurpc_mixed SCHURPreconditionerMixed.cc)
add_arccore_alina_example_no_test(deflated_solver DeflatedSolver.cc)
add_arccore_alina_example(poisson2d Poisson2D.cc)

if (TARGET eigen_target)
  add_arccore_alina_example(solver_eigen Solver.cc)
  target_link_libraries(solver_eigen PUBLIC eigen_target)
  target_compile_definitions(solver_eigen PRIVATE SOLVER_BACKEND_EIGEN)

  add_arccore_alina_example_no_test(schurpc_mixed_eigen SCHURPreconditionerMixed.cc)
  target_link_libraries(schurpc_mixed_eigen PUBLIC eigen_target)
  target_compile_definitions(schurpc_mixed_eigen PRIVATE BLOCK_TYPE_EIGEN)
endif()

add_arccore_alina_example(call_lib AlinaLibraryUsage.cc)

if (TARGET cuda_target)
  enable_language(CUDA)
  function(add_cuda_example example)
    add_executable(${example}_cuda ${example}.cc)
    set_source_files_properties(${example}.cc PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(${example}_cuda PUBLIC SOLVER_BACKEND_CUDA)
    target_include_directories(${example}_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../tests" )
    target_link_libraries(${example}_cuda PUBLIC arccore_alina cuda_target arccore_cuda_compile_flags Boost::program_options)
    add_test(NAME alina_${example}_cuda COMMAND ${example}_cuda ${SAMPLE_MATRIX_ARGS})
  endfunction()

  add_cuda_example(solver)
  add_cuda_example(schur_pressure_correction)
  add_cuda_example(cpr_drs)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Handle samples using MPI

add_library(arccore_alina_mpi_example INTERFACE)
target_link_libraries(arccore_alina_mpi_example INTERFACE
  arccore_alina ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_compile_definitions(arccore_alina_mpi_example INTERFACE "ARCCORE_ALINA_BLOCK_SIZES=${ARCCORE_ALINA_BLOCK_SIZES_SEQ}")

if (TARGET eigen_target)
  target_link_libraries(arccore_alina_mpi_example INTERFACE eigen_target)
endif()

function(add_mpi_example_test target test_name nb_core)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${ALINA_SAMPLE_BUILD_DIR}")
  target_link_libraries(${target} PUBLIC arccore_alina_samples_lib arccore_alina_mpi_example)
  set(TEST_NAME "alina_sample_mpi${nb_core}_${test_name}")
  add_test(NAME ${TEST_NAME}
    WORKING_DIRECTORY "${ALINA_SAMPLE_BUILD_DIR}"
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nb_core} ./${target} ${ARGN}
  )
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "ARCANE_NB_TASK=2")
endfunction()

function(add_mpi_example example sources nb_core)
  add_executable(${example} ${sources} ${ARGN})
  add_mpi_example_test(${example} ${example} ${nb_core})
endfunction()

function(add_mpi_example_no_test example sources)
  add_executable(${example} ${sources})
  target_link_libraries(${example} PUBLIC arccore_alina_mpi_example)
endfunction()

# Il faut mettre 1 seul coeur pour certains tests
# car ils utilisent openmp et sinon on se retrouve avec trop
# de processus

add_mpi_example(check_direct DistributedCheckDirect.cc 4)
add_mpi_example(runtime_sdd DistributedRuntimeSDD.cc 2)
add_mpi_example(runtime_sdd_3d DistributedRuntimeSDD3D.cc 2)
add_mpi_example(runtime_bp DistributedRuntimeBP.cc 2)
add_mpi_example(call_mpi_lib DistributedAlinaLibraryUsage.cc 2)
add_mpi_example(test_transpose DistributedTranspose.cc 4)
add_mpi_example(test_spmm DistributedSPMM.cc 4)
add_mpi_example(spmm_scaling DistributedSPMMScaling.cc 4)
add_mpi_example(mpi_solver DistributedSolver.cc 2)
add_mpi_example(mpi_basic_solver DistributedBasicSolver.cc 2)
add_mpi_example(mpi_complex DistributedComplex.cc 2)

add_mpi_example_test(mpi_basic_solver mpi_basic_solver_spectral_radius1 2 -p precond.coarsening.estimate_spectral_radius=1)
add_mpi_example_test(mpi_basic_solver mpi_basic_solver_spectral_radius2 2 -p precond.coarsening.estimate_spectral_radius=1 -p precond.coarsening.power_iters=4)

# Il faut un argument 'matrix' pour le test
add_mpi_example_no_test(schur_pc_mpi DistributedSCHUR.cc)

# Il faut un fichir de partition pour le test
add_mpi_example_no_test(solve_mm_mpi DistributedMatrixMarketSolver.cc)

# Exception dans boost lors du test
add_mpi_example_no_test(cpr_mpi DistributedCPR.cc)

if (TARGET arccon::Parmetis)
  add_mpi_example_no_test(partition DistributedPartition.cc 2)
  target_link_libraries(partition PUBLIC arccon::Parmetis ${Boost_PROGRAM_OPTIONS_LIBRARY})
endif()

if (TARGET cuda_target)
  foreach(example runtime_sdd runtime_sdd_3d schur_pc_mpi mpi_solver)
    add_executable(${example}_cuda ${example}.cc)
    set_source_files_properties(${example}.cc PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(${example}_cuda PUBLIC SOLVER_BACKEND_CUDA)
    target_link_libraries(${example}_cuda PUBLIC
      arccore_alina
      cuda_target
      arccore_cuda_compile_flags
      Boost::program_options
    )
  endforeach()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/plot_result
  ${CMAKE_CURRENT_BINARY_DIR}/plot_result
  COPYONLY
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
