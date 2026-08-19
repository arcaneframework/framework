#----------------------------------------------------------------------------
# Tests for Alina
#----------------------------------------------------------------------------
add_library(arccore_alina_test INTERFACE)

#----------------------------------------------------------------------------
# Add test if Eigen is found
#----------------------------------------------------------------------------
find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)
if (EIGEN_INCLUDE)
  add_library(alina_eigen_target INTERFACE)
  target_include_directories(alina_eigen_target INTERFACE ${EIGEN_INCLUDE})
  target_compile_definitions(alina_eigen_target INTERFACE ARCCORE_ALINA_HAVE_EIGEN)
endif()

# ----------------------------------------------------------------------------

set(SOURCE_FILES
  SampleProblemCommon.h
  TestAlinaLib.cc
  TestCSRMatrixView.cc
  TestQR.cc
  TestSkylineLU.cc
  TestComplexErf.cc
  TestSolverBuiltin.cc
  TestSolverComplex.cc
  TestSolverBlockCSR.cc
  TestSolverBuiltinNonScalar.cc
  TestIO.cc
  TestStaticMatrix.cc
)
if (TARGET alina_eigen_target)
  list(APPEND SOURCE_FILES
    TestEigenSolver.cc
    TestSolverEigen.cc
    TestSolverEigenNonScalar.cc
  )
endif()

set(ACCELERATOR_FILES
)

arccore_add_component_test_executable(alina
  FILES ${SOURCE_FILES} ${ACCELERATOR_FILES}
)

target_link_libraries(arccore_alina.tests PUBLIC
  arccore_alina
  arccore_accelerator
  GTest::GTest
  GTest::Main
)

if (TARGET alina_eigen_target)
  target_link_libraries(arccore_alina.tests PUBLIC alina_eigen_target)
endif()
if (ARCCORE_ENABLE_TBB)
  target_compile_definitions(arccore_alina.tests PRIVATE ARCCORE_HAS_ACCELERATOR_THREAD)
endif()


# NOTE:
# Some tests use OpenMP
# We hardcode the number of threads, otherwise we use all
# available cores.

gtest_discover_tests(arccore_alina.tests
  DISCOVERY_TIMEOUT 30
)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Tests using MPI

set(MPI_SOURCE_FILES
  TestMainMpi.h
  TestMainMpi.cc
  TestDistributedSolver.cc
)
arccore_add_component_test_executable(alina_mpi
  FILES
  ${MPI_SOURCE_FILES}
  )
target_link_libraries(arccore_alina_mpi.tests PRIVATE arccore_alina GTest::GTest)

# Since the tests must be run with 'mpiexec', we do not use 'gtest_discover_tests'
# but we manually add the tests via the 'alina_add_test' macro

macro(alina_mpi_add_test)
  set(options)
  set(oneValueArgs TEST_NAME NB_PROC)
  set(multiValueArgs)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_test(
    NAME alina_test_mpi.${ARGS_TEST_NAME}-mpi${ARGS_NB_PROC}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARGS_NB_PROC} ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:arccore_alina_mpi.tests> ${MPIEXEC_POSTFLAGS} "--gtest_filter=alina_test_mpi.${ARGS_TEST_NAME}"
  )
endmacro()

alina_mpi_add_test(TEST_NAME BasicSolver NB_PROC 1)
alina_mpi_add_test(TEST_NAME BasicSolver NB_PROC 2)
alina_mpi_add_test(TEST_NAME BasicSolver NB_PROC 4)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
