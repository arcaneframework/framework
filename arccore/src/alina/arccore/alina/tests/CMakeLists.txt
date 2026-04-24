#----------------------------------------------------------------------------
# Tests for Alina
#----------------------------------------------------------------------------
add_library(arccore_alina_test INTERFACE)

if (NOT Boost_USE_STATIC_LIBS)
  target_compile_definitions(arccore_alina_test INTERFACE BOOST_TEST_DYN_LINK)
endif ()

# ----------------------------------------------------------------------------

set(SOURCE_FILES
  SampleProblemCommon.h
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
if (TARGET eigen_target)
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
target_include_directories(arccore_alina.tests INTERFACE ${Boost_INCLUDE_DIRS})

target_link_libraries(arccore_alina.tests PUBLIC
  arccore_alina
  arccore_accelerator
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  GTest::GTest
  GTest::Main
)

if (TARGET eigen_target)
  target_link_libraries(arccore_alina.tests PUBLIC eigen_target)
endif()
if (ARCCORE_ENABLE_TBB)
  target_compile_definitions(arccore_alina.tests PRIVATE ARCCORE_HAS_ACCELERATOR_THREAD)
endif()


# NOTE:
# Certains tests utilisent OpenMP
# On fixe en dur le nombre de threads sinon on utilise tous
# les coeurs disponibles.

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
target_include_directories(arccore_alina_mpi.tests PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(arccore_alina_mpi.tests PRIVATE arccore_alina GTest::GTest)

# Comme il faut lancer les tests avec 'mpiexec', on n'utilise pas 'gtest_discover_tests'
# mais on ajoute manuellement les tests via la macro 'alina_add_test'

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
