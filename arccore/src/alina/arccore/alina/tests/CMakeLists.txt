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
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
