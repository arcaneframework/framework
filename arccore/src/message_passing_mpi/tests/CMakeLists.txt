arccore_add_component_test_executable(message_passing_mpi
  FILES
  TestMain.h
  TestMain.cc
  TestMpi.cc
  )

target_link_libraries(arccore_message_passing_mpi.tests PUBLIC arccore_message_passing_mpi GTest::GTest)

# Comme il faut lancer les tests avec 'mpiexec', on n'utilise pas 'gtest_discover_tests'
# mais on ajoute manuellement les tests via la macro 'mp_add_test'

macro(mp_add_test)
  set(options)
  set(oneValueArgs TEST_NAME NB_PROC)
  set(multiValueArgs)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_test(
    NAME MessagePassingMpi.${ARGS_TEST_NAME}-mpi${ARGS_NB_PROC}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARGS_NB_PROC} ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:arccore_message_passing_mpi.tests> ${MPIEXEC_POSTFLAGS} "--gtest_filter=MessagePassingMpi.${ARGS_TEST_NAME}"
  )
endmacro()

mp_add_test(TEST_NAME Simple NB_PROC 1)
mp_add_test(TEST_NAME Simple NB_PROC 2)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
