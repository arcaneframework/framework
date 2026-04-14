#message(STATUS "Tests for CEA part")
message(STATUS "ARCANE_TEST_DRIVER=${ARCANE_TEST_DRIVER}")
SET(TEST_PATH ${ARCANE_CEA_SOURCE_PATH}/tests)
if(NOT ARCANE_CEA_MESH_PATH)
  SET(ARCANE_CEA_MESH_PATH ${ARCANE_CEA_SOURCE_PATH}/meshes)
endif()

add_executable(arcanecea_tests_exec ${Arcane_SOURCE_DIR}/src/arcane/tests/ArcaneTestMain.cc)
arcane_target_set_standard_path(arcanecea_tests_exec)

set_target_properties(arcanecea_tests_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
if (WIN32)
  set_target_properties(arcanecea_tests_exec PROPERTIES SUFFIX ".exe")
endif()

target_link_libraries(arcanecea_tests_exec PRIVATE ${ADD_NO_AS_NEEDED} arcane_tests_lib arcane_cea_tests)

set(ARCANE_TEST_PATH ${CMAKE_BINARY_DIR}/src/arcane/tests)
set(ARCANE_TEST_WORKDIR ${ARCANE_TEST_PATH})
set(ARCANE_TEST_CASEPATH ${CMAKE_CURRENT_SOURCE_DIR})
set(ARCANE_TEST_EXECNAME arcanecea_tests_exec)
set(ARCANE_TEST_DOTNET_ASSEMBLY ${LIBRARY_OUTPUT_PATH}/ArcaneCeaTest.dll)
include(${Arcane_SOURCE_DIR}/src/arcane/tests/ArcaneTests.cmake)

# Pour l'instant, les soudures ne sont pas compatibles
# avec le partitionnement parallele lorqu'on utilise le partitionnement interne
if (Lima_FOUND AND ARCANE_ADDITIONAL_MESHES_PATH)
  ARCANE_ADD_TEST(tied_interface3 testTiedInterface-3.arc "-m 10")
  ARCANE_ADD_TEST(tied_interface_multi1 testTiedInterface-4.arc "-m 2")
  ARCANE_ADD_TEST_PARALLEL(tied_interface3_lb testLoadBalance-tiedinterface-1.arc 4 "-m 10")
endif()

ARCANE_ADD_TEST_PARALLEL(splitsd1 testPartition-SplitSD.arc 4)
arcane_add_test_parallel(splitsd1_legacy_partition_init testPartition-SplitSD.arc 4 "-We,ARCANE_USE_LEGACY_INIT_LOADBALANCEMNG,1")
ARCANE_ADD_TEST_PARALLEL(loadbalance_splitsd1 testLoadBalanceHydro-SplitSD.arc 4 "-m 30")

if (Parmetis_FOUND AND Lima_FOUND)
  ARCANE_ADD_TEST_PARALLEL(case_part_metis_mli testCasePart-1.arc 4)
  ARCANE_ADD_TEST_PARALLEL(case_part_metis_2_mli testCasePart-2.arc 4)
endif ()








if (SLOOP_FOUND)
  # ALEPH STD TEST
  ARCANE_ADD_TEST_PARALLEL(aleph_sloop testAlephSloop.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph_sloop testAlephSloop.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph_sloop testAlephSloop.arc 4)
  # ALEPH ODD TEST
  ARCANE_ADD_TEST_PARALLEL(aleph_sloop_odd testAlephSloopOdd.arc 3)
  ARCANE_ADD_TEST_PARALLEL(aleph_sloop_odd testAlephSloopOdd.arc 4)
  ARCANE_ADD_TEST_PARALLEL(aleph_sloop_odd testAlephSloopOdd.arc 7)
  ARCANE_ADD_TEST_PARALLEL(aleph_sloop_odd testAlephSloopOdd.arc 8)
  # ALEPH DELETE TEST
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_sloop testAlephSloopDelete.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_sloop testAlephSloopDelete.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_sloop testAlephSloopDelete.arc 4)
  # ALEPH KAPPA TEST
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_sloop testAlephSloop.arc 2 -arcane_opt nb_sub_domain 1 -arcane_opt idle_service AlephKappa)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_sloop testAlephSloop.arc 4 -arcane_opt nb_sub_domain 2 -arcane_opt idle_service AlephKappa)
endif ()



#####################
# HyODA Mixed Cells #
#####################
if (ARCANE_HAS_ACCELERATOR_API)
  ARCANE_ADD_TEST_SEQUENTIAL(hyoda_mixed_cells testHyodaMixedCells.arc)
endif()

# Fonction pour recopier les maillages issus de répertoires
# qui ne sont pas dans les dépots arcane (pour des raisons de taille)
function(ARCANE_CEA_COPY_MESH mesh_name)
  if (ARCANE_ADDITIONAL_MESHES_PATH)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ARCANE_ADDITIONAL_MESHES_PATH}/${mesh_name}
      ${ARCANE_TEST_PATH}/${mesh_name})
  endif()
endfunction()
function(ARCANE_CEA_COPY_MESH_DIRECTORY mesh_dir_name)
  if (ARCANE_ADDITIONAL_MESHES_PATH)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${ARCANE_ADDITIONAL_MESHES_PATH}/${mesh_dir_name}
      ${ARCANE_TEST_PATH}/${mesh_dir_name})
  endif()
endfunction()

#############
# GEOMETRIC #
#############
ARCANE_ADD_TEST_SEQUENTIAL(geometric1 testGeometric-1.arc)




# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
