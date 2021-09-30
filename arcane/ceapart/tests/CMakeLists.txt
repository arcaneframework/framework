message(STATUS "Tests for CEA part")
message(STATUS "ARCANE_TEST_DRIVER=${ARCANE_TEST_DRIVER}")
SET(TEST_PATH ${ARCANE_CEA_SOURCE_PATH}/tests)
if(NOT ARCANE_CEA_MESH_PATH)
  SET(ARCANE_CEA_MESH_PATH ${ARCANE_CEA_SOURCE_PATH}/meshes)
endif()

add_executable(arcanecea_tests_exec ${Arcane_SOURCE_DIR}/src/arcane/tests/ArcaneTestMain.cc)
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

if (LIMA_FOUND)
  ARCANE_ADD_TEST(lima_mesh_mli lima-1.arc "-m 10")
  ARCANE_ADD_TEST(lima_mesh_unf lima-2.arc "-m 10")
  ARCANE_ADD_TEST(lima_hydro lima-hydro.arc "-m 10")

  ARCANE_ADD_TEST_PARALLEL(lima_mesh_mli_1proc_rep4 lima-1.arc 4 -m 10 -R 4)
  ARCANE_ADD_TEST_PARALLEL(lima_mesh_mli_3proc_rep4 lima-1.arc 12 -m 10 -R 4)

  ARCANE_ADD_TEST_PARALLEL(lima_mesh_unf_1proc_rep4 lima-2.arc 4 -m 10 -R 4)
  ARCANE_ADD_TEST_PARALLEL(lima_mesh_unf_3proc_rep4 lima-2.arc 12 -m 10 -R 4)

  ARCANE_ADD_TEST_PARALLEL(lima_hydro-1proc_rep4 lima-hydro.arc 4 -m 10 -R 4)
  ARCANE_ADD_TEST_PARALLEL(lima_hydro-3proc_rep4 lima-hydro.arc 12 -m 10 -R 4)

  ARCANE_ADD_TEST(lima_ev6 lima-3.arc "-m 10")
  ARCANE_ADD_TEST(lima_ev6_mli2 lima-4.arc "-m 10")
endif ()

if (LIMA_FOUND)
  ARCANE_ADD_TEST(raymesh_intersection_1 testRayMeshIntersection-1.arc "-m 2")
endif()

# Pour l'instant, les soudures ne sont pas compatibles
# avec le partitionnement parallele lorqu'on utilise le partitionnement interne
if (LIMA_FOUND AND ARCANE_ADDITIONAL_MESHES_PATH)
  ARCANE_ADD_TEST(tied_interface3 testTiedInterface-3.arc "-m 10")
  ARCANE_ADD_TEST(tied_interface_multi1 testTiedInterface-4.arc "-m 2")
  ARCANE_ADD_TEST_PARALLEL(tied_interface3_lb testLoadBalance-tiedinterface-1.arc 4 "-m 10")
endif()

ARCANE_ADD_TEST_PARALLEL(splitsd1 testPartition-SplitSD.arc 4)
ARCANE_ADD_TEST_PARALLEL(loadbalance_splitsd1 testLoadBalanceHydro-SplitSD.arc 4 "-m 30")

if (PARMETIS_FOUND AND LIMA_FOUND)
  ARCANE_ADD_TEST_PARALLEL(case_part_metis_mli testCasePart-1.arc 4)
  ARCANE_ADD_TEST_PARALLEL(case_part_metis_2_mli testCasePart-2.arc 4)
endif ()

ARCANE_ADD_TEST(material1 testMaterial-1.arc "-m 10")
ARCANE_ADD_TEST_SEQUENTIAL(material1_task4 testMaterial-1.arc "-K 4" "-m 10")

# Les trois tests suivants doivent faire le même nombre d'itérations et avoir le
# même nombre de mailles (pour test)
ARCANE_ADD_TEST(material2 testMaterial-2.arc "-m 13")
ARCANE_ADD_TEST(material2_opt1 testMaterial-2-opt1.arc "-m 13")
ARCANE_ADD_TEST(material2_opt3 testMaterial-2-opt3.arc "-m 13")
ARCANE_ADD_TEST(material2_opt5 testMaterial-2-opt5.arc "-m 13")
ARCANE_ADD_TEST(material2_opt7 testMaterial-2-opt7.arc "-m 13")
ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv2 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,2)
ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv3 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,3)
ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv6 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,6)
ARCANE_ADD_TEST_CHECKPOINT_SEQUENTIAL(material_checkpoint testMaterial-checkpoint.arc 3 3)
ARCANE_ADD_TEST_CHECKPOINT_SEQUENTIAL(material_checkpoint_recreate testMaterial-checkpoint-recreate.arc 3 3)

ARCANE_ADD_TEST_SEQUENTIAL(material2_task4 testMaterial-2task.arc "-K 4" "-m 1")

ARCANE_ADD_TEST(material_sync1 testMaterial-sync-1.arc)
ARCANE_ADD_TEST_PARALLEL_THREAD(material_sync1 testMaterial-sync-1.arc 4)
ARCANE_ADD_TEST_PARALLEL(material_sync2 testMaterial-sync-2.arc 4)
ARCANE_ADD_TEST_PARALLEL(material_sync2_v3 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,3)

ARCANE_ADD_TEST(material3 testMaterial-3.arc "-m 20")
# NOTE Ajoute test optmisation uniquement en sequentiel car pour l'instant cela
# ne marche pas en parallele a cause de la suppression de mailles.
ARCANE_ADD_TEST_SEQUENTIAL(material3_opt1 testMaterial-3-opt1.arc "-m 20")
ARCANE_ADD_TEST_SEQUENTIAL(material3_opt3 testMaterial-3-opt3.arc "-m 20")
ARCANE_ADD_TEST_SEQUENTIAL(material3_opt5 testMaterial-3-opt5.arc "-m 20")
ARCANE_ADD_TEST_SEQUENTIAL(material3_opt7 testMaterial-3-opt7.arc "-m 20")
ARCANE_ADD_TEST_SEQUENTIAL(material3_opt7_trace testMaterial-3-opt7.arc "-m 20" "-We,ARCANE_TRACE_ENUMERATOR,1")

ARCANE_ADD_TEST_PARALLEL(material3_opt7_lb testMaterial-3-opt7-lb.arc 4 "-m 20")

ARCANE_ADD_TEST_SEQUENTIAL(material1_simd1 testMaterialSimd-1.arc)

ARCANE_ADD_TEST(cartesian1 testCartesianMesh-1.arc "-m 20")
ARCANE_ADD_TEST(cartesian2D-1 testCartesianMesh2D-1.arc "-m 20")
ARCANE_ADD_TEST_PARALLEL(cartesian2D-1_repart testCartesianMesh2D-1.arc 4 "-m 20" "-We,TEST_PARTITIONING,1")
if (LIMA_FOUND)
  ARCANE_ADD_TEST(cartesian2D-lima testCartesianMesh2D-2.arc "-m 20")
endif()
ARCANE_ADD_TEST_SEQUENTIAL(adiadvection-1 testAdiAdvection-1.arc "-m 20")
arcane_add_test(amr-cartesian2D-1 testAMRCartesianMesh2D-1.arc "-m 20")
arcane_add_test(amr-checkpoint-cartesian2D-1 testAMRCartesianMesh2D-1.arc -c 3 -m 5)


#################################
# CARTESIAN MESH GENERATOR TEST #
#################################
# Ces tests ont besoin d'Aleph
# TODO: faire des tests sans avoir besoin de Aleph
if (TARGET arcane_aleph_hypre)
  ARCANE_ADD_TEST_SEQUENTIAL(cartesianMeshGenerator testCartesianMeshGenerator.arc "-m 1")
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGenerator testCartesianMeshGenerator.arc 8 "-m 1")

  ARCANE_ADD_TEST_SEQUENTIAL(cartesianMeshGenerator2D testCartesianMeshGenerator2D.arc)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGenerator2D testCartesianMeshGenerator2D.arc 4)

  ARCANE_ADD_TEST_SEQUENTIAL(cartesianMeshGenerator2D-meshservice testCartesianMeshGenerator2D-meshservice.arc)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGenerator2D-meshservice testCartesianMeshGenerator2D-meshservice.arc 4)

  ARCANE_ADD_TEST_SEQUENTIAL(cartesianMeshGeneratorModulo testCartesianMeshGeneratorModulo.arc)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGeneratorModulo testCartesianMeshGeneratorModulo.arc 3)

  ARCANE_ADD_TEST_SEQUENTIAL(cartesianMeshGeneratorOrigin testCartesianMeshGeneratorOrigin.arc)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGeneratorOrigin testCartesianMeshGeneratorOrigin.arc 4)

  ARCANE_ADD_TEST_SEQUENTIAL(cartesianMeshGeneratorOrigin2D testCartesianMeshGeneratorOrigin2D.arc)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGeneratorOrigin2D testCartesianMeshGeneratorOrigin2D.arc 4)
endif()


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

# Pas actif pour l'instant car en échec
arcane_add_test_sequential(cartesianpatch1 unitCartesianPatch1.arc)

#####################
# HyODA Mixed Cells #
#####################
ARCANE_ADD_TEST_SEQUENTIAL(hyoda_mixed_cells testHyodaMixedCells.arc)

# Fonction pour recopier les maillages issus de répertoires
# qui ne sont pas dans les dépots arcane (pour des raisons de taille)
function(ARCANE_CEA_COPY_MESH mesh_name)
  if (ARCANE_ADDITIONAL_MESHES_PATH)
    exec_program(${CMAKE_COMMAND} ARGS -E copy_if_different ${ARCANE_ADDITIONAL_MESHES_PATH}/${mesh_name}
      ${ARCANE_TEST_PATH}/${mesh_name})
  endif()
endfunction()
function(ARCANE_CEA_COPY_MESH_DIRECTORY mesh_dir_name)
  if (ARCANE_ADDITIONAL_MESHES_PATH)
    exec_program(${CMAKE_COMMAND} ARGS -E copy_directory ${ARCANE_ADDITIONAL_MESHES_PATH}/${mesh_dir_name}
      ${ARCANE_TEST_PATH}/${mesh_dir_name})
  endif()
endfunction()

if (LIMA_FOUND)
  ARCANE_CEA_COPY_MESH(tube5x5x100.mli)
  ARCANE_CEA_COPY_MESH(tube5x5x100.unf)
  ARCANE_CEA_COPY_MESH(tube2x2x4.unf)
  ARCANE_CEA_COPY_MESH(tube3x3x10.unf)
  ARCANE_CEA_COPY_MESH(sphere1.mli)
  ARCANE_CEA_COPY_MESH(sphere2.mli)
  ARCANE_CEA_COPY_MESH(sphere_multi_soudure.unf)
  ARCANE_CEA_COPY_MESH(sod_cart_lima.unf)
  #ARCANE_CEA_COPY_MESH(soudures_3.mli)
  ARCANE_CEA_COPY_MESH_DIRECTORY(sphere_multi_soudure_cut_4)
  #ARCANE_CEA_COPY_MESH_DIRECTORY(soudures_3_cut_4)
endif()

#############
# GEOMETRIC #
#############
ARCANE_ADD_TEST_SEQUENTIAL(geometric1 testGeometric-1.arc)

###################
# WRAPPING '.Net' #
###################
if (TARGET arcanecea_test_cs)
  message(STATUS "Adding '.Net' tests")
  if (MONO_EXEC)
    arcane_add_test_sequential(material2_cs testMaterial-2-opt7-cs.arc -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} -m 4)
  endif()
  arcane_add_test_sequential(material2_cs_coreclr testMaterial-2-opt7-cs.arc -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} --dotnet-runtime=coreclr -m 4)
  arcane_add_csharp_test_sequential(material3_cs testMaterial-2-opt7-cs.arc -m 4)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
