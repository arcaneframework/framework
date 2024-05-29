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

if (Lima_FOUND)
  ARCANE_ADD_TEST(lima_mesh_mli lima-1.arc "-m 10")
  ARCANE_ADD_TEST(lima_mesh_unf lima-2.arc "-m 10")
  ARCANE_ADD_TEST(lima_hydro lima-hydro.arc "-m 10")
  ARCANE_ADD_TEST(lima_hydro_meshservice lima-hydro-meshservice.arc "-m 10")

  ARCANE_ADD_TEST_PARALLEL(lima_mesh_mli_1proc_rep4 lima-1.arc 4 -m 10 -R 4)
  ARCANE_ADD_TEST_PARALLEL(lima_mesh_mli_3proc_rep4 lima-1.arc 12 -m 10 -R 4)

  ARCANE_ADD_TEST_PARALLEL(lima_mesh_unf_1proc_rep4 lima-2.arc 4 -m 10 -R 4)
  ARCANE_ADD_TEST_PARALLEL(lima_mesh_unf_3proc_rep4 lima-2.arc 12 -m 10 -R 4)

  ARCANE_ADD_TEST_PARALLEL(lima_hydro-1proc_rep4 lima-hydro.arc 4 -m 10 -R 4)
  ARCANE_ADD_TEST_PARALLEL(lima_hydro-3proc_rep4 lima-hydro.arc 12 -m 10 -R 4)

  ARCANE_ADD_TEST(lima_ev6 lima-3.arc "-m 10")
  ARCANE_ADD_TEST(lima_ev6_mli2 lima-4.arc "-m 10")
endif ()

if (Lima_FOUND)
  ARCANE_ADD_TEST(raymesh_intersection_1 testRayMeshIntersection-1.arc "-m 2")
endif()

# Pour l'instant, les soudures ne sont pas compatibles
# avec le partitionnement parallele lorqu'on utilise le partitionnement interne
if (Lima_FOUND AND ARCANE_ADDITIONAL_MESHES_PATH)
  ARCANE_ADD_TEST(tied_interface3 testTiedInterface-3.arc "-m 10")
  ARCANE_ADD_TEST(tied_interface_multi1 testTiedInterface-4.arc "-m 2")
  ARCANE_ADD_TEST_PARALLEL(tied_interface3_lb testLoadBalance-tiedinterface-1.arc 4 "-m 10")
endif()

ARCANE_ADD_TEST_PARALLEL(splitsd1 testPartition-SplitSD.arc 4)
ARCANE_ADD_TEST_PARALLEL(loadbalance_splitsd1 testLoadBalanceHydro-SplitSD.arc 4 "-m 30")

if (Parmetis_FOUND AND Lima_FOUND)
  ARCANE_ADD_TEST_PARALLEL(case_part_metis_mli testCasePart-1.arc 4)
  ARCANE_ADD_TEST_PARALLEL(case_part_metis_2_mli testCasePart-2.arc 4)
endif ()

if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test(material1 testMaterial-1.arc "-m 10" "-We,ARCANE_MATERIALSYNCHRONIZER_ACCELERATOR_MODE,1")
  arcane_add_test_parallel_thread(material1 testMaterial-1.arc 4 "-m 10" "-We,ARCANE_MATERIALSYNCHRONIZER_ACCELERATOR_MODE,1")
  arcane_add_test_parallel(material1_legacy_sync testMaterial-1.arc 4 "-m 10" "-We,ARCANE_MATERIAL_LEGACY_SYNCHRONIZE,1")
  if (LZ4_FOUND)
    arcane_add_test_sequential(material1_lz4 testMaterial-1.arc "-m 10" "-We,ARCANE_MATERIAL_DATA_COMPRESSOR_NAME,LZ4DataCompressor")
  endif()
  arcane_add_test_sequential_task(material1 testMaterial-1.arc 4 "-m 10")

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
  ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv7 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  ARCANE_ADD_TEST_PARALLEL(material2_opt3_syncv8 testMaterial-2-opt3.arc 4 -m 13 -We,ARCANE_MATSYNCHRONIZE_VERSION,8)
  ARCANE_ADD_TEST_CHECKPOINT_SEQUENTIAL(material_checkpoint testMaterial-checkpoint.arc 3 3)
  ARCANE_ADD_TEST_CHECKPOINT_SEQUENTIAL(material_checkpoint_recreate testMaterial-checkpoint-recreate.arc 3 3)

  arcane_add_test_sequential_task(material2 testMaterial-2task.arc 4 "-m 1")

  ARCANE_ADD_TEST(material_sync1 testMaterial-sync-1.arc)
  ARCANE_ADD_TEST_PARALLEL_THREAD(material_sync1 testMaterial-sync-1.arc 4)
  ARCANE_ADD_TEST_PARALLEL(material_sync2 testMaterial-sync-2.arc 4)
  ARCANE_ADD_TEST_PARALLEL(material_sync2_v3 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,3)
  arcane_add_test(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  arcane_add_test_parallel_thread(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  arcane_add_test_message_passing_hybrid(material_sync2_v7 CASE_FILE testMaterial-sync-2.arc NB_MPI 3 NB_SHM 4 ARGS -We,ARCANE_MATSYNCHRONIZE_VERSION,7)

  arcane_add_accelerator_test_parallel(material_sync2_v6 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,6)
  arcane_add_accelerator_test_parallel(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  arcane_add_accelerator_test_parallel(material_sync2_v8 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,8)
  arcane_add_accelerator_test_parallel_thread(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)

  ARCANE_ADD_TEST(material3 testMaterial-3.arc "-m 20")
  # NOTE Ajoute test optmisation uniquement en sequentiel car pour l'instant cela
  # ne marche pas en parallele a cause de la suppression de mailles.
  ARCANE_ADD_TEST_SEQUENTIAL(material3_opt1 testMaterial-3-opt1.arc "-m 20")
  ARCANE_ADD_TEST_SEQUENTIAL(material3_opt3 testMaterial-3-opt3.arc "-m 20")
  ARCANE_ADD_TEST_SEQUENTIAL(material3_opt5 testMaterial-3-opt5.arc "-m 20")
  ARCANE_ADD_TEST_SEQUENTIAL(material3_opt7 testMaterial-3-opt7.arc "-m 20")
  if(NOT ARCANE_DISABLE_PERFCOUNTER_TESTS)
    arcane_add_test_sequential(material3_opt7_trace testMaterial-3-opt7.arc "-m 20" "-We,ARCANE_TRACE_ENUMERATOR,1")
  endif()

  ARCANE_ADD_TEST_PARALLEL(material3_opt7_lb testMaterial-3-opt7-lb.arc 4 "-m 20")

  ARCANE_ADD_TEST_SEQUENTIAL(material1_simd1 testMaterialSimd-1.arc)
endif()

ARCANE_ADD_TEST(cartesian1 testCartesianMesh-1.arc "-m 20")
arcane_add_accelerator_test_sequential(cartesian1 testCartesianMesh-1.arc "-m 20")
arcane_add_test(cartesian2D-1 testCartesianMesh2D-1.arc "-m 20")
arcane_add_accelerator_test_sequential(cartesian2D-1 testCartesianMesh2D-1.arc "-m 20")
ARCANE_ADD_TEST_PARALLEL(cartesian2D-1_repart testCartesianMesh2D-1.arc 4 "-m 20" "-We,TEST_PARTITIONING,1")
if (Lima_FOUND)
  ARCANE_ADD_TEST(cartesian2D-lima testCartesianMesh2D-2.arc "-m 20")
endif()
arcane_add_test(cartesian2D_coarsen1 testCartesianMesh2D-coarsen-1.arc "-m 20" "-We,ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL,1")
arcane_add_test(cartesian2D_coarsen2 testCartesianMesh2D-coarsen-2.arc "-m 20" "-We,ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL,2")
arcane_add_test(cartesian3D_coarsen2 testCartesianMesh3D-coarsen-2.arc "-m 20" "-We,ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL,2")
if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test_sequential(adiadvection-1 testAdiAdvection-1.arc "-m 20")
  arcane_add_accelerator_test_sequential(adiadvection-1 testAdiAdvection-1.arc "-m 20")
endif()

arcane_add_test(amr-cartesian2D-cell-renumbering-v1-1 testAMRCartesianMesh2D-Cell-RenumberingV1-1.arc "-m 20")
arcane_add_test(amr-cartesian2D-cell-renumbering-v1-2 testAMRCartesianMesh2D-Cell-RenumberingV1-2.arc "-m 20")
arcane_add_test(amr-cartesian2D-cell-renumbering-v4-1 testAMRCartesianMesh2D-Cell-RenumberingV4-1.arc "-m 20")
arcane_add_test(amr-cartesian3D-cell-renumbering-v1-1 testAMRCartesianMesh3D-Cell-RenumberingV1-1.arc "-m 20")
arcane_add_test(amr-cartesian3D-cell-renumbering-v4-1 testAMRCartesianMesh3D-Cell-RenumberingV4-1.arc "-m 20")
arcane_add_test_sequential(amr-cartesian3D-cell-renumbering-v1-2 testAMRCartesianMesh3D-Cell-RenumberingV1-2.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-cell-renumbering-v1-2 testAMRCartesianMesh3D-Cell-RenumberingV1-2.arc 8 "-m 20")

arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-cell-renumbering-v1-1 testAMRCartesianMesh2D-Cell-RenumberingV1-1.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-cell-renumbering-v1-2 testAMRCartesianMesh2D-Cell-RenumberingV1-2.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian3D-cell-renumbering-v1-1 testAMRCartesianMesh3D-Cell-RenumberingV1-1.arc 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-cell-renumbering-v1-2 testAMRCartesianMesh3D-Cell-RenumberingV1-2.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-cell-renumbering-v1-2 testAMRCartesianMesh3D-Cell-RenumberingV1-2.arc 8 3 5)

arcane_add_test(amr-cartesian2D-cell-renumbering-v2-1 testAMRCartesianMesh2D-Cell-RenumberingV2-1.arc "-m 20")
arcane_add_test(amr-cartesian2D-cell-renumbering-v2-2 testAMRCartesianMesh2D-Cell-RenumberingV2-2.arc "-m 20")
arcane_add_test(amr-cartesian3D-cell-renumbering-v2-1 testAMRCartesianMesh3D-Cell-RenumberingV2-1.arc "-m 20")
arcane_add_test_sequential(amr-cartesian3D-cell-renumbering-v2-2 testAMRCartesianMesh3D-Cell-RenumberingV2-2.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-cell-renumbering-v2-2 testAMRCartesianMesh3D-Cell-RenumberingV2-2.arc 8 "-m 20")

arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-cell-renumbering-v2-1 testAMRCartesianMesh2D-Cell-RenumberingV2-1.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-cell-renumbering-v2-2 testAMRCartesianMesh2D-Cell-RenumberingV2-2.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian3D-cell-renumbering-v2-1 testAMRCartesianMesh3D-Cell-RenumberingV2-1.arc 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-cell-renumbering-v2-2 testAMRCartesianMesh3D-Cell-RenumberingV2-2.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-cell-renumbering-v2-2 testAMRCartesianMesh3D-Cell-RenumberingV2-2.arc 8 3 5)

arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-1 testAMRCartesianMesh2D-PatchCartesianMeshOnly-1.arc "-m 20")
arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-2 testAMRCartesianMesh2D-PatchCartesianMeshOnly-2.arc "-m 20")
arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-3 testAMRCartesianMesh2D-PatchCartesianMeshOnly-3.arc "-m 20")
arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-4 testAMRCartesianMesh2D-PatchCartesianMeshOnly-4.arc "-m 20")
arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-5 testAMRCartesianMesh2D-PatchCartesianMeshOnly-5.arc "-m 20")

arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-1 testAMRCartesianMesh2D-PatchCartesianMeshOnly-1.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-2 testAMRCartesianMesh2D-PatchCartesianMeshOnly-2.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-3 testAMRCartesianMesh2D-PatchCartesianMeshOnly-3.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-4 testAMRCartesianMesh2D-PatchCartesianMeshOnly-4.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-5 testAMRCartesianMesh2D-PatchCartesianMeshOnly-5.arc 3 5)

arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-1.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-1.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-2.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-2.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-3.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-3.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-4.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-4.arc 8 "-m 20")

arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-1.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-1.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-2.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-2.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-3.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-3.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-4.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-4.arc 8 3 5)

arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-1 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-1.arc "-m 20")
arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-2 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-2.arc "-m 20")
arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-3 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-3.arc "-m 20")
arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-4 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-4.arc "-m 20")
arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-5 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-5.arc "-m 20")
arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-6 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-6.arc "-m 20")
arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc "-m 20")
arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-8 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-8.arc "-m 20")

arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-1 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-1.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-2 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-2.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-3 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-3.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-4 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-4.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-5 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-5.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-6 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-6.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-8 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-8.arc 3 5)

arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-1.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-1.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-2.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-2.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-3.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-3.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-4.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-4.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-5 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-5.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-5 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-5.arc 4 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-6 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-6.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-6 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-6.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc 8 "-m 20")

arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-1.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-1.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-2.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-2.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-3.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-3.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-4.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-4.arc 8 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-5 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-5.arc 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-6 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-6.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-6 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-6.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc 8 3 5)

arcane_add_test(amr-cartesian2D-V1-coarse-1 testAMRCartesianMesh2D-V1-coarse-1.arc "-m 20" "-We,ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL,2")
arcane_add_test(amr-cartesian2D-V1-coarse-2 testAMRCartesianMesh2D-V1-coarse-2.arc "-m 20")
arcane_add_test(amr-cartesian2D-V1-coarse-3 testAMRCartesianMesh2D-V1-coarse-3.arc "-m 10")
arcane_add_test(amr-cartesian2D-V1-coarse-4 testAMRCartesianMesh2D-V1-coarse-4.arc "-m 10")
arcane_add_test(amr-cartesian3D-V1-coarse-1 testAMRCartesianMesh3D-V1-coarse-1.arc "-m 10" "-We,ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL,2")
arcane_add_test_sequential(amr-cartesian3D-V1-coarse-2 testAMRCartesianMesh3D-V1-coarse-2.arc "-m 10")
arcane_add_test_parallel_thread(amr-cartesian3D-V1-coarse-2 testAMRCartesianMesh3D-V1-coarse-2.arc 8 "-m 10")

# Ce test prend du temps (environ 30 secondes) donc on ne l'active pas par défaut
if (BIG_TEST)
  arcane_add_test_sequential(amr-cartesian3D-V1-coarse-3 testAMRCartesianMesh3D-V1-coarse-3.arc "-m 10")
  arcane_add_test_parallel_thread(amr-cartesian3D-V1-coarse-3 testAMRCartesianMesh3D-V1-coarse-3.arc 8 "-m 10")
endif()

# Tests sur le renumérotation des faces en cartésien
# L'idéal est d'avoir pas mal de sous-domaines pour être sur que le
# nombre de sous-domaines dans chaque direction est différent.
# Du coup on utilise le mode mémoire partagée au lieu de MPI pour
# que les tests fonctionnent bien même si la machine n'a pas beaucoup
# de coeurs.
arcane_add_test_sequential(cartesian2d_face_renumbering testCartesianMesh2D-face-renumbering.arc "-m 10")
arcane_add_test_parallel_thread(cartesian2d_face_renumbering testCartesianMesh2D-face-renumbering.arc 6 "-m 10")
arcane_add_test_sequential(cartesian3d_face_renumbering testCartesianMesh3D-face-renumbering.arc "-m 10")
arcane_add_test_parallel_thread(cartesian3d_face_renumbering testCartesianMesh3D-face-renumbering.arc 12 "-m 10")
arcane_add_test_sequential(cartesian3d_face_renumbering_v3 testCartesianMesh3D-face-renumbering-v3.arc "-m 10")
arcane_add_test_parallel_thread(cartesian3d_face_renumbering_v3 testCartesianMesh3D-face-renumbering-v3.arc 12 "-m 10")
arcane_add_test_sequential(cartesian3d_face_edge_renumbering testCartesianMesh3D-face-edge-renumbering.arc "-m 10")
arcane_add_test_parallel_thread(cartesian3d_face_edge_renumbering testCartesianMesh3D-face-edge-renumbering.arc 12 "-m 10")

# Tests sur le partitionnement avec grille en cartésien
arcane_add_test(cartesian_grid_partitioning testCartesianMesh-grid-partitioning.arc "-m 10")
arcane_add_test_parallel(cartesian_grid_partitioning_6proc testCartesianMesh-grid-partitioning6.arc 6 "-m 10")
arcane_add_test_parallel(cartesian_grid_partitioning_12proc testCartesianMesh-grid-partitioning12.arc 12 "-m 10")

arcane_add_test_sequential(cartesian3d_grid_partitioning testCartesianMesh3D-grid-partitioning.arc "-m 10")
arcane_add_test_parallel(cartesian3d_grid_partitioning_12proc testCartesianMesh3D-grid-partitioning.arc 12 "-m 10")

#################################
# CARTESIAN MESH GENERATOR TEST #
#################################
# Ces tests ont besoin d'Aleph
# TODO: faire des tests sans avoir besoin de Aleph
if (TARGET arcane_aleph_hypre)
  arcane_add_test_parallel(cartesianMeshGenerator testCartesianMeshGenerator.arc 1 "-m 1")
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGenerator testCartesianMeshGenerator.arc 8 "-m 1")

  arcane_add_test_parallel(cartesianMeshGenerator2D testCartesianMeshGenerator2D.arc 1)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGenerator2D testCartesianMeshGenerator2D.arc 4)

  arcane_add_test_parallel(cartesianMeshGenerator2D-meshservice testCartesianMeshGenerator2D-meshservice.arc 1)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGenerator2D-meshservice testCartesianMeshGenerator2D-meshservice.arc 4)

  arcane_add_test_parallel(cartesianMeshGeneratorModulo testCartesianMeshGeneratorModulo.arc 1)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGeneratorModulo testCartesianMeshGeneratorModulo.arc 3)

  arcane_add_test_parallel(cartesianMeshGeneratorOrigin testCartesianMeshGeneratorOrigin.arc 1)
  ARCANE_ADD_TEST_PARALLEL(cartesianMeshGeneratorOrigin testCartesianMeshGeneratorOrigin.arc 4)

  arcane_add_test_parallel(cartesianMeshGeneratorOrigin2D testCartesianMeshGeneratorOrigin2D.arc 1)
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

if (Lima_FOUND)
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

# Material Heat
foreach(test_index 1 2 3 4)
  foreach(opt_level 0 3 7 11 15)
    set(TEST_MODIFICATION_FLAG ${opt_level})

    # Test sans équilibrage
    set(TEST_ACTIVE_LOAD_BALANCE "false")
    set(_TEST_FILENAME "${ARCANE_TEST_PATH}/testMaterialHeat-${test_index}-opt${opt_level}.arc")
    configure_file("testMaterialHeat-${test_index}.arc.in" "${_TEST_FILENAME}")

    # Test avec équilibrage
    set(TEST_ACTIVE_LOAD_BALANCE "true")
    set(_TEST_FILENAME_LB "${ARCANE_TEST_PATH}/testMaterialHeat-${test_index}-opt${opt_level}-lb.arc")
    configure_file("testMaterialHeat-${test_index}.arc.in" "${_TEST_FILENAME_LB}")
    if (ARCANE_HAS_ACCELERATOR_API)
      arcane_add_test_sequential(material_heat${test_index}_opt${opt_level} ${_TEST_FILENAME})
      arcane_add_test_parallel_thread(material_heat${test_index}_opt${opt_level} ${_TEST_FILENAME} 4)
      arcane_add_test(material_heat${test_index}_lb_opt${opt_level} ${_TEST_FILENAME_LB})
      arcane_add_test_parallel_thread(material_heat${test_index}_lb_opt${opt_level} ${_TEST_FILENAME_LB} 4)
    endif()
  endforeach()
endforeach()
if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test_sequential(material_heat2_opt15_2small testMaterialHeat-2-small-opt15.arc "-We,ARCANE_DEBUG_MATERIAL_MODIFIER,2")
  arcane_add_test_sequential(material_heat2_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20")
  arcane_add_accelerator_test_sequential(material_heat2_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20")
  arcane_add_accelerator_test_sequential(material_heat2_accelerator_noqueue
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_MATERIALMNG_USE_QUEUE,0")
endif()

###################
# WRAPPING '.Net' #
###################
# TODO: Utiliser les macros génériques pour tester pour les modes
if (TARGET arcanecea_test_cs)
  message(STATUS "Adding '.Net' tests")
  if (MONO_EXEC)
    arcane_add_test_sequential(material2_cs testMaterial-2-opt7-cs.arc -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} -m 4)
  endif()
  if (TARGET arcane_dotnet_coreclr)
    arcane_add_test_sequential(material2_cs_coreclr testMaterial-2-opt7-cs.arc -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} --dotnet-runtime=coreclr -m 4)
    arcane_add_test_sequential(material_eos_cs_coreclr testMaterial-eos-cs.arc -We,ARCANE_USE_DOTNET_WRAPPER,1 --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} --dotnet-runtime=coreclr -m 4)
  endif()
  arcane_add_csharp_test_sequential(material3_cs testMaterial-2-opt7-cs.arc -m 4)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
