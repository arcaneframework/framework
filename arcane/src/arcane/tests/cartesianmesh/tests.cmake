
set(ARCANE_TEST_CASEPATH ${TEST_PATH}/cartesianmesh)


arcane_add_test(cartesian1 testCartesianMesh-1.arc "-m 20")
arcane_add_accelerator_test_sequential(cartesian1 testCartesianMesh-1.arc "-m 20")
arcane_add_test(cartesian2D-1 testCartesianMesh2D-1.arc "-m 20")
arcane_add_accelerator_test_sequential(cartesian2D-1 testCartesianMesh2D-1.arc "-m 20")
arcane_add_test_parallel(cartesian2D-1_repart testCartesianMesh2D-1.arc 4 "-m 20" "-We,TEST_PARTITIONING,1")
#if (Lima_FOUND)
#  arcane_add_test(cartesian2D-lima testCartesianMesh2D-2.arc "-m 20")
#endif()

arcane_add_test_sequential(cartesian2D-3 testCartesianMesh2D-3.arc "-m 20")
arcane_add_test_parallel(cartesian2D-3 testCartesianMesh2D-3.arc 2 "-m 20")
arcane_add_accelerator_test_sequential(cartesian2D-3 testCartesianMesh2D-3.arc "-m 20")

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
arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-6 testAMRCartesianMesh2D-PatchCartesianMeshOnly-6.arc "-m 20")
arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-7 testAMRCartesianMesh2D-PatchCartesianMeshOnly-7.arc "-m 20")

arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-1 testAMRCartesianMesh2D-PatchCartesianMeshOnly-1.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-2 testAMRCartesianMesh2D-PatchCartesianMeshOnly-2.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-3 testAMRCartesianMesh2D-PatchCartesianMeshOnly-3.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-4 testAMRCartesianMesh2D-PatchCartesianMeshOnly-4.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-5 testAMRCartesianMesh2D-PatchCartesianMeshOnly-5.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-6 testAMRCartesianMesh2D-PatchCartesianMeshOnly-6.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-7 testAMRCartesianMesh2D-PatchCartesianMeshOnly-7.arc 3 5)

arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-1.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-1.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-2.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-2.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-3.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-3.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-4.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-4.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-5 testAMRCartesianMesh3D-PatchCartesianMeshOnly-5.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-5 testAMRCartesianMesh3D-PatchCartesianMeshOnly-5.arc 8 "-m 20")

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
arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-9 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-9.arc "-m 20")

arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-1 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-1.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-2 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-2.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-3 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-3.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-4 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-4.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-5 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-5.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-6 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-6.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-8 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-8.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-9 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-9.arc 3 5)

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
if (BIG_TEST)
  arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc "-m 20")
  arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc 8 "-m 20")
endif ()

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
if (BIG_TEST)
  arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc 3 5)
  arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-7 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-7.arc 8 3 5)
endif ()

arcane_add_test(amr-cartesian2D-V1-coarse-1 testAMRCartesianMesh2D-V1-coarse-1.arc "-m 20" "-We,ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL,2")
arcane_add_test(amr-cartesian2D-V1-coarse-2 testAMRCartesianMesh2D-V1-coarse-2.arc "-m 20")
arcane_add_test(amr-cartesian2D-V1-coarse-3 testAMRCartesianMesh2D-V1-coarse-3.arc "-m 10")
arcane_add_test(amr-cartesian2D-V1-coarse-4 testAMRCartesianMesh2D-V1-coarse-4.arc "-m 10")
arcane_add_test(amr-cartesian3D-V1-coarse-1 testAMRCartesianMesh3D-V1-coarse-1.arc "-m 10" "-We,ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL,2")
arcane_add_test_sequential(amr-cartesian3D-V1-coarse-2 testAMRCartesianMesh3D-V1-coarse-2.arc "-m 10")
arcane_add_test_parallel_thread(amr-cartesian3D-V1-coarse-2 testAMRCartesianMesh3D-V1-coarse-2.arc 8 "-m 10")


# Test AMR dé-raffinement avec équilibrage.
# Pas encore actif car ne fonctionne pas
# arcane_add_test_parallel(amr-cartesian2D-V1-coarse-lb-1 testAMRCartesianMesh2D-V1-coarse-lb-1.arc 4 "-m 20")

# Ce test prend du temps (environ 30 secondes) donc on ne l'active pas par défaut
if (BIG_TEST)
  arcane_add_test_sequential(amr-cartesian3D-V1-coarse-3 testAMRCartesianMesh3D-V1-coarse-3.arc "-m 10")
  arcane_add_test_parallel_thread(amr-cartesian3D-V1-coarse-3 testAMRCartesianMesh3D-V1-coarse-3.arc 8 "-m 10")
endif ()


arcane_add_test(amr-cartesian2D-cell-coarse-zone-1 testAMRCartesianMesh2D-Cell-CoarseZone-1.arc "-m 20")
arcane_add_test(amr-cartesian2D-cell-coarse-zone-2 testAMRCartesianMesh2D-Cell-CoarseZone-2.arc "-m 20")
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-cell-coarse-zone-1 testAMRCartesianMesh2D-Cell-CoarseZone-1.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-cell-coarse-zone-2 testAMRCartesianMesh2D-Cell-CoarseZone-2.arc 3 5)

arcane_add_test_sequential(amr-cartesian3D-cell-coarse-zone-1 testAMRCartesianMesh3D-Cell-CoarseZone-1.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-cell-coarse-zone-1 testAMRCartesianMesh3D-Cell-CoarseZone-1.arc 8 "-m 20")

arcane_add_test_sequential(amr-cartesian3D-cell-coarse-zone-3 testAMRCartesianMesh3D-Cell-CoarseZone-3.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-cell-coarse-zone-3 testAMRCartesianMesh3D-Cell-CoarseZone-3.arc 8 "-m 20")

arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-cell-coarse-zone-1 testAMRCartesianMesh3D-Cell-CoarseZone-1.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-cell-coarse-zone-1 testAMRCartesianMesh3D-Cell-CoarseZone-1.arc 8 3 5)

arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-cell-coarse-zone-3 testAMRCartesianMesh3D-Cell-CoarseZone-3.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-cell-coarse-zone-3 testAMRCartesianMesh3D-Cell-CoarseZone-3.arc 8 3 5)


arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh2D-PatchCartesianMeshOnly-CoarseZone-1.arc "-m 20")
arcane_add_test(amr-cartesian2D-patch-cartesian-mesh-only-coarse-zone-2 testAMRCartesianMesh2D-PatchCartesianMeshOnly-CoarseZone-2.arc "-m 20")
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh2D-PatchCartesianMeshOnly-CoarseZone-1.arc 3 5)
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-patch-cartesian-mesh-only-coarse-zone-2 testAMRCartesianMesh2D-PatchCartesianMeshOnly-CoarseZone-2.arc 3 5)


arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-1.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-1.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-coarse-zone-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-2.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-coarse-zone-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-2.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-coarse-zone-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-3.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-coarse-zone-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-3.arc 8 "-m 20")
arcane_add_test_sequential(amr-cartesian3D-patch-cartesian-mesh-only-coarse-zone-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-4.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-patch-cartesian-mesh-only-coarse-zone-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-4.arc 8 "-m 20")
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-1.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-1.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-coarse-zone-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-2.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-coarse-zone-2 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-2.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-coarse-zone-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-3.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-coarse-zone-3 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-3.arc 8 3 5)
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-coarse-zone-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-4.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-patch-cartesian-mesh-only-coarse-zone-4 testAMRCartesianMesh3D-PatchCartesianMeshOnly-CoarseZone-4.arc 8 3 5)


arcane_add_test(amr-cartesian2D-coarse-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-CoarseZone-1.arc "-m 20")
arcane_add_test_checkpoint(amr-checkpoint-cartesian2D-coarse-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh2D-WithInitialCoarse-PatchCartesianMeshOnly-CoarseZone-1.arc 3 5)

arcane_add_test_sequential(amr-cartesian3D-coarse-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-CoarseZone-1.arc "-m 20")
arcane_add_test_parallel(amr-cartesian3D-coarse-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-CoarseZone-1.arc 8 "-m 20")
arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-CoarseZone-1.arc 3 5)
arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-coarse-patch-cartesian-mesh-only-coarse-zone-1 testAMRCartesianMesh3D-WithInitialCoarse-PatchCartesianMeshOnly-CoarseZone-1.arc 8 3 5)


if (BIG_TEST)
  arcane_add_test_sequential(amr-cartesian3D-cell-coarse-zone-2 testAMRCartesianMesh3D-Cell-CoarseZone-2.arc "-m 20")
  arcane_add_test_parallel(amr-cartesian3D-cell-coarse-zone-2 testAMRCartesianMesh3D-Cell-CoarseZone-2.arc 8 "-m 20")

  arcane_add_test_sequential(amr-cartesian3D-cell-coarse-zone-4 testAMRCartesianMesh3D-Cell-CoarseZone-4.arc "-m 20")
  arcane_add_test_parallel(amr-cartesian3D-cell-coarse-zone-4 testAMRCartesianMesh3D-Cell-CoarseZone-4.arc 8 "-m 20")

  arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-cell-coarse-zone-2 testAMRCartesianMesh3D-Cell-CoarseZone-2.arc 3 5)
  arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-cell-coarse-zone-2 testAMRCartesianMesh3D-Cell-CoarseZone-2.arc 8 3 5)

  arcane_add_test_checkpoint_sequential(amr-checkpoint-cartesian3D-cell-coarse-zone-4 testAMRCartesianMesh3D-Cell-CoarseZone-4.arc 3 5)
  arcane_add_test_checkpoint_parallel(amr-checkpoint-cartesian3D-cell-coarse-zone-4 testAMRCartesianMesh3D-Cell-CoarseZone-4.arc 8 3 5)
endif()

arcane_add_test_parallel(amr-cartesian2D-cell-reduce-nb-ghost-layers-1 testAMRCartesianMesh2D-Cell-ReduceNbGhostLayers-1.arc 4 "-m 20")
arcane_add_test_parallel(amr-cartesian2D-cell-reduce-nb-ghost-layers-2 testAMRCartesianMesh2D-Cell-ReduceNbGhostLayers-2.arc 4 "-m 20")
arcane_add_test_parallel(amr-cartesian2D-cell-reduce-nb-ghost-layers-3 testAMRCartesianMesh2D-Cell-ReduceNbGhostLayers-3.arc 4 "-m 20")
arcane_add_test_parallel(amr-cartesian2D-cell-reduce-nb-ghost-layers-4 testAMRCartesianMesh2D-Cell-ReduceNbGhostLayers-4.arc 4 "-m 20")

arcane_add_test_parallel(amr-cartesian3D-cell-reduce-nb-ghost-layers-1 testAMRCartesianMesh3D-Cell-ReduceNbGhostLayers-1.arc 8 "-m 20")


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
arcane_add_test_sequential(cartesian3d_face_edgev3_renumbering testCartesianMesh3D-face-edgev3-renumbering.arc "-m 10")
arcane_add_test_parallel_thread(cartesian3d_face_edgev3_renumbering testCartesianMesh3D-face-edgev3-renumbering.arc 12 "-m 10")

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
  arcane_add_test_parallel(cartesianMeshGenerator testCartesianMeshGenerator.arc 8 "-m 1")

  arcane_add_test_parallel(cartesianMeshGenerator2D testCartesianMeshGenerator2D.arc 1)
  arcane_add_test_parallel(cartesianMeshGenerator2D testCartesianMeshGenerator2D.arc 4)

  arcane_add_test_parallel(cartesianMeshGenerator2D-meshservice testCartesianMeshGenerator2D-meshservice.arc 1)
  arcane_add_test_parallel(cartesianMeshGenerator2D-meshservice testCartesianMeshGenerator2D-meshservice.arc 4)

  arcane_add_test_parallel(cartesianMeshGeneratorModulo testCartesianMeshGeneratorModulo.arc 1)
  arcane_add_test_parallel(cartesianMeshGeneratorModulo testCartesianMeshGeneratorModulo.arc 3)

  arcane_add_test_parallel(cartesianMeshGeneratorOrigin testCartesianMeshGeneratorOrigin.arc 1)
  arcane_add_test_parallel(cartesianMeshGeneratorOrigin testCartesianMeshGeneratorOrigin.arc 4)

  arcane_add_test_parallel(cartesianMeshGeneratorOrigin2D testCartesianMeshGeneratorOrigin2D.arc 1)
  arcane_add_test_parallel(cartesianMeshGeneratorOrigin2D testCartesianMeshGeneratorOrigin2D.arc 4)
endif()

# Pas actif pour l'instant car en échec
arcane_add_test_sequential(cartesianpatch1 unitCartesianPatch1.arc)


#################################
# DynamicCircleAMRModule :
#################################
arcane_add_test(dynamic-circle-amr-1 testDynamicCircleAMR-1.arc "-m 10")

if (BIG_TEST)
  arcane_add_test_sequential(dynamic-circle-amr-2 testDynamicCircleAMR-2.arc "-m 10")
  arcane_add_test_parallel(dynamic-circle-amr-2 testDynamicCircleAMR-2.arc 8 "-m 10")
endif ()
#arcane_add_test_sequential(dynamic-circle-amr-2-fast testDynamicCircleAMR-2.arc
#  "-m 2"
#  "-A,//meshes/mesh/generator/y/n=4"
#)
#arcane_add_test_parallel(dynamic-circle-amr-2-fast testDynamicCircleAMR-2.arc 8
#  "-m 2"
#  "-A,//meshes/mesh/generator/y/n=4"
#)

#################################
# AMRPatchTesterModule :
#################################
arcane_add_test(amr-patch-tester-1 testAMRPatchTester-1.arc "-m 10")

arcane_add_test_sequential(amr-patch-tester-2 testAMRPatchTester-2.arc "-m 10")
arcane_add_test_parallel(amr-patch-tester-2 testAMRPatchTester-2.arc 8 "-m 10")
