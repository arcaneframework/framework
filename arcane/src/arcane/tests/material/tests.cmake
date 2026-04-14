
set(ARCANE_TEST_CASEPATH ${TEST_PATH}/material)

arcane_add_test_sequential(material1_simd1 testMaterialSimd-1.arc)

arcane_add_test(material_sync1 testMaterial-sync-1.arc)
arcane_add_test_parallel_thread(material_sync1 testMaterial-sync-1.arc 4)
arcane_add_test_parallel(material_sync2 testMaterial-sync-2.arc 4)
arcane_add_test_parallel(material_sync2_v3 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,3)
arcane_add_test(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
arcane_add_test_parallel_thread(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
arcane_add_test_message_passing_hybrid(material_sync2_v7 CASE_FILE testMaterial-sync-2.arc NB_MPI 3 NB_SHM 4 ARGS -We,ARCANE_MATSYNCHRONIZE_VERSION,7)

if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_accelerator_test_parallel(material_sync2_v6 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,6)
  arcane_add_accelerator_test_parallel(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  arcane_add_accelerator_test_parallel(material_sync2_v8 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,8)
  arcane_add_accelerator_test_parallel(material_sync2_vacc testMaterial-sync-2.arc 4 -We,ARCANE_ACC_MAT_SYNCHRONIZER,1)
  arcane_add_accelerator_test_parallel_thread(material_sync2_v7 testMaterial-sync-2.arc 4 -We,ARCANE_MATSYNCHRONIZE_VERSION,7)
  arcane_add_accelerator_test_parallel_thread(material_sync2_vacc testMaterial-sync-2.arc 4 -We,ARCANE_ACC_MAT_SYNCHRONIZER,1)
endif ()

arcane_add_test(material_sync3 testMaterial-sync-3.arc)
arcane_add_test_parallel_thread(material_sync3 testMaterial-sync-3.arc 4)

#################################
# Material Heat
#################################

foreach(test_index 1 2 3 4)
  foreach(opt_level 0 3 7 11 15)
    set(TEST_MODIFICATION_FLAG ${opt_level})

    # Test sans équilibrage
    set(TEST_ACTIVE_LOAD_BALANCE "false")
    set(_TEST_FILENAME "${ARCANE_TEST_PATH}/testMaterialHeat-${test_index}-opt${opt_level}.arc")
    configure_file("${ARCANE_TEST_CASEPATH}/testMaterialHeat-${test_index}.arc.in" "${_TEST_FILENAME}")

    # Test avec équilibrage
    set(TEST_ACTIVE_LOAD_BALANCE "true")
    set(_TEST_FILENAME_LB "${ARCANE_TEST_PATH}/testMaterialHeat-${test_index}-opt${opt_level}-lb.arc")
    configure_file("${ARCANE_TEST_CASEPATH}/testMaterialHeat-${test_index}.arc.in" "${_TEST_FILENAME_LB}")
    if (ARCANE_HAS_ACCELERATOR_API)
      arcane_add_test_sequential(material_heat${test_index}_opt${opt_level} ${_TEST_FILENAME})
      arcane_add_test_parallel_thread(material_heat${test_index}_opt${opt_level} ${_TEST_FILENAME} 4)
      arcane_add_test(material_heat${test_index}_lb_opt${opt_level} ${_TEST_FILENAME_LB})
      arcane_add_test_parallel_thread(material_heat${test_index}_lb_opt${opt_level} ${_TEST_FILENAME_LB} 4)
    endif()
  endforeach()
endforeach()
if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test_sequential(material_heat2_opt15_2small testMaterialHeat-2-small-opt15.arc "-We,ARCANE_DEBUG_MATERIAL_MODIFIER,2" "-We,ARCANE_PRINT_USELESS_TRANSFORMATION,1")
  arcane_add_test_sequential(material_heat2_opt15_2small_force_transform testMaterialHeat-2-small-opt15.arc "-We,ARCANE_MATERIAL_FORCE_TRANSFORM,1")
  arcane_add_test_sequential(material_heat2_accelerator
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc"
    "-We,ARCANE_MATERIALMNG_ADDITIONAL_CAPACITY_RATIO,0.5"
    "-m 20"
  )
  arcane_add_accelerator_test_sequential(material_heat2_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20")
  arcane_add_accelerator_test_sequential(material_heat2_accelerator_noqueue
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_MATERIALMNG_USE_QUEUE,0")
  arcane_add_test_sequential_host_and_accelerator(material_heat2_accelerator_generic_copy
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_USE_GENERIC_COPY_BETWEEN_PURE_AND_PARTIAL,1")
  arcane_add_test_sequential_host_and_accelerator(material_heat2_accelerator_generic_copy_one_queue
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_USE_GENERIC_COPY_BETWEEN_PURE_AND_PARTIAL,2")
  arcane_add_test_sequential_host_and_accelerator(material_heat2_accelerator_force_multiple_resize
    "${ARCANE_TEST_PATH}/testMaterialHeat-2-opt15.arc" "-m 20" "-We,ARCANE_FORCE_MULTIPLE_COMMAND_FOR_MATERIAL_RESIZE,1")

  arcane_add_test_sequential(material_heat4_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20")
  arcane_add_accelerator_test_sequential(material_heat4_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20")
  arcane_add_test_sequential_host_and_accelerator(material_heat4_init1_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20" "-We,ARCANE_MATERIAL_NEW_ITEM_INIT,1")
  arcane_add_test_sequential_host_and_accelerator(material_heat4_init2_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20"
    "-We,ARCANE_MATERIAL_NEW_ITEM_INIT,2"  "-We,ARCANE_USE_GENERIC_COPY_BETWEEN_PURE_AND_PARTIAL,2")
  arcane_add_test_sequential_host_and_accelerator(material_heat4_init3_accelerator "${ARCANE_TEST_PATH}/testMaterialHeat-4-opt15.arc" "-m 20"
    "-We,ARCANE_MATERIAL_NEW_ITEM_INIT,3" "-We,ARCANE_ALLENVCELL_FOR_RUNCOMMAND,1")
  if(HDF5_FOUND)
    arcane_add_test(material_heat2_vtkhdfv2 testMaterialHeat-2-vtkhdfv2.arc)
  endif()

endif()
