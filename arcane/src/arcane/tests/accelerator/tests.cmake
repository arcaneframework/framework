
set(ARCANE_TEST_CASEPATH ${TEST_PATH}/accelerator)

if (ARCANE_HAS_ACCELERATOR_API)

  #---------------------------------------------------------------------------
  #---------------------------------------------------------------------------

  ## accelerator/ArcaneTestStandaloneAcceleratorMng.cc

  arcane_add_accelerator_test_all_policies(standalone_accelerator_testsum dummy.arc "-A,StandaloneAcceleratorMethod=TestSum")
  arcane_add_accelerator_test_all_policies(standalone_accelerator_testbinop dummy.arc "-A,StandaloneAcceleratorMethod=TestBinOp")
  #arcane_add_accelerator_test_sequential(standalone_accelerator_testsum dummy.arc "-A,StandaloneAcceleratorMethod=TestSum")
  #arcane_add_accelerator_test_sequential(standalone_accelerator_testbinop dummy.arc "-A,StandaloneAcceleratorMethod=TestBinOp")
  #arcane_add_accelerator_test_sequential(standalone_accelerator_testemptykernel dummy.arc "-A,StandaloneAcceleratorMethod=TestEmptyKernel")
  arcane_add_accelerator_test_all_policies(standalone_accelerator_testemptykernel dummy.arc "-A,StandaloneAcceleratorMethod=TestEmptyKernel")
  arcane_add_accelerator_test_sequential(standalone_accelerator_testsum_hostpinned dummy.arc "-A,StandaloneAcceleratorMethod=TestSum" "-We,ARCANE_DEFAULT_DATA_MEMORY_RESOURCE,HostPinned")


  #---------------------------------------------------------------------------
  #---------------------------------------------------------------------------

  ## accelerator/SimpleHydroAcceleratorService.cc

  arcane_add_accelerator_test_sequential(hydro_accelerator5 testHydroAccelerator-5.arc -m 50)
  arcane_add_accelerator_test_sequential(hydro_accelerator5_prefetch testHydroAccelerator-5.arc -m 50 "-We,ARCANE_ACCELERATOR_PREFETCH_COMMAND,1")
  arcane_add_accelerator_test_sequential(hydro_accelerator5_no_memory_pool testHydroAccelerator-5.arc -m 50 "-We,ARCANE_ACCELERATOR_MEMORY_POOL,0")
  arcane_add_accelerator_test_sequential(hydro_accelerator5_full_memory_pool testHydroAccelerator-5.arc -m 50 "-We,ARCANE_ACCELERATOR_MEMORY_POOL,7")
  arcane_add_accelerator_test_parallel(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
  arcane_add_accelerator_test_parallel_thread(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
  arcane_add_accelerator_test_message_passing_hybrid(hydro_accelerator5 CASE_FILE testHydroAccelerator-5.arc NB_SHM 2 NB_MPI 2 ARGS -m 50)
  arcane_add_test(hydro_accelerator5 testHydroAccelerator-5.arc -m 50)
  arcane_add_test_sequential(hydro_accelerator5_prefetch testHydroAccelerator-5.arc -m 50 "-We,ARCANE_ACCELERATOR_PREFETCH_COMMAND,1")
  arcane_add_test_sequential_task(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
  arcane_add_test_parallel_thread(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
  arcane_add_test_message_passing_hybrid(hydro_accelerator5 CASE_FILE testHydroAccelerator-5.arc NB_SHM 2 NB_MPI 2 ARGS -m 50)
  if (ARCANE_ACCELERATOR_RUNTIME_NAME STREQUAL cuda)
    arcane_add_accelerator_test_sequential(hydro_accelerator5_cupti testHydroAccelerator-5.arc -m 20 -We,ARCANE_CUPTI_LEVEL,1)
  endif()

  arcane_add_accelerator_test_sequential(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc -m 50)
  arcane_add_test_sequential(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc -m 50)
  arcane_add_test_sequential_task(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc 4 -m 50)

  #---------------------------------------------------------------------------
  #---------------------------------------------------------------------------

  # accelerator/NumArrayUnitTest.cc

  arcane_add_test_sequential(numarray1 testNumArray-1.arc)
  arcane_add_test_sequential_task(numarray1 testNumArray-1.arc 4)
  arcane_add_accelerator_test_sequential(numarray1 testNumArray-1.arc)

  # accelerator/RunQueueUnitTest.cc

  arcane_add_test_sequential(runqueue1 testRunQueue-1.arc)
  arcane_add_test_sequential_task(runqueue1 testRunQueue-1.arc 4)
  arcane_add_accelerator_test_sequential(runqueue1 testRunQueue-1.arc)

  # accelerator/AcceleratorViewsUnitTest.cc

  arcane_add_test_sequential(acceleratorviews1 testAcceleratorViews-1.arc)
  arcane_add_test_sequential_task(acceleratorviews1 testAcceleratorViews-1.arc 4)
  arcane_add_accelerator_test_sequential(acceleratorviews1 testAcceleratorViews-1.arc)
  arcane_add_accelerator_test_parallel(acceleratorviews1 testAcceleratorViews-1.arc 4)

  # accelerator/AcceleratorReduceUnitTest.cc

  arcane_add_test_sequential(accelerator_reduce1 testAcceleratorReduce-1.arc)
  arcane_add_test_sequential_task(accelerator_reduce1 testAcceleratorReduce-1.arc 4)
  arcane_add_accelerator_test_sequential(accelelerator_reduce1 testAcceleratorReduce-1.arc)

  # accelerator/AcceleratorScanUnitTest.cc

  arcane_add_test_sequential(accelerator_scan1 testAcceleratorScan-1.arc)
  arcane_add_test_sequential_task(accelerator_scan1 testAcceleratorScan-1.arc 4)
  arcane_add_accelerator_test_sequential(accelelerator_scan1 testAcceleratorScan-1.arc)

  # accelerator/AcceleratorFilterUnitTest.cc

  arcane_add_test_sequential(accelerator_filter1 testAcceleratorFilter-1.arc)
  arcane_add_test_sequential_task(accelerator_filter1 testAcceleratorFilter-1.arc 4)
  arcane_add_accelerator_test_sequential(accelelerator_filter1 testAcceleratorFilter-1.arc)

  # accelerator/AcceleratorPartitionerUnitTest.cc

  arcane_add_test_sequential(accelerator_partitioner1 testAcceleratorPartitioner-1.arc)
  arcane_add_test_sequential_task(accelerator_partitioner1 testAcceleratorPartitioner-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_partitioner1 testAcceleratorPartitioner-1.arc)

  # accelerator/AcceleratorSorterUnitTest.cc

  arcane_add_test_sequential(accelerator_sorter1 testAcceleratorSorter-1.arc)
  arcane_add_test_sequential_task(accelerator_sorter1 testAcceleratorSorter-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_sorter1 testAcceleratorSorter-1.arc)

  # accelerator/MeshMaterialAcceleratorUnitTest.cc

  arcane_add_test_sequential(accelerator_material1 testAcceleratorMaterials-1.arc)
  arcane_add_test_sequential_task(accelerator_material1 testAcceleratorMaterials-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_material1 testAcceleratorMaterials-1.arc)

  arcane_add_test_sequential(accelerator_material1_old testAcceleratorMaterials-1.arc "-We,ARCANE_MATERIALMNG_USE_ACCELERATOR_FOR_CONSTITUENTITEMVECTOR,0")
  arcane_add_accelerator_test_sequential(accelerator_material1_old testAcceleratorMaterials-1.arc "-We,ARCANE_MATERIALMNG_USE_ACCELERATOR_FOR_CONSTITUENTITEMVECTOR,0")

  # accelerator/MemoryCopyUnitTest.cc

  arcane_add_test_sequential(memorycopy1 testMemoryCopy-1.arc)
  arcane_add_test_sequential_task(memorycopy1 testMemoryCopy-1.arc 4)
  arcane_add_accelerator_test_sequential(memorycopy1 testMemoryCopy-1.arc)

  # accelerator/MultiMemoryCopyUnitTest.cc

  arcane_add_test_sequential(multi_memorycopy1 testMultiMemoryCopy-1.arc)
  arcane_add_test_sequential_task(multi_memorycopy1 testMultiMemoryCopy-1.arc 4)
  arcane_add_accelerator_test_sequential(multi_memorycopy1 testMultiMemoryCopy-1.arc)

  # accelerator/AcceleratorItemInfoUnitTest.cc

  arcane_add_test_sequential(accelerator_iteminfo1 testAcceleratorItemInfo-1.arc)
  arcane_add_test_sequential_task(accelerator_iteminfo1 testAcceleratorItemInfo-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_iteminfo1 testAcceleratorItemInfo-1.arc)

  # accelerator/AtomicUnitTest.cc

  arcane_add_accelerator_test_all_policies(accelerator_atomic testAcceleratorAtomic-1.arc)

  # accelerator/AcceleratorMathUnitTest.cc

  arcane_add_test_sequential(acceleratormath1 testAcceleratorMath-1.arc)
  arcane_add_accelerator_test_sequential(acceleratormath1 testAcceleratorMath-1.arc)

  # accelerator/AcceleratorLocalMemoryUnitTest.cc

  arcane_add_test_sequential(accelerator_local_memory1 testAcceleratorLocalMemory-1.arc)
  arcane_add_test_sequential_task(accelerator_local_memory1 testAcceleratorLocalMemory-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_local_memory1 testAcceleratorLocalMemory-1.arc)

  #---------------------------------------------------------------------------
  #---------------------------------------------------------------------------

  # accelerator/MiniWeatherOriginalSequential.cc

  arcane_add_test_sequential(miniweather_orig1 testMiniWeatherOriginal.arc)

  # accelerator/MiniWeatherArraySequential.cc

  arcane_add_test_sequential(miniweather_array_seq1 testMiniWeatherArraySequential.arc)

  # accelerator/MiniWeatherArray.cc

  arcane_add_test_sequential(miniweather_array1_right_layout testMiniWeatherArray.arc)
  arcane_add_test_sequential(miniweather_array1_left_layout testMiniWeatherArrayLeftLayout.arc)
  arcane_add_test_sequential_task(miniweather_array1 testMiniWeatherArray.arc 4)
  arcane_add_accelerator_test_sequential(miniweather_array1_right_layout testMiniWeatherArray.arc)
  arcane_add_accelerator_test_sequential(miniweather_array1_left_layout testMiniWeatherArrayLeftLayout.arc)
  arcane_add_accelerator_test_sequential(miniweather_array1_device testMiniWeatherArrayDevice.arc)

endif()
