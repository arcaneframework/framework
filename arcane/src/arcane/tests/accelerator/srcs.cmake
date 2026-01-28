#list(APPEND ARCANE_SOURCES
#)
#list(APPEND AXL_FILES
#)

if (ARCANE_HAS_ACCELERATOR_API)
  list(APPEND ARCANE_SOURCES
    accelerator/MiniWeatherArraySequential.cc
    accelerator/MiniWeatherOriginalSequential.cc
  )
  list(APPEND AXL_FILES
    accelerator/AcceleratorFilterUnitTest
    accelerator/AcceleratorReduceUnitTest
    accelerator/AcceleratorScanUnitTest
    accelerator/SimpleHydroAccelerator
  )
  list(APPEND ARCANE_ACCELERATOR_SOURCES
    accelerator/AtomicUnitTest.cc
    accelerator/AcceleratorFilterUnitTest.cc
    accelerator/AcceleratorItemInfoUnitTest.cc
    accelerator/AcceleratorLocalMemoryUnitTest.cc
    accelerator/AcceleratorMathUnitTest.cc
    accelerator/AcceleratorPartitionerUnitTest.cc
    accelerator/AcceleratorReduceUnitTest.cc
    accelerator/AcceleratorScanUnitTest.cc
    accelerator/AcceleratorSorterUnitTest.cc
    accelerator/AcceleratorViewsUnitTest.cc
    accelerator/ArcaneTestStandaloneAcceleratorMng.cc
    accelerator/MemoryCopyUnitTest.cc
    accelerator/MeshMaterialAcceleratorUnitTest.cc
    accelerator/MiniWeatherArray.cc
    accelerator/MultiMemoryCopyUnitTest.cc
    accelerator/NumArrayUnitTest.cc
    accelerator/RunQueueUnitTest.cc
    accelerator/SimpleHydroAcceleratorService.cc
  )
endif()
