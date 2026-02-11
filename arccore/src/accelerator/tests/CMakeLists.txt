set(SOURCE_FILES
  TestCommon.h
  TestCooperativeLaunch1.cc
  TestInit.cc
  TestReduce1.cc
  TestLoop1.cc
  TestMemoryBandwidth.cc
)

set(ACCELERATOR_FILES
  TestReduce_Accelerator.cc
  TestReduceGridStride_Accelerator.cc
  TestLoop_Accelerator.cc
  TestCooperativeLaunch_Accelerator.cc
)

arccore_add_component_test_executable(accelerator
  FILES ${SOURCE_FILES} ${ACCELERATOR_FILES}
)
arccore_accelerator_add_source_files(${ACCELERATOR_FILES})

target_link_libraries(arccore_accelerator.tests PUBLIC GTest::GTest GTest::Main)

target_link_libraries(arccore_accelerator.tests PRIVATE
  arccore_accelerator
)
if (ARCCORE_ENABLE_TBB)
  target_compile_definitions(arccore_accelerator.tests PRIVATE ARCCORE_HAS_ACCELERATOR_THREAD)
endif()

gtest_discover_tests(arccore_accelerator.tests DISCOVERY_TIMEOUT 30)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
