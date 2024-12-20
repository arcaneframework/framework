set(SOURCE_FILES
  TestInit.cc
  TestCommon.cc
  TestPartition_Kernel.cc
  TestPartition.cc
  TestReduce.cc
)

arcane_add_component_test_executable(accelerator
  FILES ${SOURCE_FILES}
  )
arcane_accelerator_add_source_files(TestReduce.cc TestPartition_Kernel.cc)

target_link_libraries(arcane_accelerator.tests PUBLIC arcane_accelerator GTest::GTest GTest::Main)

target_link_libraries(arcane_accelerator.tests PRIVATE
  "$<TARGET_NAME_IF_EXISTS:arcane_accelerator_cuda_runtime>"
  "$<TARGET_NAME_IF_EXISTS:arcane_accelerator_hip_runtime>"
  "$<TARGET_NAME_IF_EXISTS:arcane_accelerator_sycl_runtime>"
)

gtest_discover_tests(arcane_accelerator.tests DISCOVERY_TIMEOUT 30)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
