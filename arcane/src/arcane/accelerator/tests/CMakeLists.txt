set(SOURCE_FILES
  TestInit.cc
  TestCommon.cc
  TestPartition.cc
)

set(ACCELERATOR_FILES
  TestReduce.cc
  TestPartition_Kernel.cc
  TestVirtualMethod.cc
  TestVirtualMethod_Kernel.cc
)

arcane_add_component_test_executable(accelerator
  FILES ${SOURCE_FILES} ${ACCELERATOR_FILES}
)
arcane_accelerator_add_source_files(${ACCELERATOR_FILES})

target_link_libraries(arcane_accelerator.tests PUBLIC arcane_accelerator GTest::GTest GTest::Main)

# Cela est nécessaire pour utiliser un kernel défini dans une autre
# unité de compilation
if (ARCANE_ACCELERATOR_MODE STREQUAL "CUDA")
  set_property(TARGET arcane_accelerator.tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(arcane_accelerator.tests PRIVATE
  Arccore::arccore_accelerator
  "$<TARGET_NAME_IF_EXISTS:arccore_accelerator_cuda_runtime>"
  "$<TARGET_NAME_IF_EXISTS:arccore_accelerator_hip_runtime>"
  "$<TARGET_NAME_IF_EXISTS:arccore_accelerator_sycl_runtime>"
)

gtest_discover_tests(arcane_accelerator.tests DISCOVERY_TIMEOUT 30)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
