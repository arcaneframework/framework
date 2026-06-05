set(SOURCE_FILES
  TestInit.cc
  TestCommon.cc
  TestPartition.cc
)

set(ACCELERATOR_FILES
  TestReduce.cc
  TestPartition_Kernel.cc
)

# Check whether tests on virtual methods are performed
set(DO_TEST_VIRTUAL FALSE)
# These tests are only valid for CUDA and HIP
if ((ARCANE_ACCELERATOR_MODE STREQUAL "CUDA") OR (ARCANE_ACCELERATOR_MODE STREQUAL "ROCM"))
  set(DO_TEST_VIRTUAL TRUE)
endif()
# We do not test virtual methods with CUDA/Clang because
# this does not seem to work during linking.
# It seems that CMake does not correctly call Clang to
# perform the linking with separated symbols.
# of the linking), of Clang or of CMake.
if (CMAKE_CUDA_COMPILER_ID STREQUAL Clang)
  set(DO_TEST_VIRTUAL FALSE)
endif()
message(STATUS "Check adding accelerator virtual method tests = ${DO_TEST_VIRTUAL}")
if(DO_TEST_VIRTUAL)
  list(APPEND ACCELERATOR_FILES
    TestVirtualMethod.cc
    TestVirtualMethod_Kernel.cc
  )
endif()

arcane_add_component_test_executable(accelerator
  FILES ${SOURCE_FILES} ${ACCELERATOR_FILES}
)
arcane_accelerator_add_source_files(${ACCELERATOR_FILES})

target_link_libraries(arcane_accelerator.tests PUBLIC arcane_accelerator GTest::GTest GTest::Main)

# This is necessary to use a kernel defined in another
# compilation unit
if (ARCANE_ACCELERATOR_MODE STREQUAL "CUDA")
  set_property(TARGET arcane_accelerator.tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(arcane_accelerator.tests PRIVATE
  Arccore::arccore_accelerator
  "$<TARGET_NAME_IF_EXISTS:arccore_accelerator_cuda_runtime>"
  "$<TARGET_NAME_IF_EXISTS:arccore_accelerator_hip_runtime>"
  "$<TARGET_NAME_IF_EXISTS:arccore_accelerator_sycl_runtime>"
)

if (ARCANE_EXECUTE_ACCELERATOR_GOOGLETEST)
  gtest_discover_tests(arcane_accelerator.tests DISCOVERY_TIMEOUT 30)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
