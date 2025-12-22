set(SOURCE_FILES
  TestInit.cc
)

set(ACCELERATOR_FILES
  TestReduce1.cc
)

arccore_add_component_test_executable(accelerator
  FILES ${SOURCE_FILES} ${ACCELERATOR_FILES}
)
arccore_accelerator_add_source_files(${ACCELERATOR_FILES})

target_link_libraries(arccore_accelerator.tests PUBLIC GTest::GTest GTest::Main)

target_link_libraries(arccore_accelerator.tests PRIVATE
  arccore_accelerator
)

gtest_discover_tests(arccore_accelerator.tests DISCOVERY_TIMEOUT 30)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
