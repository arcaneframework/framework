arccore_add_component_test_executable(concurrency
  FILES
  TestConcurrency.cc
  )

target_link_libraries(arccore_concurrency.tests PUBLIC arccore_concurrency GTest::GTest GTest::Main)

gtest_discover_tests(arccore_concurrency.tests DISCOVERY_TIMEOUT 30)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
