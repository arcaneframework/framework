arccore_add_component_test_executable(message_passing
  FILES
  TestStat.cc
  )

target_link_libraries(arccore_message_passing.tests PUBLIC arccore_message_passing GTest::GTest GTest::Main)

gtest_discover_tests(arccore_message_passing.tests DISCOVERY_TIMEOUT 30)
