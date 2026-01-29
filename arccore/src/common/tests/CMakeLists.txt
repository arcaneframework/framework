arccore_add_component_test_executable(common
  FILES
  TestProcess.cc
  TestStringVector.cc
)

target_link_libraries(arccore_common.tests PUBLIC arccore_common GTest::GTest GTest::Main)

gtest_discover_tests(arccore_common.tests DISCOVERY_TIMEOUT 30)
