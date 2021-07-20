arcane_add_component_test_executable(utils
  FILES
  TestNumArray.cc
  )

target_link_libraries(arcane_utils.tests PUBLIC arcane_utils GTest::GTest GTest::Main)

gtest_discover_tests(arcane_utils.tests)
