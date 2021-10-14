arcane_add_component_test_executable(utils
  FILES
  TestCppNameDemangler.cc
  TestNumArray.cc
  TestDependencyInjection.cc
 )

target_link_libraries(arcane_utils.tests PUBLIC arcane_utils GTest::GTest GTest::Main)

gtest_discover_tests(arcane_utils.tests)
