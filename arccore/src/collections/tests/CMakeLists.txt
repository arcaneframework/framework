arccore_add_component_test_executable(collections
  FILES
  TestArray.cc
  TestArray2.cc
  TestStringVector.cc
  )

target_link_libraries(arccore_collections.tests PUBLIC arccore_collections GTest::GTest GTest::Main)

gtest_discover_tests(arccore_collections.tests DISCOVERY_TIMEOUT 30)
