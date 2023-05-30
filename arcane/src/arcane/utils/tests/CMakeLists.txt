arcane_add_component_test_executable(utils
  FILES
  TestNumArray.cc
  TestDependencyInjection.cc
  TestRealN.cc
  TestNumVector.cc
  TestValueConvert.cc
  TestCollections.cc
  TestHash.cc
  TestHashTable.cc
  TestMemory.cc
  )

target_link_libraries(arcane_utils.tests PUBLIC arcane_utils GTest::GTest GTest::Main)

gtest_discover_tests(arcane_utils.tests DISCOVERY_TIMEOUT 30)
