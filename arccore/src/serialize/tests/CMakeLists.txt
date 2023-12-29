arccore_add_component_test_executable(serialize
  FILES
  TestSerialize.cc
  )

target_link_libraries(arccore_serialize.tests PUBLIC arccore_serialize GTest::GTest GTest::Main)

gtest_discover_tests(arccore_serialize.tests DISCOVERY_TIMEOUT 30)
