add_executable(collections.tests TestArray.cc)

target_link_libraries(collections.tests PUBLIC arccore_collections GTest::GTest GTest::Main)

gtest_discover_tests(collections.tests)
