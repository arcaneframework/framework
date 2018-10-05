add_executable(base.tests TestString.cc TestArrayView.cc)

target_link_libraries(base.tests PUBLIC arccore_base GTest::GTest GTest::Main)

gtest_discover_tests(base.tests)
