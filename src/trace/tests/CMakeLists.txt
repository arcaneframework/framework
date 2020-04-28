add_executable(arccore_trace.tests TestTraceMng.cc)

target_link_libraries(arccore_trace.tests PUBLIC arccore_trace GTest::GTest GTest::Main)

gtest_discover_tests(arccore_trace.tests)
