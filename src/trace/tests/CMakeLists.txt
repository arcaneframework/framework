add_executable(trace.tests TestTraceMng.cc)

target_link_libraries(trace.tests PUBLIC arccore_trace GTest::GTest GTest::Main)

gtest_discover_tests(trace.tests)
