add_executable(arccore_base.tests
  TestString.cc
  TestStringView.cc
  TestArrayView.cc
  TestReferenceCounter
  )

target_link_libraries(arccore_base.tests PUBLIC arccore_base GTest::GTest GTest::Main)

gtest_discover_tests(arccore_base.tests)
