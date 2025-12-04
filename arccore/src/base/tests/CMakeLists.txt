arccore_add_component_test_executable(base
  FILES
  TestBasicDataType.cc
  TestException.cc
  TestString.cc
  TestStringView.cc
  TestArrayView.cc
  TestReferenceCounter.cc
  TestRef.cc
  TestValueFiller.cc
  )

target_link_libraries(arccore_base.tests PUBLIC arccore_base GTest::GTest GTest::Main)
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  target_compile_options(arccore_base.tests PRIVATE -fext-numeric-literals)
endif()
gtest_discover_tests(arccore_base.tests DISCOVERY_TIMEOUT 30)
