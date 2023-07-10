add_executable(SGraph.tests GraphTest.cpp)

enable_testing()

find_package(GTest CONFIG)

#target_include_directories(SGraph.tests PRIVATE ${GTEST_INCLUDE_DIRS})
#target_link_libraries(SGraph.tests SGraph GTest::GTest GTest::Main)
target_link_libraries(SGraph.tests SGraph GTest::gtest GTest::gtest_main)

include(GoogleTest)

gtest_discover_tests(SGraph.tests)

