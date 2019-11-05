cmake_minimum_required(VERSION 3.11)
project(src)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(Neo.tests
        NeoTestMain.cpp)
#target_link_libraries(src ${GTEST_LIBRAIRIES})
target_link_libraries(Neo.tests GTest::GTest GTest::Main)

gtest_discover_tests(Neo.tests)
