cmake_minimum_required(VERSION 3.13)
project(Neo)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(sgraph)
add_subdirectory(src)

enable_testing()
