cmake_minimum_required(VERSION 3.11)
project(Neo)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(sgraph)
add_subdirectory(src)

enable_testing()
