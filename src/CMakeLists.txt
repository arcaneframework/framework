cmake_minimum_required(VERSION 3.11)
project(src)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(src
        NeoTestMain.cpp)
