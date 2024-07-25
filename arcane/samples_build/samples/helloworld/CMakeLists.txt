cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
set(EXAMPLE_NAME HelloWorld)

project(HelloWorld LANGUAGES C CXX)
include(../samples.common.cmake)

add_test(NAME helloworld1 COMMAND HelloWorld HelloWorld.arc)
