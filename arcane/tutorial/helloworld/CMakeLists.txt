cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(HelloWorld LANGUAGES C CXX)

set(EXAMPLE_NAME HelloWorld)

find_package(Arcane REQUIRED)

add_executable(${EXAMPLE_NAME}
  main.cc
  HelloWorldModule.cc
  ${CMAKE_CURRENT_BINARY_DIR}/HelloWorld_axl.h
)

arcane_generate_axl(HelloWorld)

arcane_add_arcane_libraries_to_target(${EXAMPLE_NAME})

target_include_directories(${EXAMPLE_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file(HelloWorld.config ${CMAKE_CURRENT_BINARY_DIR}/HelloWorld.config COPYONLY)
configure_file(HelloWorld.arc    ${CMAKE_CURRENT_BINARY_DIR}/HelloWorld.arc    COPYONLY)

enable_testing()
add_test(NAME helloworld1 COMMAND ${EXAMPLE_NAME} HelloWorld.arc)

