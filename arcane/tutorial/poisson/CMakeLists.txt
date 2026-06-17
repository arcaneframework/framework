cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(Poisson LANGUAGES C CXX)

set(EXAMPLE_NAME Poisson)

find_package(Arcane REQUIRED)

add_executable(${EXAMPLE_NAME}
  main.cc
  PoissonModule.cc
  PoissonModule.h
  ${CMAKE_CURRENT_BINARY_DIR}/Poisson_axl.h
)

arcane_generate_axl(Poisson)

arcane_add_arcane_libraries_to_target(${EXAMPLE_NAME})

target_include_directories(${EXAMPLE_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file(Poisson.config ${CMAKE_CURRENT_BINARY_DIR}/Poisson.config COPYONLY)
configure_file(Poisson.arc    ${CMAKE_CURRENT_BINARY_DIR}/Poisson.arc    COPYONLY)

enable_testing()
add_test(NAME poisson COMMAND ${EXAMPLE_NAME} Poisson.arc)

