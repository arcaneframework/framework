cmake_minimum_required(VERSION 3.21)
project(ArcCher LANGUAGES CXX C)

if (NOT Arccore_FOUND)
  find_package(Arccore REQUIRED)
endif ()

set(REPROC++ ON)
add_subdirectory(src/reproc)

add_executable(ArcCher src/launcher/main.cc)
target_include_directories(ArcCher PUBLIC . ${CMAKE_CURRENT_BINARY_DIR} src/reproc/reproc++/include)

target_link_libraries(ArcCher PUBLIC
  Arccore::arccore_trace
  Arccore::arccore_common
  Arccore::arccore_base
)

target_link_libraries(ArcCher PUBLIC reproc++)

# TODO Pour tester, à supprimer
include(../arct/arcanet_edit.cmake)
include(../arct/arcanet_to_ctest.cmake)

#message(FATAL_ERROR "Stop")
