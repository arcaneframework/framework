cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(SimpleHeat LANGUAGES C CXX)

find_package(Arcane REQUIRED)

add_executable(SimpleHeat SimpleHeatModule.cc main.cc SimpleHeat_axl.h)

arcane_generate_axl(SimpleHeat)
arcane_add_arcane_libraries_to_target(SimpleHeat)
target_include_directories(SimpleHeat PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(SimpleHeat.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

enable_testing()
add_test(NAME simple_heat_3d COMMAND SimpleHeat ${CMAKE_CURRENT_LIST_DIR}/SimpleHeat3D.arc)
