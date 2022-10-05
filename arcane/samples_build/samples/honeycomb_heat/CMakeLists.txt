cmake_minimum_required(VERSION 3.21)
project(HoneyCombHeat LANGUAGES C CXX)

find_package(Arcane REQUIRED)

add_executable(HoneyCombHeat HoneyCombHeatModule.cc main.cc HoneyCombHeat_axl.h)

arcane_generate_axl(HoneyCombHeat)
arcane_add_arcane_libraries_to_target(HoneyCombHeat)
target_include_directories(HoneyCombHeat PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(HoneyCombHeat.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

enable_testing()
add_test(NAME honeycomb2d_heat1 COMMAND HoneyCombHeat ${CMAKE_CURRENT_LIST_DIR}/HoneyCombHeat2D.arc)
add_test(NAME honeycomb3d_heat1 COMMAND HoneyCombHeat ${CMAKE_CURRENT_LIST_DIR}/HoneyCombHeat3D.arc)
