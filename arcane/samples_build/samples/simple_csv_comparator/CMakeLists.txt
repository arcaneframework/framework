set(PROJECT_NAME stc)
cmake_minimum_required(VERSION 3.16)
project(${PROJECT_NAME} LANGUAGES CXX C)

set(BUILD_SHARED_LIBS TRUE)

find_package(Arcane REQUIRED)

add_executable(${PROJECT_NAME}
  main.cc)

add_library(SimpleTableComparatorExample1 example1/SimpleTableComparatorExample1Module.cc)
target_include_directories(SimpleTableComparatorExample1 PUBLIC .)
target_link_libraries(SimpleTableComparatorExample1 PUBLIC arcane_core)

add_library(SimpleTableComparatorExample2 example2/SimpleTableComparatorExample2Module.cc)
target_include_directories(SimpleTableComparatorExample2 PUBLIC .)
target_link_libraries(SimpleTableComparatorExample2 PUBLIC arcane_core)

add_library(SimpleTableComparatorExample3 example3/SimpleTableComparatorExample3Module.cc)
target_include_directories(SimpleTableComparatorExample3 PUBLIC .)
target_link_libraries(SimpleTableComparatorExample3 PUBLIC arcane_core)

target_link_libraries(${PROJECT_NAME} PRIVATE SimpleTableComparatorExample1 SimpleTableComparatorExample2 SimpleTableComparatorExample3)

arcane_target_add_axl(SimpleTableComparatorExample1 example1/SimpleTableComparatorExample1)
arcane_target_add_axl(SimpleTableComparatorExample2 example2/SimpleTableComparatorExample2)
arcane_target_add_axl(SimpleTableComparatorExample3 example3/SimpleTableComparatorExample3)

arcane_add_arcane_libraries_to_target(${PROJECT_NAME})
target_compile_options(${PROJECT_NAME} PUBLIC -Wpedantic)
target_include_directories(${PROJECT_NAME} PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(${PROJECT_NAME}.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
enable_testing()

configure_file(data/SimpleTableComparatorExample1.arc ${CMAKE_CURRENT_BINARY_DIR} @ONLY)
configure_file(data/SimpleTableComparatorExample2.arc ${CMAKE_CURRENT_BINARY_DIR} @ONLY)
configure_file(data/SimpleTableComparatorExample3.arc ${CMAKE_CURRENT_BINARY_DIR} @ONLY)
add_test(NAME simple_csv_comparator_example_1 COMMAND stc SimpleTableComparatorExample1.arc) # Ecriture
add_test(NAME simple_csv_comparator_example_1 COMMAND stc SimpleTableComparatorExample1.arc) # Lecture
add_test(NAME simple_csv_comparator_example_2 COMMAND stc SimpleTableComparatorExample2.arc) # Ecriture
add_test(NAME simple_csv_comparator_example_2 COMMAND stc SimpleTableComparatorExample2.arc) # Lecture
add_test(NAME simple_csv_comparator_example_3 COMMAND stc SimpleTableComparatorExample3.arc) # Ecriture
add_test(NAME simple_csv_comparator_example_3 COMMAND stc SimpleTableComparatorExample3.arc) # Lecture
