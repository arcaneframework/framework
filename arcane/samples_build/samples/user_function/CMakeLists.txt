cmake_minimum_required(VERSION 3.21)
project(UserFunction LANGUAGES C CXX)

find_package(Arcane REQUIRED)

add_executable(UserFunction UserFunctionModule.cc main.cc UserFunction_axl.h)

arcane_generate_axl(UserFunction)
arcane_add_arcane_libraries_to_target(UserFunction)
target_include_directories(UserFunction PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(UserFunction.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Command to compile C# file for functions
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ExternalFunctions.dll"
  COMMAND ${ARCANE_PREFIX_DIR}/bin/arcane_dotnet_compile "${CMAKE_CURRENT_LIST_DIR}/ExternalFunctions.cs"
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ExternalFunctions.cs"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
add_custom_target(external_functions DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ExternalFunctions.dll")
add_dependencies(UserFunction external_functions)

# Add a test to run the sample
add_test(NAME userfunction1 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/UserFunction -A,UsingDotNet=1 ${CMAKE_CURRENT_LIST_DIR}/UserFunction.arc)
