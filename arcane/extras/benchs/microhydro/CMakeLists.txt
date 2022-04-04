cmake_minimum_required(VERSION 3.18)
project(MicroHydro LANGUAGES CXX)

find_package(Arcane REQUIRED)

add_executable(MicroHydro MicroHydroModule.cc main.cc MicroHydro_axl.h AdditionalVariablesModule.cc AdditionalVariables_axl.h)

arcane_generate_axl(MicroHydro)
arcane_generate_axl(AdditionalVariables)
arcane_add_arcane_libraries_to_target(MicroHydro)
target_include_directories(MicroHydro PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(MicroHydro.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(MicroHydro.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(MicroHydro.load-balance.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Partie spécifique accélérateur
arcane_accelerator_enable()
arcane_accelerator_add_source_files(MicroHydroModule.cc)
arcane_accelerator_add_to_target(MicroHydro)

enable_testing()
# Ajout des tests
add_test(NAME microhydro COMMAND ./MicroHydro -A,MaxIteration=50 MicroHydro.arc)
if (ARCANE_HAS_CUDA)
  add_test(NAME microhydro_cuda COMMAND ./MicroHydro -A,MaxIteration=50 -A,AcceleratorRuntime=cuda MicroHydro.arc)
endif()
