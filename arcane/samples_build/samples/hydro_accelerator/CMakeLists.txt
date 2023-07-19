cmake_minimum_required(VERSION 3.21)
project(HydroAccelerator LANGUAGES C CXX)

find_package(Arcane REQUIRED)

add_executable(HydroAccelerator SimpleHydroAcceleratorService.cc ModuleSimpleHydroGeneric.cc main.cc SimpleHydro_axl.h)

arcane_generate_axl(SimpleHydro)
arcane_add_arcane_libraries_to_target(HydroAccelerator)
target_include_directories(HydroAccelerator PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(HydroAccelerator.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(HydroAccelerator.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Partie spécifique accélérateur
arcane_accelerator_enable()
arcane_accelerator_add_source_files(SimpleHydroAcceleratorService.cc)
arcane_accelerator_add_to_target(HydroAccelerator)

# Ajout des tests
add_test(NAME hydro_accelerator COMMAND ./HydroAccelerator -A,MaxIteration=50 HydroAccelerator.arc)
if (ARCANE_HAS_CUDA)
  add_test(NAME hydro_accelerator_cuda COMMAND ./HydroAccelerator -A,MaxIteration=50 -A,AcceleratorRuntime=cuda HydroAccelerator.arc)
endif()
