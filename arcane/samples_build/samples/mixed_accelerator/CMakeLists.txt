cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(MixedAccelerator LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(MixedAccelerator main.cc)
arcane_add_arcane_libraries_to_target(MixedAccelerator)

arcane_accelerator_enable()
arcane_accelerator_add_source_files(main.cc)
arcane_accelerator_add_to_target(MixedAccelerator)

# Ajoute test de l'exécutable
add_test(NAME standalone_accelerator_test1 COMMAND MixedAccelerator)
if (ARCANE_HAS_CUDA)
  add_test(NAME standalone_accelerator_test1_cuda COMMAND MixedAccelerator -A,AcceleratorRuntime=cuda)
endif()
