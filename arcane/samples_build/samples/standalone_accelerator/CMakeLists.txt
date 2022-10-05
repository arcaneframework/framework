project(StandaloneAccelerator LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(StandaloneAccelerator main.cc)
arcane_add_arcane_libraries_to_target(StandaloneAccelerator)

# Indique qu'on souhaite compiler 'main.cc' avec le support des Partie spécifique accélérateur
arcane_accelerator_enable()
arcane_accelerator_add_source_files(main.cc)
arcane_accelerator_add_to_target(StandaloneAccelerator)

# Ajoute test de l'exécutable
add_test(NAME standalone_accelerator_test1 COMMAND StandaloneAccelerator)
if (ARCANE_HAS_CUDA)
  add_test(NAME standalone_accelerator_test1_cuda COMMAND StandaloneAccelerator -A,AcceleratorRuntime=cuda)
endif()
