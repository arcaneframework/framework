cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(StandaloneSubDomainAccelerator LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(StandaloneSubDomainAccelerator main.cc)
arcane_add_arcane_libraries_to_target(StandaloneSubDomainAccelerator)

# We want to compile 'main.cc' with support for accelerators
arcane_accelerator_enable()
arcane_accelerator_add_source_files(main.cc)
arcane_accelerator_add_to_target(StandaloneSubDomainAccelerator)

# Ajoute test de l'exécutable
configure_file(plancher.msh ${CMAKE_CURRENT_BINARY_DIR}/plancher.msh COPYONLY)
configure_file(data.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# ----------------------------------------------------------------------------
enable_testing()
add_test(NAME standalone_subdomain_test1 COMMAND StandaloneSubDomainAccelerator data.arc)
if (ARCANE_HAS_ACCELERATOR)
  add_test(NAME standalone_accelerator_test1_accelerator COMMAND StandaloneSubDomainAccelerator -A,UseAccelerator=1 data.arc)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
