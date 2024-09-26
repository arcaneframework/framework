cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(StandaloneSubDomain LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(StandaloneSubDomain main.cc)
arcane_add_arcane_libraries_to_target(StandaloneSubDomain)

# Ajoute test de l'exécutable
configure_file(plancher.msh ${CMAKE_CURRENT_BINARY_DIR}/plancher.msh COPYONLY)
configure_file(square_v41.msh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(data.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# ----------------------------------------------------------------------------
# Add tests without case file
add_test(NAME standalone_subdomain_test1 COMMAND StandaloneSubDomain)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME standalone_subdomain_test1_p4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ./StandaloneSubDomain
  )
endif()

# ----------------------------------------------------------------------------
# Add tests with case file
add_test(NAME standalone_subdomain_test2 COMMAND StandaloneSubDomain data.arc)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME standalone_subdomain_test2_p4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ./StandaloneSubDomain data.arc
  )
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
