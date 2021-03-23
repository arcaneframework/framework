set(EXAMPLE_NAME MicroHydro)
project(MicroHydro LANGUAGES C CXX)
include(../samples.common.cmake)

add_library(PerfectGas eos/perfectgas/PerfectGasEOSService.cc)
target_include_directories(PerfectGas PUBLIC .)
target_link_libraries(PerfectGas PUBLIC arcane_core)

add_library(StiffenedGas eos/stiffenedgas/StiffenedGasEOSService.cc)
target_include_directories(StiffenedGas PUBLIC .)
target_link_libraries(StiffenedGas PUBLIC arcane_core)

target_link_libraries(${EXAMPLE_NAME} PRIVATE PerfectGas StiffenedGas)

arcane_target_add_axl(PerfectGas eos/perfectgas/PerfectGasEOS)
arcane_target_add_axl(StiffenedGas eos/stiffenedgas/StiffenedGasEOS)

configure_file(sod.vtk ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(sod.vtkfaces.vtk ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Commande pour lancer via 'arcane_run'
set(RUN_COMMAND ${ARCANE_PREFIX_DIR}/bin/arcane_run -E ${CMAKE_CURRENT_BINARY_DIR}/MicroHydro)

add_test(NAME microhydro1 COMMAND MicroHydro MicroHydro.arc)

add_test(NAME microhydro1_run COMMAND ${RUN_COMMAND} MicroHydro.arc)

# TODO: ajouter test avec partitionnement de maillage en mode mémoire
# partagée mais pour l'instant cela n'est pas supporté dans Arcane.
add_test(NAME microhydro1_4s COMMAND MicroHydro -A,S=4 MicroHydro.nomesh.arc)

configure_file(MicroHydro.nomesh.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)


if(MPIEXEC_EXECUTABLE)
  add_test(NAME microhydro1_p4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ./MicroHydro MicroHydro.arc
    )
  # Teste le lancement avec 'arcane_run'
  add_test(NAME microhydro1_p4_run
    COMMAND ${RUN_COMMAND} -n 4 MicroHydro.arc
    )
endif()
