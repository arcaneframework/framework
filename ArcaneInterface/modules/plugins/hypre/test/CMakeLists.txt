createExecutable(hypre_arcane_test.exe)

addSources(hypre_arcane_test.exe
        main.cc
        AlienTestModule.cc
        ArcaneCodeService.cc
        )

arcane_target_add_axl(hypre_arcane_test.exe AlienTest)

linkLibraries(hypre_arcane_test.exe
        Alien::alien_core
        Alien::alien_semantic_move
        alien_plugins_solvers
        alien_arcane_tools
        arcane_core
        arcane_std
        arcane_mpi
        arcane_impl
        arcane_mesh
        arcane_utils
        arcane_thread
        )

commit(hypre_arcane_test.exe)

configure_file(AlienTest.config ${CMAKE_CURRENT_BINARY_DIR}/AlienTest.config COPYONLY)
configure_file(AlienTest.config ${CMAKE_CURRENT_BINARY_DIR}/AlienTest.config COPYONLY)
configure_file(AlienTest-Hypre.arc ${CMAKE_CURRENT_BINARY_DIR}/AlienTest-Hypre.arc COPYONLY)
configure_file(tube5x5x100.vtk ${CMAKE_CURRENT_BINARY_DIR}/tube5x5x100.vtk COPYONLY)

set(ENV (ARCANE_PARALLEL_SERVICE) "Mpi")

alien_test(
        BENCH hypre.test.arcane
        NAME scalar
        PROCS 4
        COMMAND hypre_arcane_test.exe
        OPTIONS AlienTest-Hypre.arc
        WORKING_DIRECTORY modules/AlienOpensourcePlugins/hypre/test
)
