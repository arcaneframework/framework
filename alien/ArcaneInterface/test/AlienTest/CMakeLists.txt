createExecutable(alien_arcane_test.exe)

addSources(alien_arcane_test.exe
        main.cc
        AlienTestModule.cc
        MemoryAllocationTracker.cc
        ArcaneCodeService.cc
        )

#arcane_target_add_axl(alien_arcane_test.exe AlienTest)
generateAxl(alien_arcane_test.exe
            AlienTest.axl
            NO_COPY
           )



if (TARGET intel)
    linkLibraries(alien_arcane_test.exe intel)
endif ()

linkLibraries(alien_arcane_test.exe
        Alien::alien_core
        Alien::alien_semantic_move
        alien_external_packages
        alien_ifpen_solvers
        alien_arcane_tools
        arcane_core
        arcane_std
        arcane_mpi
        arcane_impl
        arcane_mesh
        arcane_utils
        arcane_thread
        )

if (TARGET trilinos)
    linkLibraries(alien_arcane_test.exe alien_trilinos)
endif ()

commit(alien_arcane_test.exe)

if (TARGET petsc)

    configure_file(AlienTest.config ${CMAKE_CURRENT_BINARY_DIR}/AlienTest.config COPYONLY)
    configure_file(AlienTest.config ${CMAKE_CURRENT_BINARY_DIR}/AlienTest.config COPYONLY)
    configure_file(AlienTest.arc ${CMAKE_CURRENT_BINARY_DIR}/AlienTest.arc COPYONLY)
    configure_file(AlienTest-FBS.arc ${CMAKE_CURRENT_BINARY_DIR}/AlienTest-FBS.arc COPYONLY)
    configure_file(AlienTest-VBS.arc ${CMAKE_CURRENT_BINARY_DIR}/AlienTest-VBS.arc COPYONLY)
    configure_file(tube5x5x100.vtk ${CMAKE_CURRENT_BINARY_DIR}/tube5x5x100.vtk COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienTest.config
            AlienTest.arc
            AlienTest-FBS.arc
            AlienTest-VBS.arc
            tube5x5x100.vtk
            )

endif ()

#Examples !

get_property(ALIEN_COMPILE_DEFS SOURCE main.cc PROPERTY COMPILE_DEFINITIONS)
set(ALIEN_EXAMPLE_NAME "AlienExample.exe")

configure_file(Makefile.in Makefile.example @ONLY)

install(FILES
        ${CMAKE_BINARY_DIR}/ALIEN/axl/AlienTest_axl.h
        AlienTestModule.h
        AlienTestModule.cc
        AlienTestOptionTypes.h
        MemoryAllocationTracker.h
        MemoryAllocationTracker.cc
        main.cc
        DESTINATION share/examples
        )

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Makefile.example
        DESTINATION share/examples
        RENAME Makefile
        )


if (TARGET petsc)

    set(ENV (ARCANE_PARALLEL_SERVICE) "Mpi")

    alien_test(
            BENCH alien.test.arcane
            NAME scalar
            PROCS 4
            COMMAND alien_arcane_test.exe
            OPTIONS AlienTest.arc
            WORKING_DIRECTORY test/AlienTest)

    alien_test(
            BENCH alien.test.arcane
            NAME block
            PROCS 4
            COMMAND alien_arcane_test.exe
            OPTIONS AlienTest-FBS.arc
            WORKING_DIRECTORY test/AlienTest)

    alien_test(
            BENCH alien.test.arcane
            NAME vblock
            PROCS 4
            COMMAND alien_arcane_test.exe
            OPTIONS AlienTest-VBS.arc
            WORKING_DIRECTORY test/AlienTest)

endif ()
