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
            USER_INSTALL_PREFIX ALIEN
           )

# Les hooks pour les mallocs sont obsolètes et n'existent
# sur certaines plateformes (par exemple ubuntu 22.04)
if (ARCANE_HAS_MALLOC_HOOKS)
  target_compile_definitions(alien_arcane_test.exe PRIVATE TEST_HAS_MALLOC_HOOKS)
endif()


if (TARGET intel)
    linkLibraries(alien_arcane_test.exe intel)
endif ()

linkLibraries(alien_arcane_test.exe
        Alien::alien_core
        Alien::alien_semantic_move
        alien_arcane_tools
        arcane_full
        )

if (TARGET alien_external_packages)
    linkLibraries(alien_arcane_test.exe alien_external_packages)
endif ()

if (TARGET alien_ifpen_solvers AND NOT WIN32)
    linkLibraries(alien_arcane_test.exe alien_ifpen_solvers)
endif ()

if (TARGET trilinos)
    linkLibraries(alien_arcane_test.exe alien_trilinos)
endif ()

commit(alien_arcane_test.exe)

if (TARGET petsc)

    configure_file(AlienTest.config ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest/AlienTest.config COPYONLY)
    configure_file(AlienTest.config ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest/AlienTest.config COPYONLY)
    configure_file(AlienTest.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest/AlienTest.arc COPYONLY)
    configure_file(AlienTest-FBS.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest/AlienTest-FBS.arc COPYONLY)
    configure_file(AlienTest-VBS.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest/AlienTest-VBS.arc COPYONLY)
    configure_file(tube5x5x100.vtk ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest/tube5x5x100.vtk COPYONLY)

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
            BENCH test.arcane
            NAME scalar
            PROCS 4
            COMMAND alien_arcane_test.exe
            OPTIONS AlienTest.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest)

    alien_test(
            BENCH test.arcane
            NAME block
            PROCS 4
            COMMAND alien_arcane_test.exe
            OPTIONS AlienTest-FBS.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest)

    alien_test(
            BENCH test.arcane
            NAME vblock
            PROCS 4
            COMMAND alien_arcane_test.exe
            OPTIONS AlienTest-VBS.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienTest)

endif ()
