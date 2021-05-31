enable_language(Fortran)
createExecutable(alien_interface_fortran.exe)

set_target_properties(alien_interface_fortran.exe PROPERTIES LINKER_LANGUAGE CXX)

addSources(alien_interface_fortran.exe
           Program.f90
           main.c
           )


linkLibraries(alien_interface_fortran.exe
        mpifort
        Boost::boost
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_interface_c
        alien_external_packages
        alien_ifpen_solvers
        )

if (TARGET trilinos)
    linkLibraries(alien_interface_fortran.exe alien_trilinos)
endif ()

if (TARGET hpddm)
    linkLibraries(alien_interface_fortran.exe alien_hpddm)
endif ()

commit(alien_interface_fortran.exe)

install(
        TARGETS alien_interface_fortran.exe
        RUNTIME DESTINATION bin
)

configure_file(solver.json ${CMAKE_CURRENT_BINARY_DIR}/solver.json COPYONLY)
alien_test( BENCH alien.interface.fortran
            NAME seq
            PROCS 1
            COMMAND alien_interface_fortran.exe
            WORKING_DIRECTORY test/AlienInterfaceC/Fortran)

alien_test( BENCH alien.interface.fortran
            NAME parallel
            PROCS 4
            COMMAND alien_interface_fortran.exe
            WORKING_DIRECTORY test/AlienInterfaceC/Fortran)
