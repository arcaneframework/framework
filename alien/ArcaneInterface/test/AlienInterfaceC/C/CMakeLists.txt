createExecutable(alien_interface_c.exe)

set_target_properties(alien_interface_c.exe PROPERTIES LINKER_LANGUAGE CXX)

addSources(alien_interface_c.exe
           main.c
           )
           

linkLibraries(alien_interface_c.exe
        Boost::boost
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_interface_c
        alien_external_packages
        alien_ifpen_solvers
        )

if (TARGET trilinos)
    linkLibraries(alien_interface_c.exe alien_trilinos)
endif ()

if (TARGET hpddm)
    linkLibraries(alien_interface_c.exe alien_hpddm)
endif ()

commit(alien_interface_c.exe)

install(
        TARGETS alien_interface_c.exe
        RUNTIME DESTINATION bin
)

alien_test( BENCH alien.interface.c
            NAME seq
            PROCS 1
            COMMAND alien_interface_c.exe
            WORKING_DIRECTORY test/AlienInterfaceC/C)

alien_test( BENCH alien.interface.c
            NAME parallel
            PROCS 4
            COMMAND alien_interface_c.exe
            WORKING_DIRECTORY test/AlienInterfaceC/C)

