

createExecutable(alien_interface_c.exe)

#set_target_properties(alien_interface_c.exe PROPERTIES LINKER_LANGUAGE CXX)

addSources(alien_interface_c.exe
           main.c
           )
           
  
linkLibraries(alien_interface_c.exe
        Boost::boost
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_interface_c
        )
        
if (TARGET alien_core_solvers)
    linkLibraries(alien_interface_c.exe alien_core_solvers)
endif ()

if (TARGET alien_external_packages)
    linkLibraries(alien_interface_c.exe alien_external_packages)
endif ()

if (TARGET alien_ifpen_solvers)
    linkLibraries(alien_interface_c.exe alien_ifpen_solvers)
endif ()

if (TARGET trilinos)
    linkLibraries(alien_interface_c.exe alien_trilinos)
endif ()

if (TARGET hpddm)
    linkLibraries(alien_interface_c.exe alien_hpddm)
endif ()

commit(alien_interface_c.exe)

createExecutable(alien_interface_with_param_c.exe)

#set_target_properties(alien_interface_with_param_c.exe PROPERTIES LINKER_LANGUAGE CXX)

addSources(alien_interface_with_param_c.exe
           main_with_parameters.c
           )
           

linkLibraries(alien_interface_with_param_c.exe
        Boost::boost
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_interface_c
        )

if (TARGET alien_core_solvers)
    linkLibraries(alien_interface_with_param_c.exe alien_core_solvers)
endif ()

if (TARGET alien_external_packages)
    linkLibraries(alien_interface_with_param_c.exe alien_external_packages)
endif ()

if (TARGET alien_ifpen_solvers)
    linkLibraries(alien_interface_with_param_c.exe alien_ifpen_solvers)
endif ()

if (TARGET trilinos)
    linkLibraries(alien_interface_with_param_c.exe alien_trilinos)
endif ()

if (TARGET hpddm)
    linkLibraries(alien_interface_with_param_c.exe alien_hpddm)
endif ()

commit(alien_interface_with_param_c.exe)

install(
        TARGETS alien_interface_c.exe alien_interface_with_param_c.exe
        RUNTIME DESTINATION bin
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)

alien_test( BENCH interface.c
            NAME seq
            PROCS 1
            COMMAND alien_interface_c.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)

alien_test( BENCH interface.c
            NAME parallel
            PROCS 4
            COMMAND alien_interface_c.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)

alien_test( BENCH interface.c.param
            NAME seq
            PROCS 1
            COMMAND alien_interface_with_param_c.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)

alien_test( BENCH interface.c.param
            NAME parallel
            PROCS 4
            COMMAND alien_interface_with_param_c.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)
