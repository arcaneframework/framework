enable_language(Fortran)
createExecutable(alien_interface_fortran.exe)

#set_target_properties(alien_interface_fortran.exe PROPERTIES LINKER_LANGUAGE CXX)

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
        )

if(TARGET alien_core_solvers)
    linkLibraries(alien_interface_fortran.exe alien_core_solvers)
endif()

if(TARGET alien_external_packages)
    linkLibraries(alien_interface_fortran.exe alien_external_packages)
endif()

if(TARGET alien_ifpen_solvers)
    linkLibraries(alien_interface_fortran.exe alien_ifpen_solvers)
endif()

if (TARGET intel)
    linkLibraries(alien_interface_fortran.exe intel)
endif ()

if (TARGET trilinos)
    linkLibraries(alien_interface_fortran.exe alien_trilinos)
endif ()

if (TARGET hpddm)
    linkLibraries(alien_interface_fortran.exe alien_hpddm)
endif ()

commit(alien_interface_fortran.exe)

createExecutable(alien_interface_fortran_param.exe)

set_target_properties(alien_interface_fortran_param.exe PROPERTIES LINKER_LANGUAGE CXX)

addSources(alien_interface_fortran_param.exe
           ProgramWithParameters.f90
           main.c
           )


linkLibraries(alien_interface_fortran_param.exe
        mpifort
        Boost::boost
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_interface_c
        )

if(TARGET alien_external_packages)
    linkLibraries(alien_interface_fortran_param.exe alien_external_packages)
endif()

if(TARGET alien_ifpen_solvers)
    linkLibraries(alien_interface_fortran_param.exe alien_ifpen_solvers)
endif()

if (TARGET intel)
    linkLibraries(alien_interface_fortran_param.exe intel)
endif ()

if (TARGET trilinos)
    linkLibraries(alien_interface_fortran_param.exe alien_trilinos)
endif ()

if (TARGET hpddm)
    linkLibraries(alien_interface_fortran_param.exe alien_hpddm)
endif ()

if (TARGET fortran)
    linkLibraries(alien_interface_fortran_param.exe fortran)
endif ()

if (TARGET gfortran)
    linkLibraries(alien_interface_fortran_param.exe gfortran)
endif ()

commit(alien_interface_fortran_param.exe)


install(
        TARGETS alien_interface_fortran.exe alien_interface_fortran_param.exe
        RUNTIME DESTINATION bin
)

configure_file(solver.json ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/Fortran/solver.json COPYONLY)

alien_test( BENCH interface.fortran
            NAME seq
            PROCS 1
            COMMAND alien_interface_fortran.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/Fortran)

alien_test( BENCH interface.fortran
            NAME parallel
            PROCS 4
            COMMAND alien_interface_fortran.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/Fortran)


alien_test( BENCH interface.fortran.param
            NAME seq
            PROCS 1
            COMMAND alien_interface_fortran_param.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/Fortran)
            
alien_test( BENCH interface.fortran.param
            NAME parallel
            PROCS 4
            COMMAND alien_interface_fortran_param.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/Fortran)
            
