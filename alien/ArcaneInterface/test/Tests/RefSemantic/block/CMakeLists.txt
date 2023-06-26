cmake_minimum_required(VERSION 3.0)

foreach (BUILDER Profiled Stream Import)

    createExecutable(refmvhandlers.Block${BUILDER})

    addSources(refmvhandlers.Block${BUILDER} ${BUILDER}.cpp BuilderTestBlockFramework.cpp)

    if (TARGET intel)
        linkLibraries(refmvhandlers.Block${BUILDER} intel)
    endif ()

    if (TARGET alien_external_packages)
        linkLibraries(refmvhandlers.Block${BUILDER}
                alien_external_packages
                )
    endif ()

    if (TARGET alien_ifpen_solvers)
        linkLibraries(refmvhandlers.Block${BUILDER}
                alien_ifpen_solvers
                )
    endif ()

    #if (TARGET trilinos)
    #    linkLibraries(refmvhandlers.Block${BUILDER} ALIEN-Trilinos)
    #endif ()

    linkLibraries(refmvhandlers.Block${BUILDER}
            Boost::program_options
            Alien::alien_core
            Alien::alien_semantic_ref
            )

    target_include_directories(refmvhandlers.Block${BUILDER}
            PUBLIC ${PROJECT_SOURCE_DIR}/test
            )

    commit(refmvhandlers.Block${BUILDER})

    install(
            TARGETS refmvhandlers.Block${BUILDER}
            RUNTIME DESTINATION api
    )

endforeach ()

foreach (BUILDER VBlockStream)

    createExecutable(refmvhandlers.${BUILDER})

    addSources(refmvhandlers.${BUILDER} ${BUILDER}.cpp BuilderTestVBlockFramework.cpp)

    if (TARGET alien_external_packages)
        linkLibraries(refmvhandlers.${BUILDER}
                alien_external_packages
                )
    endif ()

    if (TARGET alien_ifpen_solvers)
        linkLibraries(refmvhandlers.${BUILDER}
                alien_ifpen_solvers
                )
    endif ()

    #if (TARGET trilinos)
    #    linkLibraries(refmvhandlers.${BUILDER}
    #            ALIEN-Trilinos
    #            )
    #endif ()

    linkLibraries(refmvhandlers.${BUILDER}
            Alien::alien_core
            Alien::alien_semantic_ref
            Boost::program_options
            )

    if (TARGET intel)
        linkLibraries(refmvhandlers.${BUILDER} intel)
    endif (TARGET intel)

    target_include_directories(refmvhandlers.${BUILDER}
            PUBLIC ${PROJECT_SOURCE_DIR}/test
            )

    commit(refmvhandlers.${BUILDER})

    install(
            TARGETS refmvhandlers.${BUILDER}
            RUNTIME DESTINATION api
    )

endforeach ()

foreach (REFMVTESTS
        Profiled
        Stream)

    if (TARGET petsc)
        alien_test(
                BENCH refmvhandlers.block
                NAME ${REFMVTESTS}.petsc
                PROCS 4
                COMMAND refmvhandlers.Block${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=petsc
                --solver=bicgs
                --precond=bjacobi
                --tol=1.e-100
                --max-iter=100
                --block-size=5
        )
        alien_test(
                BENCH refmvhandlers.block
                NAME ${REFMVTESTS}.petsc.lu
                COMMAND refmvhandlers.Block${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=petsc
                --solver=lu
        )
    endif ()

    if (TARGET hypre)
        alien_test(
                BENCH refmvhandlers.block
                NAME ${REFMVTESTS}.hypre
                PROCS 4
                COMMAND refmvhandlers.Block${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=hypre
                --solver=bicgstab
                --precond=euclid
                --tol=1.e-10
                --max-iter=100
                --block-size=5
        )
    endif ()

    if (TARGET ifpsolver)
        alien_test(
                BENCH refmvhandlers.block
                NAME ${REFMVTESTS}.ifpsolver
                PROCS 4
                COMMAND refmvhandlers.Block${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=ifpsolver
                --solver=bicgs
                --precond=ilu0
                --tol=1.e-10
                --max-iter=100
                --block-size=5
                --output-level=1
        )
    endif ()

    if (TARGET mcgsolver)
        alien_test(
                BENCH refmvhandlers.block
                NAME ${REFMVTESTS}.mcgsolver
                PROCS 4
                COMMAND refmvhandlers.Block${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=mcgsolver
                --solver=bicgstab
                --precond=none
                --tol=1.e-10
                --max-iter=100
                --kernel=cpu_cblas_bcsr
                --block-size=3
                --output-level=1
        )
    endif ()

    if (TARGET mtl)
        alien_test(
                BENCH refmvhandlers.block
                NAME ${REFMVTESTS}.mtl
                COMMAND refmvhandlers.Block${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=mtlsolver
                --solver=bicgstab
                --precond=ilu0
                --tol=1.e-10
                --max-iter=100
                --block-size=5
                --output-level=1
        )
    endif ()

endforeach ()
