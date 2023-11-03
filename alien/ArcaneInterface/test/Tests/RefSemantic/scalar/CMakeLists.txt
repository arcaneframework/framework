cmake_minimum_required(VERSION 3.0)

SET(REFMV_SCALAR_TEST_LIST
        Direct
        DirectByDoK
        Profiled
        Stream
        Import)

if (TARGET arcane)
    LIST(APPEND REFMV_SCALAR_TEST_LIST DirectByIndexManager)
endif (TARGET arcane)

macro(compile_test BUILDER)

    createExecutable(refmvhandlers.Scalar${BUILDER})

    addSources(refmvhandlers.Scalar${BUILDER} ${BUILDER}.cpp)

    if (TARGET intel)
        linkLibraries(refmvhandlers.Scalar${BUILDER} intel)
    endif (TARGET intel)

    linkLibraries(refmvhandlers.Scalar${BUILDER}
            Boost::program_options
            Alien::alien_core
            Alien::alien_semantic_ref
            )

    if (TARGET alien_external_packages)
        linkLibraries(refmvhandlers.Scalar${BUILDER}
                alien_external_packages
                )
    endif ()

    if (TARGET alien_ifpen_solvers)
        linkLibraries(refmvhandlers.Scalar${BUILDER}
                alien_ifpen_solvers
                )
    endif ()

    #if (TARGET trilinos)
    #    linkLibraries(refmvhandlers.Scalar${BUILDER} ALIEN-Trilinos)
    #endif ()

    target_include_directories(refmvhandlers.Scalar${BUILDER}
            PUBLIC ${PROJECT_SOURCE_DIR}/test
            )

    install(
            TARGETS refmvhandlers.Scalar${BUILDER}
            RUNTIME DESTINATION api
    )

endmacro()

foreach (BUILDER IN LISTS REFMV_SCALAR_TEST_LIST)
    compile_test(${BUILDER})

    addSources(refmvhandlers.Scalar${BUILDER}
            BuilderTestFramework.cpp
            )

    if (TARGET petsc)
        linkLibraries(refmvhandlers.Scalar${BUILDER}
                petsc
                )
    endif (TARGET petsc)

    commit(refmvhandlers.Scalar${BUILDER})
endforeach ()

# RedistributorTestFramework
createExecutable(refmvhandlers.ScalarRedistributor)

addSources(refmvhandlers.ScalarRedistributor
        RedistributorTestFramework.cpp
        Profiled.cpp
        )

linkLibraries(refmvhandlers.ScalarRedistributor
        Boost::program_options
        Alien::alien_core
        Alien::alien_semantic_ref
        )

if (TARGET alien_external_packages)
    linkLibraries(refmvhandlers.ScalarRedistributor
            alien_external_packages
            )
endif ()

if (TARGET alien_ifpen_solvers)
    linkLibraries(refmvhandlers.ScalarRedistributor
            alien_ifpen_solvers
            )
endif ()

#if (TARGET trilinos)
#    linkLibraries(refmvhandlers.ScalarRedistributor
#            ALIEN-Trilinos
#            )
#endif ()

target_include_directories(refmvhandlers.ScalarRedistributor
        PUBLIC ${PROJECT_SOURCE_DIR}/test
        )

commit(refmvhandlers.ScalarRedistributor)

install(
        TARGETS refmvhandlers.ScalarRedistributor
        RUNTIME DESTINATION api
)

# EndRedistributorTestFramework

# RedistributorAlgebraTestFramework

createExecutable(refmvhandlers.ScalarRedistributorAlgebra)

addSources(refmvhandlers.ScalarRedistributorAlgebra
        RedistributorAlgebraTestFramework.cpp
        Profiled.cpp
        )

linkLibraries(refmvhandlers.ScalarRedistributorAlgebra
        Boost::program_options
        Alien::alien_core
        Alien::alien_semantic_ref
        )

if (TARGET alien_external_packages)
    linkLibraries(refmvhandlers.ScalarRedistributorAlgebra
            alien_external_packages
            )
endif ()

if (TARGET alien_ifpen_solvers)
    linkLibraries(refmvhandlers.ScalarRedistributorAlgebra
            alien_ifpen_solvers
            )
endif ()

#if (TARGET trilinos)
#    linkLibraries(refmvhandlers.ScalarRedistributorAlgebra
#            ALIEN-Trilinos
#            )
#endif ()

target_include_directories(refmvhandlers.ScalarRedistributorAlgebra
        PUBLIC ${PROJECT_SOURCE_DIR}/test
        )

commit(refmvhandlers.ScalarRedistributorAlgebra)

install(
        TARGETS refmvhandlers.ScalarRedistributorAlgebra
        RUNTIME DESTINATION api
)

# EndRedistributorAlgebraTestFramework

SET(REFMV_SCALAR_REDISTRIBUTION_TEST_LIST
        pair
        unique)

SET(REFMV_SCALAR_REDISTRIBUTION_METHOD_LIST
        dok
        csr)

foreach (STRATEGY IN LISTS REFMV_SCALAR_REDISTRIBUTION_TEST_LIST)
    foreach (METHOD IN LISTS REFMV_SCALAR_REDISTRIBUTION_METHOD_LIST)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME RedistributorAlgebra.${STRATEGY}.${METHOD}
                PROCS 4
                COMMAND refmvhandlers.ScalarRedistributorAlgebra
                OPTIONS
                --size=100
                --redist-strategy=${STRATEGY}
                --redist-method=${METHOD}
        )

        if (TARGET petsc)
            alien_test(
                    BENCH refmvhandlers.scalar
                    NAME Redistributor.petsc.${STRATEGY}.${METHOD}
                    PROCS 4
                    COMMAND refmvhandlers.ScalarRedistributor
                    OPTIONS
                    --size=100
                    --solver-package=petsc
                    --solver=bicgs
                    --precond=bjacobi
                    --tol=1.e-10
                    --max-iter=100
                    --redist-strategy=${STRATEGY}
                    --redist-method=${METHOD}
            )
        endif (TARGET petsc)

        if (TARGET hypre)
            alien_test(
                    BENCH refmvhandlers.scalar
                    NAME Redistributor.hypre.${STRATEGY}.${METHOD}
                    PROCS 4
                    COMMAND refmvhandlers.ScalarRedistributor
                    OPTIONS
                    --size=100
                    --solver-package=hypre
                    --solver=gmres
                    --precond=euclid
                    --tol=1.e-10
                    --max-iter=100
                    --redist-strategy=${STRATEGY}
                    --redist-method=${METHOD}
            )
        endif ()

        if (TARGET ifpsolver)
            alien_test(
                    BENCH refmvhandlers.scalar
                    NAME Redistributor.ifpsolver.${STRATEGY}.${METHOD}
                    PROCS 4
                    COMMAND refmvhandlers.ScalarRedistributor
                    OPTIONS
                    --size=100
                    --solver-package=ifpsolver
                    --solver=bicgs
                    --precond=ilu0
                    --tol=1.e-10
                    --max-iter=100
                    --output-level=1
                    --redist-strategy=${STRATEGY}
                    --redist-method=${METHOD}
            )
        endif ()
    endforeach ()
endforeach ()

compile_test(ReaderWriter)
commit(refmvhandlers.ScalarReaderWriter)
alien_test(
        BENCH refmvhandlers.scalar
        NAME ReaderWriter
        PROCS 4
        COMMAND refmvhandlers.ScalarReaderWriter
)

if (TARGET hdf5 AND TARGET mpi AND TARGET petsc)
    compile_test(ExportImport)
    commit(refmvhandlers.ScalarExportImport)
    alien_test(
            BENCH refmvhandlers.scalar
            NAME ExportImport
            COMMAND refmvhandlers.ScalarExportImport
    )
endif ()

foreach (REFMVTESTS
        Direct
        DirectByDoK
        Profiled
        Stream)

    if (TARGET petsc)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME ${REFMVTESTS}.petsc
                PROCS 4
                COMMAND refmvhandlers.Scalar${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=petsc
                --solver=bicgs
                --precond=bjacobi
                --tol=1.e-10
                --max-iter=100
        )
        alien_test(
                BENCH refmvhandlers.scalar
                NAME ${REFMVTESTS}.petsc.lu
                COMMAND refmvhandlers.Scalar${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=petsc
                --solver=lu
        )

        if (TARGET mumps)
            alien_test(
                    BENCH refmvhandlers.scalar
                    NAME ${REFMVTESTS}.petsc.mumps
                    PROCS 4
                    COMMAND refmvhandlers.Scalar${REFMVTESTS}
                    OPTIONS
                    --size=100
                    --solver-package=petsc
                    --solver=mumps
            )
        endif ()

    endif (TARGET petsc)

    if (TARGET hypre)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME ${REFMVTESTS}.hypre
                PROCS 4
                COMMAND refmvhandlers.Scalar${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=hypre
                --solver=gmres
                --precond=euclid
                --tol=1.e-10
                --max-iter=100
        )
        alien_test(
                BENCH refmvhandlers.scalar
                NAME ${REFMVTESTS}.hypre.diag
                PROCS 4
                COMMAND refmvhandlers.Scalar${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=hypre
                --solver=bicgstab
                --precond=diag
                --tol=1.e-9
                --max-iter=100
        )
    endif ()

    if (TARGET ifpsolver)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME ${REFMVTESTS}.ifpsolver
                PROCS 4
                COMMAND refmvhandlers.Scalar${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=ifpsolver
                --solver=bicgs
                --precond=ilu0
                --tol=1.e-10
                --max-iter=100
                --output-level=1
        )
    endif ()

    if (TARGET mcgsolver)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME ${REFMVTESTS}.mcgsolver
                PROCS 4
                COMMAND refmvhandlers.Scalar${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=mcgsolver
                --solver=bicgstab
                --precond=none
                --tol=1.e-10
                --max-iter=100
                --kernel=cpu_cblas_bcsr
                --output-level=2
        )

    endif ()

    if (TARGET mtl)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME ${REFMVTESTS}.mtl
                COMMAND refmvhandlers.Scalar${REFMVTESTS}
                OPTIONS
                --solver-package=mtlsolver
                --solver=bicgstab
                --precond=ilu0
                --tol=1.e-10
                --max-iter=100
                --output-level=1
        )
    endif ()

endforeach ()

if (TARGET arcane)
    alien_test(
            BENCH refmvhandlers.scalar
            NAME DirectByIndexManager.petsc
            PROCS 4
            COMMAND refmvhandlers.ScalarDirectByIndexManager
            OPTIONS
            --size=80
            --solver-package=petsc
            --solver=bicgs
            --precond=bjacobi
            --tol=1.e-10
            --max-iter=100
    )
endif ()
