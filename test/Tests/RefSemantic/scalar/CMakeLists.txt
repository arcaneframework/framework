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
            ALIEN
            ALIEN-RefSemantic
            )

    if (TARGET ALIEN-ExternalPackages)
        linkLibraries(refmvhandlers.Scalar${BUILDER}
                ALIEN-ExternalPackages
                )
    endif ()

    if (TARGET ALIEN-IFPENSolvers)
        linkLibraries(refmvhandlers.Scalar${BUILDER}
                ALIEN-IFPENSolvers
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
        ALIEN
        ALIEN-RefSemantic
        )

if (TARGET ALIEN-ExternalPackages)
    linkLibraries(refmvhandlers.ScalarRedistributor
            ALIEN-ExternalPackages
            )
endif ()

if (TARGET ALIEN-IFPENSolvers)
    linkLibraries(refmvhandlers.ScalarRedistributor
            ALIEN-IFPENSolvers
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
        ALIEN
        ALIEN-RefSemantic
        )

if (TARGET ALIEN-ExternalPackages)
    linkLibraries(refmvhandlers.ScalarRedistributorAlgebra
            ALIEN-ExternalPackages
            )
endif ()

if (TARGET ALIEN-IFPENSolvers)
    linkLibraries(refmvhandlers.ScalarRedistributorAlgebra
            ALIEN-IFPENSolvers
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

foreach (STRATEGY IN LISTS REFMV_SCALAR_REDISTRIBUTION_TEST_LIST)
    alien_test(
            BENCH refmvhandlers.scalar
            NAME RedistributorAlgebra.${STRATEGY}
            PROCS 4
            COMMAND refmvhandlers.ScalarRedistributorAlgebra
            OPTIONS
            --size=100
            --redist-strategy=${STRATEGY}
    )

    if (TARGET petsc)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME Redistributor.petsc.${STRATEGY}
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
        )
    endif (TARGET petsc)

    if (TARGET hypre)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME Redistributor.hypre.${STRATEGY}
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
        )
    endif ()

    if (TARGET ifpsolver)
        alien_test(
                BENCH refmvhandlers.scalar
                NAME Redistributor.ifpsolver.${STRATEGY}
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
        )
    endif ()
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
                COMMAND refmvhandlers.Scalar${REFMVTESTS}
                OPTIONS
                --size=100
                --solver-package=mcgsolver
                --solver=bicgstab
                --precond=ilu0
                --tol=1.e-10
                --max-iter=100
                --kernel=mckernel
                --output-level=1
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
