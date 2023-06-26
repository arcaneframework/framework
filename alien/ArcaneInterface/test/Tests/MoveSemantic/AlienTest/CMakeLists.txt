find_package(Boost COMPONENTS program_options REQUIRED)

createExecutable(alien_test.exe)

addSources(alien_test.exe main.cpp)

if (TARGET intel)
    linkLibraries(alien_test.exe intel)
endif ()

linkLibraries(alien_test.exe
        Boost::program_options
        Alien::alien_core
        Alien::alien_semantic_move
        )

if (TARGET alien_external_packages)
    linkLibraries(alien_test.exe
            alien_external_packages
            )
endif ()

if (TARGET alien_ifpen_solvers)
    linkLibraries(alien_test.exe
            alien_ifpen_solvers
            )
endif ()

target_include_directories(alien_test.exe
        PUBLIC ${PROJECT_SOURCE_DIR}/test)

commit(alien_test.exe)

install(
        TARGETS alien_test.exe
        RUNTIME DESTINATION bin
)

if (TARGET petsc)
    alien_test(
            BENCH alien.test
            NAME petsc
            COMMAND alien_test.exe
            PROCS 4
            OPTIONS
            --nx=10
            --ny=10
            --solver-package=petsc
            --solver=bicgs
            --precond=bjacobi
            --tol=1.e-10
            --max-iter=100
            --builder=1
    )
endif ()

if (TARGET hypre)
    alien_test(
            BENCH alien.test
            NAME hypre
            PROCS 4
            COMMAND alien_test.exe
            OPTIONS
            --nx=10
            --ny=10
            --solver-package=hypre
            --solver=bicgstab
            --precond=euclid
            --tol=1.e-10
            --max-iter=100
            --builder=1
    )
    alien_test(
            BENCH alien.test
            NAME hypre.diag
            PROCS 4
            COMMAND alien_test.exe
            OPTIONS
            --nx=10
            --ny=10
            --solver-package=hypre
            --solver=bicgstab
            --precond=diag
            --tol=1.e-10
            --max-iter=100
            --builder=1
    )
endif ()

if (TARGET ifpsolver)
    alien_test(
            BENCH alien.test
            NAME ifpsolver
            PROCS 4
            COMMAND alien_test.exe
            OPTIONS
            --nx=10
            --ny=10
            --solver-package=ifpsolver
            --solver=bicgs
            --precond=ilu0
            --tol=1.e-10
            --max-iter=100
            --output-level=1
            --builder=1
    )
endif ()

if (TARGET mcgsolver)
    alien_test(
            BENCH alien.test
            NAME mcgsolver
            PROCS 4
            COMMAND alien_test.exe
            OPTIONS
            --nx=10
            --ny=10
            --solver-package=mcgsolver
            --solver=bicgstab
            --precond=none
            --tol=1.e-10
            --max-iter=1000
            --kernel=cpu_cblas_bcsr
            --output-level=2
    )
endif ()

if (TARGET mtl)
    alien_test(
            BENCH alien.test
            NAME mtl
            COMMAND alien_test.exe
            OPTIONS
            --nx=10
            --ny=10
            --solver-package=mtlsolver
            --solver=bicgstab
            --precond=ilu0
            --tol=1.e-10
            --max-iter=100
            --output-level=1
    )
endif ()
