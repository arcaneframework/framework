cmake_minimum_required(VERSION 3.13)

add_subdirectory(scalar)
add_subdirectory(block)

if (TARGET GTest::GTest)

    set(TEST_LIST
            Vector
            VectorBuilder
            MatrixDirectBuilder
            CompositeVector
            CompositeMatrix
            CompositeMatrixToMCG
            BlockMatrix
            VBlockMatrix
            EigenSolver)

    if (USE_CXX14)
        set(TEST_LIST ${TEST_LIST} MVExpr)
    endif (USE_CXX14)

    createExecutable(gtest.refmvhandlers)

    addSources(gtest.refmvhandlers
            main.cpp)

    foreach (test_name ${TEST_LIST})
        addSources(gtest.refmvhandlers Test${test_name}.cc)
    endforeach ()

    if (TARGET intel)
        linkLibraries(gtest.refmvhandlers intel)
    endif ()

    if (TARGET alien_core_solvers)
        linkLibraries(gtest.refmvhandlers
                      alien_core_solvers
                     )
    endif ()
    
    if (TARGET alien_external_packages)
        linkLibraries(gtest.refmvhandlers
                      alien_external_packages
                      )
    endif ()

    if (TARGET alien_ifpen_solvers)
        linkLibraries(gtest.refmvhandlers
                      alien_ifpen_solvers
                      )
    endif ()

    linkLibraries(gtest.refmvhandlers
            Boost::program_options
            GTest::GTest
            Alien::alien_core
            Alien::alien_semantic_ref
            )

    target_include_directories(gtest.refmvhandlers PUBLIC
            ${PROJECT_SOURCE_DIR}/test
            )

    commit(gtest.refmvhandlers)

    install(
            TARGETS gtest.refmvhandlers
            RUNTIME DESTINATION refmvhandlers/unitary
    )

    foreach (test_name ${TEST_LIST})
        alien_test(
                BENCH refmvhandlers.gtest
                NAME ${test_name}
                COMMAND gtest.refmvhandlers
                OPTIONS --gtest_filter=Test${test_name}*
        )
    endforeach ()

endif ()



