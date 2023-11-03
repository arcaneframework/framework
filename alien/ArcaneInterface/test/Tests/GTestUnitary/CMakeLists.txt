# test avec GTEST

set(SEQ_TEST_LIST
        Space
        )

set(PARALLEL_TEST_LIST
        BackEnds
        Timestamp
        CompositeSpace
        #SubMatrix
        Universe
        Distribution
        SpacePartition
        DoKMatrix
        Redistributor
        )

if (TARGET arcane)
    list(APPEND PARALLEL_TEST_LIST IndexManager)
endif ()

#set (DEP_LIBRARIES)
#if(TARGET ALIEN-CEA)
#  list(APPEND PARALLEL_TEST_LIST DoKMatrix)
#  list(APPEND DEP_LIBRARIES ALIEN-CEA)
#endif (TARGET ALIEN-CEA)

#if(TARGET ALIEN-IFPEN)
#  list(APPEND PARALLEL_TEST_LIST Redistributor)
#  list(APPEND DEP_LIBRARIES ALIEN-IFPEN)
#endif (TARGET ALIEN-IFPEN)

createExecutable(gtest.unitary)

addSources(gtest.unitary
        main.cpp)

foreach (test_name ${SEQ_TEST_LIST})
    addSources(gtest.unitary Test${test_name}.cc)
endforeach ()

foreach (test_name ${PARALLEL_TEST_LIST})
    addSources(gtest.unitary Test${test_name}.cc)
endforeach ()

linkLibraries(gtest.unitary
        GTest::GTest
        Alien::alien_core
        )

if (TARGET arcane_core)
else ()
    linkLibraries(gtest.unitary alien_strong_options)
endif ()

target_include_directories(gtest.unitary
        PUBLIC ${PROJECT_SOURCE_DIR}/test)

commit(gtest.unitary)

install(
        TARGETS gtest.unitary
        RUNTIME DESTINATION bin
)

foreach (test_name ${SEQ_TEST_LIST})
    alien_test(
            BENCH alien.gtest
            NAME ${test_name}
            COMMAND gtest.unitary
            OPTIONS --gtest_filter=Test${test_name}*
    )
endforeach ()

foreach (test_name ${PARALLEL_TEST_LIST})
    alien_test(
            BENCH alien.gtest
            NAME ${test_name}
            PROCS 2
            COMMAND gtest.unitary
            OPTIONS --gtest_filter=Test${test_name}*
    )
endforeach ()
