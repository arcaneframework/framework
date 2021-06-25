enable_testing()
find_package(GTest REQUIRED)

add_executable(hypre_gtest_move main.cpp
        move_use.cpp)

target_link_libraries(hypre_gtest_move
        PRIVATE GTest::GTest
        PRIVATE Alien::alien_core Alien::alien_semantic_move
        Alien::alien_hypre
        )

add_executable(hypre_gtest_ref main.cpp
        ref_use.cpp)

target_link_libraries(hypre_gtest_ref
        PRIVATE GTest::GTest
        PRIVATE Alien::alien_core Alien::alien_semantic_ref
        Alien::alien_hypre
        )

include(LoadAlienTest)

alien_test(
        BENCH hypre_gtest_move
        NAME use_solve_move
        PROCS 2
        COMMAND hypre_gtest_move
)

alien_test(
        BENCH hypre_gtest_ref
        NAME use_solve_ref
        PROCS 2
        COMMAND hypre_gtest_ref
)
