enable_testing()
find_package(GTest REQUIRED)

add_executable(hypre_gtest main.cpp
        move_use.cpp
        ref_use.cpp)

target_link_libraries(hypre_gtest
        PRIVATE GTest::GTest
        PRIVATE Alien::alien_core Alien::alien_semantic_move
        Alien::alien_semantic_ref
        Alien::hypre_wrapper
        )

include(LoadAlienTest)

alien_test(
        BENCH hypre_gtest
        NAME use_solve
        PROCS 2
        COMMAND hypre_gtest
)
