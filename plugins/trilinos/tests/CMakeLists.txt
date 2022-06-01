enable_testing()
find_package(GTest REQUIRED)

add_executable(trilinos_gtest main.cpp
        move_use.cpp
        ref_use.cpp)

target_link_libraries(trilinos_gtest
        PRIVATE GTest::GTest
        PRIVATE Alien::alien_core Alien::alien_semantic_move
        Alien::alien_semantic_ref
        Alien::trilinos_wrapper
        )

include(LoadAlienTest)

alien_test(
        BENCH trilinos_gtest
        NAME use_solve
        PROCS 2
        COMMAND trilinos_gtest
)
