add_executable(test_hypre test_hypre.cpp)

target_link_libraries(test_hypre PUBLIC
        alien_hypre
        ALIEN-RefSemantic)

#include(LoadAlienTest)
#
##-----------------------------------------------------------
## Tests
##-----------------------------------------------------------
#
#alien_test(
#        BENCH hypre
#        NAME hypre.default
#        PROCS 4
#        COMMAND test_hypre
#)

#-----------------------------------------------------------
# Bench recap
#-----------------------------------------------------------

print_bench_informations()

