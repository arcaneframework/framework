include(LoadAlienTest)

#-----------------------------------------------------------
# Without Arcane
#-----------------------------------------------------------

add_subdirectory(Tests)

#-----------------------------------------------------------
# Arcane
#-----------------------------------------------------------

if (TARGET arcane_core)

    add_subdirectory(ArcaneLaplacian)

    add_subdirectory(AlienTest)

    add_subdirectory(AlienBench)

endif ()

#-----------------------------------------------------------
# Bench recap
#-----------------------------------------------------------

print_bench_informations()
