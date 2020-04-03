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

    # Ce test doit etre refactoré pour ne plus
    # utiliser l'API CEA Move-Semantic
    # add_subdirectory(AlienTest)

    add_subdirectory(AlienBench)

endif ()

#-----------------------------------------------------------
# Bench recap
#-----------------------------------------------------------

print_bench_informations()
