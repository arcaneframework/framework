
include(LoadBuildSystem)
include(LoadAlienTest)

#-----------------------------------------------------------
# Without Arcane
#-----------------------------------------------------------

add_subdirectory(Tests)
add_subdirectory(AlienInterfaceC)

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
