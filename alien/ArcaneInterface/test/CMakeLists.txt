
include(${ALIEN_CMAKE_CONFIG_PATH}/LoadAlienTest.cmake)

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
if(NOT WIN32)
    add_subdirectory(AlienBench)
  endif()
  
endif ()

#-----------------------------------------------------------
# Bench recap
#-----------------------------------------------------------

print_bench_informations()
