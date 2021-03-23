set(EXAMPLE_NAME EOS)

project(EOS LANGUAGES C CXX)
include(../samples.common.cmake)

arcane_target_add_axl(EOS PerfectGasEOS)
target_sources(EOS PRIVATE PerfectGasEOSService.cc)

# Ajoute une cible sous la forme d'une bibliothèque
# dynamique pour exécuter le code depuis un runtime
# '.Net' (mono ou coreclr)
add_library(EOSLib EOSModule.cc PerfectGasEOSService.cc EOS_axl.h)
arcane_add_arcane_libraries_to_target(EOSLib)

add_subdirectory(csharp)

add_test(NAME eos1 COMMAND EOS EOS.arc)
