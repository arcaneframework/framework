project(StandaloneSubDomain LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(StandaloneSubDomain main.cc)
arcane_add_arcane_libraries_to_target(StandaloneSubDomain)

# Ajoute test de l'exécutable
configure_file(plancher.msh ${CMAKE_CURRENT_BINARY_DIR}/plancher.msh COPYONLY)
add_test(NAME standalone_subdomain_test1 COMMAND StandaloneSubDomain)
