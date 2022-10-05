project(DirectCartesian LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(DirectCartesian main.cc)
arcane_add_arcane_libraries_to_target(DirectCartesian)

# Recopie le jeu de données
configure_file(cartesian1.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
add_test(NAME direct_cartesian COMMAND DirectCartesian cartesian1.arc)
