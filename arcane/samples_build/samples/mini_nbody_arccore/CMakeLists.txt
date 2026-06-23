cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(MiniNbodyArccore LANGUAGES C CXX)

find_package(Arccore REQUIRED)
add_executable(MiniNbodyArccore main.cc)
target_link_libraries(MiniNbodyArccore PUBLIC Arccore::arccore_accelerator)

# Pour ajouter le support des accélérateurs
arccore_accelerator_enable()
arccore_accelerator_add_source_files(main.cc)
arccore_accelerator_add_to_target(MiniNbodyArccore)

# Recopie le jeu de données
add_test(NAME mini_nbody_arccore COMMAND MiniNbodyArccore)
